"""
CDR3 extraction utility — splits a paired antibody (VH + linker + VL) into
its two CDR3 regions using conserved motif anchoring.

抗体 CDR3 的位置依赖两个保守 motif:
  - Heavy chain (VH) CDR3:夹在 [YQ]YC ... WG[QHRK]G 之间
      e.g. "...YYC ARETLFLQVRE... WGQG..."
                ^CDR3-H starts^                ^ends^
  - Light chain (VL) CDR3:夹在 [YQ]YC ... FG[QHRK]G 之间
      e.g. "...YYC QQYGSSPLRT FGQG..."

我们没用 IMGT alignment(序列本身没带 gap),所以用 motif 锚定。失败率应该
< 5%——主要是 high-temperature sampling 偶尔把保守 Cys/Trp/Phe 也改掉的序列。
"""

import re

# FR4 起点 motif:
# Heavy: WG[QRHKM][GA] (典型 WGQG / WGRG)
# Light: FG[QRGSH][GA] (典型 FGQG / FGGG)
# 我们用 "last FR4-motif + walk back to nearest C" 算法,而不是
# anchor 在 YYC——因为 YYC 在 high-diversity sampling 下偶尔会被
# 改成 AYC / VYC / YHC 之类,导致 regex 失败。
HEAVY_END_RE = re.compile(r'WG[QRHKM][GA]')
LIGHT_END_RE = re.compile(r'FG[QRGSH][GA]')

# CDR3 length sanity bounds (skip 提取结果在这之外的)
MIN_CDR3_LEN = 3
MAX_CDR3_LEN = 35

# 配对抗体的 linker(VH + GGGGSGGGGS + VL)
LINKER = "GGGGSGGGGS"

def split_vh_vl(paired_seq: str):
    """Split paired sequence on linker. Returns (vh, vl) or (None, None) if linker absent."""
    if LINKER not in paired_seq:
        return None, None
    idx = paired_seq.index(LINKER)
    vh = paired_seq[:idx]
    vl = paired_seq[idx + len(LINKER):]
    return vh, vl

def _find_leftmost_c_in_window(seq: str, end_pos: int):
    """
    在 [end_pos - MAX_CDR3_LEN, end_pos - MIN_CDR3_LEN] 窗口里
    找 LEFTMOST C — 这是保守的 pre-CDR3 Cys(避开 CDR3 内部偶尔出现的 Cys)
    """
    valid_start = max(0, end_pos - MAX_CDR3_LEN - 2)
    valid_end = end_pos - MIN_CDR3_LEN + 1
    for i in range(valid_start, valid_end):
        if seq[i] == 'C':
            return i
    return -1

def extract_cdr3_heavy(vh_seq: str):
    """
    Extract VH CDR3.
    Strategy: 找 VH 序列里**最后一个 W**(几乎一定是 FR4 起点的 W),
    然后向前找 leftmost C(避开 CDR3 内部 Cys).
    """
    # rfind W in VH
    w_pos = vh_seq.rfind('W')
    if w_pos == -1:
        return None
    # FR4-W 通常离 VH 结尾 8-15 AA(FR4 = WGxGTLVTVSS),如果 W 太靠前(< 50)或者
    # 太靠尾(< 4 AA from end),很可能不是 FR4-W
    if w_pos < 50 or w_pos > len(vh_seq) - 3:
        return None
    cys_pos = _find_leftmost_c_in_window(vh_seq, w_pos)
    if cys_pos == -1:
        return None
    cdr3 = vh_seq[cys_pos + 1:w_pos]
    return cdr3 if MIN_CDR3_LEN <= len(cdr3) <= MAX_CDR3_LEN else None

def extract_cdr3_light(vl_seq: str):
    """
    Extract VL CDR3.
    Strategy: 找最后一个 'FG' dipeptide(FR4 起点),walk back to leftmost C.
    """
    # 找最后一个 'F' followed by 'G' (FR4 starts with FG)
    fg_pos = -1
    for i in range(len(vl_seq) - 1):
        if vl_seq[i] == 'F' and vl_seq[i+1] == 'G':
            fg_pos = i
    if fg_pos == -1:
        return None
    # FR4-F 同理:太靠前或太靠尾不像 FR4-F
    if fg_pos < 50 or fg_pos > len(vl_seq) - 3:
        return None
    cys_pos = _find_leftmost_c_in_window(vl_seq, fg_pos)
    if cys_pos == -1:
        return None
    cdr3 = vl_seq[cys_pos + 1:fg_pos]
    return cdr3 if MIN_CDR3_LEN <= len(cdr3) <= MAX_CDR3_LEN else None

def extract_cdr3_pair(paired_seq: str):
    """
    From a full paired antibody seq, extract both CDR3s.
    Returns dict {vh_cdr3, vl_cdr3, ok} or {ok: False, reason: ...}
    """
    vh, vl = split_vh_vl(paired_seq)
    if vh is None:
        return {"ok": False, "reason": "no_linker"}

    cdr3_h = extract_cdr3_heavy(vh)
    cdr3_l = extract_cdr3_light(vl)

    if cdr3_h is None and cdr3_l is None:
        return {"ok": False, "reason": "no_motif_either_chain"}
    if cdr3_h is None:
        return {"ok": False, "reason": "no_motif_heavy"}
    if cdr3_l is None:
        return {"ok": False, "reason": "no_motif_light"}

    return {"ok": True, "vh_cdr3": cdr3_h, "vl_cdr3": cdr3_l}

def parse_fasta(path: str):
    """Yield (header, seq) tuples from a FASTA file."""
    header = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            line = line.rstrip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_parts)
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)
        if header is not None:
            yield header, "".join(seq_parts)

if __name__ == "__main__":
    # quick smoke test
    test_seq = ("QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGR"
                "VTMTTDTSTSTAYMELRSLRSDDTAVYYCARETLFLQVREFPVYDYDIWGQGTMVTVSS"
                "GGGGSGGGGS"
                "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASNRATGI"
                "PDRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYGSSPLRTFGQGTRVEIK")
    result = extract_cdr3_pair(test_seq)
    print("Test extraction result:")
    print(f"  ok        : {result['ok']}")
    if result["ok"]:
        print(f"  VH CDR3   : {result['vh_cdr3']} ({len(result['vh_cdr3'])} AA)")
        print(f"  VL CDR3   : {result['vl_cdr3']} ({len(result['vl_cdr3'])} AA)")
    else:
        print(f"  reason    : {result['reason']}")

