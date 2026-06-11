"""
lib_regions.py — extend lib_cdr3.py's conserved-motif anchoring from CDR3 to
ALL 7 IMGT/Kabat-style regions per chain: FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4.

Same philosophy as lib_cdr3.py: NO IMGT alignment (sequences carry no gaps),
so we anchor on conserved framework residues/motifs and slice between them.

Anchor residues used (conserved across human VH / Vkappa / Vlambda germlines):
  Chain start  FR1
  Cys1  (1st conserved Cys, ~pos 22)              -> end of FR1
  Trp_FR2 (heavy "W[VILAF][RK]Q" = WVRQ;          -> end of CDR1 / start of FR2
           light "W[YFH][QL]"     = WYQQ)
  Trp_EW  (heavy "W[ILMVF][GSAN]" = the LEW(M/I/V)G Trp) -> end of FR2 / start of CDR2
  FR3 start motif (heavy "R[VFLI][TVS][IML]" = RVTM/RFTI;  -> end of CDR2 / start of FR3
                   light "G[VIL]P[SDANE]RF" = GVPSRF)
  Cys2  (2nd conserved Cys, pre-CDR3 ~pos 104)    -> end of FR3   (SAME anchor lib_cdr3 uses)
  Trp/Phe_FR4 (heavy "WGxG"; light "FGxG")        -> end of CDR3 / start of FR4 (SAME as lib_cdr3)
  Chain end  FR4

CDR3 itself reuses lib_cdr3's exact logic, so CDR3 numbers reproduce the original
cdr3_metrics.json bit-for-bit. The new regions are built around those same anchors.

Region boundary convention (matches antibody convention):
  - conserved Cys belongs to the FRAMEWORK to its left (FR1 ends with Cys1; FR3 ends with Cys2)
  - conserved Trp_FR2 belongs to FR2 (CDR1 ends just before it)
  - FR4 starts at the WGxG / FGxG motif

NOTE on CDR1: the Cys1->Trp_FR2 segment is the "CDR1 loop region" (Cys-to-Trp).
It is a few residues longer than the strict Kabat CDR-H1/L1 because, without
alignment, the conserved Cys is the only robust left anchor. It is internally
consistent and the diversity conclusions are unaffected.
"""

import re
from lib_cdr3 import (
    split_vh_vl,
    _find_leftmost_c_in_window,
    parse_fasta,
    LINKER,
    MIN_CDR3_LEN,
    MAX_CDR3_LEN,
)

# Region order (per chain)
REGION_ORDER = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]

# Heavy-chain conserved motifs
H_FR2_TRP = re.compile(r"W[VILAFM][RK]Q")        # WVRQ / WIRQ / WVKQ  (FR2 Trp)
H_EW_TRP  = re.compile(r"W[ILMVF][GSAN]")          # WMG / WIG / WVG / WVS (LEW.G Trp, FR2 end)
H_FR3_BEG = re.compile(r"R[VFLI][TVSA][IMLV]")     # RVTM / RFTI / RVTI  (FR3 start)
H_FR4_TRP = re.compile(r"WG[QRHKM][GA]")           # WGQG / WGRG         (FR4 start)

# Light-chain conserved motifs
L_FR2_TRP = re.compile(r"W[YFH][QLY][QLRKH]")      # WYQQ / WFQQ / WYLQ / WYQH (FR2 Trp)
L_FR2_END = re.compile(r"[LMVFI][ILVM][YFH]")      # MIY / LIY / LLY ...  (FR2 end, before CDR2)
L_FR3_ANCHOR = re.compile(r"[RK]F[ST]G[SDN]")      # RFSGS / RFSGD / KFSGS (very conserved, ~4aa into FR3)
L_FR4_BEG = re.compile(r"FG[QRGSHKA][GA]")          # FGQG / FGGG         (FR4 start)

def _slice(seq, a, b):
    s = seq[a:b]
    return s if s else None

def extract_regions_heavy(vh):
    """Return dict region->str for the heavy chain, or {} fields None on failure."""
    out = {r: None for r in REGION_ORDER}

    # --- right-side anchors (identical to lib_cdr3.extract_cdr3_heavy) ---
    w_fr4 = vh.rfind("W")
    if w_fr4 == -1 or w_fr4 < 50 or w_fr4 > len(vh) - 3:
        return out, "no_fr4_heavy"
    c2 = _find_leftmost_c_in_window(vh, w_fr4)        # pre-CDR3 Cys
    if c2 == -1:
        return out, "no_cys2_heavy"
    cdr3 = vh[c2 + 1:w_fr4]
    if not (MIN_CDR3_LEN <= len(cdr3) <= MAX_CDR3_LEN):
        return out, "cdr3_len_heavy"

    out["CDR3"] = cdr3
    out["FR4"] = vh[w_fr4:]

    # --- left-side anchors ---
    c1 = vh.find("C")                                 # 1st conserved Cys
    if c1 == -1 or c1 > 40:
        return out, "no_cys1_heavy"
    out["FR1"] = vh[:c1 + 1]

    m_fr2 = H_FR2_TRP.search(vh, c1 + 1)              # FR2 Trp (WVRQ)
    if not m_fr2 or m_fr2.start() > c1 + 30:
        return out, "no_fr2trp_heavy"
    w1 = m_fr2.start()
    out["CDR1"] = _slice(vh, c1 + 1, w1)

    m_ew = H_EW_TRP.search(vh, w1 + 1)               # LEW.G Trp (FR2 end)
    if not m_ew or m_ew.start() > w1 + 25:
        return out, "no_ewtrp_heavy"
    cdr2_start = m_ew.end()
    out["FR2"] = _slice(vh, w1, cdr2_start)

    m_fr3 = H_FR3_BEG.search(vh, cdr2_start + 4)     # FR3 start (RVTM)
    if not m_fr3 or m_fr3.start() >= c2:
        return out, "no_fr3_heavy"
    fr3_start = m_fr3.start()
    out["CDR2"] = _slice(vh, cdr2_start, fr3_start)
    out["FR3"] = _slice(vh, fr3_start, c2 + 1)

    return out, "ok"

def extract_regions_light(vl):
    out = {r: None for r in REGION_ORDER}

    # --- right-side anchors (identical to lib_cdr3.extract_cdr3_light) ---
    fg = -1
    for i in range(len(vl) - 1):
        if vl[i] == "F" and vl[i + 1] == "G":
            fg = i
    if fg == -1 or fg < 50 or fg > len(vl) - 3:
        return out, "no_fr4_light"
    c2 = _find_leftmost_c_in_window(vl, fg)
    if c2 == -1:
        return out, "no_cys2_light"
    cdr3 = vl[c2 + 1:fg]
    if not (MIN_CDR3_LEN <= len(cdr3) <= MAX_CDR3_LEN):
        return out, "cdr3_len_light"

    out["CDR3"] = cdr3
    out["FR4"] = vl[fg:]

    # --- left-side anchors ---
    c1 = vl.find("C")
    if c1 == -1 or c1 > 40:
        return out, "no_cys1_light"
    out["FR1"] = vl[:c1 + 1]

    m_fr2 = L_FR2_TRP.search(vl, c1 + 1)             # FR2 Trp (WYQQ)
    if not m_fr2 or m_fr2.start() > c1 + 30:
        return out, "no_fr2trp_light"
    w1 = m_fr2.start()
    out["CDR1"] = _slice(vl, c1 + 1, w1)

    m_fr2end = L_FR2_END.search(vl, w1 + 8)          # ..MIY/LIY (FR2 end, before CDR-L2)
    if not m_fr2end or m_fr2end.start() > w1 + 22:
        return out, "no_fr2end_light"
    cdr2_start = m_fr2end.end()
    out["FR2"] = _slice(vl, w1, cdr2_start)

    # FR3 start: anchor on the very conserved RFSGS, then walk back to the
    # FR3-starting Gly (FR3 begins "G[VX]P/SxRF...", RFSGS is ~4 aa in).
    m_fr3 = L_FR3_ANCHOR.search(vl, cdr2_start)      # RFSGS
    if not m_fr3 or m_fr3.start() >= c2:
        return out, "no_fr3_light"
    fr3_start = m_fr3.start() - 4
    g = vl.rfind("G", max(cdr2_start, m_fr3.start() - 6), m_fr3.start())
    if g != -1:
        fr3_start = g
    if fr3_start <= cdr2_start:
        return out, "no_fr3_light"
    out["CDR2"] = _slice(vl, cdr2_start, fr3_start)
    out["FR3"] = _slice(vl, fr3_start, c2 + 1)

    return out, "ok"

def extract_regions_pair(paired_seq):
    """
    Full paired antibody -> all regions for both chains.
    Returns dict:
      ok: bool (True if BOTH chains fully extracted, i.e. CDR2/FR3 found)
      vh: {region: str}, vl: {region: str}
      vh_reason, vl_reason
    Region strings may be None individually even when ok is False; callers can
    use whatever regions are present (per-region success is tracked separately).
    """
    vh, vl = split_vh_vl(paired_seq)
    if vh is None:
        return {"ok": False, "reason": "no_linker"}

    vh_regions, vh_reason = extract_regions_heavy(vh)
    vl_regions, vl_reason = extract_regions_light(vl)

    return {
        "ok": (vh_reason == "ok" and vl_reason == "ok"),
        "vh": vh_regions,
        "vl": vl_regions,
        "vh_reason": vh_reason,
        "vl_reason": vl_reason,
    }

if __name__ == "__main__":
    test = ("QVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGR"
            "VTMTTDTSTSTAYMELRSLRSDDTAVYYCARETLFLQVREFPVYDYDIWGQGTMVTVSS"
            "GGGGSGGGGS"
            "EIVLTQSPGTLSLSPGERATLSCRASQSVSSSYLAWYQQKPGQAPRLLIYGASNRATGI"
            "PDRFSGSGSGTDFTLTISSLEPEDFAVYYCQQYGSSPLRTFGQGTRVEIK")
    r = extract_regions_pair(test)
    print("ok:", r["ok"], "| vh:", r["vh_reason"], "| vl:", r["vl_reason"])
    print("\nHEAVY:")
    for reg in REGION_ORDER:
        v = r["vh"][reg]
        print(f"  {reg:5} ({len(v) if v else 0:2} aa): {v}")
    print("\nLIGHT:")
    for reg in REGION_ORDER:
        v = r["vl"][reg]
        print(f"  {reg:5} ({len(v) if v else 0:2} aa): {v}")

