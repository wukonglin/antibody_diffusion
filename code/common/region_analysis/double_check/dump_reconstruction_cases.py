#!/usr/bin/env python3
"""
dump_reconstruction_cases.py  —  RUN ON AIDA   (double-check #3)

Dumps a random sample of teacher-forced reconstruction cases so the per-region
AAR result can be eyeballed: for each sampled test sequence it prints the
original AA string, the reconstructed AA string, a mismatch track, and the
region each position falls in. This shows WHERE reconstruction errors land
(boundary vs interior of CDR1 / CDR2 / CDR3).

Place next to the original eval script (so the model imports resolve) and run:

    cd ~/ab_ld4lg && source .venv/bin/activate
    python scripts/dump_reconstruction_cases.py \
        --data /mnt/beegfs/bulk/mirror/yl4259/ab_ld4lg/processed \
        --ae-ckpt /mnt/beegfs/bulk/mirror/yl4259/ab_ld4lg/runs/ae/autoencoder_latest.pt \
        --out /mnt/beegfs/bulk/mirror/yl4259/ab_ld4lg/eval_reports/reconstruction_cases.txt \
        --split test --n-cases 20 --seed 0

Then scp the .txt back and send it to me.
"""
import argparse, json, re, sys, random
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# import the model code (works under either repo layout)
HERE = Path(__file__).resolve()
for up in [HERE.parent] + list(HERE.parents)[:4]:
    for c in (up, up / "src"):
        if c.is_dir():
            sys.path.insert(0, str(c))
try:
    from code.diffusion.LD4LG.autoencoder import AutoencoderConfig, LanguageAutoencoder
    from code.diffusion.LD4LG.data import PairedAntibodyDataset, make_collate_fn
    from code.diffusion.LD4LG.tokenizer import AATokenizer
except Exception:
    from ab_ld4lg.autoencoder import AutoencoderConfig, LanguageAutoencoder
    from ab_ld4lg.data import PairedAntibodyDataset, make_collate_fn
    from ab_ld4lg.tokenizer import AATokenizer

# region extraction to per-position spans (mirrors lib_regions.py)
LINKER = "GGGGSGGGGS"
MIN_CDR3, MAX_CDR3 = 3, 35
REGION_ORDER = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
H_FR2_TRP = re.compile(r"W[VILAFM][RK]Q")
H_EW_TRP  = re.compile(r"W[ILMVF][GSAN]")
H_FR3_BEG = re.compile(r"R[VFLI][TVSA][IMLV]")
L_FR2_TRP = re.compile(r"W[YFH][QLY][QLRKH]")
L_FR2_END = re.compile(r"[LMVFI][ILVM][YFH]")
L_FR3_ANCHOR = re.compile(r"[RK]F[ST]G[SDN]")

def _leftmost_c(seq, end):
    vs = max(0, end - MAX_CDR3 - 2); ve = end - MIN_CDR3 + 1
    for i in range(vs, ve):
        if seq[i] == "C":
            return i
    return -1

def heavy_spans(vh):
    out = {r: None for r in REGION_ORDER}
    w4 = vh.rfind("W")
    if w4 == -1 or w4 < 50 or w4 > len(vh) - 3: return out
    c2 = _leftmost_c(vh, w4)
    if c2 == -1 or not (MIN_CDR3 <= w4 - (c2 + 1) <= MAX_CDR3): return out
    out["CDR3"] = (c2 + 1, w4); out["FR4"] = (w4, len(vh))
    c1 = vh.find("C")
    if c1 == -1 or c1 > 40: return out
    out["FR1"] = (0, c1 + 1)
    m = H_FR2_TRP.search(vh, c1 + 1)
    if not m or m.start() > c1 + 30: return out
    w1 = m.start(); out["CDR1"] = (c1 + 1, w1)
    me = H_EW_TRP.search(vh, w1 + 1)
    if not me or me.start() > w1 + 25: return out
    cs = me.end(); out["FR2"] = (w1, cs)
    m3 = H_FR3_BEG.search(vh, cs + 4)
    if not m3 or m3.start() >= c2: return out
    out["CDR2"] = (cs, m3.start()); out["FR3"] = (m3.start(), c2 + 1)
    return out

def light_spans(vl):
    out = {r: None for r in REGION_ORDER}
    fg = -1
    for i in range(len(vl) - 1):
        if vl[i] == "F" and vl[i + 1] == "G": fg = i
    if fg == -1 or fg < 50 or fg > len(vl) - 3: return out
    c2 = _leftmost_c(vl, fg)
    if c2 == -1 or not (MIN_CDR3 <= fg - (c2 + 1) <= MAX_CDR3): return out
    out["CDR3"] = (c2 + 1, fg); out["FR4"] = (fg, len(vl))
    c1 = vl.find("C")
    if c1 == -1 or c1 > 40: return out
    out["FR1"] = (0, c1 + 1)
    m = L_FR2_TRP.search(vl, c1 + 1)
    if not m or m.start() > c1 + 30: return out
    w1 = m.start(); out["CDR1"] = (c1 + 1, w1)
    me = L_FR2_END.search(vl, w1 + 8)
    if not me or me.start() > w1 + 22: return out
    cs = me.end(); out["FR2"] = (w1, cs)
    m3 = L_FR3_ANCHOR.search(vl, cs)
    if not m3 or m3.start() >= c2: return out
    fr3 = m3.start() - 4
    g = vl.rfind("G", max(cs, m3.start() - 6), m3.start())
    if g != -1: fr3 = g
    if fr3 <= cs: return out
    out["CDR2"] = (cs, fr3); out["FR3"] = (fr3, c2 + 1)
    return out

def region_label_track(paired):
    """Return a per-position list of region labels ('' where unassigned)."""
    labels = [""] * len(paired)
    if LINKER not in paired:
        return labels
    idx = paired.index(LINKER); vh = paired[:idx]; vl = paired[idx + len(LINKER):]
    voff = idx + len(LINKER)
    for reg, sp in heavy_spans(vh).items():
        if sp:
            for j in range(sp[0], sp[1]): labels[j] = reg
    for j in range(idx, voff):
        labels[j] = "LINK"
    for reg, sp in light_spans(vl).items():
        if sp:
            for j in range(voff + sp[0], voff + sp[1]): labels[j] = reg
    return labels

# short region tag for the annotation row
TAG = {"FR1": "1", "CDR1": "a", "FR2": "2", "CDR2": "b", "FR3": "3",
       "CDR3": "C", "FR4": "4", "LINK": ".", "": " "}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ae-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n-cases", type=int, default=20)
    ap.add_argument("--pool", type=int, default=5000, help="sample cases from the first POOL test seqs")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    sd = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)
    cfg = AutoencoderConfig(**sd["cfg"])
    ae = LanguageAutoencoder(cfg).to(device); ae.load_state_dict(sd["model"]); ae.eval()

    ds = PairedAntibodyDataset(args.data, args.split, max_len=cfg.max_source_len)
    pool = min(args.pool, len(ds))
    random.seed(args.seed)
    idxs = sorted(random.sample(range(pool), min(args.n_cases, pool)))
    tok = AATokenizer(); collate = make_collate_fn(tok.pad_id, tok.bos_id)
    dl = DataLoader(torch.utils.data.Subset(ds, idxs), batch_size=8, shuffle=False, collate_fn=collate)
    itos = tok.itos

    lines = []
    def emit(s=""):
        lines.append(s)

    emit(f"Reconstruction cases  (split={args.split}, n={len(idxs)}, seed={args.seed})")
    emit("region tags:  1=FR1 a=CDR1 2=FR2 b=CDR2 3=FR3 C=CDR3 4=FR4 .=linker")
    emit("mismatch row: '^' = reconstructed != original at that position")
    emit("=" * 100)

    case_no = 0
    with torch.no_grad():
        for batch in dl:
            src = batch["source_tokens"].to(device)
            di  = batch["decoder_input"].to(device)
            tgt = batch["target_tokens"].to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                _, logits = ae(src, di, tgt)
            preds = logits.argmax(-1)
            for k in range(tgt.shape[0]):
                case_no += 1
                t_ids = tgt[k].tolist(); p_ids = preds[k].tolist()
                orig, recon = [], []
                for j, tid in enumerate(t_ids):
                    ch = itos[tid] if 0 <= tid < len(itos) else "?"
                    if ch in ("<eos>", "<pad>"): break
                    if ch in ("<bos>", "<unk>"): ch = "X"
                    orig.append(ch)
                    pch = itos[p_ids[j]] if 0 <= p_ids[j] < len(itos) else "?"
                    if pch in ("<bos>", "<eos>", "<pad>", "<unk>"): pch = "X"
                    recon.append(pch)
                orig = "".join(orig); recon = "".join(recon)
                labels = region_label_track(orig)
                tagrow = "".join(TAG.get(labels[j], " ") for j in range(len(orig)))
                mism = "".join("^" if (j < len(recon) and recon[j] != orig[j]) else " "
                               for j in range(len(orig)))

                # per-region correct/total for this case
                per = {}
                for j in range(len(orig)):
                    r = labels[j]
                    if r in ("", "LINK"): continue
                    c, n = per.get(r, (0, 0))
                    per[r] = (c + (1 if j < len(recon) and recon[j] == orig[j] else 0), n + 1)
                summ = "  ".join(f"{r}:{c}/{n}" for r, (c, n) in
                                 sorted(per.items(), key=lambda x: REGION_ORDER.index(x[0]))
                                 if r in REGION_ORDER)

                emit(f"\n--- case {case_no} (dataset idx {idxs[case_no-1]}, len {len(orig)}) ---")
                # print in 80-char blocks so it stays readable
                W = 80
                for off in range(0, len(orig), W):
                    emit("orig : " + orig[off:off+W])
                    emit("recon: " + recon[off:off+W])
                    emit("mism : " + mism[off:off+W])
                    emit("reg  : " + tagrow[off:off+W])
                    emit("")
                emit("per-region (correct/total): " + summ)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines))
    print("\n".join(lines[:40]))
    print(f"\n[done] wrote {args.out}  ({case_no} cases)")

if __name__ == "__main__":
    main()
