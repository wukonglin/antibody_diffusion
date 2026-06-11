#!/usr/bin/env python3
"""
eval_recovery_aar_byregion.py  —  RUN ON AIDA

Per-region version of eval_recovery_aar.py. Same teacher-forced encode-decode
reconstruction, but accumulates Amino-Acid Recovery (AAR) **per antibody region**
(FR1/CDR1/FR2/CDR2/FR3/CDR3/FR4, heavy & light) plus a **CDR3 per-position
zoom-in**. Region boundaries are found with the same conserved-motif anchoring
used in the region-diversity analysis (no IMGT alignment needed).

Place this file next to the original eval script (so the `code...` imports resolve),
e.g. in ~/ab_ld4lg/ or cs4782-final-project/code/common/, and run:

    python eval_recovery_aar_byregion.py \
        --data /mnt/beegfs/bulk/mirror/yl4259/ab_ld4lg/processed \
        --ae-ckpt /mnt/beegfs/bulk/mirror/yl4259/ab_ld4lg/runs/ae/autoencoder_latest.pt \
        --out /mnt/beegfs/bulk/mirror/yl4259/ab_ld4lg/eval_reports/recovery_aar_byregion.json \
        --split test --n-samples 5000

Then scp recovery_aar_byregion.json back and send it to me; I'll plot it.
"""
import argparse, json, re, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# import path: be robust to either repo layout
# cs4782-final-project:  code/diffusion/LD4LG/autoencoder.py  -> `from code.diffusion.LD4LG...`
# ab_ld4lg:              src/ab_ld4lg/autoencoder.py          -> `from ab_ld4lg...`
# We add the script's directory, its parents (up to 4 levels), and any `src/`
# under them to sys.path, then try both import styles. If both fail, set
# PYTHONPATH manually, e.g.:  PYTHONPATH=~/ab_ld4lg/src python eval_recovery_aar_byregion.py ...
HERE = Path(__file__).resolve()
_cands = []
for up in [HERE.parent] + list(HERE.parents)[:4]:
    _cands += [up, up / "src"]
for c in _cands:
    if c.is_dir():
        sys.path.insert(0, str(c))
_imp_err = None
try:
    from code.diffusion.LD4LG.autoencoder import AutoencoderConfig, LanguageAutoencoder
    from code.diffusion.LD4LG.data import PairedAntibodyDataset, make_collate_fn
    from code.diffusion.LD4LG.tokenizer import AATokenizer
except Exception as e1:
    _imp_err = e1
    try:
        from ab_ld4lg.autoencoder import AutoencoderConfig, LanguageAutoencoder
        from ab_ld4lg.data import PairedAntibodyDataset, make_collate_fn
        from ab_ld4lg.tokenizer import AATokenizer
    except Exception as e2:
        sys.exit(f"[import error] could not import the model code.\n"
                 f"  code.* layout: {_imp_err}\n  ab_ld4lg.* layout: {e2}\n"
                 f"Fix: run this script from the same folder as your original "
                 f"eval_recovery_aar.py, or set PYTHONPATH to the dir that contains "
                 f"the `code` package (cs4782) or the `ab_ld4lg` package (e.g. ~/ab_ld4lg/src).")

# Region extraction -> POSITION SPANS (mirrors lib_regions.py exactly)
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
    if c2 == -1: return out
    if not (MIN_CDR3 <= w4 - (c2 + 1) <= MAX_CDR3): return out
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
    if c2 == -1: return out
    if not (MIN_CDR3 <= fg - (c2 + 1) <= MAX_CDR3): return out
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

def region_spans(paired):
    """yield (chain, region, abs_start, abs_end) in paired-string coords."""
    if LINKER not in paired: return []
    idx = paired.index(LINKER); vh = paired[:idx]; vl = paired[idx + len(LINKER):]
    voff = idx + len(LINKER); spans = []
    for reg, sp in heavy_spans(vh).items():
        if sp: spans.append(("vh", reg, sp[0], sp[1]))
    for reg, sp in light_spans(vl).items():
        if sp: spans.append(("vl", reg, voff + sp[0], voff + sp[1]))
    return spans

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ae-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n-samples", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    print(f"[device] {device}")

    sd = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)
    cfg = AutoencoderConfig(**sd["cfg"])
    ae = LanguageAutoencoder(cfg).to(device); ae.load_state_dict(sd["model"]); ae.eval()
    print(f"[model] loaded {args.ae_ckpt}")

    ds = PairedAntibodyDataset(args.data, args.split, max_len=cfg.max_source_len)
    n = len(ds) if args.n_samples == 0 else min(args.n_samples, len(ds))
    tok = AATokenizer(); collate = make_collate_fn(tok.pad_id, tok.bos_id)
    dl = DataLoader(torch.utils.data.Subset(ds, list(range(n))),
                    batch_size=args.batch, shuffle=False, collate_fn=collate)
    itos = tok.itos

    # accumulators
    reg_acc = {f"{ch}_{r}": [0, 0] for ch in ("vh", "vl") for r in REGION_ORDER}
    overall = [0, 0]
    CDR3_MAX = 35
    cdr3_pos = {ch: [[0, 0] for _ in range(CDR3_MAX)] for ch in ("vh", "vl")}  # from start
    cdr3_pos_end = {ch: [[0, 0] for _ in range(CDR3_MAX)] for ch in ("vh", "vl")}  # from end

    t0 = time.time()
    with torch.no_grad():
        for bi, batch in enumerate(dl):
            src = batch["source_tokens"].to(device)
            di  = batch["decoder_input"].to(device)
            tgt = batch["target_tokens"].to(device)
            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16,
                                    enabled=(device.type == "cuda")):
                _, logits = ae(src, di, tgt)
            preds = logits.argmax(-1)
            correct = (preds == tgt)                     # (B,L) bool
            tgt_cpu = tgt.cpu().tolist(); cor_cpu = correct.cpu().tolist()
            for k in range(len(tgt_cpu)):
                ids = tgt_cpu[k]; cor = cor_cpu[k]
                # decode to AA string + keep token-index alignment (stop at eos/pad)
                aa = []; pos = []
                for j, tid in enumerate(ids):
                    t = itos[tid] if 0 <= tid < len(itos) else "<unk>"
                    if t == "<eos>" or t == "<pad>": break
                    if t in ("<bos>", "<unk>"):
                        aa.append("X"); pos.append(j); continue
                    aa.append(t); pos.append(j)
                aa = "".join(aa)
                for ch, reg, a, b in region_spans(aa):
                    c = sum(1 for j in range(a, b) if cor[pos[j]])
                    reg_acc[f"{ch}_{reg}"][0] += c
                    reg_acc[f"{ch}_{reg}"][1] += (b - a)
                    if reg == "CDR3":
                        L = b - a
                        for off in range(L):
                            if off < CDR3_MAX:
                                cdr3_pos[ch][off][0] += int(cor[pos[a + off]]); cdr3_pos[ch][off][1] += 1
                            eoff = L - 1 - off
                            if eoff < CDR3_MAX:
                                cdr3_pos_end[ch][eoff][0] += int(cor[pos[a + off]]); cdr3_pos_end[ch][eoff][1] += 1
                # overall over the real residues
                for j in range(len(aa)):
                    overall[0] += int(cor[pos[j]]); overall[1] += 1
            if (bi + 1) % 20 == 0:
                print(f"  batch {bi+1}/{len(dl)}  overall AAR {overall[0]/max(1,overall[1]):.4f}")

    def aar(p): return p[0] / p[1] if p[1] else None
    out = {
        "split": args.split, "n_sequences": n,
        "overall_aar": aar(overall),
        "per_region_aar": {k: {"n_correct": v[0], "n_total": v[1], "aar": aar(v)}
                           for k, v in reg_acc.items()},
        "cdr3_aar_by_position_from_start": {ch: [aar(p) for p in cdr3_pos[ch]] for ch in ("vh", "vl")},
        "cdr3_aar_by_position_from_end":   {ch: [aar(p) for p in cdr3_pos_end[ch]] for ch in ("vh", "vl")},
        "elapsed_seconds": time.time() - t0,
        "ae_ckpt": str(args.ae_ckpt),
        "comment": "Teacher-forced per-region AAR; regions via conserved-motif anchoring (same as region-diversity analysis).",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n=== overall AAR {out['overall_aar']*100:.2f}% ===")
    print("per-region AAR:")
    for ch in ("vh", "vl"):
        row = "  " + ch.upper() + " " + "  ".join(
            f"{r}={out['per_region_aar'][f'{ch}_{r}']['aar']*100:.1f}%"
            for r in REGION_ORDER if out['per_region_aar'][f'{ch}_{r}']['aar'] is not None)
        print(row)
    print(f"[done] wrote {args.out}  ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()

