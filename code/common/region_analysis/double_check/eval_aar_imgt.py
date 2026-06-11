#!/usr/bin/env python3
"""
eval_aar_imgt.py  —  RUN ON AIDA   (double-check #4, full version)

Same teacher-forced reconstruction AAR as eval_recovery_aar_byregion, but the
region boundaries come from STANDARD IMGT numbering instead of our motif anchoring.
If "CDR3 > CDR1/CDR2" survives under proper IMGT boundaries, the result is real
and not an artifact of our motif definitions.

Numbering backend is anarcii (the neural ANARCII), called directly. abnumber is
NOT used: abnumber hard-imports the classic `anarci` module, which is not installed
in this venv (pip pulled in `anarcii`, a different package), so every Chain() call
fails. anarcii works directly.

    pip install anarcii        # already present as an abnumber dependency
Run:
    cd ~/ab_ld4lg && source .venv/bin/activate
    python scripts/eval_aar_imgt.py \
        --data /mnt/beegfs/bulk/mirror/yl4259/ab_ld4lg/processed \
        --ae-ckpt /mnt/beegfs/bulk/mirror/yl4259/ab_ld4lg/runs/ae/autoencoder_latest.pt \
        --out /mnt/beegfs/bulk/mirror/yl4259/ab_ld4lg/eval_reports/recovery_aar_imgt.json \
        --split test --n-samples 2000
Then scp recovery_aar_imgt.json back and send it to me.
"""
import argparse, json, sys, time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# model imports (work under either repo layout)
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

try:
    from anarcii import Anarcii
except Exception as e:
    sys.exit("Could not import anarcii. Install with:  pip install anarcii\n"
             f"(import error: {e})")

LINKER = "GGGGSGGGGS"
REGION_ORDER = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]

# Standard IMGT region boundaries (by IMGT position number).
def region_of(num):
    if   1  <= num <= 26:  return "FR1"
    elif 27 <= num <= 38:  return "CDR1"
    elif 39 <= num <= 55:  return "FR2"
    elif 56 <= num <= 65:  return "CDR2"
    elif 66 <= num <= 104: return "FR3"
    elif 105 <= num <= 117: return "CDR3"
    elif 118 <= num <= 128: return "FR4"
    return None

def regions_from_numbering(result):
    """Given one anarcii result dict, return list of (region, local_residue_index).
    local_residue_index is the 0-based position within the chain string that was
    numbered. Gap entries ('-') consume no residue. Returns None on failure."""
    if not isinstance(result, dict) or result.get("error") is not None:
        return None
    numbering = result.get("numbering")
    if not numbering:
        return None
    out = []
    ri = result.get("query_start", 0)
    for (num, _ins), aa in numbering:
        if aa == "-":
            continue
        reg = region_of(num)
        if reg is not None:
            out.append((reg, ri))
        ri += 1
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--ae-ckpt", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--n-samples", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--anarcii-mode", default="speed", choices=["speed", "accuracy"])
    args = ap.parse_args()

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    print(f"[device] {device}")
    sd = torch.load(args.ae_ckpt, map_location="cpu", weights_only=False)
    cfg = AutoencoderConfig(**sd["cfg"])
    ae = LanguageAutoencoder(cfg).to(device); ae.load_state_dict(sd["model"]); ae.eval()

    ds = PairedAntibodyDataset(args.data, args.split, max_len=cfg.max_source_len)
    n = len(ds) if args.n_samples == 0 else min(args.n_samples, len(ds))
    tok = AATokenizer(); collate = make_collate_fn(tok.pad_id, tok.bos_id)
    dl = DataLoader(torch.utils.data.Subset(ds, list(range(n))),
                    batch_size=args.batch, shuffle=False, collate_fn=collate)
    itos = tok.itos

    # one shared numbering model (neural ANARCII); CPU to match the AE run
    numberer = Anarcii(seq_type="antibody", mode=args.anarcii_mode,
                       batch_size=128, cpu=(device.type == "cpu"), verbose=False)

    reg_acc = {f"{ch}_{r}": [0, 0] for ch in ("vh", "vl") for r in REGION_ORDER}
    n_anarci_fail = 0
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
            correct = (preds == tgt).cpu().tolist(); tgt_cpu = tgt.cpu().tolist()

            # decode each sequence to AA; remember the token position of each residue
            seqs = []     # per-sequence dict or None
            chains = []   # flat list of chain strings (vh, vl, vh, vl, ...)
            for k in range(len(tgt_cpu)):
                ids = tgt_cpu[k]; cor = correct[k]
                aa = []; pos = []
                for j, tid in enumerate(ids):
                    t = itos[tid] if 0 <= tid < len(itos) else "<unk>"
                    if t in ("<eos>", "<pad>"): break
                    if t == "<bos>": continue
                    if t == "<unk>": aa.append("X"); pos.append(j); continue
                    aa.append(t); pos.append(j)
                aa = "".join(aa)
                if LINKER not in aa:
                    seqs.append(None); continue
                idx = aa.index(LINKER); vh = aa[:idx]; voff = idx + len(LINKER); vl = aa[voff:]
                seqs.append({"pos": pos, "cor": cor, "voff": voff})
                chains.append(vh); chains.append(vl)

            # number every chain in this batch in one call
            res = numberer.number(chains) if chains else {}
            results = list(res.values()) if isinstance(res, dict) else list(res)
            if len(results) != len(chains):
                sys.exit(f"[fatal] anarcii returned {len(results)} results for "
                         f"{len(chains)} chains; ordering assumption broken.")

            ci = 0
            for s in seqs:
                if s is None:
                    n_anarci_fail += 1
                    continue
                hres = results[ci]; lres = results[ci + 1]; ci += 2
                pos = s["pos"]; cor = s["cor"]; voff = s["voff"]
                hs = regions_from_numbering(hres)
                ls = regions_from_numbering(lres)
                if not hs and not ls:
                    n_anarci_fail += 1
                    continue
                for reg, ri in (hs or []):
                    if ri >= len(pos): continue
                    j = pos[ri]
                    reg_acc[f"vh_{reg}"][0] += 1 if cor[j] else 0
                    reg_acc[f"vh_{reg}"][1] += 1
                for reg, ri in (ls or []):
                    ai = voff + ri
                    if ai >= len(pos): continue
                    j = pos[ai]
                    reg_acc[f"vl_{reg}"][0] += 1 if cor[j] else 0
                    reg_acc[f"vl_{reg}"][1] += 1

            if (bi + 1) % 5 == 0:
                print(f"  batch {bi+1}/{len(dl)}  (anarci fails so far: {n_anarci_fail})")

    def aar(p): return p[0] / p[1] if p[1] else None
    out = {
        "split": args.split, "n_sequences": n, "region_scheme": "IMGT (anarcii)",
        "n_anarci_fail": n_anarci_fail,
        "per_region_aar": {kk: {"n_correct": v[0], "n_total": v[1], "aar": aar(v)}
                           for kk, v in reg_acc.items()},
        "elapsed_seconds": time.time() - t0, "ae_ckpt": str(args.ae_ckpt),
        "comment": "Teacher-forced per-region AAR with standard IMGT boundaries via "
                   "anarcii; compare against motif-anchored recovery_aar_byregion.json.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n=== per-region AAR (IMGT boundaries) ===")
    for ch in ("vh", "vl"):
        print("  " + ch.upper() + "  " + "  ".join(
            f"{r}={100*out['per_region_aar'][f'{ch}_{r}']['aar']:.1f}%"
            for r in REGION_ORDER if out['per_region_aar'][f'{ch}_{r}']['aar'] is not None))
    print(f"[done] wrote {args.out}  ({time.time()-t0:.1f}s, {n_anarci_fail} ANARCI fails)")

if __name__ == "__main__":
    main()
