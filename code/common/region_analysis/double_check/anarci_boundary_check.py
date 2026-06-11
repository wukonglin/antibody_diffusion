#!/usr/bin/env python3
"""
anarci_boundary_check.py  —  double-check #4

Cross-checks our motif-anchored region boundaries against the standard IMGT
numbering from ANARCI (via the `abnumber` wrapper). For a sample of paired
sequences it extracts CDR1/CDR2/CDR3 of each chain two ways:
  (a) our motif anchoring (lib_regions.extract_regions_pair)
  (b) IMGT numbering (abnumber.Chain, scheme="imgt")
and reports, per region, how often the two agree exactly, the mean length
difference, and the mean residue overlap. If the two largely agree, our AAR-by-
region numbers are effectively using IMGT regions and the "CDR3 > CDR1/CDR2"
result is not an artifact of the motif boundaries.

No model is needed — just sequences. Runs wherever abnumber/ANARCI is installed.

Install (once):
    pip install abnumber          # pulls in ANARCI + HMMER
Run:
    python anarci_boundary_check.py \
        --fasta /path/to/any_paired.fasta \
        --n 200 --seed 0 \
        --out anarci_boundary_check.json

A good --fasta is the training subset (train_subset_50k.fasta, real sequences)
or any one of the generated sample fastas.
"""
import argparse, json, os, random, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_regions import extract_regions_pair          # our motif method
from lib_cdr3 import split_vh_vl, parse_fasta

try:
    from abnumber import Chain
except Exception as e:
    sys.exit("Could not import abnumber. Install it with:  pip install abnumber\n"
             f"(import error: {e})")

REGIONS = ["CDR1", "CDR2", "CDR3"]

def imgt_cdrs(chain_seq):
    """Return {CDR1,CDR2,CDR3: str} via IMGT numbering, or None on failure."""
    try:
        c = Chain(chain_seq, scheme="imgt")
        return {"CDR1": c.cdr1_seq, "CDR2": c.cdr2_seq, "CDR3": c.cdr3_seq}
    except Exception:
        return None

def overlap(a, b):
    """Length of the longest common substring (residue overlap proxy)."""
    if not a or not b:
        return 0
    # simple LCS-substring
    best = 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            cur = dp[j]
            dp[j] = prev + 1 if a[i-1] == b[j-1] else 0
            best = max(best, dp[j]); prev = cur
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="anarci_boundary_check.json")
    args = ap.parse_args()

    recs = [s for _, s in parse_fasta(args.fasta)]
    random.seed(args.seed)
    if len(recs) > args.n:
        recs = random.sample(recs, args.n)

    # accumulators per (chain, region)
    stat = {f"{ch}_{r}": {"n": 0, "exact": 0, "len_motif": 0, "len_imgt": 0,
                          "len_absdiff": 0, "overlap": 0}
            for ch in ("vh", "vl") for r in REGIONS}
    examples = defaultdict(list)
    n_seqs = n_both = 0

    for seq in recs:
        n_seqs += 1
        mr = extract_regions_pair(seq)
        if mr.get("reason") == "no_linker":
            continue
        vh, vl = split_vh_vl(seq)
        imgt = {"vh": imgt_cdrs(vh), "vl": imgt_cdrs(vl)}
        if imgt["vh"] is None or imgt["vl"] is None:
            continue
        n_both += 1
        for ch in ("vh", "vl"):
            for r in REGIONS:
                m = mr[ch][r]; g = imgt[ch][r]
                if not m or not g:
                    continue
                s = stat[f"{ch}_{r}"]
                s["n"] += 1
                s["exact"] += int(m == g)
                s["len_motif"] += len(m); s["len_imgt"] += len(g)
                s["len_absdiff"] += abs(len(m) - len(g))
                s["overlap"] += overlap(m, g)
                if m != g and len(examples[f"{ch}_{r}"]) < 5:
                    examples[f"{ch}_{r}"].append({"motif": m, "imgt": g})

    out = {"fasta": args.fasta, "n_sequences": n_seqs,
           "n_both_methods_ok": n_both, "per_region": {}, "examples": dict(examples)}
    print(f"sequences: {n_seqs}   both methods OK: {n_both}\n")
    print(f"{'chain/region':14}{'n':>6}{'exact%':>9}{'len_motif':>11}{'len_imgt':>10}"
          f"{'|dlen|':>8}{'overlap':>9}")
    for ch in ("vh", "vl"):
        for r in REGIONS:
            s = stat[f"{ch}_{r}"]; n = s["n"] or 1
            row = {"n": s["n"],
                   "exact_match_pct": 100 * s["exact"] / n,
                   "mean_len_motif": s["len_motif"] / n,
                   "mean_len_imgt": s["len_imgt"] / n,
                   "mean_abs_len_diff": s["len_absdiff"] / n,
                   "mean_overlap": s["overlap"] / n}
            out["per_region"][f"{ch}_{r}"] = row
            print(f"{ch.upper()+' '+r:14}{s['n']:>6}{row['exact_match_pct']:>8.1f}%"
                  f"{row['mean_len_motif']:>11.1f}{row['mean_len_imgt']:>10.1f}"
                  f"{row['mean_abs_len_diff']:>8.2f}{row['mean_overlap']:>9.1f}")

    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[done] wrote {args.out}")
    print("\nReading the table:")
    print("  exact% high + |dlen| small  -> our motif regions ~ IMGT; result is robust.")
    print("  large |dlen| (esp. CDR1/CDR2) -> our boundaries differ from IMGT; re-run")
    print("  AAR with IMGT regions before trusting the CDR3-vs-CDR1/2 ordering.")

if __name__ == "__main__":
    main()
