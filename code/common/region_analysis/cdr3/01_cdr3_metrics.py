#!/usr/bin/env python3
"""
01_cdr3_metrics.py
For every config (plain DPLM, swept DPLM, LD4LG w=1/1.5/2/3/5) compute:

   CDR3 4-gram diversity (VH CDR3, VL CDR3, combined)
   Full-sequence 4-gram diversity (as a baseline)
   CDR3 length distribution (mean, std, min, max, percentiles)
   CDR3 extraction success rate (fraction of generated seqs with a conserved motif)

Why these metrics matter:
  - Full-sequence diversity is dominated by the framework regions (the framework
    is nearly identical within a V-family), which masks the real differences in
    model behavior.
  - CDR3 is the truly hypervariable region and the source of binding specificity;
    it is what antibody design cares about most.
  - If LD4LG / DPLM are nearly identical on the framework (both learn it trivially),
    the real model difference lives in CDR3 — this is what turns the full-sequence
    Pareto trade-off into a region-aware one.

Output:
  results/cdr3_metrics.json — full numerical results
  console               — a human-readable comparison table

Usage:
  python 01_cdr3_metrics.py
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# add lib_cdr3.py to the import path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_cdr3 import extract_cdr3_pair, parse_fasta

# Config: all configs to analyze

RESULTS_ROOT = Path("/Users/susi/Documents/Claude/Projects/DL Final/ab_ld4lg_results")
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# (config_name, samples_dir, display_name)
CONFIGS = [
    ("dplm_default",   "samples_dplm_stochastic", "DPLM default (T=1.0, p=0.95)"),
    ("dplm_tuned",     "samples_dplm_tuned",      "DPLM tuned (T=1.3, p=0.99)"),
    ("ld4lg_w1p0",     "samples_ld4lg_w1p0",      "LD4LG w=1.0"),
    ("ld4lg_w1p5",     "samples_ld4lg_w1p5",      "LD4LG w=1.5"),
    ("ld4lg_w2p0",     "samples",                 "LD4LG w=2.0 (plain)"),
    ("ld4lg_w3p0",     "samples_ld4lg_w3p0",      "LD4LG w=3.0"),
    ("ld4lg_w5p0",     "samples_ld4lg_w5p0",      "LD4LG w=5.0"),
]

# 18 cells in fixed iteration order
CELLS = []
for iso in ["IGHM", "IGHG", "IGHA"]:
    for vfam in ["IGHV1", "IGHV3", "IGHV4"]:
        for loc in ["K", "L"]:
            CELLS.append(f"{iso}_{vfam}_{loc}")

# Metric primitives

def four_gram_diversity(sequences):
    """4-gram diversity = #unique_4grams / #total_4grams over a list of strings."""
    if not sequences:
        return 0.0
    total = 0
    unique = set()
    for s in sequences:
        for i in range(len(s) - 3):
            gram = s[i:i+4]
            unique.add(gram)
            total += 1
    return len(unique) / total if total > 0 else 0.0

def length_stats(lengths):
    """Return dict of summary stats for a list of ints."""
    if not lengths:
        return {"n": 0}
    arr = np.array(lengths)
    return {
        "n":      int(len(arr)),
        "mean":   float(arr.mean()),
        "std":    float(arr.std()),
        "min":    int(arr.min()),
        "max":    int(arr.max()),
        "p25":    float(np.percentile(arr, 25)),
        "p50":    float(np.percentile(arr, 50)),
        "p75":    float(np.percentile(arr, 75)),
        # Coarse histogram for plotting later
        "histogram_bins": list(range(5, 31)),  # 5..30 inclusive
        "histogram_counts": np.histogram(arr, bins=list(range(5, 32)))[0].tolist(),
    }

# Per-config analysis

def analyze_config(name, samples_dir_name, display):
    """Run all CDR3 metrics for one config, return dict."""
    samples_dir = RESULTS_ROOT / samples_dir_name
    if not samples_dir.exists():
        print(f"  {samples_dir} not found — skipping")
        return None

    print(f"\n-> {display}")
    print(f"   dir: {samples_dir}")

    full_seqs = []
    vh_cdr3s = []
    vl_cdr3s = []
    n_total = 0
    n_extract_ok = 0
    n_no_linker = 0
    n_no_heavy_motif = 0
    n_no_light_motif = 0

    for cell in CELLS:
        fpath = samples_dir / f"{cell}.fasta"
        if not fpath.exists():
            print(f"   missing cell file: {cell}")
            continue
        for _, seq in parse_fasta(str(fpath)):
            n_total += 1
            full_seqs.append(seq)
            res = extract_cdr3_pair(seq)
            if res["ok"]:
                vh_cdr3s.append(res["vh_cdr3"])
                vl_cdr3s.append(res["vl_cdr3"])
                n_extract_ok += 1
            else:
                if res["reason"] == "no_linker":
                    n_no_linker += 1
                elif res["reason"] == "no_motif_heavy":
                    n_no_heavy_motif += 1
                elif res["reason"] == "no_motif_light":
                    n_no_light_motif += 1
                elif res["reason"] == "no_motif_either_chain":
                    n_no_heavy_motif += 1
                    n_no_light_motif += 1

    # Metrics
    vh_lens = [len(s) for s in vh_cdr3s]
    vl_lens = [len(s) for s in vl_cdr3s]
    combined_cdr3s = [vh + vl for vh, vl in zip(vh_cdr3s, vl_cdr3s)]

    result = {
        "name": name,
        "display": display,
        "samples_dir": str(samples_dir),
        "n_sequences_total": n_total,
        "n_cdr3_extraction_ok": n_extract_ok,
        "extraction_failure_breakdown": {
            "no_linker":         n_no_linker,
            "no_heavy_motif":    n_no_heavy_motif,
            "no_light_motif":    n_no_light_motif,
        },
        "extraction_success_rate": (n_extract_ok / n_total) if n_total else 0.0,
        # 4-gram diversity at different scopes
        "div_4gram_full_seq":  four_gram_diversity(full_seqs),
        "div_4gram_vh_cdr3":   four_gram_diversity(vh_cdr3s),
        "div_4gram_vl_cdr3":   four_gram_diversity(vl_cdr3s),
        "div_4gram_combined_cdr3": four_gram_diversity(combined_cdr3s),
        # Length distributions
        "vh_cdr3_length": length_stats(vh_lens),
        "vl_cdr3_length": length_stats(vl_lens),
    }

    # short console summary
    print(f"   {n_total} sequences, CDR3 extracted: {n_extract_ok} ({100*n_extract_ok/max(n_total,1):.1f}%)")
    print(f"   4-gram diversity:")
    print(f"      full seq    : {result['div_4gram_full_seq']:.4f}")
    print(f"      VH CDR3     : {result['div_4gram_vh_cdr3']:.4f}")
    print(f"      VL CDR3     : {result['div_4gram_vl_cdr3']:.4f}")
    print(f"      combined    : {result['div_4gram_combined_cdr3']:.4f}")
    print(f"   VH CDR3 length: mean={result['vh_cdr3_length'].get('mean', 0):.1f}, "
          f"p50={result['vh_cdr3_length'].get('p50', 0):.0f}, "
          f"range=[{result['vh_cdr3_length'].get('min', 0)}, {result['vh_cdr3_length'].get('max', 0)}]")
    print(f"   VL CDR3 length: mean={result['vl_cdr3_length'].get('mean', 0):.1f}, "
          f"p50={result['vl_cdr3_length'].get('p50', 0):.0f}, "
          f"range=[{result['vl_cdr3_length'].get('min', 0)}, {result['vl_cdr3_length'].get('max', 0)}]")
    return result

# Main

def main():
    print("=" * 75)
    print("CDR3 region-decomposed metrics for all configs")
    print("=" * 75)

    all_results = []
    for name, samples_dir, display in CONFIGS:
        res = analyze_config(name, samples_dir, display)
        if res is not None:
            all_results.append(res)

    # Save full JSON
    out_path = OUTPUT_DIR / "cdr3_metrics.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n Saved: {out_path}")

    # Summary table (the thing you'll cite in report / office hour)
    print("\n" + "=" * 95)
    print("SUMMARY — 4-gram diversity by region")
    print("=" * 95)
    header = f"{'Config':<32} {'Full-seq':>10} {'VH CDR3':>10} {'VL CDR3':>10} {'Comb. CDR3':>12} {'Extract%':>10}"
    print(header)
    print("-" * 95)
    for r in all_results:
        print(f"{r['display']:<32} "
              f"{r['div_4gram_full_seq']:>10.4f} "
              f"{r['div_4gram_vh_cdr3']:>10.4f} "
              f"{r['div_4gram_vl_cdr3']:>10.4f} "
              f"{r['div_4gram_combined_cdr3']:>12.4f} "
              f"{100*r['extraction_success_rate']:>9.1f}%")
    print("=" * 95)

    print("\n" + "=" * 95)
    print("SUMMARY — CDR3 length (mean ± std,  p50,  range)")
    print("=" * 95)
    header = f"{'Config':<32} {'VH CDR3':>22} {'VL CDR3':>22}"
    print(header)
    print("-" * 95)
    for r in all_results:
        vh = r["vh_cdr3_length"]
        vl = r["vl_cdr3_length"]
        if vh.get("n", 0) > 0:
            vh_str = f"{vh['mean']:.1f}±{vh['std']:.1f}, p50={vh['p50']:.0f}, [{vh['min']},{vh['max']}]"
        else:
            vh_str = "(no data)"
        if vl.get("n", 0) > 0:
            vl_str = f"{vl['mean']:.1f}±{vl['std']:.1f}, p50={vl['p50']:.0f}, [{vl['min']},{vl['max']}]"
        else:
            vl_str = "(no data)"
        print(f"{r['display']:<32} {vh_str:>22} {vl_str:>22}")
    print("=" * 95)

    print(f"\nDone. JSON output at: {out_path}")
    print(f"Plot from JSON if needed; this script prints summary inline.\n")

if __name__ == "__main__":
    main()

