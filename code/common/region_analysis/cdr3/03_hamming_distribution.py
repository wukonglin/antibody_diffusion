#!/usr/bin/env python3
"""
03_hamming_distribution.py
对每个 config,算每条生成序列到训练集**最近邻**的 Hamming distance,
然后画分布——比 "exact match yes/no" + "Hamming-3 binary check" 更
rigorous 的 memorization analysis.

为啥用 Hamming distance distribution:
  - 你们 §4 已 check exact match (0/9216) 和 Hamming-3 near-match (0%)
  - 但这只是 binary 信号 — 不知道模型 距 training distribution **有多远**
  - 完整 distribution 告诉你:
      Mass 集中在低 Hamming (e.g. 20-40):接近训练分布(in-distribution)
      Mass 在高 Hamming (60+):extrapolation 到分布外
  - 不同 config 的 distribution shape 区分 in-domain interpolation
    vs out-of-domain exploration — 这是 "diversity" 的另一种 lens

Hamming distance 定义在等长序列上,所以我们做 length-matched comparison:
  - 对每条 generated seq (长度 L):
      只 跟训练集里长度 = L 的序列比 Hamming
      取 min Hamming 作为 "距 training set 最近的距离"

Compute cost: 50K 训练 / 平均长度 bucket size 3K × 9216 generated × 7 configs
            ≈ 200M ops × O(L=240),~3-5 分钟 with numpy vectorization
"""

import json
import sys
import os
from collections import defaultdict
from pathlib import Path
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_cdr3 import parse_fasta

# Config

RESULTS_ROOT = Path("/Users/susi/Documents/Claude/Projects/DL Final/ab_ld4lg_results")
TRAIN_FASTA = RESULTS_ROOT / "train_subset_50k.fasta"
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIGS = [
    ("dplm_default",   "samples_dplm_stochastic", "DPLM default (T=1.0, p=0.95)"),
    ("dplm_tuned",     "samples_dplm_tuned",      "DPLM tuned (T=1.3, p=0.99)"),
    ("ld4lg_w1p0",     "samples_ld4lg_w1p0",      "LD4LG w=1.0"),
    ("ld4lg_w1p5",     "samples_ld4lg_w1p5",      "LD4LG w=1.5"),
    ("ld4lg_w2p0",     "samples",                 "LD4LG w=2.0 (plain)"),
    ("ld4lg_w3p0",     "samples_ld4lg_w3p0",      "LD4LG w=3.0"),
    ("ld4lg_w5p0",     "samples_ld4lg_w5p0",      "LD4LG w=5.0"),
]

CELLS = []
for iso in ["IGHM", "IGHG", "IGHA"]:
    for vfam in ["IGHV1", "IGHV3", "IGHV4"]:
        for loc in ["K", "L"]:
            CELLS.append(f"{iso}_{vfam}_{loc}")

# Encode sequence string to uint8 numpy array

AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {c: i for i, c in enumerate(AA)}

def seq_to_array(s):
    """Encode AA string to uint8 numpy. Non-standard AA -> 255 (won't match)."""
    a = np.full(len(s), 255, dtype=np.uint8)
    for i, c in enumerate(s):
        if c in AA_IDX:
            a[i] = AA_IDX[c]
    return a

# Load training subset -> group by length

def load_train_by_length():
    """
    Returns dict {length: np.ndarray (n_seqs, length) uint8}
    grouping training seqs by length for length-matched comparison.
    """
    if not TRAIN_FASTA.exists():
        print(f"ERROR: training subset not found at {TRAIN_FASTA}")
        print(f"You need to:")
        print(f"  1. ssh into AIDA")
        print(f"  2. cd ~/ab_ld4lg && python /path/to/aida_extract_train_subset.py")
        print(f"  3. scp the output 50K FASTA back to {RESULTS_ROOT}/")
        sys.exit(1)

    print(f"Loading training subset from {TRAIN_FASTA}")
    by_len = defaultdict(list)
    n_total = 0
    for _, seq in parse_fasta(str(TRAIN_FASTA)):
        by_len[len(seq)].append(seq_to_array(seq))
        n_total += 1

    print(f"Loaded {n_total:,} training sequences")
    print(f"Length range: {min(by_len.keys())}–{max(by_len.keys())}")

    # Convert each length bucket to single 2D array
    out = {}
    for L, seqs in by_len.items():
        if len(seqs) < 5:
            continue   # 太少,跳过这个长度
        out[L] = np.stack(seqs)
    return out

# Compute min Hamming to training set for one generated seq

def min_hamming_to_train(gen_array, train_by_len):
    """gen_array: shape (L,) uint8. train_by_len[L]: shape (N_L, L)."""
    L = len(gen_array)
    if L not in train_by_len:
        return None  # no length match
    train_arr = train_by_len[L]
    # Vectorized Hamming: count positions where gen != train
    diffs = (train_arr != gen_array[None, :]).sum(axis=1)
    return int(diffs.min())

# Per-config analysis

def analyze_config(name, samples_dir_name, display, train_by_len):
    samples_dir = RESULTS_ROOT / samples_dir_name
    if not samples_dir.exists():
        print(f"  ️  {samples_dir} not found — skipping")
        return None

    print(f"\n-> {display}")
    t0 = time.time()

    hammings = []
    n_total = 0
    n_no_length_match = 0

    for cell in CELLS:
        fpath = samples_dir / f"{cell}.fasta"
        if not fpath.exists():
            continue
        for _, seq in parse_fasta(str(fpath)):
            n_total += 1
            gen_arr = seq_to_array(seq)
            h = min_hamming_to_train(gen_arr, train_by_len)
            if h is None:
                n_no_length_match += 1
            else:
                hammings.append(h)

    if not hammings:
        print(f"   no valid comparisons!")
        return None

    arr = np.array(hammings)
    elapsed = time.time() - t0
    print(f"   {n_total} sequences in {elapsed:.1f}s "
          f"({n_no_length_match} skipped — no length match in train subset)")
    print(f"   Hamming to nearest training seq:")
    print(f"      min     : {arr.min()}   ← 0 = exact match")
    print(f"      max     : {arr.max()}")
    print(f"      mean    : {arr.mean():.1f}")
    print(f"      p25     : {np.percentile(arr, 25):.0f}")
    print(f"      p50     : {np.percentile(arr, 50):.0f}")
    print(f"      p75     : {np.percentile(arr, 75):.0f}")
    print(f"      # near (≤3) : {(arr <= 3).sum()}  ({100*(arr <= 3).mean():.2f}%)")
    print(f"      # near (≤10): {(arr <= 10).sum()} ({100*(arr <= 10).mean():.2f}%)")

    # histogram (bins of width 5, from 0 to 100)
    hist_bins = list(range(0, 105, 5))
    hist_counts, _ = np.histogram(arr, bins=hist_bins)

    return {
        "name": name,
        "display": display,
        "n_sequences": n_total,
        "n_no_length_match": n_no_length_match,
        "n_compared": len(hammings),
        "min":  int(arr.min()),
        "max":  int(arr.max()),
        "mean": float(arr.mean()),
        "std":  float(arr.std()),
        "p25":  float(np.percentile(arr, 25)),
        "p50":  float(np.percentile(arr, 50)),
        "p75":  float(np.percentile(arr, 75)),
        "n_near_3":  int((arr <= 3).sum()),
        "n_near_10": int((arr <= 10).sum()),
        "histogram_bins":   hist_bins,
        "histogram_counts": hist_counts.tolist(),
        # Save raw (compressed) for plotting later
        "raw_hammings": arr.tolist(),
    }

def main():
    print("=" * 75)
    print("Hamming distance distribution: generated -> nearest training seq")
    print("=" * 75)

    train_by_len = load_train_by_length()

    all_results = []
    for name, samples_dir, display in CONFIGS:
        res = analyze_config(name, samples_dir, display, train_by_len)
        if res is not None:
            all_results.append(res)

    out_path = OUTPUT_DIR / "hamming_distribution.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n Saved: {out_path}")

    # Summary table
    print("\n" + "=" * 105)
    print("SUMMARY — Hamming distance to nearest training seq (length-matched)")
    print("=" * 105)
    header = f"{'Config':<32} {'min':>5} {'p25':>6} {'p50':>6} {'p75':>6} {'mean':>7} {'#≤3':>6} {'#≤10':>7}"
    print(header)
    print("-" * 105)
    for r in all_results:
        print(f"{r['display']:<32} {r['min']:>5} "
              f"{r['p25']:>6.0f} {r['p50']:>6.0f} {r['p75']:>6.0f} {r['mean']:>7.1f} "
              f"{r['n_near_3']:>6} {r['n_near_10']:>7}")
    print("=" * 105)

    # Plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {
        "dplm_default": "#888",
        "dplm_tuned":   "#d62728",
        "ld4lg_w1p0":   "#aecae8",
        "ld4lg_w1p5":   "#87b3d8",
        "ld4lg_w2p0":   "#1f77b4",
        "ld4lg_w3p0":   "#155a8a",
        "ld4lg_w5p0":   "#0c3d5f",
    }
    for r in all_results:
        bins = r["histogram_bins"]
        counts = r["histogram_counts"]
        # Convert to density
        total = sum(counts)
        density = [c / total for c in counts]
        # Bin centers for line plot
        centers = [(bins[i] + bins[i+1]) / 2 for i in range(len(counts))]
        ax.plot(centers, density,
                marker="o", markersize=4, linewidth=2,
                color=colors.get(r["name"], "black"),
                label=r["display"])

    ax.set_xlabel("Hamming distance to nearest training sequence (length-matched)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Memorization-vs-novelty profile across 7 configs\n"
                 "Mass at low Hamming = in-distribution; high Hamming = extrapolation",
                 fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig4_hamming_distribution.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig4_hamming_distribution.pdf", bbox_inches="tight")
    plt.close()
    print(f"\n Saved fig4: Hamming distance distribution plot")

if __name__ == "__main__":
    main()

