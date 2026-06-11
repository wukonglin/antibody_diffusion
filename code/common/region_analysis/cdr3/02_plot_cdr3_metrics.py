#!/usr/bin/env python3
"""
02_plot_cdr3_metrics.py
从 01_cdr3_metrics.py 产生的 JSON 画 3 张 office-hour-ready figures:

  Figure 1: LD4LG CFG plateau at CDR3 level (the headline)
            - x: CFG weight w
            - y: VH/VL CDR3 4-gram diversity
            - DPLM-default + DPLM-tuned 作 horizontal reference lines

  Figure 2: CDR3 vs full-seq diversity decomposition
            - 横向 bar chart,7 configs,full-seq 和 CDR3 并列
            - 视觉证明 "framework 掩盖了真正的 model behavior"

  Figure 3: CDR3 length distribution
            - histogram overlay,3 个 main config

输出:
  results/fig1_ld4lg_cfg_plateau_cdr3.{png,pdf}
  results/fig2_cdr3_vs_fullseq_diversity.{png,pdf}
  results/fig3_cdr3_length_distribution.{png,pdf}
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "results"
JSON_PATH = OUTPUT_DIR / "cdr3_metrics.json"

def load_results():
    with open(JSON_PATH) as f:
        results = json.load(f)
    return {r["name"]: r for r in results}

# Figure 1: LD4LG CFG plateau at CDR3 level

def plot_fig1_cfg_plateau(results):
    """LD4LG CFG sweep on CDR3 vs DPLM reference lines."""
    ld4lg_configs = [
        ("ld4lg_w1p0", 1.0),
        ("ld4lg_w1p5", 1.5),
        ("ld4lg_w2p0", 2.0),
        ("ld4lg_w3p0", 3.0),
        ("ld4lg_w5p0", 5.0),
    ]
    w_values = [w for _, w in ld4lg_configs]
    vh_cdr3 = [results[name]["div_4gram_vh_cdr3"] for name, _ in ld4lg_configs]
    vl_cdr3 = [results[name]["div_4gram_vl_cdr3"] for name, _ in ld4lg_configs]

    dplm_default_vh = results["dplm_default"]["div_4gram_vh_cdr3"]
    dplm_default_vl = results["dplm_default"]["div_4gram_vl_cdr3"]
    dplm_tuned_vh = results["dplm_tuned"]["div_4gram_vh_cdr3"]
    dplm_tuned_vl = results["dplm_tuned"]["div_4gram_vl_cdr3"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # --- VH CDR3 panel ---
    ax1.axhline(dplm_default_vh, color="#888", linestyle="--", linewidth=1.5,
                label=f"DPLM default = {dplm_default_vh:.3f}")
    ax1.axhline(dplm_tuned_vh,   color="#d62728", linestyle="--", linewidth=1.5,
                label=f"DPLM tuned   = {dplm_tuned_vh:.3f}")
    ax1.plot(w_values, vh_cdr3, marker="o", markersize=10, color="#1f77b4",
             linewidth=2.5, label="LD4LG CFG sweep")
    # 标注 plateau range
    vh_range_pct = 100 * (max(vh_cdr3) - min(vh_cdr3)) / min(vh_cdr3)
    ax1.annotate(f"LD4LG range: only +{vh_range_pct:.1f}%\n(plateau confirmed at full 18-cell)",
                 xy=(3, np.mean(vh_cdr3)), xytext=(1.5, 0.36),
                 fontsize=10, color="#1f77b4",
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2))
    ax1.set_xlabel("CFG weight $w$", fontsize=12)
    ax1.set_ylabel("VH CDR3 4-gram diversity", fontsize=12)
    ax1.set_title("VH CDR3 — LD4LG CFG plateau\n(full 18-cell × 512)", fontsize=13)
    ax1.set_xticks(w_values)
    ax1.set_xlim(0.7, 5.3)
    ax1.set_ylim(0.20, 0.55)
    ax1.legend(loc="lower right", fontsize=10)
    ax1.grid(alpha=0.3)

    # --- VL CDR3 panel ---
    ax2.axhline(dplm_default_vl, color="#888", linestyle="--", linewidth=1.5,
                label=f"DPLM default = {dplm_default_vl:.3f}")
    ax2.axhline(dplm_tuned_vl,   color="#d62728", linestyle="--", linewidth=1.5,
                label=f"DPLM tuned   = {dplm_tuned_vl:.3f}")
    ax2.plot(w_values, vl_cdr3, marker="o", markersize=10, color="#1f77b4",
             linewidth=2.5, label="LD4LG CFG sweep")
    vl_range_pct = 100 * (max(vl_cdr3) - min(vl_cdr3)) / min(vl_cdr3)
    ax2.annotate(f"LD4LG range: only +{vl_range_pct:.1f}%",
                 xy=(3, np.mean(vl_cdr3)), xytext=(1.5, 0.22),
                 fontsize=10, color="#1f77b4",
                 arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.2))
    ax2.annotate("DPLM-default L-chain\nessentially mode-collapsed",
                 xy=(2.0, dplm_default_vl + 0.005), xytext=(3.0, 0.10),
                 fontsize=10, color="#666",
                 arrowprops=dict(arrowstyle="->", color="#666", lw=1.0))
    ax2.set_xlabel("CFG weight $w$", fontsize=12)
    ax2.set_ylabel("VL CDR3 4-gram diversity", fontsize=12)
    ax2.set_title("VL CDR3 — LD4LG CFG plateau\n(full 18-cell × 512)", fontsize=13)
    ax2.set_xticks(w_values)
    ax2.set_xlim(0.7, 5.3)
    ax2.set_ylim(0.0, 0.45)
    ax2.legend(loc="center right", fontsize=10)
    ax2.grid(alpha=0.3)

    fig.suptitle("LD4LG CFG plateau confirmed at CDR3 level + full 18-cell scale\n"
                 "(Even at maximum guidance $w=5$, LD4LG CDR3 diversity stays below DPLM-tuned)",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig1_ld4lg_cfg_plateau_cdr3.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig1_ld4lg_cfg_plateau_cdr3.pdf", bbox_inches="tight")
    plt.close()
    print(f" Saved fig1: LD4LG CFG plateau (VH range +{vh_range_pct:.1f}%, VL range +{vl_range_pct:.1f}%)")

# Figure 2: CDR3 vs full-seq diversity decomposition

def plot_fig2_cdr3_vs_fullseq(results):
    """Bar chart showing 10-15x gap between CDR3 and full-seq diversity."""
    order = ["dplm_default", "dplm_tuned",
             "ld4lg_w1p0", "ld4lg_w1p5", "ld4lg_w2p0", "ld4lg_w3p0", "ld4lg_w5p0"]
    labels = ["DPLM\ndefault", "DPLM\ntuned",
              "LD4LG\nw=1", "LD4LG\nw=1.5", "LD4LG\nw=2", "LD4LG\nw=3", "LD4LG\nw=5"]
    full = [results[name]["div_4gram_full_seq"] for name in order]
    vh_c = [results[name]["div_4gram_vh_cdr3"] for name in order]
    vl_c = [results[name]["div_4gram_vl_cdr3"] for name in order]

    x = np.arange(len(order))
    w = 0.27

    fig, ax = plt.subplots(figsize=(13, 5.5))
    b1 = ax.bar(x - w, full, w, label="Full sequence", color="#aaa")
    b2 = ax.bar(x,     vh_c, w, label="VH CDR3",       color="#1f77b4")
    b3 = ax.bar(x + w, vl_c, w, label="VL CDR3",       color="#ff7f0e")

    # 在 CDR3 bar 上标多少倍 of full-seq
    for i, name in enumerate(order):
        ratio_vh = vh_c[i] / max(full[i], 1e-6)
        ax.text(x[i],        vh_c[i] + 0.01, f"×{ratio_vh:.0f}", ha="center", fontsize=9, color="#1f77b4")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("4-gram diversity", fontsize=12)
    ax.set_title("CDR3 vs full-sequence diversity:\nframework dominates full-seq metric, masking real model behavior",
                 fontsize=13)
    ax.legend(loc="upper left", fontsize=11)
    ax.set_ylim(0, 0.6)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig2_cdr3_vs_fullseq_diversity.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig2_cdr3_vs_fullseq_diversity.pdf", bbox_inches="tight")
    plt.close()
    print(" Saved fig2: CDR3 vs full-seq diversity decomposition")

# Figure 3: CDR3 length distribution

def plot_fig3_length_distribution(results):
    """Histogram of VH/VL CDR3 lengths for 3 main models."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))

    colors = {"dplm_default": "#888", "dplm_tuned": "#d62728", "ld4lg_w2p0": "#1f77b4"}
    labels = {"dplm_default": "DPLM default", "dplm_tuned": "DPLM tuned", "ld4lg_w2p0": "LD4LG (w=2)"}

    for key in ["dplm_default", "dplm_tuned", "ld4lg_w2p0"]:
        r = results[key]
        # Reconstruct from histogram (we saved bins and counts)
        vh = r["vh_cdr3_length"]
        bins = vh["histogram_bins"]
        counts = vh["histogram_counts"]
        # normalize to density
        widths = np.diff(bins + [bins[-1] + 1])
        total = sum(counts)
        density = [c / max(total, 1) for c in counts]
        ax1.bar(bins, density, width=0.9, alpha=0.55, color=colors[key],
                label=f"{labels[key]} (mean={vh['mean']:.1f}, std={vh['std']:.1f})")

        vl = r["vl_cdr3_length"]
        bins = vl["histogram_bins"]
        counts = vl["histogram_counts"]
        total = sum(counts)
        density = [c / max(total, 1) for c in counts]
        ax2.bar(bins, density, width=0.9, alpha=0.55, color=colors[key],
                label=f"{labels[key]} (mean={vl['mean']:.1f}, std={vl['std']:.1f})")

    ax1.set_xlabel("VH CDR3 length (AA)", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title("VH CDR3 length distribution", fontsize=13)
    ax1.legend(fontsize=10)
    ax1.set_xlim(2, 30)
    ax1.grid(axis="y", alpha=0.3)

    ax2.set_xlabel("VL CDR3 length (AA)", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title("VL CDR3 length distribution", fontsize=13)
    ax2.legend(fontsize=10)
    ax2.set_xlim(2, 30)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle("CDR3 length distribution: DPLM-default produces tight,\n"
                 "near-deterministic lengths; LD4LG matches natural antibody variation",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "fig3_cdr3_length_distribution.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "fig3_cdr3_length_distribution.pdf", bbox_inches="tight")
    plt.close()
    print(" Saved fig3: CDR3 length distribution")

# Main

def main():
    results = load_results()
    print(f"Loaded {len(results)} configs from {JSON_PATH}")
    plot_fig1_cfg_plateau(results)
    plot_fig2_cdr3_vs_fullseq(results)
    plot_fig3_length_distribution(results)
    print(f"\n3 figures saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()

