#!/usr/bin/env python3
"""
02_plot_region_metrics.py — region-decomposed figures, mirroring the original
fig1/fig2/fig3 style but extended to ALL 7 regions.

  figR1_region_diversity_profile  — 4-gram diversity across 7 regions (VH/VL),
                                     key configs side by side  (extends fig2)
  figR2_region_cfg_plateau        — LD4LG CFG sweep per CDR (CDR1/CDR2/CDR3),
                                     DPLM refs                 (extends fig1)
  figR3_region_length_distribution— length distribution per region (VH/VL),
                                     frameworks tight, CDRs spread (extends fig3)
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "results"
data = {r["name"]: r for r in json.load(open(OUT / "region_metrics.json"))}
REGIONS = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
IS_CDR = {"FR1": 0, "CDR1": 1, "FR2": 0, "CDR2": 1, "FR3": 0, "CDR3": 1, "FR4": 0}

def div(cfg, chain, reg):
    return data[cfg]["regions"][reg][chain]["div_4gram"]

# FIG R1: diversity profile across regions
def fig_profile():
    show = [("dplm_default", "DPLM default", "#888888"),
            ("dplm_tuned",   "DPLM tuned",   "#d62728"),
            ("ld4lg_w2p0",   "LD4LG (w=2)",  "#1f77b4")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(REGIONS)); w = 0.26
    for ax, chain, title in [(ax1, "vh", "Heavy chain (VH)"), (ax2, "vl", "Light chain (VL)")]:
        for j, (cfg, lab, col) in enumerate(show):
            vals = [div(cfg, chain, r) for r in REGIONS]
            ax.bar(x + (j-1)*w, vals, w, label=lab, color=col)
        # shade CDR columns
        for i, r in enumerate(REGIONS):
            if IS_CDR[r]:
                ax.axvspan(i-0.5, i+0.5, color="#ffe9b3", alpha=0.35, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(REGIONS)
        ax.set_ylabel("4-gram diversity"); ax.set_title(title)
        ax.set_ylim(0, 0.55); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("Diversity is concentrated in the CDRs (shaded); frameworks are near-constant across all models\n"
                 "CDR3 ≫ CDR2 ≈ CDR1 ≫ FR1–FR4 — the per-region view the full-sequence metric hides",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "figR1_region_diversity_profile.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "figR1_region_diversity_profile.pdf", bbox_inches="tight")
    plt.close(); print(" figR1_region_diversity_profile")

# FIG R2: CFG plateau per CDR
def fig_plateau():
    ld = [("ld4lg_w1p0",1.0),("ld4lg_w1p5",1.5),("ld4lg_w2p0",2.0),
          ("ld4lg_w3p0",3.0),("ld4lg_w5p0",5.0)]
    ws = [w for _,w in ld]
    cdrs = [("CDR1","#2ca02c","o"),("CDR2","#ff7f0e","s"),("CDR3","#1f77b4","D")]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, chain, title in [(ax1,"vh","Heavy chain (VH)"),(ax2,"vl","Light chain (VL)")]:
        for reg, col, mk in cdrs:
            y = [div(c, chain, reg) for c,_ in ld]
            ax.plot(ws, y, marker=mk, markersize=8, linewidth=2.3, color=col, label=f"LD4LG {reg}")
            # DPLM tuned reference for this region
            ax.axhline(div("dplm_tuned", chain, reg), color=col, ls="--", lw=1, alpha=0.6)
        ax.set_xlabel("CFG weight $w$"); ax.set_ylabel("4-gram diversity")
        ax.set_title(title); ax.set_xticks(ws); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="center right")
    fig.suptitle("CFG plateau holds in every CDR, not just CDR3\n"
                 "Solid = LD4LG CFG sweep; dashed = DPLM-tuned reference (same colour per region)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "figR2_region_cfg_plateau.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "figR2_region_cfg_plateau.pdf", bbox_inches="tight")
    plt.close(); print(" figR2_region_cfg_plateau")

# FIG R3: region length distribution
def fig_length():
    cfg = "ld4lg_w2p0"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
    cdr_palette = {"CDR1":"#2ca02c","CDR2":"#ff7f0e","CDR3":"#1f77b4"}
    for ax, chain, title in [(ax1,"vh","Heavy chain (VH)"),(ax2,"vl","Light chain (VL)")]:
        # plot only the CDRs (frameworks are delta spikes; their fixed lengths annotated below)
        for reg in ["CDR1","CDR2","CDR3"]:
            ls = data[cfg]["regions"][reg][chain]["length"]
            bins = ls["histogram_bins"]; counts = ls["histogram_counts"]
            tot = sum(counts) or 1
            dens = [c/tot for c in counts]
            ax.bar(bins, dens, width=0.9, alpha=0.55, color=cdr_palette[reg],
                   label=f"{reg} (μ={ls.get('mean',0):.1f}, σ={ls.get('std',0):.1f})")
        # annotate fixed framework lengths
        fr_txt = "Frameworks (σ≈0):\n" + "\n".join(
            f"  {r} = {data[cfg]['regions'][r][chain]['length'].get('mean',0):.0f} aa fixed"
            for r in ["FR1","FR2","FR3","FR4"])
        ax.text(0.97, 0.97, fr_txt, transform=ax.transAxes, ha="right", va="top",
                fontsize=9, color="#444", bbox=dict(boxstyle="round", fc="#f0f0f0", ec="#bbb"))
        ax.set_xlim(2, 30); ax.set_xlabel("Region length (AA)"); ax.set_ylabel("Frequency")
        ax.set_title(title); ax.grid(axis="y", alpha=0.3); ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("Region length distributions (LD4LG w=2): frameworks are length-invariant (σ≈0, boxed),\n"
                 "CDRs carry the length variation — widest in CDR3, the junctional region",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    fig.savefig(OUT / "figR3_region_length_distribution.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "figR3_region_length_distribution.pdf", bbox_inches="tight")
    plt.close(); print(" figR3_region_length_distribution")

if __name__ == "__main__":
    fig_profile(); fig_plateau(); fig_length()
    print("done")

