#!/usr/bin/env python3
"""
Plot per-region reconstruction AAR from two independent region-definition methods
side by side: our conserved-motif anchoring vs standard IMGT numbering (anarcii).
Agreement across both methods shows the "VH CDR3 > CDR1/CDR2" result is real and
not an artifact of the motif boundaries.

Motif numbers are read from results/region_aar.json (the 5000-seq run).
IMGT numbers are the recovery_aar_imgt.json run (anarcii, 1000 test seqs, 0 fails).
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
REGIONS = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
IS_CDR = [r.startswith("CDR") for r in REGIONS]

# motif-anchored AAR (read from the existing results json)
motif_json = json.load(open(RES / "region_aar.json"))["per_region_aar"]
motif = {ch: [100 * motif_json[f"{ch}_{r}"]["aar"] for r in REGIONS] for ch in ("vh", "vl")}

# IMGT (anarcii) AAR from recovery_aar_imgt.json
imgt = {
    "vh": [97.6, 94.3, 97.3, 92.3, 97.7, 97.0, 99.4],
    "vl": [98.7, 96.5, 98.8, 96.6, 99.0, 97.4, 99.8],
}

C_MOTIF, C_IMGT = "#3B6FB6", "#E08A3C"
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
x = np.arange(len(REGIONS)); w = 0.38

for ax, ch, title in zip(axes, ("vh", "vl"), ("Heavy chain (VH)", "Light chain (VL)")):
    # shade CDR columns
    for i, cdr in enumerate(IS_CDR):
        if cdr:
            ax.axvspan(i - 0.5, i + 0.5, color="#F2C94C", alpha=0.16, zorder=0)
    b1 = ax.bar(x - w / 2, motif[ch], w, label="Motif anchoring (ours)", color=C_MOTIF, zorder=3)
    b2 = ax.bar(x + w / 2, imgt[ch], w, label="IMGT numbering (ANARCII)", color=C_IMGT, zorder=3)
    for b in (b1, b2):
        for r in b:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.12,
                    f"{r.get_height():.1f}", ha="center", va="bottom", fontsize=7.5)
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(REGIONS, fontsize=10)
    ax.set_ylim(90, 101)
    ax.grid(axis="y", color="#DDDDDD", zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

axes[0].set_ylabel("Reconstruction accuracy (AAR, %)", fontsize=11)

# annotate the key heavy-chain ordering
axvh = axes[0]
axvh.annotate("CDR3 > CDR1 > CDR2\nunder BOTH methods",
              xy=(5, 97.0), xytext=(3.1, 91.4),
              fontsize=9.5, weight="bold", color="#B5532A",
              ha="center",
              arrowprops=dict(arrowstyle="->", color="#B5532A", lw=1.4))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
           fontsize=10.5, bbox_to_anchor=(0.5, 1.02))
fig.suptitle("Per-region reconstruction accuracy: two independent region definitions agree",
             y=1.07, fontsize=13.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.99])
for ext in ("png", "pdf"):
    fig.savefig(RES / f"figR8_motif_vs_imgt.{ext}", dpi=170, bbox_inches="tight")
print("wrote figR8_motif_vs_imgt.png / .pdf")
