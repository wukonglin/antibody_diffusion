#!/usr/bin/env python3
"""
04_plot_region_aar.py — plot per-region reconstruction accuracy (AAR) +
CDR3 per-position zoom-in, from recovery_aar_byregion.json (run on AIDA).

  figR5_region_aar          — per-region AAR bars (VH/VL, 7 regions), CDRs shaded
  figR6_cdr3_aar_zoom       — CDR3 per-position AAR (from start & from end)
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

SRC = "/sessions/practical-keen-lamport/mnt/uploads/recovery_aar_byregion.json"
OUT = Path("/sessions/practical-keen-lamport/mnt/outputs/region_work/results")
d = json.load(open(SRC))
pr = d["per_region_aar"]
REGIONS = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
IS_CDR = {"FR1": 0, "CDR1": 1, "FR2": 0, "CDR2": 1, "FR3": 0, "CDR3": 1, "FR4": 0}

# figR5: per-region AAR bars
def fig_region_aar():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    x = np.arange(len(REGIONS))
    for ax, ch, title, col in [(ax1, "vh", "Heavy chain (VH)", "#1f77b4"),
                               (ax2, "vl", "Light chain (VL)", "#ff7f0e")]:
        vals = [100 * pr[f"{ch}_{r}"]["aar"] for r in REGIONS]
        bars = ax.bar(x, vals, color=[col if IS_CDR[r] else "#cccccc" for r in REGIONS],
                      edgecolor="#444", linewidth=0.6)
        for i, r in enumerate(REGIONS):
            if IS_CDR[r]:
                ax.axvspan(i - 0.5, i + 0.5, color="#ffe9b3", alpha=0.35, zorder=0)
            ax.text(i, vals[i] + 0.15, f"{vals[i]:.1f}", ha="center", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(REGIONS)
        ax.set_title(title); ax.set_ylim(90, 100.5)
        ax.set_ylabel("Reconstruction accuracy AAR (%)")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Autoencoder reconstruction accuracy by region (test set, overall AAR = {100*d['overall_aar']:.2f}%)\n"
                 "Frameworks reconstruct near-perfectly; the CDRs (shaded) are hardest — the variable region the latent must compress",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "figR5_region_aar.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "figR5_region_aar.pdf", bbox_inches="tight")
    plt.close(); print(" figR5_region_aar")

# figR6: CDR3 per-position zoom-in
def fig_cdr3_zoom():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.0))
    for ax, ch, title in [(ax1, "vh", "Heavy chain — VH CDR3"),
                          (ax2, "vl", "Light chain — VL CDR3")]:
        start = [100 * v for v in d["cdr3_aar_by_position_from_start"][ch] if v is not None]
        end = [100 * v for v in d["cdr3_aar_by_position_from_end"][ch] if v is not None]
        n = 15 if ch == "vh" else 11   # show first ~mean-length positions
        ax.plot(range(1, len(start[:n]) + 1), start[:n], marker="o", color="#1f77b4",
                lw=2.2, label="from CDR3 start (N-term)")
        ax.plot(range(1, len(end[:n]) + 1), end[:n], marker="s", color="#d62728",
                lw=2.2, label="from CDR3 end (C-term)")
        ax.set_xlabel("Position offset within CDR3 (AA)")
        ax.set_ylabel("Reconstruction accuracy AAR (%)")
        ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("CDR3 reconstruction accuracy, position by position\n"
                 "Accuracy dips toward the middle of CDR3 — the junctional core (V(D)J + N-additions) is hardest to recover",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "figR6_cdr3_aar_zoom.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT / "figR6_cdr3_aar_zoom.pdf", bbox_inches="tight")
    plt.close(); print(" figR6_cdr3_aar_zoom")

if __name__ == "__main__":
    fig_region_aar(); fig_cdr3_zoom()
    # also save a tidy copy of the json with the results
    json.dump(d, open(OUT / "region_aar.json", "w"), indent=2)
    print("done")

