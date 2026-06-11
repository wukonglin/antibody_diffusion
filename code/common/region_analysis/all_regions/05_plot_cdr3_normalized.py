import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("/sessions/practical-keen-lamport/mnt/uploads/recovery_aar_byregion.json"))
OUT = Path("/sessions/practical-keen-lamport/mnt/outputs/region_work/results")
overall = 100 * d["overall_aar"]

# median CDR3 lengths (from earlier region metrics): VH~15, VL~10
LEN = {"vh": 15, "vl": 10}
COL = {"vh": "#1f77b4", "vl": "#d62728"}
LAB = {"vh": "Heavy chain (VH CDR3)", "vl": "Light chain (VL CDR3)"}

def nc_profile(ch):
    """Stitch a N->C profile: each position attributed to its nearer-end measurement."""
    L = LEN[ch]
    s = d["cdr3_aar_by_position_from_start"][ch]
    e = d["cdr3_aar_by_position_from_end"][ch]
    xs, ys = [], []
    for j in range(L):
        dN, dC = j, L - 1 - j
        v = s[dN] if dN <= dC else e[dC]
        if v is None:
            continue
        xs.append(j / (L - 1))
        ys.append(100 * v)
    return np.array(xs), np.array(ys)

fig, ax = plt.subplots(figsize=(11, 5.6))

# three CDR3 sub-zones (intuitive): V-anchored N-term | junctional core | J-anchored C-term
ax.axvspan(0.00, 0.30, color="#cfe3f7", alpha=0.5, lw=0)
ax.axvspan(0.30, 0.70, color="#f9d5d3", alpha=0.5, lw=0)
ax.axvspan(0.70, 1.00, color="#d6ecd6", alpha=0.5, lw=0)
ax.text(0.15, 91.55, "V-gene anchored", ha="center", va="bottom", fontsize=9, color="#33617f")
ax.text(0.50, 91.55, "junctional core (D + N-additions)", ha="center", va="bottom", fontsize=9, color="#9c4540")
ax.text(0.85, 91.55, "J-gene anchored", ha="center", va="bottom", fontsize=9, color="#3f7a3f")

for ch in ("vh", "vl"):
    x, y = nc_profile(ch)
    ax.plot(x, y, marker="o", markersize=6, linewidth=2.6, color=COL[ch], label=LAB[ch])

ax.axhline(overall, color="#555", linestyle="--", linewidth=1.3,
           label=f"overall AAR = {overall:.1f}%")
ax.set_xlim(0, 1); ax.set_ylim(91, 100)
ax.set_xlabel("Relative position within CDR3  (0 = N-terminus  →  1 = C-terminus)", fontsize=12)
ax.set_ylabel("Reconstruction accuracy AAR (%)", fontsize=12)
ax.set_title("CDR3 reconstruction accuracy along the chain (N → C)\n"
             "Both ends (germline-anchored) reconstruct well; accuracy dips in the junctional core",
             fontsize=13)
ax.grid(alpha=0.3); ax.legend(loc="upper center", fontsize=10, ncol=3, framealpha=0.95)
fig.tight_layout()
fig.savefig(OUT / "figR6b_cdr3_normalized.png", dpi=150, bbox_inches="tight")
fig.savefig(OUT / "figR6b_cdr3_normalized.pdf", bbox_inches="tight")
print("saved figR6b_cdr3_normalized")
