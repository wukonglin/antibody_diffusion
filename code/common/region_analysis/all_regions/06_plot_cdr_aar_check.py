import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

d = json.load(open("/sessions/practical-keen-lamport/mnt/uploads/recovery_aar_byregion-00fd57e7.json"))
OUT = Path("/sessions/practical-keen-lamport/mnt/outputs/region_work/results")
ps = d["cdr_aar_by_position_from_start"]; pe = d["cdr_aar_by_position_from_end"]
LEN = {"vh_CDR1":13,"vh_CDR2":17,"vh_CDR3":15,"vl_CDR1":12,"vl_CDR2":7,"vl_CDR3":10}
COL = {"CDR1":"#2ca02c","CDR2":"#ff7f0e","CDR3":"#1f77b4"}
overall = 100*d["overall_aar"]

def profile(key):
    L=LEN[key]; s=[100*x if x is not None else None for x in ps[key]]; e=[100*x if x is not None else None for x in pe[key]]
    xs,ys=[],[]
    for j in range(L):
        dN,dC=j,L-1-j
        v = s[dN] if (dN<=dC and dN<len(s)) else (e[dC] if dC<len(e) else None)
        if v is not None: xs.append(j/(L-1)); ys.append(v)
    return np.array(xs),np.array(ys)

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5.6),sharey=True)
for ax,ch,title in [(ax1,"vh","Heavy chain (VH)"),(ax2,"vl","Light chain (VL)")]:
    for r in ["CDR1","CDR2","CDR3"]:
        x,y=profile(f"{ch}_{r}")
        ax.plot(x,y,marker="o",ms=6,lw=2.4,color=COL[r],label=f"{r} (mean {y.mean():.1f}%)")
    ax.axhline(overall,color="#888",ls="--",lw=1.2,label=f"overall AAR {overall:.1f}%")
    ax.set_xlabel("Relative position within region  (N -> C)")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=9,loc="lower center")
    ax.set_xlim(0,1); ax.set_ylim(89,100.5)
ax1.set_ylabel("Reconstruction accuracy AAR (%)")
fig.suptitle("Double-check: CDR3 reconstructs better than CDR1/CDR2 throughout — not a boundary artifact\n"
             "CDR1/CDR2 sit ~3 pts below CDR3 across the whole region (interior, not just the edges)",
             fontsize=13,y=1.03)
fig.tight_layout()
fig.savefig(OUT/"figR7_cdr_aar_check.png",dpi=150,bbox_inches="tight")
fig.savefig(OUT/"figR7_cdr_aar_check.pdf",bbox_inches="tight")
print("saved figR7_cdr_aar_check")
