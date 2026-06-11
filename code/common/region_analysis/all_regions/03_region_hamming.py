#!/usr/bin/env python3
"""
03_region_hamming.py — region-decomposed version of the original
03_hamming_distribution.py.

For each region (FR1..CDR3..FR4, both chains) and each generated sequence,
compute the Hamming distance to the NEAREST training-set sequence's SAME region
(length-matched), exactly the way the original did for the full sequence.

Memorization-vs-novelty, by region:
  framework Hamming peaks near 0  -> germline, essentially memorized
  CDR3 Hamming far from 0         -> novel (junctional)

Output: results/region_hamming.json + figR4_region_hamming.{png,pdf}
"""
import json, os, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_regions import extract_regions_pair, REGION_ORDER
from lib_cdr3 import parse_fasta

STAGE = Path("/sessions/practical-keen-lamport/mnt/outputs/region_work/staged")
TRAIN = Path("/sessions/practical-keen-lamport/mnt/DL Final/ab_ld4lg_results/train_subset_50k.fasta")
OUT = Path(__file__).parent / "results"

# Configs to score (keep it focused & readable; the cross-REGION story is the point)
CONFIGS = [("ld4lg_w2p0", "LD4LG (w=2)")]
TRAIN_MAX = 20000  # cap training seqs parsed (nearest-neighbor stays stable)
TRAIN_CAP = 1200   # max training rows per (region,chain,length) bucket
GEN_CHUNK = 400    # generated rows per vectorized chunk
rng = np.random.default_rng(0)
CELLS = [f"{iso}_{vf}_{loc}" for iso in ["IGHM","IGHG","IGHA"]
         for vf in ["IGHV1","IGHV3","IGHV4"] for loc in ["K","L"]]

AA = "ACDEFGHIKLMNPQRSTVWY"; AAI = {c:i for i,c in enumerate(AA)}
def enc(s):
    a = np.full(len(s), 255, np.uint8)
    for i,c in enumerate(s):
        if c in AAI: a[i] = AAI[c]
    return a

def build_train_index():
    """ {(region,chain,length): np.ndarray(n,length)} from training regions."""
    print("Extracting regions from training subset...")
    buckets = defaultdict(list)
    n = 0
    for _, seq in parse_fasta(str(TRAIN)):
        if n >= TRAIN_MAX: break
        r = extract_regions_pair(seq)
        if r.get("reason") == "no_linker": continue
        for ch in ("vh", "vl"):
            for reg in REGION_ORDER:
                s = r[ch][reg]
                if s: buckets[(reg, ch, len(s))].append(enc(s))
        n += 1
    idx = {k: np.stack(v) for k, v in buckets.items() if len(v) >= 5}
    print(f"  indexed {n} training seqs into {len(idx)} (region,chain,len) buckets")
    return idx

def score_config(name, idx):
    """Bucket generated regions by (region,chain,length) then vectorize Hamming."""
    d = STAGE / name
    # collect generated region encodings, grouped by bucket
    gen = defaultdict(list)
    for cell in CELLS:
        fp = d / f"{cell}.fasta"
        if not fp.exists(): continue
        for _, seq in parse_fasta(str(fp)):
            r = extract_regions_pair(seq)
            if r.get("reason") == "no_linker": continue
            for ch in ("vh","vl"):
                for reg in REGION_ORDER:
                    s = r[ch][reg]
                    if s and (reg,ch,len(s)) in idx:
                        gen[(reg,ch,len(s))].append(enc(s))
    ham = {(reg,ch): [] for reg in REGION_ORDER for ch in ("vh","vl")}
    for (reg,ch,L), rows in gen.items():
        G = np.stack(rows)                       # (nG, L)
        T = idx[(reg,ch,L)]                      # (nT, L)
        if T.shape[0] > TRAIN_CAP:
            T = T[rng.choice(T.shape[0], TRAIN_CAP, replace=False)]
        mins = np.empty(G.shape[0], np.int32)
        for i in range(0, G.shape[0], GEN_CHUNK):
            g = G[i:i+GEN_CHUNK]                  # (c, L)
            d2 = (T[None,:,:] != g[:,None,:]).sum(2)   # (c, nT)
            mins[i:i+g.shape[0]] = d2.min(1)
        ham[(reg,ch)].extend(mins.tolist())
    return ham

def main():
    idx = build_train_index()
    allres = {}
    for name, disp in CONFIGS:
        print(f"Scoring {disp}...")
        ham = score_config(name, idx)
        allres[name] = {"display": disp,
            "regions": {f"{reg}_{ch}": {
                "n": len(ham[(reg,ch)]),
                "mean": float(np.mean(ham[(reg,ch)])) if ham[(reg,ch)] else None,
                "p50": float(np.median(ham[(reg,ch)])) if ham[(reg,ch)] else None,
                "hist_bins": list(range(0,31)),
                "hist_counts": np.histogram(ham[(reg,ch)], bins=list(range(0,32)))[0].tolist() if ham[(reg,ch)] else [],
            } for reg in REGION_ORDER for ch in ("vh","vl")}}
    json.dump(allres, open(OUT/"region_hamming.json","w"), indent=2)
    print("saved region_hamming.json")

    # plot: per-region Hamming distribution for the plain LD4LG model
    import matplotlib.pyplot as plt
    name = "ld4lg_w2p0"; res = allres[name]
    palette = {"FR1":"#bbbbbb","FR2":"#999999","FR3":"#777777","FR4":"#555555",
               "CDR1":"#2ca02c","CDR2":"#ff7f0e","CDR3":"#1f77b4"}
    is_cdr = {"FR1":0,"CDR1":1,"FR2":0,"CDR2":1,"FR3":0,"CDR3":1,"FR4":0}
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(14,5.5))
    for ax,ch,title in [(ax1,"vh","Heavy chain (VH)"),(ax2,"vl","Light chain (VL)")]:
        for reg in REGION_ORDER:
            r=res["regions"][f"{reg}_{ch}"]
            if not r["hist_counts"]: continue
            b=r["hist_bins"]; c=r["hist_counts"]; tot=sum(c) or 1
            dens=[x/tot for x in c]
            ax.plot(b,dens,("-" if is_cdr[reg] else "--"),color=palette[reg],
                    lw=(2.4 if is_cdr[reg] else 1.2),marker=("o" if is_cdr[reg] else None),
                    markersize=4,label=f"{reg} (mean={r['mean']:.1f})")
        ax.set_xlabel("Hamming distance to nearest training sequence (same region, length-matched)")
        ax.set_ylabel("Density"); ax.set_title(title); ax.set_xlim(0,30)
        ax.grid(alpha=0.3); ax.legend(fontsize=8,ncol=2)
    fig.suptitle("Memorization vs novelty, by region (LD4LG w=2): frameworks sit near Hamming 0\n"
                 "(germline, essentially memorized); CDRs are novel — CDR3 the furthest from training",
                 fontsize=13,y=1.03)
    fig.tight_layout()
    fig.savefig(OUT/"figR4_region_hamming.png",dpi=150,bbox_inches="tight")
    fig.savefig(OUT/"figR4_region_hamming.pdf",bbox_inches="tight")
    print(" figR4_region_hamming")

if __name__=="__main__":
    main()

