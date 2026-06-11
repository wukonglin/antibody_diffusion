#!/usr/bin/env python3
"""
01_region_metrics.py — region-decomposed metrics for ALL 7 regions per chain,
across all 7 configs. Mirrors 01_cdr3_metrics.py exactly (same 4-gram diversity
and length-stats definitions) but for FR1/CDR1/FR2/CDR2/FR3/CDR3/FR4.

Output: results/region_metrics.json
"""
import json, os, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib_regions import extract_regions_pair, REGION_ORDER
from lib_cdr3 import parse_fasta

STAGE = Path("/sessions/practical-keen-lamport/mnt/outputs/region_work/staged")
OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

CONFIGS = [
    ("dplm_default", "DPLM default (T=1.0, p=0.95)"),
    ("dplm_tuned",   "DPLM tuned (T=1.3, p=0.99)"),
    ("ld4lg_w1p0",   "LD4LG w=1.0"),
    ("ld4lg_w1p5",   "LD4LG w=1.5"),
    ("ld4lg_w2p0",   "LD4LG w=2.0 (plain)"),
    ("ld4lg_w3p0",   "LD4LG w=3.0"),
    ("ld4lg_w5p0",   "LD4LG w=5.0"),
]
CELLS = [f"{iso}_{vf}_{loc}" for iso in ["IGHM","IGHG","IGHA"]
         for vf in ["IGHV1","IGHV3","IGHV4"] for loc in ["K","L"]]

def four_gram_diversity(seqs):
    if not seqs: return 0.0
    total=0; uniq=set()
    for s in seqs:
        for i in range(len(s)-3):
            uniq.add(s[i:i+4]); total+=1
    return len(uniq)/total if total>0 else 0.0

def length_stats(lengths):
    if not lengths: return {"n":0}
    a=np.array(lengths)
    return {"n":int(len(a)),"mean":float(a.mean()),"std":float(a.std()),
            "min":int(a.min()),"max":int(a.max()),
            "p25":float(np.percentile(a,25)),"p50":float(np.percentile(a,50)),
            "p75":float(np.percentile(a,75)),
            "histogram_bins":list(range(0,51)),
            "histogram_counts":np.histogram(a,bins=list(range(0,52)))[0].tolist()}

def analyze(name, display):
    d = STAGE/name
    if not d.exists():
        print(f"  skip {name} (missing)"); return None
    # per chain, per region collect sequences
    full=[]
    vh={r:[] for r in REGION_ORDER}; vl={r:[] for r in REGION_ORDER}
    n_total=0; vh_full_ok=0; vl_full_ok=0
    for cell in CELLS:
        fp=d/f"{cell}.fasta"
        if not fp.exists(): continue
        for _,seq in parse_fasta(str(fp)):
            n_total+=1; full.append(seq)
            r=extract_regions_pair(seq)
            if r.get("reason")=="no_linker": continue
            if r["vh_reason"]=="ok": vh_full_ok+=1
            if r["vl_reason"]=="ok": vl_full_ok+=1
            for reg in REGION_ORDER:
                if r["vh"][reg]: vh[reg].append(r["vh"][reg])
                if r["vl"][reg]: vl[reg].append(r["vl"][reg])
    res={"name":name,"display":display,"n_sequences_total":n_total,
         "div_4gram_full_seq":four_gram_diversity(full),
         "vh_full_extract_ok":vh_full_ok,"vl_full_extract_ok":vl_full_ok,
         "regions":{}}
    for reg in REGION_ORDER:
        res["regions"][reg]={
            "vh":{"div_4gram":four_gram_diversity(vh[reg]),
                  "n":len(vh[reg]),"length":length_stats([len(s) for s in vh[reg]])},
            "vl":{"div_4gram":four_gram_diversity(vl[reg]),
                  "n":len(vl[reg]),"length":length_stats([len(s) for s in vl[reg]])},
        }
    print(f"-> {display}: {n_total} seqs | VH-ok {100*vh_full_ok/max(n_total,1):.1f}% VL-ok {100*vl_full_ok/max(n_total,1):.1f}%")
    print(f"   full-seq 4gram div = {res['div_4gram_full_seq']:.4f}")
    print(f"   {'region':6}{'VH div':>9}{'VL div':>9}{'VH len':>9}{'VL len':>9}")
    for reg in REGION_ORDER:
        rv=res["regions"][reg]
        vhl=rv["vh"]["length"]; vll=rv["vl"]["length"]
        print(f"   {reg:6}{rv['vh']['div_4gram']:>9.4f}{rv['vl']['div_4gram']:>9.4f}"
              f"{vhl.get('mean',0):>9.1f}{vll.get('mean',0):>9.1f}")
    return res

def main():
    out=[]
    for name,disp in CONFIGS:
        r=analyze(name,disp)
        if r: out.append(r)
    p=OUTPUT_DIR/"region_metrics.json"
    json.dump(out, open(p,"w"), indent=2)
    print(f"\n saved {p}")

if __name__=="__main__":
    main()

