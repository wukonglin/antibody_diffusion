import glob, os, re, shutil
from lib_cdr3 import parse_fasta

UP = "/sessions/practical-keen-lamport/mnt/uploads"
STAGE = "/sessions/practical-keen-lamport/mnt/outputs/region_work/staged"

def config_of(h):
    T = re.search(r"T=([0-9.]+)", h)
    topp = re.search(r"topp=([0-9.]+)", h)
    cfg = re.search(r"cfg=([0-9.]+)", h)
    if "model=DPLM" in h:                      # DPLM: distinguished by T
        return "dplm_default" if (T and T.group(1)=="1.0") else ("dplm_tuned" if (T and T.group(1)=="1.3") else f"dplm_T{T.group(1) if T else '?'}")
    # LD4LG: no model field, only cfg weight
    w = cfg.group(1) if cfg else "?"
    m = {"1.0":"ld4lg_w1p0","1.5":"ld4lg_w1p5","2.0":"ld4lg_w2p0","3.0":"ld4lg_w3p0","5.0":"ld4lg_w5p0"}
    return m.get(w, f"ld4lg_w{w}")

def cell_of(h):
    iso=re.search(r"iso=(\w+)",h).group(1); vf=re.search(r"vfam=(\w+)",h).group(1); loc=re.search(r"loc=(\w+)",h).group(1)
    return f"{iso}_{vf}_{loc}"

buckets={}
for fp in glob.glob(os.path.join(UP,"*.fasta")):
    recs=list(parse_fasta(fp))
    if not recs: continue
    cfg=config_of(recs[0][0]); cell=cell_of(recs[0][0])
    buckets.setdefault(cfg,{})[cell]=fp

os.makedirs(STAGE, exist_ok=True)
for cfg,cells in buckets.items():
    d=os.path.join(STAGE,cfg); os.makedirs(d,exist_ok=True)
    for cell,fp in cells.items():
        dst=os.path.join(d,f"{cell}.fasta")
        with open(fp) as r, open(dst,"w") as w: w.write(r.read())

print("Configs staged:")
for cfg in sorted(buckets):
    nseq=sum(1 for _ in parse_fasta(list(buckets[cfg].values())[0]))
    print(f"  {cfg:14} {len(buckets[cfg])} cells  (~{nseq}/cell)")

