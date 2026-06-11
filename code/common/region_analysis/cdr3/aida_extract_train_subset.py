#!/usr/bin/env python3
"""
aida_extract_train_subset.py — RUN ON AIDA
从 train.metadata.tsv 随机抽 50,000 条 paired antibody sequence
存成 FASTA,然后 scp 回 Mac 用于 Hamming distance distribution.

为啥 50K:
  - Full 训练集 ~1.74M 太大;100K 也 OK 但 ~25MB;50K 已经够 stable
  - 跟生成集 9216 比足够 represent training distribution
  - scp 回 Mac 约 12 MB,几秒钟
"""

import csv
import random

TSV_PATH = "/mnt/beegfs/bulk/mirror/yl4259/bio-diffusion/train.metadata.tsv"
OUT_PATH = "/mnt/beegfs/bulk/mirror/yl4259/ab_ld4lg/train_subset_50k.fasta"
N_SAMPLES = 50_000
SEED = 42

print(f"Reading {TSV_PATH}...")
all_seqs = []
with open(TSV_PATH, newline="") as f:
    reader = csv.DictReader(f, delimiter="\t")
    # 找 pair_input_seq 列 (column name 应该叫这个或类似的)
    fieldnames = reader.fieldnames
    seq_col = None
    for candidate in ["pair_input_seq", "paired_seq", "sequence", "seq"]:
        if candidate in fieldnames:
            seq_col = candidate
            break
    if seq_col is None:
        print(f"ERROR: can't find sequence column in {fieldnames}")
        raise SystemExit(1)
    print(f"Using column: {seq_col}")
    for i, row in enumerate(reader):
        seq = row[seq_col].strip()
        if seq:
            all_seqs.append(seq)
        if i % 200_000 == 0 and i > 0:
            print(f"  read {i:,} rows...")

print(f"Total training sequences: {len(all_seqs):,}")

random.seed(SEED)
subset = random.sample(all_seqs, min(N_SAMPLES, len(all_seqs)))
print(f"Sampled {len(subset):,} for subset")

with open(OUT_PATH, "w") as f:
    for i, seq in enumerate(subset):
        f.write(f">train_{i}\n{seq}\n")

print(f"Wrote {OUT_PATH}")
print(f"\nNow on Mac:")
print(f"  scp yl4259@aida2.cac.cornell.edu:{OUT_PATH} \\")
print(f"      '/Users/susi/Documents/Claude/Projects/DL Final/ab_ld4lg_results/'")

