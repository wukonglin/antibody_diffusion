#!/usr/bin/env python3
"""
aida_extract_train_subset.py — RUN ON AIDA
Randomly sample 50,000 paired antibody sequences from train.metadata.tsv, save
them as FASTA, then scp back to the Mac for the Hamming-distance distribution.

Why 50K:
  - the full training set (~1.74M) is too large; 100K also works but is ~25MB;
    50K is already stable enough.
  - 50K is enough to represent the training distribution relative to the 9216
    generated sequences.
  - scp back to the Mac is ~12 MB, a few seconds.
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
    # find the paired-sequence column (named pair_input_seq or similar)
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

