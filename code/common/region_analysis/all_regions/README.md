# By-region analysis (all 7 regions) — extension of the CDR3 analysis

Same pipeline as the original CDR3 analysis, extended from CDR3 to **all 7 antibody
regions** per chain: **FR1, CDR1, FR2, CDR2, FR3, CDR3, FR4**, for both VH (heavy)
and VL (light), across all **7 configs** (DPLM default/tuned + LD4LG w=1/1.5/2/3/5).

## Method (identical philosophy to `lib_cdr3.py`)

No IMGT alignment — regions are sliced between **conserved framework motifs**, exactly
the way the original code anchored CDR3 on the pre-CDR3 Cys and the FR4 `WGxG`/`FGxG`
motif. The CDR3 boundary code is reused unchanged, so **CDR3 numbers reproduce the
original `cdr3_metrics.json` exactly** (verified: DPLM-default VH CDR3 = 0.26240,
VL CDR3 = 0.04329, identical to the original).

Added anchors: 1st conserved Cys (FR1 end), FR2 Trp (`WVRQ` heavy / `WYQQ` light),
the `LEW.G` Trp (FR2 end, heavy), and the FR3-start motif (`RVTM` heavy / `RFSGS`
light). See `lib_regions.py`.

Extraction success on the generated sequences: heavy 99%+, light 96–99% for the
low-temperature configs. For **DPLM-tuned** (T=1.3) the high sampling temperature
mutates some conserved anchor residues, so full 7-region extraction drops to ~50–57%;
CDR3 itself is unaffected. The non-CDR3 diversity for DPLM-tuned is therefore a mild
**under**-estimate (the most-mutated sequences are the ones that fail extraction).

## Key result

4-gram diversity is concentrated in the CDRs and near-zero in the frameworks, in
**every** model — the per-region view the full-sequence metric hides:

`CDR3 ≫ CDR2 ≈ CDR1 ≫ FR1–FR4`

Framework regions are also **length-invariant** (σ≈0: FR1=22, FR2=14, FR3=30, FR4=11 aa
heavy), while the CDRs carry all the length variation — widest in CDR3, the junctional
region. The LD4LG CFG plateau seen for CDR3 holds for **CDR1 and CDR2** too.

## Files

- `figR1_region_diversity_profile.{png,pdf}` — 4-gram diversity across all 7 regions,
  key configs (extends original fig2)
- `figR2_region_cfg_plateau.{png,pdf}` — LD4LG CFG sweep per CDR, CDR1/CDR2/CDR3
  (extends original fig1)
- `figR3_region_length_distribution.{png,pdf}` — per-region length distributions
  (extends original fig3)
- `region_metrics.json` — full numerical results (every region × chain × config)
- `region_metrics_summary.csv` — flat table (config, chain, region, div, len, n)
- `lib_regions.py` — region extraction (extends `lib_cdr3.py`)
- `01_region_metrics.py`, `02_plot_region_metrics.py` — compute + plot scripts
