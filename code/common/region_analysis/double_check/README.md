# double_check — region-AAR verification

Independent checks that the surprising per-region finding — in the heavy chain,
**CDR3 reconstructs better than CDR1 and CDR2** — is real and not an artifact of
how we define region boundaries.

## Scripts

| File | Check | Runs where |
|---|---|---|
| `dump_reconstruction_cases.py` | #3 — dump teacher-forced reconstruction cases (orig vs recon, per-position region tags) so errors can be eyeballed | AIDA (needs AE checkpoint) |
| `anarci_boundary_check.py` | #4 (cheap) — compare our motif CDR boundaries against IMGT numbering on raw sequences; no model needed | anywhere with anarcii |
| `eval_aar_imgt.py` | #4 (full) — recompute per-region AAR using standard IMGT boundaries from ANARCII instead of our motifs | AIDA (needs AE checkpoint) |
| `07_plot_motif_vs_imgt.py` | plots motif vs IMGT per-region AAR side by side (figR8) | local |

`eval_aar_imgt.py` numbers chains with `anarcii` directly. It does NOT use
`abnumber`: abnumber hard-imports the classic `anarci` module, which is absent in
the AIDA venv (pip pulls in `anarcii`, a different package), so every call fails.

## Result

Both region definitions agree (numbers within ~1–2 points everywhere):

| VH | CDR1 | CDR2 | CDR3 |
|---|---|---|---|
| Motif anchoring | 93.8 | 94.3 | 96.9 |
| IMGT (ANARCII)  | 94.3 | 92.3 | 97.0 |

Under standard IMGT boundaries the ordering CDR3 > CDR1 > CDR2 holds (and is
stronger). See `results/figR8_motif_vs_imgt.png`.
