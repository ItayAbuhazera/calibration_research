---
type: observation
status: closed_historical
evidence_scope: unpublished_repository_analysis
project: rgc
evidence_strength: 2
tags: [rgcl, rgcc, tost, equivalence, margin]
---

# RGCL-RGCC equivalence is metric- and regime-dependent

## Evidence source

`geometric/GeometricInternalCalibration/GeometricInternalCalibration/Experiments/generate_rgcc_rgcl_tost_table.py`
(canonical script, default `--absolute-margin 0.005`, i.e. δ=0.5pp — the same
margin as the published paper), run into at least three sibling output
directories with different row compositions:
`calibration_comparison_mce_adaptive/tost_rgcc_rgcl/tost_rgcc_vs_rgcl_ece.csv`,
`calibration_comparison_separate/tost_rgcc_rgcl/tost_rgcc_vs_rgcl_ece.csv`,
`calibration_comparison_new/tost_rgcc_rgcl/tost_rgcc_vs_rgcl_ece.csv`; plus
`latex_tables_statistics_n/table_sgc_vs_faiss_tost_{10,20,30}pct.tex` (relative
margin, different pairing).

## Measured fact

The paper (Table 2) reports RGCL-vs-RGCC equivalence in 9 of 13 model–dataset
combinations at δ=0.5pp. The repository's own artifacts do not reproduce one
canonical "9 of 13":

- `calibration_comparison_mce_adaptive`: 13 rows (CIFAR-100 set swaps in
  dinov2_large and drops resnet152) → 9 Equivalent / 1 Inconclusive / 3
  RGCC-lower. **CIFAR-100/densenet121 is Equivalent here** — contradicting the
  paper's own stated non-equivalent list for that same combination.
- `calibration_comparison_separate`: 14 rows (both dinov2_large and
  CIFAR-100/resnet152 present, different per-row `n`) → 9 Equivalent / 4
  Inconclusive / 1 RGCC-lower. Same headline "9," different denominator (14)
  and different failing set.
- `calibration_comparison_new`: only 10 rows, no Tiny-ImageNet at all → 7
  Equivalent / 3 Inconclusive.
- A separate table (`table_sgc_vs_faiss_tost_*pct.tex`) tests a **different
  pairing** (SGC vs. an SGC-FAISS backend, not RGCL-vs-RGCC) using *relative*
  margins (10/20/30% of mean) instead of the paper's fixed δ: "Equivalence
  established: 0/3" at every relative margin tested. This must not be
  conflated with the RGCL-vs-RGCC question above.

## Interpretation

The margin itself (δ=0.5pp, absolute) is largely consistent across the
RGCL-vs-RGCC tables; what varies is the **row composition and sample size**
feeding the same test, which changes both the denominator and which specific
combinations are called equivalent. Equivalence conclusions in this line of
work depend on the metric (absolute vs. relative margin), the exact
dataset/model set included, and the sample size per cell — not on a single
stable ground truth.

## What this does NOT establish

- It does not establish that the published paper's Table 2 is wrong — the
  paper's own reported numbers are the highest-precedence source; this note
  records an **unresolved conflict** between repository artifacts and the
  paper's stated exception list for CIFAR-100/densenet121, not a correction to
  the paper.
- It does not establish "RGCL and RGCC are universally equivalent" or "RGCC is
  universally better" — neither claim is supported by any single artifact
  found.
- It does not identify which of the sibling directories, if any, represents
  the "final" analysis; no run manifest resolving this was found.

## Related project

[[Semantic Geometric Calibration RGC]]

## Related later observations

- [[Geometric calibration space layer and binning are regime-dependent]] (an
  earlier, differently-caused instance of "no single configuration dominates")
