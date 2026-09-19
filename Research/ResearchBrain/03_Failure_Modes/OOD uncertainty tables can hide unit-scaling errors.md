---
type: failure_mode
status: closed_historical
evidence_scope: unpublished_repository_analysis
project: rgc
tags: [ood, unit-scaling, sentinel-value, tost]
---

# OOD uncertainty tables can hide unit-scaling errors

## Scope correction — read before citing

The evidence found is **not** from an OOD-metric table, and is **not**
confirmed to be a literal unit-scaling (e.g. percentage-vs-fraction) bug. It
is a plausible sentinel/placeholder-value artifact found in an *equivalence*
table for a different comparison (SGC vs. an SGC-FAISS backend). Kept under
this title per the originating task's requested structure, but scoped
precisely below — do not read this as a confirmed OOD unit-scaling bug.

## Evidence

`geometric/GeometricInternalCalibration/GeometricInternalCalibration/latex_tables_statistics_n/table_sgc_vs_faiss_tost_{10,20,30}pct.tex`
(duplicated under `calibration_comparison_mce_adaptive/tost_ece_tables/`).
Rows for CIFAR-10/DenseNet-121 and CIFAR-10/ResNet-152 render as
`0.59±0.17` (SGC) vs. `0.42±100.00` (SGC-FAISS) — an implausible standard
deviation of exactly `100.00`, alongside a CI/Diff field rendered as `"---"`
and TOST p = 1.000.

## Interpretation

This pattern is consistent with an `n=1` (undefined-variance) case being
rendered through a sentinel/fallback value instead of being excluded or
flagged, in the LaTeX table-generation code. The exact injection point was
not conclusively traced (candidate region:
`Experiments/generate_sgc_tables.py`, lines ~1600–1660, functions
`format_ece_value`/`format_accuracy_value`). A separate check of
`verify_ood_reporting.py`'s explicit unit test for `*_improvement_vs_*_pct`
OOD formulas found no percent/fraction inconsistency there — the OOD
percentage-formula code itself appears internally consistent.

## What this does NOT establish

- It does not establish a confirmed unit-scaling bug in any OOD-labeled table.
- It does not establish the exact line of code responsible for the sentinel
  value.
- It does not establish how many other rows across the repository's many
  `latex_tables_statistics*` variants share this pattern — only two rows were
  directly confirmed.

## Related project

[[Semantic Geometric Calibration RGC]]

## Related

[[RGCL-RGCC equivalence is metric- and regime-dependent]]
