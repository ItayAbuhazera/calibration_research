---
type: observation
status: closed_historical
evidence_scope: unpublished_repository_analysis
project: rgc
evidence_strength: 3
tags: [k-neighbours, ood, id-calibration, ablation]
---

# Neighbour count controls OOD more than ID calibration

## Evidence source

`geometric/GeometricInternalCalibration/GeometricInternalCalibration/tables/k_ablation/table_k_trend_sgc.tex`,
backed by `tables/k_ablation/k_ablation_aggregated_results.csv` (SGC/RGCL
feature mode).

## Measured fact

As the nearest-neighbour count k varies 1→300:

- In-distribution ECE is flat/non-monotonic: 1.03% → 1.02% → 0.96% → 0.99% →
  1.04% → 1.06%, within roughly ±0.25–0.32 std — no clear trend.
- OOD detection AUROC rises **monotonically**: 0.700 → 0.737 → 0.774 → 0.788 →
  0.800 → 0.842 → 0.843 (+14.3 points from k=1 to k=300).

## Interpretation

Increasing the neighbour count used in the geometric score has a strong,
monotonic effect on OOD-detection quality but little to no effect on
in-distribution calibration quality — the two objectives respond very
differently to this hyperparameter.

## What this does NOT establish

- It does not establish a mechanism for *why* k affects OOD detection more
  than ID calibration.
- It does not establish that this generalizes beyond SGC/RGCL feature mode
  (RGCC was not confirmed to show the same pattern in this artifact).
- It does not by itself recommend a "best" k — that depends on whether OOD
  detection or ID calibration is the deployment priority.

## Related project

[[Semantic Geometric Calibration RGC]]

## Related later observations

- [[ID calibration gains do not imply OOD detection gains]] (the corresponding
  failure-mode framing of this same dissociation)
