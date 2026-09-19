---
type: observation
status: closed_historical
evidence_scope: unpublished_repository_analysis
project: rgc
evidence_strength: 3
tags: [spp, rgcl, rgcc, latency, preprocessing]
---

# SPP changes the calibration-latency tradeoff rather than acting as free compression

## Evidence source

`geometric/GeometricInternalCalibration/GeometricInternalCalibration/timing_analysis/tableA_offline_online.csv`
and `tableB_offline_breakdown.csv` (duplicated under `Results/`, `Results1-3/`,
`calibration_comparison_separate_save_sample_time/`).

## Measured fact

RGCL's offline feature-extraction time (spatial pyramid pooling + random
projection) is 16×–44× that of RGCC (raw coordinate sampling, no
pooling/projection) across configurations:

| Config | RGCL offline | RGCC offline | Ratio |
|---|---|---|---|
| CIFAR-10 / ResNet-18 (n=22) | 461.9s | 16.9s | ~27x |
| CIFAR-10 / DenseNet-121 (n=6) | 708.2s | 43.8s | ~16x |
| CIFAR-100 / ResNet-50 (n=10) | 1162.3s | 33.2s | ~35x |
| SVHN / DenseNet-121 (n=4) | 3103.3s | 70.1s | ~44x |

Online per-sample latency is nearly identical between the two (e.g. 2.464ms
vs. 2.468ms for CIFAR-10/DenseNet-121) — the entire cost differential is in
one-time offline feature extraction, not inference.

## Interpretation

Spatial Pyramid Pooling is not a "free" dimensionality-reduction step relative
to raw coordinate sampling — it is a large, one-time offline cost, consistent
with but far more precisely quantified than the paper's qualitative statement
that RGCL costs more offline than RGCC (paper Fig. 4/5, Section 5.5).

## What this does NOT establish

- It does not establish that this offline cost matters in practice for a
  given deployment — the paper explicitly frames RGC as suited to settings
  with stable calibration data, where a one-time cost is amortized.
- It does not establish anything about calibration quality; this is a pure
  compute-cost measurement, separate from the ECE/equivalence findings.

## Related project

[[Semantic Geometric Calibration RGC]]

## Related later observations

[[RGCL-RGCC equivalence is metric- and regime-dependent]]
