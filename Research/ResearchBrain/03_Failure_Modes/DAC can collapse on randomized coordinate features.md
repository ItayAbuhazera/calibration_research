---
type: failure_mode
status: closed_historical
evidence_scope: unpublished_repository_analysis
project: rgc
tags: [dac, randomized-coordinates, ablation, cifar100]
---

# DAC can collapse on randomized coordinate features

## Evidence

`geometric/GeometricInternalCalibration/GeometricInternalCalibration/comprehensive_analysis.comparisons.json`,
key `"DAC: Original vs DAC with Random Coordinates"`.

## Measured fact

Applying DAC's own density-aware calibration mechanism to RGCC-style
randomized-coordinate features (instead of DAC's native prescribed layer
features) produces a sharp, dataset-specific degradation, entirely on
CIFAR-100:

| Config | DAC (native) ECE | DAC (random coords) ECE | Cohen's d |
|---|---|---|---|
| augmix_cifar100_resnet18 | 0.073 | 0.324 | -42.4 |
| baseline_brier_cifar100_resnet18 | 0.047 | 0.232 | -30.9 |
| baseline_cross_entropy_cifar100_resnet101 | 0.032 | 0.239 | -32.1 |
| baseline_cross_entropy_cifar100_resnet18 | 0.033 | 0.218 | — |
| baseline_focal_adaptive_cifar100_resnet18 | 0.029 | 0.202 | — |

Effect sizes of −30 to −42 are extreme, consistent with a systematic collapse
rather than run-to-run noise. CIFAR-10 rows in the same comparison table show
small/non-significant differences.

## Interpretation

DAC's density-estimation mechanism appears to depend on structural properties
of its native, deliberately-chosen layer features that randomized coordinate
sampling does not preserve — at least on CIFAR-100. This is a failure of
*substituting DAC's feature source*, not a failure of RGCC's own calibration
mapping (RGCC uses isotonic regression on rank-normalized separation scores,
not DAC's density estimator).

## What this does NOT establish

- It does not establish why CIFAR-100 specifically triggers this collapse
  while CIFAR-10 does not.
- It does not establish anything about RGCC's own reported performance
  (Table 1/2 of the published paper) — RGCC does not use DAC's density
  estimator.
- It is a single comparison table; no seed-level breakdown was located beyond
  the mean/effect-size summary.

## Related project

[[Semantic Geometric Calibration RGC]]

## Related

[[Direct DAC calibration on randomized coordinates]] (the resulting kill)
