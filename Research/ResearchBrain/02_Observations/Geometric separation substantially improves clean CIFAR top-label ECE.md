---
type: observation
status: closed_historical
evidence_scope: unpublished_repository_analysis
project: geometric-separation
evidence_strength: 2
tags: [geometry, ece, cifar, mobilenet, efficientnet]
---

# Geometric separation substantially improves clean CIFAR top-label ECE

## Evidence source

`geometric/GeometricCalibration/master_geometric_comparison.csv` and
`merged_complete_results/{cifar10,cifar100}_{mobilenet,efficientnet}/` in the
`GeometricCalibration` repository. Not covered by
[[Uncertainty Estimation Based on Geometric Separation]], which uses only
RF/GB/CNN, not MobileNet/EfficientNet.

## Measured fact

On clean (uncorrupted) CIFAR-10 and CIFAR-100, the best available geometric
calibration configuration substantially reduces top-label ECE vs. the
uncalibrated model for both extra models:

| Dataset | Model | Uncal ECE | Best geometric ECE | Reduction |
|---|---|---|---|---|
| CIFAR-10 | MobileNet | 0.0198 | 0.00971 | ~51% |
| CIFAR-10 | EfficientNet | 0.0188 | 0.01103 | ~41% |
| CIFAR-100 | MobileNet | 0.0580 | 0.01347 | ~77% |
| CIFAR-100 | EfficientNet | 0.0701 | 0.01368 | ~80% |

On GTSRB, the pattern does not hold: EfficientNet's best available geometric
configuration is *worse* than uncalibrated (0.0312 → 0.04052) even on clean
data; MobileNet's is only mildly better (0.0366 → 0.02891, ~21%).

## Interpretation

Geometric separation (in this repository's implementation) reduces clean
top-label ECE substantially on CIFAR for two additional CNN backbones beyond
those tested in the published paper — consistent with, but not proof of, the
paper's general claim generalizing to more architectures.

## What this does NOT establish

- This is a "best config per dataset/model" summary, not a paired/multi-seed
  comparison with confidence intervals in the modern sense — it lacks the
  paired-testing discipline used in the current (`GeometricFullCalibration`)
  work.
- It does not establish that this specific best configuration is stable across
  seeds: the underlying seed counts for some configurations are short of the
  nominal count (see
  [[Historical geometric calibration ECE exports contain invalid perfect scores]]
  for related seed-count integrity flags in the same result family).
- It does not generalize to GTSRB, where the same method with the same
  freedom to pick the best configuration is actually *worse* than uncalibrated
  for one of the two models.

## Related project

[[Geometric Separation]]

## Related later observations

- [[Semantic geometry is more corruption-resistant than pixel geometry but not shift-stable]]
- [[Geometric calibration space layer and binning are regime-dependent]]
