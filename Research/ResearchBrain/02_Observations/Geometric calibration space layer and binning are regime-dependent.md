---
type: observation
status: closed_historical
evidence_scope: unpublished_repository_analysis
project: geometric-separation
evidence_strength: 2
tags: [geometry, layer-selection, binning, regime-dependence]
---

# Geometric calibration space, layer, and binning are regime-dependent

## Evidence source

`geometric/GeometricCalibration/master_geometric_comparison.csv` (one row per
dataset/model recording the best-performing space/layer/binning
configuration).

## Measured fact

The best-performing configuration is not constant across datasets or even
across models on the same dataset:

- CIFAR-10 (both models): best space = physical (pixel), binned.
- CIFAR-100 (both models): best space = physical (pixel), **unbinned**.
- GTSRB (both models): best space = semantic, unbinned, but the best **layer**
  differs by architecture (`features_18_0` for MobileNet vs.
  `features_6_0_block_0` for EfficientNet).

Note on terminology: "binned"/"unbinned" here names the geometric
calibrator's internal fast-separation binning strategy (an implementation
choice inside the geometric score itself), not the number of ECE histogram
bins used for evaluation.

## Interpretation

No single (space, layer, binning) configuration dominates across datasets or
architectures in this repository's historical exploration; the right choice
appears to be regime-dependent.

## What this does NOT establish

- This is a "best per cell" table, not a sensitivity analysis — it does not
  show how close the alternatives were, nor whether the differences are
  statistically meaningful given the seed-count issues noted in
  [[Historical geometric calibration ECE exports contain invalid perfect scores]].
- It does not establish a rule for predicting the right configuration for a
  new dataset/architecture in advance.
- It does not by itself explain *why* GTSRB behaves so differently from
  CIFAR-10/100 (semantic space, no binning, and architecture-specific layer,
  vs. physical space for CIFAR).

## Related project

[[Geometric Separation]]

## Related later observations

- [[Geometric separation substantially improves clean CIFAR top-label ECE]]
- [[Semantic geometry is more corruption-resistant than pixel geometry but not shift-stable]]
- This regime-dependence anticipates the later, separately-evidenced
  [[RGCL-RGCC equivalence is metric- and regime-dependent]] in the RGC project
  — both show that a single fixed representation/config choice does not
  dominate universally, though the mechanisms and codebases differ.
