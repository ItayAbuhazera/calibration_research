---
type: killed_idea
project: rgc
evidence_scope: unpublished_repository_analysis
reason: DAC's density-estimation mechanism collapses sharply on CIFAR-100 when given randomized-coordinate features instead of its native layer features
tags: [dac, randomized-coordinates, negative-result]
---

# Direct DAC calibration on randomized coordinates

## Original idea

Substitute DAC's prescribed, architecture-specific layer features with
RGCC-style randomized coordinate sampling, to get DAC's density-aware
calibration mechanism without DAC's layer-selection tuning burden.

## Why it died

On CIFAR-100, DAC's ECE degrades sharply and specifically when its feature
source is swapped to randomized coordinates (e.g.
`baseline_cross_entropy_cifar100_resnet18`: 0.033 → 0.218 ECE), with extreme
effect sizes (Cohen's d ranging −30 to −42). CIFAR-10 shows small,
non-significant differences in the same comparison table.

## Evidence that killed it

[[DAC can collapse on randomized coordinate features]]

## Scope of the kill

Killed only for: applying **DAC's own density-estimation mechanism** directly
to randomized-coordinate features, specifically evidenced as a failure mode
on CIFAR-100. Not a claim about RGCC itself (which pairs randomized
coordinates with its own rank-normalization + isotonic mapping, not DAC's
density estimator, and performs well per the published paper).

## Conditions under which to revisit

Only if DAC's density estimator is modified to be robust to the different
statistical structure of randomized-coordinate features (e.g. a
regularization or bandwidth-selection change specifically motivated by this
failure), and re-tested on CIFAR-100 with adequate seeds.

## Related project

[[Semantic Geometric Calibration RGC]]
