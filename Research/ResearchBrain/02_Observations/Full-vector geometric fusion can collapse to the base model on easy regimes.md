---
type: observation
status: open
date: 2026-04-27
project: full-vector-calibration
evidence_strength: 2
tags: [dataset-regime, fusion, decision-change]
---

# Full-vector geometric fusion can collapse to the base model on easy regimes

## Observation

On CIFAR-10 / ResNet-18, the selected full-vector fusion beta was zero and the
fusion metrics matched the base model.

On Tiny-ImageNet / ResNet-50, fusion selected beta=10, changed the argmax on
~5.3% of samples, produced positive net flips, and improved accuracy and NLL.

## Counter-evidence to the "hard regime" reading

CIFAR-100-C is a hard regime by any reasonable definition, yet the fixed beta=30
vector arm is roughly level with the head there (**[inference]** - that figure is
derived by subtracting two paired CIs in
[[2026-09-15 RGC Shift Recoverability]], not read off a reported
number). So "harder regime -> geometry earns decision changes" does not survive
the third data point.

Confounds separating the three points: ResNet-18 vs ResNet-50 vs ResNet-101,
validation-selected beta vs beta fixed at 30, and covariate shift vs clean
in-distribution difficulty. Any of the three could carry the effect.

## Candidate latent pattern

The operational value of geometric decision changes may be regime-dependent -
but the vault currently has three points, three architectures, and two beta
selection rules, so "regime" is not yet isolated from anything else.

## What this does NOT establish

Calling one benchmark "easy" and another "hard" is only shorthand. The runs also
differ in:
- number of classes,
- architecture,
- checkpoints,
- feature distributions,
- error prevalence.

## Decisive future test

Hold model family and training recipe fixed and vary a controlled difficulty /
shift dimension. Ask whether oracle geometric complementarity and selected fusion
strength increase with that dimension.
