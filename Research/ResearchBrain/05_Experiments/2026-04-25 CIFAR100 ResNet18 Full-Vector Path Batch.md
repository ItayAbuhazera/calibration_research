---
type: experiment
status: completed_exploratory
date: 2026-04-25
project: full-vector-calibration
benchmark: CIFAR-100
model: ResNet-18
seeds: [21, 22, 23, 24, 25]
preregistered: false
source: github
tags: [calibration, full-vector, pareto-frontier, exploratory]
---

# 2026-04-25 CIFAR-100 / ResNet-18 full-vector path batch

## Provenance

This batch was completed before its card was filled. The repository explicitly
marks it as exploratory.

GitHub source:
`research_log/experiments/2026-04-25_cifar100-resnet18_path-decision-batch/`

## Key structural result

The batch exposed a strong trade-off between two calibration goals:

- `rgcl_tail_vector_scaling` (earlier informally called `gc_dac`) had much better
  **top-label ECE** than full-vector distance fusion.
- `full_vector_distance_fusion` had much better **NLL / full-distribution quality**
  and better accuracy.

This led to the accepted decision that one baseline is insufficient for hybrid
methods: compare against both a full-vector baseline and a top-label specialist.

Method naming resolved 2026-09-15 by author decision: the top-label specialist is called `rgcl_tail_vector_scaling` everywhere; `gc_dac` was an earlier informal name for the same method. This is a naming choice, not a re-check of the run outputs.

## Important negative result

The proposed anchored rank-geometric mixture selected:
- lambda = 0 on every CIFAR-100 seed;
- alpha = 0 on every CIFAR-100 seed.

So the mixture collapsed to the anchored-model-tail endpoint rather than using
the rank-geometric component.

## Another failure

`rgcl_tail_dirichlet` worsened NLL on the base-correct subset, and the effect was
not confined to confidently wrong predictions.

## Decision that followed

Path A was adopted: characterize the **Pareto frontier** between top-label
calibration and full-vector proper scoring rather than claiming a universally
dominating hybrid.

## Related notes

- [[Top-label calibration and full-vector proper scoring form a Pareto frontier]]
- [[Rank-geometric anchored mixture collapses to zero weight]]
- [[High-confidence anchoring can block useful full-vector decision changes]]
- [[GC-DAC confidence-gated anchoring on CIFAR-100]]
