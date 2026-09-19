---
type: experiment
status: completed_exploratory
date: 2026-04-25
project: full-vector-calibration
benchmark: CIFAR-10
model: ResNet-18
seeds: [21, 22, 23, 24, 25]
preregistered: false
source: github
tags: [calibration, full-vector, decision-change, exploratory]
---

# 2026-04-25 CIFAR-10 / ResNet-18 full-vector path batch

## Provenance

This batch was already completed before its experiment card was filled.
Treat its conclusions as **exploratory**, not pre-registered.

GitHub source:
`research_log/experiments/2026-04-25_cifar10-resnet18_path-decision-batch/`

## Key readout

- Base accuracy: ~0.94028
- Base NLL: ~0.25325
- Base top-label ECE: ~0.03966
- `full_vector_distance_fusion` selected beta = **0** across the aggregate and
  therefore reduced to the base probabilities on this easy setting.
- One-vs-rest beta calibration improved NLL and top-label ECE while allowing a
  small amount of decision change.

## Observation extracted

[[Full-vector geometric fusion can collapse to the base model on easy regimes]]

## What this suggests

Geometry-aware decision changes are not uniformly useful. Their value may depend
on dataset difficulty, number of classes, base-model error structure, or the
amount of complementary geometric information available.

## What this does NOT prove

It does not establish a monotonic relationship with "difficulty". CIFAR-10,
CIFAR-100, and Tiny-ImageNet differ along many axes.

## Related notes

- [[Top-label calibration and full-vector proper scoring form a Pareto frontier]]
- [[2026-04-25 CIFAR100 ResNet18 Full-Vector Path Batch]]
- [[2026-04-27 TinyImageNet ResNet50 Full-Vector Path Batch]]
