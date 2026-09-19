---
type: experiment
status: completed_exploratory
date: 2026-04-27
project: full-vector-calibration
benchmark: Tiny-ImageNet
model: ResNet-50
seeds: [11, 12, 13, 14, 15]
preregistered: false
source: github
tags: [calibration, full-vector, decision-change, exploratory]
---

# 2026-04-27 Tiny-ImageNet / ResNet-50 full-vector path batch

## Provenance

The card was filled retroactively after the batch had already completed.
Treat this as exploratory evidence.

GitHub source:
`research_log/experiments/2026-04-27_tiny-imagenet-resnet50_path-decision-batch/`

## Key readout

Approximate aggregate means:

- Base accuracy: **0.5750**
- Base NLL: **1.7199**
- Full-vector distance fusion accuracy: **0.5819**
- Full-vector distance fusion NLL: **1.6921**
- Fusion argmax-change rate: **~5.30%**
- Fusion net flips: **~68.8**
- Selected fusion beta: **10**

Thus, unlike CIFAR-10, the geometric full-vector method actually changed
decisions and improved both accuracy and NLL in this setting.

At the same time, the anchored rank-geometric mixture again selected
`lambda=0, alpha=0`.

## Observation extracted

[[Full-vector geometric fusion can collapse to the base model on easy regimes]]
and its converse: on a harder 200-class setting, geometry changed decisions and
produced positive net flips.

## Missing

No confidence interval is recorded for the accuracy or NLL gain, although the
batch ran five seeds. This is currently the only cross-dataset positive result
for geometric decision changes, so it should not be cited as a gain until a
paired seed-level CI is computed from the existing run outputs.

## Caution

This is not a controlled "difficulty" experiment: dataset, model architecture,
class count, and training regime all change together.

## Related notes

- [[Top-label calibration and full-vector proper scoring form a Pareto frontier]]
- [[Rank-geometric anchored mixture collapses to zero weight]]
