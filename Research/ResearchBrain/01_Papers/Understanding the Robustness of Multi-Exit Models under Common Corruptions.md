---
type: paper
status: abstract_verified
year: 2022
venue: "arXiv 2212.01562; peer-review venue not found"
url: https://arxiv.org/abs/2212.01562
tags: [early-exit, overthinking, cifar-c, oracle-exit, prior-art-audit]
---

# Understanding the Robustness of Multi-Exit Models under Common Corruptions

Mehra, Seto, Jaitly, Theobald. Abstract verified by the lead 2026-09-29; table values reported by subagents only.

## Why this matters to me
Establishes "correct earlier, wrong later" **under common corruptions** and quantifies how little of it realistic policies recover.

## Core contribution
Multi-exit models (SDN-style) under CIFAR-C: "early-exiting at the first correct exit … significant boost in accuracy (~10%) over
exiting at the last layer. However, with realistic early-exit strategies … a marginal improvement in accuracy (1%)"; shift "widens the gap
… by 5% on average"; defines underthinking and overthinking metrics.

## Benchmark / task
CIFAR-10-C / CIFAR-100-C; VGG-16, ResNet-56 multi-exit models (subagent reading).

## Strongest result
Oracle-vs-realistic gap ≈ 10 vs ≈ 1 points under corruption.

## Failure / limitation
Oracle "first correct exit" counts include multiplicity/chance agreement; exits are trained heads (whether IC-only on a frozen backbone:
unverified); no per-example learned corrector with harm accounting.

## What this changes in my beliefs
The oracle headroom is known to be large and mostly unrecoverable by simple policies under shift — T1's phenomenon is taken; any
contribution must be about prospective selection, not headroom.

## Related observations
[[Oracle any-layer recoverability counts overstate correction headroom]].

## Novelty relevance
SAME PHENOMENON under corruption (T1/T2 oracle).
