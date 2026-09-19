---
type: killed_idea
project: rgc
evidence_scope: unpublished_repository_analysis
non_canonical_source: true
reason: composite selector was the worst of four compared layer-choice strategies on the recovered AugMix/CIFAR-10 protocol
tags: [layer-selection, negative-result]
---

# Composite unsupervised layer selection for SGC

## Source caveat

Evidence for this kill is recoverable only from a non-canonical backup
directory (`_backup_unique_from_GeometricInternalCalibration_1_20260627/`) —
see [[Composite geometry scores do not reliably select calibration layers]]
for the full caveat. The canonical repository's `robust_layer_selector.py`
implementing this idea has no callers and no output artifacts.

## Original idea

Use an unsupervised composite geometric-separation score to automatically
select which intermediate layer to use for representation-space geometric
calibration, avoiding both manual architecture-specific tuning (DAC/TULIP) and
RGC's randomized-sampling approach.

## Why it died (scoped narrowly)

On the recovered protocol — 12 experiments, AugMix/CIFAR-10,
DenseNet-121/ResNet-18/ResNet-50 × seeds 11–14 — the composite selector had
the **worst mean ECE of four compared methods** (0.016435, vs. 0.002983 for
the oracle and 0.007208 for a naive fixed-default layer with no selection at
all), and beat the naive fixed default in only 3 of 12 experiments (25%).

## Evidence that killed it

[[Composite geometry scores do not reliably select calibration layers]]

## Scope of the kill

Killed only for: this specific composite unsupervised scoring function, on
this recovered AugMix/CIFAR-10 protocol, n=12, backup-sourced. Not a claim
that no unsupervised layer-selection score could ever work — nor a claim
about RGC's own randomized-sampling approach, which sidesteps this problem
entirely rather than trying to solve it.

## Conditions under which to revisit

Only with a re-implementation in the canonical codebase, evaluated with
adequate seeds and at least one additional benchmark family, and only if a
new composite score is motivated by a mechanism explaining why the earlier
one failed (e.g. why "looks separated" did not track "predicts calibration
error" here).

## Related project

[[Semantic Geometric Calibration RGC]]
