---
type: observation
status: replicated_exploratory
date: 2026-04-27
project: full-vector-calibration
evidence_strength: 3
tags: [mixture, negative-result, anchoring]
---

# Rank-geometric anchored mixture collapses to zero weight

## Observation

The validation-selected anchored rank-geometric mixture selected zero weight for
its geometric mixture component on all CIFAR-100 seeds.

The same `lambda=0, alpha=0` collapse appears again in the Tiny-ImageNet
multi-seed aggregate.

## Interpretation

Under the tested objective and implementation, the extra rank-geometric mixture
component is not earning its complexity.

## Important nuance

On CIFAR-10 the aggregate is less clean, so the strongest statement is about
CIFAR-100 plus Tiny-ImageNet, not universal impossibility.

## Consequence

Do not resurrect this method through more hyperparameter tuning on the same test
settings. Revisit only if a genuinely new regime or mechanism predicts a reason
the component should receive non-zero weight.
