---
type: observation
status: structural
date: 2026-06-21
project: decision-improving-calibration
evidence_strength: 5
tags: [temperature-scaling, argmax, theorem]
---

# Sample-dependent scalar temperature cannot change the predicted class

## Statement

For any strictly positive scalar temperature `T(z)`,

argmax_k z_k = argmax_k z_k / T(z).

This remains true even when the temperature depends arbitrarily on the sample's
logit vector.

## Consequence

Global TS and per-sample scalar PTS may improve probability calibration but
cannot improve accuracy through changing the selected class.

## Research use

This creates a structural control group for decision-improving calibration:
methods that genuinely alter decisions need class-dependent corrections or some
other mechanism that changes relative class scores.
