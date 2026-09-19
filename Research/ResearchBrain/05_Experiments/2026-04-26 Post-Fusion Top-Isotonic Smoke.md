---
type: experiment
status: smoke_completed
date: 2026-04-26
project: full-vector-calibration
benchmark: CIFAR-100
model: ResNet-18
preregistered: partially_corrected
source: github
tags: [isotonic, argmax, structural-error, smoke]
---

# 2026-04-26 Post-fusion top-isotonic smoke

## Original hypothesis

Apply isotonic regression only to the top coordinate of the
`full_vector_distance_fusion` output, then proportionally renormalize the tail.
The original pre-registration assumed this was effectively argmax-preserving.

## Smoke result

That structural assumption was false.

On CIFAR-100 / ResNet-18 / seed 21:

- `argmax_change_rate_vs_fusion = 0.1712`
- mean top probability before recalibration: ~0.712
- mean top probability after recalibration: ~0.601

Lowering the top probability and rescaling the remaining mass can make a
previously non-top class overtake the original top class.

## Scientific lesson

A transform that directly changes the top probability and redistributes the
remaining probability mass is **not** necessarily decision-preserving, even if
it does not explicitly reorder logits.

The method was correctly reclassified from decision-preserving to
decision-changing; the pre-registered metric bounds themselves were not moved.

## Observation extracted

[[Top-coordinate recalibration plus tail renormalization can change argmax]]

## Why this belongs in the vault

This is exactly the kind of structural mistake a research memory should preserve:
otherwise the same intuitive-but-false assumption can reappear in a later method.
