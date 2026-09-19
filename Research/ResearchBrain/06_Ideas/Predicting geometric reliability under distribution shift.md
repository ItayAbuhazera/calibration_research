---
type: idea
status: candidate
novelty: unknown
top_tier_potential: unknown
benchmark: CIFAR-100-C
project: rgc-shift
tags: [geometry, reliability, ood, selective-decision]
---

# Predicting geometric reliability under distribution shift

## One-sentence contribution

Instead of asking whether geometry improves predictions on average, predict
when a geometric correction is trustworthy under shift and control decisions
using that reliability estimate.

## Phenomenon

[[Geometry contains complementary accuracy information under corruption]]
[[Validation-fitted neighbourhood reliability features fail under corruption]]
[[Per-class geometry exposes more oracle headroom than global kNN]]

## Core hypothesis

There exists a representation-derived reliability signal beyond the current
local neighbourhood statistics that predicts when geometry should override the
head under distribution shift.

## Why this may be new

Unknown. Requires an adversarial literature audit before investment.

## Benchmark

CIFAR-100-C initially. Must later include another architecture / dataset family
if the mechanism survives.

## Decisive POC

First determine whether the failure is:
- weak in-domain observability, or
- transfer collapse under corruption.

Only if observability exists should richer reliability modelling be attempted.

## Kill criterion

If even oracle-supervised diagnostic models using reasonable representation
summaries cannot predict geometry-better-than-head out of sample, stop.

## Reviewer #2 attack

"This is just learning when kNN is right."

The project only becomes interesting if it isolates a general reliability
phenomenon, demonstrates failure of standard uncertainty / density signals, and
shows a principled risk-controlled decision advantage.

## Current verdict

PROMISING OBSERVATION, NOT YET A RESEARCH PROJECT.
