---
type: experiment
status: results_missing_from_repo
date: 2026-06-21
project: decision-improving-calibration
benchmark: CIFAR-100
model: ResNet-18
seed: 21
preregistered: true
source: github
tags: [glad-pi, net-flips, actionable-calibration]
---

# 2026-06-21 Decision-Improving Calibration pilot

## Status

The repository contains the pre-run reference and states that the run started,
but the inspected document does **not** contain completed results. Therefore this
note records the design only; do not treat it as evidence.

## Structural premise

Positive scalar temperature transforms — including sample-dependent scalar
temperatures — preserve logit ordering and therefore cannot improve accuracy by
changing decisions.

The benchmark therefore separates methods by whether corrections are
sample-dependent and class-dependent.

## Primary planned hypothesis

GLAD-PI should produce positive net beneficial flips while keeping top-label ECE
and NLL within pre-specified tolerances.

## Why this is relevant now

This is an earlier manifestation of the broader research pattern:

**uncertainty/calibration signal → operational decision change**

That is directly relevant to current work on actionable confidence.

## Next action

Locate the actual run outputs before converting this design into observations.
