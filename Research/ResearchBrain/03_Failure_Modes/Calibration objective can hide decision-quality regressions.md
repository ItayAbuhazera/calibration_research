---
type: failure_mode
status: open
project: full-vector-calibration
tags: [metric-mismatch, ece, decision-quality]
---

# Calibration objective can hide decision-quality regressions

## Failure

A method can substantially improve top-label ECE while worsening NLL, Brier, or
accuracy.

## Evidence

[[Top-label calibration and full-vector proper scoring form a Pareto frontier]]

## Why it matters

Optimizing one calibration summary can produce a misleading "better calibrator"
claim if the intended downstream decision depends on the full probability vector
or argmax quality.

## Guardrail

Every future calibration experiment should declare which of these is primary:
- confidence calibration,
- full-distribution proper scoring,
- decision accuracy,
- selective / action utility.
