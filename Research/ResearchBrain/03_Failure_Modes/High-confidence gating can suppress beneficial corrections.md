---
type: failure_mode
status: open
project: full-vector-calibration
tags: [gating, anchoring, decision-change]
---

# High-confidence gating can suppress beneficial corrections

## Failure

A gate or anchor that treats high base confidence as a reason to preserve the
head can remove exactly the decision-changing behavior responsible for accuracy
gains.

## Evidence

[[High-confidence anchoring can block useful full-vector decision changes]]

## Connection to current work

This failure mode should be remembered when designing any "commit when safe"
mechanism: confidence in the current answer is not the same as expected utility
of keeping the current action.
