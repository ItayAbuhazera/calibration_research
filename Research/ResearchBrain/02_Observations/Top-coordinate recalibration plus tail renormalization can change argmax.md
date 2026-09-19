---
type: observation
status: structural
date: 2026-04-26
project: full-vector-calibration
evidence_strength: 4
tags: [argmax, isotonic, probability-transform]
---

# Top-coordinate recalibration plus tail renormalization can change argmax

## Observation

The post-fusion top-isotonic smoke changed the argmax on 17.12% of the examined
CIFAR-100 samples.

## Mechanism

If the calibrated top probability is reduced from `u` to `u'`, proportional tail
renormalization multiplies all non-top probabilities by

(1 - u') / (1 - u).

A sufficiently large factor can push a previous runner-up above the recalibrated
top class.

## General lesson

Do not infer "decision preserving" merely because a post-hoc method targets one
probability coordinate. Prove argmax invariance algebraically or test it.

## Related

- [[2026-04-26 Post-Fusion Top-Isotonic Smoke]]
