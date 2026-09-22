---
type: hypothesis
status: not-supported-under-this-protocol
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C, ResNet-101, checkpoints 2 and 4 (development)
novelty: unknown
tags: [H-ATLAS, layers, pooling, metric, gate]
---

# H-ATLAS-01 — Frozen representations expose practically accessible information for correcting classification errors

Experiment: [[2026-09-21 Representation Atlas Program]]. Refutation conditions frozen before outcomes (repo spec §8):
* **H-A spatial information:** clean-selected grid2/SPP candidate beats the clean-selected GAP candidate in macro corruption net utility by ≥ 0.5 pp in both checkpoints. Refuted otherwise.
* **H-B complementarity:** best hidden candidate's unique repairs exceed its unique harms relative to the output-space candidate, both checkpoints. Refuted otherwise.
* **H-C metric:** raw_l2 or Mahalanobis beats unit_l2 on clean-selection net gain by ≥ 0.5 pp for ≥ 1 pooling in both checkpoints. Refuted otherwise.
* **H-D gate:** F1 beats F0 (same candidate) by ≥ 0.10 pp in both checkpoints and hidden-F1 beats output-F1. Refuted otherwise.
* **H-E practical:** hidden-F1 post-temperature ≥ +0.5 pp over base and ≥ +0.25 pp over the strongest output control with non-worsening NLL and Brier, same direction in both checkpoints. Refuted otherwise.
**Status (2026-09-21): H-A…H-E all refuted** in development (2 checkpoints): spatial−GAP +0.11/−0.03 pp; unique repairs−harms −70/+80; metric gain ≤ +0.48 pp; F1−F0 −0.02/0.00 pp; hidden-F1 +0.07/+0.29 pp over base and −0.01/+0.04 pp over the strongest output control. Scope: development cells, clean-selected candidates (layer4), frozen Mahalanobis estimator, ridge gate on ≤151 clean disagreements. Not evidence that mid-layer alternatives are unusable: they exist (10–14 % of base errors) but harm 38–52 % of base-correct examples. See [[2026-09-21 Representation Atlas Program]].

**Correction (2026-09-21):** "refuted" above means the frozen mechanical thresholds were not met; see [[2026-09-21 Fixed Deep Candidate Gate Study]] and `results/fixed_gate/report/hypothesis_audit.json` for which components have interval evidence against the threshold magnitude (H-A, H-D, H-E, layer4 candidates only), which are seed-dependent/uncertain (H-B, H-C raw_l2). Follow-up [[H-GATE-01 Candidate selection versus gate utility mismatch]].
