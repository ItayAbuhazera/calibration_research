---
type: hypothesis
status: not-supported-under-this-protocol
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C, ResNet-101, checkpoints 2 and 4 (development)
novelty: unknown
tags: [H-GATE, candidate-selection, gate, layer3]
---

# H-GATE-01 — Selecting candidates by standalone accuracy (E[D]) discards a selectively deployable alternative (E[gD])

Experiment: [[2026-09-21 Fixed Deep Candidate Gate Study]]. Go criterion (frozen, practical, not a formal test): a fixed deep-layer3 2×2 candidate with a clean-trained gate improves held-out clean utility of Z1 over Z0 with the same sign and paired intervals excluding 0 in both checkpoints, transfers to the 12 corruption cells, and reaches +0.5 pp over base and +0.25 pp over the output control without worse NLL/Brier. Kill criterion: no consistent held-out benefit.
**Status:** not supported in development. Z1−Z0 clean +0.10/+0.13 pp (intervals include 0), corruption −0.04/−0.04; deep Z1 macro +0.075/+0.21 pp over base ≈ output-evidence gates and output/layer4 controls; no growth with n or m. Does not show that a different global selection algorithm cannot help, nor that information is absent.
