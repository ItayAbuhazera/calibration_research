---
type: failure_mode
status: open
project: Full-Vector Geometric Calibration
date: 2026-09-28
tags: [parameterization, ridge, readout, G1, regime-map]
---

# Equivalent feature spans can differ materially under finite-sample regularized readouts

## Failure

Two inputs with the same linear span — (Z, H_L) and (Z, P_ker H_L), where P_ker is the null-space projector of the head W (so P_row H_L is an affine image of Z) — give materially different accuracy when fitted with the same ridge-penalized, per-coordinate-standardized readout at a finite label budget.

## Where observed

* G1 (ResNet-101, T-8k×1, anchored, target labels): G − C = **+0.72 [+0.61, +0.83] / +0.50 [+0.39, +0.61] pp** (checkpoints 2 / 4); `GeometricFullCalibration/results/g1/report/g1_aggregate.json` (`GminusC`). [[2026-09-28 G1 Conditional Accessibility Gatekeeper]]
* Regime-map follow-up (ResNet-50 fine-tuned states b*): Z+K − C1d ≈ **+0.12 to +0.32 pp**; P_row (a linear image of z) moved accuracy by −0.74 to +0.56 pp; `results/regime_map_followup/report/stage2_anchored.json`. [[2026-09-26 Regime-Map Follow-up Capacity and Ceiling Controls]]
* Not applicable to state (a) of the regime map, where K was built on a normalized u and the spans are not equal.

## Why existing signal/rule fails

A ridge penalty on standardized coordinates is not invariant to invertible linear reparameterization: removing the row-space block, rescaling, or re-standardizing changes the effective prior, and at strong regularization (all these arms selected the strongest grid λ) the prior dominates.

## Competing explanations

Regularization geometry (above); λ-selection noise on a coarse grid; optimizer tolerance (ruled out as the main cause: all fits converged, no retries).

## What would falsify this failure mode

Equivalent-span arms agreeing within ≈ 0.1 pp under the same protocol at comparable budgets.

## Related observations

[[A full-rank penultimate target readout does not reproduce the compact layer3 probe increment in ResNet-101]].

## Research opportunity / implication

**Sub-band effects from high-dimensional readouts are not interpretable as representation-information differences without a parameterization audit.** Any future contrast involving 2048-d features should report an equivalent-span (or reparameterized) arm and treat effects smaller than its band as uninterpretable.
