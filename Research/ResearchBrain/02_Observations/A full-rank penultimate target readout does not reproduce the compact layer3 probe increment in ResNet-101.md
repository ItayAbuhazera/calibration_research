---
type: observation
status: open
date: 2026-09-28
project: Full-Vector Geometric Calibration
evidence_strength: 2
tags: [G1, penultimate, compact-summary, target-supervised, resnet101]
---

# A full-rank penultimate target readout does not reproduce the compact layer3 probe increment in ResNet-101

## Observation

In ResNet-101 (CIFAR-100, checkpoints 2 and 4) on the 12 exposed CIFAR-100-C development cells, an anchored linear readout fitted with 8k target labels gains +1.88 / +1.80 pp macro accuracy over the anchored Z-only readout when given the 100-d layer3.22 probe logits P, but only +0.27 / +0.39 pp when given the full 2048-d penultimate representation H_L. Adding H_L to (Z, P) lowers accuracy by 1.38 / 1.16 pp.

## Evidence

[[2026-09-28 G1 Conditional Accessibility Gatekeeper]]; `GeometricFullCalibration/results/g1/report/g1_aggregate.json` (arms B, C, D vs A at T-8k1); fits `results/g1/fits/seed{2,4}/T-8k1/`; 95% image-group bootstrap intervals in `g1_table.md`. All H_L arms selected the strongest ridge on the grid (λ = 0.1; the extension to 1.0 was worse).

## What it does NOT establish

**This does not establish that the relevant information is absent from H_L.** It concerns one readout family (anchored linear, single shared ridge λ, per-coordinate standardization), one budget (8k target labels; 2.5k also tested), two checkpoints, exposed development cells, and target-supervised (oracle) fitting. It says nothing about depth specificity (G1 found none beyond the parameterization band), deployability, or other architectures. It differs from fine-tuned ResNet-50 (b10), where a full h_L readout matched the layer3 increment ([[2026-09-26 Regime-Map Follow-up Capacity and Ceiling Controls]]).

## Possible mechanisms

(Interpretation.) Estimation cost of a 2048-d × 100-class readout at 8k rows forces heavy shrinkage that also suppresses the P coefficients in the joint fit; the clean-supervised probe is a compact, pre-trained summary that is cheaper to use.

## Related failure modes

[[Equivalent feature spans can differ materially under finite-sample regularized readouts]].

## Related hypotheses

[[H-ACCESS-01 Mid-depth evidence adds target-fitted value beyond (Z, H_L) in end-to-end ResNet-101 under corruption]] (not supported at the tested budget).

## Decisive next test

None authorized on this branch (Stage 0 closed under frozen Outcome C).
