---
type: hypothesis
status: not_supported_at_tested_budget (G1 outcome C)
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C (12 development cells), ResNet-101, checkpoints 2 and 4
novelty: limited (ILC few-shot implies the direction for a CIFAR-trained ResNet-18 at an OOD-selected layer; not measured conditionally)
tags: [H-ACCESS, conditional-accessibility, penultimate, stage0, G1]
---

# H-ACCESS-01 Mid-depth evidence adds target-fitted value beyond (Z, H_L) in end-to-end ResNet-101 under corruption

## Formal statement

Claim scope: ResNet-101 (CIFAR-100, `baseline_cross_entropy`), checkpoints 2 and 4; the 12 exposed CIFAR-100-C development cells;
evidence Z (native logits), H_L (layer4 GAP, 2048-d), P_3.22 (cached layer3.22 clean-probe logits); anchored linear readout
q = softmax(z + f(x)) fitted with target labels (T-8k×1); top-1 accuracy. Statement: Acc_T(Z, H_L, P_3.22) − Acc_T(Z, H_L) is
practically material (≥ +0.5 pp), beyond a recipe-matched redundant summary of H_L (P_4.2) and not matched by a competent
independent predictor (Z_other). A positive result means additional **operationally accessible** evidence for this readout and
budget only.

## Why it follows from evidence

Stage 0: (Z, P_3.22) − (Z) = +2.87 / +2.63 pp at T-8k×1; the layer3 plateau vs layer4.x ≈ 0; at matched 100-d, P_3.22 beats
P_4.2 by ≈ +3 pp ([[2026-09-24 Stage 0 Evidence Ablation]]). But no H_L arm was ever run on ResNet-101, and in fine-tuned
ResNet-50 a full-rank h_L target readout recovered ≈ the layer3 increment (D − C1d from +0.04 to +0.14 at b3/b10; +0.51 vs +0.09
at b1 across seeds; [[2026-09-26 Regime-Map Follow-up Capacity and Ceiling Controls]]). The joint arm (Z, h_L, P) was never fit anywhere.

## Competing explanation

E1 head bottleneck (H_L already has it); E3 sample efficiency (P is a compact summary easier to estimate from 8k labels);
E4 generic diverse second predictor; E5 parameterization/regularization; E6 target recalibration (cancels in the nested contrast).
Prediction overlap: E1 and E3 both predict D ≈ C at large budgets; only the budget comparison separates them, and two budgets cannot
establish asymptotic sufficiency.

## Benchmark

Stage-0 folds/regimes on the exposed development cells only; no reserved families; no checkpoints 1/3/5.

## Baselines

Anchored Z-only; (Z, P_3.22); (Z, H_L); (Z, H_L, P_4.2); (Z, H_L, Z_other); (Z, P_ker H_L) equivalent span; shuffled-P.

## Decisive experiment

[[2026-09-28 G1 Conditional Accessibility Gatekeeper]]; frozen spec `GeometricFullCalibration/docs/g1_conditional_access_spec.md`.
Primary type: mechanism discrimination (target-supervised diagnostic). Uncertainty tested: readout expressiveness / finite-sample
estimation. Exposure: target labels of the 12 exposed cells for fitting; development reuse ([[Representation Correction Exposure Ledger]]).

## Outcome-to-decision matrix

See the experiment card (outcomes A–F). Only A leaves anything eligible.

## Kill criterion

Outcomes B, C, D, E or F of the frozen G1 rule.

## Go criterion

Outcome A in both checkpoints (makes a robustness stage *eligible*; does not start it).
Practical threshold and uncertainty limitations: +0.5 pp material, +0.2 pp negligible upper bound (Stage-0 conventions); intervals
conditional on fitted CV predictions; two checkpoints only.

## Prior-art threats

ILC (Uselis & Oh, ICLR 2025; few-shot +1.6 pp on CIFAR-100-C for ResNets, replacement estimand, OOD-selected layer); Head2Toe
(ICML 2022; joint all-layer readout incl. pre-logits, frozen, VTAB); Attentive multi-layer fusion (arXiv 2601.09322; frozen ViTs, no
corruption); DFR (ICLR 2023; penultimate retraining).

## Update 2026-09-28 (G1 result; appended)
[[2026-09-28 G1 Conditional Accessibility Gatekeeper]]: frozen outcome **C (sample-efficiency / capacity confound)** in both checkpoints. Δ_cond = +0.24 pp [+0.20, +0.29] (both), below the +0.5 material scale and inside the equivalent-span band (|G − C| = 0.72 / 0.50); the joint (Z, H_L, P) readout is 1.2–1.4 pp worse than (Z, P); (Z, H_L) adds only +0.27 / +0.39 over Z vs +1.88 / +1.80 for (Z, P). Not supported at the tested budget/readout; absence of the information from H_L is **not** shown. Branch stopped per the frozen rule.
