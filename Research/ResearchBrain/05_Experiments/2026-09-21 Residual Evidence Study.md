---
type: experiment
status: completed_stopped_by_frozen_gate
date: 2026-09-21
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C (12 development cells), ResNet-101, checkpoint seeds 2 and 4, corrected_v2_train_norm
preregistered: true
evidence_scope: unpublished_repository_analysis
tags: [residual-readout, spatial-pooling, class-distance, logit-controls, decision-utility, null-result]
---

# 2026-09-21 — Residual decision-information study (accessible-information diagnostic)

Frozen spec (hash-verified, results appended below its marker): repo `Research/GeometricFullCalibration/docs/residual_evidence_study_spec.md`
(sha256 `bd928cd8…dd3a`; `results/residual_study/freeze_manifest.json`). Predecessors: [[2026-09-21 Layer-Selection Pilot]], [[2026-09-21 Normalization Audit and Corrected Protocol]],
[[2026-09-20 Full-Vector DAC POC]] (legacy protocol, closed). Hypothesis: [[H-RESID-01 Source-learnable residual decision information at layer3.22]]. Theory: [[Theory Plan - Decision Utility, Layers, Compression and Risk Control]].

## Question
Does a specified spatial or class-distance representation expose useful, **source-learnable** decision information beyond strong full-logit readouts, using the same correction family for every source and accounting for
capacity, selection and calibration cost? *Accessible-information diagnostic:* not a Bayes bound, not a new calibrator.

## Design (verified implementation facts)
One layer, `layer3.22` (no layer search). Evidence: G (GAP → fixed 100-d Gaussian map), S (2×2 pool flattened → fixed map), DG / DS (class-wise K_c=5 radii from GAP / 2×2 banks, 100-d, unprojected),
O (fixed random ReLU features of logits), DL (class radii in centered-normalized logit space from the *same labeled bank* = label-access control). Common form `softmax(B(z) + W_F φ_F)` with a frozen
output-only anchor `B` (identity / VS / ridge-to-identity matrix scaling), `λ ∈ {1e-4,1e-2,1,100}` plus exact zero, fit on 2 500 FIT rows; two clean-selection policies (NLL; Decision = max accuracy s.t. NLL ≤ anchor+0.01);
primary = hidden-pool Decision policy vs output-pool Decision policy. Stage 0 verified splits/indices/checkpoint hashes; Stage 1 used only existing predictions.

## Results (measured; full tables in repo spec §16)
* **Stage 1 (existing predictions, frozen challengers):** evaluation-only oracle union of the geometric challengers adds only **+0.46 pp** beyond the union of the two full-logit controls (limited oracle reading; no gate trained).
* **Development Stage 2 (12 cells × 2 seeds):** primary +0.102 pp over base; output-Decision control +0.118 pp; anchor +0.108 pp; VS +0.128 pp. Every fixed arm with λ ≥ 100 is within ±0.05 pp of the anchor; λ ≤ 1e-2 collapses accuracy by 3–19 pp (10 000 parameters on 2 500 rows); clean selection picked the top of the λ grid (‖W‖_F ≈ 0.1–0.2) almost everywhere.
* **Frozen gate: FAIL on criteria 1 (gain +0.102 < 0.50), 2 (margins −0.026/−0.006/−0.016 < 0.25), 3 (−0.043 seed 2, +0.012 seed 4; 1/4 families positive), 4 (NLL +0.098, ECE +0.059 vs native DAC).** Criteria 5–6 pass. **Decision: stop**; no confirmation launched.
* Other measured facts: spatial (S) repairs almost nothing that GAP (G) missed (repaired-set Jaccard 0.87); DG/DS Jaccard 0.81; DL/O ≥ hidden evidence; the NLL policy in seed 4 chose a λ=1 arm that lowered accuracy and worsened mean true-class rank (9.52 vs base 8.55) while the Decision policy chose the near-anchor λ=100 arm; interventions concentrate in the lowest base-margin quintile.

## Decision map (pre-declared; what the outcome supports)
* S beats G / output controls? **No** → no evidence of accessible spatial information in this pipeline (the earlier +0.3–0.44 pp additive-β signal did not reproduce as a common-readout advantage).
* G/S beat DG/DS? No (all ≈ anchor) → no evidence these summaries/readouts were the limiting factor.
* DG/DS help where scalar FV-DAC did not? **No.**
* DL/O match hidden evidence? **Yes (slightly ahead)** → no demonstrated internal-representation advantage.
* Decision selection helps where NLL selection does not? Weak, one of two checkpoints, and the "helped" arm ≈ anchor → not supported as a result; subject to (unrun) confirmation.
* Clean benefit reverses on corruptions? Clean-selection signal itself ≈ 0 → transfer is secondary.
* Only probability metrics improve? Neither: calibration is at the anchor's level (worse than native DAC's density temperature).
* **Conclusion:** stop this bounded source-only readout direction *at this sample/compression budget*. Not a claim that no information exists anywhere in the network; the dominant limit observed is finite-sample learnability from 2 500 clean fit rows.

## What this does NOT establish
Absence of information in the un-pooled tensor or other layers; anything for readouts with more fit labels or other parametrizations (e.g. low-rank/structured); cross-seed or cross-architecture behaviour (two development checkpoints);
population Bayes gap; significance. Development conditions were already inspected by earlier experiments.

## Measurements handed to the theory work
Retained rank (participation ratio of φ of 100: G ≈ 38, S ≈ 60, DG ≈ 2.1, DS ≈ 2.9, O ≈ 30, DL ≈ 10); projection distortion (G,S: mean ratio ≈ 1.0, sd 0.07, p95 rel. squared-distance error 0.26–0.29; JL sufficient dims 9 184 / 2 473 / 515 for ε=0.1/0.2/0.5 ≫ 100);
selected λ at the grid boundary; fit-vs-select NLL gaps (λ=1: 0.15–0.5 nats; λ=100: none); layer dims 1024 (GAP) / 4096 (2×2); corrected base margins (interventions only in the lowest quintile); clean → corruption signed-utility changes; costs (bank query 0.05–0.12 s per 10 000 queries).
No population Bayes gap is estimated.

## Unresolved / provenance
* Corrected benchmark corruption cells (job array 21533078) still running when written; clean-cell reconciliation done. The benchmark fits TS/VS on the whole validation split; this study refits them on FIT rows (VS clean ΔNLL +0.017/+0.023 vs the benchmark's).
* **IJCAI-era CIFAR-100 preprocessing (unresolved, no claim made):** the canonical RGC repo's `run_post_hoc_calibration.py` uses the legacy `get_test_loader`, and stored `calibration_comparison/ablation_*_cifar100_resnet101_seed{2,4}.json` record uncalibrated accuracy 76.20 / 76.31 % (legacy-normalization values; corrected would be 76.59 / 76.51 %). Which artifacts the published tables used is not established.

## Artifacts
Repo: `Calibrators/residual_readout.py`, `Experiments/{residual_evidence_study, residual_stage0_stage1, aggregate_residual_study, residual_dev_extra_diagnostics, crosscheck_residual_vs_benchmark}.py`, `tests/test_residual_evidence_study.py` (12 tests),
`scripts/residual_study_{dev,confirm}.sbatch`; `results/residual_study/{stage0_stage1,dev,freeze_manifest.json,val_train_overlap.json}`. Frozen-state hashes: seed 2 `8bcde030…`, seed 4 `cd4f2e16…`.
