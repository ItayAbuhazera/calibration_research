---
type: experiment
status: planned_frozen_before_fitting
date: 2026-09-24
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C (12 development cells), ResNet-101, checkpoints 2 and 4, corrected_v2_train_norm
preregistered: true
experiment_type: mechanism discrimination (target-supervised development diagnostic)
parent_question: Is the Stage 0 probe-logit gain specific to layer3.22, or generic fusion of a diverse second predictor under target supervision?
exposure_ledger: "[[Representation Correction Exposure Ledger]]"
evidence_scope: unpublished_repository_analysis (development; not confirmation; not a deployable method)
tags: [stage0, ablation, evidence-source, probe-logits, generic-fusion]
---

# 2026-09-24 — Stage 0 evidence ablation: layer4.2 probe, another checkpoint's logits, and all cached layer probes

Parent: [[2026-09-22 Stage 0 Probe-Logit Increment Study]]; hypothesis (proposed) [[H-STAGE0-01 Layer3.22 probe evidence is redundant with the logits on clean data but complementary under corruption, and clean supervision cannot identify the useful combination]] (explanation 5, generic fusion, and explanation 1 are the ones this contrast can separate). Frozen specification: repo `docs/stage0_evidence_ablation_spec.md` (authoritative; read it for every choice). Workflow: uncommitted working-tree version of `docs/research_workflow.md` and current `Templates/Experiment.md` (the updated workflow files had not landed in git when this card was written).

## Question

Claim scope: ResNet-101, CIFAR-100, checkpoints 2 and 4, the 12 exposed CIFAR-100-C development cells, the Stage 0 folds/regimes/objective; only the second evidence source `P` changes; linear multinomial readout; top-1 accuracy. Target-supervised diagnostic; not a deployable method; not an information ceiling.

## Competing explanations and predictions

* **Depth-specific** (the `layer3.22` evidence is needed): a final-layer probe `P_A` (layer4.2, ≈ the head's own features) gives a much smaller target-fitted increment than `layer3.22`; another network's logits `P_B` do not reproduce the Stage 0 pattern.
* **Generic fusion under shift:** `P_B` gives the same pattern (clean increment ≈ 0, Δ_S ≈ 0, Δ_T ≫ 0), and `P_A` gives a comparable increment.
* **Generic diversity:** across the 12 layer probes Δ_T tracks only how often `P` disagrees with `Z`, and `P_B` lies on the same curve.
Predictions overlap partly: a depth-independent diversity effect would satisfy the second and third at once.

## Experiment and primary contrast

Primary type: mechanism discrimination. Uncertainty tested: readout expressiveness / transfer. **Primary contrast:** D_A = Δ_T(P_layer3.22 | Z) − Δ_T(P_A | Z) at T-8k×1, per checkpoint, paired image-group bootstrap (2,000 resamples). Minimum controls: `q_Z` refit compared with Stage 0's `q_Z` (determinism check), convergence audit, λ-edge audit. Claim class: correctness of a fitted readout.
Per `P` also reported: standalone accuracy, disagreement rate with `Z` (clean, per cell), Δ_S, gap, clean-view increment, at 8k×1 (and 2.5k×1 for P_A, P_B).

## Setup

Sources: P_A = `raw__probe_logits[:, 11, :]`; P_B = the other checkpoint's atlas `u0` logits on the same images; P_C = the other cached layer probes (indices 0–7, 9, 10; T-8k×1 and S-8k×1 only); P_ref = Stage 0 outputs reused. Regimes: T-8k×1, S-8k×1, T-2.5k×1, S-2.5k×1 for P_A and P_B (primary 8k×1). Data/fitting access: identical to Stage 0 (same folds, roles, target-label access); no reserved data (checkpoints 1/3/5, 11 unused families) and no new feature extraction. Resources: CPU only. 280 planned runs (560 arm-fits).

## Fixed choices

Everything of Stage 0 unchanged (folds, λ grid, objective without ½ factor, standardization, L-BFGS and recovery policy, bootstrap). Snapshot-from-commit provenance; post-hoc additions labelled.

## Primary metrics / baselines

Accuracy (primary), NLL, Brier; baselines: target-fitted and clean-fitted `q_Z`, and the Stage 0 `layer3.22` `q_ZP`.

## Outcome-to-decision matrix (frozen; thresholds in the spec §4; all must hold in both checkpoints)

| Possible outcome | Explanation supported/weakened | Still unresolved | Next decision |
|---|---|---|---|
| **R1:** Δ_T(P_A) ≥ 0.5 × Δ_T(P_layer3.22) at T-8k×1 | Depth-specific story weakened (final-layer evidence gives a comparable target-fitted gain) | whether gain is generic fusion or something shared by deep probes | Stop the depth-specific story |
| **R2:** P_B: Δ_T ≥ +1.0 pp (interval excludes 0), Δ_S ≤ +0.5 pp, clean increment within ±0.5 pp | Gap is generic fusion under shift; not specific to layer probes | cause of the clean/target gap | Treat Stage 0 as generic fusion |
| **R3:** across 12 layers Spearman(disagreement, Δ_T) ≥ 0.8 and P_B within 2 × residual RMSE of the line | Generic diversity, depth adds nothing beyond disagreement | why diversity helps only with target labels | Treat as generic diversity |
| None of R1–R3 (verdict = first met in order R1, R2, R3) | Depth-specific pattern survives for this readout and these cells | mechanism; other readouts; confirmation | Propose the confirmation / paper package for a separate decision; do not start it |

Practical scale: "near zero" = within ±0.5 pp; "clearly positive" = ≥ +1.0 pp with interval excluding 0. Intervals are conditional on the fitted CV predictions; two non-independent checkpoints; 12 layer points is descriptive.

## Results

*(to be appended after runs complete)*

## Protocol deviations

None yet.

## Post-mortem and allocation

*(to be completed after results)*

## Amendments / engineering recovery

* 2026-09-24: card and spec frozen before any ablation fit.

## Pre-results addendum (appended 2026-09-24, before any aggregate output was read)

Written while the ablation jobs were running and the aggregate had not been opened. The frozen spec, rules R1–R3, thresholds and running jobs are unchanged. What had been seen before this addendum: the Stage 0 results (including the `layer3.22` reference Δ_T/Δ_S), and one fast-mode smoke test of the other-checkpoint evidence (seed 2, S-2.5k×1, fold 0, single λ, discarded) whose clean-view accuracy gain was about +4 pp.

The analyses below are **secondary and descriptive, with no decision attached**; they cannot change the R1–R3 verdict, and R1–R3 is reported first.

1. **Per evidence source, standalone and overlap.** Standalone accuracy of `P` (clean and per cell); disagreement rate of argmax `P` with argmax `Z` (clean and per cell); and the rate at which `P` and `Z` are both wrong on the same image (clean and per cell).
2. **Gap comparison.** G(`layer3.22`) − G(other-checkpoint logits), with G = Δ_T − Δ_S (12-cell macro), at 8k×1 and 2.5k×1, per checkpoint, using a paired image-group bootstrap in which both sources' per-image effects are resampled with the same indices.
3. **Clean increment versus recoverable share, across all 13 sources** (12 layer probes including `layer3.22`, plus the other checkpoint's logits): scatter of the S-8k×1 clean-view increment against Δ_S/Δ_T at 8k×1, with Spearman correlation, per checkpoint. Descriptive only. Δ_S/Δ_T is unstable when Δ_T is near zero; sources with Δ_T < 0.5 pp will be marked.

**Note on R2.** R2 requires a near-zero clean-view increment. The smoke test above, seen before this addendum was written, suggested a clean increment of about +4 pp for the other checkpoint's logits (a second network's logits are a strong ensemble partner on clean images). An R2 failure is therefore expected and is weak evidence on its own about depth specificity; item 2 (the gap comparison, which removes the clean-image ensemble effect from the comparison) carries the informative contrast.

**Order of reading.** (a) Check that each refit `q_Z` reproduces the Stage 0 `q_Z`; if any differs beyond numerical noise (max absolute probability difference > 1e-6), stop and report before reading any contrast. (b) Report the frozen R1–R3 verdict. (c) Then the secondaries above. Nothing else is started by this addendum.

## Determinism check — recorded 2026-09-24 02:30, before any contrast was read (STOP RULE TRIPPED)

Per the addendum, each refit `q_Z` was compared with the Stage 0 `q_Z` (max absolute probability difference > 1e-6 = stop and report). Only the check block of `results/stage0_ablation/report/ablation_aggregate.json` and a diagnostic on the 280 files were read; no R1–R3 value, contrast or secondary output has been read.
* 97 of 280 files exceed 1e-6; maximum 2.1e-4. 183 files are identical (deviation exactly 0), including all 20+20 files of the T-2.5k×1 and S-2.5k×1 regimes for every evidence source.
* Deviations occur only at 8k×1: T-8k×1 53 of 120 files, S-8k×1 44 of 120. Selected λ matches Stage 0 in all 280 files. Every ablation arm-fit converged with no retries and no λ at a grid edge (560 arm-fits).
* Prediction effect: 37 argmax differences out of 7,280,000 predictions (0.0005 %); at most one image flips in any file.
* The seed-4 S-2.5k×1 fold-0 file that was rewritten by the earlier agent rerun (Stage 0 deviation 6) agrees with the fresh refits (max deviation 6e-8 and 0), so that file is consistent with a clean refit.
* Not established: the cause. Candidate: floating-point reduction order in multithreaded BLAS at the larger row count (Stage 0 jobs set only `OMP_NUM_THREADS`; the ablation jobs also set `MKL_NUM_THREADS` and `OPENBLAS_NUM_THREADS`, and nodes differ). This was not tested.
* Operational effect is far below the accuracy scale of any contrast, but the stop rule was written as a threshold and it was crossed; continuing is left to the researcher.

Aggregation history: the first aggregate job (21648590) hit its 1-hour limit (slow shared-disk I/O); the automatic engineering retry (21652108, same snapshot and code) completed in 3 min 20 s; the post-hoc secondary job (21652109, snapshot `stage0_abl_v2_a4cfd2046536`) completed. Outputs exist and are unread.

