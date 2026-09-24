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

## Pre-reading note — noise bound on the `q_Z` deviations (2026-09-24, before any contrast was read)

Decision (researcher): treat the `q_Z` deviations as numerical noise **if** a worst-case bound is below 0.1 pp. Script `atlas/stage0_ablation_noise_bound.py` (reads only `q_Z` predictions; output `results/stage0_ablation/report/noise_bound.json`). Entries are argmax differences between the refit `q_Z` and the Stage 0 `q_Z`, summed over the 5 folds: **12 corrupted cells / clean copy** (N = 10,000 images per condition).

| evidence | s2 T-8k1 | s2 S-8k1 | s2 T-2.5k1 | s2 S-2.5k1 | s4 T-8k1 | s4 S-8k1 | s4 T-2.5k1 | s4 S-2.5k1 |
|---|---|---|---|---|---|---|---|---|
| L11 | 0/0 | 2/0 | 0/0 | 0/0 | 1/0 | 1/0 | 0/0 | 0/0 |
| xckpt | 0/0 | 1/0 | 0/0 | 0/0 | 1/0 | 2/0 | 0/0 | 0/0 |
| L0 | 0/0 | 1/0 | – | – | 1/0 | 1/0 | – | – |
| L1 | 0/0 | 1/0 | – | – | 1/0 | 1/0 | – | – |
| L2 | 0/0 | 1/0 | – | – | 0/0 | 1/0 | – | – |
| L3 | 0/0 | 1/0 | – | – | 1/0 | 1/0 | – | – |
| L4 | 0/0 | 1/0 | – | – | 1/0 | 1/0 | – | – |
| L5 | 0/0 | 1/0 | – | – | 1/0 | 1/0 | – | – |
| L6 | 0/0 | 1/0 | – | – | 1/0 | 1/0 | – | – |
| L7 | 0/0 | 1/0 | – | – | 1/0 | 1/0 | – | – |
| L9 | 0/0 | 1/0 | – | – | 1/0 | 1/0 | – | – |
| L10 | 0/0 | 1/0 | – | – | 1/0 | 1/0 | – | – |

Worst case = every differing prediction flips correctness against the true class, effect = differences / N. Largest single-cell bound 0.010 pp (one image in one cell); largest 12-cell macro bound 0.0008 pp per entry; largest bound for any Δ_T − Δ_S gap 0.0025 pp; clean-view increments 0 (no clean differences at all). Total 37 differences. D_A shares one refit `q_Z` arm, so its bound equals a single entry's. **All bounds are below the 0.1 pp stop threshold, so reading proceeds.** Sensitivity check to be reported with the primary: D_A recomputed with Stage 0's `q_Z` as the common reference for both arms (labelled sensitivity). After reading, not as a gate: one T-8k×1 fold rerun with OMP/MKL/OpenBLAS threads pinned to 1, checked against Stage 0.

## Results (read 2026-09-24 after the noise bound; artifacts `results/stage0_ablation/report/`; aggregation from snapshot `stage0_abl_v1_e27abb99ee2b`, secondaries from `stage0_abl_v2_a4cfd2046536`)

**Frozen verdict [pre-specified]: none of R1, R2, R3 is met (both checkpoints) → "the depth-specific pattern survives"; the rule says to propose the confirmation / paper package for a separate decision and not to start it.** A "survives" verdict is not confirmation and establishes no cause.

* **R1 (P_A = layer4.2).** Δ_T(P_A) at T-8k×1 is −0.41 pp (seed 2) and −0.36 pp (seed 4) against +2.87 / +2.63 for `layer3.22`; ratio −0.14, far below 0.5. **Primary contrast D_A = +3.28 pp [3.00, 3.56] (seed 2) and +2.99 pp [2.72, 3.24] (seed 4).** Sensitivity (labelled; Stage 0's `q_Z` as the common reference for both arms): identical to the precision reported (+3.28 [3.00, 3.56], +2.99 [2.72, 3.24]).
* **R2 (P_B = other checkpoint's logits) fails, as the addendum anticipated, on the clean-increment condition and also on Δ_S.** Δ_T = +3.55 [3.32, 3.81] / +2.51 [2.28, 2.73] pp (clearly positive); Δ_S = +3.46 / +2.53 pp (not ≤ +0.5); S-8k×1 clean increment = +2.43 / +2.39 pp (not near zero).
* **R3 fails.** Spearman(disagreement with `Z`, Δ_T) over the 12 layer probes = +0.20 (seed 2) and +0.19 (seed 4), below 0.8. P_B lies above the layer line (residual +2.00 / +1.15 pp against RMSE 0.97 / 0.96; outside 2×RMSE for seed 2 only). Δ_T is non-monotone in depth: ≈ +1 pp at layers 0–1, rising to ≈ +2.6–2.9 pp at layers 5–8, then ≈ 0 to −0.4 pp at layers 9–11, whereas disagreement with `Z` falls monotonically with depth.

**Secondaries [declared in the pre-results addendum; descriptive, no decision attached].**
* **Item 1 (standalone / overlap; clean; 12-cell macro; seed 2 / seed 4).** `layer3.22` probe: accuracy 73.4 / 73.1 clean, 45.5 / 45.9 macro; disagreement with `Z` 0.22 clean, 0.38 / 0.41 macro; both-wrong 0.18 clean, 0.44 / 0.43 macro. Layer4.2: 76.6 / 76.7 clean, 50.0 / 51.0 macro; disagreement 0.05 clean, 0.09 macro; both-wrong 0.22 clean, 0.48 macro. Other checkpoint's logits: 76.5 / 76.6 clean, 51.0 / 50.1 macro; disagreement 0.23 clean, 0.42 macro; both-wrong 0.16 clean, 0.41 macro. Per-cell values are in `ablation_secondary.json`.
* **Item 2 (gap comparison G(layer3.22) − G(other-checkpoint logits), G = Δ_T − Δ_S, paired image-group bootstrap).** 8k×1: +3.27 [2.99, 3.54] (seed 2), +2.94 [2.67, 3.22] (seed 4), with G(layer3.22) = +3.35 / +2.92 and G(other checkpoint) = +0.09 / −0.02. 2.5k×1: +0.86 [0.64, 1.09] / +0.81 [0.60, 1.04], with G(other checkpoint) = +0.62 / +0.36.
* **Item 3 (clean increment vs Δ_S/Δ_T, 13 sources, descriptive).** Spearman +0.40 (seed 2), +0.24 (seed 4), n = 13. Every same-network layer probe has a negative or ≈0 S-fit effect at 8k×1 (Δ_S from −1.42 to +0.29 pp; share Δ_S/Δ_T negative for all layers with Δ_T ≥ 0.5 pp) and a clean increment within about ±0.7 pp, while the other checkpoint's logits sit alone at share ≈ 1.0 with a clean increment of +2.4 pp. Layers 10 and 11 have Δ_T < 0.5 pp (ratio unstable). Figure: `ablation_clean_increment_vs_share.png`.

**Numerical-noise disclosure.** Refit `q_Z` differed from Stage 0's in 97 of 280 files (max 2.1e-4; 37 argmax differences of 7.28 M predictions); worst-case effect on any declared contrast ≤ 0.010 pp (single cell) and ≤ 0.0025 pp for any gap, below the 0.1 pp stop threshold; convergence audit clean (560 arm-fits, 0 unconverged, 0 λ at a grid edge).
**Thread-pinning check (after reading; not a gate).** One T-8k×1 fold (seed 4, fold 1, evidence `layer3.22`) rerun on node `ise-cpu-intl-14` with OMP/MKL/OpenBLAS threads = 1 (job 21657682, 1 CPU, 4 min): does **not** reproduce Stage 0 exactly (max probability difference 3.2e-5 for `q_Z`, 4.5e-5 for `q_ZP`, 0 argmax differences, same λ). It agrees with the earlier 4-thread refit to 2e-6. The Stage 0 task ran on `ise-cpu128-09` (128-core node) and the ablation refits on `ise-cpu-intl-*` (40-core Intel nodes). So thread count is not the cause; a difference between CPU/node types is the leading candidate (one fold, untested on a matching node). For future exact reproducibility, pin the node type (constraint or node list) as well as the threads.

## Interpretation (labelled; not established)

* The Stage 0 pattern (target-fit gain, clean-fit null) is **not reproduced by another network's logits**: they recover under clean supervision (Δ_S ≈ Δ_T, clean increment +2.4 pp). This argues against "generic fusion under shift" as the account of the clean-fit failure for the same-network probes, and the R2 failure alone would have been weak evidence (as pre-declared); the gap comparison and item 3 carry it.
* The evidence for depth is coarse: the target-fit gain is a **mid-depth plateau (layers ≈ 5–8, `layer3.22` included), not specific to `layer3.22`**, is absent at the final layers, and the layer curve is not explained by disagreement with `Z` alone (R3). Same-network probes at all depths have Δ_S ≤ ~0.3 pp.
* Not tested: why same-network intermediate probes are not clean-recoverable while another network's logits are (clean-fit non-identifiability, shift of the probe distribution, and readout mismatch remain open); nonlinear readouts; other architectures; confirmation on any reserved resource.

## Post-mortem and allocation (interim; researcher decision pending)

* Primary contrast: D_A ≈ +3.0 to +3.3 pp with narrow intervals; the depth-specific pattern is not weakened by the final-layer or other-checkpoint controls.
* Claim supported: for this readout, budget and these cells, the target-fit gain from same-network intermediate probes is largest at mid-depth and is not matched by the final-layer probe; only the other checkpoint's logits behave differently, and they behave like an ensemble partner. Not supported: a mechanism, a deployable method, uniqueness of `layer3.22`.
* Allocation: the frozen rule proposes a confirmation / paper package; it is **not started**, needs a frozen candidate and explicit authorization to consume a reserved resource. Nothing pushed beyond commit `ad33f6c`.
* Review status: single-author agent-assisted; the noise-bound and thread-pin notes are self-audits.

## Provenance addendum — CPU models (2026-09-24; engineering, not scientific; corrects the node comparison in the thread-pin note)

`lscpu` on representative nodes (tiny pinned jobs, `results/stage0_ablation/provenance/lscpu_*.txt`): `ise-cpu-intl-*` = Intel Xeon E5-2680 v2 @ 2.80 GHz (AVX, no AVX2); `ise-cpu128-*` = AMD EPYC 7702P; `ise-cpu256-*` = AMD EPYC 7763. Only these four nodes were measured; the family-to-model mapping for the other nodes is inferred from the node name and was not checked. Both Stage 0 and the ablation ran on a mix of these families (Stage 0: mostly 128-core AMD with some Intel and 256-core nodes; ablation: mostly 256-core AMD plus some 128-core and Intel), so the earlier statement that Stage 0 ran on 128-core nodes and the refits on Intel nodes held only for the single thread-pin fold.
`q_Z` deviation (files with max probability difference > 1e-6, 8k×1 regimes only; 2.5k×1 = 0 of 40) by CPU pair of the Stage 0 task and the refit task: EPYC7702P→EPYC7702P 6/11; EPYC7702P→EPYC7763 49/174; EPYC7702P→Intel 16/19; EPYC7763→EPYC7763 0/9; EPYC7763→other 2/3; Intel→any 24/24 (`provenance/deviation_by_cpu_pair.json`). Every pair involving the Intel CPU deviated, and same-model pairs are not always identical (EPYC7702P→EPYC7702P 6/11), so CPU model is associated with, but does not determine, the deviation; reduction order in multithreaded numerical kernels remains the candidate. Thread pinning alone did not reproduce Stage 0 (previous note). Treated as engineering: worst-case effect on any contrast ≤ 0.010 pp. For exact reproducibility future runs should pin node type and record the CPU model.

## Proposed exposure-ledger entry (2026-09-24; not written to the ledger because the ledger carries a separate task's uncommitted edits)
*"CIFAR-100 test images (same 10,000 IDs), clean plus the 12 development cells; checkpoints 2 and 4; evidence-source ablation (layer probes at 12 depths, the other checkpoint's logits): T-regime labels used for target-supervised fits, S regime clean labels; further development reuse of the exposed cells; not confirmation. Not accessed: checkpoints 1/3/5, the 11 unused CIFAR-100-C families, any new test data."*

