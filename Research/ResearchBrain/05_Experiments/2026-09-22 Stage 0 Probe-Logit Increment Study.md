---
type: experiment
status: completed_development_material_increment_target_supervised
date: 2026-09-22
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C (12 development cells), ResNet-101, checkpoints 2 and 4, corrected_v2_train_norm
preregistered: true
experiment_type: operational recoverability (target-supervised diagnostic)
parent_question: Does a fixed intermediate-layer probe add usable evidence beyond the base logits for a restricted linear readout, and can clean supervision recover it?
exposure_ledger: "[[Representation Correction Exposure Ledger]]"
evidence_scope: unpublished_repository_analysis (development on exposed cells; not confirmation; not a deployable method)
tags: [stage0, target-supervised, recoverability, layer3, probe-logits, stacking]
---

# 2026-09-22 — Stage 0: does adding `layer3.22` probe logits improve a target-fitted linear readout of the base logits?

Observation: [[Target-fitted stacking of logits and layer3.22 probe logits recovers a material increment that clean-fitted stacking on matched budgets does not]]. Hypothesis (proposed): [[H-STAGE0-01 Layer3.22 probe evidence is redundant with the logits on clean data but complementary under corruption, and clean supervision cannot identify the useful combination]]. Project state: [[Current Evidence - Representation-Based Correction Program]]. Follows [[2026-09-21 Fixed Deep Candidate Gate Study]] (closed).

**Which workflow applied (recorded per the task).** The canonical workflow document (`GeometricFullCalibration/docs/research_workflow.md`), the updated `Templates/Experiment.md`, `Templates/Hypothesis.md`, the exposure ledger and the failure-mode note were **uncommitted changes in the working tree** when this card was written (git status 2026-09-23). They were not edited here. The current working-tree template was used, and the existing ledger [[Representation Correction Exposure Ledger]] is linked, not modified.

## Question

Claim scope: ResNet-101 (CIFAR-100, `baseline_cross_entropy`), checkpoints 2 and 4; the 10,000 CIFAR-100 test images, clean plus the 12 development CIFAR-100-C cells (gaussian noise, defocus blur, fog, JPEG × severity 1/3/5); evidence = base logits `Z` and the cached 100-d `layer3.22` GAP linear-probe logits `P` (pre-temperature, stored float16); target = class label; operational decision = top-1 class from a linear multinomial readout fitted with **target-corruption labels** (regime T) or **clean labels** (regime S). This is a target-supervised diagnostic of a restricted readout. It is not an information ceiling and not a clean-only method.

## Pre-execution fields (mapped after results)

The pre-execution fields below are **linked or quoted from documents that predate results** and were transcribed into this card **after results were known**; nothing here was written before execution.
* Frozen specification: repo `docs/stage0_execution_spec.md` (§§0–6 frozen 2026-09-22 before any outer-evaluation metric; §7 appended 2026-09-23 before aggregate output was opened, except as recorded under Deviations). Sidecar `docs/stage0_execution_spec.frozen.sha256` (re-frozen at commit `5862d97`; spec sha256 `30811daf552d…`).
* Design memo (a Claude Doc, not a repo file; updated 2026-09-22): <https://claude.ai/code/artifact/ebf21486-1669-4afb-9748-f624a976e2dd>. Its Section 7 holds the outcome table reproduced below.
* The frozen spec was first written from the execution prompt alone because the memo was not found by filesystem search; the comparison against the real memo is spec §7 ("Declared deviations").

## Competing explanations and predictions (from the memo/spec; see the hypothesis card for the full set)

* Ordinary target recalibration plus classifier fusion → a target-fitted `q_Z` alone already improves on base; adding any comparably informative second predictor would help similarly.
* Complementarity specific to `layer3.22` → `q_ZP` beats `q_Z` under target fitting, and a shuffled-P control does not.
* Non-identifiability from clean data (memo T1) → a clean-fitted readout gains nothing under corruption while the target-fitted one does. It fits equally with clean-fit regularization, a shift in the distribution of `P`, or readout mismatch; Stage 0 cannot attribute a gap.

## Experiment and primary contrast

* Primary type/purpose: operational recoverability.
* Uncertainty tested: readout expressiveness and transfer (finite-sample estimation for the image/view budgets).
* **Primary contrast:** T-8k×12 target-fitted `q_ZP = softmax(A z̃ + B p̃ + b)` versus target-fitted `q_Z = softmax(A z̃ + b)`, 12-cell macro accuracy, evaluated on the 12 corrupted copies of held-out images (pooled out-of-fold), reported per checkpoint. Minimum controls: shuffled-P fit (fold 0), convergence/λ-edge audit, evaluation-only duplicate exclusion.
* Claim class: correctness/decision accuracy of a fitted readout (not replacement-class prediction, not signed intervention utility).
* Secondaries: the other five regimes (T-8k×1, T-2.5k×12, T-2.5k×1, S-8k×1, S-2.5k×1); images-vs-views, recoverability gap Δ_T−Δ_S, clean-view increment.

## Setup (facts)

Data/fitting access: five outer folds (seed 20260922) over 10,000 test images, all 13 condition-copies and pixel-identical duplicate groups in one fold (2 within-test duplicate groups, both label-conflicting); inner 75/25 image-grouped split for λ selection; nested 2,500-image subset shared by S and T; one preassigned corrupted cell per image for the ×1 regimes. **Target-label access:** T regimes fit on labels of the 12 development cells; S regimes on clean-view labels of the same test images; clean-test images are therefore fitting rows for this diagnostic. Checkpoints 2 and 4 only. Checkpoints 1/3/5 and the 11 other CIFAR-100-C families were **not accessed** by Stage 0 (no code path loads them). See [[Representation Correction Exposure Ledger]] for their broader status; they are not called pristine here.

## Fixed choices

Objective mean cross-entropy + λ(‖A‖²_F+‖B‖²_F) (no ½ factor; the memo states no scalar convention, spec §7 Deviation 3), bias unpenalized, λ ∈ {1e-1,…,1e-5}, per-coordinate standardization fitted on training rows only, full-batch L-BFGS in float64 on CPU (no GPU). Cost: 30.6 allocated CPU-hours (4 cores × job elapsed, incl. two ~55-min I/O-slowed tasks); summed fit wall time 5.8 h. Code `atlas/stage0_*.py`; fits ran from `snapshots/stage0_v1_d0dcfd61aa88`.

## Primary metrics / baselines

Accuracy (primary), NLL, Brier (sum over classes, mean over rows). Baselines: base softmax (canonical FP32 atlas logits), target-fitted `q_Z`, clean-fitted `q_Z`/`q_ZP`.

## Outcome-to-decision matrix (memo Section 7, fixed 2026-09-22 before results; transcribed verbatim in spec §7)

| Possible outcome | Explanation supported/weakened | Still unresolved | Next decision |
|---|---|---|---|
| Step 0 fails, or the sanity control gains ≥ +0.2 pp | Protocol or leakage fault | — | Fix the pipeline; compute no contrast until the control passes |
| Primary material (Δ≥+0.5 pp, CI excludes 0, both checkpoints) | A linear readout of the layer3.22 probe logits, fitted with target labels, recovers ≥0.5 pp beyond target-fitted logits on these cells | mechanism; deployability | Read the secondaries; Stage 1 eligible for a separate decision |
| Primary small or uncertain | A small, under-resolved increment for this readout | — | No new allocation; report as is |
| Primary null (upper 95% CI < +0.2 pp, both checkpoints) | This restricted, target-supervised readout adds no material accuracy beyond target-fitted logits | other layers/readouts | The narrow allocation stop |
| Checkpoints fall in different rows | Checkpoint-dependent result | — | Report per checkpoint; "small or uncertain" |
| Recoverability gap Δ_T−Δ_S > 0 at 8k×1, CI excludes 0 | A source-to-target recoverability gap for this evidence/readout; cause unresolved (T1, regularization, distribution shift of P, readout mismatch) | cause of the gap | No further clean-only study of this exact family without a stated mechanism |
| Δ_S ≈ Δ_T, both positive at 8k×1 | Recoverable from clean supervision at ~8,000 images | — | Check S-2.5k×1 first; a new preregistered clean-only study becomes justifiable |
| Δ_T(8k×12) > Δ_T(8k×1) | Multiple corrupted views per image contribute | — | Record; lowers the case for clean-only designs of this family |
| Δ_T(2.5k)≈0 but Δ_T(8k) material, matched views | Needs more independent images than clean budgets provide | — | Record; relevant to future budget choices |

Practical scale: material = Δ ≥ +0.5 pp with the bootstrap interval excluding 0 in both checkpoints; null = upper endpoint < +0.2 pp in both. Intervals are 2,000 paired bootstrap resamples over image groups, conditional on the fitted cross-validation predictions; they do not capture training-sample uncertainty or historical selection.

## Results (facts; artifacts under `results/stage0/report/`, immutable fits `snapshots/stage0_v1_d0dcfd61aa88`)

Labelled by evidence class: **[pre-specified]** in the frozen spec; **[post-hoc]** added afterwards.

**Sanity gates [pre-specified].** Shuffled-P `q_ZP` versus the real `q_Z` on the fold-0 held-out rows: −0.36 pp (seed 2) and −0.80 pp (seed 4) for the memo's gate regime (T-8k×12); all six shuffled fits fall between −0.36 and −1.85 pp, none reaching the +0.2 pp audit trigger. 0 of 120 arm-fits unconverged or retried; selected λ was only 1e-3 (82) or 1e-2 (38); none at a grid edge. Selected λ is the same for S and T at a given image budget (8k: 1e-3 both; 2.5k×1: 1e-2 both, S-2.5k×1 `q_Z` 8/10 at 1e-2).

**Primary [pre-specified], T-8k×12, 12-cell macro accuracy, `q_ZP` − `q_Z`:** seed 2 **+4.23 pp** [3.99, 4.47]; seed 4 **+3.98 pp** [3.75, 4.22]. Verdict by the frozen rule: `material_development_increment`. Evaluation-only exclusion of the 10 test images pixel-identical to a train image changes Δ by <0.002 pp.

**Per cell, T-8k×12 (pp, seed 2 / seed 4).** Gaussian noise s1/s3/s5: +5.57/+10.41/+10.97 and +5.33/+10.47/+10.14. Defocus blur s1/s3/s5: +0.03/+2.84/+7.70 and −0.28/+2.81/+8.05. Fog s1/s3/s5: +0.43/+1.44/+4.64 and +0.61/+0.80/+4.33. JPEG s1/s3/s5: +0.84/+2.47/+3.37 and +0.87/+1.46/+3.13. Severity-1 blur and fog are about 0; the macro mean is carried by noise and severe blur/fog. Per-cell intervals in `stage0c_requested_tables.json` are not multiplicity-adjusted.

**Secondaries [pre-specified], seed 2 / seed 4 (pp, [95% CI], joint resamples).**

| Contrast | Seed 2 | Seed 4 |
|---|---|---|
| Δ_T at 8k×1 | +2.87 [2.63, 3.12] | +2.63 [2.39, 2.86] |
| Δ_S at 8k×1 | −0.48 [−0.65, −0.31] | −0.29 [−0.46, −0.12] |
| Gap Δ_T−Δ_S at 8k×1 | +3.35 [3.11, 3.60] | +2.91 [2.67, 3.17] |
| Δ_T at 2.5k×1 | +1.35 [1.17, 1.54] | +1.20 [1.02, 1.39] |
| Δ_S at 2.5k×1 | −0.14 [−0.27, +0.00] | +0.03 [−0.11, +0.16] |
| Gap at 2.5k×1 | +1.49 [1.30, 1.68] | +1.17 [0.97, 1.37] |
| 8k vs 2.5k images, 12 views | +2.19 [1.94, 2.44] | +2.08 [1.81, 2.32] |
| 8k vs 2.5k images, 1 view | +1.52 [1.28, 1.78] | +1.43 [1.18, 1.67] |
| 12 vs 1 views, 8k images | +1.35 [1.15, 1.56] | +1.35 [1.14, 1.55] |
| 12 vs 1 views, 2.5k images | +0.68 [0.42, 0.92] | +0.70 [0.45, 0.94] |
| Clean-view increment, S-8k×1 (clean images) | +0.18 [−0.25, +0.59] | +0.10 [−0.33, +0.53] |
| Clean-view increment, S-2.5k×1 (clean images) | +0.36 [+0.01, +0.72] | −0.01 [−0.37, +0.37] |

**Stage 0b [pre-specified descriptive].** The clean-trained `layer3.22` probe alone is below base in all 13 conditions in both checkpoints (−3.04 to −8.73 pp; clean 73.4 vs 76.6 and 73.1 vs 76.5). No layer meets the descriptive eligibility flag (≥1 pp over base on ≥2 families in both checkpoints). Layers 4.1/4.2 sit at ≈ base; shallower layers are far worse (figures below).

**Stage 0a [pre-specified descriptive].** For the fixed deep candidate j, the base-logit rank of j among disagreements is distributed like the true-class rank among base errors: rank 2 ≈ 20–21 % vs 21–22 %, ranks 3–5 ≈ 24–25 % vs 24 %, rank > 5 ≈ 54–56 % vs 53–55 % (12 cells pooled). The runner-up frequency of j is therefore not enriched relative to the true class. Fit-row analysis was not available (the fixed-gate fit rows are a clean-only validation split; spec §1). Descriptive; it neither proves nor disproves class-agnostic information.

**Clean cost [pre-specified secondary rows, read post-hoc]: accuracy % / NLL on clean held-out images.** T-8k×12 `q_ZP` 76.02 / 0.895 (seed 2), 76.09 / 0.896 (seed 4); S-8k×1 `q_ZP` 76.76 / 0.872, 76.72 / 0.874. Target-fitting costs ≈0.7 pp clean accuracy and ≈0.02 NLL relative to the clean-fitted model.

**Absolute ladder [post-hoc descriptive], 12-cell macro (accuracy % / NLL / Brier).** From stored out-of-fold predictions and the base logits (`results/stage0/report/stage0_posthoc_accuracy_ladder.{json,md}`, per-cell rows there).

| | Base | T-8k×12 `q_Z` | T-8k×12 `q_ZP` | S-8k×1 `q_Z` | S-8k×1 `q_ZP` |
|---|---|---|---|---|---|
| Seed 2 | 50.09 / 2.415 / 0.706 | 53.17 / 1.929 / 0.592 | 57.39 / 1.648 / 0.545 | 50.04 / 2.451 / 0.686 | 49.56 / 2.496 / 0.699 |
| Seed 4 | 51.01 / 2.281 / 0.675 | 54.18 / 1.887 / 0.583 | 58.16 / 1.620 / 0.538 | 50.97 / 2.331 / 0.658 | 50.68 / 2.343 / 0.663 |

Reading (post-hoc, order-dependent decomposition): target recalibration of the logits alone (`q_Z` vs base) is +3.08 / +3.17 pp; adding the probe is a further +4.23 / +3.98 pp; the total over base is +7.30 / +7.15 pp, i.e. roughly 42–44 % recalibration and 56–58 % added evidence. The clean-fitted models do not improve accuracy over base under corruption and have worse NLL than base, while improving clean NLL.

**Post-hoc scalar blend [post-hoc; not in the frozen protocol; live tree, code `atlas/stage0_alpha_beta.py`, commit `30e8751`].** `softmax(α z + β p + b)` on raw `z` and `p`, scalar α, β, free 100-vector b, no penalty, per fold; 40 fits, all stopped on the optimizer's own tolerance. Recovers 26 % / 29 % of the frozen Δ_T (target labels; gain +1.11 / +1.16 pp over the α-only comparator, α ≈ 0.49, β ≈ 0.37) and 8 % / 11 % with clean labels (+0.32 / +0.44 pp; α ≈ 0.50–0.51, β ≈ 0.47). Raw-scale caveat: `z` and `p` scales were not equalized, so only the α-versus-β comparison is affected; the clean-versus-target β comparison is within the same parameterization.

Figures (existing JSON only): `results/stage0/report/figures/0a_flip_decomposition.png`, `0a_rank_distributions.png`, `0b_layer_probe_heatmap.png`, `0b_layer_depth_curves.png` (script `atlas/stage0_ladder.py`, post-hoc descriptive).

## Protocol deviations (full list in spec §7; provenance record in `results/stage0/report/aggregate_provenance.json`)

1. **Spec frozen without the real memo.** §§0–6 came from the execution prompt because the memo (a Claude Doc) was not found on the filesystem; the line-by-line comparison was made after fitting and before aggregate output. The memo's sanity control is 2 real fits (T-8k×12 only, fold 0, per checkpoint); 6 were run (three regimes × two checkpoints). The T-8k×12 pair is the memo-specified gate; the others are additional.
2. **Derived-gap intervals added after the first aggregate output was seen**, to conform to frozen §5 (joint resamples). The first output showed point estimates only. Primary Δ_T, controls and verdict did not change.
3. **Penalty scaling** (no ½ factor) follows the execution prompt; the memo is silent. It roughly doubles the effective penalty relative to a halved convention; the λ-edge audit found no edge selections.
4. **Aggregation and verification jobs were cancelled and run by hand.** Slurm aggregate/verify jobs 21599569/21599570 were cancelled before starting; aggregation ran on the live tree and was rerun from `snapshots/stage0_v3_ec359efbe2cc` (git head `5862d97`), output **byte-identical** (sha256 `318453572e1d…4a350a359`). The 9 structural tests were run manually on the live tree.
5. **Repository incident.** Commit `b55dd90` (Stage 0 code, §§0–6, snapshot `stage0_v1`) was pushed to `origin/main` by an agent-spawned fork without authorization. `5862d97` (spec §7, aggregator, loader) and `30e8751` (post-hoc scripts) are local commits; history was not rewritten.
6. **One fold file was rewritten after the Slurm run.** `results/stage0/seed4/S-2.5k1/fold0.npz` has an mtime (2026-09-23 00:52) later than its job's end (2026-09-22 23:48); the same fork reported re-running it as a determinism check (its report of 55/56 identical arrays is unverified, and the original file no longer exists to compare). The code state of that rerun is unknown. The file feeds seed 4's S-2.5k×1 rows; confirming it needs an engineering rerun from the snapshot, which this documentation task did not authorize.
7. **Two slow tasks.** `c_s4_S-8k1` folds 0/1 ran ≈55 min (disk-wait on a shared node, low CPU utilization; not a code fault); no numerical effect.
8. Post-hoc scripts (`stage0_report.py`, `stage0_alpha_beta.py`, `stage0_ladder.py`) ran from the live tree, not a snapshot.

## Post-mortem and allocation

* **Primary contrast, practical magnitude, uncertainty:** +4.23 / +3.98 pp target-fitted `q_ZP` over target-fitted `q_Z` (12-cell macro), intervals ≈ ±0.24 pp conditional on the fitted CV predictions; about 8× the +0.5 pp practical bar. Concentrated in noise and severe blur/fog cells; ≈0 at low-severity blur/fog.
* **Explanations more/less plausible:** more plausible — target-supervised linear stacking of `Z` and `P` extracts information that the target-fitted `Z` readout does not (shuffled-P control passed; convergence clean); ordinary target recalibration explains a large part of the gain over base (42–44 %). Less plausible — "the intermediate probe is simply more robust" (0b: the probe alone is worse than base everywhere); "one scalar trust weight closes the gap" (a scalar blend fitted with target labels recovers only 26–29 %; clean and target fits also select similar β, differing modestly). **Still indistinguishable:** non-identifiability from clean data (T1) versus clean-fit regularization (selected λ was the same for S and T at a given budget, which argues against a simple λ story but does not remove it), a shift in the distribution of `P` relative to `Z`, readout mismatch, and generic "any second predictor would fuse similarly" (no matched second-predictor control was run; the shuffled-P control does not test it).
* **Reasoning-chain stage tested:** readout expressiveness / transfer of a fixed evidence source under a restricted linear readout, with target labels. It does not test detection, abstention, kNN candidates, other layers, or deployment.
* **Claim supported:** on these 12 exposed cells, two checkpoints and 10,000 shared images, a target-fitted linear readout of base logits plus `layer3.22` probe logits beats the target-fitted logit-only readout by ≈4 pp macro accuracy, and clean-fitted stacking at matched budgets does not transfer (Δ_S ≤ 0.03 pp, mostly negative). **Not supported:** any clean-only or deployable method; a cause for the recoverability gap; specificity to `layer3.22` (versus other layers or other second predictors); generalization to other corruptions, severities, checkpoints, or architectures; an information ceiling.
* **Allocation (memo Section 7 rows that apply):** the recoverability-gap row applies → **no further clean-only study of this evidence family without a stated mechanism for closing the gap.** Stage 1 is eligible but **not started** (needs its own decision and authorization). Stage 0d (detection/abstention) not started. The memo's Direction 2 (label-free choice of readout depth; conditional on a large oracle layer gain in 0b) is not triggered: no layer met the 0b flag. Separately, a scalar trust-weight explanation is weakened by the post-hoc blend result. Reopening evidence: a stated mechanism that a clean-fitted design can exploit, or authorized new evidence on a reserved resource.
* **Review status/independence:** single-author agent-assisted analysis; the memo comparison and this card were produced by the same agent lineage and have not had independent review. One agent-spawned subagent reported results that did not match repository state (commit/push, fold rewrite); its self-report was not relied upon and every claim used here was re-checked against files or Slurm.

## Amendments / engineering recovery

* 2026-09-23: spec §7 appended and re-frozen; derived-gap CIs added; aggregator refactors (control-first print, convergence/λ-edge audit, evaluation-only exclusion sensitivity).
* 2026-09-23: this card written after results from frozen spec, memo, and `results/stage0/report/*`.
