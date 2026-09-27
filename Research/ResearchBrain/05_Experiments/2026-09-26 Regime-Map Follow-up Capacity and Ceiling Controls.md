---
type: experiment
status: closed_inconclusive_stop
date: 2026-09-26
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C (12 development cells), ImageNet-pretrained ResNet-50 (seven states: a; b1/b3/b10 x two fine-tuning seeds)
preregistered: true
experiment_type: capacity and second-classifier controls (target-supervised development diagnostic, ORACLE)
parent_question: Is the target-fitted increment of the layer3 probe logits P over Z specific to intermediate-layer information, or is it a capacity, second-classifier or head-discard effect?
exposure_ledger: "[[Representation Correction Exposure Ledger]]"
evidence_scope: unpublished_repository_analysis (development; not confirmation; not a deployable method)
tags: [regime-map, follow-up, anchored-readout, oracle-diagnostic, inconclusive]
---

# 2026-09-26 — Regime-map follow-up: capacity, second-classifier and ceiling controls

Parent: [[2026-09-24 Regime-Map Pilot]]. Hypothesis: [[H-STAGE0-01 Layer3.22 probe evidence is redundant with the logits on clean data but complementary under corruption, and clean supervision cannot identify the useful combination]].

**Pre-declared outcome (`decision` in `results/regime_map_followup/report/stage2_anchored.json`): INCONCLUSIVE — stop.** Stage 3 (label budget) was not run: the stop covers it (its purpose was sizing a phenomenon for continuation; at stop it cannot change anything). No further runs on this line.

## Frozen specification
`docs/regime_map_followup_spec_v1.md` (sha256 `9214ab74…a6986`), superseded in its decision table and comparators by `docs/regime_map_followup_amendment_1.md` (sha256 `31ee68be…6d2f`); both unedited, sidecars verified. Every Δ_T below is an **ORACLE target-label diagnostic** (fits use the labels of the 12 exposed development cells).
Readouts are anchored: q = softmax(z + f(x)), λ path {1e-1…1e-5} plus f = 0. Fits: T-8k×1, 5 folds, 12-cell macro, 2000-resample paired grouped bootstrap (one shared index array per state). Decision states: b10_s1 and b10_s2 only (must agree).

## Gates and decision (applied to b10_s1, b10_s2)
* Gate 1 (D < 0.5 pp, or C1b ≥ 50% of D, in either seed): not triggered (D = +0.79, +0.84; C1b fraction −0.469, −0.463).
* Gate 2 (C1e ≥ 50% of D in both seeds): not triggered (−0.411, −0.380).
* HEAD-DISCARD: not met (Δ_T(C1d) = +0.65, +0.80 < D).
* INTERMEDIATE-SPECIFIC: not met (D − Δ_T(C1d) intervals [−0.06, +0.32] and [−0.15, +0.24] include 0).
* Anything else → INCONCLUSIVE, stop.

## Stage 2 table (anchored, Δ_T over anchored Z-only, pp, 95% intervals)
| state | D = Δ_T(Z,P) | C1b frac | C1e (Z,P_L) Δ_T / frac of D | C1c (h_L) | C1d (Z,h_L) | K | Z+K | D − Δ_T(C1d) |
|---|---|---|---|---|---|---|---|---|
| a | +2.08 [+1.93, +2.22] | 0.129 | +0.30 / 0.146 | +0.84 | +0.89 | +1.29 | +1.35 | +1.19 [+1.02, +1.38] |
| b1_s1 | +2.46 [+2.22, +2.69] | 0.039 | +0.57 / 0.233 | +1.96 | +1.95 | +2.02 | +2.07 | +0.51 [+0.27, +0.75] |
| b3_s1 | +1.50 [+1.29, +1.72] | 0.075 | +0.13 / 0.088 | +1.48 | +1.45 | +1.69 | +1.73 | +0.05 [−0.16, +0.26] |
| **b10_s1** | **+0.79 [+0.62, +0.96]** | −0.469 | **−0.32 [−0.45, −0.20] / −0.411** | +0.69 | **+0.65 [+0.46, +0.84]** | +0.91 | **+0.97 [+0.76, +1.21]** | **+0.14 [−0.06, +0.32]** |
| b1_s2 | +1.83 [+1.60, +2.07] | −0.005 | −0.01 / −0.007 | +1.74 | +1.74 | +1.82 | +1.91 | +0.09 [−0.14, +0.33] |
| b3_s2 | +1.59 [+1.39, +1.79] | 0.011 | −0.19 / −0.118 | +1.56 | +1.54 | +1.59 | +1.70 | +0.05 [−0.17, +0.25] |
| **b10_s2** | **+0.84 [+0.66, +1.02]** | −0.463 | **−0.32 [−0.44, −0.19] / −0.380** | +0.85 | **+0.80 [+0.62, +0.97]** | +1.01 | **+1.08 [+0.86, +1.29]** | **+0.04 [−0.15, +0.24]** |

Intervals for the remaining cells are in the artifacts. `P_row` sanity (reported, not gated): macro accuracy differs from anchored Z-only by −0.74 to +0.56 pp; mean |Δprob| 0.0014–0.0022.

## Artifacts
* Stage 1: `results/regime_map_followup/report/stage1_anchored.json`, `stage1_anchored_table.md`; per-fold `results/regime_map_followup/stage1/<state>/fold{f}.npz`.
* Stage 2: `results/regime_map_followup/report/stage2_anchored.json`, `stage2_anchored_table.md`; per-fold `results/regime_map_followup/stage2/`; features/heads/probes `results/regime_map_followup/hL/<state>/`.
* Results record: `docs/regime_map_followup_results_stage1_2.md`. Code: `atlas/followup_*.py`; snapshots `followup_s1_ea2dfacff7fe`, `followup_s2_be595d5404ce`.
* Validation: consistency checks passed ((a) head refit reproduced z exactly, b* max|Δz| ≤ 1.8e-5 vs 1e-2); 0 unconverged/retried fits; anchored Z-only reproduced Stage 1 (1 differing prediction in b1_s2, 0 elsewhere).

## Descriptive, not decision-bearing
1. At b10, C1b and C1e are negative, and C1c, C1d and K recover increments of similar size to D; the D − Δ_T(C1d) intervals include 0.
2. *(Noticed after the decision.)* In state (a) (single fit, descriptive only), D − Δ_T(C1d) = +1.19 [+1.02, +1.38], falling to about 0 by b3 in both seeds.
3. *(Noticed after the decision.)* The P_row sanity check shows parameterization alone moves macro accuracy by up to 0.74 pp, so between-arm differences below that size (e.g. Z+K vs C1d) are not interpretable.

Items 2 and 3 authorize no follow-up.

## Does not establish
* Every Δ_T is an **oracle target-label diagnostic**, not a deployable quantity and not an information ceiling.
* **Absence of information is not shown**: INCONCLUSIVE means the pre-declared rule did not separate the explanations, not that the layer3 evidence is redundant or specific.
* Nothing about other backbones, resolutions, probe layers, corruptions, or a cause of the recoverability gap.
* Exposure: same 12 development cells and labels as the parent (see the ledger; not edited here).
