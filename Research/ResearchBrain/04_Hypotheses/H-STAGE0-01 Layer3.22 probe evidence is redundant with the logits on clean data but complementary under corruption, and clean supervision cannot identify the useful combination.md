---
type: hypothesis
status: proposed
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C, ResNet-101, checkpoints 2 and 4 (development)
novelty: unknown
tags: [H-STAGE0, recoverability, identifiability, probe-logits, layer3.22, proposed]
---

# H-STAGE0-01 Layer3.22 probe evidence is redundant with the logits on clean data but complementary under corruption, and clean supervision cannot identify the useful combination

**Status: proposed, created at the researcher's request. Nothing is established.** Experiment: [[2026-09-22 Stage 0 Probe-Logit Increment Study]]; observation: [[Target-fitted stacking of logits and layer3.22 probe logits recovers a material increment that clean-fitted stacking on matched budgets does not]].

## Formal statement

Claim scope: ResNet-101 (CIFAR-100), fixed `layer3.22` GAP probe logits `P` and base logits `Z`, linear multinomial readouts, CIFAR-100-C corruptions like the 12 development cells, top-1 accuracy. Statement: (i) with clean supervision at matched image/view budgets, adding `P` to `Z` gives no accuracy gain under corruption (approximately the clean-view increment), while (ii) with target-corruption supervision it gives a material gain, and (iii) the difference is because the useful combination of `Z` and `P` under corruption is not identifiable from clean data. Stage 0 observed (i) and (ii) on development cells; it did not test (iii).

## Why it follows from evidence

Stage 0: target-fitted +4.23 / +3.98 pp (T-8k×12); clean-fitted Δ_S −0.48 / −0.29 pp (8k×1), gap +3.35 / +2.91 pp; clean-view increment ≈ +0.1 to +0.2 pp (interval includes 0); the probe alone is below base in all 13 conditions. The memo's T1 corollary ([[Theory Plan - Decision Utility, Layers, Compression and Risk Control]], Memo-T1) says clean data carry no population information about the hidden part of a gate when the source insufficiency ε_S is 0; it is not offered as the explanation of any result.

## Competing explanation

Each with a prediction that would separate it from the hypothesis:

1. **Non-identifiability from clean data (T1) — the hypothesis.** Predicts: clean-fitted gain stays ≈ 0 as clean budget grows and under any regularization; a target-fitted gain grows with the number of independent target images (Stage 0: +2.1 pp from 2.5k→8k images).
2. **Clean-fit regularization.** Predicts: the clean gap shrinks when clean-fit λ (or a different penalty/standardization) is varied more widely than the fixed grid. Stage 0 evidence: selected λ was the same for S and T at a given budget, which counts against a simple version, but the grid was fixed and coarse.
3. **Shift in the distribution of P relative to Z.** Predicts: an unlabelled per-condition standardization of `P` and `Z` (no labels) closes part of the gap; the gap tracks severity and family (Stage 0: concentrated in noise and severe blur/fog). Requires an unlabelled target batch, so it is a different access regime and would be reported separately.
4. **Readout mismatch.** Predicts: a different readout family (nonlinear, or a different combination of `Z` and `P`) fitted on clean data recovers part of the gain; a linear clean fit does not.
5. **Ordinary target recalibration plus classifier fusion — the strongest mundane explanation.** Predicts: a target-fitted `q_Z` already captures a large share (observed 42–44 % of the gain over base, post-hoc), and replacing `P` by another comparably informative predictor (another layer's probe, an independently trained head) gives a comparable target-fitted increment. It also predicts the gain is not specific to `layer3.22`. No such control was run; the shuffled-`P` control does not test it.

Prediction overlap / non-distinguishability: explanations 1–4 all predict "target gain, clean null" and are not separated by Stage 0. Explanation 5 differs from 1–4 only through a matched second-predictor control. The post-hoc scalar-blend result (a single α, β weight recovers 26–29 % with target labels, 8–11 % with clean labels) weakens "one scalar trust weight" as the mechanism but does not separate 1–4.

## Benchmark

Same fixed inputs and folds as Stage 0 for any development follow-up; any confirmation needs a reserved resource and explicit authorization (see [[Representation Correction Exposure Ledger]]). Checkpoints 1/3/5 and the 11 unused families are not pristine by default and were not accessed by Stage 0.

## Baselines

Base softmax; target-fitted `q_Z` (recalibration alone); clean-fitted `q_Z`; matched second-predictor `q_ZQ` for explanation 5; noise-matched or shuffled `P`.

## Decisive experiment

Primary type: mechanism discrimination (operational recoverability with matched controls). Primary contrast and minimum controls: clean-fitted versus target-fitted increment of adding `P` to `Z` at matched budgets, together with a matched second-predictor arm and a wider clean-λ path. Uncertainty tested: readout expressiveness, transfer, finite-sample estimation. Exposure/access: fit and selection roles declared before evaluation; target-label diagnostics declared separately from clean-only fitting; a confirmation would consume a reserved resource only with explicit approval. **No such experiment is authorized by this card.**

## Outcome-to-decision matrix

| Possible outcome | Explanation supported/weakened | Still unresolved | Next decision |
|---|---|---|---|
| A matched second predictor gives a target-fitted increment comparable to `P` | Explanation 5 (generic fusion) supported; specificity to `layer3.22` weakened | whether clean data could identify any of them | stop attributing the gain to `layer3.22`; record |
| A widened clean-λ or unlabelled-standardization arm recovers a large part of the gap (see criteria) | Explanations 2 or 3 supported; T1 weakened for this readout | true identifiability limit | redesign a clean-only contrast around the recovering mechanism |
| Clean-fitted gain stays ≈ 0 across the added controls while target-fitted gain persists | T1-type non-identifiability more plausible for this readout; not proven | which of 1–4 | no further clean-only study of this family without a stated mechanism (already in force) |
| Target-fitted gain does not reproduce on the confirmation resource | Stage 0 gain was development-specific | — | downgrade the observation |

## Kill criterion

Kill (for a future confirmation): (a) a clean-fitted readout with matched budget and a widened λ path recovers ≥ 50 % of the target-fitted increment, or (b) a matched second predictor gives ≥ 50 % of the `P` increment, or (c) the target-fitted increment on the authorized reserved resource has an upper 95 % endpoint < +0.2 pp.

## Go criterion

Go (for a future confirmation of the specific claim): on the authorized reserved resource, target-fitted increment ≥ +0.5 pp with its interval excluding 0 in each checkpoint used, clean-fitted increment ≤ +0.2 pp (upper endpoint below the gate) at matched budgets, and a matched second predictor recovering < 50 % of the `P` increment. Practical thresholds are task-specific; there is no universal CI rule, and intervals remain conditional on the fitted CV predictions.

## Prior-art threats

Target-label stacking of intermediate layers is known (the memo cites Uselis et al. 2025 and Lee et al. 2023 for target-label re-reading of intermediate layers); last-layer retraining with extra inputs is the ordinary reading of a target-fit win. Novelty is unknown and is not claimed.

## Update 2026-09-24 (evidence ablation; appended, nothing above rewritten)

Source: [[2026-09-24 Stage 0 Evidence Ablation]] (ResNet-101, checkpoints 2 and 4, 12 development cells; development evidence).
* **The generic-fusion competitor (explanation 5's "any comparably informative second predictor gives the same pattern") is falsified in this setting.** The one non-same-network source tested, `Z_other` (the other checkpoint's logits), is source-recoverable: Δ_S ≈ Δ_T (+3.46 / +2.53 vs +3.55 / +2.51 pp at 8k×1) with a clean increment of +2.4 pp, whereas every same-network layer probe has Δ_S ≤ about +0.3 pp.
* **Conditional redundancy survives, but on one contrasting source.** The statement that probe evidence is redundant with the logits on clean data and complementary under corruption is not contradicted; the contrast against it is a single source (`Z_other`).
* **Confound stays open: "same network" and "clean-redundant" are fully confounded** in the current data (all same-network probes are clean-redundant; the only cross-network source is clean-complementary). This is what the pretrained-model regime map (Phase 2, frozen, not started) is designed to separate.
* **Wording.** Use "stage-specific (layer3.x plateau)", not "layer3.22-specific": Δ_T is ≈ +2.6 to +2.9 pp across layer3.7–layer3.22 and ≈ 0 at layer4.1. `P_4.2` reads the same penultimate features as `Z`, so the depth claim rests mainly on layer4.0/4.1 ≈ 0 (layer4.0 ≈ +0.7 pp) and on the layer3 plateau, not on the primary contrast D_A alone.
* **Status:** proposed; nothing established.

## Caveat added 2026-09-24 (recipe confound on the deep-layer leg; appended)
The "layer4.x ≈ 0" leg of the depth claim (layer4.0 Δ_T ≈ +0.7, layer4.1 ≈ 0, layer4.2 < 0) was measured with layer-pilot probes whose selected λ = 0.01 sat at the **upper edge of the λ grid** {1e-4, 1e-3, 1e-2}, with inner-fit NLL still falling at the edge (Phase 1B card). A possible recipe confound: better-tuned layer4 probes could differ from what was measured. The contrast against the same-network mid-depth plateau and the `Z_other` contrast are unaffected by this (the `Z_other` result does not use these probes). The regime-map pilot (spec v2) applies a grid-edge extension rule to every probe.

## Update 2026-09-26 (regime-map pilot verdict; appended, nothing above rewritten)
Source: [[2026-09-24 Regime-Map Pilot]], frozen spec v2 (`docs/regime_map_pilot_spec_v2.md`), artifact `results/regime_map/report/regime_aggregate.json` (keys `rules`, `rules_by_seed`, `states.<s>.gap_8k1`, `states.<s>.S_clean_increment_8k1`, `gap_b10_minus_gap_a_seed{1,2}`).
* **Verdict: row 4 fired in both fine-tuning seeds** (`rules.row` = 4, `rules_by_seed` = {'seed1': 4, 'seed2': 4}). **"Clean redundancy controls recoverability" is killed for this evidence** (`layer3` GAP probe logits on an ImageNet-pretrained ResNet-50 with clean redundancy manipulated by fine-tuning dose, CIFAR-100, the 12 development cells).
* Numbers behind the row (8k×1, 12-cell macro, pp, 95% image-bootstrap interval): clean increment of state (a) +3.82 [+3.20, +4.43]; b10 clean increment -0.20 [-0.66, +0.24] (seed 1), -0.74 [-1.19, -0.31] (seed 2); gap(a) +2.77 [+2.43, +3.09]; gap(b10) +1.53 [+1.26, +1.78] (seed 1), +2.04 [+1.79, +2.28] (seed 2); gap(b10) − gap(a) -1.24 [-1.63, -0.86] (seed 1), -0.73 [-1.12, -0.33] (seed 2).
* **Does not establish:** (i) **Δ_T uses target labels**: it is an oracle ceiling of a restricted linear readout, not a deployable quantity, and the fits use exposed development cells; (ii) **the root cause of the recoverability gap is unresolved**: clean non-identifiability, source-fit regularization, shift of the probe evidence under corruption, readout mismatch and target-adaptation capacity are not separated; (iii) any claim for other backbones, resolutions, probe layers or corruptions; (iv) that clean redundancy is irrelevant in general, only that, as manipulated here, it does not control the gap. Two fine-tuning seeds; state (a) is a single fit; intervals are image-bootstrap only.

## Correction 2026-09-26 — increments are relative to the 8k refit, not to the model's own output (appended; nothing above rewritten)
The clean increment, Δ_S, Δ_T and gap reported above are all differences between two **8k-row refits** (Z+H minus Z-only, same protocol). The Z-only refit is itself not the model's output: its accuracy differs from the base head by a state-dependent amount. Per-state (refit − base) and (Z+H − base), in pp, **clean / macro-12**, read from `results/regime_map/report/regime_aggregate.json` (`states.<s>.standalone.acc_Z_clean`, `acc_Z_macro12`; `states.<s>.regimes.<S-8k1|T-8k1>.q_Z_clean_acc`, `q_Z_macro12_acc`, `q_ZP_clean_acc`, `q_ZP_macro12_acc`):

| state | base Z (clean / macro-12, %) | S-8k×1 Z-only refit − base | S-8k×1 Z+H − base | T-8k×1 Z-only refit − base | T-8k×1 Z+H − base |
|---|---|---|---|---|---|
| a | 72.35 / 48.76 | -4.22 / -3.38 | -0.40 / -1.36 | -7.47 / +0.62 | -4.08 / +5.41 |
| b1_s1 | 65.36 / 39.56 | +8.18 / +5.40 | +9.83 / +6.86 | +5.85 / +11.88 | +6.66 / +15.07 |
| b3_s1 | 77.60 / 46.99 | +1.56 / +1.29 | +1.40 / +1.46 | -0.79 / +8.34 | -1.26 / +10.15 |
| b10_s1 | 84.25 / 53.68 | -1.29 / -1.15 | -1.49 / -1.86 | -3.29 / +5.15 | -4.90 / +5.96 |
| b1_s2 | 68.97 / 43.47 | +5.03 / +2.79 | +6.08 / +3.03 | +2.91 / +9.58 | +3.30 / +12.03 |
| b3_s2 | 74.92 / 45.04 | +3.67 / +3.15 | +3.46 / +2.64 | +1.35 / +10.20 | +0.84 / +12.23 |
| b10_s2 | 84.58 / 53.30 | -0.96 / -1.16 | -1.70 / -2.18 | -3.38 / +5.84 | -4.98 / +6.86 |

* **The row-4 verdict is unchanged** (it is a statement about the frozen rule applied to the reported differences).
* **The statement that the frozen backbone has "+3.8 pp clean complementary information" is not supported relative to the base output**: the +3.82 pp is Z+H minus the S-8k×1 Z-only refit, and that refit is below the base head on clean (see the table); relative to the base output the source-fitted Z+H is below it on clean for state (a).
* Nothing else in this note is reinterpreted by this correction.

