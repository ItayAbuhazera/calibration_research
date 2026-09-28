---
type: experiment
status: completed_development_outcome_C_stop
date: 2026-09-28
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C (12 development cells), ResNet-101, checkpoints 2 and 4, corrected_v2_train_norm
preregistered: true
experiment_type: mechanism discrimination (target-supervised development diagnostic, ORACLE label access)
parent_question: Is the Stage-0 predictive increment evidence specific to the intermediate representation, or already available from the penultimate state H_L?
exposure_ledger: "[[Representation Correction Exposure Ledger]]"
tags: [G1, conditional-accessibility, penultimate, stage0, gatekeeper]
---

# 2026-09-28 — G1: conditional accessibility gatekeeper

Hypothesis: [[H-ACCESS-01 Mid-depth evidence adds target-fitted value beyond (Z, H_L) in end-to-end ResNet-101 under corruption]].
Parents: [[2026-09-22 Stage 0 Probe-Logit Increment Study]], [[2026-09-24 Stage 0 Evidence Ablation]], [[2026-09-26 Regime-Map Follow-up Capacity and Ceiling Controls]].
**Frozen specification (authoritative):** `GeometricFullCalibration/docs/g1_conditional_access_spec.md`, sha256 `b860895f2256…49bd77`, committed `4f7523f` before any G1 code or outcome.
Code commit `c56c484`; immutable snapshot `snapshots/g1_5ea1dcbb3e37`.

## Question

Claim scope: ResNet-101 checkpoints 2 and 4; the 10,000 test images × clean + 12 exposed cells; evidence Z, H_L (layer4 GAP 2048-d,
new extraction), P_3.22, P_4.2, Z_other; target = class label; decision = top-1 of an anchored linear readout fitted with target labels.
**Target-supervised diagnostic; not deployable; not an information ceiling.**

## Competing explanations and predictions

- E1 head bottleneck → C recovers B's gain; D ≈ C.
- E2 depth-specific operational accessibility → D − C ≥ +0.5 pp, beyond E and F controls.
- E3 sample efficiency / compression → D − C shrinks from 2.5k to 8k while C catches up with B.
- E4 generic diverse predictor → F − C comparable to D − C.
- E5 parameterization → |G − C| comparable to the effect.
- E6 target recalibration → cancels in D − C (all arms target-fit on the same rows).

## Experiment and primary contrast

- Primary type: mechanism discrimination. Uncertainty tested: readout expressiveness / finite-sample estimation.
- **Primary contrast:** Δ_cond = Acc_T(Z, H_L, P_3.22) − Acc_T(Z, H_L), T-8k×1, 12-cell macro, per checkpoint.
- Controls: E = (Z, H_L, P_4.2); F = (Z, H_L, Z_other); G = (Z, P_ker H_L) equivalent span; D-shuf (fold 0); budget T-2.5k×1 for A–D.
- Claim class: correctness of a fitted readout.

## Setup

Stage-0 fold plan, inner 75/25 image-grouped λ selection, preassigned corrupted cell; anchored fitter from the regime-map follow-up
(unchanged) plus the frozen grid-edge rule (λ larger = stronger ridge). Label access: T-labels of the 12 exposed cells; clean reported
descriptively. Not accessed: 11 reserved families, checkpoints 1/3/5.

## Outcome-to-decision matrix

| Possible outcome | Explanation supported/weakened | Still unresolved | Next decision |
|---|---|---|---|
| A material conditional accessibility | E2 supported; E1/E3/E4 weakened at this budget | robustness, source access, mechanism | robustness stage *eligible*, not started |
| B penultimate sufficiency | E1 supported; E2 weakened | why the head discards target-useful directions | stop the intermediate-specific branch |
| C sample-efficiency / capacity | E3/E5 supported | asymptotic sufficiency | stop |
| D generic diversity | "complementary predictive view", not depth | same- vs cross-network views | stop the depth framing |
| E mixed / inconclusive | none | checkpoint dependence | stop |
| F invalid | none | engineering / regularization | one engineering recovery, else stop |

Practical scale: material +0.5 pp (CI excludes 0, above the equivalent-span band), negligible = upper 95% bound < +0.2 pp (Stage-0 conventions).

## Results (2026-09-28; artifacts `GeometricFullCalibration/results/g1/report/g1_aggregate.json`, `g1_table.md`; fits `results/g1/fits/seed{2,4}/<regime>/<arm>/fold*.npz`; extraction `results/g1/hL/seed{2,4}/`)

**Validity [pre-specified].** Extraction gate passed: max |z_atlas − (H_L W_effᵀ + b_eff)| = 2.3e-6 / 2.1e-6 (tolerance 1e-2), model forward = atlas logits exactly, temp = 1.0, argmax agreement 1.0 (`results/g1/hL/seed*/consistency.json`). 104 fit tasks + 2 extraction + aggregation completed; every final fit converged, 0 retries. Every H_L-containing arm selected λ = 0.1 at 8k (λ = 1.0 at 2.5k) — the strong (heavier-ridge) end — and the frozen extension one decade stronger was worse, so **0 unresolved edges**. D-shuf − C (fold 0) = +0.15 / +0.17 pp (< +0.2: V4 passes).

**Frozen decision: Outcome C — SAMPLE-EFFICIENCY / CAPACITY CONFOUND** (labels `cap`/`cap`), triggered by the nesting check.

| quantity (pp; 12-cell macro; 95% image-group bootstrap) | checkpoint 2 | checkpoint 4 |
|---|---|---|
| **Δ_cond = D − C (8k, primary)** | **+0.24 [+0.20, +0.28]** | **+0.24 [+0.20, +0.29]** |
| Δ_cond (2.5k) | +0.03 [+0.02, +0.05] | +0.04 [+0.02, +0.06] |
| D − E (vs recipe-matched P_4.2) | +0.21 [+0.18, +0.26] | +0.22 [+0.18, +0.26] |
| F − C (Z_other conditional) | +0.51 [+0.45, +0.56] | +0.36 [+0.31, +0.41] |
| D − F | −0.27 [−0.32, −0.22] | −0.12 [−0.17, −0.07] |
| G − C (equivalent span) | +0.72 [+0.61, +0.83] | +0.50 [+0.39, +0.61] |
| **D − B (nesting)** | **−1.38 [−1.53, −1.24]** | **−1.16 [−1.30, −1.03]** |
| B − A (anchored Stage-0 analogue, 8k) | +1.88 [+1.73, +2.05] | +1.80 [+1.65, +1.95] |
| C − A (8k) | +0.27 [+0.19, +0.34] | +0.39 [+0.31, +0.47] |
| (C−B)8k − (C−B)2.5k | −0.23 [−0.41, −0.04] | −0.13 [−0.31, +0.05] |

Absolute 12-cell macro accuracy (%), 8k: A 52.24 / 53.25; B 54.12 / 55.04; C 52.51 / 53.63; D 52.74 / 53.88; E 52.53 / 53.66; F 53.01 / 54.00; G 53.23 / 54.14.
**Post-hoc descriptive (fold 0 only; not decision-bearing):** D − C +0.17 / +0.22, E − C +0.00 / +0.03, D-shuf − C +0.15 / +0.17 pp — real P's conditional increment is close to that of 100 shuffled columns on this fold.
Cost: 32.5 four-core task-hours of fitting (measured) + ≈ 2.5 GPU-minutes extraction.

## Protocol deviations

None. The grid-edge rule was exercised (all H_L arms) and resolved in every fold. Test suite: 9/9 new G1 tests pass; the full suite has 26 failures + 5 errors that are identical without the G1 files (pre-existing, unrelated).

## Post-mortem and allocation

- **Primary contrast:** Δ_cond = +0.24 pp in both checkpoints, intervals [+0.20, +0.29] — above the +0.2 negligible bound, below the +0.5 material scale, and **smaller than the equivalent-span parameterization band** (|G − C| = 0.72 / 0.50). Not a material conditional increment.
- **What decided the outcome:** the joint readout (Z, H_L, P) is 1.2–1.4 pp *worse* than (Z, P). With 2,248 standardized inputs, 8k target rows and one shared ridge λ (selected at the strong end), the readout cannot use P as well as the 200-d readout does. Full-rank H_L itself adds only +0.27 / +0.39 over Z, versus +1.88 / +1.80 for the 100-d P.
- **Explanations:** E1 (head bottleneck: a readout on H_L recovers Stage 0) **not supported at this budget** — the H_L readout recovers ≈ 15–20 % of B − A. E2 (depth-specific increment beyond H_L) **not supported** at the material scale and within the parameterization band; its fold-0 size is similar to shuffled-P. E3 in its compact-summary sense is **consistent** (high-dimensional readouts are estimation-limited), but its *budget* signature was not observed (Δ_cond grew from +0.03 to +0.24 with more labels; C did not catch up). E4: Z_other adds more after H_L than P does. E5 is **large**: equivalent spans differ by 0.5–0.7 pp. E6 cancels.
- **Still indistinguishable:** whether H_L contains the Stage-0 information in a form a better-regularized or larger-budget readout could use, vs. the information being more accessible (linearly, at 8k labels) through the compact mid-depth probe. Two budgets cannot settle asymptotic sufficiency.
- **Claim supported (development, target-supervised):** at 8k target labels with the anchored single-λ linear readout, the Stage-0 increment is carried by the compact 100-d P summary; a full-rank H_L readout does not reproduce it and a joint readout loses most of it. **Not supported:** that the information is absent from H_L; that it is depth-specific; any mechanism.
- **Allocation (frozen row C):** **stop** this branch; no further budgets, readouts, regularizers, layers or checkpoints. The intermediate-specific framing of Stage 0 is not established; neither is the head-bottleneck framing.
- **Review status:** single-author, agent-assisted; decision computed by the pre-frozen, unit-tested rule.
