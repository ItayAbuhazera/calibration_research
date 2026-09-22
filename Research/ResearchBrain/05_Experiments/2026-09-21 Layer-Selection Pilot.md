---
type: experiment
status: completed
date: 2026-09-21
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C (12 development cells), ResNet-101, checkpoint seeds 2 and 4, corrected_v2_train_norm
preregistered: true
evidence_scope: unpublished_repository_analysis
tags: [layer-selection, dac, fv-dac-successor, decision-utility, pooling]
---

# 2026-09-21 — Layer-selection pilot (L ∈ {1,4,6,8})

Spec (frozen before any corrected corruption cell): repo `docs/layer_selection_pilot_spec.md` §§0–12;
results §13, interpretation §14 (appended). Hypothesis: [[H-LAYER-01 Selected internal layers add decision value beyond logits]].
Predecessors: [[2026-09-20 Full-Vector DAC POC]] (closed, legacy protocol — untouched),
[[2026-09-21 Normalization Audit and Corrected Protocol]]. Theory: [[Theory Plan - Decision Utility, Layers, Compression and Risk Control]].

## Question
Does access to additional appropriately *selected* internal layers improve clean-fitted decision correction and probability quality
under corruption, and is any benefit specific to geometric statistics? Separates (A) layer choice, (B) pooling, (C) distances vs probes,
(D) readout, (E) clean-to-shift transfer, (F) preprocessing.

## Design (verified implementation facts)
12 candidate Bottleneck block outputs (`layer1.0 … layer4.2`, dims 256–2048), GAP+L2 (native-DAC operation); logits are a control, not a layer.
Family A = `softmax((z − β·mean_{l∈A} r_l)/S_DAC)`, K_c=5 (inherited from the legacy-protocol FV-DAC pilot), corrected-protocol native-DAC S_DAC,
β by NLL on inner-FIT. Family B = equal-weight average of clean-fitted temperature-calibrated linear probes (+ full-logit probe, final-representation
probe). Greedy forward selection on inner-SELECT NLL → nested A₁⊂A₄⊂A₆⊂A₈, plus depth-spaced controls and a permuted-label control. Split roles
`layer_pilot_split_v1`: bank/probe-fit = train 45k, inner-FIT/SELECT = 2.5k/2.5k validation, clean test and 12 cells evaluation-only. Two seeds
declared before shift evaluation. **The 12 cells are previously inspected development conditions.**

## Results (measured; details and per-arm table in repo spec §13)
* Frozen §9 verdict: **`no_material_evidence_to_continue_tested_family`.** Best candidates: `B_greedy_L4` +0.095 pp, `A_greedy_L4` +0.076 pp
  (bar: ≥ +0.5 pp with ≥ +0.25 pp in each seed; promising needs +1.0 pp). Vector Scaling (logit-only): **+0.190 pp**.
* Family A: L=1/4/6/8 greedy ΔAcc +0.025 / +0.076 / −0.008 / −0.024 pp; depth-spaced +0.043 / +0.089 / +0.094 / +0.091. Calibration cost:
  ECE 0.20–0.21 vs corrected native DAC 0.132 (NLL 2.36–2.40 vs 2.255). Permuted-label control ≈ 0 flips.
* Family B greedy: +0.095 (L=4, NLL 2.201, ECE 0.115 — better than native DAC) → −0.124 (L=6) → −0.884 (L=8). Full-logit probe +0.039 (NLL 2.65).
* Greedy inner-SELECT NLL is minimal at L=1–4 (Family A s2 min at L=3, s4 at L=1); more layers worsen it.
* Effect sign depends on corruption family (Family A L=4: −0.68 pp gaussian_noise, +0.68 pp jpeg) and shrinks with severity; clean-test gains
  (+0.35–0.56 pp) do not predict corruption gains. Flip base-error enrichment vs top-2-margin-matched null ≈ 1.06–1.09.
* Sensitivity (pre-declared 2×2 pooling, Family A, selection not re-run): +0.443 / +0.368 / +0.312 / +0.327 pp, W>H 19/17/16/14 of 24 — still below the
  floor; ECE 0.21.
* Exploratory, post-hoc (`Experiments/layer_pilot_exploratory_diagnostics.py`, uses target labels descriptively): AUC of layer4.1/4.2 distance
  difference for separating repairs (W) from harms (H) among top-2 pairs ≈ 0.81–0.82 vs 0.81 for the base logit margin alone; within-class layer
  correlation ρ≈0.6; equal-weight d′ of greedy sets is non-monotone in L (peaks at L=2–4).

## Interpretation (labelled)
* **Unresolved interpretation:** no accuracy gain in this pilot is separable from what a logit-only readout achieves; layer choice by clean NLL did not
  beat depth-spaced; more layers did not help monotonically (consistent with the correlated-layer mechanism in the theory plan, not proven).
* **Hypothesis-generating (not a result):** pooling — 2×2 spatial pooling raised the same operator's ΔAcc by ≈ 0.3–0.4 pp.
* Does **not** establish: absence of information in the un-pooled tensor; that no readout could help; anything across seeds beyond n=2; anything about
  other architectures/datasets; significance.

## Artifacts
Repo `Research/GeometricFullCalibration/`: `Experiments/layer_selection_pilot.py`, `Calibrators/layer_readouts.py`, `Experiments/aggregate_layer_pilot.py`,
`Experiments/layer_pilot_spatial_sensitivity.py`, `tests/test_layer_selection_pilot.py`; `results/layer_pilot/checkpoint_seed{2,4}/`,
`results/layer_pilot/aggregate/`, `results/layer_pilot/spatial_grid2/`; state hashes seed 2 `209405a4…`, seed 4 `03e4572f…`.
Slurm: `scripts/layer_pilot_run.sbatch`, `scripts/layer_pilot_spatial_grid2.sbatch`, `scripts/corrected_v2_*.sbatch`.

## Not done (conditional follow-ups only)
Target-labelled oracle suite; BN-Adapt suite; spatial pooling as a primary; certificate role for risk control; other corruption families / seeds.

## Corrections appended 2026-09-21 (second pass; results and frozen verdict unchanged)
* The verdict is a **practical** one under a frozen rule: corrected-protocol arms were small positive on average (+0.03 to +0.10 pp) — "no material evidence", not "zero".
* "A probe-average gain is a calibration effect, not geometry" is an **untested attribution**; only the metric change is measured.
* The exploratory AUCs (≈0.81–0.82 geometry vs 0.81 margin) do not show zero incremental information given the logits.
* The layer-addition threshold and ρ≈0.6 are illustrative and model-dependent.
* The pooling sensitivity motivates a *prospective* comparison ([[2026-09-21 Residual Evidence Study]]); it is not confirmation evidence.
