# Stage 0 evidence ablation — is the probe gain specific to `layer3.22`? (frozen specification)

**Frozen:** 2026-09-24, before any ablation fit is run. §§0–6 are not to be edited
after results are opened; results go in a separate appended section.
**Card:** `ResearchBrain/05_Experiments/2026-09-24 Stage 0 Evidence Ablation.md`.
**Parent:** `docs/stage0_execution_spec.md` (frozen; its results, deviations and
provenance are unchanged by this study). Hypothesis being probed:
`H-STAGE0-01` (proposed). This is a new bounded development diagnostic, not a
reopening of any closed study, and it authorizes no reserved data and no new
feature extraction.

## 0. Question

Stage 0 found that a target-fitted linear readout of base logits `Z` plus the
`layer3.22` probe logits gains materially over the target-fitted logit-only readout, while
clean-fitted stacking does not. Two explanations that Stage 0 could not separate:

* **Depth-specific:** something about the `layer3.22` evidence is needed.
* **Generic fusion under shift:** any comparably diverse second predictor, including another
  network's logits, gives the same pattern under target supervision.

Claim scope: ResNet-101, CIFAR-100, checkpoints 2 and 4, the 12 development cells, the
frozen Stage 0 folds/regimes/objective; linear multinomial readout; target-supervised
diagnostic. Not a deployable method, not an information ceiling.

## 1. Evidence sources (only `P` changes)

Everything else is the Stage 0 code path unchanged: `atlas/stage0_folds` plan (seed 20260922),
inner 75/25 split, nested 2,500 subset, per-image cell assignment, λ grid
`{1e-1,…,1e-5}`, the L-BFGS fitter and convergence/retry policy, per-coordinate training-only
standardization, `q_ZP = softmax(A z̃ + B p̃ + b)`, mean-NLL objective without a ½ factor, and
the 2,000-resample grouped paired bootstrap.

| id | evidence `P` (100 columns) | source |
|---|---|---|
| P_ref | `layer3.22` probe logits (candidate index 8) | Stage 0 outputs, reused, not refit |
| P_A | `layer4.2` probe logits (index 11) | `results/layer_pilot/checkpoint_seed{s}/<cell>/per_sample.npz`, `raw__probe_logits[:, 11, :]` |
| P_B | the other checkpoint's base logits on the same images (seed 2 ↔ seed 4) | `results/atlas/seed{6−s}/u0/<cell>.npz`, key `logits`; row alignment checked by labels |
| P_C | each of the other cached probe layers (indices 0–7, 9, 10), together with P_A and P_ref giving all 12 layers | same file, index `k` |

Probe logits are the raw stored (pre-temperature, float16-quantized) values, as in Stage 0.
`Z` is the canonical FP32 atlas logits of the checkpoint under study.

## 2. Regimes and fits

* P_A and P_B: T-8k×1, S-8k×1, T-2.5k×1, S-2.5k×1 (primary: 8k×1).
* P_C (the other ten layers): T-8k×1 and S-8k×1 only (secondary, descriptive; the
  layer curve uses Δ_T at 8k×1). P_ref comes from Stage 0.
* All five outer folds, both checkpoints. The `q_Z` arm does not use `P`; it is refit anyway and
  compared with the Stage 0 `q_Z` outputs as a determinism check (reported, not a gate).
  Planned: 28 evidence–regime combinations × 5 folds × 2 checkpoints = 280 runs (560 arm-fits).
* CPU only. Engineering recovery (convergence, memory, timeout, packaging) is allowed and
  logged; anything that changes features, objective, grid, folds or exclusions is a
  scientific change requiring an amendment.

## 3. Quantities reported

Per evidence source `P` and checkpoint:
* Δ_T(P|Z) and Δ_S(P|Z) at 8k×1 and 2.5k×1: 12-cell macro accuracy of `q_ZP` minus `q_Z`
  (pp), pooled out-of-fold, grouped bootstrap intervals; gap Δ_T − Δ_S (joint resamples);
  clean-view increment (S-fit `q_ZP` − `q_Z` on clean held-out images).
* Standalone accuracy of `P` (argmax of the raw logits, clean and per cell), and the
  disagreement rate of argmax `P` with argmax `Z` (clean and per cell; macro over the 12 cells).
* Also NLL and Brier of `q_ZP` and `q_Z` per regime.

**Primary contrast.** D_A = Δ_T(P_ref|Z) − Δ_T(P_A|Z) at T-8k×1, per checkpoint, with a
paired image-group bootstrap (2,000 resamples, one shared draw over the two regimes' per-image
effect arrays; duplicate groups move together).

## 4. Decision rules (frozen; evaluated in this order, all reported)

Definitions. "Near zero" means a point estimate within ±0.5 pp (the Stage 0 practical bar).
"Clearly positive" means ≥ +1.0 pp with the interval excluding 0. All conditions must hold in
**both** checkpoints; if a condition holds in one checkpoint only, the rule is not met and the
result is reported as checkpoint-dependent.

* **R1 — depth-specific story stops.** Δ_T(P_A|Z) ≥ 0.5 × Δ_T(P_ref|Z) at T-8k×1 (point
  estimates), both checkpoints.
* **R2 — generic fusion under shift.** For P_B at 8k×1: Δ_T clearly positive; Δ_S near zero or
  negative (point estimate ≤ +0.5 pp); clean-view increment near zero. (This is the Stage 0
  pattern for `layer3.22`, reproduced with another network's logits.)
* **R3 — generic diversity.** Over the 12 layer probes (P_ref, P_A and the ten P_C): the
  Spearman correlation between disagreement-with-`Z` (macro over the 12 cells) and Δ_T at
  T-8k×1 is ≥ 0.8, **and** P_B's point lies within 2 × the residual RMSE of the least-squares
  line of Δ_T on disagreement fitted to the 12 layer points. (With 12 points and two
  non-independent checkpoints this is descriptive. Correlation with layer depth is reported for
  transparency and is not part of the rule.)
* **Verdict.** The first of R1, R2, R3 that is met, in that order; the others met are listed.
  If none is met: **the depth-specific pattern survives** — propose the confirmation / paper
  package for a separate decision. Do not start it. Confirmation would need a frozen candidate
  and explicit authorization to consume a reserved resource.

Nothing here establishes a cause. A positive R1/R2/R3 verdict weakens the depth-specific
reading only for this readout, budget and these cells; "survives" is not confirmation.

## 5. Data and fitting access; exposure

Same 10,000 test images and 12 development cells, checkpoints 2 and 4, target-label access for
T regimes and clean-label access for S regimes, exactly as Stage 0. P_B uses the other
checkpoint's cached base logits (both checkpoints are already in use). Checkpoints 1/3/5 and the
11 unused CIFAR-100-C families are not accessed. This diagnostic is another development use of
the exposed rows; it adds an exposure entry, not confirmation.

## 6. Provenance rules

Fits and aggregation run from an immutable snapshot of a commit that contains this file.
Post-hoc analyses added afterwards are labelled post-hoc. Job ledger:
`results/stage0_ablation/ledger.json`. No push without the researcher's approval.

## 7. Results

*(appended after all runs complete; §§0–6 unedited above this line)*
