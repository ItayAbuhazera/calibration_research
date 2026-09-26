# Regime-map follow-up — is the target-fitted increment capacity, the discarded last-layer directions, or the intermediate evidence? (frozen specification v1)

**Frozen:** 2026-09-26, before any computation of this follow-up. §§0–11 are not to be edited after results are opened;
results go in an appended §12. **Status: written, NOT launched; awaiting the researcher's approval** (thresholds are
the researcher's; items marked **[ADAPTATION]** are the smallest adaptations I could find to make an ambiguous or
inapplicable part of the design executable, and need the researcher's confirmation before launch).
**Parents:** `docs/regime_map_pilot_spec_v2.md` (frozen, complete; row 4 fired in both fine-tuning seeds) and
`docs/stage0_execution_spec.md`. Neither is edited. Card to be created at approval:
`ResearchBrain/05_Experiments/2026-09-26 Regime-Map Follow-up.md`.

## 0. Scope and labelling

Same states, fine-tuning seeds, corruption cells, folds and λ protocol as Phase 2 (7 states: `a`; `b1_s1, b3_s1, b10_s1`;
`b1_s2, b3_s2, b10_s2`; 12 development cells = {gaussian_noise, defocus_blur, fog, jpeg_compression} × severity {1, 3, 5};
Stage 0 fold plan seed 20260922). **No new corruption family, no reserved resource, no other model.** **Every
target-label fit in this study is an ORACLE DIAGNOSTIC**: it uses the labels of the exposed evaluation domain and
is an upper-reference quantity for a restricted readout, not a deployable method. Output directories, file names, table
columns, cards and prose carry the prefix `oracle_` / the label "ORACLE DIAGNOSTIC (target labels)".

## 1. Question

Phase 2 established that the target-fitted increment Δ_T of the (Z, H) readout over the Z-only readout is large and is not
recovered by source (clean) fitting, in every state, and that clean redundancy does not control that gap. This follow-up asks
where the target-fitted increment comes from:
1. **Capacity:** would a target-fitted readout with more inputs or a bigger feature space, but no intermediate-layer information,
   give the same increment (Check 1)?
2. **Discarded last-layer directions:** does the increment live in the part of the final representation `h_L` that the network's
   own head discards, i.e. its kernel directions (Check 2)?
3. **Label budget:** how does each increment depend on the number of labelled target images (Check 3, descriptive)?

## 2. Definitions (verified from the code; see `results/regime_map/` producers)

* `h_L` ∈ ℝ²⁰⁴⁸: global average pool of the `layer4` output of the ResNet-50 (`atlas/regime_map.py::forward_feats`, `g4`).
* `Z` ∈ ℝ¹⁰⁰: the state's own logits. States b\*: `fc(h_L)`, `fc` = `Linear(2048, 100)` (weight `[100, 2048]`, bias `[100]`, in the
  fine-tuning checkpoints). State (a): a linear head fitted on the frozen backbone's train features,
  `logits = ((L2norm(h_L) − μ)/σ) W_(a) + b_(a)` with `W_(a)` `[2048, 100]`, μ, σ `[2048]` (`regime_map.py::fit_linear`, `apply_linear`;
  λ selected by inner-FIT NLL).
* **`H` = `P` ∈ ℝ¹⁰⁰: the raw logits of the GAP linear probe on the last block of `layer3`** (`g3` ∈ ℝ¹⁰²⁴; `P = ((L2norm(g3) − μ)/σ) W_p + b_p`,
  `W_p` `[1024, 100]`; `regime_map.py::build_state`), stored as `p` in `results/regime_map/<state>/<cond>.npz`. **[ADAPTATION]** In the
  design text "H" and "dim(H)" are read as this 100-dimensional probe-logit evidence, because that is what the Phase 2 readout uses; the underlying
  intermediate representation (`g3`, 1024-d) is not an input of any arm here.
* Readout (Stage 0 fitter, unchanged; `atlas/stage0_fit.py::fit_arm`, called by `atlas/stage0c_run.py::run_one`): per-coordinate standardization
  fitted on the fit rows only, `q = softmax(X̃W + b)`, objective mean cross-entropy + λ‖W‖²_F (no ½ factor, bias unpenalized), λ ∈ {1e-1,…,1e-5}
  chosen by inner-validation NLL (grouped 75/25 image split of the outer-training images, `results/stage0/shared/fold_plan.json`),
  refit on all outer-training rows at the selected λ; float64 L-BFGS (strong-Wolfe), tolerance and retry policy of Stage 0.
* Target fit (T-8k×1): per outer fold ≈ 8,000 outer-training images (per class 67–93, mean 80.0), **one view per image** — the image's preassigned
  corrupted cell (≈ 666–668 images per cell), one pooled model for all cells; evaluation on the held-out fold's images at all 12 corrupted cells; predictions
  pooled out-of-fold over the 5 folds (10,000 images per cell). `Δ_T(arm)` = 12-cell macro of [accuracy(arm) − accuracy(Z-only)] on the same held-out rows
  (`atlas/regime_aggregate.py::arrays`). The Z-only comparator is fitted with exactly the same rows, inner split, λ path and refit.

## 3. Data to be produced (nothing exists on disk yet)

Verified absent: no `h_L` (or any `layer3`/`layer4` feature) file exists for any state; the head `W` of state (a) and the probe weights are not stored;
the b\* head weights are in `results/regime_map/ckpt/s{1,2}/b{1,3,10}.pt`. Extraction (GPU; strict fp32, same pipeline and resolution policy as Phase 2):
* all 7 states: `h_L` for the 13 conditions × 10,000 test images (float32, 1.06 GB per state); b\* also `W, b` from the checkpoint;
* state (a) additionally: the train (45,000) and validation (5,000) `h_L` to **refit the head** with its stored λ (`results/regime_map/a/summary.json`,
  `head_linear_layer4_gap.selected_lambda`), giving `W, b, μ, σ`.
* **Z and P are taken from the existing artifacts** (`z`, `p` in `results/regime_map/<state>/<cond>.npz`) for every arm, so all arms share one `Z` and `H`.
**Consistency checks (report; frozen thresholds [ADAPTATION: mine, for review]):** b\*: max |z_stored − (W h_L + b)| ≤ 1e-2 over all rows and conditions; (a): the refit head
must reproduce the stored argmax on ≥ 99.9% of rows (and mean |Δz| reported). **If a check fails, stop and report; nothing is fitted.**

## 4. Arms (all target-fitted, ORACLE DIAGNOSTIC, identical protocol of §2; each reported as an increment over the Z-only readout)

| id | inputs to the readout | dim |
|---|---|---|
| Z-only (C1a) | `z` | 100 |
| (Z,H) | `z, p` | 200 |
| **C1b** | `z` and `φ(z̃)` = ReLU(R z̃), where `z̃` is `z` standardized with the arm's own fit-row statistics and `R` ∈ ℝ¹⁰⁰ˣ¹⁰⁰ is fixed: `R = default_rng(20260927).standard_normal((100,100)) / 10` (same `R` for all states, folds, seeds); `φ` is standardized per coordinate with the same fit rows **[ADAPTATION]**: "Z expanded to dim(H)" read as 100 random ReLU features appended to `z` so the arm has the same 200 inputs as (Z,H) | 200 |
| **C1c** | `h_L` alone (DFR-style last-layer retraining on target data), per-coordinate standardized | 2048 |
| **C1d** | `z` and `h_L` | 2148 |
| **K** (Check 2) | `P_ker u` alone | 2048 (rank ≤ 1948) |
| **Z+K** (Check 2) | `z` and `P_ker u` | 2148 |
| **Prow** (sanity) | `P_row u` alone | 2048 (rank ≤ 100) |

Check 2 definitions: `u` is the head's own input vector — `u = h_L` for b\*; **[ADAPTATION]** for state (a) `u = (L2norm(h_L) − μ)/σ`, because the (a)
head acts on that space. With the head `z = W_h u + b`, `W_h` ∈ ℝ¹⁰⁰ˣ²⁰⁴⁸ (transposed for (a)), `P_row = W_h⁺ W_h` and `P_ker = I − P_row` (Moore–Penrose pseudo-inverse
in float64). **Sanity check:** `P_row u = W_h⁺(z − b)` is an invertible linear image of `z`, so a readout on `Prow` should match Z-only up to the affine map; report the discrepancy
(difference of macro accuracy, mean absolute difference of predicted probabilities on the evaluation rows). Per-coordinate standardization and the ridge penalty are not invariant
to that map, so a small non-zero discrepancy is expected and is a reported quantity, not a gate. Unit tests (planned): `P_row² = P_row`, `W_h P_ker = 0`, `P_row u + P_ker u = u`.

**Residuals (per state, per fine-tuning seed):** `R_x = Δ_T(Z,H) − Δ_T(C1x)` for x ∈ {b, c, d}; **`R = min_x R_x = Δ_T(Z,H) − max(Δ_T(C1b), Δ_T(C1c), Δ_T(C1d))`**.

## 5. Check 3 — label budget (descriptive)

Arms Z-only, (Z,H), C1b, C1c, C1d at target-fit sizes **n ∈ {2, 5, 20, all} labelled images per class** ("all" = the full T-8k×1 fit rows of §2, no draw).
For n ∈ {2, 5, 20}: within each outer fold, draw n images per class uniformly without replacement from the fold's outer-training images, `default_rng([20260928, fold, n, draw])`,
**5 draws** (draw = 0…4; draw d over the 5 folds forms one pooled out-of-fold evaluation). Each drawn image keeps its preassigned corrupted view and its inner-fit/inner-val role from the
Stage 0 plan **[ADAPTATION: inherited, not re-split; at n = 2 the inner-validation set has ≈ 50 images, so λ selection is noisy — a stated limitation]**.
Report per arm and size: the increment over the same-draw Z-only readout, its mean and SD across the 5 draws; and, for every outcome, the **increment at 5 labels/class as a
fraction of the all-label increment**.

## 6. Statistics

Paired bootstrap over evaluation images (duplicate groups move together), 2,000 resamples, **one shared resample-index array per state/fine-tuning seed used by every arm** (so R and all
differences are paired); `R` and its interval are recomputed inside each resample (max taken per resample). Fine-tuning seeds and states are reported separately and never pooled in a
decision. Intervals are conditional on the fitted CV predictions and contain no training-seed variance. Reproduction check: Z-only and (Z,H) `Δ_T` recomputed here must match
`results/regime_map/report/regime_aggregate.json` (`states.<s>.regimes.T-8k1.delta_acc_pp`) within 0.05 pp **[ADAPTATION: mine; Stage 0 noise bound was 0.01 pp]**, else stop and report.

## 7. Decision table (frozen before any computation; thresholds are the researcher's)

Decision-bearing fits **[ADAPTATION — needs confirmation]:** the design says "in both seeds" but state (a) has a single fit. The rule is evaluated on **D = {b10_s1, b10_s2, a}**;
"in both seeds" is read as "in every fit of D", and **an outcome holds only if every fit in D supports it, otherwise INCONCLUSIVE**. States b1/b3 are reported descriptively.

Evaluate in this order (the first that holds is the outcome):

1. **HEAD-DISCARD:** for every fit in D: `R_b ≥ 0.5 pp` (C1b does not bring R below 0.5) **and** `min(R_c, R_d) < 0.5 pp` (C1c or C1d does) **and** `Δ_T(Z+K) ≥ 0.75 × Δ_T(C1d)` (with `Δ_T(C1d) > 0`; point estimates).
   Reading: mechanism candidate — the head discards shift-relevant directions.
   *Precedence note [ADAPTATION]:* the KILL condition as worded (R < 0.5) would also be met whenever C1c or C1d alone absorbs the increment, which would make HEAD-DISCARD unreachable; it is therefore tested first.
2. **KILL (capacity):** for every fit in D: `R < 0.5 pp` **or** the 95% interval of R includes 0. Reading: the gap is target-adaptation capacity; close the regime-map line.
3. **INTERMEDIATE-SPECIFIC:** for every fit in D: `R ≥ 1.0 pp` with the 95% interval of R excluding 0. Reading: H carries information not available in `h_L` or capacity-matched `Z`.
4. **Otherwise, or any disagreement among the fits of D: INCONCLUSIVE — stop.** No extra seeds, cells or variants.

The label-budget report (§5) is produced for every outcome. Nothing here establishes a mechanism outside the tested readout family; every outcome is an oracle-diagnostic statement.

## 8. Reporting

Per state and fine-tuning seed: `Δ_T` with intervals for Z-only (0 by definition), (Z,H), C1b, C1c, C1d, K, Z+K, Prow; `R_b, R_c, R_d, R` with intervals; selected λ per arm and fold
(with convergence/retry and grid-edge audit); absolute macro/clean accuracies of every arm; the consistency and reproduction checks; Check 3 tables. All outputs labelled ORACLE DIAGNOSTIC.

## 9. Cost estimate and job plan (estimates; nothing is launched)

Measured inputs (read from artifacts): extraction throughput `results/regime_map/smoke/smoke.json` key `extract_fp32_img_per_s`; Stage 0 fitter wall time per T-8k×1 fold (both arms, 100- and 200-d inputs)
`results/regime_map/fits/*/T-8k1/fold*.npz` key `wall_time_s` (35 files).
* **GPU (extraction):** 6 states × 130,000 + state (a) 180,000 images ⇒ 960,000 images at the measured throughput ⇒ ≈ 0.16 GPU-h (566 s), plus a few minutes for the head refit and `W`/`h_L` writes;
  storage 1.06 GB per state (7.5 GB). **≈ 0.2 GPU-h total.**
* **CPU (fits):** per (state, fold) 8 arms at full size (input dims 100, 200, 200, 2048, 2148, 2048, 2148, 2048) plus 5 arms × 3 sizes × 5 draws. **Extrapolated assuming fit time ∝ rows × input dimension** from the measured
  ≈ 55 s per arm at ≈ 150 inputs and 8,000 rows: ≈ 67 min for the full-size arms and ≈ 48 min for the budget arms per (state, fold) on 4 cores, i.e. **≈ 115 min per task, 35 tasks (7 states × 5 folds), ≈ 270 CPU-hours allocated**;
  wall ≈ 2 h per task if all 35 run in parallel. This is an upper-end linear extrapolation (not measured; BLAS efficiency may improve it, iteration counts for ill-conditioned wide inputs may worsen it). A GPU implementation of the same fitter would be much
  faster but would need a CPU/GPU objective-and-gradient parity test first (not part of this plan unless requested).
* **Job plan (after approval):** (1) unit tests + extraction (7 GPU jobs) → (2) consistency and reproduction checks (CPU, minutes) → (3) fit array `(state, fold)` = 35 CPU tasks, outputs per arm: per-image correctness and true-class NLL for the 12 cells (no probability arrays), λ, convergence flags → (4) aggregation with the frozen decision function (CPU). Immutable snapshot of a commit containing this spec; ledger `results/regime_map_followup/ledger.json`; node type and `lscpu` recorded (Stage 0 numerics note); engineering recovery only; no push without approval.

## 10. Exposure

Development reuse of the same exposed images and cells on the same models; target labels of the 12 development cells used for oracle fits (as Phase 2). Not accessed: any other CIFAR-100-C family, checkpoints 1/3/5, new test data. Ledger entry to be appended when the workflow ledger is unblocked (kept in the card until then).

## 11. Planned implementation notes (not written yet)

`atlas/followup_extract.py` (h_L, W), `atlas/followup_fit.py` (arms, budgets; reuses `stage0_fit.fit_arm`; C1b feature map as a fit-time transform), `atlas/followup_rules.py` (§7 as a pure function; unit-tested for every branch,
HEAD-DISCARD-before-KILL precedence, and the D-agreement rule), tests for the projector identities and the stratified draws.

## 12. Results

*(appended after the runs; §§0–11 unedited above this line)*
