# Regime-map follow-up — amendment 1 (frozen; supersedes the decision table and the comparators of v1)

**Frozen:** 2026-09-26, before any computation of the follow-up. Amends `docs/regime_map_followup_spec_v1.md`
(sha256 `9214ab74e906bab64e3c1adfe4643ca0177805b277f11cdbd9f36b86808a6986`, **left unedited**). Where this amendment and v1 conflict, this
amendment governs; v1 §§0–3, 6 (statistics), 8–10 stay in force except as stated. Results go in an appended §9.
**Every target-label fit is an ORACLE DIAGNOSTIC** (v1 §0): file names, columns, cards and prose carry `oracle_` / that label. No new corruption
family, no reserved resource, no other model.

## 0. Why (researcher's reason, recorded)

In the Phase 2 table the Z-only comparator is an 8k-row refit whose clean accuracy differs from the base head by state-dependent amounts (−4.2 pp for (a),
+8.2 / +5.0 for b1, +1.6 / +3.7 for b3, −1.3 / −1.0 for b10 under S-8k×1; the per-state values are in the Phase 2 card's correction section, read from
`results/regime_map/report/regime_aggregate.json`). Increments were therefore measured against comparators of state-dependent quality. Anchored readouts remove that
dependence.

## 1. A1 — anchored readouts are primary

For every arm the readout is **residual with a frozen output-only anchor**:

    q(x) = softmax(z + f(x)),   f(x) = X̃(x) W + b     (linear in the arm's inputs)

`z` is the state's stored logits (`results/regime_map/<state>/<cond>.npz`, key `z`), used unchanged (no temperature). Everything else follows Stage 0: `X̃` per-coordinate
standardized with statistics of the fit rows only; **[CONVENTION — stated because it differs from the residual-evidence study]** the ridge is Stage 0's: mean cross-entropy + λ‖W‖²_F
(no ½, no 1/K), bias `b` unpenalized, zero init, float64 L-BFGS with the Stage 0 tolerances and retry policy, so that "same λ protocol as Phase 2" holds. `Calibrators/residual_readout.py`
uses `λ‖W_c‖²/(2K)`, no bias and a different grid; the two λ scales relate as `λ_res = 200 · λ_stage0`, and class-centering is immaterial at the ridge optimum.
**λ path:** {1e-1, 1e-2, 1e-3, 1e-4, 1e-5} **plus the candidate "f = 0"** (W = 0, b = 0: exactly the base head). Selection: minimum inner-validation NLL (same grouped 75/25 image split as Phase 2,
inherited from `results/stage0/shared/fold_plan.json`); ties (|ΔNLL| < 1e-9) → f = 0, then the larger λ. Refit at the selected λ on all outer-training rows; f = 0 needs no refit.
Same 5 folds, T-8k×1 fit rows (one view per image), same evaluation rows (held-out fold, 12 corrupted cells, plus the clean copy for the clean columns).

**Reported for every arm:** (i) `Δ_T(arm) = anchored(arm) − anchored(Z-only)`, 12-cell macro accuracy, pp; (ii) `anchored(arm) − base` (base = accuracy of the state's own argmax-z on the same rows), macro-12 and clean.
The unanchored Phase 2 quantities (clean increment, Δ_S, Δ_T, gap) are kept unchanged as a **secondary row for continuity**, read from `regime_aggregate.json`, not refit.
`D = anchored Δ_T(Z, P)`. "Explains x %" = `Δ_T(control) / D` (per state and fine-tuning seed), with its bootstrap interval (ratio computed per resample).

## 2. A2 — arms

| id | inputs to f | role |
|---|---|---|
| Z-only | `z` | anchored comparator |
| (Z, P) | `z, p` — `P` = the 100-d source-fitted layer3 probe logits, unchanged | `D` |
| **C1b** | `z` and `φ(z̃)` = ReLU(R z̃), R = `default_rng(20260927).standard_normal((100,100))/10` (100 random ReLU features of standardized `z`, both standardized with fit rows) | **capacity control** |
| **C1e = (Z, P_L)** | `z` and `p_L` — the 100-d logits of a probe on `h_L` fitted with **P's exact protocol** (data: the state's 45k train features; L2-normalize then z-score; λ grid {1e-4, 1e-3, 1e-2} with the grid-edge rule of spec v2 §3; `Calibrators/layer_readouts.fit_probe`, full-batch L-BFGS) | **second-classifier control**: (Z, P_L) vs (Z, P) separates "layer3 information" from "any second clean-fitted classifier" |
| C1c | `h_L` alone (per-coordinate standardized), anchored | **ceiling reference** (not a capacity control) |
| C1d | `z` and `h_L`, anchored | **ceiling reference** (not a capacity control) |
| K | `P_ker u` alone | Check 2 |
| Z+K | `z` and `P_ker u` | Check 2 (`Z + P_ker h_L`) |
| Prow | `P_row u` alone | Check 2 sanity |

`u`, `P_row`, `P_ker` as in v1 §4 (for state (a) `u = (L2norm(h_L) − μ)/σ`, the head's standardized input space; `W_h` from the refit head). C1c and C1d have ≈ 2048 inputs and are **excluded from any capacity residual**; the v1 residual `R`
and its rules are dropped. H stays the implemented `P` (v1 §2); the underlying 1024-d representation is not an input of any arm.

## 3. A3 — decision states

**b10_s1 and b10_s2 carry the decision and must agree.** State (a), b1 and b3 are descriptive only (their rows are reported, never enter a gate or the decision).

## 4. A4 — gates and decision table (replaces v1 §7)

All quantities: anchored, target-fitted, 12-cell macro, per fine-tuning seed, on the pooled out-of-fold predictions; gates use **point estimates** and the intervals are reported **[CONVENTION: the researcher's wording gives no
interval rule for the gates]**. Bootstrap: paired over evaluation images (duplicate groups move together), 2,000 resamples, **one shared resample-index array per state used by every arm**.

* **Gate 1** (existing `z`/`p` arrays only — Stage 1): **STOP** if `D < 0.5 pp` in **either** b10 seed (effect too small to matter). **STOP** if C1b explains ≥ 50 % of `D` in **either** b10 seed (capacity).
* **Gate 2** (after `h_L` extraction — Stage 2): **STOP** if `Δ_T(Z, P_L)` explains ≥ 50 % of `D` in **both** b10 seeds (second-classifier effect, not layer3 information).
* **HEAD-DISCARD:** `Δ_T(C1d) ≥ D` **and** `Δ_T(Z + P_ker h_L) ≥ 75 % × Δ_T(C1d)` (with `Δ_T(C1d) > 0`), in **both** b10 seeds.
* **INTERMEDIATE-SPECIFIC:** `D − Δ_T(C1d) > 0` with the 95 % interval **excluding 0**, in **both** b10 seeds.
* **Anything else, or disagreement between the two seeds: INCONCLUSIVE — stop.** No extra seeds, cells, layers or variants.

Stages are run in this order and **each gate stops the run**; a stop is reported with the table and nothing beyond the pre-declared row. HEAD-DISCARD and INTERMEDIATE-SPECIFIC are mutually exclusive (`Δ_T(C1d) ≥ D` versus `D − Δ_T(C1d) > 0`).

## 5. A5 — label budget (Stage 3; descriptive; only if Gate 2 passes)

Sizes **n ∈ {5, 20, all}** labelled images per class (2 is dropped). Arms: Z-only, (Z, P), C1b, C1e, C1c, C1d (anchored). **λ is fixed at the value selected at "all" for the same arm, state and fold — an ORACLE choice, declared as such**; no inner split is used at
n < all (so the inner-validation noise of v1 disappears; an arm whose "all" selection was f = 0 stays f = 0). Draws: within each outer fold, n images per class uniformly without replacement, `default_rng([20260928, fold, n, draw])`, 5 draws (draw *d* over the 5 folds forms one pooled out-of-fold evaluation);
each drawn image keeps its preassigned corrupted view. Report per arm and size the increment over the same-draw anchored Z-only readout (mean and SD across the 5 draws) and `Δ(5/class) / Δ(all)`.

## 6. A6 — Check 2 conventions and consistency thresholds (items 4, 5, 8 of v1 approved)

* Standardized space for state (a): `u = (L2norm(h_L) − μ)/σ` (the head's own input space); for b\*: `u = h_L`.
* The `Prow` discrepancy (difference of macro accuracy and mean absolute difference of predicted probabilities against anchored Z-only) is **reported, not gated**.
* **Consistency thresholds (explicit):** b\*: `max |z_stored − (W h_L + b)| ≤ 1e-2` over all extracted rows and all 13 conditions (W, b from the fine-tuning checkpoint); (a): the head refitted on the (a) train features at the stored λ (`results/regime_map/a/summary.json`,
  `head_linear_layer4_gap.selected_lambda`) must reproduce the stored argmax of `z` on **≥ 99.9 %** of all extracted rows (mean |Δz| reported). **If the (a) head refit fails, stop and report; nothing further is fitted.**
  `C1e`: no numerical threshold; the probe's selected λ, edge status, train/val accuracy are reported.
* Stage 1 also has a reproduction check: the unanchored Phase 2 `Δ_T` is read from the artifact (not refit); the anchored Z-only and (Z, P) fits at the f = 0 candidate must equal the base head exactly (0 differing predictions) — a test of the anchoring code, asserted in the run.

## 7. Stages

* **Stage 1 (CPU only; existing `z`/`p` arrays):** anchored Z-only, (Z, P), C1b for all 7 states × 5 folds (105 arm-fits, 35 (state, fold) tasks). Apply Gate 1. Report the per-state × seed table: anchored Z-only − base, (Z, P) − base, `D`, the C1b fraction (each with intervals) and the secondary unanchored row.
* **Stage 2 (only if Gate 1 passes):** `h_L` extraction for all 7 states (train 45k + val 5k + test 130k; ≈ 0.2 GPU-h), the (a) head refit and consistency checks, `C1e`, `C1c`, `C1d`, `K`, `Z+K`, `Prow`. Apply Gate 2, then HEAD-DISCARD / INTERMEDIATE-SPECIFIC / INCONCLUSIVE. Report the table with the C1e, C1c and C1d columns.
* **Stage 3 (only if Gate 2 passes):** the label budget of §5.
Costs: Stage 1 ≈ 105 arm-fits at ≈ 55 s per 4-core arm (`results/regime_map/fits/*/T-8k1/fold*.npz`, key `wall_time_s`, mean over 35 files ÷ 2 arms) ≈ 1.6 h of 4-core time, i.e. ≈ 6 CPU-hours allocated. Stage 2 and 3 estimates: v1 §9 (extrapolated), to be re-stated before Stage 2/3 launch; nothing beyond Stage 1 is launched by this amendment's freeze.

## 8. Execution and provenance

Immutable snapshot of a commit containing this amendment; jobs and ledger under `results/regime_map_followup/`; node type and `lscpu` recorded per task (Stage 0 numerics note); engineering recovery only; no push of results without the researcher's approval; vault notes append-only.
Unit tests before any launch: anchored fitter equals the Stage 0 fitter when the anchor is zero; the f = 0 candidate reproduces the base head exactly; λ path/tie rules; the gate and decision functions on synthetic inputs (every branch, both-seed and either-seed rules).

## 9. Results

*(appended after the runs; §§0–8 unedited above this line)*
