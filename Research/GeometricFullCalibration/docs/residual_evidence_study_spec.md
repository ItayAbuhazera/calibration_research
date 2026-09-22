# Residual decision-information study — frozen specification

**Frozen:** 2026-09-21, after Stage 0 (provenance) and Stage 1 (cheap diagnostics from existing
predictions) and after the code was written and unit-tested, **before any Stage-2 prediction
was produced or inspected.** Text above the marker `<!-- END FROZEN -->` is hashed
(`results/residual_study/freeze_manifest.json`); results and interpretation are appended
below the marker only. Namespace: `results/residual_study/` (nothing under `results/layer_pilot/`,
`results/studyAB/` or `results/fv_dac/` is written).

This is an **accessible-information diagnostic**. It does not estimate a Bayes-optimal
information content, does not claim a new calibrator, and a reproducible null that closes the
hypothesis is a successful outcome. The scalar/class-distance mean-pooling family remains
**closed** under its own rule (`docs/layer_selection_pilot_spec.md` §13); the FV-DAC verdict
(legacy protocol) is untouched.

> **Question.** Does a specified spatial or class-distance representation expose useful,
> source-learnable decision information beyond strong full-logit readouts, when the same correction
> family is used for every evidence source and capacity, selection and calibration cost are accounted for?

## 0. Corrections adopted before running (interpretation; originals preserved and appended-to)

1. For a fixed alternative `j` and base `i`, `D=1{j=Y}−1{i=Y}`; the optimal *evidence-measurable binary gate* has
   value `E[(E[D|F])_+]`. It equals `Acc*(F)−Acc(base)` only when the alternative may be chosen as the best
   `F`-measurable class (a multiclass action set); it does **not** generally attain unrestricted multiclass Bayes accuracy.
2. A conservative JL sufficient dimension larger than our embedding dimension means the bound does not certify the
   compression we use; it is not a necessary dimension and does not show distance preservation is practically impossible.
3. Probe averaging improved probability metrics; whether geometry played a role was **not tested**. Observed metric
   change and attribution are separate statements.
4. Small positive gains are small positive gains. "No material evidence" is the practical verdict, not "zero".
5. Similar marginal AUCs for geometry and margin do not show zero incremental information given the full logits; higher AUC alone does not show usefulness.
6. The layer-addition threshold (0.8–0.95) is model-dependent (equicorrelated Gaussian, equal weights, ρ≈0.6); an illustration, not a law.
7. Target utility is not identifiable under **unrestricted** shift from unlabeled evidence; restricted shift families can change this.

## 1. Verified provenance (Stage 0; `results/residual_study/stage0_stage1/`)

* Preprocessing: `corrected_v2_train_norm` everywhere; benchmark intermediates and fitted state are stamped corrected.
* Native DAC baseline variant: **five-source benchmark DAC** (`conv1, layer1..layer4`, logits NOT a source, k=200). The paper's six-source variant is not used.
* Pilot ↔ benchmark (clean, both seeds): identical test labels; argmax agreement of base/TS/VS/native DAC 0.9995–0.9998 (TF32 vs strict-fp32 forward passes); metric-definition crosscheck max |Δacc| 1e-4, |ΔNLL| 3.2e-5, |ΔECE| 5.6e-4; inner-FIT and inner-SELECT index hashes equal the pilot's.
* Benchmark **corruption** cells (job array 21533078) are not available (pending behind the benchmark clean fit 21532856); pilot ↔ benchmark reconciliation for corruption cells is therefore *not yet possible* and is stated as such. This study does not depend on them.
* Index manifests (`manifest_seed{2,4}.json`): original-CIFAR-index partition rebuilt exactly as `data/cifar100.py::get_train_valid_loader` (`np.random.seed(seed)`; shuffle; first 10 % validation); every benchmark train/val row is found in the rebuilt set (45 000 / 5 000, 0 missing). CIFAR-100 contains bit-identical duplicate images with different labels, so row→original-index mapping is ambiguous for those rows (5–7 label disagreements after hash mapping; 44 989–44 991 unique indices); **the benchmark arrays' labels are authoritative.**
* Checkpoint sha256 (best_model.pth): seed1 `ccc2e65a…`, seed2 `bf04fa32…`, seed3 `b9f22464…`, seed4 `5903abd2…`, seed5 `768bbf00…` (full values in `stage0_reconciliation.json`).

## 2. Split roles (default plan verified, not reconstructed)

reference bank = train (45 000) — provides class labels to the class-distance banks; fit = inner-FIT (2 500); clean selection = inner-SELECT (2 500); both from `make_inner_validation_split(select_fraction=0.5, seed=123)` on the seed's validation split (5 000); evaluation = clean test (10 000) and corruption cells. New learned readout parameters and **all** feature standardizers use FIT rows only. Selection uses only clean-selection labels. Evaluation labels never enter fitting or selection. Development seeds (2, 4) use the benchmark-materialized arrays (row order as materialized); confirmation seeds materialize deterministically in original-index-partition order.

**Declared deviation.** Native DAC's own weights are fitted on the *whole* validation split (benchmark practice; development seeds load the frozen pickle, confirmation seeds refit in-pipeline with the same class). It is a calibration reference only and does not enter any residual arm.

**Label-access asymmetry (disclosed).** DG/DS/DL use the 45 000 labelled reference rows; G/S/O do not. Every readout receives the same 2 500 fit labels. Parameter matching does not equalize supervision; the **DL** arm (same labeled bank in logit space) is the control for this.

## 3. Exposure ledger

| checkpoint | exposure before this study |
|---|---|
| seed 4 | legacy Phase 0/1; **FV-DAC pilot** (legacy, 12 cells; k_c=5 selected here); layer pilot (12 cells + clean test); RGC shift/recoverability |
| seed 2 | legacy Phase 0/1; layer pilot (12 cells + clean test); RGC shift/recoverability |
| seeds 1, 3, 5 | legacy Phase 0/1 baseline sweep (12 cells + clean) and RGC shift/recoverability (5 seeds). **Not** used in the layer pilot or in designing this study (layer, pooling, k_c, thresholds came from seeds 2, 4 and the FV-DAC seed-4 pilot). Held-out with respect to this design, **not** unexposed to the benchmark. |

Shared across all checkpoints: the same 10 000 CIFAR-100 test images and CIFAR-C corruptions of them. Validation/train overlap across checkpoints (`results/residual_study/val_train_overlap.json`): a checkpoint's validation images lie in another checkpoint's *training* set ≈ 4 460–4 520 of 5 000 times, and two checkpoints' validation sets share ≈ 480–540 images. New checkpoints are not new test images; confirmation is **held-out-checkpoint / condition replication**, not an untouched independent-data experiment.

The development corruption cells (`gaussian_noise, defocus_blur, fog, jpeg_compression` × severity 1/3/5) are inherited from the pilot and are **development evidence**.

## 4. Stage 1 (existing predictions, no fitting)

From the pilot's stored per-example predictions of the frozen challengers `{vector_scaling, A_greedy_L4, A_depth_L4, B_greedy_L4, B_logit_probe, B_final_repr_probe, A_logit_space}`: W/H/U, intervention rate, `W/total`, `W/(W+H)`, `(W−H)/N`, `W/#base-errors`, candidate correctness given a flip on a base error `W/(W+U)`, flip rate on base errors vs on base-correct examples (detection), runner-up correctness among base errors (from full-precision ranks), evaluation-only oracle union of base with each challenger and with small listed sets, repaired-set overlap and disagreement with the full-logit controls. **Oracle union = an evaluation-only label oracle choosing among already-fixed predictions**: not a deployment method, not a Bayes bound, not headroom over all readouts. No gate and no corruption-labelled probe is trained.

## 5. Stage 2 — evidence sources (one layer, `layer3.22`, exactly the verified post-activation Bottleneck output; no layer search)

Let `h(x)∈R^{1024}` be `layer3.22` GAP → L2 normalization and `g(x)∈R^{4096}` the flattened adaptive 2×2 average pool of the same tensor → L2 normalization (pooling → flatten → L2, the pilot's order). `K=100`.

| ID | raw evidence | `φ_F(x) ∈ R^{100}` (all standardized with FIT-only mean and `√(var+1e-6)`) |
|---|---|---|
| G | `h` | `P_G h`, `P_G ∈ R^{100×1024}`, entries `N(0,1/100)`, seed 20260922; **no** post-projection renormalization |
| S | `g` | `P_S g`, `P_S ∈ R^{100×4096}`, entries `N(0,1/100)`, seed 20260923 |
| DG | `h` vs class banks | `r^{GAP}(x) ∈ R^{100}`: `r_k`=Euclidean distance to the `K_c`-th nearest neighbour among train rows of class `k` (bank = 45 000 GAP vectors) — full vector, unprojected |
| DS | `g` vs class banks | same operator on the 2×2 bank |
| O | logits `z` | `ReLU(B ẑ + c)`, `ẑ` = fit-standardized logits, `B_ij∼N(0,1/100)`, `c_i∼N(0,1)`, seed 20260924 |
| DL | `z` vs class banks | same radii operator on the bank of **centered, L2-normalized train logits**, same labels/`K_c`/bank size |

Audited operator: `Calibrators/full_vector_dac.py::ClassConditionalKNN` + `normalized_euclidean`, true Euclidean. **`K_c = 5`**, inherited from the closed FV-DAC pilot (selected there on clean inner-SELECT from {5,20,200} under the *legacy* protocol; verified in `docs/full_vector_dac_experiment.md` §19.3). No `k` sweep, no centroids/kernels/covariance metric/rank normalization. Maps and seeds are data-independent and not tuned. Before fitting, the run records projection distortion of G and S on 2 000 fixed clean fit pairs, the participation ratio / entropy rank / 90 %-variance components of every `φ`, and the Dasgupta–Gupta sufficient dimension for `n=45 000` (`4 ln n/(ε²/2−ε³/3)`: 9 184 at ε=0.1, 2 473 at ε=0.2, 515 at ε=0.5) — disclosure of the compression, **not** a guarantee. A negative G/S result has the scope of these 100-d compressed readouts.

## 6. Common correction family (exact equations)

**Anchor** `B(z)`, fitted on FIT, selected on SELECT by NLL (ties → earlier candidate). Candidates, in this order: identity (`A=I,b=0`); Vector Scaling (`VectorScaling()` defaults, output taken as `log q`); regularized full matrix scaling
`B(z)=Az+b`, `min mean-NLL + λ(‖A−I‖_F²+‖b‖²)/(2K)`, `λ∈{1e-4,1e-2,1,100}`. Temperature Scaling (`TemperatureScaling()` defaults), Vector Scaling and native DAC are also reported as controls. **The anchor is then frozen.**

**Residual readout** for every `F∈{G,S,DG,DS,O,DL}` and `λ∈{1e-4,1e-2,1,100}`, plus the exact zero candidate (`W=0`, i.e. the anchor):

`t_F(x) = B(z(x)) + W_F φ_F(x)`, `q_F = softmax(t_F)`,
`min_W  mean_NLL(q_F, y) + λ ‖W_c‖_F² / (2K)`, `W ∈ R^{K×100}`, `W_c = W − mean_k W_k` (class rows centered, removing the common softmax offset).

Same FIT rows, feature dimension (100), parameter count (10 000), optimizer (float64 full-batch L-BFGS, strong-Wolfe, ≤ 500 iterations, one deterministic continuation to ≤ 2 000 if `‖∇‖_∞>1e-4`), zero initialization and grid for every source. Non-convergence after the continuation is an *engineering failure* (gate criterion 6), never a scientific null. No inverse-margin/uncertainty gate; no native-DAC sample temperature inside these arms; the grid is not expanded.

This is a **diagnostic residual readout**, more flexible than minimal FV-DAC and not, by itself, a novel method.

## 7. Selection policies (clean-selection data only; no new fits)

From the fitted pool, independently for the **hidden pool** `{G,S,DG,DS}×λ ∪ {zero}` and the **output pool** `{O,DL}×λ ∪ {zero}`:

* **NLL policy**: minimize SELECT NLL (ties: larger λ — so the anchor, λ=∞, wins any exact tie — then fixed family order `G,S,DG,DS,O,DL`).
* **Decision policy**: maximize SELECT accuracy subject to `NLL ≤ NLL(anchor)+0.01`; ties: lower NLL, then larger λ (the anchor counts as λ=∞), then family order. The anchor is always feasible. The 0.01 nats allowance is an empirical selection tolerance, **not** a population-risk certificate. No ECE gate at selection.

The **primary deployed procedure is the hidden-pool Decision policy** (`proc::hidden_decision`); its comparator is the output-pool Decision policy (`proc::output_decision`). NLL policies are diagnostics of objective mismatch. The hidden pool has more families than the output pool: this search asymmetry is reported and **selection is not capacity matching**; fresh-checkpoint confirmation is essential. Family-level Decision/NLL selections (each family + zero) are also stored, for reporting G-vs-S and DG-vs-DS overlap. No target-specific selection. Different checkpoints may select different arms with the same algorithm.

## 8. Development evaluation (seeds 2 and 4; clean test + the 12 inherited cells)

Controls: base, TS, VS, the selected anchor, native DAC (five-source), all 24 fixed arms, the four procedures, exact zero (= anchor). Old pilot arms are context only. Reported, with the definitions fixed here: accuracy (macro over conditions per checkpoint, then over checkpoints; every condition also listed), differences vs base/anchor/VS/output-Decision; NLL, summed multiclass Brier, 15-bin fixed ECE, adaptive ECE, classwise ECE (`utils/unified_metrics.py`); W/H/U and the Stage-1 intervention metrics; base and corrected true-class ranks (computed from float64 probabilities, stored as integers) and corrected top-two margins; intervention rate and signed utility `(W−H)/n` by **base-margin** quintiles and, for the primary procedure, **correction-strength** quintiles (`max_k|t_F−B|`), with **bin edges fixed on clean-selection data**; matched hidden-vs-output paired 2×2 tables and rank changes on identical images (not two unrelated AUCs); G-vs-S and DG-vs-DS repaired-set overlap; bank-query time per 10 000 queries, bank bytes, fit parameters, selected λ. The best corruption row of any table is descriptive only.

## 9. Development gate (frozen; research-budget decisions, not theory)

`proc::hidden_decision` proceeds to confirmation only if **all** hold (checkpoint-averaged unless stated):

1. macro corruption accuracy ≥ base + **0.50 pp**;
2. macro accuracy exceeds each of VS, the anchor and `proc::output_decision` by ≥ **0.25 pp**;
3. its mean advantage over `proc::output_decision` is > 0 in **each** checkpoint and > 0 in ≥ **3 of the 4** corruption families (averaging severities and checkpoints);
4. mean corruption NLL ≤ native DAC + **0.02** and top-label ECE ≤ native DAC + **0.02** (Brier and per-condition deterioration also reported);
5. clean test: accuracy ≥ anchor − **0.20 pp** and NLL ≤ anchor + **0.01**;
6. provenance/numerical checks pass (all fits converged, all cells present and evaluation-only under the corrected protocol, all controls present, Stage-0 reconciliation OK).

If it fails: the study closes, with no new seeds, layers, pooling sizes, `k`, nonlinear classifiers, gates or covariance variants; a suggestive secondary result does not reopen the gate.

## 10. Conditional confirmation (only if §9 passes; frozen algorithm, no changes)

Seeds **1, 3, 5** (checkpoints verified present; exposure ledger §3), clean test + all **15** CIFAR-100-C corruptions × severities 1–5, each checkpoint's procedure fit entirely from clean data with the same maps, grid and policies; no corruption fitting; evaluate the deployed procedures and required controls only. A separate summary is kept for the 11 corruption families **not** used in development. Primary contrast: `proc::hidden_decision − proc::output_decision`, macro over conditions and checkpoints, **two-sided 95 % paired image-level bootstrap** (2 000 resamples of the 10 000 original image IDs, shared by all checkpoints and conditions; corruption copies are not independent observations); seed-specific effects reported. The interval is conditional on these three checkpoints and is not seed-population uncertainty nor selection-adjusted for the research search; secondary intervals are descriptive.

A **strong positive** requires: gate criteria 1, 2, 4, 5, 6 on the confirmation macro; a positive `hidden_decision − output_decision` gain in **every** checkpoint; and a lower bound > 0 of the primary 95 % interval. Even then no new architecture is launched and no novelty is claimed: the deliverable is a concrete second-architecture replication proposal and the narrowest supported mechanism.

## 11. Budget, fit counts, compute

Per checkpoint: 24 residual fits + 4 matrix-scaling anchor fits + TS + VS (+ native DAC refit for confirmation seeds); float64 CPU L-BFGS on 2 500×100 (seconds). Features: one layer at `d=1024` (GAP) and `4096` (2×2); banks in GPU memory 45 000×(1 024+4 096+100)×4 B = 0.94 GB plus the five native-DAC banks (0.7 GB); per-cell arrays ≈ 0.3 GB. Bank-query cost measured per 10 000 queries. Smoke run: ≈ 25 s fit + 6 s/cell. Estimated total: development ≈ 0.5 GPU-h (limit 12), confirmation ≈ 2–3 GPU-h for 3×76 cells (limit 24); ≤ 60 GB RAM per job. Existing benchmark jobs 21532856/21533078 are not touched.

## 12. Code map

`Calibrators/residual_readout.py` (math), `Experiments/residual_evidence_study.py` (runner), `Experiments/residual_stage0_stage1.py`, `Experiments/aggregate_residual_study.py`, `tests/test_residual_evidence_study.py`, `scripts/residual_study_*.sbatch`. File hashes, git HEAD and worktree diff hash are in `results/residual_study/freeze_manifest.json`.

## 13. Decision map (pre-declared reading of outcomes)

S beats G and output controls ⇒ evidence for accessible spatial information in this pipeline. G/S beat DG/DS ⇒ these summaries/readouts are limiting (not that all distance methods fail). DG/DS help where scalar FV-DAC did not ⇒ using the summary was limiting (the classifier remains a known readout construction). DL/O match hidden evidence ⇒ no demonstrated internal-representation advantage. Decision selection helps where NLL selection does not ⇒ objective mismatch supported *in the tested pool*, subject to confirmation. Clean benefit reverses on corruptions ⇒ transfer is limiting. Only NLL/Brier improve ⇒ a probability-estimation result, not decision correction. No meaningful gain after controls ⇒ stop this bounded source-only readout direction at this sample/compression budget; **not** a claim that no information exists anywhere in the network.

<!-- END FROZEN -->

## 16. Results — Stage 0, Stage 1, development Stage 2 (appended 2026-09-21; text above the marker unedited and hash-verified)

**Disclosures.** (a) A 1 500-row / 2-cell smoke run of the runner on seed 4 was inspected for bugs only before freezing; its outputs were deleted. (b) `Experiments/residual_dev_extra_diagnostics.py` (descriptive tables) was written after the freeze; it fits and selects nothing. (c) Frozen-spec hash `bd928cd8ee39d2f6ec6ba68e8b814193be180a721a73d6fc11e58d1f8c46dd3a`; code hashes in `results/residual_study/freeze_manifest.json` (verified unchanged after the run). (d) Frozen-state hashes: seed 2 `8bcde030…1a902`, seed 4 `cd4f2e16…41e0`.

### 16.1 Stage 0 (`results/residual_study/stage0_stage1/stage0_reconciliation.json`)
Job status when queried: corrected benchmark fit `21532856` (tasks 2, 4) RUNNING (~1 h 25 m, still inside its always-on RGCL/GC-DAC stages) and evaluation array `21533078` PENDING on that dependency; nothing relaunched. Corrected benchmark corruption cells therefore do not yet exist; the pilot/study numbers are reconciled to the benchmark on the **clean** cell only (labels identical; argmax agreement 0.9995–0.9998; metric crosscheck ≤ 1e-4 acc / 3.2e-5 NLL / 5.6e-4 ECE; inner-split hashes equal; five-source native DAC in both). This study's own baselines (base, TS, VS, native DAC) are computed in-pipeline from the same checkpoints and frozen benchmark native-DAC pickles.

### 16.2 Stage 1 — candidate headroom over frozen challengers (12 corruption cells × seeds 2, 4; base accuracy 50.55 %)
| challenger | intervention rate | W | H | U | W/total | W/(W+H) | W / base errors | ΔAcc (pp) |
|---|---|---|---|---|---|---|---|---|
| vector_scaling | 8.8 % | 4170 | 3714 | 13257 | 0.197 | 0.529 | 4.2 % | +0.190 |
| A_greedy_L4 | 6.1 % | 2526 | 2344 | 9699 | 0.173 | 0.519 | 2.6 % | +0.076 |
| A_depth_L4 | 1.5 % | 699 | 486 | 2321 | 0.199 | 0.590 | 0.7 % | +0.089 |
| B_greedy_L4 | 14.3 % | 6142 | 5914 | 22333 | 0.179 | 0.509 | 6.4 % | +0.095 |
| B_logit_probe | 12.5 % | 5317 | 5223 | 19359 | 0.178 | 0.504 | 5.3 % | +0.039 |
| B_final_repr_probe | 8.6 % | 3468 | 3627 | 13430 | 0.169 | 0.489 | 3.5 % | −0.066 |
| A_logit_space | 0.8 % | 394 | 285 | 1253 | 0.204 | 0.580 | 0.4 % | +0.045 |

Runner-up (top-2) is correct on 25.7 % of base errors. **Evaluation-only oracle unions** (headroom over base, pp): with all seven challengers +4.66; with the two full-logit controls (VS, logit probe) +2.93; with the three geometric challengers (A_greedy_L4, A_depth_L4, A_logit_space) +1.14; full-logit + geometric +3.39, i.e. the geometric challengers add **+0.46 pp** to the union of the full-logit controls; each single challenger's union with base: VS +1.74, A_greedy_L4 +1.05, B_greedy_L4 +2.56, B_logit_probe +2.22. **Reading (limited):** an oracle choosing per example between these already-fixed predictions could gain at most ≈ 0.46 pp from the geometric challengers beyond the full-logit controls; the large unions (+2.9 to +4.7 pp) mostly reflect that any two ≈50 %-accurate, weakly-agreeing predictors have a large oracle union with ≈ 50 % of their flips being harmful. This rules out a *useful gate over this particular candidate set* at more than ≈ 0.46 pp of geometry-specific headroom; it is not a Bayes bound and does not bear on other readouts. No gate was trained.

### 16.3 Development Stage 2 (seeds 2, 4; 12 cells; ΔAcc in pp vs base; NLL/Brier/ECE = mean over 12 cells then seeds)
Anchor selected: matrix scaling with λ=100 in both seeds. Native DAC (five-source): NLL 2.255, Brier 0.663, ECE 0.132. TS: 2.274 / 0.669 / 0.145. Base: 2.348 / 0.691 / 0.191.

| arm | ΔAcc s2 / s4 / avg | NLL | Brier | ECE | W / H / U | clean ΔAcc |
|---|---|---|---|---|---|---|
| Vector Scaling | +0.078 / +0.178 / +0.128 | 2.478 | 0.700 | 0.206 | 4722 / 4415 / 14690 | +0.22 |
| anchor (MS λ=100) | +0.061 / +0.155 / +0.108 | 2.344 | 0.689 | 0.189 | 1064 / 805 / 2923 | +0.08 |
| G λ=1e-4 / 1e-2 / 1 / 100 | −15.3 / −7.2 / −0.23 / +0.12 (avg) | 7.30 / 3.87 / 2.452 / 2.343 | – | 0.474 / 0.305 / 0.195 / 0.189 | – | −16.6 / −6.4 / +0.03 / +0.05 |
| S λ=1e-4 / 1e-2 / 1 / 100 | −19.4 / −7.5 / −0.17 / +0.12 | 7.75 / 3.69 / 2.435 / 2.344 | – | 0.504 / 0.290 / 0.191 / 0.189 | – | −24.7 / −8.4 / −0.09 / +0.08 |
| DG λ=1e-4 / 1e-2 / 1 / 100 | −11.9 / −4.8 / −0.38 / +0.11 | 6.69 / 3.70 / 2.621 / 2.368 | – | 0.436 / 0.281 / 0.200 / 0.192 | – | −9.6 / −2.9 / +0.17 / +0.10 |
| DS λ=1e-4 / 1e-2 / 1 / 100 | −10.1 / −4.3 / −0.19 / +0.12 | 6.25 / 3.58 / 2.540 / 2.354 | – | 0.426 / 0.287 / 0.202 / 0.191 | – | −9.0 / −3.0 / +0.11 / +0.07 |
| O λ=1e-4 / 1e-2 / 1 / 100 | −6.9 / −4.1 / −0.06 / +0.13 | 9.75 / 4.95 / 2.453 / 2.341 | – | 0.483 / 0.360 / 0.190 / 0.188 | – | −6.5 / −3.5 / +0.20 / +0.10 |
| DL λ=1e-4 / 1e-2 / 1 / 100 | −6.4 / −3.2 / +0.05 / +0.15 | 9.90 / 4.59 / 2.449 / 2.345 | – | 0.472 / 0.321 / 0.196 / 0.189 | – | −6.0 / −2.7 / +0.19 / +0.13 |
| **proc hidden Decision (primary)** | +0.035 / +0.168 / **+0.102** | 2.353 | 0.690 | 0.191 | 1291 / 1047 / 4021 | +0.085 |
| proc hidden NLL | +0.035 / −0.029 / +0.003 | 2.415 | 0.694 | 0.194 | 3248 / 3241 / 12359 | +0.135 |
| proc output Decision (= output NLL) | +0.078 / +0.157 / **+0.118** | 2.389 | 0.691 | 0.194 | 3440 / 3158 / 11282 | +0.285 |

(Per-arm W/H/U, Brier and adaptive/classwise ECE for every fixed arm: `results/residual_study/dev/aggregate/arm_table.csv`; every condition: `…/checkpoint_seed*/<cell>/cell_metrics.json`.) All 28 fits per seed converged (no continuation needed beyond what is recorded). Selected procedures: seed 2 hidden Decision = hidden NLL = `DS_l100`, output Decision = output NLL = `DL_l100`; seed 4 hidden Decision `S_l100`, hidden NLL `DS_l1`, output (both) `DL_l1`. **Every family-level selection except seed-4 DS-NLL/DL/O-NLL is λ=100, the top of the grid, with ‖W‖_F ≈ 0.1–0.2** (an almost-zero correction); λ=1 arms already lose 0.1–0.5 pp and λ ≤ 1e-2 destroy accuracy (fit accuracy 100 %, 10 000 parameters on 2 500 rows).

**Mechanical gate (§9) — FAIL.** (1) macro gain over base **+0.102 pp** (needs ≥ 0.50): fail. (2) margins over VS / anchor / output-Decision **−0.026 / −0.006 / −0.016 pp** (need ≥ 0.25): fail. (3) advantage over output-Decision per checkpoint **−0.043 (seed 2) / +0.012 (seed 4)** pp; per family: gaussian_noise −0.098, defocus_blur +0.143, fog −0.018, jpeg_compression −0.090 (1 of 4 positive): fail. (4) NLL **+0.098** and ECE **+0.059** worse than native DAC (Brier +0.028; NLL worse by > 0.02 in 13 of 24 conditions, Brier worse in 24 of 24): fail. (5) clean vs anchor: accuracy +0.01 pp, NLL −0.0016: pass. (6) provenance/numerics: pass. **Failing: 1, 2, 3, 4.** Under §9 the study closes; no confirmation was launched.

### 16.4 Diagnostics for the theory (`aggregate/extra_diagnostics.json`)
* **Selected regularization at the grid boundary:** λ=100 (‖W‖_F 0.1–0.2 of a 10 000-parameter map). Fit-vs-select NLL for λ=1 arms: fit 0.38–0.71, select 0.85–0.89 (gap 0.15–0.5 nats); for λ=100: fit ≈ 0.86, select ≈ 0.84–0.87 (no gap, no gain). Clean-selection NLL of hidden vs output evidence at λ=100 is indistinguishable (seed 2: G/S/DG/DS 0.840–0.842, O/DL 0.839–0.840, anchor 0.8435).
* **Retained rank** (participation ratio of φ on fit rows, of 100): G 38–39, S 59–61, DG 2.1–2.2, DS 2.8–2.9, O 30, DL 10. Radius vectors are dominated by one shared direction (DG/DS top-1 variance fraction ≈ 0.57 in the smoke run; participation ratio ≈ 2–3), i.e. the standardized per-class radii carry little independent variation.
* **Projection distortion** (2 000 fixed clean fit pairs, ‖P(u−v)‖/‖u−v‖): G mean 0.996–0.997 (sd 0.07), S 0.984–1.004 (sd 0.07); 95th-percentile relative squared-distance error 0.26–0.29. JL sufficient dimensions for n=45 000: 9 184 (ε=0.1), 2 473 (0.2), 515 (0.5) ≫ 100: the bound certifies nothing here; no guarantee is claimed.
* **Interventions are concentrated in the lowest base-margin quintile** (primary procedure: intervention rate 10.2 % / 5.1 % in the lowest quintile for seed 2 / 4, exactly 0 in the other four) — a consequence of the strongly regularized correction; signed utility there +10 / +46 per 10⁴ examples. Correction-strength quintiles: seed 2 utility per 10⁴ +12, +6, +15, +3, **−7** (top strength quintile negative); seed 4 +21, +19, +7, +12, +23.
* **Mean true-class rank over the 12 cells** (base 9.055 / 8.545 for seeds 2 / 4): hidden Decision 9.042 / 8.504; output Decision 9.032 / 8.993; hidden **NLL** policy seed 4 **9.518** (worse than base) — the NLL-selected λ=1 arm damages ranking while the Decision-selected λ=100 arm does not. Matched hidden-vs-output (same images): seed 2 only-hidden-correct 262 vs only-output 314; seed 4 2158 vs 2144 (differences of order the count noise).
* **G vs S / DG vs DS repaired-set overlap** (family Decision-selected arms, corruption cells pooled, seed 2): G∩S = 555 of 636 (Jaccard 0.87); DG∩DS = 641 of 795 (0.81); each family repairs 586–823 examples; S-only-vs-G ≈ 30, DS-only-vs-DG ≈ 70 — spatial evidence repairs almost nothing that GAP evidence missed. (Seed 4's DL family is an outlier: DL selects the λ=1 arm with 2 617 repairs and 2 774-example union.)
* **Cost**: bank query per 10 000 queries — DG 0.06 s, DS 0.12 s, DL 0.05 s on an RTX 4090; banks 184 MB (GAP) + 737 MB (2×2) + 18 MB (logit); 10 000 fitted parameters per readout; all fits < 30 s.
* **Source→corruption signed utility change:** clean-test ΔAcc vs base of the primary procedure +0.10 / +0.07 pp; corruption +0.035 / +0.168 pp; the outputs-only Decision control on clean +0.12 / +0.45 pp and on corruption +0.078 / +0.157 pp. Family heterogeneity of the primary (pp): gaussian_noise −0.055, defocus_blur +0.152, fog +0.190, jpeg +0.120; by severity 1/3/5: +0.130 / +0.088 / +0.087.

### 16.5 Mechanical decision and interpretation
**Decision: STOP.** The development gate failed on criteria 1, 2, 3 and 4; no new seeds, layers, pooling sizes, `k`, nonlinear classifiers, gates or covariance variants were tried. Provenance and numerical checks passed, so this is a scientific null for the tested combination, not an engineering failure and not resource-blocked.

* **Weakened:** that a source-fitted 100-d residual readout of `layer3.22` (GAP, 2×2, class-radius, or logit-space evidence) adds accessible decision information beyond a matrix-scaling anchor on ≈2 500 fit rows. Hidden evidence did not beat output evidence (+0.102 vs +0.118 pp), spatial did not beat GAP, and radius summaries did not beat pooled features; DL/O ≥ hidden evidence in this pool. The earlier +0.3–0.44 pp exploratory 2×2 signal under the *additive-β* correction did not reproduce as an accuracy advantage under a common residual readout (S_λ100 +0.117 pp ≈ G_λ100 +0.117 pp ≈ anchor +0.108 pp).
* **Supported (tested pool, two checkpoints):** objective mismatch in one of two checkpoints — the NLL policy picked a λ=1 hidden arm that lowered mean corruption accuracy and true-class rank while the Decision policy did not — but this is not confirmed and the "helped" arm is essentially the anchor (net +16 flips vs the anchor).
* **Unidentified:** whether information exists that a readout with more than 2 500 clean fit labels (or another parametrization) could use; the dominant observed limit is **finite-sample learnability under the clean fit budget** (clean selection prefers a near-zero correction; unregularized fits interpolate the 2 500 rows), not a demonstrated absence of information in the network. Any transfer effect is secondary because the clean-selection signal itself is ≈ zero. The old scalar-distance claim is unaffected.
* **Calibration:** criterion 4 is structurally hard for every anchor-based arm — the strong output-only anchor keeps the base's shift calibration (NLL 2.344, ECE 0.189 vs native DAC 2.255 / 0.132) because native DAC's sample-dependent density temperature was deliberately excluded (§6). Criteria 1–3 fail on accuracy alone.

### 16.6 Addendum (2026-09-21, after §16.1 was written)
* **Job status update.** Benchmark fit array `21532856` finished (`COMPLETED`, tasks 2 / 4 in 1 h 34 m / 1 h 28 m); evaluation array `21533078` has started (tasks 0–5 RUNNING, remainder PENDING). The corrected benchmark **corruption** cells are still being produced; `Experiments/crosscheck_residual_vs_benchmark.py` reconciles them cell by cell when they exist (`results/residual_study/stage0_stage1/corruption_reconciliation.json`; currently only the two clean cells).
* **Clean reconciliation of this study's in-pipeline baselines** vs the benchmark: base, native DAC, TS agree to ≤ 3e-5 NLL, ≤ 6e-4 ECE, 0 accuracy difference, argmax agreement ≥ 0.9996. **Vector Scaling differs by design**: the benchmark fits TS/VS on the *whole* 5 000-row validation split (`fit_inputs=logits_val`), whereas this study (compatible roles) refits them on the 2 500 FIT rows only — ΔNLL +0.017 / +0.023, argmax agreement 0.976 / 0.973, accuracy ≤ 0.22 pp apart on clean test. Consequently the layer pilot's TS/VS baselines (frozen benchmark pickles) saw the pilot's inner-SELECT rows — a role incompatibility for the *baselines* only (evaluation labels were not touched); it favours those baselines slightly and does not change the pilot's verdict (its candidates failed by ≥ 0.4 pp).
* **IJCAI-era provenance trace (cheap; no claim about publications).** In the canonical RGC repository, `Experiments/run_post_hoc_calibration.py:1120` builds the CIFAR-100 clean test loader with `Data.cifar100.get_test_loader` (ImageNet statistics; the file is byte-identical to this repository's legacy version, dated 2026-01-07). Stored artifacts `calibration_comparison/ablation_baseline_cross_entropy_cifar100_resnet101_seed{2,4}.json` (2026-01-11) record **uncalibrated accuracy 76.20 % (seed 2) and 76.31 % (seed 4)** and seed-4 ECE 0.04895; the paired diagnostic gives legacy (ImageNet-statistics) 76.20 % / 76.32 % and ECE₁₅ 0.04886, versus **corrected 76.59 % / 76.51 %**. So those artifacts were produced with the *legacy* test normalization. **Not established:** which artifacts/versions the published IJCAI tables were built from (no commit/log linking them was found). Publication claims were not modified and no one was contacted.
