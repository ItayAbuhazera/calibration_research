# Fixed deep-layer candidate, gate information and effective sample size — frozen specification

Status: development experiment (2026-09-21). Checkpoints 2 and 4 only; existing clean roles; the exact 12 exposed corruption cells.
This document is frozen (hash recorded in `docs/fixed_gate_study_spec.frozen.sha256` and `results/fixed_gate/freeze_manifest.json`)
BEFORE any gate of this study is fitted or any per-example outcome of the fixed deep candidate is aggregated by this study's code.
Text below `<!-- END FROZEN -->` is appended results/interpretation only and is not covered by the hash.

## 0. Honest exposure statement

* The atlas (`docs/atlas_program_spec.md` §13) already showed, on the 2 000-image atlas subset and on the clean fit/selection/calibration rows,
  that deep-layer3 spatial candidates repair more base errors than layer4 candidates but also harm many base-correct examples.
  Choosing "layer3.22 + 2×2" for THIS study is therefore development-informed by outcomes already seen. It is NOT a clean-selected
  candidate under the atlas rule (its clean-selection net utility is about −0.16, far below the layer4 winners).
* Provenance of `layer3.22` (verified in `docs/layer_selection_pilot_spec.md` §13): NLL-greedy L=1 pick of the layer pilot, both seeds, for a
  GAP additive-β readout — a different objective and readout than kNN gating. It is a provenance link, not an endorsement for this readout.
* Parameter fitting in this study uses clean-fit rows only; the design (which candidate, which comparison) is development-informed.
* Roles have historical exposure (inner-FIT/SELECT of earlier studies; the atlas split of the old SELECT role into selection/calibration).
  This is not a new or pristine dataset. Nothing here is confirmation evidence and no earlier verdict or threshold is changed.

## 1. Question

The atlas selected candidates by standalone E[D]; a gated system realizes E[g(X)D], D = 1{j=Y} − 1{i=Y}. Question: did standalone selection
discard a selectively deployable alternative, or is useful intervention information inaccessible from the allowed evidence? We do not assume
lack of clean supervision; we test the fixed candidate at the existing budget (n = 2 500 clean-fit rows) and at nested smaller prefixes.

## 2. Fixed candidate (exactly one; no other pooling, k, metric, projection or covariance transform)

`layer3.22` post-activation Bottleneck output (1024×8×8) → `adaptive_avg_pool2d(2)` → flatten (4096-d, no projection) → L2 normalization →
exact Euclidean kNN, k = 50, against the same 45 000-row labelled clean training bank (unit-normalized), deterministic (distance, index) ordering
(`atlas/knn.py`, TF32 off). p_geo(c|x) = (n_c(x) + π_c)/(50 + 1), π = source-bank class prior; j(x) = argmax_c p_geo (lowest index on ties).
Implemented by `atlas.stage_b --mode F --site layer3.22 --pool grid2 --metric unit_l2` (unchanged atlas code), written to `results/fixed_gate/seed{2,4}/knn_F/`.
Validation check: the recomputed neighbours must agree with the atlas `knn_unit/layer3.22__grid2.npz` neighbours on the 31 000 atlas queries
(reported, not asserted equal to the last bit).

Controls (frozen from EXISTING atlas clean-selection manifests, not reselected):
* layer4 GAP: seed 2 `layer4.1` unit_l2, seed 4 `layer4.2` raw_l2 (`results/atlas/seed*/shortlist.json`, `metric_selection.json`).
* output-space: logit-space kNN, Mahalanobis metric in both seeds (`metric_selection.json` → `output`).
They receive the same four gate families at n = 2 500 only. They are controls, not winners to promote.

## 3. Data roles (atlas roles, verified against `results/atlas/seed*/roles.json`)

reference bank = 45 000 training rows (labels used to form p_geo only); fit = 2 500 rows; selection = 1 250; calibration = 1 250 (stratified halves of the
old 2 500-row SELECT). Evaluation: all 10 000 clean test images and the same 12 development cells (gaussian_noise, defocus_blur, fog,
jpeg_compression × severities 1, 3, 5), matched original image IDs across methods. No reference-bank row is used for gate fitting; supervision is not increased.
Base logits for EVERY method in this study come from one artifact: `results/atlas/seed*/u0/*.npz` (FP32, TF32 off, batch 250). Predictions from other precision
pipelines are never mixed in (§9 reconciles them separately).

## 4. Gate: four prespecified feature families (ridge regression of D on disagreements)

For base class i = argmax z (base logits z, 100 classes), candidate j, p = softmax(z) (original model probabilities), disagreement iff j ≠ i.

* C0 (5): z_i − z_j; top-two logit margin z_(1) − z_(2); p_i; p_j; entropy H(p).
* C1 (9): C0 + [p_geo(j) − p_geo(i); entropy of p_geo; k-th neighbour radius (distance to the 50th neighbour in the candidate's metric); log(max(pre-L2 pooled norm, 1e-12))].
  For the output candidate the "pooled norm" is the norm of the centred logit vector (inherited atlas convention).
* Z0 (105): C0 + the full centred logit vector z − mean(z) (100).
* Z1 (109): Z0 + the same four geometric features as C1.

No labels, correctness flags, corruption identity, severity or test-dependent normalization enters a gate. No class one-hots, no hidden coordinates.
Standardization: mean/std (population, + 1e-6) of the FIT SUBSET's disagreement rows only; features with std < 1e-8 on that subset are recorded as constant and
become identically 0 after standardization.
Fit: minimize mean_r (D_r − b − w·F_r)² + λ‖w‖², r over that subset's disagreement rows, b unpenalized (= mean D of those rows), closed form
w = (XᵀX + λ m I)⁻¹ Xᵀ(D − b), m = number of disagreement rows; λ ∈ {0.01, 1, 100}. Predictions clipped to [−1, 1]. g = 1{j ≠ i and D̂ > θ}; g = 0 when i = j.
θ ∈ {0, 0.02, 0.05, 0.10, 0.20}. Natural W/H/U prevalence retained; nothing dropped or reweighted. Fewer than 5 disagreements → degenerate (never-intervene only).
Selection on the 1 250 clean-selection rows: maximize Σ gD / n_selection over {never} ∪ {λ}×{θ} (full-population denominator, unchanged examples included). Ties:
fewer interventions, then larger λ, then fixed order (λ ascending index, θ ascending). `never` counts as λ = ∞.
Separately reported controls, NOT selectable: never-intervene (≡ native DAC probabilities/base argmax) and always-intervene-on-disagreement (g = 1{j ≠ i}) and candidate-alone.
Effective design rank (rank of [1 | standardized X] on the disagreement rows) and ridge effective degrees of freedom tr(X(XᵀX + λ m I)⁻¹Xᵀ) are reported for every fit.

Primary gate-information contrast (fixed): Z1 minus Z0, fixed deep candidate, n = 2 500, reported per checkpoint. C0/C1 and the layer4/output controls are diagnostics.
The primary is never chosen by corruption performance. C1 > C0 alone is insufficient to claim an advantage over full-logit evidence.

## 5. Learning curve without new labels

Deep candidate only. Three fixed class-stratified nested orders (order seeds 20261001, 20261002, 20261003): within each class, fit rows are shuffled with the
order seed; rows are then emitted by rounds (round r takes the r-th shuffled row of each class, classes visited in a per-round permutation from the same generator).
Prefixes n ∈ {625, 1250, 2500}; the n = 2 500 prefix is the same row SET for every order and is fitted once (not counted as replication). The same nested rows are used for
all four families; selection/calibration rows are fixed. Fit and standardize independently per prefix. Record n, m (disagreements), W/H/U, effective rank/df.
Orders are a sensitivity analysis, not independent datasets. Layer4 and output controls: n = 2 500 only.
Reported for each (family, order, n): train MSE of D̂ vs D on the fit-subset disagreements; held-out MSE on the calibration-row disagreements (clean-generalization set, evaluated
BEFORE those rows' labels are used for temperature fitting and from frozen gates only) and on clean-test disagreements; each MSE alongside the constant-mean predictor's MSE;
coverage, rescues (W among interventions), harms (H), net gain, against both n and m. No gate or n is selected from these evaluations.

## 6. Freeze, then evaluate

Stage order enforced in code: (1) extraction; (2) fit + select → `gates_frozen.json` (sha256 recorded) using fit/selection rows and labels only; (3) evaluation reads the
frozen file and refuses to run without it: calibration rows first (generalization diagnostics), then temperatures, then clean test and the 12 cells.
Pipelines evaluated per checkpoint: base; native DAC (FIT-refit, atlas `u0`); TS, VS, matrix scaling (refit on FIT rows; matrix-scaling λ chosen on selection rows, as in the atlas);
candidate-alone (deep, layer4, output); never; always; C0, C1, Z0, Z1 for the deep candidate (all prefixes/orders) and for layer4 and output candidates (n = 2 500).
Probabilities: q_raw = q_DAC where g = 0 and p_geo where g = 1 (atlas construction); the exact argmax of q_raw must equal the declared action (i if g = 0 else j) — counted and reported.
Final scalar temperature: q_T = softmax(log(clip(q_raw, 1e-12, 1)) / T), T > 0 by 1-D bounded search (log T ∈ [log 0.05, log 20]) minimizing NLL on the 1 250 calibration rows only,
fitted after decisions are frozen, for every probability pipeline including controls. Both raw and post-T are reported for both clean and corruption; we never report only the better one.
Primary probability metrics: NLL and sum-convention multiclass Brier. ECE secondary. Ground-truth class ranks are computed on unrounded float64 probabilities before any storage cast.

Per seed / split / cell: accuracy difference vs base (pp), W, H, U, coverage (interventions/N), W/(W+H+U), W/(W+H), rescued fraction W/base-errors, global harmful-intervention
probability H/N, net utility (W−H)/N, W−H identity check, NLL, Brier, mean GT rank.
Utility-bin diagnostic: bins = quintile edges of D̂ over the clean-SELECTION disagreement rows of the primary (deep Z1, n = 2 500) gate; on clean test and per corruption family
(pooled over severities) report the signed utility mean and W/H/U proportions per bin; flag source-to-target sign reversals. Descriptive only.
Uncertainty: paired image-level bootstrap (2 000 resamples, seed 20261010): resample the 10 000 original image IDs, keeping all conditions and all method predictions of an image together;
statistic = macro-mean over the 12 cells (and clean separately) of the accuracy / NLL difference. Per checkpoint; not IID over cells; not selection-adjusted; not a population statement.

## 7. Pre-declared reading of outcomes (practical, not formal tests)

Practical targets from earlier studies (unchanged): +0.5 pp over base and +0.25 pp over the strongest output control on the 12-cell macro, without worse NLL/Brier.
Missing them is not a refutation of any signal. Outcome cases (Brief §9): (1) Z1 > Z0 held-out clean utility and consistent transfer; (2) clean improvement reversing under corruption;
(3) output-evidence gates equal on the deep candidate; (4) compact helps but full-logit matches; (5) neither yields useful held-out intervention; (6) growing benefit with m across orders.
"Consistent" = same sign and paired-bootstrap interval excluding zero in BOTH checkpoints for the held-out clean net gain; anything less is reported as uncertain.

## 8. Not authorized / out of scope

Fresh checkpoints, seeds 1/3/5, other corruption families or architectures, other layers/pooling/k/metrics, nonlinear gates, conformal wrappers, TTA, moving bank rows into gate fitting,
commits/pushes, contacting anyone, editing `06_Ideas/`, changing publication claims.

## 9. Numerical and latency audit (separate from the science)

* Mismatch audit: for every benchmark condition with outputs, identify image IDs where atlas-FP32 argmax ≠ benchmark argmax; on those IDs plus a seeded random 1 000 control images per condition,
  rerun the same checkpoint in (a) FP32 strict batch 250 (atlas path), (b) FP32 strict batch 256, (c) TF32 conv+matmul batch 256, (d) fp16 autocast batch 256 (the benchmark's `predict_proba` path uses
  `torch.amp.autocast("cuda")`), and compare each with the benchmark probabilities. A precision explanation is accepted only if a matching run reproduces the benchmark argmax.
* Latency: the fixed configuration, backbone-only vs full pipeline measured interleaved in the same process, resident GPU bank, synchronized timing, warm-up, ≥ 200 (batch 1) / ≥ 50 (batch 256)
  repetitions, median and p95; hook-only overhead measured separately; no optimization is added.

<!-- END FROZEN -->
