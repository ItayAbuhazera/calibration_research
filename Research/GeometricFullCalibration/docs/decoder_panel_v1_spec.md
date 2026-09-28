# Decoder Panel v1 — frozen measurement protocol (specification)

Status: **FROZEN v1 (2026-09-28)** — immutable except for a formally documented bug amendment; sidecar `decoder_panel_v1_spec.frozen.sha256`.
Frozen after the engineering audit (synthetic targets only) and before any scientific Decoder-Panel output existed.
Implementation: `atlas/decoder_panel.py` (`PANEL_VERSION = "decoder_panel_v1"`, `SPACE_VERSION = "dp1-space-1"`). Tests:
`tests/test_decoder_panel.py`. Durable journal: `docs/decoder_panel_v1_master_report_2026-09-28.md`. Resource plan:
`docs/decoder_panel_v1_resource_plan.md`.

## 0. Purpose and scope of claims

Decoder Panel v1 is the **default measurement instrument** whenever a claim of the form "evidence channel E adds / does not add usable
evidence beyond Z" is made in this program. It exists because two completed experiments showed that the decoder regime can be as large
as the scientific contrast: G1 (equivalent-span parameterization sensitivity +0.72 / +0.50 pp > the +0.24 pp conditional increment) and
N1a (77 / 80 selector fits at the weak-regularization grid edge).

It is a **sensitivity instrument, not a search**. It never establishes information-theoretic absence. Vocabulary (fixed):

- **prediction content** — statistical information in E relevant to Y (not measurable by any finite panel);
- **accessibility** — whether a specified decoder family exploits E under a finite sample / compute / supervision regime;
- **panel-robust accessibility** — an effect present across multiple preregistered, structurally distinct families;
- **decoder-specific accessibility** — an effect observable only under a narrow family;
- **action information** — information about the action advantage Δ_a or which action has higher utility.

Strongest permitted negative statement: *"No practically meaningful usable increment was detected across the preregistered Decoder
Panel v1 under the tested sample, compute and supervision regime."* Never: "there is no information in H".

## 1. The six families (exactly six; closed set)

| id | family | model | preprocessing (fitted on the rows being fitted only) | selection |
|---|---|---|---|---|
| D1 | linear | L2-penalized linear predictor: (anchored) multinomial / binary logistic, or ridge for scalar regression | per-coordinate standardization | exhaustive λ grid |
| D2 | poly2 | features [x̃, std(TS₂(x̃))] → L2 linear head; TS₂ = `PolynomialCountSketch(degree=2, gamma=1, coef0=0)`; **degree fixed at 2** (linear terms explicit, second-order interactions sketched; no O(d²) materialization) | x̃ = standardized x; sketch outputs standardized | exhaustive (sketch dim × λ) grid |
| D3 | lgbm | LightGBM gradient-boosted trees on **raw coordinates** (no PCA, no standardization; axis-aligned splits are the family) | none | Optuna TPE |
| D4 | mlp | `Linear(d, w) → GELU → Dropout(p) → Linear(w, K)`, **exactly one hidden layer**, AdamW, batch 256, output layer zero-initialized | per-coordinate standardization | Optuna TPE |
| D5 | knn | standardization → `SparseRandomProjection` to the JL dimension `johnson_lindenstrauss_min_dim(n_fit, eps=0.5)` **only if** d exceeds it → brute-force Euclidean kNN; class scores log((n_c + 1)/(k + K)) (Laplace α = 1); anchored classification: logits = offset + β·score | standardization; projection (fixed seed) | exhaustive (k × weighting × β) grid |
| D6 | rff | standardization → `RBFSampler` (random Fourier features of the RBF kernel, 1024 components) → standardization → L2 linear head; γ = multiplier × γ_med, γ_med = 1 / median‖x_i − x_j‖² on a fixed 500-row sample of the (standardized) fit rows (label-free) | as stated | exhaustive (γ multiplier × λ) grid |

Why these six (structural distinctness): D1 global linear; D2 global second-order polynomial; D3 axis-aligned piecewise-constant
partitions; D4 learned smooth compression; D5 local neighbourhood geometry; D6 smooth stationary-kernel geometry.

**Prohibited after freezing:** adding a seventh family; replacing a family (e.g. LightGBM → CatBoost, kNN → metric learning, RBF → other
kernel); raising polynomial degree; adding MLP layers; adding Optuna trials, widening grids or changing spaces because a result is
negative. A Decoder Panel v2 requires an independent methodological justification (not motivated by rescuing a result), is frozen before
use and is applied prospectively only.

## 2. Tasks and anchoring

- `classification` (K ≥ 2; binary = K = 2) — optionally **anchored**: logits = offset + g(x) where offset is a fixed per-row score
  (e.g. native logits Z). D1/D2/D6: offset enters the softmax (as `atlas.followup_anchored`); D3: offset = LightGBM `init_score`, raw
  boosted score added to it; D4: output layer zero-initialized, network output added to offset; D5: offset + β·log p_knn.
- `regression` (scalar, squared loss): D1/D2/D6 closed-form ridge (mean squared error + λ‖w‖²), D3 L2 regression, D4 MSE, D5 weighted
  neighbour mean.
- action selection = classification over action-advantage classes with a caller-supplied decision objective (N1a-DP).
- **Anchor-only candidate:** in anchored tasks the candidate g = 0 (logits = offset) is always part of the selection set for every family
  (the analogue of G1's f = 0); ties go to g = 0.

## 3. Frozen grids / search spaces (identical for every evidence arm, base model, fold and study)

- **λ grid (D1, D2, D6):** {1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7}; objective mean loss + λ‖W‖²_F (bias unpenalized; the
  repository convention, no ½); float64 L-BFGS (strong Wolfe, history 20, tol_grad 1e-8, tol_change 1e-11, max_iter 2000, one retry at
  6000); zero init at the strongest λ, then warm-started strong → weak (strictly convex: warm starts change numerics only); refit re-traces
  the same path up to the selected λ.
- **D2 sketch dims:** {512, 1024, 2048}.
- **D6 γ multipliers:** {0.25, 0.5, 1, 2, 4} × γ_med; 1024 components.
- **D5:** k ∈ {5, 15, 30, 50, 100}; weighting ∈ {uniform, distance}; β ∈ {0.25, 0.5, 1, 2, 4} (anchored classification only; otherwise
  β = 1); JL eps = 0.5; label-free distortion audit ‖Rx_i − Rx_j‖/‖x_i − x_j‖ on a fixed 500-row sample of the fit rows (median, 5th and
  95th percentile, min, max) stored for every projected fit.
- **D3 LightGBM (Optuna TPE):** num_leaves int [4, 64] log; max_depth ∈ {−1, 4, 6, 8}; min_data_in_leaf int [10, 200] log;
  learning_rate [0.03, 0.3] log; feature_fraction [0.1, 1.0]; lambda_l1 [1e-8, 10] log; lambda_l2 [1e-8, 10] log; **max_delta_step = 2.0 (fixed)**; ≤ 1000 rounds; early
  stopping 50 rounds on the inner-validation rows (native multi_logloss / binary_logloss / l2); refit on all training rows with
  round(mean best iteration over inner splits) rounds; deterministic mode, force_col_wise, num_threads = cpus-per-task.
- **D4 MLP (Optuna TPE):** width ∈ {64, 128, 256}; lr [1e-4, 3e-2] log; weight_decay [1e-6, 1e-1] log; dropout ∈ {0, 0.1, 0.2};
  ≤ 200 epochs; early stopping on inner-validation loss (patience 20; best-epoch weights); refit on all training rows for
  round(mean best epoch) epochs. Hidden-layer count is not a hyperparameter.
- **Optuna budget:** N_TRIALS = **50** trials per study for both D3 and D4 (fixed by the Stage-B convergence audit, §10), TPESampler with a derived
  seed, n_jobs = 1, in-memory storage. One sampler seed per study; never several seeds with best-of selection.

## 4. Nested selection (all families)

For every outer training set: inner split(s) → evaluate every grid configuration / Optuna trial on every inner split → objective =
mean over inner splits → select → **refit on all permitted outer-training rows** → evaluate **once** on outer-test rows. Every transform
(scaler, projection, sketch, RFF γ, early stopping) is fitted inside the fit on the rows it is given and stored on the fitted object; the
outer-test rows and labels are never passed to `fit_decoder`. Outer-test labels may not influence scaling, projection, γ, early
stopping, hyperparameter choice, architecture or budget.

## 5. HPO fairness

Within a family, every evidence arm, base model and outer fold receives the identical search-space definition, trial/grid budget, inner
split construction, selection objective and seed protocol. **Selected hyperparameters may differ** across arms, base models and folds
(fairness = same search opportunity). Scientific contrasts are always within family: Δ_m = V_m(Z, H) − V_m(Z). Never MLP(Z,H) vs
Linear(Z). No base model receives extra trials.

## 6. Seeds

Master seed 20260928. Study seed = first 31 bits of sha256("20260928|experiment|sorted key=value ids|family"). Trial seeds (LightGBM,
MLP init/shuffle) derive from (study seed, trial number, inner split); refit seed from (study seed, "refit"). Projection / sketch / RFF
seeds are fixed per family (20260928 + 11 / 13 / 17) and identical across arms and folds. Every study persists: study id, sampler seed,
space version, all trial parameters and per-split objectives, best iterations, selected configuration, runtimes.

## 7. HPO regimes (two, mutually exclusive, enforced by `HPOContext.validate`)

- **`target_pooled`** (G1-DP): target-supervised accessibility diagnostic; exactly one image-grouped inner split of the target-labelled
  training rows; the unit preserves the original experiment's fitting unit. Label everywhere: **TARGET-CELL/TARGET-POOLED HPO —
  ACCESSIBILITY DIAGNOSTIC ONLY**; never interpreted as deployment, source-trained generalization or unseen-shift transfer.
- **`shift_transfer`** (N1a-DP and any future held-out-environment study): the outer held-out environment (e.g. corruption family, all
  severities) never enters HPO, early stopping, preprocessing, thresholds or refit; inner model selection = leave-one-TRAINING-environment-
  out (each allowed training environment held out once; objective = macro over these inner environments); inner fit/val rows image-
  disjoint; refit on all allowed training environments; evaluate once on the held-out environment. No per-severity tuning (severity
  heterogeneity is a result). `validate()` refuses a context that mixes the two regimes.

## 8. Selection objectives

The objective must match the scientific task and is frozen in each experiment's spec: multiclass prediction → inner-validation NLL;
action selection → a decision estimand (realized policy utility / negative route regret), never AUROC.

## 9. Required tests (all in `tests/test_decoder_panel.py`)

Training-rows-only preprocessing (all six families, sentinel evaluation rows); objective receives inner-validation rows only; G1-DP fit
never sees outer-test labels; deterministic inner splits; deterministic study seeds and reproducible Optuna studies; identical spaces and
budgets across matched arms; sketch / JL / RFF reproducibility; JL target and distortion audit; LightGBM on raw coordinates; MLP exactly one
hidden layer; RFF γ training-only; held-out N1a-DP family never in selection and image disjointness (inner and outer); Z_other excluded from
N1a-DP evidence; HPO regimes cannot be mixed; unique output paths; restart skips only checksum-validated outputs; G1-DP arms preserve
G1 meanings and shuffle rule; regression/binary support.

## 10. Engineering audit (Stage B; synthetic teacher targets, no scientific labels)

Full numbers: `docs/decoder_panel_v1_resource_plan.md`. Summary: (i) Optuna budget rule — raise 30 → 50 if any audited study still
improved > 0.25 % (relative) from 20 → 30 trials; the G1-scale MLP did (0.37 % CPU, 1.47 % GPU), so **N_TRIALS = 50** for D3 and D4.
(ii) Anchored LightGBM exploded without a leaf cap (Newton steps under saturated softmax anchors) → `max_delta_step = 2.0` is part of D3.
(iii) MLP CPU vs GPU: GPU 2–7× faster per study but CPU ≤ 19 min per study with far more free slots → **D4 runs on CPU**.
(iv) kNN uses an exact blockwise NumPy search (sklearn's threaded brute force returned out-of-range indices under mixed OpenMP runtimes).
(v) JL audit on G1 arms: target dim 418, distortion 5th/95th pct 0.94/1.06. (vi) λ grid, sketch dims {512, 1024, 2048}, 1024 RFF
components and the kNN grid are feasible as specified (largest single study ≈ 3–6 h on 8 cores); nothing was reduced for cost.

## 11. SLURM execution design

One Slurm array task = one experiment unit (all HPO sequential inside it; Optuna n_jobs = 1; no per-trial jobs; no shared Optuna
database; in-memory studies persisted as compact JSON). Separate arrays per decoder family (different resource classes); array index →
unit mapping is deterministic and written to a manifest before submission. OMP/MKL/OPENBLAS/NUMEXPR threads = LightGBM num_threads =
cpus-per-task; sklearn n_jobs ≤ cpus-per-task. Heavy inputs are staged to node-local `/tmp/dp_<job>_<task>` (this cluster has no
`$SLURM_TMPDIR`), verified by sha256, removed on exit; only results/logs are written to the persistent tree. Outputs are written
atomically with a `.done.json` completion marker holding sha256 of every output; reruns skip only validated units; every array index is
independently restartable. Aggregators refuse incomplete arrays. Per-family resource classes, array granularity (G1-DP: one task per (checkpoint, fold, arm) for linear/poly2/rff/lgbm,
one task per (checkpoint, fold) for mlp/knn; N1a-DP: one task per (base, held-out family, fold)), concurrency (%20 heavy / %10 light) and
the core-hour estimate are in the resource plan; time limits ≤ 24 h (multi-day limits blocked backfill on this cluster).

## 12. Interpretation rules (panel-level; each experiment spec instantiates numbers)

- Always report the full decoder-family vector (Δ_linear, Δ_poly2, Δ_lgbm, Δ_mlp, Δ_knn, Δ_rff) per base model; never max_m Δ_m alone;
  never select the family with the narrowest interval.
- Categories: **CROSS-FAMILY ROBUST** (material effect in the same direction in ≥ 4 of 6 valid families in every base model, none
  materially reversed), **FAMILY-SPECIFIC** (material in 1–3 valid families, ≥ 2 valid families at most negligible), **PANEL-NEGATIVE**
  (no valid family material, ≥ 4 valid families with the interval inside the negligible band), **INCONCLUSIVE** (otherwise, including base-
  model disagreement or < 5 valid families). Experiment specs fix the materiality scale and validity checks.
- Panel failure is never evidence of information-theoretic absence.
