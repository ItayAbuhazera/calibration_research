# G1-DP — Decoder Panel accessibility audit of G1 (frozen specification v1)

Status: **FROZEN v1 (2026-09-28)**, sidecar `g1_dp_spec.frozen.sha256`. Frozen before any G1-DP fit, bundle or raw-H_3.22 extraction existed.
Panel spec sha256 `c27a1df454889bb706e9ded852275683f7be2d3316a5d05603ca0b3246c5d50d` (commit `0dc2041`).
Instrument: Decoder Panel v1 (`docs/decoder_panel_v1_spec.md`, frozen first). Code: `atlas/g1dp.py`, `atlas/g1dp_extract.py`,
`atlas/g1dp_aggregate.py`, `atlas/g1dp_rules.py`. Journal: `docs/decoder_panel_v1_master_report_2026-09-28.md`.

**TARGET-POOLED HPO — ACCESSIBILITY DIAGNOSTIC ONLY (target-label access on exposed development cells).** Not deployment, not
source-trained generalization, not unseen-shift transfer. A NEW experiment: it does not alter G1's frozen Outcome C, which remains the
historical verdict of the G1 Conditional Accessibility Gatekeeper.

## 0. Question

Does the apparent accessibility difference between the compact Stage-0 evidence P_3.22 and the penultimate representation H_L survive
changes in decoder family? G1 (anchored linear, T-8k×1): B − A = +1.88 / +1.80 pp, C − A = +0.27 / +0.39 pp, so B − C = +1.62 / +1.41 pp;
conditional D − C = +0.24 / +0.24 pp; equivalent-span sensitivity |G − C| = 0.72 / 0.50 pp.

Primary within-family contrast: **Δ_BC(m) = Acc_m(B) − Acc_m(C)** (12-cell macro top-1 accuracy, pp) for each family m of the panel.
Secondary frozen contrasts: **Δ_HC(m) = Acc_m(H) − Acc_m(C)** (raw layer3.22 vs raw layer4 representation; raw tier) and
**Δ_DC(m) = Acc_m(D) − Acc_m(C)** (G1's conditional increment).

## 1. Fitting unit (repository-verified correction to the task prompt)

The task prompt assumed G1 fitted one readout per corruption cell. The repository (`atlas/g1_fit.py`, G1 spec §3) shows ONE target-
supervised readout per (checkpoint, regime, arm, outer fold), fitted on T-8k×1 rows pooled over the 12 cells (each outer-train image once,
at its preassigned corrupted cell `cell_assignment_local`) and evaluated on the outer-test images under all 13 conditions. G1-DP preserves
that unit: HPO unit = **checkpoint × outer fold × evidence arm × decoder family** (pooled over cells). Hyperparameters are therefore not
cell-specific. Only T-8k×1 is run (T-2.5k is not rerun).

## 2. Data (already exposed; no new data)

- Checkpoints: ResNet-101 CIFAR-100 `baseline_cross_entropy` seeds 2 and 4; corrected_v2 protocol; the 12 exposed CIFAR-100-C cells
  (gaussian_noise, defocus_blur, fog, jpeg_compression × severity 1/3/5) + clean (evaluation only, descriptive).
- Folds: Stage-0 plan `results/stage0/shared/fold_plan.json` (5 outer folds, image-grouped inner 75/25 split `inner_fit_mask`).
- Blocks: z (atlas logits), P_3.22 (`raw__probe_logits[:, 8]`), P_4.2 (`[:, 11]`), Z_other (other checkpoint's atlas logits),
  H_L (`results/g1/hL/seed{s}/h_<cond>.npy`, G1 extraction; consistency passed), labels via `atlas.stage0_data` (all its assertions).
- **Raw tier — H_3.22_raw (new extraction, label-free):** the exact input of the P_3.22 probe: L2-normalized global-average-pooled
  `layer3.22` output, 1024-d, strict FP32, batch 250 (`atlas/g1dp_extract.py`). The probe's own z-scoring is affine and is subsumed by
  every family's standardization (and irrelevant for LightGBM). **Gate (frozen):** reconstructing P_3.22 with the saved probe
  (`probe_weights.pt`, index 8) must give max |P_rec − P_cache| ≤ 0.05 (cache is float16; |P| ≤ ~32) and argmax agreement ≥ 0.999
  over every row of all 13 conditions, and the forward logits must match the atlas logits (max |Δz| ≤ 1e-2), in both checkpoints.
  If the gate fails: the raw tier (arms H, I, Hs) is dropped from interpretation and the report states: "G1-DP tests decoder dependence
  of the existing compact P_3.22 evidence, but cannot fully test raw layer3-vs-layer4 accessibility." No other layer, pooling rule or
  preprocessing may be substituted. Note (recorded, not changed): H_L follows G1's definition (GAP, not L2-normalized); H_3.22_raw
  follows the probe's definition (GAP, L2-normalized).
- Not accessed: reserved CIFAR-100-C families, checkpoints 1/3/5, new images, new label budgets, other layers.

## 3. Arms (frozen)

| arm | inputs to g (all anchored: logits = z + g) | role |
|---|---|---|
| A | z | Z-only reference |
| B | z, P_3.22 | Stage-0 compact evidence |
| C | z, H_L | full penultimate representation |
| D | z, H_L, P_3.22 | G1 primary conditional arm |
| E | z, H_L, P_4.2 | redundant-summary control |
| F | z, H_L, Z_other | diverse-predictor control |
| H | z, H_3.22_raw | raw-tier: raw layer3.22 representation |
| I | z, H_L, H_3.22_raw | raw-tier: both raw representations |
| Cs | z, H_L~ | dimension-matched shuffle of C |
| Hs | z, H_3.22_raw~ | dimension-matched shuffle of H |
| Ds | z, H_L, P_3.22~ | = G1 D-shuf, now on all 5 folds |

Shuffle rule (frozen, = G1 D-shuf): the shuffled block's image identity is permuted within each partition (inner-fit +10, inner-val +20,
outer-train refit +30, outer-test +40; `np.random.default_rng(20260923 + offset)`), taking the permuted image's value at the row's own
cell / evaluation condition; z, labels, folds, cells and dimensionality are unchanged. Selection uses the inner-partition permutation;
the refit uses the outer-train permutation. G1's G (P_ker) arm is not repeated (its role was a linear-parameterization sensitivity).

## 4. Decoders and HPO

All six Decoder Panel v1 families, frozen grids/spaces/budgets of the panel spec, anchored classification (K = 100, offset = z), HPO regime
`target_pooled`, selection objective = inner-validation multiclass NLL on the Stage-0 inner split (fit rows = inner-fit images, val rows =
inner-val images, each at its preassigned cell), anchor-only candidate included, refit on all outer-train rows, single evaluation on the
outer-test images × 13 conditions. Study seeds from ids {seed, fold, arm, regime = T-8k1}.

## 5. Statistics

Per checkpoint and family: predictions pooled over the 5 outer folds (every image held out once); per-image 12-cell macro accuracy;
paired image-group bootstrap (`stage0_aggregate.group_id_for_bootstrap`), B = 2000, **one resample-index array per checkpoint shared by
all arms and all six families** (seed 20261010 + 60000 + checkpoint); 95 % percentile intervals, conditional on the fitted CV
predictions (no training-seed variance). Checkpoints 2 and 4 are reported separately and never pooled.

Reported per family and checkpoint (pp): B − A, C − A, D − C, D − B, E − C, F − C, **B − C**, H − A, **H − C**, I − C, I − H, and the
controls Cs − A, Hs − A, Ds − C; accuracy of every arm; inner-validation NLL and selected hyperparameters per fold; clean accuracy
(descriptive). The full family vector is always shown; no best-of-panel quantity is reported as a result.

## 6. Validity (per family, per checkpoint; a family counts only if valid in both checkpoints)

- V1 (global) H_L consistency (existing) passed; raw tier additionally requires the H_3.22 gate (§2).
- V2 completeness: all 11 arms × 5 folds present with validated `.done.json` markers.
- V3 convergence (D1/D2/D6): every final refit of arms A, B, C, D, H, I converged (after the frozen retry).
- V4 strong-edge (D1/D2/D6): selected λ = 1e1 (strongest grid value) in ≥ 2 of 5 folds for any of A, B, C, H → invalid.
  (Weak-edge selections are reported, not invalidating: λ = 1e-7 on standardized inputs is numerically the unpenalized fit.)
- V5 shuffle controls (Stage-0 convention): Cs − A < +0.2 pp, Hs − A < +0.2 pp and Ds − C < +0.2 pp (point estimates, all folds).
  (Hs only when the raw tier is valid.)
- V6 (D5) JL distortion audit: every projected fit has 5th percentile ≥ 0.5 and 95th percentile ≤ 1.5 (the eps = 0.5 band).

## 7. Classification of a contrast (per family, per checkpoint; material scale m = 0.5 pp, Stage-0 convention)

- **POS**: Δ ≥ +0.5 and lower 95 % bound > 0.  **REV**: Δ ≤ −0.5 and upper bound < 0.
- **NULL** (at most negligible at the material scale): the interval lies inside (−0.5, +0.5).
- **UNC**: otherwise.

## 8. Cross-family verdict (frozen; applied separately to Δ_BC [primary], Δ_HC [raw tier] and Δ_DC [conditional])

Let V = families valid in both checkpoints. If |V| < 5 → **INCONCLUSIVE (validity)**. Otherwise, first match:

1. **CROSS-FAMILY ROBUST (+)**: ≥ 4 families of V are POS in both checkpoints and no family of V is REV in either.
   (**CROSS-FAMILY ROBUST (−)**: same with REV/POS exchanged.)
2. **FAMILY-SPECIFIC**: 1–3 families of V are POS (or REV) in both checkpoints and ≥ 2 families of V are NULL in both checkpoints.
3. **PANEL-NEGATIVE**: no family of V is POS or REV in either checkpoint and ≥ 4 families of V are NULL in both checkpoints.
4. **INCONCLUSIVE**: otherwise (including checkpoint disagreement and imprecision).

Interpretation (Δ_BC): (1+) compact mid-depth evidence is panel-robustly more accessible than H_L under this target-pooled finite-sample
regime — still NOT "H_L lacks the information"; (2) the G1 gap is decoder-family-specific (if the POS families are only linear-head
families, the original G1 gap was decoder/estimation dependent) — no robust representation-depth claim; (3) no material accessibility
difference under the panel; (4) no stable interpretation. Δ_HC answers the same question for the raw representations (not the probe
summary); Δ_DC re-measures G1's conditional increment. If the raw tier is invalid, Δ_HC is not interpreted.

| possible outcome (Δ_BC) | supported / weakened | still unresolved | next decision |
|---|---|---|---|
| CROSS-FAMILY ROBUST (+) | compact P_3.22 more accessible than H_L across decoder families | information content; depth mechanism; source regime | record; G1 historical verdict unchanged; no new rescue |
| FAMILY-SPECIFIC | G1 gap depends on decoder family | which inductive bias carries it | record; no depth claim |
| PANEL-NEGATIVE | no material gap once decoders vary | — | record; Stage-0 increment was readout-specific |
| INCONCLUSIVE | none | precision / validity | record; no follow-up by default |

No outcome authorizes new data, new layers, new decoder families, new budgets or reopening Stage 0.

## 9. Execution

Stages (afterok chains, all from an immutable snapshot `python -m atlas.snapshot g1dp`): C1 GPU extraction per checkpoint
(`atlas.g1dp_extract`) → C2 CPU bundle per (checkpoint, fold) (`atlas.g1dp bundle`) → D one Slurm array per family over the frozen unit
mapping (`atlas.g1dp fit_index`) → E/F aggregation (`atlas.g1dp_aggregate`, refuses incomplete arrays) → report. Resource classes and
concurrency: `docs/decoder_panel_v1_resource_plan.md`. Engineering recovery (time/memory/I-O resubmission of failed indices) allowed;
any scientific change after outcomes are visible is forbidden.
