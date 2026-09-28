# G1 — Conditional accessibility gatekeeper (frozen specification v1)

Frozen 2026-09-28 before any G1 code was written or any G1 outcome existed. Parent commit `f990341`.
Authorization: researcher instruction of 2026-09-28 (design, implement, execute on exposed development data;
no reserved families; no checkpoints 1/3/5). **TARGET-SUPERVISED DIAGNOSTIC (ORACLE-LABEL ACCESS).**

## 0. Question and estimand

In the ResNet-101 CIFAR-100 checkpoints 2 and 4 (`baseline_cross_entropy`), on the 12 exposed CIFAR-100-C development
cells (gaussian_noise, defocus_blur, fog, jpeg_compression x severity 1/3/5), does the fixed Stage-0 evidence P_3.22
(cached 100-d logits of the clean-trained layer3.22 GAP probe, `raw__probe_logits[:, 8, :]`) add target-fitted top-1 accuracy
to an anchored linear readout that already has the native logits Z **and** the full penultimate representation H_L
(layer4 output, global-average-pooled, 2048-d)?

    Delta_cond = Acc_T[q(Z, H_L, P_3.22)] - Acc_T[q(Z, H_L)]        (primary; T-8k x 1; 12-cell macro accuracy, pp)

A positive value means only: additional operationally accessible evidence for this restricted target-fitted readout at this
finite label budget. It does not mean information absent from H_L, information-theoretic novelty, causality, deployability,
source accessibility, or uniqueness to layer3.

## 1. Reconciliation recorded before freezing (verified against code/artifacts)

1. State (a) spans (regime-map follow-up): `atlas/followup_stage2.py::projectors` uses u = (L2norm(h) - mu)/sigma for state (a),
   while C1d uses raw h. (Z, P_ker u) and (Z, h) therefore do NOT share a linear span in (a); they do in b* (u = h). An earlier
   audit statement to the contrary was wrong for (a).
2. Lambda direction: `stage0_fit` / `followup_anchored` minimize mean CE + lambda ||W||_F^2. lambda = 0.1 is the STRONGEST penalty on
   the grid {1e-1..1e-5}. All state-(a) H_L arms selected 0.1 in 5/5 folds, i.e. they sat at the strong-regularization edge and may
   have preferred even stronger regularization (between 0.1 and f = 0). State (a)'s D - C1d = +1.19 pp is therefore
   regularization-edge-limited and is not used as clean motivation.
3. ResNet-50 D - Delta_T(C1d) by state [artifact `results/regime_map_followup/report/stage2_anchored.json`]: b1_s1 +0.51 [+0.27, +0.75];
   b1_s2 +0.09 [-0.14, +0.33]; b3_s1 +0.05 [-0.16, +0.26]; b3_s2 +0.05 [-0.17, +0.25]; b10_s1 +0.14 [-0.06, +0.32]; b10_s2 +0.04 [-0.15, +0.24].
   Seeds can disagree (b1); the G1 rule reports per checkpoint and never pools disagreement away.
4. Stage-0 conventions [`docs/stage0_execution_spec.md` §7 memo table]: material = Delta >= +0.5 pp with the 95% interval excluding 0 in
   both checkpoints; null = upper 95% endpoint < +0.2 pp in both; a sanity/shuffle control gaining >= +0.2 pp = protocol fault.
   G1 keeps all three unchanged.
5. Parameterization: Z+K - C1d = +0.12 to +0.32 pp in the six fine-tuned ResNet-50 states despite identical spans; P_row moved
   -0.74 to +0.56 pp. The G1 primary contrast is nested and shares the identical H_L coordinates, so this does not invalidate it; the
   equivalent-span arm (G) is a sensitivity diagnostic (not a validity gate) used to bound mechanism claims (§6).
6. Z_other [artifact `results/stage0_ablation/report/ablation_aggregate.json`]: Delta_T = +3.55 / +2.51 pp, Delta_S = +3.46 / +2.53 pp
   (checkpoint 2 / 4, unanchored Stage-0 fitter). Z_other is the only competent independent predictor available without new data.

## 2. Data, exposure, provenance

- Images: the 10,000 CIFAR-100 test images, conditions `spec.CONDITIONS` (clean + 12 cells) from `results/atlas/shared/test_sets_full.npy`.
- Z: canonical strict-FP32 atlas logits `results/atlas/seed{s}/u0/<cond>.npz["logits"]` (the Stage-0 Z). Labels/Z/P through
  `atlas.stage0_data.load_cell` (all its cross-artifact assertions apply). Z_other: the other checkpoint's atlas logits (2 <-> 4).
- P_3.22 = probe index 8, P_4.2 = probe index 11 of `results/layer_pilot/checkpoint_seed{s}/<cond>/per_sample.npz["raw__probe_logits"]`
  (clean-trained layer-pilot probes on z-scored L2-normalized GAP features; float16 cache).
- **New artifact:** H_L for checkpoints 2 and 4, 13 conditions: hook the `layer4` output of the CIFAR ResNet-101, mean over (H, W)
  (identical to the model's `avg_pool2d(out, 4)` on the 4x4 map), strict FP32 (`atlas.common.setup_torch`), batch 250, stored float32.
  Head: W_eff = fc.weight / temp, b_eff = fc.bias / temp (the model divides logits by `self.temp`).
- **Extraction consistency gate (before any fit):** max |z_atlas - (H_L W_eff^T + b_eff)| <= 1e-2 over every row of all 13 conditions,
  per checkpoint (the frozen follow-up threshold; observed there <= 1.8e-5). Failure -> stop; engineering recovery only.
- Label access: target (T) fits use labels of the 12 exposed development cells (one preassigned cell per image, Stage-0 plan).
  Clean-condition accuracy is reported descriptively only; no clean labels are used for fitting.
- **Not accessed:** the 11 reserved CIFAR-100-C families, checkpoints 1/3/5, any new images.

## 3. Fitting protocol

- Folds/regimes: Stage-0 plan `results/stage0/shared/fold_plan.json` (seed 20260922), 5 outer folds, inner 75/25 image-grouped split,
  one preassigned corrupted cell per image (`cell_assignment_local`). Regimes: **T-8k x 1** (all outer-train images) and
  **T-2.5k x 1** (the plan's `nested_2500_local_positions`, inheriting the inner split and cell assignment).
- Readout: anchored q = softmax(z + f(x)), f linear in the arm's standardized inputs (`atlas.followup_anchored.fit_anchored`, unchanged):
  mean CE + lambda ||W||_F^2, bias unpenalized, zero init, float64 L-BFGS, Stage-0 tolerances and retry policy. Standardization per
  coordinate on the inner-fit rows for selection and on all outer-train rows for the final refit (as in the follow-up).
- Lambda path: f = 0 (lambda = inf) plus {1e-1, 1e-2, 1e-3, 1e-4, 1e-5}; selection = minimum inner-validation NLL, ties -> f = 0 then
  larger lambda (`followup_anchored.pick`). **Grid-edge rule (every arm):** if the selected finite lambda is 1e-1, also evaluate 1e0 and,
  if 1e0 is then selected, 1e1; if the selected lambda is 1e-5, evaluate 1e-6 and, if selected, 1e-7. A selection still at the outermost
  evaluated finite value after the extensions is an **unresolved edge**. Selected lambda, edge status, convergence, retries, wall time
  are stored per (arm, fold).
- Nested C/D/E/F/D-shuf use identical rows, folds, Z anchor, raw H_L coordinates, per-coordinate standardization rule and lambda protocol.

## 4. Arms (frozen)

| Arm | Inputs to f | Regimes | Role |
|---|---|---|---|
| A | z | 8k, 2.5k | anchored Z-only reference |
| B | z, P_3.22 | 8k, 2.5k | Stage-0 continuity |
| C | z, H_L | 8k, 2.5k | full penultimate control |
| **D** | **z, H_L, P_3.22** | **8k, 2.5k** | **primary conditional arm** |
| E | z, H_L, P_4.2 | 8k | recipe-matched redundant-summary control (probe of H_L itself) |
| F | z, H_L, Z_other | 8k | competent diverse-predictor control |
| G | z, P_ker H_L (P_ker = I - W_eff^+ W_eff) | 8k | equivalent-span sensitivity (same span as C) |
| D-shuf | z, H_L, P_3.22 with image identity permuted within each of {inner-fit, inner-val, outer-train, outer-eval} (seed 20260923) | 8k, fold 0 only | leakage / regularization sanity |

No layer sweep, no PCA/MLP variants, no clean-regime refits, no geometry scores.

## 5. Statistics

Per checkpoint: predictions pooled over the 5 outer folds (every image held out once); 12-cell macro accuracy per image; paired
image-group bootstrap (groups from `stage0_aggregate.group_id_for_bootstrap`), B = 2000, **one shared resample-index array per checkpoint
for all arms and both budgets** (seed 20261010 + 40000 + checkpoint). 95% percentile intervals. D-shuf is fold-0 only: its contrast is
Acc(D-shuf) - Acc(C) on fold-0 held-out images (point estimate). Intervals are conditional on the fitted CV predictions; no training-seed
variance; the two checkpoints are two trained models, not a population.

Reported quantities (pp): Delta_cond(8k) = D - C; Delta_cond(2.5k); Delta_E = D - E; Delta_F = F - C; D - F; s = |G - C| (8k);
C - B (8k, 2.5k); D - B (nesting); B - A (Stage-0 continuity); Delta_cond(2.5k) - Delta_cond(8k); (C - B)(8k) - (C - B)(2.5k); D-shuf - C.

## 6. Decision rule (frozen; evaluated per checkpoint, then combined)

Thresholds: m = +0.5 pp (material), n = +0.2 pp (negligible upper bound) — Stage-0 conventions, allocation thresholds, not truths.

**Validity (any failure -> outcome F):**
V1 extraction consistency passes in both checkpoints. V2 every final fit of A–G (both budgets) converged (after the frozen retry).
V3 neither C nor D has an unresolved lambda edge in >= 2 of 5 folds in either regime/checkpoint. V4 D-shuf - C < +0.2 pp in both checkpoints.

**Per-checkpoint label (first match):**
1. `cap` (sample-efficiency / capacity) if any of: (i) nesting failure D - B < -0.5 pp at 8k; (ii) E3 pattern: Delta_cond(2.5k) - Delta_cond(8k) >= 0.5
   with lower bound > 0 AND (C - B)(8k) - (C - B)(2.5k) >= 0.5 with lower bound > 0 AND the 8k class is not MAT; (iii) MAT at 8k but
   NOT (Delta_E >= 0.5 with lower bound > 0).
2. `div` if MAT at 8k and the upper bound of (D - F) < +0.5 (P's conditional increment not materially larger than Z_other's).
3. `acc` if MAT at 8k.
4. `neg` if NEG at 8k and max(Acc C, Acc G) >= Acc B - 0.5 (H_L recovers the Stage-0-like gain within the material scale).
5. `mid` otherwise.

MAT = Delta_cond >= 0.5 with lower bound > 0 AND Delta_cond > s. NEG = upper bound of Delta_cond < 0.2.

**Combined outcome:** F if invalid; if both checkpoints share a label: acc -> **A**, neg -> **B**, cap -> **C**, div -> **D**;
otherwise (different labels, or `mid`) -> **E**.

| Outcome | Supported | Weakened | Unknown | Branch |
|---|---|---|---|---|
| A MATERIAL CONDITIONAL ACCESSIBILITY | E2: P_3.22 adds operational evidence after (Z, H_L), beyond a redundant H_L summary and beyond what Z_other adds | E1, E3 (at this budget), E4 | source accessibility, reserved-family robustness, mechanism | Stage-0 increment recorded as depth-specific operational accessibility (development). Robustness stage becomes *eligible*, not started. |
| B PENULTIMATE SUFFICIENCY | E1: H_L already carries the usable evidence; Stage 0 = head bottleneck + target adaptation | E2 | why the native head discards target-useful directions (DFR-type) | **Stop** the intermediate-specific branch. |
| C SAMPLE-EFFICIENCY / CAPACITY | E3/E5: P is a compact summary easier to estimate, or the effect is a probe-summary artifact | E2 as a representation-content claim | asymptotic sufficiency (two budgets cannot show it) | **Stop**; no further budgets or readouts. |
| D GENERIC DIVERSITY | P is useful, but so is any competent diverse view; noun = "complementary predictive view" | depth specificity of E2 | whether same-network views differ from cross-network ones in kind | **Stop** the depth-specific framing. |
| E MIXED / INCONCLUSIVE | none | none | checkpoint dependence | **Stop**; no third checkpoint. |
| F INVALID | none | none | engineering / regularization limit | One engineering recovery if engineering; otherwise record optimization-limited and **stop**. |

Only A leaves anything eligible; nothing is started automatically.

## 7. Execution envelope

CPU fits via Slurm (`cpu` partition, 4 cores, node CPU model logged), GPU extraction on `rtx4090`. Estimated ~60–80 four-core CPU-hours
+ < 0.2 GPU-h. Jobs run from an immutable snapshot (`python -m atlas.snapshot g1`). Engineering recovery (I/O, memory, time limits,
re-submission) allowed; any scientific change after outcomes are visible is forbidden (stop and report instead).
