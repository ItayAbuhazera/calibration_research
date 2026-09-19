# Decision-Improving Calibration: Pre-Run Reference

## Scalar-Temperature Structural-Impossibility Lemma

**Theorem.** For any function $T: \mathbb{R}^K \to \mathbb{R}_{>0}$ (not necessarily constant),

$$\arg\max_k z_k = \arg\max_k \frac{z_k}{T(z)}.$$

*Proof.* Dividing all logits by a strictly positive scalar $T(z)$ preserves their relative order.
Hence neither global temperature scaling nor per-sample parameterized temperature scaling (PTS)
can change the predicted class. $\square$

**Implication for the 2×2 structural argument:**
Any method that applies a positive scalar function of the logits — even a complex sample-adaptive
one — is structurally incapable of benefiting accuracy through decision changes. This includes PTS.
It belongs to the *sample-dependent, not class-dependent, argmax-invariant* cell.

---

## 2×2 Structural Classification Matrix

| | **Not class-dependent** | **Class-dependent** |
|---|---|---|
| **Not sample-dependent** | temperature_scaling<br>*(global scalar, argmax-invariant)* | vector_scaling, beta_calibration, ovr_isotonic, odir_dirichlet<br>*(global class-specific, can change argmax)* |
| **Sample-dependent** | parameterized_temperature_scaling (PTS)<br>*(per-sample scalar, argmax-invariant by lemma)* | GLAD-PI, full_vector_geometric_fusion, kcal_lite, trust_score_switch, aar_lightweight<br>*(jointly sample×class or sample×scalar+class additive, can change argmax)* |

**Key insight:** The decision-improvement potential lives in the bottom-right cell —
methods that are both sample-dependent and class-dependent. GLAD-PI is explicitly
designed for this cell via its permutation-equivariant per-class correction.

---

## Decision-Audit Field Definitions

All methods with access to base probabilities receive a `decision_audit` sub-dict:

| Field | Definition |
|-------|-----------|
| `base_accuracy` | Fraction of test samples where base model prediction is correct |
| `method_accuracy` | Fraction of test samples where calibrated prediction is correct |
| `accuracy_delta` | `method_accuracy - base_accuracy` |
| `argmax_change_rate` | Fraction of samples where calibrated argmax ≠ base argmax |
| `changed_to_correct_count` | Samples where base was wrong, method is correct (N_flip_right) |
| `changed_to_wrong_count` | Samples where base was correct, method is wrong (N_flip_wrong) |
| `wrong_to_wrong_count` | Samples where base was wrong, method is wrong (different error) |
| `correct_to_correct_count` | Samples where base was correct, method is still correct (unchanged) |
| `net_flips` | `changed_to_correct_count - changed_to_wrong_count` |
| `net_flip_rate` | `net_flips / N` |
| `changed_to_correct_rate` | `changed_to_correct_count / N` |
| `changed_to_wrong_rate` | `changed_to_wrong_count / N` |
| `flip_to_correct_count` | Alias for `changed_to_correct_count` |
| `flip_to_wrong_count` | Alias for `changed_to_wrong_count` |

All `da_*` prefixed columns in `summary_metrics.csv` correspond to these fields.

---

## Intended Full Audit Grid

The following methods should all appear in the same `summary_metrics.json` for a
valid decision-improvement comparison:

**Argmax-invariant baselines (structural control group):**
- `base_model`
- `temperature_scaling`
- `parameterized_temperature_scaling` (PTS) — `--enable_pts_baseline`
- `rgcl`, `rgcc`, `gc_dac`, `gc_tulip`

**Argmax-changing standard calibrators:**
- `vector_scaling`
- `beta_calibration`
- `ovr_isotonic`
- `odir_dirichlet`

**Geometry-aware baselines:**
- `full_vector_geometric_fusion_{mode}` / `full_vector_geometric_fusion_rgcl_{mode}` — `--enable_full_vector_geometric_fusion`
- `softmax_knn_blend` / `softmax_knn_blend_rgcl` — `--enable_knn_blend_baseline`
- `kcal_lite` / `kcal_lite_rgcl` — `--enable_kcal_lite_baseline`
- `rgcl_neighbor_correction_trust`, `rgcl_neighbor_correction_separation` — `--enable_rgcl_neighbor_correction`

**Stage B methods (to be added):**
- `trust_score_original_diagnostic`, `trust_score_original_switch` — `--enable_trust_score_baseline`
- `aar_lightweight` — `--enable_aar_lightweight`
- `glad_pi` — `--enable_glad_pi`

**External (official AAR or other):**
- Any method via `--external_method_json path/to/external_methods.json`

---

## External Method Import

To evaluate official AAR outputs (or any external probability matrix):

```bash
python Experiments/run_unified_benchmark.py \
  --dataset cifar100 --model resnet18 --seed 12 \
  ... (other flags) ... \
  --external_method_json external_methods.json
```

Where `external_methods.json` has the schema:

```json
{
  "methods": {
    "aar_official": {
      "test_probs": "/path/to/aar_official_test_probs.npy",
      "metadata": {
        "can_change_argmax": true,
        "method_family": "external_full_vector_posthoc",
        "tuning_procedure": "external",
        "tuning_split": "external_or_validation",
        "tuning_objective": "external"
      }
    }
  }
}
```

Requirements: `test_probs` must be a `.npy` file of shape `[N_test, C]` with rows summing to 1.

---

## Intended Smoke Command (Stage A complete, Stage B pending)

```bash
python Experiments/run_unified_benchmark.py \
  --dataset cifar100 --model resnet18 --seed 12 \
  --enable_pts_baseline \
  --enable_full_vector_geometric_fusion \
  --fusion_feature_source rgcl \
  --enable_knn_blend_baseline \
  --enable_kcal_lite_baseline --kcal_k_per_class 5 \
  --enable_rgcl_neighbor_correction \
  --inner_val_fraction 0.5 \
  --inner_val_seed 123 \
  --output_dir results/unified_benchmark/cifar100/resnet18/seed12/decision_ready_smoke
```

After Stage B is implemented, add:

```bash
  --enable_trust_score_baseline \
  --enable_aar_lightweight \
  --enable_glad_pi \
  --glad_pi_beta_grid 0,0.01,0.03,0.1,0.3,1,3,10 \
  --glad_pi_nll_tolerance 0.05
```

---

## Experiment Card Template

Fill this in **before** looking at test-set results to avoid post-hoc tuning.

---

## Experiment Card: CIFAR-100 Decision-Improvement Run — v1 (PRE-REGISTERED)

**Date pre-registered:** 2026-06-21

**Dataset / Model / Seed:** CIFAR-100 / ResNet-18 / seed 21

**Note:** Originally pre-registered as seed 12 but no trained model exists for that seed. Switched to seed 21 (first available seed with a prior benchmark run).

**Output directory:** `results/unified_benchmark/cifar100/resnet18/seed21/decision_run_v1`

**Method list:**
- base_model (reference)
- temperature_scaling (argmax-invariant control)
- parameterized_temperature_scaling / PTS (argmax-invariant control, `--enable_pts_baseline`)
- vector_scaling, beta_calibration, ovr_isotonic, odir_dirichlet (standard argmax-changing)
- gc_dac, rgcl, rgcc, gc_tulip, anchored_model_tail (geometry-top-label methods)
- full_vector_geometric_fusion_rgcl_{modes} (`--enable_full_vector_geometric_fusion --fusion_feature_source rgcl`)
- softmax_knn_blend_rgcl (`--enable_knn_blend_baseline`)
- kcal_lite_rgcl (`--enable_kcal_lite_baseline --kcal_k_per_class 5`)
- rgcl_neighbor_correction_trust, rgcl_neighbor_correction_separation (`--enable_rgcl_neighbor_correction`)
- trust_score_original_diagnostic, trust_score_original_switch (`--enable_trust_score_baseline`)
- aar_lightweight (`--enable_aar_lightweight`) — NOT official AAR
- glad_pi (`--enable_glad_pi`) — primary hypothesis method

**Primary hypothesis:**
GLAD-PI achieves `net_flips >= 30` on the CIFAR-100 10k test set
AND `top_label_ece_delta <= +0.005` AND `nll_delta <= +0.05` relative to base_model.

**Primary metric:** `decision_audit.net_flips` in `summary_metrics.json`

**Secondary metrics (for structural comparison):**
- Does GLAD-PI beat the simpler switch baselines (trust_score_original_switch,
  rgcl_neighbor_correction_trust) in net_flips?
- Is ECE well-calibrated (not just accurate)?

**ECE / NLL tolerance (pre-specified, evaluated against base_model):**
- `top_label_ece` delta <= +0.005 (within 0.5 percentage points of base)
- `nll` delta <= +0.05 (within 5% relative to base model NLL)

**Go / No-Go rule:**
- **GO** if: `glad_pi.net_flips > 0` AND `nll_delta <= 0.05` AND `ece_delta <= 0.005`
- **NO-GO** if: `glad_pi.net_flips <= 0` OR either tolerance is violated

**Direction if NO-GO:**
- If net_flips > 0 but tolerance violated: investigate ECE/NLL regularisation; do NOT tune post-hoc.
- If net_flips <= 0: GLAD-PI does not improve decisions on this dataset/model. Trust Score switch or
  neighbor correction may still be positive — report those separately.

**Comparison baseline for NLL tolerance:** `temperature_scaling` NLL (not base_model NLL, since TS
is a standard calibration reference).

**Inner validation split:** `--inner_val_fraction 0.5 --inner_val_seed 123`
(50% fit / 50% select; ~2500 samples each on CIFAR-100 5k val)

**GLAD-PI hyperparameters (pre-specified, not tuned post-hoc):**
- Beta grid: `0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10`
- NLL tolerance: `0.05`
- Hidden dim: `64`, LR: `1e-3`, weight_decay: `1e-4`, epochs: `100`
- Feature source: `rgcl` (RGCL per-class 1-NN distances)
- Temperature branch: disabled

**Caveats (pre-registered):**
- AAR-lightweight is NOT official AAR. Uses 1-NN distance as atypicality proxy.
  Official AAR will be run separately via `--external_method_json`.
- Trust Score and neighbor_correction_trust are equivalent under the same RGCL feature space
  (documented in method metadata `equivalence_note`).
- Single seed (seed 12); this is a go/no-go pilot, not a multi-seed significance test.

**Exact command:**
```bash
mkdir -p logs
conda run -n geo_cuda12 python Experiments/run_unified_benchmark.py \
  --dataset cifar100 --model resnet18 --seed 21 \
  --enable_pts_baseline \
  --enable_full_vector_geometric_fusion \
  --fusion_feature_source rgcl \
  --enable_knn_blend_baseline \
  --enable_kcal_lite_baseline --kcal_k_per_class 5 \
  --enable_rgcl_neighbor_correction \
  --enable_trust_score_baseline \
  --enable_aar_lightweight \
  --enable_glad_pi \
  --glad_pi_beta_grid 0,0.01,0.03,0.1,0.3,1,3,10 \
  --glad_pi_nll_tolerance 0.05 \
  --inner_val_fraction 0.5 \
  --inner_val_seed 123 \
  --output_dir results/unified_benchmark/cifar100/resnet18/seed21/decision_run_v1 \
  2>&1 | tee logs/cifar100_resnet18_seed21_decision_run_v1.log
```

**Run started:** 2026-06-21

**Results recorded:** (fill in after run completes)

---

---

## Inner Validation Split Semantics

The `--inner_val_fraction` flag controls the **selection fraction** — the portion
of the validation set reserved for held-out hyperparameter selection (e.g. GLAD-PI
beta grid selection, trust-score threshold selection).

- `--inner_val_fraction 0.5` → 50% fit, 50% select (default)
- The default 0.5 is chosen because `net_flips` is discrete and noisy;
  fewer than ~2500 selection samples (50% of CIFAR-100 5000-val) would give
  unreliable beta selection.
- For PTS: only the fit split is used (early-stopping based; no grid selection needed).
- For GLAD-PI: both fit (training) and select (beta grid) splits are required.
