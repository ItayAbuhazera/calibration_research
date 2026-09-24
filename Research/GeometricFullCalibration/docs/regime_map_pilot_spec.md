# Regime-map pilot — is source-recoverability of intermediate evidence controlled by its clean redundancy given Z? (frozen specification)

**Frozen:** 2026-09-24, before any fit of this pilot. §§0–9 are not to be edited after results are
opened; results go in an appended §10. **Status: prepared, NOT launched.** Launching needs the
researcher's explicit go.
**Card:** `ResearchBrain/05_Experiments/2026-09-24 Regime-Map Pilot.md`.
**Parents:** `docs/stage0_execution_spec.md` (frozen, complete) and `docs/stage0_evidence_ablation_spec.md`
(frozen, complete). This is a new bounded development diagnostic; it reopens no closed study, uses
no reserved resource, and authorizes no confirmation.

## 0. Question and confound being separated

Stage 0 (ResNet-101 trained from scratch, checkpoints 2/4) found that a target-fitted readout gains from the
mid-depth (`layer3.7–3.22`) probe logits `P`, while the clean-fitted readout does not (Δ_S ≤ ≈ +0.3 pp).
The evidence ablation found that another checkpoint's logits `Z_other` **are** source-recoverable. "Same
network" and "clean-redundant given `Z`" are therefore **fully confounded** in the existing data.

> **Question.** Is the source-recoverability of intermediate evidence `P` controlled by its **clean
> redundancy given `Z`**, rather than by coming from the same network or from mid-depth?

Claim scope: development diagnostic; ImageNet-pretrained ResNet-50, CIFAR-100, the 12 development
cells, one training seed per state. Not a deployable method, not an information ceiling, not confirmation.

## 1. Backbone, data, resolution policy

* **Backbone:** torchvision `resnet50`, `ResNet50_Weights.IMAGENET1K_V1`, file `resnet50-0676ba61.pth`,
  **sha256 `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`** (matches torchvision's
  file-name hash prefix; asserted in code; torchvision 0.18.1 / torch 2.3.1). V1 rather than V2 because V2's
  training recipe (heavy augmentation, EMA, long schedule) is a larger departure from a plain supervised
  backbone.
* **Data:** CIFAR-100. Train on the benchmark 45,000-image train split and use the 5,000-image validation split
  (same arrays as the layer pilot, `results/studyAB/phase0_corrected_v2/.../splits`) only for probe/head
  hyperparameters and monitoring. Evaluation on the 10,000 test images × 13 conditions (clean + 12 development
  cells: `results/atlas/shared/test_sets_full.npy`), identical to Stage 0.
* **Resolution policy (one for all states): corrupt at native 32×32, then upsample to 224×224.** Images are the
  stored 8-bit CIFAR / CIFAR-100-C arrays (CIFAR-normalized floats; inverted and rounded to exact 8-bit pixels),
  bilinearly upsampled to 224×224 on the GPU, normalized with ImageNet mean/std. **Why:** the pretrained stem,
  strides and BN statistics are matched to ~224-pixel inputs; a modified 3×3 stem would discard the pretrained
  `conv1` and make the "pretrained" claim vacuous; corruptions are defined at 32×32, so applying them at native
  size and upsampling matches how CIFAR-100-C is defined; and it costs little (below). One normalization is used
  for train and test in every state (the earlier train/test normalization mismatch cannot recur). Extraction runs
  in strict fp32 (TF32 off).

## 2. Model states ("regimes" of the map)

| state | definition |
|---|---|
| **a** | frozen ImageNet backbone; `Z` = linear head trained on CIFAR-100 train (multinomial LR on L2-normalized, z-scored layer4 GAP features, λ ∈ {1e-4,…,1} by inner-FIT NLL) |
| **b1, b3** | the same backbone fine-tuned end to end; snapshots at the end of epoch 1 and epoch 3 of the schedule below (**dose**) |
| **b10** (b-final) | the same run at epoch 10; `Z` = the fine-tuned network's own logits for b1/b3/b10 |

Fine-tuning recipe (one run, seed 20260924, checkpoints saved along it): 10 epochs, SGD momentum 0.9 (Nesterov),
lr 0.01 (backbone) and 0.1 (new head) with per-step cosine decay to 0, weight decay 5e-4 (no decay on BN/bias),
batch 128, random 32×32 crop (pad 4) and horizontal flip before upsampling, fp16 autocast, on the 45k train split.
The new head is initialized N(0, 0.01²). Doses are epochs 1 and 3 of a 10-epoch cosine schedule (states along
one trajectory, not separate truncated schedules). Dose order by amount of adaptation: a < b1 < b3 < b10.

## 3. Evidence `P`

`P` = raw logits of a GAP linear probe on the **last block of `layer3`** (1024-d GAP), trained on **each state's
own 45k train features** with the **layer-pilot recipe unchanged across states**:
`Calibrators/layer_readouts.fit_probe` (L2-normalized GAP → z-score → multinomial LR, full-batch L-BFGS strong-Wolfe,
history 20, ≤ 200 iterations, ½λ‖W‖² convention of that recipe), λ ∈ {1e-4, 1e-3, 1e-2} chosen by inner-FIT NLL
(inner split `make_inner_validation_split(select_fraction=0.5, seed=123)` of the 5k validation split), scalar
temperature fitted and stored but **not applied** (raw logits are used, as in Stage 0). Selected λ and edge status
are reported per state. Differences from Stage 0: `P` and `Z` are stored as float32, not float16.

## 4. Protocol (Stage 0 code unchanged)

Stage 0 code path (`atlas/stage0c_run.py`, `stage0_fit`, folds seed 20260922, inner 75/25 split, nested 2,500
subset, per-image cell assignment, λ grid `{1e-1,…,1e-5}`, no-½ objective, L-BFGS recovery policy, 2,000-resample
grouped bootstrap) is used unchanged; only the artifact source (`--regime-state`) differs. Regimes T-8k×1
(primary), S-8k×1, T-2.5k×1, S-2.5k×1 for each of the four states, all five folds, on the 12 development cells;
S fits are also scored on the clean held-out images to get the **clean increment**. Planned: 4 states × 4 regimes ×
5 folds = 80 runs (160 arm-fits), CPU.
**Scale:** one training seed per state (one fine-tuning run; one frozen backbone). **The image-bootstrap intervals
do not capture training-seed variance**; all intervals are conditional on the fitted CV predictions.

## 5. Quantities and frozen decision rules

`Δ_T(state)`, `Δ_S(state)` = 12-cell macro accuracy of `q_ZP` − `q_Z` at 8k×1 under target / source fitting;
**gap(state) = Δ_T − Δ_S** (12-cell macro, pp); **clean increment(state)** = S-8k×1 `q_ZP` − `q_Z` on clean
held-out images. Intervals: paired image-group bootstrap; `gap(b10) − gap(a)` uses one shared resample of both
states' per-image effects. Doses are the four states ordered a < b1 < b3 < b10.

Rows are checked **top to bottom; the first that applies is the verdict** (implemented and unit-tested in
`atlas/regime_aggregate.py::evaluate_rules`):

| # | Condition | Reading | Decision |
|---|---|---|---|
| 1 | **Manipulation check fails:** clean increment(a) ≤ +0.3 pp, or its interval includes 0 | Regime (a) did not create a clean-complementary source from the same network | Inconclusive; report only |
| 2 | gap(b-final) interval includes 0, **or** gap(b-final) < +1.0 pp | The Stage 0 pattern does not transfer to a fine-tuned pretrained model | Restrict the claim to from-scratch ResNet-101 |
| 3 | gap(a) ≥ 0.8 × gap(b-final) | `P` is clean-complementary yet still not source-recoverable | Kill "clean redundancy controls recoverability" |
| 4 | gap(a) ≤ 0.5 × gap(b-final) **and** the interval of gap(b-final) − gap(a) excludes 0 **and** the dose condition holds | Supports clean redundancy as the controlling variable | Propose a confirmation and baselines package; do not start it |
| 5 | otherwise | Inconclusive | Report |

**Dose condition (frozen definition of "across doses the gap does not fall as the clean increment falls"):** for
every ordered pair of states (i, j) with clean increment(i) − clean increment(j) ≥ 0.3 pp, gap(j) ≥ gap(i) − 0.3 pp
(point estimates). Nothing here establishes a cause; row 4 would only justify proposing a confirmation.

**Reported regardless of the verdict:** absolute accuracies per state (base `Z`, target-fitted `q_Z`, `q_ZP`,
source-fitted `q_Z`, `q_ZP`; 12-cell macro and clean; also NLL/Brier); standalone accuracy of `P` and `Z`,
disagreement of argmax `P` with argmax `Z` and both-wrong rate (clean and per cell); gaps and clean increments at
2.5k×1; the probe and head hyperparameters and edge flags; base accuracy and robustness will differ across states,
so comparisons are of gaps **within** each state. The manipulation check and the dose values are reported per state.

## 6. Data and fitting access; exposure

Same 10,000 CIFAR-100 test images and 12 development cells as Stage 0 (labels used for T-regime fits, clean labels
for S), now on a **new model family**; training data are CIFAR-100 train/validation only. **Reserved resources are
not accessed:** checkpoints 1/3/5, the 11 unused CIFAR-100-C families, and any new test data. Exposure-ledger entry
(to be added to the authoritative ledger; see the card): development reuse of the 12 cells on an ImageNet-pretrained
ResNet-50 (four states), with fit/evaluation label roles as Stage 0; not confirmation.

## 7. Cost (measured in the smoke test, RTX 4090; `results/regime_map/smoke/smoke.json`)

Measured: fine-tuning 1,715 img/s (fp16 autocast, batch 128); strict-fp32 extraction 1,695 img/s; peak 6.6 GB.

| item | images | GPU-hours |
|---|---|---|
| fine-tuning, 10 epochs × 44,928 images | 449,280 | 0.07 |
| feature/probe extraction, per state (45k train + 5k val + 130k test) × 4 states | 720,000 | 0.12 |
| head and probe fits (GPU L-BFGS, ~1–2 min per state) | — | ≈ 0.05 |
| **subtotal** | | **≈ 0.25** |
| contingency ×2 (node retries, slower nodes) | | **≈ 0.5 GPU-h total** |

Stage 0 fits (CPU only, not GPU): 80 runs at ≈ 2–3 min on 4 cores ≈ 9 CPU-hours allocated. **Total GPU ≈ 0.5 h
≪ 24 GPU-h, so no cheaper variant is needed.**

## 8. Execution envelope

Immutable snapshot of a commit that contains this file; jobs and ledger under `results/regime_map/`. Prepared submit
script `atlas/regime_submit.py` (fine-tune → state extraction → Stage 0 fits → aggregation with `afterok`); **not run**.
Engineering recovery (node/CUDA failures, memory, timeout) is allowed and logged; a change of features, objective, grid,
folds, resolution policy, recipe, or exclusions is a scientific change requiring an amendment. No reserved data, no
push without approval. CPU numerics: pin node type and record `lscpu` (Stage 0 noise note).

## 9. Pre-launch verification (smoke test, not results)

Loader/schema, weights hash, fp32 extraction, fine-tune step, probe and head fits, and the Stage 0 loader format were
exercised on 600 train / 500 val / 300 test images (3 conditions) on one RTX 4090 (`smoke.json`): all passed. Numbers
from the smoke test are not used for any decision. The Stage 0 fit path with `--regime-state` was checked by feeding it a copy of the Stage 0 `layer3.22` arrays in the regime-map format (S-2.5k×1, fold 0, seed slot 0): the `q_ZP` probabilities matched Stage 0's to 0.0 (same node type not guaranteed in general; see §8).

## 10. Results

*(appended after the runs; §§0–9 unedited above this line)*
