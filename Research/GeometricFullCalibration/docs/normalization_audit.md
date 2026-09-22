# Input-normalization audit and repair (2026-09-21)

**Verdict: the train/test mismatch is confirmed** for CIFAR-100 (and, in the
unified benchmark's corruption path, for CIFAR-10-C). It is repaired by one
authoritative specification, `utils/preprocessing_protocol.py`, selected from
**checkpoint provenance only** — never from test accuracy. Legacy behaviour is
preserved verbatim and labelled `legacy_v1_mixed_norm`; the repaired behaviour is
`corrected_v2_train_norm`. Labels used below: **[verified]** = read from code /
measured from an artifact in this session.

The earlier note in `docs/full_vector_dac_experiment.md` §2.1.2 described this
correctly in its second version (train/val CIFAR, test/CIFAR-C ImageNet); this
audit adds execution-path evidence, the third variant, and the paired diagnostic.

## 1. Provenance table

| stage | actual normalization | evidence |
|---|---|---|
| **checkpoint training** | CIFAR: mean (0.4914, 0.4822, 0.4465), std (0.2023, 0.1994, 0.2010); random-crop(4)+flip augmentation | [verified] `Experiments/train_model.py:317-323` calls `data.cifar100.get_train_valid_loader(augment=True)`; that function's `normalize` is CIFAR (`data/cifar100.py:50-53`) and is applied to train, valid (`:56-70`). The checkpoint dir's `config.json`/`results.json` match `train_model.py`'s output layout and fields (`baseline_cross_entropy_cifar100_resnet101_seed4/`). `data/cifar100.py` in the RGC training repo is **byte-identical** (`diff` empty; file dated 2026-01-07, checkpoints 2026-01-10). |
| **training-time validation record** | CIFAR | [verified] `results.json → evaluation_results` (seed 4: loss 0.886969, acc 77.0, ECE 0.044766) — **reproduced** by re-evaluating the seed-rebuilt validation split under CIFAR statistics (loss 0.886946, acc 77.00, ECE15 0.044841); the ImageNet-statistics value differs (loss 0.896345). Seed 2: recorded 0.881487 / 77.78 vs CIFAR-stats 0.881432 / 77.78 vs ImageNet-stats 0.902084 / 77.24. `results/preprocessing_audit/preprocessing_diagnostic_seed{2,4}.json` block A. |
| **reference bank** (DAC/kNN) | CIFAR (train split, `augment=False`) | [verified] `utils/model_utils.py::get_data_loaders` cifar100 branch calls `cifar100_loaders(augment=False)`; `run_unified_benchmark.py:4192` extracts native-DAC train features through that `train_loader`. Cached `train_raw.npy` value range −2.43…2.75 = (0−0.4914)/0.2023 … (1−0.4465)/0.2010. |
| **fitting** (validation inner-FIT; native DAC weights; TS/VS/…) | CIFAR (validation split) | [verified] same loader; cached `val_raw.npy` range −2.43…2.75. |
| **selection** (validation inner-SELECT) | CIFAR | [verified] same. |
| **clean test — legacy** | **ImageNet**: mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) | [verified] `data/cifar100.py::get_test_loader` (original lines 143-148); `get_data_loaders` → `cifar100_test(...)`; cached legacy `test_raw.npy` range −2.12…2.64 = (0−0.485)/0.229 … (1−0.406)/0.225. |
| **corruption test — legacy** | **ImageNet** | [verified] hard-coded `transforms.Normalize(mean=[0.485,…], std=[0.229,…])` in `run_unified_benchmark.py` (HEAD lines 3840-3844) for *every* `--dataset`; cached corrupted `test_raw.npy` range −2.12…2.64. For CIFAR-10 this is also a mismatch (`data/cifar10.py` train/test are CIFAR-normalized). |
| clean test / corruption — **corrected** | CIFAR (= training) | `utils/preprocessing_protocol.py`; `--preprocessing_protocol corrected_v2_train_norm` (default of the benchmark). |

Not double-normalized [verified]: the caches hold **post-transform** tensors
(`get_all_data_as_numpy` collects loader output), values in the ranges above,
and normalization is applied exactly once (test
`test_normalization_applied_exactly_once`). Channel order is RGB (torchvision
CIFAR datasets and the CIFAR-C `.npy` files are both RGB HWC uint8 → `ToTensor`
→ CHW float in [0,1]); dtype float32; no resize at 32 px; augmentation only in
the training loader. Preprocessing does **not** differ across the checkpoints
(one loader for all CIFAR-100 architectures trained by `train_model.py`). Tiny
ImageNet train/valid/test loaders all use ImageNet statistics (consistent);
CIFAR-10 clean test uses CIFAR statistics (consistent), only CIFAR-10-C in the
unified benchmark was mismatched. **Not audited, and refused by the corrected
protocol:** SVHN, PACS, DINOv2 image-size paths (`utils/model_utils.py` resize
wrapper is unchanged).

### Other variants found

* `Experiments/extract_corruption_features.py::_test_transform` used a **third**
  set (CIFAR-100 dataset statistics 0.5071/0.4865/0.4409, 0.2673/0.2564/0.2762)
  while its comment claimed to match the clean test pipeline. Amended to use the
  specification (`--preprocessing_protocol`; `--verify_against` compares under
  the protocol of the cell verified). No stored study output referenced any cell it
  produced.
* `Experiments/run_fv_dac_experiment.py::_corruption_transform` replicates the
  legacy ImageNet transform on purpose (the closed FV-DAC pilot used the frozen
  legacy Phase 0/1 state). It is now marked LEGACY in its docstring; the FV-DAC
  outputs stay as historical legacy-protocol results.
* **The legacy paper-reproduction path has the same clean-test issue.**
  `Experiments/run_post_hoc_calibration.py::get_data_loaders` (and
  `utils.model_utils.get_data_loaders`, used by `compare_dac_geometric.py`) call
  `cifar100.get_test_loader`, i.e. CIFAR-100 **clean-test** inputs were
  ImageNet-normalized for the RGC/IJCAI-era experiments too, unless a script
  built its own transform (`compare_dac_geometric.py:8873-8882`,
  `run_post_hoc_calibration.py:1292` use CIFAR statistics for their own CIFAR-C
  loaders). Those paths are **unchanged by default** (AGENTS.MD: keep the paper
  workflow reproducible); `run_post_hoc_calibration.py` gained an opt-in
  `--preprocessing_protocol`. **Whether any published CIFAR-100 clean number is
  affected is not established here and is not claimed** — it needs a
  dedicated check by the paper authors (see the paired-diagnostic magnitudes
  below for the size of the effect on the base model).

## 2. What changed

| item | change |
|---|---|
| `utils/preprocessing_protocol.py` (new) | authoritative constants per dataset/split/protocol, `eval_transform`, stamps, `require_compatible` |
| `data/cifar100.py::get_test_loader`, `utils/model_utils.py::get_data_loaders` | opt-in `preprocessing_protocol` parameter; default None = legacy, bit-for-bit |
| `Experiments/run_unified_benchmark.py` | `--preprocessing_protocol` (**default corrected — a deliberate default change for correctness**); CIFAR-C transform from the specification; protocol enters `_args_fingerprint` only when non-legacy (legacy fingerprints unchanged); `_guard_preprocessing_protocol` rejects incompatible `--fitted_state_dir`, `intermediates/` and `--reuse_non_metric_from`, then stamps corrected dirs; stamp recorded in `fit_once_provenance.json` and summary metadata. Legacy directories are never written to (unstamped = legacy). |
| `Experiments/extract_corruption_features.py`, `run_post_hoc_calibration.py` | see above |
| tests | `tests/test_preprocessing_protocol.py` (16); `tests/test_full_vector_dac.py::test_canonical_benchmark_behaviour_is_unchanged` narrowed from "runner byte-identical to HEAD" to "only the documented amendment lines were removed" |
| historical Slurm scripts | `scripts/phase0_1_*.sbatch`, `scripts/fv_dac_*.sbatch` are unchanged; because the benchmark default is now corrected, re-running them against their legacy output trees is **rejected** by the guard. To reproduce a legacy result deliberately add `--preprocessing_protocol legacy_v1_mixed_norm`. New corrected scripts: `scripts/corrected_v2_{fit_clean,evaluate_corruption}.sbatch`. |
| corrected trees | `results/studyAB/phase0_corrected_v2/` (new); `results/studyAB/phase0/` is untouched and legacy-labelled |

## 3. Artifact compatibility

| artifact | status |
|---|---|
| everything under `results/studyAB/phase0/**` (logits, features, `train_raw/val_raw/test_raw`, banks, fitted `*.pkl`/`*.pt`, per-sample arrays, summaries) | **legacy-labelled, incompatible with corrected**: test/corruption inputs used ImageNet statistics. Rejected by the guard (unstamped = legacy). Every method fitted there consumed CIFAR-normalized train/val but was evaluated on ImageNet-normalized test: **refit is required for a corrected run** (fitting inputs are unchanged, but the cached feature/logit *test* arrays and any calibrator whose fit consumed test-path arrays are invalid; the safe rule applied is: refit everything under a new tree). |
| `results/fv_dac/**` | legacy-labelled (built on the legacy frozen native DAC state and legacy test transform). Historical verdict unchanged. |
| checkpoints (`best_model.pth`) | valid — the repair changes evaluation inputs only. |
| materialized *train* and *validation* split arrays | numerically the same under both protocols (both CIFAR-normalized), but they live inside legacy directories; the corrected run re-materializes them and asserts identical labels/membership (test `test_split_membership_and_labels_unchanged_by_protocol`). |
| CIFAR-100-C raw `.npy` files, CIFAR-100 raw data | valid (raw pixels). |
| Tiny ImageNet, CIFAR-10 clean-test artifacts | protocol-invariant (train == test statistics); unaffected. CIFAR-10-C artifacts from the unified benchmark: legacy-labelled. |

## 4. Paired clean-input diagnostic (bounded; same images, same checkpoint)

`Experiments/preprocessing_diagnostic.py`, TF32 disabled, 10 000 clean CIFAR-100
test images; `results/preprocessing_audit/preprocessing_diagnostic_seed{2,4}.json`.

| | seed 4 | seed 2 |
|---|---|---|
| test accuracy legacy → corrected | 0.7632 → 0.7651 (+0.19 pp) | 0.7620 → 0.7659 (+0.39 pp) |
| test NLL legacy → corrected | 0.9377 → 0.9247 | 0.9506 → 0.9284 |
| top-1 agreement | 93.17 % | 92.89 % |
| corrected fixes / breaks a legacy decision (W / H) | 205 / 186 | 224 / 185 |
| mean abs. logit shift | 0.083 | 0.082 |
| val accuracy ImageNet-stats vs CIFAR-stats (same images) | 0.7696 vs 0.7700 | 0.7724 vs 0.7778 |

Representation / neighbour block (native-DAC layers, k = 200, bank = train under
CIFAR statistics), seed 4 (seed 2 similar):

| layer | dim | mean paired \|Δ s_l\| legacy-vs-corrected | test/val mean s_l ratio, legacy → corrected | top-10 neighbour overlap |
|---|---|---|---|---|
| conv1 | 64 | 0.0290 | 1.020 → 0.996 | 3.9 % |
| layer1 | 256 | 0.0089 | 0.954 → 0.995 | 20.7 % |
| layer2 | 512 | 0.0082 | 0.978 → 0.997 | 46.6 % |
| layer3 | 1024 | 0.0049 | 0.995 → 0.999 | 71.1 % |
| layer4 | 2048 | 0.0322 | 1.004 → 0.997 | 70.1 % |

Reading (labels: **measured**): under the legacy protocol clean-test kNN
statistics differed from the validation statistics by up to ≈ 4.6 % at `layer1`;
under the corrected protocol two clean, unseen splits are exchangeable to
≤ 0.6 %. The base-model effect is small in accuracy (+0.2 to +0.4 pp on clean
test) but the *neighbour structure* moves substantially at shallow layers. **This
does not establish how much any published or legacy calibration result changed**;
it quantifies the input perturbation only. Whether the mismatch changed any historical conclusion is an **unresolved interpretation**; the corrected baseline rows in `results/studyAB/phase0_corrected_v2/` allow that comparison for the two pilot seeds, but no such comparison has been made for the closed FV-DAC verdict, which stands as a legacy-protocol result.
