---
type: project
status: completed
canonical_repo: ../geometric/GeometricCalibration
parent_project: null
paper: "[[Uncertainty Estimation Based on Geometric Separation]]"
tags: [geometry, calibration, fast-separation, corruption]
---

# Geometric Separation

## Research question

A model's native confidence is only one signal for estimating whether a
prediction is correct. Does the geometric separation of an input from its
same-class and competing-class training examples provide a better signal,
once both are calibrated the same way post-hoc? (Paper framing, see
[[Uncertainty Estimation Based on Geometric Separation]].)

## Core method (high level)

- Compute a scalar geometric-separation score from the distance of an input
  to its nearest same/predicted-class training examples versus its nearest
  competing-class training examples.
- Map that scalar to a calibrated probability via standard post-hoc
  calibration (isotonic regression), exactly as one would calibrate a model's
  native confidence.
- Fast-separation and several data-reduction methods (pooling, PCA, random
  pixel/training-set sampling, k-means) address the runtime cost of computing
  distances against the full training set.

See the paper note for the exact definitions/formulas; they are not
reconstructed here.

## Published contribution

See [[Uncertainty Estimation Based on Geometric Separation]] for the full,
paper-sourced account (scope: MNIST/Fashion/GTSRB/SignLang/CIFAR-10/tabular,
RF/GB/CNN, raw-pixel-only geometry).

## Repository-only findings

Everything below is from the `GeometricCalibration` repository only. It goes
beyond the published paper (extra models, a semantic/deep-feature geometry
variant, and a corruption evaluation the paper does not run) and carries the
repository's own historical analysis discipline, **not** the paper's. None of
it should be read as amending what the paper claims.

### A. Clean CIFAR calibration (MobileNet / EfficientNet)

Evidence: `master_geometric_comparison.csv`,
`statistical_analysis_by_config/statistical_summary_*_RS103-120.csv`,
`merged_complete_results/{cifar10,cifar100,gtsrb}_{mobilenet,efficientnet}/`.

| Dataset | Model | Uncal ECE | Best geometric ECE | Best config | Reduction |
|---|---|---|---|---|---|
| CIFAR-10 | MobileNet | 0.0198 | 0.00971 | geometric_physical, binned, layer `features_18_0` | ~51% |
| CIFAR-10 | EfficientNet | 0.0188 | 0.01103 | geometric_physical, binned, `avgpool` | ~41% |
| CIFAR-100 | MobileNet | 0.0580 | 0.01347 | geometric_physical, unbinned, `features_18_0` | ~77% |
| CIFAR-100 | EfficientNet | 0.0701 | 0.01368 | geometric_physical, unbinned, `avgpool` | ~80% |
| GTSRB | MobileNet | 0.0366 | 0.02891 | geometric_semantic, unbinned | ~21% |
| GTSRB | EfficientNet | 0.0312 | 0.04052 | geometric_semantic, unbinned | **worse** (+30%) |

On CIFAR-10/CIFAR-100, geometric separation substantially reduces clean
top-label ECE for both extra (non-paper) models. On GTSRB/EfficientNet, the
best available geometric configuration is *worse* than uncalibrated even on
clean data — a genuine counter-example, not a universal win.

These are historical exploratory analyses: single best-config summaries per
dataset/model, without the paired/matched-seed testing discipline used in the
current (`GeometricFullCalibration`) work. See integrity flags below regarding
seed counts behind these numbers.

Full note: [[Geometric separation substantially improves clean CIFAR top-label ECE]]

### B. Clean-to-synthetic-corruption reversal

Evidence: `utils/data_augmentation.py` (`transform_test_set_noise`, `_shift`,
`_rotate`, `_all`), result columns `ECE_noise/ECE_shift/ECE_rotate/ECE_all` in
`merged_complete_results/*/complete_results_*.csv`.

**These are custom synthetic corruptions (Gaussian-noise / shift / rotation,
and a combined "all"), NOT the CIFAR-C benchmark.** The repository implements
exactly 3 corruption types plus a combined variant — 4 conditions in total.
Evaluated across the 4 CIFAR dataset/model combinations (CIFAR-10/100 ×
MobileNet/EfficientNet), that gives 4 × 4 = 16 corruption-condition ×
dataset-model comparisons — this is the origin of "16," not 16 distinct
corruption types.

Raw-pixel ("`geometric_physical`") clean-selected calibrators become **worse**
than uncalibrated on the combined-corruption ECE in 100% of matched CIFAR-10
and CIFAR-100 rows (149/149). GTSRB is mixed (EfficientNet 75%, MobileNet 10%).

Full note: [[Clean-fitted geometric confidence mappings can reverse under synthetic corruption]]

### C. Semantic vs. pixel geometry

Evidence: `embedders/image_embedders.py` (`SemanticModel` ABC and multiple deep
embedders — CLIP, EfficientNet-Lite, MobileNetV3, MobileViT, SigLIP, VAE,
TinyVAE), same result CSVs as above, `Calibration_Technique` column
distinguishing `geometric_physical_*` from `geometric_semantic_*`.

- Semantic (deep-feature) geometry beats pixel geometry on combined-corruption
  ECE in 100% of matched CIFAR-10/CIFAR-100 rows — more corruption-resistant.
- Semantic geometry is **not** universally better than the uncalibrated model:
  on CIFAR-10 it is worse than uncalibrated on ~95–100% of rows; on
  CIFAR-100/EfficientNet it is worse on only ~20% of rows (i.e. mostly better
  there).
- GTSRB is architecture-dependent: semantic geometry helps
  (beats uncalibrated) on MobileNet in ~85% of rows but hurts on EfficientNet
  in ~81% of rows under the same corruption protocol.

Full note: [[Semantic geometry is more corruption-resistant than pixel geometry but not shift-stable]]

### D. Regime-dependent representation choice

Evidence: `master_geometric_comparison.csv`.

The best space (physical vs. semantic), binning strategy, and layer are not
constant across datasets or even across models on the same dataset: CIFAR-10
favors physical+binned; CIFAR-100 favors physical+unbinned; GTSRB favors
semantic+unbinned but with a *different* best layer per architecture
(`features_18_0` for MobileNet vs. `features_6_0_block_0` for EfficientNet).
("Binned"/"unbinned" here is the geometric calibrator's internal
fast-separation binning strategy, not the ECE histogram bin count.)

Full note: [[Geometric calibration space layer and binning are regime-dependent]]

## Historical integrity flags

- **Platt zero-ECE artifact** — *suspicious artifact / likely mechanism*.
  The Platt calibrator reports exactly `0.0` ECE (all of ECE/ECE_noise/
  ECE_shift/ECE_rotate/ECE_all) in 224/237 matched rows across CIFAR-10/100.
  Plausible mechanism (not fully proven as sole cause): `PlattCalibrator.fit`
  (`calibrators/calibrators.py` L292–321) trains an unregularized per-class
  logistic regression on a single scalar feature, which can saturate
  `predict_proba` to exactly `1.0` for high-confidence points; combined with
  the binning bug below, those points are silently dropped from every ECE bin,
  driving computed ECE toward 0.
- **ECE binning bug at confidence == 1.0** — *verified implementation issue*.
  `utils/metrics.py`, `CalibrationMetrics.ece()` (and `.mce()`/`.ace()`/
  `.binned_likelihood()`) computes
  `bin_indices = np.digitize(confidence, bin_boundaries) - 1` with
  `bin_boundaries = np.linspace(0, 1, n_bins + 1)`. `np.digitize` (default
  `right=False`) maps any confidence `>= 1.0` to index `n_bins`, one past the
  last valid bin (`0..n_bins-1`); the aggregation loop `for i in
  range(n_bins)` never visits that index, so points with confidence exactly
  1.0 contribute zero error while still counting in the denominator —
  systematically deflating ECE whenever any prediction saturates to exactly
  1.0. This is the production ECE path (`utils/calibration.py`,
  `scripts/main_layer.py`, `n_bins=15` or `20`).
- **`fast_separation` / `StabilitySpace` alias duplication** — *verified
  implementation issue*. Two independently-defined `StabilitySpace` classes
  exist with identical constructor signatures: `utils/utils.py` (CPU-only) and
  `geometric/stability_space.py` (adds GPU args). `calibrators/geometric_calibrators.py`
  L13–14 contains a live import of the GPU version with the CPU version's
  import commented out directly below it — direct evidence of manual,
  undocumented switching between two maintained copies (silent-divergence
  risk).
- **Random-state / seed count discrepancies** — *verified implementation
  issue*. Filenames imply a nominal RS103–120 range (18 seeds), but actual
  seed counts per dataset/model are 10–11 (`N_Random_States` column in
  `statistical_analysis_by_config/statistical_summary_*.csv`) — already short
  of nominal. Some individual technique/layer/binning cells are far shorter
  than the rest of their own table: e.g. `geometric_semantic_binned_avgpool`
  at `layer_avgpool`/binned in the GTSRB/EfficientNet summary has `N=1`
  against `N=10` everywhere else in that file; several CIFAR-100/EfficientNet
  cells at `layer_avgpool`+unbinned drop to `N=5–7` against `N=10–11`
  elsewhere.

Full note: [[Historical geometric calibration ECE exports contain invalid perfect scores]]

## Unresolved questions

- Does clean-to-corruption reversal reflect loss of the underlying geometric
  signal, drift in the signal-to-correctness calibration map, or an
  inappropriate space/layer choice under shift? (Competing explanations are
  not adjudicated by this archaeology pass — see the corresponding failure
  mode note.)
- Is there a representation/layer selection rule that transfers across
  datasets and architectures, or is the regime-dependence in section D
  fundamental?
- These historical results use ECE only; a corrected evaluation using
  NLL/Brier/selective risk has not been done on this repository's exports.
- The paper's scalar (top-label, predicted-class) construction vs. a
  full-vector construction is not tested anywhere in this repository.

## Killed / parked directions

[[Raw-pixel geometric separation as a shift-robust calibrator]] — killed
narrowly, scoped to clean-fitted raw-pixel geometry under this repository's
synthetic-corruption protocol.

## Artifact ledger

| Finding | Artifact | Evidence type | Integrity |
|---|---|---|---|
| Clean CIFAR ECE reduction, MobileNet/EfficientNet | `master_geometric_comparison.csv`, `merged_complete_results/{cifar10,cifar100}_*` | exploratory repository run | exploratory — seed counts below nominal |
| GTSRB/EfficientNet clean geometric calibration worse than uncalibrated | `master_geometric_comparison.csv` | exploratory repository run | exploratory |
| Raw-pixel clean-to-corruption reversal (100% of CIFAR10/100 rows) | `merged_complete_results/*/complete_results_*.csv` (`ECE_all` vs `uncal`) | exploratory repository run (custom corruptions, not CIFAR-C) | exploratory |
| Semantic > pixel geometry under corruption; not universal vs. uncalibrated | same result CSVs, `Calibration_Technique` column | exploratory repository run | exploratory |
| Best space/layer/binning varies by dataset & model | `master_geometric_comparison.csv` | exploratory repository run | exploratory |
| Platt exactly-0 ECE across clean + 3 corruptions | `merged_complete_results/*/complete_results_*.csv` (Platt rows) | implementation inspection + CSV pattern | known metric bug (interacting with binning bug) |
| ECE binning bug at confidence==1.0 | `utils/metrics.py::CalibrationMetrics.ece()` | implementation inspection | known metric bug |
| Duplicate `StabilitySpace` (CPU/GPU) with manual import switching | `utils/utils.py`, `geometric/stability_space.py`, `calibrators/geometric_calibrators.py` L13-14 | implementation inspection | known code-duplication risk |
| Seed counts short of nominal RS103-120, some cells as low as N=1 | `statistical_analysis_by_config/statistical_summary_*.csv` | implementation/CSV inspection | under-replicated |

## Descendant project

[[Semantic Geometric Calibration RGC]]
