---
type: paper
status: core
year: 2026
venue: IJCAI-ECAI (preliminary preprint — not the version of record)
role: own_work
tags: [rgc, rgcl, rgcc, geometry, calibration, randomized-sampling]
---

# Semantic Geometric Calibration in Randomized Neural Feature Space

Abuhazera, Cohen, Einziger. IJCAI-ECAI 2026 preprint.

## Research question

Prior representation-space geometric calibrators (DAC, TULIP) need
architecture-specific layer selection. Is careful layer selection actually
required for effective geometric calibration, or can randomized sampling
replace it without a statistically significant loss?

## Method — RGC

Four-step pipeline (Algorithm 1):

1. **Random Index Sampling** — sample a fixed-size subset of representation
   components uniformly at random, once, and reuse it at calibration and test
   time (no performance-based, validation-adaptive layer choice).
2. **Feature Construction** — map sampled components to a compact, L2-normalized
   embedding `z(x)`.
3. **Geometric Scoring** — nearest-same-class vs. nearest-different-class
   distance in `z(x)`-space (Eq. 11), the same separation-score family as
   [[Uncertainty Estimation Based on Geometric Separation]], now applied to a
   learned/aggregated representation instead of raw pixels.
4. **Calibration Mapping** — rank-normalize scores (empirical CDF against the
   validation set) then fit isotonic regression to correctness.

Two instantiations of steps 1–2:

- **RGCL**: samples `L` intermediate *layers* uniformly at random, applies
  Spatial Pyramid Pooling (4×4/2×2/1×1) per sampled layer, projects each pooled
  vector with an i.i.d. Gaussian random projection to dimension `d`, sums the
  projected vectors, then L2-normalizes.
- **RGCC**: samples `K` individual activation *coordinates* uniformly at random
  from the global concatenated-activation space (no layer subsetting, no SPP,
  no projection) — cheaper feature construction, lower preprocessing cost.

Default hyperparameters used throughout (stated by the paper, not tuned per
dataset): **L = 6, d = K = 256**.

## Published results

- Datasets: CIFAR-10, CIFAR-100, Tiny-ImageNet. Architectures: ResNet-18/50/101/152,
  DenseNet-121, DINOv2-Large (linear probe). 13 model–dataset combinations used
  for the headline table (Table 1) and equivalence tests (Table 2).
- RGCL reduces ECE by **~78% on average** vs. uncalibrated, up to **~97%** on
  CIFAR-100/DINOv2-Large (Table 1). Average rank ~1.71 across all 14 rows/methods
  (1.21 excluding RGCC), first-or-second in 12/14 settings.
- Against the best non-RGC baseline (mostly DAC/Isotonic), RGCL improves ECE by
  ~12% on average, up to ~19% on individual configurations.
- **RGCL vs. RGCC (Table 2, TOST equivalence, margin δ = 0.5 pp):** formal
  equivalence established in **9 of 13** model–dataset combinations. The 4
  combinations that do **not** establish equivalence are TINY/resnet101,
  TINY/resnet152, TINY/resnet50, and CIFAR100/densenet121 — the paper reports
  this as inconclusive (higher variance / fewer paired trials), not as
  inequivalence.
- **RGCL vs. layer-selected baselines GC(DAC)/GC(TULIP) (same table/margin):**
  equivalence established in **10 of 13** combinations — a *different* count
  from the RGCL-vs-RGCC comparison; do not conflate the two "9 vs 13" and
  "10 vs 13" numbers.
- Randomized layer sampling is argued to work (Section 5.4) because layers with
  strong geometric separation empirically tend to have disproportionately large
  L2 norm after pooling/projection (5.5×–34.6× early-to-late increase across
  three architectures on CIFAR-100), so summing projected sampled layers
  implicitly up-weights the informative ones. The paper explicitly labels this
  "an empirical explanation rather than a worst-case guarantee."
- Computational trade-off (Section 5.5): RGCL has the highest one-time
  offline preprocessing cost; RGCC substantially reduces it but remains higher
  than DAC/Fast-Separation preprocessing. Online, RGCL and RGCC have **nearly
  identical inference throughput** (same scoring procedure); DAC is faster than
  both online; Fast-Separation (pixel-space) is slower and scales with input
  dimensionality.
- Hyperparameter sensitivity (Fig. 1): across `L ∈ {2,4,6,8,10,12}` and
  `d ∈ {64,...,1024}` (30 configurations), mean ECE stays within [1.10, 1.22]%
  — the chosen (L=6, d=256) is not a cherry-picked optimum.

## Published limitations (stated by the authors)

- Evaluation is **in-distribution image classification only**.
- Corruption, domain shift, and non-vision modalities are explicitly named as
  future work — not evaluated in this paper.
- The geometric index (nearest-neighbor structure over training representations)
  must be rebuilt/updated whenever the calibration data changes; this is named
  as a limitation of the offline-preprocessing design, not solved here.
- The "why RGCL works" norm-dominance explanation is explicitly qualified: in
  architectures where late-layer norms do not dominate, the implicit weighting
  effect may be weaker.

## Repository follow-up / unpublished evidence

See [[Semantic Geometric Calibration RGC]] (project page) for the
separately-sourced account of what the `GeometricInternalCalibration` repository
contains beyond this paper: an unsupervised composite layer-selector experiment,
OOD-detection-vs-ID-calibration dissociations, an RGCL/RGCC equivalence analysis
using a different margin than this paper's δ=0.5pp, and other integrity flags.
These are **repository-only, unpublished** findings and are kept out of this
note deliberately — do not read them as amendments to what this paper claims.

## Direct descendants in the vault

- [[Full-Vector Geometric Calibration]] — extends RGC-style geometric scoring
  from top-label calibration into full-vector / decision-changing calibration,
  KCal comparisons, and shift/recoverability questions.
- [[2026-04-25 CIFAR100 ResNet18 Full-Vector Path Batch]]
- [[2026-04-27 TinyImageNet ResNet50 Full-Vector Path Batch]]
- [[2026-09-15 RGC Shift Recoverability]]
