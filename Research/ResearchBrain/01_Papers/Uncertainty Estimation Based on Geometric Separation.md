---
type: paper
status: core
year: 2023
venue: JMLR
role: research_line_root
tags: [geometry, uncertainty, calibration, fast-separation]
---

# Uncertainty Estimation Based on Geometric Separation

Chouraqui, Cohen, Einziger, Leman. JMLR 23 (2023).

## Research question

Can the geometric distance of an input from existing training inputs serve as
a better signal for confidence estimation than a model's native confidence,
once both are passed through the same post-hoc calibration machinery?

## Method

- **Separation** (Definition 3): for input `x`, let `F_M(x)` be same-predicted-class
  training points and `F̄_M(x)` be all other training points. `x` is *safe* if it
  is closer to `F_M(x)` than to `F̄_M(x)`, *dangerous* otherwise. The separation
  score is the signed radius of the maximal ball around `x` that stays on one
  side of this boundary (positive for safe, negative for dangerous).
- **Fast-separation** (Definition 5): a cheap approximation using only
  `D(x, F̄_M(x)) − D(x, F_M(x))`, over any metric (not just L2), avoiding the
  expensive triplet search of the exact separation measure.
- **Calibration**: the scalar (fast-)separation score is mapped to a confidence
  value by fitting isotonic regression (or a sigmoid) against empirical accuracy
  on a held-out validation set — the same post-hoc recipe used for calibrating a
  model's native confidence, just applied to a new signal instead.
- **Dimensionality/runtime reduction**: pooling, max-pooling, PCA, bilinear
  resize, random pixel sampling, random training-set subsampling (`Randset`),
  and k-means centroid replacement are evaluated as ways to make the
  training-set distance computation tractable for near real-time use.

## Published evidence (scope)

- Datasets: MNIST, Fashion-MNIST, GTSRB, Sign-Language-MNIST, CIFAR-10, plus two
  tabular datasets (wine quality, airline passenger satisfaction).
- Models: Random Forest, Gradient Boosted Trees, and a CNN. **No MobileNet or
  EfficientNet, no ImageNet-scale models.**
- Signal is computed on **raw (normalized) pixel/feature space only** — there is
  no semantic/deep-representation variant of geometric separation in this paper.
- Table 2: fast-separation (`S_M`) and exact separation (`S̄_M`) give
  near-identical ECE and both beat Iso, Platt, SBC, HB, BBQ, Beta, TS, ETS in
  almost every dataset/model cell; the one loss is CNN on Fashion-MNIST
  (−5.6%, Table 3).
- Reported improvement: "up to 99%" ECE reduction vs. alternatives, model- and
  dataset-dependent (Table 3).
- Dimensionality-reduction methods (pooling/PCA/RBI/Randpix/Randset/k-means)
  preserve most of the ECE benefit while giving large throughput gains (Figs. 6–8).
- Norm choice (L1/L2/L∞, Table 1): no norm is universally best across datasets.
- Tabular data: a smaller, non-uniform 1–77% accuracy improvement; a few cases
  where plain isotonic regression on the model's native confidence wins instead.

## Assumptions and scope stated by the paper

- Inputs must be normalized (same size/scale) for the distance computation to
  be meaningful across inputs.
- The method is evaluated against the **uncalibrated** model, and separately
  against other post-hoc calibrators applied to the model's native confidence —
  it does not modify the underlying model.
- No corruption/shift evaluation and no CNN-middle-layer ("semantic") geometric
  variant are run; both are explicitly named as future work.

## Published limitations (stated by the authors)

- Method assumes normalized, fixed-size inputs; variable-sized images are future
  work.
- Exact separation is computationally expensive; fast-separation and the
  reduction methods trade some fidelity for speed, though the paper reports
  the loss as small.
- Future direction stated explicitly in the Conclusion: replace raw images with
  a CNN middle-layer latent vector as the geometric feature space — this is the
  seed of the later semantic/representation-space line, but it is **not**
  executed or evaluated in this paper.

## Repository follow-up / unpublished evidence

See [[Geometric Separation]] (project page) for the full, separately-sourced
account of what the `GeometricCalibration` repository contains beyond this
paper — clean CIFAR results on additional (deep) models, a semantic-vs-pixel
geometry comparison, a clean-to-corruption reversal, and several integrity
flags in the historical exports. These are **repository-only findings**, not
part of what this paper claims, and are kept out of this note deliberately.

## Direct descendants in the vault

- [[Semantic Geometric Calibration in Randomized Neural Feature Space]] —
  moves the geometric signal from raw pixel space into intermediate neural
  representation space, exactly the direction sketched in this paper's
  Conclusion.
- [[Geometry contains complementary accuracy information under corruption]]
- [[Predicting geometric reliability under distribution shift]]
