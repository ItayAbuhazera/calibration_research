---
type: observation
status: closed_historical
evidence_scope: unpublished_repository_analysis
project: geometric-separation
evidence_strength: 2
tags: [geometry, semantic, pixel, corruption, gtsrb]
---

# Semantic geometry is more corruption-resistant than pixel geometry but not shift-stable

## Evidence source

`geometric/GeometricCalibration/embedders/image_embedders.py` (multiple deep
embedders behind a `SemanticModel` interface — CLIP, EfficientNet-Lite,
MobileNetV3, MobileViT, SigLIP, VAE, TinyVAE) and the `Calibration_Technique`
column (`geometric_physical_*` vs. `geometric_semantic_*`) in
`merged_complete_results/*/complete_results_*.csv`.

## Important distinction

**This is the repository's own custom synthetic-corruption protocol
(Gaussian noise / shift / rotation, and a combined "all" — 4 conditions total,
implemented in `utils/data_augmentation.py`), NOT the CIFAR-C benchmark.**
See [[CIFAR-C]] for the actual benchmark note used elsewhere in this vault.
Do not read "corruption" in this note as CIFAR-C.

## Measured fact

Across the CIFAR-10/CIFAR-100 × MobileNet/EfficientNet combinations
(4 dataset-model combos × 4 corruption conditions):

- Semantic (deep-feature) geometry beats pixel ("physical") geometry on
  combined-corruption ECE in **100%** of matched rows.
- Semantic geometry is **not** universally better than the uncalibrated model:
  on CIFAR-10 it is worse than uncalibrated in ~95–100% of rows; on
  CIFAR-100/EfficientNet it is worse in only ~20% of rows (i.e. mostly better
  there).
- On GTSRB, the result is architecture-dependent: semantic geometry beats
  uncalibrated on MobileNet in ~85% of rows but is worse than uncalibrated on
  EfficientNet in ~81% of rows, under the identical corruption protocol.

## Interpretation

Moving from raw-pixel to semantic (deep-feature) geometry improves relative
robustness to this repository's synthetic corruptions, but does not by itself
guarantee an improvement over the uncalibrated model, and the direction of
that comparison depends on the architecture (GTSRB).

## What this does NOT establish

- This is not evidence about CIFAR-C or any standard corruption benchmark.
- It does not establish which architectural property of EfficientNet vs.
  MobileNet drives the GTSRB reversal.
- It does not establish that the *current* project's RGC-shift work
  (CIFAR-100-C, a different codebase, a different geometric construction, and
  a different question — routing vs. raw ECE) shares a mechanism with this
  historical result. See caution below.

## Relationship to the current RGC-shift observation — not the same failure mechanism

[[Geometry contains complementary accuracy information under corruption]] (the
current `GeometricFullCalibration` project, CIFAR-100-C, RGC-style features)
is a **different, later observation**: it finds oracle-level complementary
accuracy information under real CIFAR-100-C corruption, using a different
geometric construction and a different research question (whether the head
should be overridden), not a raw ECE comparison of physical vs. semantic
geometry. **Do not treat this note and that one as the same phenomenon** —
link them as related lineage, not as duplicate evidence for one claim.

## Related project

[[Geometric Separation]]

## Related later observations

- [[Geometric separation substantially improves clean CIFAR top-label ECE]]
- [[Geometry contains complementary accuracy information under corruption]] (related lineage, not the same mechanism)
