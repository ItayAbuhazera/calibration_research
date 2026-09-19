---
type: failure_mode
status: closed_historical
evidence_scope: unpublished_repository_analysis
project: geometric-separation
scope: raw_pixel_geometry_custom_corruption_only
tags: [geometry, corruption, reversal, calibration]
---

# Clean-fitted geometric confidence mappings can reverse under synthetic corruption

## Scope (read narrowly)

The measured phenomenon is narrow: **a mapping selected/fitted on clean data
using the historical raw-pixel ("`geometric_physical`") geometric signal
performed poorly under this repository's synthetic-corruption protocol.**
This is not a general claim that "geometry fails under shift."

## Important distinction

The corruptions here are the repository's own custom synthetic transforms
(Gaussian noise, shift, rotation, and a combined "all" —
`utils/data_augmentation.py`), **not the CIFAR-C benchmark** (see [[CIFAR-C]]).

## Evidence

`geometric/GeometricCalibration/merged_complete_results/*/complete_results_*.csv`,
columns `ECE_all` (combined-corruption ECE) vs. `uncal`. Raw-pixel
clean-selected geometric calibrators show `ECE_all` worse than the
uncalibrated model in 100% of matched CIFAR-10/CIFAR-100 rows (149/149, both
MobileNet and EfficientNet). GTSRB is mixed: EfficientNet 75% (30/40 rows)
worse, MobileNet only 10% (4/40 rows) worse.

## Competing explanations (not adjudicated by this archaeology pass)

1. The geometric separation signal itself degrades under the pixel-level
   perturbation (the underlying distances become less informative).
2. The signal is still informative, but the fitted signal→correctness mapping
   (isotonic regression, fit on clean/validation data) no longer matches the
   corrupted distribution — a calibration-map-drift explanation distinct from
   signal loss.
3. Raw-pixel space is simply the wrong representation choice under these
   corruptions, and a different representation (semantic geometry) would not
   show the same reversal — partially supported by
   [[Semantic geometry is more corruption-resistant than pixel geometry but not shift-stable]],
   though semantic geometry is also not universally better than uncalibrated.
4. A historical metric artefact — e.g. an interaction with the ECE binning bug
   described in
   [[Historical geometric calibration ECE exports contain invalid perfect scores]] —
   could inflate or deflate the apparent reversal for specific rows. Not ruled
   out for any individual row in this dataset.

## What this does NOT establish

- It does not establish which of the four explanations above is correct, or
  whether more than one is simultaneously true.
- It does not establish anything about CIFAR-C or natural distribution shift.
- It does not establish that semantic (deep-feature) geometry solves this —
  semantic geometry is more corruption-resistant in relative terms but is
  itself not universally better than the uncalibrated model (see the related
  observation).
- It is **not the same finding** as the current project's
  [[Validation-fitted neighbourhood reliability features fail under corruption]]
  (a different codebase, a different geometric construction — RGC-style, not
  pixel-level "physical" geometry — and a different question, routing rather
  than raw ECE reversal). Treat the two as historically related but
  mechanistically distinct until a shared cause is demonstrated.

## Related project

[[Geometric Separation]]

## Related

- [[Semantic geometry is more corruption-resistant than pixel geometry but not shift-stable]]
- [[Raw-pixel geometric separation as a shift-robust calibrator]] (the resulting kill)
- [[Historical geometric calibration ECE exports contain invalid perfect scores]]
