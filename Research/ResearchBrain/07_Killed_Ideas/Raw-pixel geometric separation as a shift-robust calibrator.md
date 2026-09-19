---
type: killed_idea
project: geometric-separation
evidence_scope: unpublished_repository_analysis
reason: clean-fitted raw-pixel geometric calibrators reverse under the repository's synthetic-corruption protocol
tags: [geometry, pixel, corruption, negative-result]
---

# Raw-pixel geometric separation as a shift-robust calibrator

## Scope of the kill (read narrowly)

Killed only for: **clean-fitted raw-pixel ("`geometric_physical`") geometric
calibration, evaluated under the `GeometricCalibration` repository's recovered
synthetic-corruption protocol** (custom Gaussian noise / shift / rotation /
combined — not CIFAR-C).

**Not killed:** raw-pixel geometry in general, raw-pixel geometry under any
other shift protocol, or geometry as a signal family more broadly. Semantic
(deep-feature) geometry is a *different*, more corruption-resistant
representation choice within the same repository and is not covered by this
kill (see [[Semantic geometry is more corruption-resistant than pixel geometry but not shift-stable]]
— though it is also not universally better than uncalibrated).

## Evidence that killed it

[[Clean-fitted geometric confidence mappings can reverse under synthetic corruption]] —
raw-pixel clean-selected calibrators are worse than uncalibrated in 100% of
matched CIFAR-10/CIFAR-100 rows under this protocol.

## Why it died

For the specific combination of (raw pixel space) × (clean-only fitting) ×
(this repository's synthetic corruption family), the resulting calibrator was
reliably harmful rather than merely unhelpful.

## Conditions under which to revisit

Only with a genuinely different fitting protocol (e.g. corruption-aware
fitting, not clean-only) or a different corruption family (e.g. CIFAR-C) — not
by re-tuning the same clean-fitted raw-pixel construction on the same protocol.

## Related project

[[Geometric Separation]]
