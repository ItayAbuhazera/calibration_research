---
type: paper
status: audited
year: 2024
venue: ICML
short_name: TULIP
tags: [layers, representations, uncertainty]
---

# Transitional Uncertainty with Layered Intermediate Predictions

## Why this matters

Uses intermediate-network information for uncertainty estimation in a
single-pass setting.

## Link

Direct conceptual neighbour to randomized multi-layer geometry in RGC.

## Literature extraction (audit 2026-09-15)

- **Signal:** disagreement / transitional information from intermediate representations before later layers collapse feature distinctions, coupled to a single-pass approximate-GP uncertainty estimator.
- **Target:** predictive uncertainty.
- **Decision:** confidence or uncertainty-aware downstream use; the paper is not a router between a base head and a geometric expert.
- **Can change argmax?** Not the central claim: its contribution is uncertainty estimation in a single-pass model, not post-hoc replacement of a fixed head's label.
- **Assumptions:** useful uncertainty-relevant distinctions exist in intermediate layers and can be preserved without undermining label-relevant compression.
- **Benchmark:** standard image-classification uncertainty benchmarks plus class imbalance, complex architectures, and medical/CT modalities.
- **Distribution shift?** Challenging deployment regimes are evaluated, but the paper is not a direct clean-to-corruption reliability-transfer study.
- **Does it ask when geometry itself is reliable?** No. It asks where in a network uncertainty information survives.
- **Strongest vault overlap:** supports the possibility that the current gate misses the relevant representation layer rather than merely needing another scalar neighbourhood statistic.
- **Remaining gap:** it does not evaluate ``geometry-better-than-head`` or operational recovery of complementary decisions.
- **Primary source:** https://proceedings.mlr.press/v235/benkert24a.html
