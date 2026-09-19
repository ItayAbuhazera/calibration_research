---
type: paper
status: audited
year: 2023
venue: NeurIPS
short_name: AAR
tags: [atypicality, density, recalibration]
---

# Beyond Confidence — Reliable Models Should Also Consider Atypicality

## Why this matters

AAR makes atypicality an explicit second axis beyond model confidence.

## Relevance

It is a strong conceptual comparator for any claim that representation-space
density or distance provides information not already present in logits.

## Important implementation note

The repository's `AARLightweightCalibrator` is explicitly **not** the official
AAR method; it uses a nearest-neighbour atypicality proxy instead of the
paper's GMM-based class-conditional atypicality. Keep paper conclusions separate
from results of the lightweight approximation.

## Literature extraction (audit 2026-09-15)

- **Signal:** input and class atypicality from class-conditional Gaussian density in penultimate embeddings, combined with model confidence.
- **Target:** conditional miscalibration, lower accuracy, and prediction-set coverage variation associated with atypical inputs/classes.
- **Decision:** atypicality-aware recalibration and prediction-set construction.
- **Can change argmax?** Yes. AAR includes a class-dependent additive correction in log-probability space; the authors explicitly attribute its accuracy changes to that term.
- **Assumptions:** class-conditional embedding density is adequately represented by shared-covariance Gaussians; atypicality measured on the calibration/reference distribution remains meaningful for the evaluated groups.
- **Benchmark:** ImageNet, CIFAR-10/100, MNLI; long-tailed ImageNet/CIFAR/Places; Alpaca-7B classification tasks; Fitzpatrick17k skin-lesion case study.
- **Distribution shift?** No controlled corruption/OOD shift experiment; long-tail and demographic subgroup analyses are not substitutes for that evaluation.
- **Does it ask when geometry itself is reliable?** It stratifies reliability by geometric atypicality, but does not test whether atypicality identifies when a second geometric predictor should replace the head.
- **Strongest vault overlap:** directly challenges a confidence-only gate and demonstrates that an internal density signal can support decision-changing post-hoc correction.
- **Remaining gap:** no oracle complementarity analysis, no clean-to-corruption transfer assessment, and the current repository proxy is not the official AAR method.
- **Primary source:** https://arxiv.org/abs/2305.18262
