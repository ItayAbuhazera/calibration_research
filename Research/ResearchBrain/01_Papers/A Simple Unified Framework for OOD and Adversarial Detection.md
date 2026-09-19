---
type: paper
status: audited
year: 2018
venue: NeurIPS
tags: [mahalanobis, ood, class-conditional-geometry]
---

# A Simple Unified Framework for OOD and Adversarial Detection

## Why this matters

Canonical class-conditional representation-space distance baseline using
Mahalanobis structure across deep features.

## Link to current work

Useful ancestor for the general proposition that internal feature geometry can
carry error / OOD information beyond the softmax output.

## Literature extraction (audit 2026-09-15)

- **Signal:** maximum class-conditional Mahalanobis log-density from multiple hidden layers, using class means and a tied covariance estimated on labeled training data.
- **Target:** abnormality / OOD or adversarial status, rather than correctness of the predicted ID label.
- **Decision:** flag OOD/adversarial inputs; the paper also shows a class-incremental use after detecting an unknown sample.
- **Can change argmax?** The primary detector does not. Its Gaussian discriminant classifier can define a different class rule, but this is not the main operational evaluation.
- **Assumptions:** deep features are approximately class-conditionally Gaussian with shared covariance; distance to the nearest fitted class tracks abnormality.
- **Benchmark:** CIFAR-10/100 and SVHN with OOD datasets including TinyImageNet/LSUN; adversarial detection; noisy-label and low-sample stress tests.
- **Distribution shift?** Yes for OOD and adversarial inputs, but not as a clean-fitted local-competence gate under common corruption.
- **Does it ask when geometry itself is reliable?** No. It assumes the density score is the detector and evaluates detector discrimination, not its conditional reliability for selecting between two predictors.
- **Strongest vault overlap:** canonical class-conditional alternative to global kNN, relevant to the observed extra per-class oracle headroom.
- **Remaining gap:** OOD detection is not the same target as ``geometry-better-than-head`` on shifted but in-label-space samples.
- **Primary source:** https://arxiv.org/abs/1807.03888
