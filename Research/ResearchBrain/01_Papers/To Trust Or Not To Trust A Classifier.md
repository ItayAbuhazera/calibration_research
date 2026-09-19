---
type: paper
status: audited
year: 2018
venue: NeurIPS
short_name: Trust Score
tags: [trust, nearest-neighbors, selective-decision]
---

# To Trust Or Not To Trust A Classifier

## Why this matters

Trust Score asks a question close to the current bottleneck: whether local
geometry can distinguish trustworthy from untrustworthy model decisions.

## Read it for

- the geometric correctness signal;
- its assumptions;
- how it behaves when local neighbourhood structure shifts;
- how it differs from using geometry merely as a probability calibrator.

## Link

[[Validation-fitted neighbourhood reliability features fail under corruption]]

## Literature extraction (audit 2026-09-15)

- **Signal:** ratio of distance to the nearest alternative class high-density set versus distance to the predicted-class high-density set; can use intermediate representations.
- **Target:** correctness / trustworthiness of an already-trained classifier's hard prediction.
- **Decision:** trust, flag, reject, or send the existing prediction to a human; it does not prescribe a correction label.
- **Can change argmax?** No; the score leaves the classifier's label unchanged.
- **Assumptions:** labeled reference data estimate class high-density sets; relevant geometry is low- or medium-dimensional / manifold-like; high trust corresponds to agreement with the Bayes classifier under its nonparametric assumptions.
- **Benchmark:** UCI tasks plus MNIST, SVHN, CIFAR-10, and CIFAR-100; the CIFAR-100 result is reported as essentially negative.
- **Distribution shift?** No controlled corruption or OOD benchmark.
- **Does it ask when geometry itself is reliable?** It asks when the *head prediction* is reliable using geometry, not when a geometry-based alternative prediction is reliable under shift.
- **Strongest vault overlap:** almost exactly the label-free local-competence question behind the killed neighbourhood-statistics gate.
- **Remaining gap:** no shift-transfer test and no head-versus-geometry oracle objective; high-dimensional degradation is an explicit warning for the current representation setting.
- **Primary source:** https://arxiv.org/abs/1805.11783
