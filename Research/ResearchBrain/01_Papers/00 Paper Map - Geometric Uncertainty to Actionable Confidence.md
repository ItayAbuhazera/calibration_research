---
type: paper_map
status: active
date: 2026-09-15
tags: [reading-map, geometry, calibration, actionable-confidence]
---

# Paper Map — Geometric Uncertainty → Actionable Confidence

This is not a generic calibration bibliography. A paper belongs here if it
helps explain one of the research links:

**representation / geometry → uncertainty signal → calibrated probability/risk → operational decision**

## Tier A — Must be in the vault

### [[Uncertainty Estimation Based on Geometric Separation]]
Foundation of the geometric-separation research line. Keep it as the root note.

### [[Semantic Geometric Calibration in Randomized Neural Feature Space]]
RGC / RGCL. Your direct continuation of the geometric-separation line.

### [[Beyond In-Domain Scenarios - Robust Density-Aware Calibration]]
DAC. Essential because it also derives calibration information from hidden-space
neighbourhood density and explicitly targets distribution shift.

### [[Taking a Step Back with KCal]]
**Yes, KCal belongs in the vault.** It is one of the closest conceptual neighbours:
representation-space geometry + class-conditional density/KDE + calibration.
It is especially important because your repository already contains a controlled
factorial study separating KCal's learned projection, density estimator, kernel,
reference bank, and fusion choices from RGCL-style alternatives.

### [[To Trust Or Not To Trust A Classifier]]
Trust Score. Important for the question "when should geometry override the head?"
It converts neighbour geometry into a correctness/trust signal rather than only
a calibrated probability.

### [[Beyond Confidence - Reliable Models Should Also Consider Atypicality]]
Atypicality-Aware Recalibration (AAR). Directly relevant to density / typicality
as a second signal beyond softmax confidence.

### [[A Simple Unified Framework for OOD and Adversarial Detection]]
Mahalanobis confidence in representation space. Important ancestor for
class-conditional feature-space distance as uncertainty evidence.

### [[Transitional Uncertainty with Layered Intermediate Predictions]]
TULIP. Relevant to multi-layer internal uncertainty signals and connects directly
to RGC's randomized intermediate representations.

## Tier B — Calibration foundations you should link, but do not over-note

### [[On Calibration of Modern Neural Networks]]
Temperature scaling / modern calibration baseline.

### [[Beyond Temperature Scaling - Dirichlet Calibration]]
Full-vector multiclass calibration. Particularly important for your discovered
top-label-ECE versus full-vector-proper-scoring frontier.

### [[Benchmarking Neural Network Robustness to Common Corruptions]]
CIFAR-C benchmark source. Needed for any corruption-shift observation.

## Tier C — Bridge to the next PhD direction

### [[SelectiveNet]]
Risk-coverage / reject-option framing. Helps move from "confidence score" to
"accept or abstain".

### [[Conformal Risk Control]]
A formal bridge from a score to a controlled operational risk target.

### [[Deep k-Nearest Neighbors]]
Useful ancestor for layer-wise nearest-neighbour evidence and representation-space
nonconformity.

## Reading policy

Do not create 20-page summaries. For each paper record:
1. the exact signal it uses;
2. what is calibrated / predicted;
3. what operational decision it supports;
4. the benchmark;
5. the strongest failure or limitation;
6. one link to your own observation graph.

## Most important conceptual triangle right now

- KCal: **can latent density produce calibrated full distributions?**
- Trust Score / AAR: **can geometry or atypicality indicate when the model is wrong?**
- Your current RGC-shift observation: **useful geometry exists, but can we know when to act on it under shift?**

That triangle is more useful for future idea discovery than another long list of
generic calibration papers.
