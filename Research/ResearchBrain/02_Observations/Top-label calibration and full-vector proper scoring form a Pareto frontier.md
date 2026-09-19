---
type: observation
status: supported_exploratory
date: 2026-04-25
project: full-vector-calibration
evidence_strength: 3
tags: [pareto-frontier, ece, nll, brier]
---

# Top-label calibration and full-vector proper scoring form a Pareto frontier

## Observation

On the CIFAR-100 multi-seed batch, top-label-specialized geometric methods and
full-vector distance fusion optimize different objectives.

A particularly clear comparison:

- `rgcl_tail_vector_scaling` top-label ECE: ~0.0321
- `full_vector_distance_fusion` top-label ECE: ~0.0884
- paired top-label-ECE difference: ~-0.0564, CI excluding zero favorably

But:

- `rgcl_tail_vector_scaling` NLL: ~1.4646
- fusion NLL: ~1.3036
- `rgcl_tail_vector_scaling` accuracy: ~0.6317
- fusion accuracy: ~0.6576


## Method naming

Method naming resolved 2026-09-15 by author decision: the top-label specialist is called `rgcl_tail_vector_scaling` everywhere; `gc_dac` was an earlier informal name for the same method. This is a naming choice, not a re-check of the run outputs.

## Interpretation

The method that is better calibrated in the top-label sense is not necessarily
better as a full predictive distribution or as a decision rule.

## Why this matters

A single scalar "calibration quality" ranking can hide a real multi-objective
trade-off.

## Research consequence

Whenever a new method claims improvement, ask separately:

1. Does it improve confidence calibration of the chosen class?
2. Does it improve the full predictive distribution?
3. Does it improve decisions / accuracy?
4. At what operating point?

## Scope

Exploratory, carded after the run, CIFAR-100 / ResNet-18 only. The vault never
records fusion accuracy against the **base model** on CIFAR-100 - only against
the top-label specialist - so "fusion improves decisions" is not established here.

## Related

- [[2026-04-25 CIFAR100 ResNet18 Full-Vector Path Batch]]
- [[High-confidence anchoring can block useful full-vector decision changes]]
- [[Top-coordinate recalibration plus tail renormalization can change argmax]]

## Connection to the RGC paper's own scope

[[Semantic Geometric Calibration in Randomized Neural Feature Space]] states
its own evaluation is top-label-ECE-first ("Since RGC calibrates top-label
confidence, ECE is our primary metric"), checking Adaptive-ECE/MCE only as
secondary robustness checks — not full-vector proper scoring or decision
accuracy. This note's Pareto-frontier finding is a repository-only,
exploratory counterpart to that published scope choice: it does not show the
paper's ECE claims are wrong, only that optimizing top-label ECE and
optimizing the full predictive distribution are different objectives in this
project's own later, separate full-vector experiments.
