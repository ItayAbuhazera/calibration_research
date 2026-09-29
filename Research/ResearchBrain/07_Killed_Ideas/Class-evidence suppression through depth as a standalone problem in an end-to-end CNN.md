---
type: killed_idea
date: 2026-09-29
project: Full-Vector Geometric Calibration
reason: phenomenon and control design already published; own evidence shows no population-level overthinking; killed by prior-art red-team, not by experiment
tags: [class-evidence-evolution, overthinking, T1, prior-art-audit, red-team]
---

# Class-evidence suppression through depth as a standalone problem in an end-to-end CNN

## Original idea
Characterize, for final errors of an end-to-end CIFAR-100 ResNet-101 under CIFAR-100-C, whether true-class evidence that was top-ranked at
intermediate depth is later suppressed, beyond base-rate depth flips ([[H-EVO-01 True-class evidence suppression is distinguishable from base-rate depth flips in ResNet-101 under corruption]]).

## Why it died
Phenomenon published in CNNs: SDN (up to 50 % of errors, clean), Mehra et al. under CIFAR-C, and Corruption Depth (Neural Networks 2024;
abstract only: per-sample depth "until the misclassification persists" under common corruptions). The base-rate control design is
published (CALRD, IJCAI 2026, direction signature vs successes; Wrong Before Right). In our substrate every clean-trained probe is below the
head at every depth in every condition ([[Clean-trained depth probes in ResNet-101 are below the head at every depth in every tested condition]]),
and mid-depth candidates are wrong on ≈40 % of correct rows. The question's only stated value was to feed T3, whose POC is DO NOT RUN.

## Evidence that killed it
`GeometricFullCalibration/docs/internal_computation_recoverability_redteam_2026-09-29.md` §5; [[Prior Art Map - Internal Computation Recoverability and Selective Repair]].

## What this does NOT establish
Not that suppression is absent in this model (never measured); not a claim about other architectures or substrates.

## Conditions under which to revisit
A substrate where some clean depth probe beats the head (population-level overthinking), or a reopened T3.
