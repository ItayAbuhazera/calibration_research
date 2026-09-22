---
type: observation
status: verified_implementation_issue
date: 2026-09-21
project: Full-Vector Geometric Calibration
evidence_strength: 4
evidence_scope: unpublished_repository_analysis
integrity_label: verified implementation issue
tags: [preprocessing, normalization, benchmark-integrity]
---

# Legacy benchmark evaluated CIFAR-100 with ImageNet statistics on a CIFAR-trained checkpoint

## Observation
[[2026-09-21 Normalization Audit and Corrected Protocol]]: in the unified benchmark (and the
legacy paper-reproduction path's clean-test loader) CIFAR-100 clean-test and CIFAR-100-C inputs
were normalized with ImageNet statistics while the checkpoints, the DAC reference bank and all
fitting/selection data used CIFAR statistics. Evidence: `data/cifar100.py` lines 50-53 vs 143-148;
cached array value ranges; reproduction of the checkpoint's recorded validation loss only under
CIFAR statistics. Effect size on the clean-test base model: +0.2 to +0.4 pp accuracy when
corrected; kNN statistics shift by up to ≈4.6 % at `layer1`.

## Why it matters
Every legacy Phase 0/1 number and the closed FV-DAC pilot are a **compound shift** (corruption +
normalization mismatch). It is confounded especially for methods that compare test queries to a
train-built reference bank.

## What this does NOT establish
That any legacy conclusion reverses; that the published RGC/IJCAI clean CIFAR-100 numbers are
affected (same loader, unresolved); that the mismatch is small for other datasets (only CIFAR-100
and CIFAR-10-C-in-the-benchmark were mismatched; Tiny ImageNet and CIFAR-10 clean-test were consistent).

Related: [[Full-Vector Geometric Calibration]], [[CIFAR-C]], [[2026-09-20 Full-Vector DAC POC]].
