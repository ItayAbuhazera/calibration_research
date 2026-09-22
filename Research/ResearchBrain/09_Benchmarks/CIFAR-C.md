---
type: benchmark
status: active
source_paper: Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
tags: [corruption, shift, cifar-c]
---

# CIFAR-C

## What it tests

Common image corruptions across severity levels.

## Current use in this vault

Primary shift benchmark for:
[[2026-09-15 RGC Shift Recoverability]]

## Important caution

A finite set of fixed corruption families is not a population sample of all
possible distribution shifts. Seed-wise intervals over model / RGC randomness
must not be interpreted as uncertainty over the space of corruptions.

## Caveat added 2026-09-21
Normalization: in the unified benchmark CIFAR-C inputs were ImageNet-normalized while CIFAR-100 checkpoints are trained on CIFAR statistics (legacy protocol).
Results tagged `legacy_v1_mixed_norm` are a compound shift; use `corrected_v2_train_norm` for CIFAR-C evaluation:
[[2026-09-21 Normalization Audit and Corrected Protocol]]. The 12 cells used in [[2026-09-20 Full-Vector DAC POC]] and [[2026-09-21 Layer-Selection Pilot]] are
development conditions, not confirmation data.
