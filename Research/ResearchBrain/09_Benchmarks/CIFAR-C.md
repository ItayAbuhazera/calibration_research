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
