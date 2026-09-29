---
type: killed_idea
date: 2026-09-29
project: Full-Vector Geometric Calibration
reason: no identifying edit exists that is both non-oracle and non-trivial; oracle version and decodability-vs-causality dissociation published; killed by red-team analysis, not by experiment
tags: [causal, activation-patching, T4, prior-art-audit, red-team]
---

# Non-oracle causal recoverability via internal edits in a fixed classifier

## Original idea
Show that editing an implicated internal representation at depth l, then running the original downstream network, restores the correct
final prediction for errors under natural corruption (T4 of the territory audit).

## Why it died
Edit classes: class-directed edits (toward a probe's class) make restoration a consequence of the edit direction, which is uninformative.
Class-agnostic statistics restoration is BN-adapt / TENT, already per-sample harm-audited ([[To Adapt or Not to Adapt - Selective Adaptation for VLMs]]).
Oracle clean patching is DeepCorrect's correction-priority procedure; the observational-vs-causal dissociation under clean→corrupted
patching is published for vision ([[Causality is not Decodability - Counting ViTs]]), and surgical fine-tuning already locates corruption
repair early.

## Evidence that killed it
`GeometricFullCalibration/docs/internal_computation_recoverability_redteam_2026-09-29.md` §8.

## What this does NOT establish
Nothing about where corruption damage localizes in our ResNet-101 (the per-example oracle patching profile was never measured; it
remains a descriptive localization, not a research problem).

## Conditions under which to revisit
An edit family that is label-free, not class-directed, not a TTA statistic update, and has a stated identifying contrast.
