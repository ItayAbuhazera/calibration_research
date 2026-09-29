---
type: killed_idea
date: 2026-09-29
project: Full-Vector Geometric Calibration
reason: substantially the same problem defined and measured in prior work (incl. CIFAR-100-C); killed by prior-art audit, not by experiment
tags: [recoverability, T2, prior-art-audit]
---

# Recoverability of final errors as a standalone new concept

## Original idea
Distinguish failure detection ("the prediction is wrong") from recoverability ("internal computation identifies the correct
alternative") and study which final errors are recoverable.

## Why it died
The distinction and the measurement already exist: Vertical Fusion (arXiv 2607.10391) introduces "recoverability: the capacity of
intermediate representations to correct last-layer failures" and measures it incl. CIFAR-100-C; SelfChecker (ICSE 2021) separates alarm
from "advice in the form of an alternative prediction"; Orgad et al. (ICLR 2025) and KAPPA (ICML 2026) separate detection from the
correct answer / knowledge from prediction in LLMs; our own Atlas and fixed-gate studies measured correct-alternative rates on base errors.

## Evidence that killed it
[[Vertical Fusion - Recoverability in ViT Hierarchies]]; [[Self-Checking Deep Neural Networks in Deployment]]; [[Prior Art Map - Internal Computation Recoverability and Selective Repair]]; territory audit report §7.

## What remains useful
A chance-corrected, harm-accounted recoverability measure on a native head as the candidate-generation half of
[[H-SELREP-01 Multi-depth internal evidence discriminates repair from harm beyond output evidence under held-out corruption]]; see
[[Oracle any-layer recoverability counts overstate correction headroom]].

## Conditions under which to revisit
None as a standalone concept; only as a component of a selection or causal result.

## Correction 2026-09-29 — evidence re-based (verdict unchanged: killed)

Red-team full-text check (`GeometricFullCalibration/docs/internal_computation_recoverability_redteam_2026-09-29.md` §3 E2, §6): Vertical Fusion's recovery rates (18–76 %, Table 1) are **oracle any-layer counts on 16 clean
datasets**. For CIFAR-100-C it reports only fusion accuracy and an oracle row (Table 3), so "measures it incl. CIFAR-100-C" above is
overstated. The kill rests on SelfChecker (label-free internal "advice", Table III lowers CIFAR-100 accuracy 66.79 → 66.16 / 69.52 → 68.85;
lead-verified) and on the explicit detection vs correct-answer distinction (Orgad et al.; KAPPA).
