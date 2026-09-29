---
type: paper
status: abstract_verified
year: 2025
venue: "arXiv 2510.09794 (2025-10-10)"
url: https://arxiv.org/abs/2510.09794
tags: [causality, decodability, activation-patching, vit, T4, red-team]
---

# Causality ≠ Decodability, and Vice Versa: Lessons from Interpreting Counting ViTs

Huang, Chang. Abstract verified by the lead 2026-09-29; body not read.

## Why this matters to me
It already shows, in vision, the observational-vs-causal separation that T4 wanted: activation patching across clean–corrupted image
pairs vs linear-probe decodability by depth.

## Core contribution (abstract)
"middle-layer object tokens exert strong causal influence despite being weakly decodable, whereas final-layer object tokens support
accurate decoding yet are functionally inert"; "decodability and causality reflect complementary dimensions of representation".

## Failure / limitation
Counting ViTs; controlled corruption pairs, not natural distribution shift; no per-example error repair.

## Novelty relevance
SAME PHENOMENON for T4's core separation → contributes to [[Non-oracle causal recoverability via internal edits in a fixed classifier]] (killed).
Source: `GeometricFullCalibration/docs/internal_computation_recoverability_redteam_2026-09-29.md` §4, §8.
