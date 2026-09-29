---
type: paper
status: abstract_verified_body_via_subagent
year: 2026
venue: "arXiv 2607.10391 (2026-07-11), 'Under review'"
url: https://arxiv.org/abs/2607.10391
tags: [recoverability, intermediate-probes, vit, cifar-100-c, prior-art-audit]
---

# Vertical Fusion: Condensing Internal Representations for Robust ViT Classification

Di Salvo, Rai, Damirchi, Meza De la Jara, Doerrich, Lents, Ledig. Abstract verified by the lead 2026-09-29; §3.2.1 and §5.3 read by a
subagent (not re-verified).

## Why this matters to me
It already **names our T2**: "the notion of recoverability: the capacity of intermediate representations to correct last-layer
failures" (abstract).

## Core contribution
Independent probes at every depth of frozen ViTs (DINOv2); Recovery Rate = (a_oracle − a_L)/(100 − a_L) where a_oracle counts any layer
correct (§3.2.1); VFusion learns a low-dimensional aggregation of the hierarchy.

## Benchmark / task
16 datasets; shift sets incl. CIFAR-10-C, CIFAR-100-C, ImageNet-200-C, CelebA, Waterbirds (§5.3, Table 3; subagent reading).

## Strongest result
"intermediate probes correctly classify 18% to 76% of samples that the last-layer probe misclassifies"; VFusion closes 45 % of the gap
between the best single layer and the oracle.

## Failure / limitation
Oracle any-layer metric without chance/multiplicity correction; no per-example harm accounting; fuses for every sample (no KEEP/APPLY);
probe-on-frozen-backbone, not a native end-to-end head.

## What this changes in my beliefs
T2 as a concept is taken; only a chance-corrected, harm-accounted measurement on a native head remains. See [[Oracle any-layer recoverability counts overstate correction headroom]].

## Novelty relevance
SAME PHENOMENON / names T2 → T2 KILLED as standalone ([[Recoverability of final errors as a standalone new concept]]).

## Follow-up
Re-check revised versions before any submission (unrefereed; same group as [[Representation Trajectories Matters]]).

## Correction 2026-09-29 — lead full-text read (red-team `GeometricFullCalibration/docs/internal_computation_recoverability_redteam_2026-09-29.md` §3 E2)
Body now lead-verified from the arXiv HTML. **Table 1's recovery rates (18–76 %) are for 16 clean datasets only.** CIFAR-100-C appears
only in Table 3 (accuracy: best layer 73.1, VFusion 74.4, Oracle 81.0); no recovery rate is reported there. Recoverability is an oracle
any-layer count over MLP probes on frozen DINOv2. The "Benchmark / task" line above is correct about the shift sets, but the recoverability
*metric* was not reported under shift. T2's kill no longer rests on this paper's CIFAR-100-C numbers.
