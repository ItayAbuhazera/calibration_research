---
type: paper
status: abstract_verified
year: 2026
venue: "ECCV 2026 (per arXiv comments); arXiv 2609.08367 (2026-09-08)"
url: https://arxiv.org/abs/2609.08367
tags: [selective-adaptation, repair-vs-harm, natural-shift, tta, prior-art-audit]
---

# To Adapt or Not to Adapt? Selective Adaptation for Vision-Language Models

Jiang, Liang, Liang, He, Tan. Abstract and venue comment verified by the lead 2026-09-29; body details via subagent.

## Why this matters to me
Takes the **problem structure** of our T3: a prospective per-sample decision whether to apply a correction, with beneficial vs harmful
(correct→wrong) outcomes, under natural shift.

## Core contribution
"a new problem of selective adaptation, which aims to determine whether a given test sample should undergo adaptation or be skipped";
Cross-Augmentation Similarity (CAS) gate built from outputs across augmentations (not internal features).

## Benchmark / task
CLIP TTA; ImageNet-A/V/R/K and fine-grained sets (subagent reading).

## Strongest result
Harmful + negligible adaptations are the large majority of samples (subagent: > 90 %); a simple output-based gate helps.

## Failure / limitation
The correction is TTA of a VLM, not an internal alternative class in an end-to-end CNN; gate uses outputs only.

## What this changes in my beliefs
Selective repair-vs-harm under natural shift is not a new problem; our residue is a settings intersection (internal evidence, native head,
held-out corruption families) — see [[H-SELREP-01 Multi-depth internal evidence discriminates repair from harm beyond output evidence under held-out corruption]].

## Novelty relevance
SAME PROBLEM STRUCTURE for T3. Neighborhood not saturated — re-check before any submission.
