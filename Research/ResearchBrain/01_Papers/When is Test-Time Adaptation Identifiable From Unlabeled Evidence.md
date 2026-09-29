---
type: paper
status: read_html
year: 2026
venue: "arXiv 2609.11235 (2026-09-10)"
url: https://arxiv.org/abs/2609.11235
tags: [tta, identifiability, action-selection, keep-vs-apply, cifar-100-c, red-team]
---

# When is Test-Time Adaptation Identifiable From Unlabeled Evidence?

Jhawar, Wang. Full text (arXiv HTML) read by the lead 2026-09-29, keyword-level.

## Why this matters to me
It formalizes the question behind any KEEP/APPLY selector: "does the evidence given to the selector contain enough information to
determine the best action at all?" It separates "a weak selector versus an information channel that cannot support the desired decision".

## Core contribution
Exact boundary in a finite-batch Gaussian TTA model (doing nothing vs mean recentering). On CIFAR-100-C and DomainNet-126, changing only
deployment structure can reverse the oracle action while order-blind global evidence stays unchanged.

## Failure / limitation
Deployment-level (batch/stream) action selection for TTA, not per-example internal corrections.

## Novelty relevance
ADJACENT. Does not move a verdict. Sharpens the POC-T3 critique: a negative selector result cannot by itself distinguish a weak selector
from an insufficient channel ([[H-SELREP-01 Multi-depth internal evidence discriminates repair from harm beyond output evidence under held-out corruption]]). Source: `GeometricFullCalibration/docs/internal_computation_recoverability_redteam_2026-09-29.md` §4, §11.
