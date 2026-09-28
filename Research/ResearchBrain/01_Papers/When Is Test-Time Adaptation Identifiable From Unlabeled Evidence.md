---
type: paper
status: read_html
year: 2026
venue: preprint (arXiv, submitted 2026-09-10; no venue stated)
url: https://arxiv.org/abs/2609.11235
tags: [tta, identifiability, action-selection, evidence-channel]
---

# When Is Test-Time Adaptation Identifiable From Unlabeled Evidence? (Jhawar & Wang)

Read 2026-09-28 from the arXiv HTML; theorems checked as stated there, proofs not re-derived.

## Why this matters to me

Gives the framing for action selection: a selector can only be as good as the ambiguity its evidence channel resolves.

## Core contribution

Selector sees Z_n = φ_n(f_0, X_1..X_n) (entropy, margins, source logits, moments, stream statistics); actions {Keep, Tent, DeYO, ROID, test-time normalization}; oracle = minimum prospective target risk. Thm 1: identical evidence laws with different unique oracle actions ⇒ any selector errs ≥ 1/2 (equal prior). Thm 2: identifiable iff all evidence-compatible worlds share an optimal action. Thm 3: Gaussian finite-batch Keep/Recenter boundary δ_c(n) = Θ(n^{-1/2}). Prop 1: invariances ignored by the channel create non-identifiability.

## Benchmark / task

CIFAR-100-C (675 worlds) and DomainNet-126 (90 worlds).

## Strongest result

157 of 270 matched IID/correlated pairs with the same images change oracle action; stream-aware evidence (Z3) lowers mean regret to 0.312 pp vs 1.170 pp (MORPHEUS NC).

## Failure / limitation

Actions are per-deployment (batch/stream), not per-example; evidence channels are output/stream-level — no matched internal-vs-output comparison; 1-D Gaussian theory.

## What this changes in my beliefs

The right target for our action question is ambiguity in **action advantage** under the available evidence, not error prediction.

## Related observations

—

## Novelty relevance

COMPONENT / BUILDING BLOCK (identifiability framing); not a collision with a per-example internal-evidence action study.

## Follow-up

Framing used by the proposed N1a action-ambiguity audit ([[2026-W40]]).
