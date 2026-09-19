---
type: observation
status: open
date: 2026-09-15
project: rgc-shift
evidence_strength: 3
tags: [geometry, oracle-headroom, ood]
---

# Geometry contains complementary accuracy information under corruption

## Observation

The oracle union of head and geometry is materially better than the head alone
under CIFAR-100-C: ~4.385 pp headroom for per-class geometry and ~3.798 pp for
global kNN.

## Evidence

[[2026-09-15 RGC Shift Recoverability]], [[2026-09-15 G3 Headroom Null]]

## What it does NOT establish

It does not establish that the useful cases are identifiable without labels.
The tested label-free gate captured essentially none of the available headroom.

**Update 2026-09-15 — null now run, see [[2026-09-15 G3 Headroom Null]].**
A disagreement-matched, no-information stochastic null produces only
~0.31–0.34 pp headroom — an order of magnitude below geometry's 3.80–4.39 pp.
The headroom is not simply "any two disagreeing predictors" by construction;
`blocked_by: missing-oracle-union-null` is removed on this basis.

**But a second, independently-trained head (not geometry) recovers roughly
double the headroom (8.89 pp) and rescue precision (0.223) of either geometry
arm, at a lower disagreement rate.** So while geometry clears the
no-information bar, it does not currently clear the "ordinary second
classifier" bar. Read this note's title as "geometry contains complementary
information above a no-information null", not as "geometry is a specially
informative source of complementary information" — the latter is not
supported pending a direct geometry-vs-independent-head statistical
comparison (paired CIs not yet computed).

**Update 2026-09-15 (later same day)**: paired CIs were computed and the
independent-head gap is real (both CIs exclude zero — see
[[2026-09-15 G3 Headroom Null]] Results). Whether that gap is "just"
standalone predictor strength or something type-specific was then tested
directly in [[2026-09-15 G3 Controlled Complementarity]] and landed
**inconclusive**: no existing predictor spans geometry's accuracy range
(0.19-0.30) to test against — everything else available clusters at
0.48-0.50. Keep the strength explanation labeled correlational.

Scope: one benchmark, one architecture, five seeds, no per-corruption breakdown.

## Possible mechanisms

- Geometry is useful only in a subset of corruptions / samples not separable by
  the current neighbourhood statistics.
- The geometry/head error patterns are complementary but reliability is not
  stable from validation to corruption.
- Per-class geometry may preserve class-conditional structure that global kNN
  loses.

## Related hypotheses

[[H-RGC-01 Reliability mapping shifts under corruption]]
[[H-RGC-02 Richer representation features predict geometric reliability]]

## Historical precedent (different mechanism, related lineage only)

An earlier project in the same lineage found that semantic (deep-feature)
geometry is more corruption-resistant than raw-pixel geometry, but not
universally better than the uncalibrated model — see
[[Semantic geometry is more corruption-resistant than pixel geometry but not shift-stable]].
That result used a different codebase, a different geometric construction,
and a raw-ECE comparison rather than an oracle-headroom/routing question. See
[[Research Lineage]] for why these are recorded as historically related, not
as the same finding.

## Decisive next test

Condition directly on corruption severity / corruption family only as a
diagnostic (not as a deployable method) and test whether oracle recoverability
concentrates in specific regimes.
