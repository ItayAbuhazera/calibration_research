---
type: observation
status: open
date: 2026-09-15
project: rgc-shift
evidence_strength: 3
tags: [perclass, global-knn, geometry]
---

# Per-class geometry exposes more oracle headroom than global kNN

## Observation

Per-class geometry exposes about **0.587 pp** more oracle-union headroom than
global kNN, with a paired 95% CI that excludes zero.

## Evidence

[[2026-09-15 RGC Shift Recoverability]], [[2026-09-15 G3 Headroom Null]]

## Update 2026-09-15 — null baseline run

Both per-class geometry (4.385 pp) and global kNN (3.798 pp) clear a
disagreement-matched no-information null (0.34 pp / 0.31 pp respectively) by
a similar margin — see [[2026-09-15 G3 Headroom Null]]. The 0.587 pp
per-class-over-global-kNN gap itself was not re-tested against a null in
this pass; the open question below (disagreement rate vs. information)
remains unresolved.

## Why this matters

The useful signal is not merely "nearest-neighbour structure exists". The
class-conditional geometry appears to preserve additional complementary
information.

## What it does NOT establish

The gated accuracies are indistinguishable, so the extra per-class headroom is
not yet operationally recoverable.

More basically: **oracle headroom rises with disagreement, not only with
information.** Standalone geometry accuracy and head-geometry disagreement rate
were not recorded for either arm, so the +0.587 pp is equally consistent with
per-class geometry simply disagreeing with the head more often. Record both
rates before reading this as a statement about class-conditional structure.

## Decisive next test

Compare where the extra per-class-only oracle wins occur:
- corruption family,
- severity,
- head confidence,
- contrast magnitude,
- representation layer / seed regime.
