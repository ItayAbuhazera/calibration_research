---
type: hypothesis
status: proposed
project: rgc-shift
benchmark: CIFAR-100-C
novelty: unknown
tags: [representation, conditional-information]
---

# H-RGC-02 Richer representation features predict geometric reliability

## Formal statement

The current scalar neighbourhood statistics are insufficient, but richer
representation-derived features contain information about whether geometry
will outperform the head.

Conceptually:

I(Z_rich ; G | R_basic, S_head) > 0

where G means "geometry is the better decision".

## Why it follows from evidence

Per-class geometry has oracle headroom that the current label-free gate cannot
recover.

## Baselines

- current four-feature gate;
- head confidence / entropy / margin only;
- combined basic gate + head confidence.

## Decisive experiment

Use a strictly held-out diagnostic split to compare incremental predictive
value of richer representation summaries. Do not jump directly to a large
neural gate; first test whether any incremental signal exists.

## Kill criterion

No stable incremental discrimination / risk-coverage gain beyond basic
features and head confidence.

## Go criterion

Consistent incremental value across seeds and at least two corruption families.
