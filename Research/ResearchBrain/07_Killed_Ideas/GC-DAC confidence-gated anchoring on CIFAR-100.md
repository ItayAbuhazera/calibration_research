---
type: killed_idea
date: 2026-04-25
project: full-vector-calibration
reason: CIFAR-100 multi-seed evidence refuted the gated-anchor path
tags: [anchoring, gc-dac, negative-result]
---

# GC-DAC confidence-gated anchoring on CIFAR-100

## Original idea

Use GC-DAC confidence to decide where to apply an anchored top-label method and
where to fall back to a stronger full-vector calibrator.

## Why it died

On CIFAR-100:
- the rank-geometric mixture selected zero mixture weight;
- anchored variants hurt NLL, including on base-correct examples;
- distribution-adaptive confidence strata did not reveal a region where the
  anchored method overtook full-vector distance fusion on NLL.

## Naming

"GC-DAC" is the name of the gating *idea*. The anchored method it gated is
`rgcl_tail_vector_scaling`; its NLL-strong comparator is full-vector distance
fusion (see [[Top-label calibration and full-vector proper scoring form a Pareto frontier]]).

## Decision

Adopt the Pareto-frontier characterization instead of continuing to tune the
gated-anchor path.

## Revisit condition

Only revisit if a genuinely new dataset shows an anchored method beating the
full-vector baseline in a pre-specified confidence stratum or if a new mechanism,
not threshold tuning, predicts where the anchor should help.
