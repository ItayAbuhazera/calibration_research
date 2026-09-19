---
type: observation
status: reframed_insufficient_evidence
evidence_scope: unpublished_repository_analysis
project: rgc
evidence_strength: 1
tags: [g-function, feature-substitution, ablation]
---

# Geometric scoring matters more than feature substitution

## Evidence-based correction to this note's title

**Archaeology did not find evidence establishing this claim as stated.** The
title is kept (per the originating task's requested structure) but the
content below reports what was actually found, which is narrower and partly
different.

## What "g_function" actually is in this repository

`Experiments/analyze_g_functions.py` / `analyze_g_functions_multiseed.py`
reconstruct and RMSE-compare the **learned isotonic calibration curve** `G(x)`
(mapping normalized separation score → probability) between RGCL and RGCC —
not the geometric scoring rule itself. The intent (per an in-code comment) is
"if RMSE is low, the Coordinate method captures the same signal as the Layer
method." **No numeric RMSE summary artifact was found on disk** (only PNGs and
timing JSONs) — the comparison's actual outcome is not recoverable from this
snapshot.

No script was found that swaps the geometric *scoring rule* (e.g. separation
vs. some alternative) independent of the feature representation, despite ~20
named candidate metrics existing in
`Experiments/layer_selection.py::unit_weights_for_all_metrics()`.

## What was actually found: a feature-pooling-representation swap

`comprehensive_analysis.comparisons.json`, key
`"Geometric: SGC vs Geo+DAC Features (SPP+JL vs spatial avg+L2)"`: holding the
calibrator fixed and swapping the feature-pooling representation (RGCL's
SPP+JL vs. a DAC-style spatial-average+L2 pooling):

- Mostly non-significant on CIFAR (2 of 9 comparisons significant).
- On SVHN, the DAC-style pooling is dramatically better:
  `svhn/resnet18`: 0.02517 (SPP+JL) vs. 0.00096 (DAC-style), p=0.041.

## Interpretation

Feature representation (pooling method) can matter a great deal in specific
regimes (SVHN) and not at all in others (most of CIFAR) — but this is a
statement about representation substitution, not a comparison establishing
that the scoring rule matters "more than" representation. That comparison
does not exist in this repository.

## What this does NOT establish

- It does not establish "geometric scoring matters more than feature
  substitution" — no such ablation was found.
- It does not establish that RGCL and RGCC's calibration curves are or are not
  similar (the `g_function` RMSE comparison's result was not recoverable).
- The SVHN result is a single dataset/architecture instance and should not be
  generalized without replication.

## Related project

[[Semantic Geometric Calibration RGC]]

## Related later observations

[[RGCL-RGCC equivalence is metric- and regime-dependent]]
