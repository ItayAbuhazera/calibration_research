---
type: failure_mode
status: closed_historical
evidence_scope: unpublished_repository_analysis
project: rgc
non_canonical_source: true
tags: [layer-selection, oracle, augmix, negative-result]
---

# Composite geometry scores do not reliably select calibration layers

## Source-precedence caveat — read before citing

The canonical repository
(`geometric/GeometricInternalCalibration/GeometricInternalCalibration/`)
contains the composite unsupervised layer-selection module
(`robust_layer_selector.py`: `RobustLayerSelector`, Borda rank aggregation,
`within_eps_hit_rate`, `compute_top_k_recall`, `analyze_regret_distribution`),
but it has **zero callers anywhere in the canonical codebase** and produced no
output artifacts on this snapshot — in the canonical tree this is unused/dead
code. The concrete 12-experiment result below exists **only** in
`_backup_unique_from_GeometricInternalCalibration_1_20260627/aaai_full_experiments/results/analyze_result.ipynb`,
a sibling backup directory that `RESEARCH_WORKSPACE.md` says not to use unless
explicitly needed. It is used here, flagged, because it is the only surviving
evidence for an important historical negative result — do not treat it as
canonical, and do not use it as a source for any other claim without the same
caveat.

## Correction to the originating task's assumed number

The originating instruction for this note assumed "the selector matched the
oracle ECE layer only 1/12 times." **This exact metric was not found.** No
oracle-index-match statistic of that form exists in the recovered notebook.
The real, recoverable metric is different and is reported below; the
discrepancy is recorded here rather than silently resolved.

## Evidence

12 recovered experiments: AugMix/CIFAR-10, DenseNet-121/ResNet-18/ResNet-50 ×
seeds 11–14. Four layer-choice strategies compared: Composite Score (the
unsupervised selector), ECE-Only selector, Optimal (oracle, i.e. the layer
with the actual best ECE), and Physical (a fixed default layer, no
selection).

Mean ECE across the 12 experiments:

| Method | Mean ECE | "Beats Physical" rate |
|---|---|---|
| Optimal (oracle) | 0.002983 | 11/12 (91.7%) |
| ECE-Only selector | 0.007201 | 5/12 (41.7%) |
| Physical (fixed default) | 0.007208 | — (baseline) |
| **Composite Score selector** | **0.016435** | **3/12 (25.0%)** |

The composite unsupervised selector is the **worst of the four methods
compared** — worse even than the naive fixed-default layer, at roughly 5.5×
the oracle's ECE.

## Interpretation

Geometry that "looks" strongly separated by an unsupervised composite score
is not necessarily geometry that predicts calibration error. In this recovered
protocol, the composite selector actively hurt relative to doing nothing
(the fixed default) in 9 of 12 experiments.

## What this does NOT establish

- n=12, single benchmark family (AugMix/CIFAR-10), backup-sourced — this is
  under-replicated evidence, not a settled multi-benchmark conclusion.
- It does not establish that *no* unsupervised composite score could work —
  only that this particular implementation, on this protocol, did not.
- It does not establish the same mechanism as the current project's
  [[Validation-fitted neighbourhood reliability features fail under corruption]]
  — that is a different codebase, a different signal (routing gate features,
  not a layer-selection score), and a different task. Link as related lineage
  only; see [[Research Lineage]].

## Related project

[[Semantic Geometric Calibration RGC]]

## Related

[[Composite unsupervised layer selection for SGC]] (the resulting kill)
