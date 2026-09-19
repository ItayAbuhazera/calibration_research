---
type: failure_mode
status: closed_historical
evidence_scope: unpublished_repository_analysis
project: rgc
tags: [ood, id-calibration, dissociation]
---

# ID calibration gains do not imply OOD detection gains

## Evidence

Design intent: `create_oracle_symlinks.py::generate_detailed_report()`
(~lines 1204–1385) explicitly builds *separate* "positive improvement" filters
for ID ECE (intended output `per_architecture_positive_improvements.csv`) and
for OOD AUROC (intended output
`per_architecture_ood_positive_improvements.csv`) — a configuration only
counts as an "OOD win" if it beats baselines on OOD independent of its ECE
result. **Generated instances of these specific files were not located on
disk** in this snapshot (input directory
`calibration_comparison_results/random_ablation/ood` not confirmed present).

Concrete stand-in: [[Neighbour count controls OOD more than ID calibration]] —
from k=50→300, ID ECE mildly *worsens* (0.99%→1.06%) while OOD AUROC keeps
improving (0.788→0.843).

## Interpretation

The authors' own reporting design treats ID-calibration improvement and
OOD-detection improvement as independent axes worth separating — consistent
with, and given concrete quantitative support by, the k-neighbour-count
ablation, where the two metrics move in different directions over the same
hyperparameter range.

## What this does NOT establish

- The comprehensive comparison table the code was designed to produce was not
  located — this note rests on (a) confirmed code design intent and (b) one
  concrete ablation (k), not a broad multi-configuration table.
- It does not establish that ID and OOD objectives are *always* in tension —
  only that they are not guaranteed to move together, and one clear
  counterexample (k) exists.

## Related project

[[Semantic Geometric Calibration RGC]]

## Related

[[Neighbour count controls OOD more than ID calibration]]
