# ADR-002: Primary and secondary baselines for hybrid methods

Status: accepted
Date: 2026-04-25
Deciders: Itay

## Context
The research question involves a tension between top-label ECE (where gc_dac and rgcl-family methods excel due to top-label isotonic calibration) and full-vector NLL/Brier/accuracy (where full_vector_distance_fusion, vector_scaling, and odir_dirichlet excel). On CIFAR-100 seeds 21-25 the gap is concrete: gc_dac NLL = 2.331 [1.59, 3.07] vs full_vector_distance_fusion NLL = 1.304 [0.92, 1.69]; gc_dac top_label_ece = 0.031 [0.013, 0.050] vs full_vector_distance_fusion top_label_ece = 0.088 [0.050, 0.127]. A single baseline cannot capture both axes.

## Decision
Primary baseline for hybrid methods is full_vector_distance_fusion. Secondary baseline for top-label calibration claims is gc_dac. Paired CIs are computed against both; the primary metric decision rule references the vs-fusion CI.

## Evidence
- per_method.gc_dac.nll.mean = 2.331175 at experiments/2026-04-25_cifar100-resnet18_path-decision-batch/aggregate.json
- per_method.full_vector_distance_fusion.nll.mean = 1.303600 at same aggregate.json
- per_method.gc_dac.top_label_ece.mean = 0.031185 at same aggregate.json
- per_method.full_vector_distance_fusion.top_label_ece.mean = 0.088435 at same aggregate.json

## Consequences
- Stop doing: comparing hybrid methods against a single baseline
- Start doing: every hybrid card reports paired deltas against both fusion (primary) and gc_dac (secondary)
- New primary baseline (if changed): full_vector_distance_fusion

## Revisit conditions
if a new method dominates fusion on NLL and dominates gc_dac on top-label ECE simultaneously.
