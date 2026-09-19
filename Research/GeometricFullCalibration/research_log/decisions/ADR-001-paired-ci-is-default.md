# ADR-001: Paired t-CI at 95% is the default comparison procedure

Status: accepted
Date: 2026-04-25
Deciders: Itay

## Context
Seeds are shared across methods within a batch by construction (same data splits, same model checkpoint, same calibration split), so paired differences remove seed-level variance that dominates unpaired comparisons at N=5. The aggregator at research_log/scripts/aggregate_batch.py implements paired t-CI with N-1 dof and a 1.96 fallback for df >= 20. Prior published Table 1 from the RGC paper shows per-metric seed std of 0.05-0.3 ECE points, which means single-seed deltas under 0.2pp are pure noise.

## Decision
Paired t-CI at 95% is the default for every method-vs-baseline comparison. Unpaired comparison is permitted only when seed sets differ and must be flagged explicitly in the card.

## Evidence
- aggregate_batch.py:_paired_vs_baseline at research_log/scripts/aggregate_batch.py: paired t-CI implementation
- per_method[*][*].n_pairs at experiments/2026-04-25_cifar100-resnet18_path-decision-batch/aggregate.json: confirms shared seed set across methods within a batch

## Consequences
- Stop doing: reporting unpaired mean +/- std deltas in cards or ADRs
- Start doing: every numeric claim in a card cites a paired CI from aggregate.json paired_vs_baseline
- New primary baseline (if changed): n/a

## Revisit conditions
if a future batch deliberately uses different seed sets across methods.
