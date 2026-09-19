# Aggregate Report -- {{batch_id}}

Seeds: {{seed_list}} | N = {{n_seeds}} | Commit: {{commit_hash}}
Baseline for paired deltas: {{baseline_name}}

## Per-method metrics (mean [95% t-CI])
{{per_method_table}}

## Paired deltas vs {{baseline_name}} (mean [95% paired CI])
`*` = CI excludes 0 in favorable direction.
{{paired_delta_table}}

## Decision-changing diagnostics
{{flip_table}}

## Hyperparameter selection stability
{{hp_table}}

## Sanity checks
- Max |1 - row_sum| across probability matrices: {{prob_sum_error}}
- Seeds present: {{seeds_present}}
- Commit hash consistent across seeds: {{commit_consistent}}
