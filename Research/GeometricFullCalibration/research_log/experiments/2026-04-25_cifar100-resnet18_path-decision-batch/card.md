# Experiment Card -- 2026-04-25_cifar100-resnet18_path-decision-batch

Status: draft | Date: 2026-04-25 | Commit: unknown

## Pre-registration (filled BEFORE seeds/ is populated)
- Hypothesis: {one sentence}
- Primary metric: {name}, direction: {lower|higher} is better
- Primary baseline: {method}
- Secondary metrics and acceptable regression bounds: {list}
- Decision rule:
  - Continue: primary 95% paired CI excludes 0 favorably AND no secondary CI excludes 0 unfavorably beyond bound
  - Modify: primary favorable but one secondary fails
  - Pause: primary CI crosses 0
  - Kill: primary CI excludes 0 unfavorably
- What would change my mind: {explicit counter-evidence}
- Seeds: 21,22,23,24,25 | Dataset/model: cifar100/resnet18
- WARNING: this card was filled retroactively from an already-completed batch. Pre-registration was not recorded before execution. Treat all conclusions from this batch as exploratory.

## Commands
See commands.sh. Environment: see env.txt.

## Results (from aggregate.json -- do not retype numbers here)
- Primary: full_vector_distance_fusion on nll at per_method.full_vector_distance_fusion.nll.mean = 1.3036004781723023 [per_method.full_vector_distance_fusion.nll.ci_lo=0.9172086521877691, per_method.full_vector_distance_fusion.nll.ci_hi=1.6899923041568354]
- Secondaries: see aggregate.md.

## Observations
- Seeds present are listed at sanity.seeds_present in aggregate.json.
- Commit hash source is listed at sanity.commit_hash_source in aggregate.json.

## Decision (filled after reviewing results)

## Recommended next experiment

