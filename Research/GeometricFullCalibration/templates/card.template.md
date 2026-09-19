# Experiment Card -- {{batch_id}}

Status: draft | Date: {{date}} | Commit: {{commit_hash}}

## Pre-registration (filled BEFORE seeds/ is populated)
- Hypothesis: {{one sentence}}
- Primary metric: {{name}}, direction: {{lower|higher}} is better
- Primary baseline: {{method}}
- Secondary metrics and acceptable regression bounds: {{list}}
- Decision rule:
  - Continue: primary 95% paired CI excludes 0 favorably AND no secondary CI excludes 0 unfavorably beyond bound
  - Modify: primary favorable but one secondary fails
  - Pause: primary CI crosses 0
  - Kill: primary CI excludes 0 unfavorably
- What would change my mind: {{explicit counter-evidence}}
- Seeds: {{list}} | Dataset/model: {{dataset/model}}

## Commands
See commands.sh. Environment: see env.txt.

## Results (from aggregate.json -- do not retype numbers here)
- Primary: {{method}} on {{metric}} at {{aggregate.json key path}} = {{value}} [{{ci_lo}}, {{ci_hi}}]
- Secondaries: see aggregate.md.

## Observations
Factual statements only, each grounded in an aggregate.json key path.

## Decision (filled after reviewing results)
- Classification: {{continue | modify | pause | kill | inconclusive}}
- Reason (must cite the pre-registered decision rule above):
- Linked ADR: ADR-{{NNN}} (draft until accepted by human)

## Recommended next experiment
One sentence. Justification must cite this card's metrics.
