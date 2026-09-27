# Experiment Card -- {{batch_id}}

Status: draft | Date: {{date}} | Commit/specification: {{commit_or_version}}

## Decision card (freeze before outcome evaluation)
- Parent question/hypothesis: {{link and branch disposition}}
- Question and claim scope: {{population; model/data regime; evidence inputs;
  target variable; operational decision}}
- Primary experiment type: {{descriptive | operational recoverability |
  mechanism discrimination | intervention | confirmation | engineering validation}}
  (secondary types/purpose: {{none or list}})
- Uncertainty tested: {{representation information | statistic retention |
  readout expressiveness | finite-sample estimation | transfer | action quality
  | net utility}}
- Competing explanations and predictions: {{E1 prediction; E2/ordinary
  baseline prediction; E3 if meaningful; say explicitly if predictions overlap}}
- Primary contrast: {{one target/metric/comparison}}. Minimum interpretive
  controls: {{list}}. Claim class: {{correctness prediction | replacement-class
  prediction | signed intervention utility}}.
- Practical scale and uncertainty: minimum useful effect {{threshold and why}};
  interval/power/design limitations {{limits}}.
- Data and fitting access: {{sources; fit/selection/evaluation roles;
  target-label access; grouped units; historical exposure; reserved resources}}.
- Execution envelope: authorized configurations {{list}}; allowed engineering
  recovery {{scope}}; CPU/GPU/resources {{limits}}; selection opportunities
  {{declared choices}}; finish conditions {{conditions}}.
- Provenance: authorization {{source}}; exposure ledger {{link}}; protocol/code
  {{links}}; artifact destination {{path}}.

## Outcome-to-decision matrix

| Possible outcome | Explanation supported/weakened | Still unresolved | Next decision |
|---|---|---|---|
| {{outcome 1}} | {{support/weaken}} | {{limit}} | {{continue / stop this branch / redesign named contrast / inconclusive}} |
| {{outcome 2}} | {{support/weaken}} | {{limit}} | {{decision}} |
| {{outcome 3 if meaningful}} | {{support/weaken}} | {{limit}} | {{decision}} |

## Commands
See commands.sh. Environment: see env.txt.

## Results (from aggregate.json -- do not retype numbers here)
- Primary: {{method}} on {{metric}} at {{aggregate.json key path}} = {{value}} [{{ci_lo}}, {{ci_hi}}]
- Secondaries: see aggregate.md.

## Verified evidence
Factual statements only, with exact artifact/key paths. Separate facts from
interpretation, conjecture, theorem assumptions, and proposed work.

## Post-mortem and allocation decision
- Primary contrast: {{practical magnitude and uncertainty}}
- Explanations more/less plausible: {{scoped interpretation}}
- Still indistinguishable: {{alternatives}}
- Reasoning-chain stage actually tested: {{stage}}
- Claim now supported / not supported: {{two scoped statements}}
- Allocation: {{continue / stop this branch / redesign named contrast / inconclusive}}.
  Reopen only if: {{specific material evidence/prediction and contrast}}.
- Review status/independence: {{none | self-audit | independent review; honest scope}}
- Linked ADR/review: {{link or n/a}}

## Amendments and engineering recovery
{{None, or dated change: engineering recovery vs scientific change; affected
artifacts, reruns, authorization, and exposure impact.}}
