# Cumulative research workflow

**Canonical workflow guidance for `GeometricFullCalibration`.** The Cursor
rule, project `AGENTS.MD`, and workspace Codex/Claude entry points point here.
It governs new research-workflow guidance and cards; frozen specifications,
historical cards, access controls, and existing authorizations retain their
own status. It does not itself authorize new scientific work or confirmation
data access.

## Aim and loop

Use: **question → competing explanations → smallest informative experiment →
verified evidence → allocation decision → justified follow-up**. Research is
cumulative when a completed contrast changes a stated decision, not when it
merely generates another variant to try.

Before a method search, name the most important unresolved prerequisite and
ask whether a direct diagnostic tests it better. Operational recoverability is
often appropriate when signal availability is genuinely unresolved. Do not
require a target-labelled oracle for every project, or a complete
mechanism/theorem before an intervention: a simple intervention can be the
most informative diagnostic. Feasibility or measurement work is allowed when
it establishes a necessary capability; label it, rather than claiming it
identifies a mechanism.

Maintain a research tree in the card and project synthesis: parent question,
current evidence, unresolved explanations, distinguishing experiment, and
branch disposition. Stopping a method does not close its broader question;
an open question does not authorize further allocation. Reopen a stopped
branch only for specified new evidence/prediction and a contrast capable of
testing it. A practical allocation stop is not scientific refutation.

## Card before execution

Use `templates/card.template.md` and, for vault-facing work,
[`ResearchBrain/Templates/Experiment.md`](../../ResearchBrain/Templates/Experiment.md).
Keep the card roughly one page;
link the detailed protocol, immutable configuration, code, and exposure
ledger. Completing it is scientific preparation, not a separate permission
gate. When scope is already authorized, write/freeze the card and proceed;
ask only for a genuinely unspecified scientific choice.

Every card states:

- **Question and claim scope:** population, model/data regime, evidence
  inputs, target variable, and operational decision.
- **Experiment type:** descriptive, operational recoverability, mechanism
  discrimination, intervention, confirmation, or engineering validation. If
  several apply, state one primary purpose.
- **Uncertainty tested:** representation information; information retained by
  a statistic; readout expressiveness; finite-sample estimation; transfer;
  action quality; and/or net utility. These are neither a fixed sequence nor
  mutually exclusive.
- **Competing explanations and predictions:** two or three when meaningful,
  including the strongest ordinary baseline explanation. Do not manufacture
  discrimination where predictions overlap.
- **Primary contrast:** one target/metric/comparison and the minimum controls
  needed to interpret it. Distinguish correctness prediction,
  replacement-class prediction, and signed intervention utility.
- **Data and fitting access:** sources; fit/selection/evaluation roles;
  target-label access; grouped units; historical exposure; and reserved
  resources.
- **Outcome-to-decision matrix; practical scale/uncertainty:** task-specific
  minimum useful effect, interval/power limitations, and what each possible
  result supports, does not establish, and changes operationally. A confidence
  interval crossing zero is not proof of no effect; an upper endpoint below a
  prespecified useful gain can support stopping at that scale if the design is
  valid. Never use a universal “CI excludes zero → continue” rule.
- **Execution envelope/provenance:** authorized configurations, allowed
  engineering recovery, resources, selection opportunities, finish
  conditions, parent hypothesis, specification/version, status,
  authorization source, and artifact links.

The required compact matrix is:

| Possible outcome | Explanation supported/weakened | Still unresolved | Next decision |
|---|---|---|---|
| {{outcome}} | {{interpretation}} | {{limit}} | continue / stop this branch / redesign a named contrast / inconclusive |

If every row says “try more variants,” redesign the contrast.

## Evidence, claims, and decision metrics

- Correct alternative labels existing, predicting when they help, and a net
  gain over a matched output-only system are different claims.
- A target-trained cross-fitted probe is a supervised diagnostic, not a Bayes
  oracle or information ceiling. Its null is limited to its features, model
  family, fitting budget, and evaluation scope. Reserve **oracle** for a
  precisely defined idealized operation (for example, evaluation-label choice
  from a frozen candidate set) and state both its bound and its non-bound.
  Do not use `Acc*` for an ordinary fitted probe.
- A finite-sample feature gain does not measure conditional mutual
  information directly: capacity, regularization, and selection matter. A
  target/source recoverability gap alone neither proves non-identifiability
  nor identifies its cause.
- A correctness/proper-score gain does not imply a better classification
  decision. A correction study needs a correction target and operational
  metric. For a fixed intervention set verify
  `DeltaAccuracy = (W - H) / N`; distinguish `W/(W+H+U)` from `W/(W+H)`.
- A shuffled control is useful but fallible: positive gain can be chance or
  changed regularization. Never reroll/retune it until it agrees with a
  desired conclusion. Coarse score bins cannot rule out smaller useful
  regions.
- A narrow method failure is a valid negative result without an information
  ceiling. Broad absence-of-information claims require much stronger evidence
  and may remain unavailable. Do not turn constrained nulls into “geometry is
  useless,” or project motifs into established findings.
- Separate observed facts, interpretation, conjecture, theorem assumptions,
  and proposed work. A title, assistant summary, simulation, paper abstract,
  or unverified citation does not verify a theorem.

## Exposure and confirmation

[`Representation Correction Exposure Ledger`](../../ResearchBrain/10_Projects/Representation%20Correction%20Exposure%20Ledger.md)
is the authoritative exposure ledger. It is keyed by dataset/split, checkpoint,
corruption family/severity, and purpose; record fit, selection, evaluation,
and design-decision label access separately, including known prior programs.
Inspect manifests/metadata freely for discovery, but do not open protected
labels, predictions, or metrics merely to judge whether protected data look
promising. Do not move, delete, or permission-lock data in documentation work.

Development reuse is legitimate. Prohibited: presenting repeated development
performance as untouched confirmation, hiding adaptive choices, or tuning a
purportedly frozen deployed method on target outcomes. Treat new corruptions
of the same images as a different generalization axis from new images or
architectures. Do not call checkpoints 1/3/5 pristine merely because a branch
did not use them; check broader history. Treat the unused eleven corruption
families as reserved confirmation only if the ledger supports that; otherwise
record their exposure and reserve only what remains unexposed.

Confirmation needs a frozen, justified candidate plus explicit authorization
to consume the reserved resource. A target-supervised diagnostic cannot
automatically be confirmed as a deployable clean-only method; declare target
diagnostic access separately from clean-only fitting.

## Roles, review, and execution

Use these logical roles as helpful, not compulsory, passes: **Scout** (primary
literature and verifiable mechanisms), **Skeptic** (alternatives/baselines and
distinguishability), **Experimentalist** (authorized frozen implementation and
engineering recovery), **Auditor** (evidence, protocol/code correspondence,
and warranted claims). One agent may do separated passes. For consequential
claims prefer an independent review context; agents/models are not independent
empirical replications. Record review status and independence honestly—never
invent sign-off or block routine work because a reviewer is unavailable.

Auditors inspect preregistration and evidence before narrative when feasible,
then access configs, manifests, split IDs, code, selected parameters,
per-example predictions, logs, and aggregates—not only `aggregate.json`.
They record an initial evidence-based conclusion, compare the write-up, and
resolve discrepancies. Do not reveal confirmation results before the protocol
permits it.

CPU/GPU choice follows engineering needs and quota, not an artificial
preference. Within authorization, act autonomously on inspection,
implementation, targeted tests, scheduling, aggregation, and packaging,
memory, timeout, or convergence recovery. Parallelize independent work within
quotas; bound the scientific search separately from available resources.

**Engineering recovery** restores the declared computation, versions affected
artifacts, reruns affected arms consistently, and logs the repair.
**Scientific change** (features, objective, selection grid, target access,
protected data, or outcome-dependent exclusions) requires an amendment and
exposure record, and runs only with existing or new authorization. A request
can authorize design plus execution: freeze design before outcome evaluation,
then execute it. Use immutable configurations, resumable ledgers, and output
completeness checks; a passing job/test is not scientific success. Do not add
workflow wrappers, enforcement services, compulsory full suites, or expensive
automation solely for this process.

## Post-experiment synthesis

Complete the card post-mortem: primary contrast/practical magnitude and
uncertainty; explanations made more/less plausible; indistinguishable
alternatives; reasoning-chain stage tested; supported and unsupported claim;
allocation decision; and specific reopening evidence. “Unknown” is valid; do
not force mechanism or promise publishability.

## Worked example — frozen Stage 0

[`stage0_execution_spec.md`](stage0_execution_spec.md) is an authorized,
frozen worked example, not a template to expand. Its primary contrast is the
target-fitted base-logit classifier versus base-plus-`layer3.22`-probe-logit
classifier on already exposed CIFAR-100-C development cells. Its primary type
is **operational recoverability**: it asks whether that fixed evidence/readout
has incremental target-fitted accuracy at the specified budget. It is not an
information ceiling, oracle, or clean-only deployment claim. Matched
source/target and image/view-count regimes are secondaries. It authorizes no
automatic correctness/abstention test, Stage 1, or new data.

Recorded outputs are present under
`results/stage0/report/` and the submitted jobs are no longer active in the
scheduler as checked 2026-09-23. Read the frozen specification and its
result/provenance artifacts for status; do not use this workflow document to
restart, broaden, or reinterpret the program.
