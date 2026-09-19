---
type: idea
status: candidate
novelty: unknown
top_tier_potential: unknown
benchmark: multi-task decision benchmark TBD
project: decision-native-foundation-models
tags: [foundation-models, confidence, calibration, selective-decision, efficiency, structured-decisions]
---

# Decision-Native Foundation Models for Calibrated Parallel Decisions

## One-sentence contribution

Test whether a single shared, schema-conditioned adapter can reuse a pretrained
foundation model's representation to produce multiple calibrated probabilistic
decisions in parallel, including for held-out decision schemas, with an
efficiency advantage measured at matched task quality or selective risk.

## Phenomenon

Many foundation-model workloads ultimately require structured actions rather
than explanatory prose: classify an intent, select one or more allowed answers,
abstain, escalate, or route to another system. Standard autoregressive
interfaces may repeatedly decode text even when several decisions share the
same input context and each answer space is known.

The candidate interface is:

```text
h = F(x)

p_i(a | x, q_i) = G(h, q_i, a)
```

Here, `x` is the input/context; `q_i` describes a decision question and its
allowed answer space; the expensive representation `h` is computed once; and
`G` is one shared, schema-conditioned mechanism rather than an independent
classifier head for every task. For several questions, the intended computation
is conceptually:

```text
                         ┌── Decision 1
Input ── Foundation ─────┼── Decision 2
         model           ├── Decision 3
                         └── Decision M
```

Whether this interface can preserve useful task performance, uncertainty, and
transfer while reducing cost is unknown. Jev / RLCD is only recent motivation
or an existence signal for decision-oriented alternatives to autoregressive
generation; unpublished company claims are not scientific evidence here, and
this note makes no claim about their architecture.

This idea is a plausible new branch of the broader actionable-confidence theme
summarized in [[Research Lineage]]:
internal representation → uncertainty/confidence → calibrated risk →
operational decision. It is not yet empirically or mechanistically connected
to the existing geometric work.

## Core hypothesis

A shared Decision Adapter, conditioned on natural-language descriptions of
questions and candidate answers, can be trained across heterogeneous tasks and
then transfer non-trivially to entirely held-out tasks and answer schemas. The
hypothesis is that its probabilities can support selective decisions,
abstention, or escalation, and that reusing one foundation-model representation
can yield increasing efficiency benefits as the number of requested decisions
`M` grows without sacrificing quality at the matched operating point.

The simplest falsifiable adapter family is:

```text
h_x = F(x)
z_q,a = E(q, a)

s(x,q,a) = phi(h_x)^T W psi(z_q,a)

P(a | x,q) = softmax_a s(x,q,a)
```

This bilinear form is a POC, not a commitment. If frozen representations are
insufficient, a stronger but still bounded variant may add lightweight
cross-attention or a schema encoder; LoRA/QLoRA is a later fallback, not the
starting point. Foundation-model pretraining is out of scope.

## Why this may be new

Unknown. The narrow question worth auditing is not whether language-model
representations can support classification, nor whether classification can be
faster than generation. It is whether one shared, schema-conditioned
probabilistic decision mechanism can jointly provide:

- heterogeneous answer spaces rather than one fixed label vocabulary;
- transfer to unseen decision schemas and preferably unseen task families;
- calibrated confidence useful for risk-controlled abstention or escalation;
- reuse of one input representation across many simultaneous decisions; and
- an efficiency advantage against strong alternatives at matched quality or
  matched selective risk.

This conjunction is only a proposed research target. It must undergo an
adversarial literature audit before serious investment, and no novelty score or
top-tier probability should be assigned yet.

## Closest prior art

The unaudited threat set includes classification or discriminative heads on
pretrained language models; prompt-based classification from token
probabilities and verbalizers; multi-task and instruction-conditioned
classification; semantic uncertainty and confidence elicitation; selective
prediction; conformal or risk-controlled LLM decisions; structured or
constrained generation; objectives targeting calibration; non-autoregressive
prediction; learned routing and cascading; embedding-based classifiers;
prototype methods; and cross-encoders that score `(input, candidate answer)`
pairs.

These are prior-art families to audit, not citations or claims that the proposed
combination is absent from the literature. The audit must search specifically
for shared schema-conditioned scoring, held-out-task or held-out-schema
transfer, calibrated probabilistic outputs, and matched-quality efficiency
evaluation. Discovery that this combination is already established is a kill
criterion, not a result to obscure.

## Benchmark

Construct a small heterogeneous multi-task decision benchmark, finalized only
after checking licensing, download/compute size, label semantics, task
diversity, contamination concerns, and compatibility with candidate baselines.
Possible families are:

- intent / high-cardinality classification: Banking77 or CLINC150;
- multi-label parallel decisions: GoEmotions or another suitable semantic
  multi-label dataset;
- topic, sentiment, or question classification: AG News, TREC, or an SST-style
  dataset; and
- multiple-choice reasoning or knowledge: ARC-Challenge or selected MMLU /
  MMLU-Pro tasks.

The suite must emphasize heterogeneous schemas and must not consist only of
binary sentiment tasks. Split by task, not merely by examples: train the same
adapter on three to five heterogeneous tasks, then hold out at least one entire
task family whose task, answer vocabulary, and schema were absent from adapter
training. In the main zero-shot evaluation, provide only natural-language
question and label descriptions and allow no task-specific fine-tuning.

Minimum baselines are:

A. autoregressive LLM answer generation;
B. constrained / structured autoregressive generation;
C. direct LM token/log-probability classification where applicable;
D. frozen representation plus a task-specific linear classifier; and
E. the shared schema-conditioned Decision Adapter.

Embedding similarity / prototype classification and cross-encoder scoring of
`(input, candidate answer)` are useful additional baselines. Some methods may
not apply identically to every task; report those limitations rather than force
an artificial comparison.

## Matched operating point

The main claim cannot be merely “faster than generation.” Compare throughput,
latency, FLOPs, or output-token cost at equal accuracy/F1, equal error rate,
equal selective risk, or equal coverage under a fixed risk constraint. A
candidate headline metric is **throughput at matched selective risk**, with
**latency / FLOPs / tokens at matched task quality** as a complementary view.

Report task-appropriate accuracy or F1, NLL, Brier score, calibration measures
(not ECE alone), AURC / risk-coverage, coverage at fixed risk, latency,
throughput, peak VRAM, and autoregressive output-token count. Measure
representation reuse explicitly as `M` increases—for example `M = 1, 4, 16,
64`—because the proposed advantage should become clearer only when several
decisions share the same input representation. Include optimized constrained
classification baselines, not only verbose free-form generation.

## Decisive POC

1. Select one open 1B–4B language model that fits on one GPU.
2. Freeze the backbone and precompute input representations where practical.
3. Train one small schema-conditioned Decision Adapter jointly across three to
   five heterogeneous tasks.
4. Hold out at least one entire task family and its answer schema.
5. Evaluate seen tasks and the zero-shot held-out task/schema for predictive
   quality, calibration, risk-coverage, abstention/escalation utility, inference
   cost, and scaling with the number of simultaneous decisions.
6. Compare against the autoregressive, constrained, token-probability,
   task-specific-head, and applicable embedding/cross-encoder baselines at
   matched operating points.

The informative positive outcome is not merely improved accuracy. It would be
that one shared adapter remains competitive, provides useful probabilistic
confidence, transfers non-trivially to unseen schemas/tasks, and becomes
materially more efficient than strong autoregressive or constrained baselines
as `M` grows. This is the result the POC is designed to test, not an expected
conclusion.

End-to-end calibration objectives, geometric uncertainty, distribution-shift
robustness, adaptive adapter-to-LLM routing, dependent or hierarchical
decisions, and task-conditioned representation extraction are deferred unless
the base phenomenon survives.

## Kill criterion

Kill or substantially reframe the idea if any of the following holds:

1. Where task-specific training is allowed, a simple task-specific linear head
   matches or beats the shared mechanism and the latter provides no useful
   transfer advantage.
2. On unseen tasks/schemas, the shared adapter collapses toward chance or is
   materially worse than ordinary prompting or token-probability baselines.
3. Calibration fails on held-out tasks or distribution shift and can be
   restored only with task-specific calibration data.
4. The efficiency advantage disappears against optimized constrained
   classification rather than verbose text generation.
5. Any speed advantage requires materially worse decisions at the same
   risk/coverage or other matched-quality operating point.
6. The prior-art audit finds that essentially the same shared,
   schema-conditioned probabilistic architecture—with unseen-task transfer and
   calibrated, risk-controlled evaluation—has already been established.

## Reviewer #2 attack

- “This is just a classifier head on an LLM.”
- “This is prompt-based classification without decoding.”
- “Of course classification is faster than generation.”
- “Zero-shot labels are just text embeddings used as classifier weights.”
- “Calibration does not make the architecture novel.”
- “You are comparing against an unnecessarily expensive generative baseline.”
- “A task-specific classifier would solve this better.”
- “Any claimed parallelism is trivial once outputs are conditionally
  independent.”

The project is interesting only if evidence—not framing—distinguishes it via
shared schema conditioning, held-out schema/task generalization,
probabilistic and risk-aware outputs, measurable representation reuse, and
matched-quality efficiency comparisons against strong baselines.

## Current verdict

PROMISING DIRECTION, REQUIRES PRIOR-ART AUDIT AND SMALL POC.
