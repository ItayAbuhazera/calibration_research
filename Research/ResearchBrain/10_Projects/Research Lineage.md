---
type: project_map
status: active
date: 2026-09-15
tags: [lineage, geometry, calibration, actionable-confidence]
---

# Research Lineage — Geometric Uncertainty to Actionable Confidence

This note is a **historical organization of four projects that share
authors, codebases, and a recurring question**, not a claim that each stage
was a theoretically necessary consequence of the one before it. Several
transitions (e.g. pixel → semantic geometry, top-label → full-vector) were
motivated by specific limitations noted at the time, and are recorded that
way below; others may simply reflect what was tractable or interesting next.

## The chain

1. **[[Geometric Separation]]** (`GeometricCalibration`, JMLR 2023) —
   raw-pixel/feature geometric distance to training examples as a confidence
   signal, calibrated post-hoc. Explicitly proposes CNN-middle-layer geometry
   as future work.
2. **[[Semantic Geometric Calibration RGC]]** (`GeometricInternalCalibration`,
   IJCAI-ECAI 2026 preprint) — moves geometry into intermediate neural
   representation space (RGCL/RGCC), replacing architecture-specific layer
   selection with randomized sampling.
3. **[[Full-Vector Geometric Calibration]]** (`GeometricFullCalibration`) —
   extends the same geometric-scoring toolkit from scalar top-label
   calibration to full-vector / decision-changing calibration, KCal
   comparisons, and reliability-routing diagnostics.
4. **RGC shift / recoverability / relative competence** — within the same
   `GeometricFullCalibration` codebase: does the geometric signal that helps
   in-distribution keep meaning under corruption, and can a system know
   *when* to act on it? See [[2026-09-15 RGC Shift Recoverability]] and the
   open hypotheses [[H-RGC-01 Reliability mapping shifts under corruption]],
   [[H-RGC-02 Richer representation features predict geometric reliability]].
5. **Current actionable-confidence research questions** — see below.

## Repeated questions across generations

These are recorded as **open questions that recur across all four stages**,
not as conclusions already reached at any stage.

1. Does geometry contain information beyond the model's own confidence?
2. Which representation (pixel, layer, randomized-layer, randomized-coordinate)
   makes that information usable?
3. Is the chosen geometry itself stable across dataset/model regimes, or does
   the best choice change per regime?
4. Does the mapping from geometric signal to correctness transfer under shift?
5. When should geometric information change the system's *decision* rather
   than merely its reported confidence?

## Why this lineage matters

Three historical precursors of the current "is the signal itself reliable"
question, spread across three different projects and codebases:

- **[[Geometric Separation]]**: a clean-fitted raw-pixel geometric confidence
  mapping reversed — became worse than uncalibrated — under the repository's
  synthetic-corruption protocol (see
  [[Clean-fitted geometric confidence mappings can reverse under synthetic corruption]]).
- **[[Semantic Geometric Calibration RGC]]**: an unsupervised, "good-looking"
  geometric layer selector did not reliably pick the layer with the best
  actual calibration error (see
  [[Composite geometry scores do not reliably select calibration layers]]).
- **RGC shift / recoverability** (current): a validation-fitted, label-free
  reliability gate could not recover the oracle headroom that geometry
  demonstrably has under CIFAR-100-C corruption (see
  [[Validation-fitted neighbourhood reliability features fail under corruption]]).

**These three observations make "is the geometric signal itself reliable and
knowable-when-to-trust" a historically recurrent question across this research
line. They do not, on their own, establish a single common mechanism** — the
representation, the corruption protocol, the task (routing vs. layer choice vs.
calibration mapping), and the metric differ across all three, and no
cross-project null or unifying experiment has been run.

## Related current-generation notes

- [[Geometry contains complementary accuracy information under corruption]]
- [[Validation-fitted neighbourhood reliability features fail under corruption]]
- [[Predicting geometric reliability under distribution shift]]
- [[Top-label calibration and full-vector proper scoring form a Pareto frontier]]

## Active side branch: full-vector DAC (2026-09-20)

[[Full-Vector Density-Aware Calibration]] /
[[H-FVDAC-01 Class-conditioned DAC density enables decision correction]] /
[[2026-09-20 Full-Vector DAC POC]] branch off **published prior art**
([[Beyond In-Domain Scenarios - Robust Density-Aware Calibration]]) rather
than off stage 3 of the chain above. It shares this codebase and the
CIFAR-100-C protocol, and it asks recurring question 5 ("when should
geometric information change the decision?"), but its mechanism is DAC's
own layer-wise kNN operator with a class-conditioned search domain — not
RGCL/GC-DAC geometry. It is **not** an attempt to rescue any prior
full-vector geometric fusion result, and nothing in it revises the earlier
negative findings.

## Possible future branch: decision-native foundation models

[[Decision-Native Foundation Models for Calibrated Parallel Decisions]] is a
candidate extension of the broader actionable-confidence theme: reuse a
foundation-model representation to produce calibrated, risk-aware operational
decisions through a shared schema-conditioned interface. It is not part of the
historical geometric lineage, and no evidence currently establishes that its
mechanism follows from—or should use—the existing geometric methods. Treat it
as a possible branch pending a prior-art audit and a bounded POC, not as the
next proven stage of this research line.

## Update 2026-09-21

FV-DAC (legacy protocol, closed) → normalization audit → [[2026-09-21 Layer-Selection Pilot]]. The recurring open *question* (not a claim, not an idea note):
does a source-fitted geometric readout carry decision value that survives corruption, and how would one know without target labels?
Cross-links: [[Theory Plan - Decision Utility, Layers, Compression and Risk Control]] (why clean-fitted utility is non-identifiable on an unlabeled target),
[[From Similarity to Decisions - PCE (AAAI submission)]] (decision-target and harm-budget framing). The pilot's null is scoped to the tested combination.

## Update 2026-09-21 (second pass)
Layer pilot → [[2026-09-21 Residual Evidence Study]] (common residual readout, output-evidence and label-access controls, frozen gate): null. The recurring open *question* narrows to learnability from a small clean fit set rather than to which representation/pooling to read; still a question, not a claim or an idea note.

## Update 2026-09-21 (third pass)
Residual-evidence study → [[2026-09-21 Representation Atlas Program]]: the open *question* narrows to how a clean-trained selector could separate helpful from harmful mid-layer alternatives (candidate presence is established; safe selection is not). Question, not a claim or idea note.

## Update 2026-09-21 (fixed deep-candidate gate study) — [[2026-09-21 Fixed Deep Candidate Gate Study]], [[H-GATE-01 Candidate selection versus gate utility mismatch]]
Status: completed (development, checkpoints 2 and 4). Fixed `layer3.22` 2×2 kNN candidate with C0/C1/Z0/Z1 ridge gates and n∈{625,1250,2500} learning curves: no consistent held-out benefit (Z1−Z0 clean +0.10/+0.13 pp, corruption −0.04/−0.04); deep gates ≈ output-evidence and layer4/output controls; practical targets not met; probability quality not improved. Deep candidate has 873/908 fit disagreements (the 52–151-event limit was layer4/output-specific) but H:W ≈ 3.8:1. Audit corrections to the atlas report appended in its card; TF32 mismatch cause verified (benchmark = TF32 convs, batch 128); seed-4 reconciliation deferred jobs 21537819–21. Measurement note for theory: standalone E[D] and gated E[gD] differ, but here neither exposed a large positive-utility region (top score bin ≈ 0 utility). Not established: absence of information; more-label benefit.
