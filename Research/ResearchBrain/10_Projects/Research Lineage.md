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

## Possible future branch: decision-native foundation models

[[Decision-Native Foundation Models for Calibrated Parallel Decisions]] is a
candidate extension of the broader actionable-confidence theme: reuse a
foundation-model representation to produce calibrated, risk-aware operational
decisions through a shared schema-conditioned interface. It is not part of the
historical geometric lineage, and no evidence currently establishes that its
mechanism follows from—or should use—the existing geometric methods. Treat it
as a possible branch pending a prior-art audit and a bounded POC, not as the
next proven stage of this research line.
