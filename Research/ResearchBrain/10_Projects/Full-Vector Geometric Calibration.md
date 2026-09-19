---
type: project
status: active
canonical_repo: ../GeometricFullCalibration
parent_project: "[[Semantic Geometric Calibration RGC]]"
paper: "[[Semantic Geometric Calibration in Randomized Neural Feature Space]]"
tags: [full-vector, decision-change, kcal, rgc-shift, actionable-confidence]
---

# Full-Vector Geometric Calibration

## Research question

RGC and its predecessors calibrate a single scalar (top-label confidence). Two
questions this project asks that the published RGC paper does not:

1. Can geometric signal be used to recalibrate the **full probability vector**
   (proper scoring, not just top-label ECE), and can it ever change the
   **predicted class** (`argmax`), not just its reported confidence?
2. Does the RGC-style geometric signal, and the mapping from that signal to
   correctness, survive **distribution shift** — and if oracle-level
   complementary information exists under shift, can a system know *when* to
   act on it without labels?

## Canonical repository

`GeometricFullCalibration/` (top level of the workspace). Note: this is the
**same codebase** that ships the official RGC paper code
(`run_paper_experiments.py`, `README_IJCAI_REVIEWERS.md`) *and* a separate,
explicitly-labeled research extension, the "unified full-vector benchmark"
(`Experiments/run_unified_benchmark.py`, `README_FULL_VECTOR_BENCHMARK.md`).
The full-vector/decision-change/shift work below is entirely in the unified
benchmark path and is **not** part of the IJCAI-ECAI reproduction path or its
claims.

## Core method (repository-only; not published)

The unified benchmark evaluates, against shared checkpoints/splits/seeds:
uncalibrated baseline; temperature/vector/one-vs-rest beta/one-vs-rest
isotonic/ODIR Dirichlet calibration; RGCL, RGCC, GC-DAC, GC-TULIP; anchored
model-tail and anchored rank-geometric-tail reconstructions; full-vector
distance fusion (`neg_distance`, `margin`, `log_trust_ratio`, `rank_log_trust`
score modes); a softmax–kNN blend; KCal-lite and full KCal (learned projection
+ RBF-KDE over a validation reference bank).

## Strands and status

### A. Full-vector / decision-changing calibration

- Structural control: a positive, sample-dependent **scalar** temperature
  transform preserves logit ordering and therefore provably cannot change the
  argmax — see [[Sample-dependent scalar temperature cannot change the predicted class]].
  This is algebra, not an empirical result, and is used as a control throughout
  this strand.
- [[2026-04-25 CIFAR10 ResNet18 Full-Vector Path Batch]] and
  [[2026-04-25 CIFAR100 ResNet18 Full-Vector Path Batch]]: exploratory
  (not pre-registered), single small seed sets. `full_vector_distance_fusion`
  selected weight β=0 (collapsed to the base model) on the easier CIFAR-10
  setting — see [[Full-vector geometric fusion can collapse to the base model on easy regimes]].
- [[Top-label calibration and full-vector proper scoring form a Pareto frontier]]:
  the strongest cross-cutting observation of this strand — improving top-label
  ECE and improving full-vector proper scoring are not the same objective and
  can trade off.
- [[2026-06-21 Decision-Improving Calibration Pilot]] (GLAD-PI): design-only —
  the repository states the run started but completed results were not found;
  **do not treat as evidence** until located.
- Several anchored/blended constructions collapsed to trivial or degenerate
  solutions on the tested regimes: [[Rank-geometric anchored mixture collapses to zero weight]],
  [[Clean-selected probability blending collapses to head-only]],
  [[High-confidence anchoring can block useful full-vector decision changes]].

### B. RGC shift / recoverability / relative competence

- [[2026-09-15 RGC Shift Recoverability]] (ResNet-101, CIFAR-100-C, 5 seeds,
  pre-registered but with two unresolved protocol deviations — `k_vote` and
  the fitting split). Headline: real oracle-union headroom exists
  (**[[Geometry contains complementary accuracy information under corruption]]**,
  **[[Per-class geometry exposes more oracle headroom than global kNN]]**), but
  the tested label-free reliability gate captured essentially none of it
  (**[[Validation-fitted neighbourhood reliability features fail under corruption]]**,
  and the resulting kill: **[[Simple neighbourhood-statistics gate for geometric correction]]**).
- Open hypotheses gating further work: [[H-RGC-01 Reliability mapping shifts under corruption]],
  [[H-RGC-02 Richer representation features predict geometric reliability]].
- Candidate (not yet a project): [[Predicting geometric reliability under distribution shift]].

## What this project does NOT yet establish

- No oracle-union null baseline exists (head + a second independent predictor),
  so oracle-headroom numbers cannot yet be read as "geometry specifically" vs.
  "any two disagreeing predictors."
- No matched-risk/matched-coverage operating point has been reported anywhere
  in this strand (tracked in the vault dashboard as an open integrity item).
- No shift benchmark other than CIFAR-100-C, and no non-ResNet architecture,
  has been tested for the shift/recoverability strand.
- The `k_vote` protocol deviation in the current headline experiment is
  unresolved; the gate kill is explicitly protocol-contingent, not final.

## Artifact ledger

| Finding | Artifact | Evidence type | Integrity |
|---|---|---|---|
| Oracle-union geometric headroom under CIFAR-100-C (~4.4pp per-class, ~3.8pp global-kNN) | [[2026-09-15 RGC Shift Recoverability]] | multi-seed aggregate (5 seeds) | exploratory — no oracle-union null |
| Label-free gate captures ~0% of oracle headroom | [[2026-09-15 RGC Shift Recoverability]] | multi-seed aggregate (5 seeds) | protocol-contingent (`k_vote` deviation) |
| Full-vector distance fusion collapses to β=0 on CIFAR-10/ResNet-18 | [[2026-04-25 CIFAR10 ResNet18 Full-Vector Path Batch]] | exploratory repository run | exploratory, not pre-registered |
| GLAD-PI decision-improving calibration pilot | [[2026-06-21 Decision-Improving Calibration Pilot]] | design only | results not located in repo |
| Scalar temperature cannot change argmax | [[Sample-dependent scalar temperature cannot change the predicted class]] | implementation/algebraic inspection | clean (structural, not empirical) |

## Ancestor project

[[Semantic Geometric Calibration RGC]]
