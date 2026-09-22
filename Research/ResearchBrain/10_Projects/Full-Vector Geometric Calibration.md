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

### C. Full-vector DAC (new branch, 2026-09-20)

A **separate** strand that shares this codebase and the CIFAR-100-C
protocol but not the RGCL/GC-DAC mechanism: [[Full-Vector Density-Aware
Calibration]] / [[H-FVDAC-01 Class-conditioned DAC density enables decision
correction]] / [[2026-09-20 Full-Vector DAC POC]]. It class-conditions the
*published* DAC operator's search domain and adds one non-negative scalar.
It is explicitly **not** a rescue of strand A's `β=0` collapse or of the
gate kill in strand B, and it does not revise either. No result has been
interpreted yet.

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

## Update 2026-09-21 — corrected preprocessing, layer-selection pilot, theory plan

### D. Normalization repair (verified implementation issue)
The unified benchmark evaluated CIFAR-100 clean test and CIFAR-100-C with ImageNet statistics on CIFAR-trained checkpoints:
[[Legacy benchmark evaluated CIFAR-100 with ImageNet statistics on a CIFAR-trained checkpoint]],
[[2026-09-21 Normalization Audit and Corrected Protocol]]. Corrected protocol `corrected_v2_train_norm` is now the benchmark default;
legacy results (incl. strand C's FV-DAC pilot) are legacy-labelled, not revised. **Unresolved:** whether the RGC/IJCAI-era clean CIFAR-100 numbers share the issue.

### E. Layer-selection pilot (strand C successor, 2026-09-21)
[[2026-09-21 Layer-Selection Pilot]] / [[H-LAYER-01 Selected internal layers add decision value beyond logits]]: 12 block-output candidates, L ∈ {1,4,6,8},
Family A (class-distance correction) and Family B (linear probes). Verdict under the frozen rule: no material evidence to continue *this tested family*
(best +0.095 pp vs Vector Scaling +0.190 pp). Pooling sensitivity is the only axis with a hint (+0.3–0.44 pp) and is hypothesis-generating.

### F. Theory work plan
[[Theory Plan - Decision Utility, Layers, Compression and Risk Control]]; link to the AAAI PCE submission:
[[From Similarity to Decisions - PCE (AAAI submission)]] (proposed extension is an idea, not a contribution).

| Finding | Artifact | Evidence type | Integrity |
|---|---|---|---|
| Legacy CIFAR-100 test/C used ImageNet stats vs CIFAR training | [[2026-09-21 Normalization Audit and Corrected Protocol]] | code + array ranges + val-loss reproduction | verified implementation issue |
| Layer pilot: no candidate arm ≥ +0.5 pp (2 seeds, 12 development cells) | [[2026-09-21 Layer-Selection Pilot]] | preregistered pilot, 2 seeds | under-replicated (n=2 seeds) |
| 2×2 pooling +0.3–0.44 pp (Family A) | [[2026-09-21 Layer-Selection Pilot]] | pre-declared sensitivity | hypothesis-generating |

## Update 2026-09-21 (second pass) — residual-evidence study
### G. Residual decision-information study
[[2026-09-21 Residual Evidence Study]] / [[H-RESID-01 Source-learnable residual decision information at layer3.22]]: `layer3.22` GAP / 2×2 / class-radius / logit-space evidence under one residual readout with a frozen output-only anchor.
Frozen gate failed (criteria 1–4); **stop.** Development: hidden-evidence procedure +0.102 pp vs output-evidence control +0.118 pp (VS +0.128, anchor +0.108). Corrections to earlier interpretation were appended to the theory plan and the pilot card.

| Finding | Artifact | Evidence type | Integrity |
|---|---|---|---|
| Residual-evidence procedure does not beat output-evidence control (2 dev checkpoints) | [[2026-09-21 Residual Evidence Study]] | preregistered dev stage, 2 seeds; confirmation not authorized | under-replicated (n=2 seeds), development conditions |
| Legacy-normalization values in RGC-repo `calibration_comparison/ablation_*` JSONs | [[2026-09-21 Normalization Audit and Corrected Protocol]] (status note) | artifact vs paired diagnostic | verified for those artifacts; publication provenance unresolved |

## Update 2026-09-21 (third pass) — representation atlas program
### H. Atlas / distance / gate program
[[2026-09-21 Representation Atlas Program]] / [[H-ATLAS-01 Accessible correction information across layers pooling and metrics]]: 34 sites × {GAP, 2×2, SPP}, kNN diagnostic, three metrics, small gates, temperature; 0.58 GPU-h. Result: alternatives exist at deep layer3 (10–14 % of base errors, vs 2.5 % logit kNN) but at 38–52 % harm on base-correct; clean-selected `layer4` candidates duplicate the head; H-A…H-E refuted. Prior closed studies untouched.

| Finding | Artifact | Evidence type | Integrity |
|---|---|---|---|
| Deep-layer3 spatially pooled kNN supplies correct alternatives on 13 % of base errors at 38–41 % harm | [[2026-09-21 Representation Atlas Program]] | target-labelled diagnostic, 2 checkpoints, 2 000-image subset | exploratory, development cells |
| Clean-selected hidden candidate + gate does not beat output controls | same | frozen mechanical evaluation | under-replicated (n=2) |

## Update 2026-09-21 (fixed deep-candidate gate study) — [[2026-09-21 Fixed Deep Candidate Gate Study]], [[H-GATE-01 Candidate selection versus gate utility mismatch]]
Status: completed (development, checkpoints 2 and 4). Fixed `layer3.22` 2×2 kNN candidate with C0/C1/Z0/Z1 ridge gates and n∈{625,1250,2500} learning curves: no consistent held-out benefit (Z1−Z0 clean +0.10/+0.13 pp, corruption −0.04/−0.04); deep gates ≈ output-evidence and layer4/output controls; practical targets not met; probability quality not improved. Deep candidate has 873/908 fit disagreements (the 52–151-event limit was layer4/output-specific) but H:W ≈ 3.8:1. Audit corrections to the atlas report appended in its card; TF32 mismatch cause verified (benchmark = TF32 convs, batch 128); seed-4 reconciliation deferred jobs 21537819–21. Measurement note for theory: standalone E[D] and gated E[gD] differ, but here neither exposed a large positive-utility region (top score bin ≈ 0 utility). Not established: absence of information; more-label benefit.
