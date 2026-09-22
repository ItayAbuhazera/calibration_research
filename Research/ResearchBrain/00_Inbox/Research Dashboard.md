# Research Dashboard

## Current highest-priority experiment
- [[2026-09-15 RGC Shift Recoverability]]

## Most recent completed experiment
- [[2026-09-20 Full-Vector DAC POC]] — **stopped by its own frozen
  continuation rule (Outcome E); H-FVDAC-01 closed as negative.** Seed 4
  only. A **weak label-aligned signal** is present (permuted-label control
  exactly null: beta=0, zero flips in all 12 cells; real labels beta=6.43),
  but the experiment does **not** establish a robust or practically useful
  decision improvement: +0.14 pp mean corruption accuracy (~10x below the
  predeclared bar), intervention precision **0.186** (68% of flips are
  wrong->different-wrong), the effect **shrinks** with severity, Vector
  Scaling with no geometry reaches +0.086 pp, and it gives back 82.5% of
  native DAC's ECE gain. 4 of 5 criteria passed; the rule requires 5, so
  seeds 1/2/3/5 and the remaining 11 corruptions were not launched.
  Re-audited 2026-09-20: earlier "the mechanism is real, not an artefact"
  withdrawn as an overclaim. **Not** an RGCL rescue; prior full-vector
  negatives stand, unrevised.

## Prior experiment history now linked
- [[2026-04-25 CIFAR10 ResNet18 Full-Vector Path Batch]]
- [[2026-04-25 CIFAR100 ResNet18 Full-Vector Path Batch]]
- [[2026-04-26 Post-Fusion Top-Isotonic Smoke]]
- [[2026-04-27 TinyImageNet ResNet50 Full-Vector Path Batch]]
- [[2026-06-21 Decision-Improving Calibration Pilot]]

## Observations by evidence strength

Sorted by the note's own `evidence_strength`. Nothing is listed as "strong"
unless its own front matter says so.

### Structural / proven (5, 4)
- [[Sample-dependent scalar temperature cannot change the predicted class]] (5, algebraic - a control, not empirical support)
- [[Top-coordinate recalibration plus tail renormalization can change argmax]] (4, one seed plus algebra)

### Supported, multi-seed (3)
- [[Geometry contains complementary accuracy information under corruption]] - blocked by missing oracle-union null
- [[Per-class geometry exposes more oracle headroom than global kNN]] - blocked by missing oracle-union null
- [[Rank-geometric anchored mixture collapses to zero weight]]
- [[Clean-selected probability blending collapses to head-only]]
- [[Top-label calibration and full-vector proper scoring form a Pareto frontier]]
- [[High-confidence anchoring can block useful full-vector decision changes]] - partly true by construction

### Weak / single-seed (2) - do not cite as support
- [[Full-vector geometric fusion can collapse to the base model on easy regimes]]
- [[Seed 1 occupies a qualitatively different kNN regime]] - low strength, but may explain much of the between-seed variance; worth raising, not citing

## Integrity queue (clear before new work)

- [x] **Oracle-union null baseline.** Head + second independently trained head, and head + noise-matched predictor. Blocks O1 and O3. — Done 2026-09-15, see [[2026-09-15 G3 Headroom Null]]: geometry clears the noise-matched null (~10x) but loses to the independent-head null (roughly half the headroom/rescue-precision); paired CIs computed, gap is real. Follow-up [[2026-09-15 G3 Controlled Complementarity]] tried to control for standalone predictor strength — **inconclusive**, no existing predictor spans geometry's accuracy range (0.19-0.30 vs. everything else at 0.48-0.50). New item below.
- [x] **Fill the 0.30-0.45 accuracy gap for a real strength-matched complementarity test.** — Partially done 2026-09-15 via a different route: [[2026-09-15 G3 Synthetic Matched-Strength Null]] built a *generic* (non-same-representation) matched-strength/matched-disagreement null from deliberately-weakened independent-head logits (no new compute). Result: **B, geometry consistent with the matched synthetic null** for the seeds-2-5 aggregate (excess headroom/rescue-precision within ~0.35pp of zero). The *same-representation* version (raw embeddings -> weakened readout, forward pass through the existing checkpoints) named in [[2026-09-15 G3 Controlled Complementarity]] is still not run and remains the one path to the stronger claim — kept open below.
- [ ] **Same-representation weakened readout.** Extract raw test-cell embeddings for the 12 corrupted cells (forward pass through the existing 5 checkpoints, no retraining) and fit one deliberately-weakened same-representation readout (nearest-centroid on a coarse layer, or a heavily-regularised linear probe) landing near 0.30-0.45 accuracy. Minimal experiment named in [[2026-09-15 G3 Controlled Complementarity]], still not run — the item above answered the generic version of this question instead.
- [x] **Re-run the gate arm at the pre-registered `k_vote=50`.** — Done properly 2026-09-15: [[2026-09-15 G3 k_vote Audit Corrected]] uses the exact `rgc_shift` (pinned `ff81032f...`) formulas for `head_neighbour_agreement`/`neighbourhood_concentration`, and the confirmed `export_recoverability.py` formula for `neighbour_margin` (`knn_radius` remains uncomputable at k=50 from cache, preserved at k_radius=200 per design). The original univariate audit — [[2026-09-15 G3 k_vote Audit]] — is now marked **SUPERSEDED — INCORRECT FEATURE DEFINITION** (its `agreement` feature was a self-invented binary proxy, not the real continuous formula) and left in place, not deleted. Corrected verdict: **k_vote CHANGE MINOR** — same practical conclusion as before, but the `head_neighbour_agreement` feature now shows a materially larger *negative* delta at k=50 (was a small negative with the wrong formula) while `neighbourhood_concentration`/`neighbour_margin` still show small positive deltas. `knn_radius` audited for the first time (invariant, near-chance).
- [ ] **Record standalone geometry accuracy and head-geometry disagreement rate** for both geometry arms.
- [ ] **Locate the GLAD-PI pilot outputs** or leave [[2026-06-21 Decision-Improving Calibration Pilot]] as design-only.
- [ ] **Report at least one matched-risk / matched-coverage comparison** (README rule 6 is currently unmet everywhere).
- [ ] **Finish the thin paper notes.** Six are `audited` (Mahalanobis, Atypicality, DAC, KCal, Trust Score, TULIP) and [[Conceptual Prior Art - Local Competence and Reliability Routing]] is written. Still thin: [[Uncertainty Estimation Based on Geometric Separation]] and [[Semantic Geometric Calibration in Randomized Neural Feature Space]] (the two parents of this line), plus Dirichlet calibration, SelectiveNet, Conformal Risk Control, Deep kNN, and the CIFAR-C source.
- [ ] **Record a CI for the Tiny-ImageNet fusion gain** (+0.69 pp accuracy, -0.028 NLL). No interval is stored, so the only cross-dataset positive result is currently a point estimate.
- [ ] **Add one shift benchmark that is not CIFAR-100-C** - natural shift (ImageNet-R/V2, CIFAR-10.1, WILDS) or ImageNet-C - and one non-ResNet architecture. [[CIFAR-C]] itself warns its corruption families are not a sample of possible shifts.
- [ ] **Write the experiment card for the actionable-confidence smoke before running it.** [[2026-W38]] queues it; no note exists.

Fixed on 2026-09-15: broken RGC-shift link; transfer-claim in the failure-mode
title (the old "Clean-fitted ... do not transfer" file is now a superseded stub -
delete it manually); "strong observations" list that ignored `evidence_strength`; `gc_dac` /
`rgcl_tail_vector_scaling` naming (author decision: `rgcl_tail_vector_scaling`).

## Open hypotheses
- [[H-RGC-01 Reliability mapping shifts under corruption]]
- [[H-RGC-02 Richer representation features predict geometric reliability]]
- [[H-FVDAC-01 Class-conditioned DAC density enables decision correction]]

## Candidate ideas
- [[Predicting geometric reliability under distribution shift]]
- [[Decision-Native Foundation Models for Calibrated Parallel Decisions]]
- [[Full-Vector Density-Aware Calibration]]

## Killed / parked
- [[Simple neighbourhood-statistics gate for geometric correction]]
- [[GC-DAC confidence-gated anchoring on CIFAR-100]]

## Reading map
- [[00 Paper Map - Geometric Uncertainty to Actionable Confidence]]

## Latest synthesis
- [[2026-W38]]

## Update 2026-09-21
- **Most recent completed experiments:** [[2026-09-21 Layer-Selection Pilot]] (frozen rule verdict: *no material evidence to continue the tested family*; best
  candidate +0.095 pp vs Vector Scaling +0.190 pp; 2 seeds; GAP) and [[2026-09-21 Normalization Audit and Corrected Protocol]] (verified: legacy CIFAR-100 test/C used ImageNet
  statistics on CIFAR-trained checkpoints; corrected protocol is now the benchmark default).
- **New integrity items:**
  - [ ] Ask the RGC/IJCAI authors to check whether published CIFAR-100 clean-test numbers used the legacy `get_test_loader` (same mismatch) — [[Legacy benchmark evaluated CIFAR-100 with ImageNet statistics on a CIFAR-trained checkpoint]].
  - [ ] Corrected-protocol rerun of the other Phase 0/1 methods if the benchmark table is to be reported (only base/TS/VS/native DAC + pilot arms exist in `results/studyAB/phase0_corrected_v2/`).
  - [ ] Read the `[unverified]` sources in [[Theory Plan - Decision Utility, Layers, Compression and Risk Control]].
- **Open hypothesis:** [[H-LAYER-01 Selected internal layers add decision value beyond logits]] (not supported under this protocol).
- **Reading:** [[From Similarity to Decisions - PCE (AAAI submission)]], [[Theory Plan - Decision Utility, Layers, Compression and Risk Control]].

## Update 2026-09-21 (second pass)
- **Most recent completed experiment:** [[2026-09-21 Residual Evidence Study]] — stopped by its frozen gate (criteria 1–4 failed); no confirmation. Closes [[H-RESID-01 Source-learnable residual decision information at layer3.22]] under this protocol; the finite-sample learnability limit (2 500 clean fit rows) is the observed bottleneck, not a demonstrated absence of information.
- **Integrity queue additions:** [ ] corrected benchmark corruption cells (array 21533078) → run `Experiments/crosscheck_residual_vs_benchmark.py`; [ ] IJCAI-era CIFAR-100 preprocessing provenance (stored RGC-repo ablation JSONs carry legacy-normalization accuracies; which artifacts fed published tables is unknown — no contact made, no claim edited); [ ] the pilot's TS/VS baselines were fit on the whole validation split (baseline-only role incompatibility, verdict unaffected).

## Update 2026-09-21 (third pass)
- **Most recent completed program:** [[2026-09-21 Representation Atlas Program]] — H-ATLAS-01 refuted under frozen conditions; new fact: correct hidden alternatives exist (deep layer3, spatial pooling helps) but with 38–52 % harm on base-correct examples; clean selection cannot pick or gate them. Corrected benchmark reconciliation: seed-2 cells done, seed-4 cells still running (array 21533078). IJCAI-era preprocessing provenance still unresolved.

## Update 2026-09-21 (fixed deep-candidate gate study) — [[2026-09-21 Fixed Deep Candidate Gate Study]], [[H-GATE-01 Candidate selection versus gate utility mismatch]]
Status: completed (development, checkpoints 2 and 4). Fixed `layer3.22` 2×2 kNN candidate with C0/C1/Z0/Z1 ridge gates and n∈{625,1250,2500} learning curves: no consistent held-out benefit (Z1−Z0 clean +0.10/+0.13 pp, corruption −0.04/−0.04); deep gates ≈ output-evidence and layer4/output controls; practical targets not met; probability quality not improved. Deep candidate has 873/908 fit disagreements (the 52–151-event limit was layer4/output-specific) but H:W ≈ 3.8:1. Audit corrections to the atlas report appended in its card; TF32 mismatch cause verified (benchmark = TF32 convs, batch 128); seed-4 reconciliation deferred jobs 21537819–21. Measurement note for theory: standalone E[D] and gated E[gD] differ, but here neither exposed a large positive-utility region (top score bin ≈ 0 utility). Not established: absence of information; more-label benefit.
