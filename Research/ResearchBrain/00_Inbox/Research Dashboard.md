# Research Dashboard

## Current highest-priority experiment
- [[2026-09-15 RGC Shift Recoverability]]

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

## Candidate ideas
- [[Predicting geometric reliability under distribution shift]]
- [[Decision-Native Foundation Models for Calibrated Parallel Decisions]]

## Killed / parked
- [[Simple neighbourhood-statistics gate for geometric correction]]
- [[GC-DAC confidence-gated anchoring on CIFAR-100]]

## Reading map
- [[00 Paper Map - Geometric Uncertainty to Actionable Confidence]]

## Latest synthesis
- [[2026-W38]]
