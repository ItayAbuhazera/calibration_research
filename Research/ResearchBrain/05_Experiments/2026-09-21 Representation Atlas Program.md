---
type: experiment
status: completed_development_all_hypotheses_refuted
date: 2026-09-21
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C (12 development cells), ResNet-101, checkpoints 2 and 4, corrected_v2_train_norm
preregistered: true
evidence_scope: unpublished_repository_analysis
tags: [atlas, pooling, spp, knn, mahalanobis, gate, probability-quality, visualization]
---

# 2026-09-21 — Representation atlas, distance diagnostics and selective correction

Newly authorized diagnostic program; **does not reopen** [[2026-09-20 Full-Vector DAC POC]], [[2026-09-21 Layer-Selection Pilot]] or [[2026-09-21 Residual Evidence Study]] (their verdicts and gates stand).
Frozen spec: repo `Research/GeometricFullCalibration/docs/atlas_program_spec.md` (hash in `results/atlas/freeze_manifest.json`); job table `docs/atlas_job_decisions.md`; run ledger `results/atlas/ledger.json`.
Hypotheses: [[H-ATLAS-01 Accessible correction information across layers pooling and metrics]]. Theory: [[Theory Plan - Decision Utility, Layers, Compression and Risk Control]].

## Plan (labels: proposed)
A. 34 sites (stem + 33 blocks) × {GAP, 2×2, SPP(1,2,4; weights 1/q, /√levels)}, exact kNN (k=50, unit-L2) diagnostic classifier with smoothed `p_geo`, versus the same construction in logit space; clean-only source shortlist (one per pooling).
B. unit-L2 vs norm-preserving Euclid vs low-rank+isotropic regularized Mahalanobis on the ≤3 selected candidates and on logits.
C. small ridge gate (F0 output/logit-derived features vs F1 + geometry) per candidate; D. NLL/Brier-fitted final temperature on a held-out calibration split, full 10 000-image evaluation, latency.
V. clustering / PCA trajectories / neighbour galleries as diagnostics. Primary outcomes ACC, NLL, Brier; ECE secondary (the earlier ECE gate is not inherited).

## Results (measured; repo spec §13, full tables `results/atlas/report/`, figures `results/atlas/figures/`)
* Cost: **0.58 GPU-hours**, ≈1.7 GB/checkpoint; latency (RTX 4090, exact 45 000-row search) batch-1 10.5 ms backbone → 10.4–12.2 ms with GAP/2×2/SPP search; SPP bank 7.7 GB.
* Atlas (target-labelled diagnostic): correct alternatives on base errors peak at **deep layer3** (kNN accuracy on base errors GAP 0.10 → 2×2 0.13 → SPP 0.13–0.14; logit-space kNN 0.025) but the same candidates are wrong on **38–52 %** of base-correct examples; net utility is positive only at `layer4.1/4.2` (≈ the head itself, +0.1–0.2 pp).
* Clean-only shortlist picked `layer4.x` in every pooling family; gate learned ≈ nothing (≤151 clean disagreements); hidden-F1 +0.07/+0.29 pp vs base, −0.01/+0.04 pp vs the strongest output control; raw NLL/Brier never beat native DAC; clean-fitted final temperature hurt under shift.
* All five frozen hypotheses (H-A…H-E) **refuted**; regularized Mahalanobis (frozen estimator) catastrophic on high-dimensional hidden features.
* Complementarity: hidden and output candidates share few repairs (Jaccard 0.34 / 0.65) but equally many unique harms; oracle-union increment of hidden over output candidates +0.8 pp (evaluation-only oracle; not a bound on a feasible gate).
* Reconciliation with the corrected benchmark: 17 cells so far — labels and pixels identical; base disagreements are near-ties (TF32); native DAC/VS differ by designed role differences; seed-4 corruption cells still running.

## Interpretation (labelled)
**Supported (development):** hidden mid-layer representations contain alternatives that output space lacks. **Weakened:** that a clean-selected hidden candidate + small gate beats matched output pipelines. **Unidentified:** candidate-selection vs intervention-selection vs shift-transfer limits (data consistent with an intervention-selection/transfer limit; the clean candidate-selection rule is itself a limit).
**Not established:** absence of usable information; that gates with more clean supervision cannot work; anything across seeds beyond n=2; significance.

## Provenance and unresolved
Frozen spec hash in `results/atlas/freeze_manifest.json`; job table `docs/atlas_job_decisions.md`; ledger `results/atlas/ledger.json`; IJCAI-era preprocessing provenance remains unresolved (see [[2026-09-21 Normalization Audit and Corrected Protocol]]).

## Corrections appended 2026-09-21 (original report above preserved) — see [[2026-09-21 Fixed Deep Candidate Gate Study]]
* "All five hypotheses refuted" = all five failed their frozen continuation thresholds. Statistically: H-A/H-D/H-E have descriptive intervals below the frozen magnitudes for the clean-selected layer4 candidates; H-B is seed-dependent (−70/+80, uncertain); H-C is at the margin for raw_l2 (+0.48 vs 0.5 pp) and negative only for the frozen Mahalanobis estimator. Detail: repo `results/fixed_gate/report/hypothesis_audit.json`.
* "10–14 % repair / 38–52 % harm": corruption-macro rates on the atlas subset at target-label-peak layer3 sites; repair rate is not net utility. Matched recomputation: deep layer3.22 2×2 repairs 12.9/13.4 % and harms 40/42 % under corruption (clean 20–21 % / 25–27 %), net −12 pp; the logit control repairs 2.5 % and harms 1.9 %.
* The +0.8 pp oracle union did not include deep-layer3 spatial candidates (exact set in the fixed-gate audit) and is not a bound on feasible gains.
* "TF32 near-ties": now verified — the benchmark path is TF32 convs at batch 128 (no autocast); reproducing it gives 0 base mismatches. The gates' 52–151 events were specific to the layer4/output candidates (deep candidate: 873/908 disagreements).
* Scope: candidate alternatives exist; useful conditional intervention value is not established; finite-sample difficulty is plausible, not identified; improved repair recall in one layer is not an end-to-end spatial-pooling gain.
