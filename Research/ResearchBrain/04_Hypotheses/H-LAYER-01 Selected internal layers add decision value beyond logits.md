---
type: hypothesis
status: not-supported-under-this-protocol
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C, ResNet-101, checkpoint seeds 2 and 4 (corrected preprocessing)
novelty: unknown
tags: [layers, decision-utility, H-LAYER]
---

# H-LAYER-01 — Clean-selected internal layers add decision value beyond a logit-only readout under corruption

Decisive experiment: [[2026-09-21 Layer-Selection Pilot]] (spec frozen in repo
`docs/layer_selection_pilot_spec.md` before results). Theory: [[Theory Plan - Decision Utility, Layers, Compression and Risk Control]] §T1, §T3.

## Statement (falsifiable, tested combination)
For at least one predeclared greedy arm with L ∈ {4,6,8} layers (Family A: `softmax((z−β·mean_l r_l)/S_DAC)`; Family B:
equal-weight average of clean-fitted linear probes), fitted and selected on clean data with GAP-pooled ResNet-101
features, the mean CIFAR-100-C ΔAccuracy is ≥ +1.0 pp (≥ +0.5 pp in each of two seeds), exceeds Vector Scaling and a
full-logit probe by ≥ 0.25 pp, has W>H in ≥ 20/24 seed×cells, keeps ECE within +0.02 and NLL within +0.05 of native DAC,
and (Family A) beats its permuted-label control by 2×.

## Go / kill (frozen)
Go = all P1–P5 for ≥ 1 candidate arm (then: replication on fresh seeds/conditions, not a claim). Kill (this family, this
protocol) = no candidate arm reaches +0.5 pp seed-avg with +0.25 pp in each seed, or those that do fail calibration/permuted control.

## Status
**Not supported under this protocol** (2026-09-21): best candidate ΔAcc +0.095 pp (`B_greedy_L4`), +0.076 pp (`A_greedy_L4`);
verdict `no_material_evidence_to_continue_tested_family`. Scope: 2 seeds, 4 development corruption families, GAP, K_c=5, linear
probes only. **Not** an information-theoretic statement; **not** evidence that layers or geometry cannot help.
Open: a pre-declared 2×2 pooling sensitivity gave ≈ +0.3–0.44 pp for Family A (still below the floor, calibration cost) —
hypothesis-generating only (axis B, pooling), needs its own preregistration.

## Update 2026-09-21
Prospective follow-up on the pooling axis: [[H-RESID-01 Source-learnable residual decision information at layer3.22]] / [[2026-09-21 Residual Evidence Study]] — the +0.3–0.44 pp additive-β 2×2 signal did **not** reproduce as an accuracy advantage under a common residual readout. Status unchanged (not supported under this protocol).
