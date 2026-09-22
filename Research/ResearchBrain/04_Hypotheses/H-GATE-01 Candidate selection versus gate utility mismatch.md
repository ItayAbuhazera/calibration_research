---
type: hypothesis
status: not-supported-under-this-protocol
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C, ResNet-101, checkpoints 2 and 4 (development)
novelty: unknown
tags: [H-GATE, candidate-selection, gate, layer3]
---

# H-GATE-01 — Selecting candidates by standalone accuracy (E[D]) discards a selectively deployable alternative (E[gD])

Experiment: [[2026-09-21 Fixed Deep Candidate Gate Study]]. Go criterion (frozen, practical, not a formal test): a fixed deep-layer3 2×2 candidate with a clean-trained gate improves held-out clean utility of Z1 over Z0 with the same sign and paired intervals excluding 0 in both checkpoints, transfers to the 12 corruption cells, and reaches +0.5 pp over base and +0.25 pp over the output control without worse NLL/Brier. Kill criterion: no consistent held-out benefit.
**Status:** not supported in development. Z1−Z0 clean +0.10/+0.13 pp (intervals include 0), corruption −0.04/−0.04; deep Z1 macro +0.075/+0.21 pp over base ≈ output-evidence gates and output/layer4 controls; no growth with n or m. Does not show that a different global selection algorithm cannot help, nor that information is absent.

**Correction (2026-09-22), verdict unchanged:** "not supported" means the go criterion (consistent held-out Z1−Z0 benefit, both signs, both checkpoints) failed — it does not mean every quantity is indistinguishable from zero. Deep-Z1-vs-base is a genuine small positive effect under corruption in both checkpoints (interval excludes 0: +0.075 pp [0.048,0.103] seed 2, +0.213 pp [0.139,0.289] seed 4); it is the geometry-specific increment (Z1−Z0) and the whole-system advantage over the output-candidate control that remain unresolved (intervals include 0 in both checkpoints). On the frozen gate's actual (narrow) intervention set, empirical utility is clearly positive: W/(W+H)=0.70 (seed 2) / 0.585 (seed 4), (W−H)/F=+22.6%/+7.2%. See the "Correction 2026-09-22" section of [[2026-09-21 Fixed Deep Candidate Gate Study]] for the full corrected table, exact W/H/U counts, and the numerical-bound scope (≈44×/≈15× smaller than the observed gains, not "orders of magnitude" as an earlier pass stated).
