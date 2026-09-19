---
type: observation
status: open
date: 2026-09-15
project: rgc-shift
evidence_strength: 2
tags: [seed, instability, knn]
---

# Seed 1 occupies a qualitatively different kNN regime

## Observation

Seed 1 has very different neighbour statistics and geometry accuracy from seeds
2-5. On clean data its median top-1 neighbour count is 200/200, while seeds 2-5
have medians around 19-28. Under corruption, its geometry accuracy is also much
higher than most other seeds.

## Why this matters

The experimental seed currently changes multiple things together: trained
checkpoint and RGC internal randomisation. Therefore this anomaly cannot yet be
attributed to layer sampling, projection, or training randomness.

## Decisive next test

Decouple:
1. fixed checkpoint × multiple RGC randomisations;
2. multiple checkpoints × fixed RGC configuration.

This is a mechanism diagnostic, not a reason to tune the main result.

## Update 2026-09-15 — precise numbers from [[2026-09-15 G3 Headroom Null]]

While investigating why an independent second head beats geometry on
oracle-union headroom (mean per-class-geometry standalone accuracy 0.294 vs.
head 0.487, averaged over 12 CIFAR-100-C cells and 5 seeds — geometry is a
much weaker standalone classifier, which alone plausibly explains most of
the rescue-precision gap), per-seed per-class geometry accuracy was computed
directly:

| seed | head_acc | per-class geometry acc | global-kNN acc |
|---|---|---|---|
| 1 | 0.4817 | **0.4787** | 0.4807 |
| 2 | 0.4856 | 0.2591 | 0.2659 |
| 3 | 0.4845 | 0.2572 | 0.2711 |
| 4 | 0.4977 | 0.1881 | 0.1853 |
| 5 | 0.4853 | 0.2872 | 0.3017 |

Seed 1's geometry accuracy (both per-class and global-kNN) is nearly
identical to its own head accuracy — categorically different from seeds
2-5, where geometry accuracy is 30-60 pp below the head. This is now a
precise, quantified confirmation of the original qualitative claim ("under
corruption, its geometry accuracy is also much higher than most other
seeds"), still n=1 anomalous seed out of 5 — the decisive next test above
(decoupling checkpoint from RGC randomisation) has not been run, so this
remains an open mechanism question, not yet an explanation.

## Update 2026-09-15 — mechanism narrowed, checkpoint strength ruled out

From [[2026-09-15 G3 Controlled Complementarity]], computed directly from
cached `neighbour_labels` using the exact `neighbour_statistics()` formula
in `rgc_shift/recoverability.py` (see that card's correction note on
`rgc-shift.zip`):

| seed | true_label_purity | neighbourhood_concentration | clean checkpoint accuracy |
|---|---|---|---|
| 1 | **0.477** | **0.914** | 75.28% (lowest of the 5) |
| 2 | 0.091 | 0.156 | 76.20% |
| 3 | 0.088 | 0.147 | 76.38% |
| 4 | 0.059 | 0.117 | 76.31% |
| 5 | 0.104 | 0.162 | 76.35% |

**Checkpoint strength is ruled out**: seed 1 has the *lowest* clean
accuracy of the 5, not the highest, so the anomaly is not "seed 1 happens to
have the best classifier." The measurable difference is neighbourhood
structure — seed 1's corrupted-cell neighbourhoods are 5-8x more
concentrated/pure than seeds 2-5's. This still does not decide between RGC
randomisation and geometry configuration as the cause (the decoupling test
below remains unrun), but narrows "qualitatively different kNN regime" to a
specific, quantified property.

**Also found**: the `perclass` and `globalknn` exports' `neighbour_labels`/
`neighbours` arrays are byte-identical per (seed, cell) — both query the
same global kNN index; only the vote/decision rule differs. So this
concentration/purity number is a property of the shared retrieval, not
independently computed twice — it is not evidence that per-class and
global-kNN geometry are independently anomalous, only that whatever is
different about seed 1's retrieval affects both arms' inputs identically.

## Update 2026-09-15 — sharper saturation numbers from the corrected k_vote audit

[[2026-09-15 G3 k_vote Audit Corrected]] recomputed
`head_neighbour_agreement` and `neighbourhood_concentration` with the exact
`rgc_shift` formulas (the previous k_vote audit's `agreement` feature was a
different, invented quantity — see that card's superseded banner). At
k_vote=50, `frac(head_neighbour_agreement == 1.0)` = **0.529** for seed 1
vs. 0.0000-0.0030 for seeds 2-5; `neighbourhood_concentration` and
`neighbour_margin` saturate at almost exactly the same fractions (0.530 /
0.0000-0.0030). Roughly half of seed 1's samples sit at the absolute
maximum of all three features simultaneously — a regime seeds 2-5
essentially never reach. This is the most precise version of this
observation to date; still n=1 anomalous seed, still does not decide
between RGC randomisation and checkpoint identity as cause (the decoupling
test below remains unrun).

## Update 2026-09-15 — fails to match any synthetic weakened-head setting

[[2026-09-15 G3 Synthetic Matched-Strength Null]] built a grid of
deliberately-weakened independent-head predictors spanning
disagreement 0.44-0.98 (temperature and Gaussian-noise families) to test
seeds 2-5's geometry against a matched-strength null. Seed 1's own
disagreement rate with its head (0.073 per-class, 0.156 global-kNN) sits
below the *entire* grid's achievable range at any accuracy — even the
mildest weakening setting produces ~0.44 disagreement. **Matching FAILED**
for seed 1 at both tolerance stages, for both arms. This is a new,
independent line of evidence for the same conclusion as the concentration/
purity numbers above: seed 1's behaviour is not reproducible by generically
degrading an ordinary independent head's predictions — whatever is
different about it is not just "seed 1 happens to disagree with its head at
an unusual rate," but a qualitatively different regime the synthetic null's
degradation mechanism cannot reach at all.
