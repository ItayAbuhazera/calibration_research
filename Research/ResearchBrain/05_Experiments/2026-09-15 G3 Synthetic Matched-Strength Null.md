---
type: experiment
status: completed
date: 2026-09-15
project: rgc-shift
benchmark: CIFAR-100-C
model: ResNet-101
seeds: [1, 2, 3, 4, 5]
preregistered: false
tags: [geometry, complementarity, strength-control, null-test, synthetic]
---

# G3 Synthetic Matched-Strength Null

## Question

[[2026-09-15 G3 Controlled Complementarity]] landed **INCONCLUSIVE — NO
ADEQUATE MATCHED-STRENGTH CONTROL** because no *real* predictor family in
this repo spans geometry's standalone-accuracy regime (~0.19-0.30 for seeds
2-5): everything else available (independent heads, full-vector fusion)
clusters at 0.48-0.50. That card's named next step required a fresh forward
pass through the checkpoints (new compute). This experiment builds the
matched-strength control a different way, without any new compute: by
deliberately weakening already-cached independent-head *logits* until they
land in geometry's accuracy/disagreement regime, then asking whether
geometry's oracle-union headroom and rescue precision exceed what that
matched synthetic null produces.

## Hypothesis tested

Same three-way split as Controlled Complementarity: (A) geometry shows
excess complementarity above a strength-and-disagreement-matched null; (B)
geometry is consistent with such a null; (C) matching fails / remains
inconclusive.

**Explicit scope, stated up front**: a synthetic null tests whether generic
degraded-head disagreement explains the geometry result. It does **not**
establish that RGC geometry is uniquely useful relative to other
same-representation readouts (that would need a real, same-representation,
weakened readout — the experiment Controlled Complementarity specified and
did not run).

## Setup

New script `GeometricFullCalibration/scripts/g3_synthetic_matched_null.py`
(additive; does not modify `g3_headroom_null.py`, `g3_kvote_audit.py`, or
`g3_controlled_complementarity.py`). No GPU inference, no retraining, no new
routing method — reads only cached `logits`, `y_pred_head`, `y_pred_knn`,
`y_true` from `exports/seed{1-5}/{perclass,globalknn}/{cell}.npz`.

**Synthetic predictor families**, built from independent heads' cached
logits (seed B != primary seed A, same 12 shift cells):

1. `temperature_gumbelmax`: `y' = argmax(logits_B / T + g)`,
   `g ~ Gumbel(0,1)` iid per class. Exactly equivalent in distribution to
   categorical sampling from `softmax(logits_B / T)` (the Gumbel-max
   trick) — verified algebraically (`argmax(z/T + g) = argmax(z + T*g)` for
   `T>0`, since argmax is invariant to positive rescaling). This single
   family therefore also stands in for a literal "additive Gumbel
   perturbation" family; implementing both separately would be a
   reparametrisation of the same operation, not a second distinct family,
   so it was not duplicated.
2. `gaussian_additive`: `y' = argmax(logits_B + sigma * eps)`,
   `eps ~ Normal(0,1)` iid per class. Genuinely distinct from (1): additive
   Gaussian noise does not respect the same scale-invariance, so it
   degrades the predictor along a different path through logit space.

Fixed, deterministic RNG per `(primary_seed, source_seed, cell, family,
param, draw)` (sha256-derived, same style as the `matched_stochastic` null's
seeding in [[2026-09-15 G3 Headroom Null]]). For a given primary seed A, one
synthetic instance at a (family, param) setting pools counts across the
other 4 seeds' logits (as "any independent head", matching how
`independent_head` pooled in Headroom Null) and across all 12 cells,
*before* taking ratios — same anti-Jensen's-inequality pooling as
[[2026-09-15 G3 Controlled Complementarity]]. Per-draw pooled ratios are
kept so across-draw spread is a genuine Monte-Carlo uncertainty on the
pooled statistic.

**Grid, and a disclosed compute-driven deviation**: the originally-declared
grid was `T ∈ {0.5,0.75,1,1.25,1.5,2,3,5,8,12}`,
`sigma ∈ {0.25,0.5,1,1.5,2,3,5,8,12,20}`, 8 Monte-Carlo draws. The shared
compute host this ran on was under heavy *external*, unrelated multi-tenant
load (load average ~31 on a 4-core machine, confirmed via `uptime` at the
time, not caused by or controllable from this task) — the full grid did not
finish in tractable wall-clock time. Thinned once, before any results were
inspected, to `T ∈ {0.5,1,1.5,3,5,12}`, `sigma ∈ {0.25,1,2,5,8,20}`, 5 draws.
That first run then revealed a genuine grid-density problem, not a
scientific finding: the thinned grid jumped from accuracy ~0.29 (T=1.5)
straight to ~0.07 (T=3), skipping over the entire 0.19-0.30 target region
for most seeds (same gap for sigma 2->5). This was **densified once more**,
before interpreting any excess/headroom numbers, to
`T ∈ {0.5,1,1.5,1.75,2,2.25,2.5,3,5,12}`,
`sigma ∈ {0.25,1,2,2.5,3,3.5,4,5,8,20}` (12+8 = 20 settings, still 5 draws).
The decision to add points was driven by "did the grid reach the declared
target region", not by which residual sign would look favorable — no
geometry excess values were computed or consulted before this second
change. Final run: 20 settings x 5 draws x 4 source seeds x 12 cells x 5
primary seeds = 24,000 synthetic-prediction evaluations, each a single
`argmax` over a (10000, 100) array.

## Metrics (per B1)

`accuracy_B`, `disagreement(A,B)`, `headroom = P(A wrong, B right)`,
`rescue_precision = P(A wrong, B right | A != B)`,
`both_wrong_disagreement = P(A wrong, B wrong | A != B)`. Computed for every
synthetic instance, and for the real references (per-class geometry,
global-kNN geometry, independent_head recomputed fresh here for pooling
consistency, full_vector_fusion), all pooled across the 12 cells per
primary seed.

## Matching (per B2)

Predeclared tolerance stage 1: `|acc diff| <= 0.02`, `|disagreement diff| <=
0.03`; stage 2 (if no stage-1 match): `<= 0.03` / `<= 0.05`; else `MATCHING
FAILED`. Applied separately per (arm, seed), and at two aggregate scopes
(seeds 2-5 pooled, all-5-seeds pooled/secondary), matching each geometry
point against the synthetic table built from *that same primary seed(s)*.

## Results

### Matching outcome

| scope | arm | tolerance | n matched | excess_headroom | excess_rescue_precision |
|---|---|---|---|---|---|
| seed 1 | per-class | **FAILED** | 0 | — | — |
| seed 1 | global-kNN | **FAILED** | 0 | — | — |
| seed 2 | per-class | stage1 | 1 | +0.0021 | +0.0023 |
| seed 2 | global-kNN | stage1 | 1 | -0.0039 | -0.0031 |
| seed 3 | per-class | stage1 | 1 | +0.0041 | +0.0046 |
| seed 3 | global-kNN | stage2 | 2 | -0.0014 | -0.0019 |
| seed 4 | per-class | stage2 | 2 | **+0.0104** | **+0.0118** |
| seed 4 | global-kNN | stage1 | 2 | +0.0085 | +0.0102 |
| seed 5 | per-class | stage1 | 1 | +0.0066 | +0.0069 |
| seed 5 | global-kNN | stage1 | 2 | -0.0018 | -0.0023 |
| **seeds 2-5 (primary)** | per-class | stage1 | 1 | **+0.0035** | **+0.0027** |
| **seeds 2-5 (primary)** | global-kNN | stage1 | 1 | **-0.0006** | **-0.0000** |
| all 5 (secondary) | per-class | stage2 | 2 | -0.0085 | -0.0098 |
| all 5 (secondary) | global-kNN | **FAILED** | 0 | — | — |

Per-setting Monte-Carlo std on headroom/rescue-precision is ~0.0001-0.0004
(tight — MC resampling noise alone is small); cross-setting spread among
matched settings for a given point is larger, up to ~0.005-0.009 where 2
settings matched, reflecting genuine sensitivity to *which* nearby setting
happens to fall inside the tolerance window with only 1-2 matches per
point. Full detail, including which settings matched and both uncertainty
components, in `results/g3_synthetic_matched_null/matching.json`.

**Seed 1 fails to match at either tolerance stage for both arms** — expected,
not a new finding. Seed 1's own disagreement rate with its head (0.073-0.156)
sits far below the 0.4-0.8 band the whole synthetic grid was built to cover
(even the mildest weakening setting, T=0.5, produces ~0.44 disagreement).
This is consistent with, and does not resolve,
[[Seed 1 occupies a qualitatively different kNN regime]] — its anomaly is
not reproducible by generically weakening an ordinary independent head.

### Primary result: seeds 2-5 aggregate

The scope the task named as primary target coverage matched cleanly at
stage-1 tolerance for **both** arms, and the excess is small and
close to zero in both directions: per-class +0.35pp headroom / +0.27pp
rescue precision; global-kNN -0.06pp / ~0.00pp. For comparison, the raw
oracle-union headroom this whole research line has been investigating is
3.80-4.89pp (see [[2026-09-15 G3 Headroom Null]],
[[2026-09-15 G3 Controlled Complementarity]]) — the matched-null excess
here is roughly an order of magnitude smaller than that headroom itself,
and smaller than the independent-head-vs-geometry gap's 95% CI half-widths
(~0.6-0.8pp) reported in Headroom Null.

### Individual seeds

Seeds 2, 3, 5 show small, sign-mixed excess (-0.4pp to +0.7pp) depending on
arm — no consistent direction. **Seed 4 is the one point that leans more
clearly positive** (+0.85 to +1.18pp across both arms, its largest single
values in the table) — still under 1.2 percentage points and comparable in
magnitude to the cross-setting spread (~0.35-0.53pp) among its own matched
settings, but worth flagging honestly rather than averaging away: it is the
one seed where "excess complementarity" is not clearly distinguishable from
matched-setting noise.

### Plots

`results/g3_synthetic_matched_null/plots/{accuracy_vs_headroom,
accuracy_vs_rescue_precision, disagreement_vs_headroom,
accuracy_disagreement_headroom}.png`. All four show geometry (seeds 2-5,
both arms) sitting essentially *on* the synthetic accuracy/disagreement ->
headroom/rescue-precision curve traced out by the temperature and Gaussian
families, visually consistent with the near-zero excess numbers above.
Seed 1 is a clear, separate outlier in the disagreement-vs-headroom plot:
far lower disagreement (~0.07-0.16) than its ~0.48 accuracy would predict
from the synthetic trend, consistent with it being excluded from matching.
Real independent heads and full-vector fusion sit at the high-accuracy end
of the same trend (extrapolated, not matched against — a secondary
observation, not part of the formal test).

## B4 Interpretation

**B. GEOMETRY CONSISTENT WITH MATCHED SYNTHETIC NULL** — for the primary
seeds-2-5 aggregate scope (both arms), which is where the target coverage
was explicitly aimed and where matching succeeded cleanly at stage-1
tolerance. Individual-seed results are consistent with this (seeds 2, 3, 5:
near-zero, sign-mixed; seed 4: a small positive excess worth continued
attention, not ignoring, but far below the raw headroom magnitude that
motivated this whole investigation).

This is **not** the same claim as Controlled Complementarity's state B
("explained by strength") being fully proven — it specifically means:
*generic degraded-head disagreement, matched on standalone accuracy and
disagreement rate alone, already reproduces geometry's oracle-union
headroom and rescue precision to within ~1pp.* It does not test, and does
not rule out, whether a same-representation weakened readout (the
experiment Controlled Complementarity specified and this one did not run)
would behave differently — that remains the one open path to distinguishing
"geometry is just a weaker predictor" from "geometry is weaker but a
different kind of predictor."

## What this does NOT establish

- Does not establish that geometry carries literally zero unique
  information — only that a generic, non-geometric, degraded predictor at
  matched strength and disagreement already accounts for the observed
  headroom to within about 1 percentage point, an order of magnitude below
  the effect size that motivated the original question.
- Does not resolve the strength-vs-type distinction from Controlled
  Complementarity in the strong sense (same-representation weakened
  readout) — only in the weaker, still meaningful sense (any-representation
  weakened readout, matched on accuracy + disagreement).
- Seed 4's mildly larger excess is not explained here — could be genuine,
  could be residual grid-matching noise (only 2 settings within tolerance,
  spread ~0.35-0.53pp). Not investigated further in this pass.
- Grid was thinned/densified twice for disclosed compute-availability
  reasons (see Setup) — a different starting grid density might shift which
  settings fall inside the matching tolerance, though the underlying
  accuracy/disagreement/headroom relationship traced by the plots is smooth
  and consistent across both thinned and densified runs where they overlap.
- n=5 seeds (4 usable for the seeds-2-5 scope), one architecture, one
  benchmark family. Do not generalize beyond ResNet-101/CIFAR-100-C.
- perclass and globalknn geometry remain correlated readouts of one shared
  retrieval (see [[Seed 1 occupies a qualitatively different kNN regime]]),
  analyzed here as two arms but not as independent evidence.

## Artifacts

- `GeometricFullCalibration/scripts/g3_synthetic_matched_null.py`,
  `scripts/g3_synthetic_matched_null_plots.py` (both new, additive).
- `GeometricFullCalibration/results/g3_synthetic_matched_null/{real_references.json,
  synthetic_raw_counts.json, synthetic_summary.json, pooled_2to5_summary.json,
  pooled_all5_summary.json, group_real_geometry.json, matching.json,
  plots/*.png}`.

## Next action

- If the seed-4 excess or the strong same-representation-readout question
  is pursued further, the concrete next step is still the one named in
  [[2026-09-15 G3 Controlled Complementarity]]: a forward pass through the
  existing 5 checkpoints to extract raw test-cell embeddings, then fit one
  deliberately-weakened same-representation readout landing in the
  0.19-0.30 accuracy band. Not run here (new compute, out of scope).
- Seed 1 remains unexplained by any generic-weakening account — the
  decoupling test named in
  [[Seed 1 occupies a qualitatively different kNN regime]] (fixed
  checkpoint x multiple RGC randomisations; multiple checkpoints x fixed
  RGC config) is still the decisive next step for that specific anomaly.
