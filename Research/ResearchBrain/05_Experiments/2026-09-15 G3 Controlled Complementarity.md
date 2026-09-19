---
type: experiment
status: completed
date: 2026-09-15
project: rgc-shift
benchmark: CIFAR-100-C
model: ResNet-101
seeds: [1, 2, 3, 4, 5]
preregistered: false
tags: [geometry, complementarity, strength-control, null-test, inconclusive]
---

# G3 Controlled Complementarity

## Question

[[2026-09-15 G3 Headroom Null]] found that an independently-trained second
head beats geometry on oracle-union headroom and rescue precision, and
flagged standalone predictor strength (head/independent-head ~48.7% vs.
geometry ~29-30% accuracy) as a CORRELATIONAL, not identified, explanation.
This experiment asks: after controlling as well as existing artifacts allow
for standalone strength and disagreement rate, does geometry show unusual
error complementarity with the primary head?

Not a new gate or method — purely a diagnostic on existing per-sample
exports.

## Hypothesis tested

Three candidate answers, stated in advance (see Interpretation states
below): (A) geometry shows excess complementarity above a strength-matched
null; (B) the independent-head gap is fully explained by strength/
disagreement; (C) existing artifacts cannot distinguish A from B.

## Setup

**Inspection phase (no training launched)** found no natural same-type
strength ladder: all 5 ResNet-101/CIFAR-100 checkpoints cluster tightly
(75.28-76.38% clean accuracy; `EXPECTED_CLEAN_ACCURACY` in
`scripts/export_recoverability.py`), no intermediate/earlier-epoch
checkpoints exist anywhere in the repo, and no other ResNet-101/CIFAR-100
training-method family exists. Same-representation readouts (Mahalanobis/
linear probe) could not be fit-and-scored from cache either: raw embeddings
are cached only for the **training** split
(`artifacts/recoverability/seed{N}/reference_arrays.npz`, used to build the
kNN reference bank) — no raw embeddings for any evaluation cell (clean/val/
12 corrupted cells) are cached anywhere. Scoring a new readout on the
shift-eval cells would require a fresh forward pass through each checkpoint
over the corrupted images — new compute, not a cache read — and was not
approved for this pass.

Per this inspection, followed the "no natural ladder" fallback: fit a
descriptive model of rescue precision and oracle-union headroom as
functions of standalone accuracy and disagreement rate, using only the
non-geometric predictor families available, then compute geometry's
residual against that fit.

**Predictor families** (all on the same 12 shift cells x 5 seeds as
[[2026-09-15 G3 Headroom Null]], sample-id aligned):
- per_class_geometry, global_knn (as before).
- **full_vector_fusion** (`y_pred_vector`) — a fourth existing predictor
  found during inspection: a full-vector distance-fusion calibrator
  (`Calibrators/geometric_calibrator.py:FullVectorDistanceFusionCalibrator`)
  frozen per seed. Standalone accuracy ~48.7% (essentially identical to the
  head), disagreement with the head only ~8.4% — a near-head-clone, not a
  diverse strength point, but included as an available non-geometric family.
- independent_head, now built per ordered (primary seed, other seed) pair
  (20 instances) rather than pooled, to give the regression more rows.

Script: `GeometricFullCalibration/scripts/g3_controlled_complementarity.py`
(new file; the prior `g3_headroom_null.py` and `g3_kvote_audit.py` were left
untouched — confirmed via file timestamps, this repo gitignores `scripts/`
and `results/` entirely so there is no git history to diff against, only
disk state). Plots:
`scripts/g3_controlled_complementarity_plots.py`. Outputs:
`results/g3_controlled_complementarity/{controlled_complementarity_raw.json,
controlled_complementarity_jackknife.json, seed1_mechanism_check.json,
plots/*.png}`.

**Metric aggregation changed from [[2026-09-15 G3 Headroom Null]]**: this
script pools counts across the 12 cells before taking ratios (sum
correct/disagree/rescue, then divide), rather than averaging per-cell
ratios. This avoids that script's noted Jensen's-inequality inconsistency.
Numbers differ slightly (third decimal) between the two cards for this
reason — both are legitimate, differently-aggregated views of the same raw
per-sample data.

## Fixed choices

Descriptive model: `LinearRegression` (rescue_precision, oracle_headroom_pp)
~ (accuracy_B, disagreement), fit on non-geometric rows only (25 rows: 5
full_vector_fusion + 20 independent_head). Residuals computed for every
predictor instance, including geometry. Uncertainty via leave-one-checkpoint-
out jackknife (5 folds, each excluding one seed's checkpoint from both the
primary and non-geometric-predictor role) rather than treating the 25 rows
as independent (they reuse only 5 independent checkpoints in overlapping
pairs).

## Primary metrics

`accuracy_B`, `disagreement`, `oracle_headroom_pp`, `rescue_precision` per
predictor instance; `residual_*_vs_strength_null` = observed minus the
descriptive model's prediction; jackknife range of that residual.

## Baselines

The non-geometric predictors are the baseline/null itself (this is what the
descriptive model is fit on).

## Go / kill rule

A (excess complementarity): geometry's residual is positive and
jackknife-stable. B (explained by strength): geometry's residual is ~0 and
jackknife-stable, or clearly negative. C (inconclusive): residual sign/
magnitude is not jackknife-stable, or predictors don't span geometry's
accuracy range.

## Results

**The accuracy gap is total**: non-geometric predictors cluster at
accuracy_B 0.48-0.50; geometry (seeds 2-5) sits at 0.19-0.30, with zero data
points in between. `accuracy_B` standard deviation across the 25
non-geometric fitting rows is 0.0055 — essentially no variance to identify
that coefficient. See
`results/g3_controlled_complementarity/plots/accuracy_vs_headroom.png` (and
the other two required plots) for the visual gap.

**One in-range exception, and it is decisive for interpretation**: seed 1's
per-class/global-kNN geometry accuracy is ~0.48, landing inside the
non-geometric cluster (not an extrapolation). Its residual vs. the fitted
strength-null model is near zero (-0.12pp per-class, +0.27pp global-kNN) and
**jackknife-stable**: excluding any one of the 5 checkpoints from the fit
moves this residual by at most ±0.08pp.

**Seeds 2-5's positive residuals are extrapolation artifacts, not a stable
finding.** Naive fit: residuals of +0.84pp to +2.41pp (headroom) and +0.061
to +0.150 (rescue precision), all positive. Jackknife: residual_headroom_pp
for individual (family, seed) combinations ranges as wide as **-1.6pp to
+14.6pp** depending on which single checkpoint is excluded from the fit
(full ranges in
`results/g3_controlled_complementarity/controlled_complementarity_jackknife.json`).
This instability is the central result: a coefficient fit on ~0.0055 std of
accuracy variance, then extrapolated ~20 points down to geometry's accuracy
range, is not identified.

**Seed-1 mechanism (new, from cache, no retraining)**: checkpoint strength
is ruled out as the explanation — seed 1's checkpoint has the *lowest*
clean accuracy of the 5 (75.28% vs. 76.20-76.38%). Neighbourhood structure is
the measurable difference: mean `neighbourhood_concentration` = 0.914 and
mean `true_label_purity` = 0.477 for seed 1, vs. 0.117-0.162 /
0.059-0.104 for seeds 2-5 (5-8x gap), using the exact
`neighbour_statistics()` formula from `rgc_shift/recoverability.py` (see
correction note below). Identical for `perclass` and `globalknn` arms
because — new finding — **the two arms' `neighbour_labels`/`neighbours`
arrays are byte-identical** (`np.array_equal` confirmed on a sampled cell):
both query the same global kNN index over the same training bank; only the
vote/decision rule that turns those neighbours into `y_pred_knn` differs.
Treat per-class geometry and global kNN as correlated readouts of one
retrieval, not two independently-constructed geometric spaces.

**Seed-1 vs. seeds 2-5 breakdown (per-class geometry, required by
protocol)**:

| | accuracy_B | disagreement | headroom_pp | rescue_precision |
|---|---|---|---|---|
| all 5 seeds | 0.294 | — | 4.39 | 0.084 |
| seed 1 only | 0.479 | 0.156 | 2.35 | 0.151 |
| seeds 2-5 | 0.248 | 0.752 | 4.89 | 0.067 |

Seed 1 does not define the all-5 conclusion in the same direction one might
guess: despite much higher standalone accuracy, seed 1 shows *lower*
headroom than seeds 2-5 (it disagrees with the head far less often, 15.6%
vs. 75.2%, because it tracks the head closely) but *higher* rescue
precision (it is right more often when it does disagree, consistent with
being a stronger classifier).

## Protocol deviations

Not preregistered. The `full_vector_fusion` family was not part of the
original G3 plan — added during inspection as an available, no-retrain-
needed predictor.

## Interpretation

**C. INCONCLUSIVE — NO ADEQUATE MATCHED-STRENGTH CONTROL.**

Existing predictors do not span geometry's accuracy regime (0.19-0.30);
everything else available clusters at 0.48-0.50. The single in-range
comparison (seed 1) is fit almost exactly by the strength-null model and is
jackknife-stable — mildly supportive of "explained by strength" (state B) —
but it is one data point, confounded with seed 1's other known anomalies
(unusually concentrated/pure neighbourhoods; see
[[Seed 1 occupies a qualitatively different kNN regime]]), not a
generalizable test. The positive residuals for seeds 2-5 are extrapolation
artifacts (jackknife range up to 16pp wide) and are not usable as evidence
for state A.

**This does not overturn [[2026-09-15 G3 Headroom Null]]**: geometry still
clearly clears the no-information disagreement-matched null from that
experiment (that comparison did not require strength-matching, since the
null was constructed to match geometry's own disagreement rate directly).
What remains unresolved is specifically whether the *gap to an independent
head* reflects "geometry is a weaker predictor, nothing more" or "geometry
is weaker but the type of error information differs" — this experiment
could not distinguish those with existing artifacts.

## Correction to prior ResearchBrain entries (2026-09-15, same day)

While inspecting for predictor families, found `GeometricFullCalibration/
rgc-shift.zip` — a gitignored zip containing a separate, pinned package
`rgc_shift` (git `ff81032f...`) that `scripts/aggregate_recoverability.py`
imports directly (`from rgc_shift.recoverability import ...`) and that
`scripts/export_recoverability.py` also imports from. **This is the actual
gate-fitting code for the RGC Shift Recoverability gate — it is not
missing, contrary to what [[2026-09-15 G3 k_vote Audit]] and the addendum on
[[Validation-fitted neighbourhood reliability features fail under
corruption]] concluded.** It was missed because it lives in a separate,
gitignored (`*.zip`) package rather than the canonical repo's own git
history, which is why `git log -S "neighbourhood_concentration"` inside
`GeometricFullCalibration` found nothing — that string only exists in
`rgc_shift`'s own separate embedded git repo.

Confirmed from reading `rgc_shift/recoverability.py`:
`neighbourhood_concentration = counts.max(axis=1) / k` — matches the
independently-derived `vote_concentration` definition used in
[[2026-09-15 G3 k_vote Audit]] exactly. But `head_neighbour_agreement`'s
real definition, `(neighbour_labels == y_head[:, None]).mean(axis=1)` (a
continuous membership fraction), is different from that audit's
approximation (a binary majority-vote match) — so that audit's `agreement`
feature was a different, not just approximate, quantity. The `margin` and
`vote_concentration` findings in that audit stand (formulas confirmed
correct); the `agreement` finding should be treated as measuring a
different feature than the real gate used, not a close approximation of it.

`neighbour_statistics()` in `rgc_shift` takes `k` implicitly from the input
array's width, so a literal k=50 vs k=200 re-run of `head_neighbour_agreement`
and `neighbourhood_concentration` (2 of the 4 real gate features) using the
*exact* production formula is possible by slicing cached `neighbour_labels`
— not attempted in this pass, out of scope for this task, but a strictly
better version of the k_vote audit than what was run, if that thread is
picked up again.

**What is still true from the prior addendum**: the package is not
installed in any available conda environment on this machine (`import
rgc_shift` fails in `geo_cuda12`), so it cannot currently be run end-to-end
without unzipping and installing it — a real (lesser) reproducibility gap,
just not "code does not exist."

## What this does NOT establish

- Does not establish state A or B definitively — that is the entire point
  of landing on C.
- The `full_vector_fusion` family's near-zero disagreement with the head
  (8.4%) makes it a weak test of the strength-vs-type question even within
  its own accuracy band; it mostly confirms that a head-correlated
  predictor produces little headroom, not a new independent fact.
- Seed-1's neighbourhood-concentration finding is a measured correlate, not
  a proven cause, of its accuracy anomaly — the decisive decoupling test
  named in [[Seed 1 occupies a qualitatively different kNN regime]] (fixed
  checkpoint x multiple RGC randomisations; multiple checkpoints x fixed RGC
  config) still has not been run.
- n=5 checkpoints total. Every "jackknife-stable" or "jackknife-unstable"
  claim above is over 5 folds, not a large-sample guarantee.

## Next action

**Minimal additional experiment for a real answer** (not run, per
instructions): extract raw test-cell embeddings for the 12 corrupted cells
via a forward pass through the existing 5 checkpoints (new compute, not
retraining — the checkpoints already exist), then fit one deliberately-
weakened same-representation readout (e.g. a nearest-centroid classifier on
a coarse/shallow layer, or a linear probe with heavy regularisation) tuned
to land near 0.30-0.45 standalone accuracy. That gives one real in-range
non-geometric comparison point instead of a 20-point extrapolation, and
would let the go/kill rule above actually resolve A vs. B.

Lower priority: the seed-1 decoupling test (checkpoint x RGC-randomisation
factorial), and a literal k=50 re-run of the two gate features now confirmed
exactly recomputable via `rgc_shift.recoverability.neighbour_statistics`.

## Addendum 2026-09-15 (same day) — synthetic matched-strength null, partial resolution

This card's own "minimal additional experiment" (a same-representation
weakened readout, requiring a fresh forward pass) was not run. A different,
no-new-compute route was tried instead:
[[2026-09-15 G3 Synthetic Matched-Strength Null]] deliberately weakens
already-cached independent-head *logits* (temperature-scaled categorical
sampling and additive Gaussian noise) to build a matched-strength,
matched-disagreement null spanning geometry's accuracy regime, without
retraining or new embedding extraction.

**Result: B. GEOMETRY CONSISTENT WITH MATCHED SYNTHETIC NULL**, for the
primary seeds-2-5 aggregate (both arms): excess headroom and rescue
precision are within ~0.35 percentage points of zero, an order of magnitude
below the 3.80-4.89pp raw headroom this line of investigation started from.
Individual seeds 2, 3, 5 are similarly small and sign-mixed; seed 4 shows a
mildly larger (~1pp) positive excess, flagged but not resolved.

**This is a partial, not full, resolution of this card's own inconclusive
verdict.** It answers the *generic*-degradation version of the
strength-vs-type question (any predictor, matched on accuracy +
disagreement, already reproduces the headroom) but not the
*same-representation* version this card specifically asked for (a weakened
readout built from the same features/geometry, not from an unrelated
independent head's logits) — the experiment named in "Next action" above
remains the one path to that stronger test. This card's own INCONCLUSIVE
verdict is left as originally recorded, above, per the instruction to
append rather than rewrite; the new, more complete picture is: *generic*
strength-matching resolves toward B, *same-representation* strength-matching
remains untested.
