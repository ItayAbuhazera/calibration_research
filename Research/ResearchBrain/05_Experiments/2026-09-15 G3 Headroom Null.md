---
type: experiment
status: completed
date: 2026-09-15
project: rgc-shift
benchmark: CIFAR-100-C
model: ResNet-101
seeds: [1, 2, 3, 4, 5]
preregistered: false
tags: [geometry, ood, recoverability, null-test, oracle-headroom]
---

# G3 Headroom Null

## Question

Is the oracle-union headroom reported for per-class geometry (~4.385 pp) and
global kNN (~3.798 pp) in [[2026-09-15 RGC Shift Recoverability]] evidence of
real, geometry-specific complementary information, or is it substantially
explained by the fact that *any* second predictor that disagrees with the head
inflates oracle-union accuracy by construction?

This directly addresses the integrity-queue item "Oracle-union null baseline"
in the Research Dashboard, which blocked
[[Geometry contains complementary accuracy information under corruption]] and
[[Per-class geometry exposes more oracle headroom than global kNN]].

## Hypothesis tested

H0: Geometry's oracle-union headroom is no larger than what a no-information
predictor, disagreement-matched to geometry's own disagreement rate, would
produce.

## Setup

Recomputed directly from cached per-sample exports — **no rerun of the
export pipeline**. Source: `GeometricFullCalibration/exports/seed{1-5}/
{perclass,globalknn}/{cell}.npz`, 12 shift cells (4 corruption types ×
severities {1,3,5}; clean/val excluded — this is a shift-headroom question),
5 seeds, N=10000 per cell.

Script: `GeometricFullCalibration/scripts/g3_headroom_null.py` (additive, new
file). Raw per-(seed,cell,arm) output:
`GeometricFullCalibration/results/g3_headroom_null/g3_headroom_null_raw.json`.
Aggregate: `.../g3_headroom_null_aggregate.json`.

Four arms compared against the same primary head, per seed/cell:

1. **per_class_geometry** — `y_pred_knn` from the `perclass` export (the
   original arm).
2. **global_knn** — `y_pred_knn` from the `globalknn` export (the original
   arm).
3. **independent_head** — a second, independently-trained ResNet-101
   checkpoint (verified by distinct SHA-256 hashes in `export_recoverability.py`,
   not a shared backbone with different downstream fitting). Pooled over the
   other 4 seeds' heads on the same fixed eval samples (confirmed `y_true`
   identical across seeds per cell).
4. **matched_stochastic_vs_{perclass,globalknn}** — the primary head's own
   prediction with a uniformly-random *other*-class flip injected at a rate
   matched to the corresponding real arm's observed disagreement rate on that
   exact (seed, cell). Deterministic RNG seeded by `sha256(seed|cell|tag)`.
   Zero real information by construction — this is the null itself.

## Fixed choices

- **Metric definitions are new** (not previously defined in this codebase or
  in the ResearchBrain vault):
  - `disagreement_rate = P(head_pred != B_pred)`
  - `oracle_headroom_pp = P(head wrong AND B right) * 100`
    (equals `accuracy(oracle-union(head,B)) - accuracy(head)`)
  - `rescue_precision = P(head wrong AND B right | head_pred != B_pred)`
    = `oracle_headroom_pp / (100 * disagreement_rate)` at the per-cell level.
- Aggregation: per-seed mean across the 12 cells, then mean ± 1.96·SEM across
  the 5 seed-level means (normal approx, same style as the original card's
  paired-seed CIs). **Caveat**: the aggregate table's `rescue_precision` is
  the mean of per-cell ratios, not the ratio of the aggregated
  `oracle_headroom_pp`/`disagreement_rate` columns — these are not
  arithmetically consistent with each other by construction (Jensen's
  inequality across cells with different disagreement rates). Do not
  cross-check one column against the other two.

## Primary metrics

`disagreement_rate`, `oracle_headroom_pp`, `rescue_precision`, each as
mean ± 95% CI across 5 seeds.

## Baselines

`matched_stochastic_vs_perclass` / `matched_stochastic_vs_globalknn` (the
null itself); `independent_head` (a second, stronger comparison point — not
originally requested as a "baseline" but included as an arm).

## Go / kill rule

Go (headroom is more than construction-by-disagreement): geometry's headroom
clears the matched-stochastic null with a non-overlapping CI.
Kill (headroom is fully explained by disagreement alone): geometry's headroom
is within the matched-stochastic null's CI.

## Results

Sanity check: recomputed `per_class_geometry` oracle_headroom_pp = 4.3853,
`global_knn` = 3.7982 — matches the original experiment card's reported
4.385 pp / 3.798 pp almost exactly, confirming this recomputation reproduces
the published aggregate.

| Arm | disagreement_rate | oracle_headroom_pp | rescue_precision |
|---|---|---|---|
| per_class_geometry | 0.6228 | 4.3853 | 0.0896 |
| global_knn | 0.5799 | 3.7982 | 0.0857 |
| independent_head (pooled, other 4 seeds) | 0.4337 | 8.8877 | 0.2231 |
| matched_stochastic_vs_perclass | 0.6229 | 0.3387 | 0.0053 |
| matched_stochastic_vs_globalknn | 0.5797 | 0.3050 | 0.0053 |

Per-seed values and CIs are in `g3_headroom_null_aggregate.json`.

**Paired seed-level CIs (independent_head minus geometry arm, 5 seeds, normal
approx):**

| Comparison | oracle_headroom_pp | rescue_precision |
|---|---|---|
| independent_head − per_class_geometry | +4.50, 95% CI [3.31, 5.69] | +0.134, 95% CI [0.095, 0.172] |
| independent_head − global_knn | +5.09, 95% CI [3.54, 6.64] | +0.137, 95% CI [0.102, 0.173] |

Both CIs exclude zero — the independent-head advantage is not just a point-
estimate artifact.

**Mechanism check — standalone accuracy, not a geometry-specific rescue
effect.** Mean standalone accuracy across the 12 cells and 5 seeds: head
0.487, per_class geometry 0.294, global_knn 0.301, other-seed heads (pooled)
0.487 — i.e. the "independent head" arm is, on average, exactly as accurate
as the primary head, while geometry is a much weaker standalone classifier.
This plausibly explains most of the rescue-precision gap without needing any
geometry-specific mechanism: a stronger standalone predictor wins more of
its disagreements with the head, independent of what kind of predictor it
is. Per-seed breakdown, and a striking exception, in
[[Seed 1 occupies a qualitatively different kNN regime]]: seed 1's per-class
geometry accuracy (0.4787) is nearly identical to its head accuracy (0.4817)
— unlike seeds 2-5, where geometry accuracy is 30-60pp below the head. This
is new, precise, corroborating evidence for that previously weak (n=1),
qualitative note.

## Protocol deviations

None from a preregistration — this experiment was not preregistered
(`preregistered: false`); it was scoped and run in a single session in
response to the integrity-queue item.

## Interpretation

The go/kill rule resolves cleanly: geometry's oracle-union headroom
(3.80–4.39 pp) is roughly an order of magnitude above the no-information,
disagreement-matched null (0.31–0.34 pp). **The headroom is not simply an
artifact of "any disagreeing predictor inflates oracle-union accuracy."**
[[Geometry contains complementary accuracy information under corruption]]
and [[Per-class geometry exposes more oracle headroom than global kNN]] can
have their `blocked_by: missing-oracle-union-null` flag removed on this
basis.

**However, the independent-head arm reframes what this means.** A second,
ordinary, independently-trained ResNet-101 — with a *lower* disagreement rate
than either geometry arm (0.434 vs 0.58–0.62) — produces roughly double the
oracle headroom (8.89 pp) and roughly double the rescue precision (0.223 vs
0.086–0.090) of either geometry arm. Geometry clears the "no information"
bar but does not clear the "an ordinary second classifier" bar — on this
evidence, plugging in a second independently-trained head recovers more
oracle-accessible complementary information than plugging in geometry does.

## What this does NOT establish

- This does not establish that geometry carries *no* useful information
  beyond a second head — only that, on this oracle-union metric, a second
  head currently does better. Geometry may still offer information a second
  head does not that isn't visible in this aggregate (e.g. concentrated in
  different corruption families/severities — not tested here).
- n=5 seeds, one architecture, one benchmark family (CIFAR-100-C). Do not
  generalize beyond ResNet-101/CIFAR-100-C without replication.
- `rescue_precision` and `disagreement_rate` are new operational metrics
  introduced in this experiment, not independently validated definitions —
  treat comparisons across arms as internally consistent, not as an
  externally-benchmarked metric.
- The independent-head result does not, by itself, mean an ensemble is a
  *better calibration method* than geometry — oracle-union headroom is an
  upper bound assuming perfect knowledge of which predictor is right per
  sample; no gate/selector was tested for the independent-head arm (that
  would be a natural next step, given the original gate's poor capture rate
  for the geometry arms).
- The standalone-accuracy mechanism check is a correlational explanation,
  not a controlled one — accuracy and predictor type are confounded here
  (geometry is both "weaker" and "a different kind of predictor" at once).
  No experiment varied standalone accuracy while holding predictor type
  fixed, or vice versa, so "the gap is just accuracy" is the leading
  hypothesis, not a proven decomposition.

## Next action

- Paired seed-level CIs for `independent_head − {per_class_geometry,
  global_knn}` are now computed (see Results) and exclude zero for both
  oracle_headroom_pp and rescue_precision — the ranking is established at
  n=5 seeds.
- Test the standalone-accuracy hypothesis directly: does rescue precision
  correlate with standalone accuracy across a wider set of "B" predictors
  (e.g. weaker/stronger independent heads, geometry at different k), holding
  predictor type fixed? Would separate "stronger classifier wins
  disagreements" from anything geometry-specific. **Done, inconclusive**:
  see [[2026-09-15 G3 Controlled Complementarity]] — no existing predictor
  spans geometry's accuracy range, so the strength explanation remains
  correlational, neither confirmed nor refuted.
- Run the seed-1 decoupling test named in
  [[Seed 1 occupies a qualitatively different kNN regime]] (fixed checkpoint
  × multiple RGC randomisations; multiple checkpoints × fixed RGC config) —
  now higher priority given the precise accuracy match found here.
- Consider whether an independent-head-vs-geometry comparison changes the
  priority of [[H-RGC-01 Reliability mapping shifts under corruption]] and
  [[H-RGC-02 Richer representation features predict geometric reliability]].
- G3 k_vote audit (recorded k_vote=200 vs pre-registered k_vote=50) is
  **blocked**: the gate-fitting code that produced
  `results/seed{N}/recoverability_{perclass,globalknn}.json`'s `gate` block
  (features including `neighbourhood_concentration`, threshold,
  `n_train_decisive`) does not exist anywhere in the canonical
  `GeometricFullCalibration` repo — not in `export_recoverability.py`, not in
  `aggregate_recoverability.py`, and `git log --all -S "neighbourhood_concentration"`
  returns zero commits. See addendum in
  [[Validation-fitted neighbourhood reliability features fail under corruption]].
