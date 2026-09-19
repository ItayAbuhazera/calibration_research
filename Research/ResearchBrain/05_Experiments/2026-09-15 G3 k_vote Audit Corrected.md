---
type: experiment
status: completed
date: 2026-09-15
project: rgc-shift
benchmark: CIFAR-100-C
model: ResNet-101
seeds: [1, 2, 3, 4, 5]
preregistered: false
supersedes: "[[2026-09-15 G3 k_vote Audit]]"
tags: [geometry, k_vote, saturation, gate, null-test, corrected]
---

# G3 k_vote Audit Corrected

## Question

Same question as the superseded [[2026-09-15 G3 k_vote Audit]]: does k=50
(pre-registered) discriminate rescue-worthy samples better than the
recorded k=200? This rerun uses the *exact* `rgc_shift` source (pinned
`ff81032f...`, unzipped from `GeometricFullCalibration/rgc-shift.zip`)
instead of an invented approximation, per the correction recorded in
[[2026-09-15 G3 Controlled Complementarity]].

## Exact source definitions (recorded verbatim, not inferred from names)

From `rgc_shift.recoverability.neighbour_statistics` (function body read in
full from the unzipped pinned commit):

```
head_neighbour_agreement    = (neighbour_labels == y_head[:, None]).mean(axis=1)
neighbourhood_concentration = counts.max(axis=1) / k
    where counts[i, c] = bincount of class c among the k nearest neighbours
    of i; k is taken implicitly from neighbour_labels.shape[1] -- there is
    no separate k argument.
true_label_purity           = (neighbour_labels == y_true[:, None]).mean(axis=1)
    (label-aware; the package's own docstring marks this "diagnosis only",
    not a legal gate feature.)
```

`neighbour_margin` and `knn_radius` are **not part of `rgc_shift` at all** —
confirmed by grepping the full unzipped package source for "margin" and
"radius": zero hits in `recoverability.py` (or anywhere else in the
package). They are computed by this repo's own
`GeometricFullCalibration/scripts/export_recoverability.py::_global_neighbour_fields`
(lines ~258-314), which is what actually produced the cached
`exports/seed{N}/{arm}/{cell}.npz` files used here and by the original
recoverability run:

```
neighbour_margin = (top1_count - top2_count) / K_VOTE
knn_radius       = distances[:, K_RADIUS - 1]
```

`K_VOTE` and `K_RADIUS` were both hardcoded to 200 in that script at the
time the exports were generated (`export_recoverability.py:50-51`) — this
is why the recorded run used k_vote=200 for both quantities despite the
pre-registration pinning k_vote=50, and confirms the two constants happened
to be equal in that run, not defined as always-equal.

**Correction to the previous ResearchBrain framing**: [[2026-09-15 G3
Controlled Complementarity]]'s correction note described all four features
as living in `rgc_shift`. That was imprecise — only `head_neighbour_agreement`
and `neighbourhood_concentration` come from `rgc_shift`; `neighbour_margin`
and `knn_radius` come from this repo's own export script. Recorded here as
the more precise account; not edited into that card's own history per the
append-don't-rewrite rule (cross-linked instead).

## Setup

New script `GeometricFullCalibration/scripts/g3_kvote_audit_corrected.py`
(the superseded `g3_kvote_audit.py` left untouched, per the reproducibility
instruction not to modify a script that generated previous results). Same
12 shift cells, 5 seeds, both arms (`perclass`, `globalknn`) as
[[2026-09-15 G3 Headroom Null]] and the superseded audit. Reads only cached
`neighbour_labels` (10000 x 200, pre-sorted nearest-first), `y_pred_head`,
`y_true`, `y_pred_knn`, `knn_radius` from the existing exports — no rerun of
any pipeline stage, no GPU inference.

`head_neighbour_agreement`, `neighbourhood_concentration`, `neighbour_margin`,
and `true_label_purity` are recomputed exactly at k in {50, 200} by slicing
`neighbour_labels[:, :k]` (exact, not an approximation — the cached array is
stored at the full width). `knn_radius` is **not** recomputed at k=50: only
a single scalar (distance to the 200th neighbour) is cached per sample, not
per-neighbour distances, so a k=50 radius is unavailable without a fresh
nearest-neighbour query against the raw embedding bank (out of scope; no
GPU inference or new embedding extraction was run). Per the task's
instruction to preserve k_radius=200, `knn_radius` is reported as a single,
k_vote-invariant feature — identical values and identical AUROC under both
conditions, by construction.

For the `globalknn` arm, the actual classifier `y_pred_knn` is itself a
k_vote-dependent majority vote with a distance-based tie-break that cannot
be exactly reconstructed at k=50 either (same missing-per-neighbour-distances
limit). The disagreement/decisiveness/label population for the AUROC
diagnostic (A1) is therefore held **fixed** at the existing k=200
`y_pred_knn` for both k_vote conditions — only the four candidate gate
*features* are recomputed at each k. This is an explicit scope choice
(continuity with the superseded audit's design), not an oversight. It does
not affect the `perclass` arm, whose `y_pred_knn` (a per-class-1NN-distance
argmin) never depended on k_vote.

## A0: descriptive statistics and saturation

Per (arm, seed, cell, k, feature): mean, std, min, p05, median, p95, max,
n_unique. Full grid (240 rows x 2 arms x 5 features (incl. knn_radius) —
`results/g3_kvote_audit_corrected/descriptive_stats.json`. Seed-level means
across the 12 cells (perclass and globalknn are numerically identical here,
confirming [[Seed 1 occupies a qualitatively different kNN regime]]'s
finding that both arms share one retrieval):

| seed | agreement mean (k50) | agreement mean (k200) | frac(agreement==1), k50 | frac(agreement==1), k200 |
|---|---|---|---|---|
| 1 | 0.8795 | 0.8841 | **0.5293** | **0.4178** |
| 2 | 0.1537 | 0.1059 | 0.0023 | 0.0000 |
| 3 | 0.1433 | 0.0988 | 0.0020 | 0.0000 |
| 4 | 0.0889 | 0.0629 | 0.0003 | 0.0000 |
| 5 | 0.1694 | 0.1163 | 0.0030 | 0.0000 |

`neighbourhood_concentration` and `neighbour_margin` saturation
(`frac(==1)`) track `head_neighbour_agreement`'s almost exactly for seed 1
(0.5300/0.4179 vs 0.5293/0.4178) and are likewise ~0 for seeds 2-5 — full
table in `descriptive_stats.json`. This is a substantially sharper,
quantified version of the seed-1 anomaly than previously available: with
the *correct* continuous formula, roughly **half of seed 1's samples sit at
the absolute maximum** of all three features simultaneously at k=50, a
regime seeds 2-5 essentially never reach.

`knn_radius` has no natural "==1" saturation (it is a raw distance, not a
bounded [0,1] statistic); per-seed means range widely (0.39-0.81 depending
on seed/corruption mix — see `descriptive_stats.json`), invariant across
k_vote by construction.

## Spearman(feature_k50, feature_k200)

Mean rho across all (seed, cell) pairs, identical for both arms (shared
retrieval):

| feature | mean rho | min | max |
|---|---|---|---|
| head_neighbour_agreement | 0.918 | 0.821 | 0.965 |
| true_label_purity | 0.921 | 0.800 | 0.975 |
| neighbourhood_concentration | 0.867 | 0.522 | 0.957 |
| neighbour_margin | **0.680** | **0.223** | 0.954 |

`head_neighbour_agreement`/`true_label_purity` rankings are largely
preserved between k=50 and k=200 (both are simple neighbour-averages,
robust to window size). `neighbour_margin`'s ranking is the most sensitive
to k_vote of the three recomputable features — on at least one (seed, cell)
its k=50 and k=200 rankings are only weakly correlated (rho=0.22).

## A1: label-aware AUROC diagnostic

Same target/label definition as the superseded audit (decisive
disagreements between head and the fixed, recorded k=200 `y_pred_knn`;
G=1 if geometry right & head wrong, G=0 if head right & geometry wrong,
both-wrong excluded). Full raw table:
`results/g3_kvote_audit_corrected/auroc_diagnostic_raw.json`; aggregate:
`auroc_diagnostic_aggregate.json`.

| arm | feature | k | all 5 seeds | seeds 2-5 | seed 1 |
|---|---|---|---|---|---|
| perclass | head_neighbour_agreement | 50 | 0.4684 | 0.4751 | 0.4417 |
| perclass | head_neighbour_agreement | 200 | 0.4871 | 0.4956 | 0.4530 |
| perclass | neighbourhood_concentration | 50 | 0.5751 | 0.6009 | 0.4720 |
| perclass | neighbourhood_concentration | 200 | 0.5691 | 0.5941 | 0.4692 |
| perclass | neighbour_margin | 50 | 0.5482 | 0.5690 | 0.4652 |
| perclass | neighbour_margin | 200 | 0.5439 | 0.5637 | 0.4649 |
| perclass | knn_radius (invariant) | 50/200 | 0.4941 | 0.4928 | 0.4990 |
| globalknn | head_neighbour_agreement | 50 | 0.5059 | 0.5132 | 0.4765 |
| globalknn | head_neighbour_agreement | 200 | 0.5433 | 0.5497 | 0.5177 |
| globalknn | neighbourhood_concentration | 50 | 0.6286 | 0.6464 | 0.5576 |
| globalknn | neighbourhood_concentration | 200 | 0.6135 | 0.6370 | 0.5192 |
| globalknn | neighbour_margin | 50 | 0.6076 | 0.6207 | 0.5555 |
| globalknn | neighbour_margin | 200 | 0.5928 | 0.6125 | 0.5140 |
| globalknn | knn_radius (invariant) | 50/200 | 0.4653 | 0.4728 | 0.4353 |

**k_vote CHANGE MINOR.**

- `neighbourhood_concentration` and `neighbour_margin` reproduce the
  superseded audit's direction: small AUROC *gains* at k=50 (seeds 2-5:
  perclass +0.7 to +2.7pt, globalknn +0.9 to +1.9pt).
- `head_neighbour_agreement`, now computed with the *real* continuous-
  membership formula rather than the superseded audit's binary
  approximation, shows the **opposite** direction and is now the largest
  single delta in the table: k=50 is *worse* than k=200 by 2.05pt
  (perclass, seeds 2-5) to 3.65pt (globalknn, seeds 2-5). The superseded
  audit's `agreement` feature also went slightly negative at k=50, but by
  under 1pt — this is a materially larger, more confident negative result
  for the real feature, not a reproduction of the old one.
- `knn_radius` is invariant by construction (k_radius preserved at 200) and
  is close to chance on its own (0.44-0.50) regardless of k_vote — it would
  not have driven the original gate's near-zero capture rate either way.
- No feature reaches strong discrimination (max 0.647, globalknn
  neighbourhood_concentration at k=50, seeds 2-5) at either k.
- Net effect of switching k_vote 200 -> 50: two of three recomputable
  features gain 1-3 AUROC points, one loses 2-4 points, none change the
  qualitative picture (all weak-to-moderate univariate discrimination).
  This does not overturn the superseded audit's practical conclusion ("k
  does not look like the main lever") but corrects the sign and roughly
  doubles the confidence on the `head_neighbour_agreement` result
  specifically, since it is no longer measuring an invented proxy feature.

## What this does NOT establish

- Still univariate discrimination of individual features, not a refitted
  multivariate gate (same limitation as the superseded audit).
- `knn_radius` at k=50 remains entirely unaudited — its cross-k behavior is
  unknown; only its invariant k=200 value is characterized here.
- The `globalknn` arm's disagreement/decisiveness population is fixed at
  the k=200 classifier for both conditions (see Setup) — this audits
  candidate gate *features* at k=50, not "what a true from-scratch k=50
  globalknn classifier would predict."
- n=5 seeds, one benchmark family. Seed 1's numbers in every table above
  are the same anomalous seed characterized in
  [[Seed 1 occupies a qualitatively different kNN regime]] — do not let it
  define the "seeds 2-5" or "all 5" conclusions (kept as separate columns
  throughout for exactly this reason).

## Artifacts

- `GeometricFullCalibration/scripts/g3_kvote_audit_corrected.py` (new,
  additive; superseded `g3_kvote_audit.py` untouched).
- `GeometricFullCalibration/results/g3_kvote_audit_corrected/{descriptive_stats.json,
  spearman.json, spearman_summary.json, auroc_diagnostic_raw.json,
  auroc_diagnostic_aggregate.json}`.
- Source of truth: `GeometricFullCalibration/rgc-shift.zip` (pinned
  `ff81032f...`, unzipped read-only to a scratch path for import — the zip
  itself was not modified), `GeometricFullCalibration/scripts/export_recoverability.py`
  lines 50-51 (`K_VOTE`/`K_RADIUS` constants) and ~258-314
  (`_global_neighbour_fields`).

## Next action

- A real k=50 `knn_radius` and a from-scratch k=50 `globalknn` classifier
  both require a fresh nearest-neighbour query against the raw embedding
  bank (not cached) — out of scope here (no GPU inference), but now a
  well-scoped, concrete next step if that thread is picked up.
- Given the small, mixed effect sizes here (same conclusion as the
  superseded audit), priority remains on the strength-vs-type question — see
  [[2026-09-15 G3 Synthetic Matched-Strength Null]].
