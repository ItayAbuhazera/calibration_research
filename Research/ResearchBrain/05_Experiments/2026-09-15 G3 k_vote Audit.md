---
type: experiment
status: superseded
superseded_by: "[[2026-09-15 G3 k_vote Audit Corrected]]"
date: 2026-09-15
project: rgc-shift
benchmark: CIFAR-100-C
model: ResNet-101
seeds: [1, 2, 3, 4, 5]
preregistered: false
tags: [geometry, k_vote, saturation, gate, null-test, superseded]
---

> **SUPERSEDED 2026-09-15 (same day) — INCORRECT FEATURE DEFINITION.**
> This card's `agreement` feature is a self-invented binary majority-vote
> approximation, built because the real `rgc_shift` package was (wrongly)
> believed missing from the repo at the time. It is not missing (see the
> correction already recorded below and in
> [[2026-09-15 G3 Controlled Complementarity]]). The real
> `head_neighbour_agreement` in `rgc_shift.recoverability.neighbour_statistics`
> is a continuous neighbour-membership fraction, not a binary vote match — a
> different quantity, not an approximation of it. `margin` and
> `vote_concentration` in this card used formulas later confirmed correct
> against the real source, but were audited only qualitatively (no
> descriptive stats, no saturation rates, no Spearman(k50,k200)). Preserved
> below for history — do not delete or edit its measured numbers. The
> corrected, exact-formula audit with full descriptive statistics is
> [[2026-09-15 G3 k_vote Audit Corrected]].

# G3 k_vote Audit

## Question

The recorded RGC Shift Recoverability run used `k_vote=200` where the
pre-registration pinned `k_vote=50`; [[Validation-fitted neighbourhood
reliability features fail under corruption]] flagged this as possibly
having washed out the local signal two of the four gate features needed.
Does k=50 actually give the underlying vote-based features more
discriminative power for identifying rescue-worthy samples than k=200?

**This is explicitly NOT a re-run of the original gate.** At the time this
was run, the gate-fitting code was believed missing from the canonical
repo. **Correction (2026-09-15, same day, see
[[2026-09-15 G3 Controlled Complementarity]])**: it is not missing — it
lives in `GeometricFullCalibration/rgc-shift.zip`, a gitignored package
`rgc_shift` that the canonical scripts import directly; the earlier search
only checked `GeometricFullCalibration`'s own git history, which does not
contain `rgc_shift`'s code. The real gate-fitting formulas are now known:
`neighbourhood_concentration = counts.max(axis=1)/k` matches this audit's
`vote_concentration` exactly, but the real `head_neighbour_agreement` is a
continuous neighbour-membership fraction, different from the binary
majority-vote-match `agreement` feature computed below — so the `agreement`
result in this card measures a different quantity than the real gate used,
not an approximation of it. The `margin` and `vote_concentration` results
are unaffected.

## Hypothesis tested

H0: k=50 gives the same or worse univariate discrimination of "geometry is
right" (among head/geometry disagreement cases) as k=200 — i.e. the
k_vote=200 deviation does not, by itself, explain the gate's failure.

## Setup

`GeometricFullCalibration/scripts/g3_kvote_audit.py` (additive, new file),
reading cached `neighbour_labels` (10000 x 200, pre-sorted nearest-first) from
`exports/seed{1-5}/{perclass,globalknn}/{cell}.npz`, same 12 shift cells as
[[2026-09-15 G3 Headroom Null]]. Raw:
`results/g3_headroom_null/g3_kvote_audit_raw.json`. Aggregate:
`.../g3_kvote_audit_aggregate.json`.

Three features recomputed at k in {50, 200} by slicing
`neighbour_labels[:, :k]` and recounting (no rerun of the kNN query):

- `agreement` — head prediction matches the k-neighbour majority vote
  (majority via plain argmax; the original pipeline's distance-based
  tie-break could not be replicated — raw per-neighbour distances are not
  cached, only a single scalar `knn_radius` at k=200 — so this is an
  approximation of the real `head_neighbour_agreement` feature).
- `margin` — (top1_count − top2_count) / k. Matches the confirmed formula in
  `export_recoverability.py:_global_neighbour_fields`.
- `vote_concentration` — top1_count / k. **New operational definition** —
  the original `neighbourhood_concentration` formula is unrecoverable (zero
  hits in full git history), so this is not a reproduction.

`knn_radius` (the fourth original gate feature) is **excluded**: only the
scalar distance to the 200th neighbour is cached, not per-neighbour
distances, so radius at rank 50 cannot be recomputed without loading the raw
embedding bank (out of scope for this pass).

Target/label: among samples where head and the existing k=200 geometry
prediction (`y_pred_knn`) disagree, label = 1 if geometry is right. Metric:
AUROC of each feature for this label, computed per (seed, cell, arm, k),
aggregated per-seed (n_disagree-weighted mean across the 12 cells), then
mean ± 95% CI (normal approx) across 5 seeds — same style as
[[2026-09-15 G3 Headroom Null]].

## Fixed choices

Same 12 corruption cells, 5 seeds, both geometry arms (perclass, globalknn)
as the headroom-null experiment.

## Primary metrics

AUROC(feature, k) and the paired delta AUROC(k=50) − AUROC(k=200), per
feature per arm, with 95% CI across 5 seeds.

## Baselines

k=200 itself is the baseline/comparison point (the recorded, deviated
configuration).

## Go / kill rule

Go (k_vote=200 plausibly explains part of the gate failure): k=50 shows a
CI-supported AUROC improvement over k=200 for the local/agreement-type
features specifically named in the original concern.
Kill (k_vote is not the explanation): no consistent, CI-supported
improvement at k=50.

## Results

| arm | feature | AUROC k=50 | AUROC k=200 | delta (50−200) | 95% CI on delta |
|---|---|---|---|---|---|
| perclass | agreement | 0.489 | 0.494 | −0.0044 | [−0.0054, −0.0034] |
| perclass | margin | 0.574 | 0.566 | +0.0075 | [−0.0011, +0.0160] |
| perclass | vote_concentration | 0.612 | 0.604 | +0.0078 | [+0.0009, +0.0146] |
| globalknn | agreement | 0.492 | 0.500 | −0.0085 | [−0.0135, −0.0036] |
| globalknn | margin | 0.615 | 0.600 | +0.0152 | [+0.0085, +0.0218] |
| globalknn | vote_concentration | 0.659 | 0.645 | +0.0143 | [+0.0084, +0.0202] |

## Protocol deviations

Not preregistered. `agreement` uses an approximate tie-break (plain argmax,
not the original distance-based tie-break) since raw per-neighbour distances
are not cached. `vote_concentration` is a new definition, not a
reproduction of the lost `neighbourhood_concentration`.

## Interpretation

Mixed, small-magnitude result — **does not strongly vindicate or refute the
k_vote=200 concern.** `margin` and `vote_concentration` (2 of 3 recomputable
features) show a small, mostly CI-supported improvement at k=50 (up to
+0.015 AUROC for globalknn), consistent with the "averaging over 200
neighbours washed out local signal" hypothesis — but the effect size is
tiny (1-2 AUROC points). `agreement` gets very slightly *worse* at k=50 for
both arms (CI excludes zero, negative), the opposite direction from the
concern. Absolute AUROC values are weak-to-moderate even at the more
favorable k=50 (0.49-0.66) — none of these three features, alone, strongly
discriminates rescue-worthy samples at either k. The kill in
[[Validation-fitted neighbourhood reliability features fail under
corruption]] is not resolved by this: even at k=50, no single feature here
looks like it would have rescued the gate on its own.

## What this does NOT establish

- This is univariate discrimination of individual features, not a
  refitted multivariate gate — the original gate combined four features
  (including `knn_radius`, excluded here) with a fitted threshold. A
  reimplemented multivariate gate at k=50 might behave differently from
  what univariate AUROC suggests.
- `knn_radius` — one of the four original features, and the one most
  directly about neighbourhood extent — could not be audited at all.
- **Superseded**: the `agreement` feature here is not an approximation of
  the real `head_neighbour_agreement` — it is a different feature (binary
  majority-vote match vs. the real continuous membership fraction). Its
  result should not be used to reason about the real gate feature at all
  (see the correction note above). `margin` and `vote_concentration` remain
  valid, formula-confirmed results.
- n=5 seeds, one benchmark family, two geometry arms. Do not generalize the
  "k barely matters" conclusion beyond this scope.

## Next action

- If a multivariate gate refit at k=50 is wanted, it needs to be built as an
  explicitly new implementation (logistic regression or similar over the 3
  recomputable features, or 4 if `knn_radius` at k=50 is made available by
  loading the raw embedding bank), not treated as a reproduction of the lost
  gate.
- Given the small effect sizes here, prioritize the independent-head
  mechanism question from [[2026-09-15 G3 Headroom Null]] over further
  k_vote work — k does not look like the main lever.
