---
type: failure_mode
status: closed_historical
evidence_scope: unpublished_repository_analysis
project: geometric-separation
tags: [ece, metric-bug, platt, binning]
---

# Historical geometric calibration ECE exports contain invalid perfect scores

## Failure

A subset of the `GeometricCalibration` repository's historical ECE exports
contain implausible exactly-zero ECE values, traceable to two interacting
issues in the metric implementation, plus separately-verified seed-count
shortfalls in the same result family.

## Evidence and verdicts

### 1. ECE binning bug at confidence == 1.0 — verified implementation issue

`utils/metrics.py`, `CalibrationMetrics.ece()` (shared by `.mce()`, `.ace()`,
`.binned_likelihood()`): computes
`bin_indices = np.digitize(confidence, bin_boundaries) - 1` with
`bin_boundaries = np.linspace(0, 1, n_bins + 1)`. `np.digitize` (default
`right=False`) maps any confidence `>= 1.0` to index `n_bins`, one past the
last valid bin index (`0..n_bins-1`). The aggregation loop
`for i in range(n_bins)` never visits that index, so points with confidence
exactly 1.0 contribute zero calibration error while still counting toward the
denominator — silently deflating ECE whenever any prediction confidence
saturates to exactly 1.0. This is the production ECE path
(`utils/calibration.py`, `scripts/main_layer.py`, `n_bins=15` or `20`).

### 2. Platt zero-ECE artifact — suspicious artifact / likely mechanism

`calibrators/calibrators.py::PlattCalibrator.fit` (L292–321) trains an
unregularized per-class logistic regression on a single scalar feature
(`class_scores.reshape(-1,1)`). When that feature already near-perfectly
separates the binary correct/incorrect label — plausible for a
high-confidence source classifier — sklearn's LR can produce large
coefficients that saturate `predict_proba` to exactly `1.0` for the most
confident points. Combined with bug #1, those points are silently dropped
from every ECE bin while still counting in `n`, driving computed ECE toward
0. Observed: Platt reports exactly `0.0` for **all five** ECE columns
(clean + 3 corruption types + combined) in 224/237 matched rows across
CIFAR-10/CIFAR-100 (100% for 4 of the 4 dataset/model combinations tested;
near-zero for a few GTSRB rows). The claim that bug #1 fully explains bug #2
is a plausible mechanism supported by code inspection, not a proven causal
chain (per source-precedence discipline: do not assert causality beyond what
is shown).

### 3. Random-state / seed count discrepancies — verified implementation issue

`statistical_analysis_by_config/statistical_summary_*_RS103-120.csv`,
`N_Random_States` column. Filenames imply a nominal RS103–120 range (18
seeds), but actual seed counts per dataset/model are only 10–11 — already
short of nominal. Specific technique/layer/binning cells are far shorter than
the rest of their own table: `geometric_semantic_binned_avgpool` at
`layer_avgpool`/binned in the GTSRB/EfficientNet summary has `N=1` against
`N=10` everywhere else in that file; several CIFAR-100/EfficientNet cells at
`layer_avgpool`+unbinned drop to `N=5–7` against `N=10–11` elsewhere in the
same file.

## Why it matters

Any historical ECE number attributed to Platt in this repository, and any row
built from the specific undercounted technique/layer/binning cells above,
should be treated as unreliable until re-derived with a corrected metric
implementation and adequate seeds. This does not invalidate the
`geometric_physical`/`geometric_semantic` comparisons elsewhere in this
project, which do not depend on the confidence==1.0 edge case in the same way
(their outputs are not observed to be saturated at exactly 1.0/0.0 in the same
pattern).

## Related project

[[Geometric Separation]]

## Related

- [[Geometric separation substantially improves clean CIFAR top-label ECE]]
- [[Clean-fitted geometric confidence mappings can reverse under synthetic corruption]]
