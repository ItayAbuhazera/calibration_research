---
type: project
status: completed
canonical_repo: ../geometric/GeometricInternalCalibration/GeometricInternalCalibration
parent_project: "[[Geometric Separation]]"
paper: "[[Semantic Geometric Calibration in Randomized Neural Feature Space]]"
tags: [rgc, rgcl, rgcc, geometry, calibration, ood]
---

# Semantic Geometric Calibration (RGC)

## Research question

Is careful, architecture-specific layer selection actually necessary for
effective representation-space geometric calibration, or can randomized
sampling replace it without a statistically significant loss? (Paper framing —
see [[Semantic Geometric Calibration in Randomized Neural Feature Space]].)

## Core method

RGC pipeline: randomized intermediate-representation sampling → compact
normalized semantic embedding → nearest-neighbour separation score → rank
normalization → isotonic correctness mapping.

- **RGCL**: uniformly samples intermediate *layers*; applies spatial pyramid
  pooling per sampled layer; random-projects to a fixed dimension; sums and
  L2-normalizes.
- **RGCC**: uniformly samples activation *coordinates* directly from the global
  concatenated-activation space; no SPP or per-layer projection; lower
  preprocessing cost.
- Default paper settings: `L = 6` layers, `d = K = 256` embedding
  dimension/coordinate count (see paper note — not re-derived here).

## Published results

See [[Semantic Geometric Calibration in Randomized Neural Feature Space]] for
the full, paper-sourced account: ~78% average ECE reduction vs. uncalibrated
(up to ~97% on CIFAR-100/DINOv2-Large), RGCL/RGCC equivalence to layer-selected
baselines (DAC/TULIP) in 10/13 combinations, RGCL-vs-RGCC equivalence in 9/13
combinations, and the offline-preprocessing-vs-online-throughput trade-off
(RGCL priciest offline, RGCC cheaper offline, DAC fastest online, RGCL/RGCC
near-identical online).

## Published limitations

- In-distribution image classification only.
- Corruption, domain shift, and non-vision modalities are explicitly named as
  future work, not evaluated.
- The geometric index must be rebuilt/updated whenever calibration data
  changes.

## Repository-only findings

Everything below is from the `GeometricInternalCalibration` repository only,
not the published paper. Labeled per finding as
`verified implementation issue`, `suspicious artifact / likely mechanism`,
`under-replicated`, or `insufficient evidence`.

### RGCL/RGCC equivalence: metric- and regime-dependent

The canonical TOST script (`Experiments/generate_rgcc_rgcl_tost_table.py`,
default `--absolute-margin 0.005`, i.e. the same δ=0.5pp as the paper)
produced **three different result tables in three sibling output
directories**, with different sample compositions and different conclusions:

- `calibration_comparison_mce_adaptive/tost_rgcc_rgcl/tost_rgcc_vs_rgcl_ece.csv`
  (13 rows, CIFAR-100 set = densenet121/dinov2_large/resnet101/18/50 — **no
  resnet152**): 9 Equivalent, 1 Inconclusive (dinov2_large), 3 "RGCC lower"
  (all three Tiny-ImageNet rows). **CIFAR-100/densenet121 is Equivalent here**
  — this directly conflicts with the published paper's own text, which lists
  CIFAR-100/densenet121 among the non-equivalent combinations.
- `calibration_comparison_separate/tost_rgcc_rgcl/tost_rgcc_vs_rgcl_ece.csv`
  (14 rows — both dinov2_large *and* CIFAR-100/resnet152 present; different
  per-row `n`, e.g. CIFAR-10/densenet121 n=18 here vs. n=25 in the file
  above): 9 Equivalent / 4 Inconclusive / 1 RGCC-lower. Same headline count
  of "9," different denominator (14, not 13) and a different failing set.
- `calibration_comparison_new/tost_rgcc_rgcl/tost_rgcc_vs_rgcl_ece.csv`
  (only 10 rows, **no Tiny-ImageNet rows at all**): 7 Equivalent / 3
  Inconclusive.
- Separately, `latex_tables_statistics_n/table_sgc_vs_faiss_tost_{10,20,30}pct.tex`
  runs the same equivalence-testing machinery with a **relative** margin
  (10/20/30% of the mean, floor 0.1pp) instead of the paper's fixed δ=0.5pp —
  but for a **different pairing** (SGC vs. an SGC-FAISS backend, not
  RGCL-vs-RGCC): "Equivalence established: 0/3" at every margin tested. Do not
  conflate this with the RGCL-vs-RGCC comparison above.

**Conflict recorded, not resolved:** the repository does not have one
canonical "9 of 13" table; it has (at least) three different row sets under
the same nominal margin, one of which contradicts the paper's own stated
exception list. Status: `verified implementation issue / under-replicated` —
this is a genuine analysis-composition discrepancy, not a margin discrepancy
as such (the margin itself is mostly stable at δ=0.5pp across the RGCL-vs-RGCC
tables).

Full note: [[RGCL-RGCC equivalence is metric- and regime-dependent]]

### SPP changes the offline/online cost trade-off, not "free compression"

`timing_analysis/tableA_offline_online.csv` / `tableB_offline_breakdown.csv`:
RGCL (SPP + random projection) offline feature-extraction time is 16×–44× that
of RGCC across configurations (e.g. CIFAR-10/ResNet-18: 461.9s vs. 16.9s;
SVHN/DenseNet-121: 3103.3s vs. 70.1s), while **online per-sample latency is
nearly identical** between the two (e.g. 2.464ms vs. 2.468ms,
CIFAR-10/DenseNet-121). Status: `verified implementation issue` (concrete,
reproducible cost split; consistent with the paper's own qualitative claim
that RGCL costs more offline, but quantifies it far more precisely than the
paper does).

Full note: [[SPP changes the calibration-latency tradeoff rather than acting as free compression]]

### Feature representation matters more than confirmed for the scoring rule

Note: this does **not** cleanly establish "geometric scoring matters more than
feature substitution" as originally framed. `g_function` in this repo
(`Experiments/analyze_g_functions*.py`) refers to the **learned isotonic
calibration curve** mapping normalized separation score → probability, not the
scoring rule itself — no script was found that swaps the scoring rule
independent of the representation, despite ~20 named candidate metrics
existing in `Experiments/layer_selection.py:unit_weights_for_all_metrics()`.
The closest real ablation found swaps the *feature-pooling representation*
(SPP+JL vs. a DAC-style spatial-average+L2 pooling) holding the calibrator
fixed (`comprehensive_analysis.comparisons.json`, key
`"Geometric: SGC vs Geo+DAC Features (SPP+JL vs spatial avg+L2)"`): mostly
non-significant on CIFAR (2/9 significant), but on SVHN the DAC-style pooling
is dramatically better (SVHN/ResNet-18: 0.02517 vs. 0.00096 ECE, p=0.041).
Status: `suspicious artifact / likely mechanism` for the SVHN result;
`insufficient evidence` for the originally-hypothesized scoring-vs-feature
question.

Full note: [[Geometric scoring matters more than feature substitution]]

### Neighbour count k controls OOD detection far more than ID calibration

`tables/k_ablation/table_k_trend_sgc.tex` /
`k_ablation_aggregated_results.csv`: as k goes 1→300 (SGC/RGCL feature mode),
ID ECE is flat/non-monotonic (1.03%→0.96%→1.04%→1.06%, within ~±0.25–0.32
std — no real trend), while OOD AUROC rises monotonically from 0.700 to 0.843
(+14.3 points). Status: `verified implementation issue` (a real ablation,
clean monotonic trend).

Full note: [[Neighbour count controls OOD more than ID calibration]]

### Composite unsupervised layer selection failed against an oracle (backup-only evidence)

**Source-precedence caveat:** the canonical repository's
`robust_layer_selector.py` (the composite unsupervised layer-selection
module) has **zero callers anywhere in the canonical codebase** and produced
no output artifacts on this snapshot — it is unused/dead code here. The
concrete 12-experiment result (AugMix/CIFAR-10, DenseNet-121/ResNet-18/
ResNet-50 × seeds 11–14) exists **only** in
`_backup_unique_from_GeometricInternalCalibration_1_20260627/aaai_full_experiments/results/analyze_result.ipynb`
— a sibling backup directory the workspace notes say not to use unless
explicitly needed. It is used here because it is the *only* surviving
evidence for this historically important negative result, and it is flagged
accordingly; do not treat it as canonical.

**The task brief's assumed number ("matched the oracle ECE layer only 1/12
times") does not match what is recoverable from this notebook** — no
oracle-index-match metric of that exact form was found. The actual recovered
metric is "beats the naive fixed-default (`Physical`) baseline": mean ECE
across the 12 experiments — Optimal (oracle) = 0.002983, ECE-Only-selector =
0.007201, Physical (fixed default) = 0.007208, **Composite Score selector =
0.016435 (worst of the four, ~5.5× the oracle's ECE)**. "Beats Physical" rate:
Composite Score 3/12 (25.0%), ECE-Only 5/12 (41.7%), Optimal 11/12 (91.7%).
Status: `under-replicated` (n=12, backup-sourced) but the qualitative
conclusion is clear and consistent with the task's intended lesson: geometry
that looks strongly separated by an unsupervised composite score is not
necessarily geometry that predicts calibration error — here it did *worse*
than the simplest fixed heuristic.

Full note: [[Composite geometry scores do not reliably select calibration layers]]

### DAC collapses on randomized-coordinate features

`comprehensive_analysis.comparisons.json`, key
`"DAC: Original vs DAC with Random Coordinates"`: DAC's own density-aware
calibration mechanism, applied to RGCC-style randomized-coordinate features
instead of its native layer features, degrades sharply and specifically on
CIFAR-100 — e.g. `baseline_cross_entropy_cifar100_resnet18`: ECE 0.033→0.218;
`augmix_cifar100_resnet18`: 0.073→0.324; Cohen's d ranging −30 to −42 (extreme
effect sizes). CIFAR-10 rows in the same table show small/non-significant
differences. Status: `verified implementation issue` (a real ablation with a
striking, reproducible, dataset-specific collapse).

Full note: [[DAC can collapse on randomized coordinate features]]

### ID calibration gains vs. OOD detection gains — dissociation

The codebase's own reporting design (`create_oracle_symlinks.py`, function
`generate_detailed_report`) explicitly separates "positive improvement" on ID
ECE from "positive improvement" on OOD AUROC into different output files —
evidence the authors treated these as separate axes. The generated instances
of those specific files were not located on disk. The clearest concrete
stand-in is the k-ablation table above: from k=50→300, ID ECE mildly
*worsens* (0.99%→1.06%) while OOD AUROC keeps improving (0.788→0.843).
Status: `suspicious artifact / likely mechanism` (design intent confirmed by
code; a comprehensive comparison table was not located).

Full note: [[ID calibration gains do not imply OOD detection gains]]

### A possible sentinel/unit artifact in an equivalence table

`latex_tables_statistics_n/table_sgc_vs_faiss_tost_{10,20,30}pct.tex`: rows
for CIFAR-10/DenseNet-121 and CIFAR-10/ResNet-152 render as `0.59±0.17` (SGC)
vs. `0.42±100.00` (SGC-FAISS) — an implausible standard deviation of exactly
`100.00`, alongside a CI/Diff shown as `"---"` and TOST p = 1.000, consistent
with an n=1 (undefined-variance) case rendered through a sentinel fallback
instead of being excluded. The exact code path was not conclusively traced
(candidate region `Experiments/generate_sgc_tables.py` lines ~1600–1660).
**Scope correction:** this is not an OOD-metric table and the anomaly is not
confirmed to be a unit-scaling (percentage-vs-fraction) bug specifically — it
is a plausible sentinel-value artifact in an SGC-vs-FAISS-backend equivalence
table. Status: `suspicious artifact / likely mechanism`, narrowly scoped.

Full note: [[OOD uncertainty tables can hide unit-scaling errors]]

### PCA vs. random projection (JL) — status: unresolved / unvalidated

The repository's `utils/compression_utils.py::SmartCompression` supports both
`method='pca'` and `method='random_projection'` (JL-style), plus dedicated
`FixedSizeSPP_JL`/`ChannelFirstSPP_JL`/`MultiScaleSPP_JL` classes. However,
`Calibrators/geometric_calibrator_new.py` hardcodes
`method='fixed_spp_jl'`, and no call site anywhere passes `method='pca'` —
PCA is present but unexercised code. **No CSV/JSON/tex artifact anywhere
compares PCA vs. JL.** There is no basis in this repository for a "PCA lost to
JL" conclusion — PCA was never actually run in a comparison, so it cannot be
said to have lost.

## Unresolved questions

- No canonical, single RGCL-vs-RGCC equivalence count exists across the
  repository's own artifacts (see above) — this remains an open conflict, not
  resolved by this archaeology pass.
- Whether randomized layer *multiplicity* (as opposed to neighbour count k)
  specifically improves OOD detection was investigated and **no artifact was
  found** (`Experiments/analyze_layer_count_ablation.py` varies layer count
  but contains no OOD/AUROC code, and produced no output artifacts on this
  snapshot). This claim is intentionally **not** added to the vault.
- A scoring-rule-independent-of-representation ablation (as opposed to the
  representation-pooling swap that was found) does not exist in this
  repository.

## Artifact ledger

| Finding | Artifact | Evidence type | Integrity |
|---|---|---|---|
| RGCL/RGCC ECE ~78% reduction avg, up to 97% | published paper, Table 1 | paper | clean |
| RGCL/RGCC vs. DAC/TULIP equivalence 10/13 | published paper, Table 2 | paper | clean |
| RGCL vs RGCC equivalence 9/13 (paper) vs. 3 different repo tables (7-9 of 10-14) | paper Table 2 vs. `calibration_comparison_{mce_adaptive,separate,new}/tost_rgcc_rgcl/*.csv` | paper vs. multi-seed aggregate | conflicting analysis definition |
| SPP offline cost 16x-44x RGCC, ~equal online latency | `timing_analysis/tableA_offline_online.csv` | multi-seed aggregate | clean |
| SPP+JL vs DAC-style pooling: SVHN outlier, CIFAR mostly n.s. | `comprehensive_analysis.comparisons.json` | multi-seed aggregate | exploratory |
| k=1→300: OOD AUROC +14.3pp, ID ECE flat | `tables/k_ablation/k_ablation_aggregated_results.csv` | multi-seed aggregate | clean |
| Composite layer selector worst of 4 methods, beats fixed baseline 3/12 | `_backup_.../aaai_full_experiments/results/analyze_result.ipynb` | single small aggregate (n=12) | under-replicated; **non-canonical backup source** |
| DAC collapses on randomized coordinates (CIFAR-100 only) | `comprehensive_analysis.comparisons.json` | multi-seed aggregate | clean, dataset-specific |
| Sentinel std=100.00 in SGC-vs-FAISS TOST table | `latex_tables_statistics_n/table_sgc_vs_faiss_tost_*.tex` | implementation/table inspection | suspicious artifact, root cause not fully traced |
| PCA vs JL comparison | `utils/compression_utils.py`, `Calibrators/geometric_calibrator_new.py` | implementation inspection | no comparison exists — unresolved/unvalidated |

## Ancestor / descendant projects

Ancestor: [[Geometric Separation]]
Descendant: [[Full-Vector Geometric Calibration]]
