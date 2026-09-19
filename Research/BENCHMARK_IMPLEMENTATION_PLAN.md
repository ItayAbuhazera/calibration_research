# Benchmark Implementation Plan — Geometric / Representation-Based Calibration

Status: **FROZEN DESIGN, NOT YET IMPLEMENTED.** No code has been written or run
against this plan. This file is the pre-registration for Phase 0–2 of the
canonical benchmark. Treat it as immutable once approved — deviations during
implementation are appended to §16, never silently edited into the design
above them.

Canonical repo for all changes: `GeometricFullCalibration/` (paths below are
relative to it unless stated otherwise). Verified directly against this
repo's own code, not assumed: `utils/coordinate_extraction.py` and
`utils/layer_utils.py` here are self-contained (not a cross-repo import from
`geometric/GeometricInternalCalibration/`), and the CIFAR-100/ResNet-101
checkpoints load `Net/resnet_cifar.py::resnet101`, which has no `maxpool`
module and pools via a bare `F.avg_pool2d` functional call (not a hookable
`nn.Module`).

---

## 1. Scientific questions

**Study A (native method benchmark).** Which complete, faithfully-run method
— own representation, own statistic, own mapper, own capacity — produces the
best calibration/decision quality on CIFAR-100/ResNet-101, clean and under
CIFAR-100-C?

**Study B (controlled factorial).**
- B1 (statistic effect): at a *fixed* representation, does the GC-derived or
  DAC-derived per-layer statistic produce better calibration, when both are
  forced through one identical pooling/aggregation/mapper pipeline?
- B2 (representation effect): at a *fixed* statistic, does corrected
  random-internal-layer selection or DAC's prescribed deterministic layer set
  produce a better representation, under that same shared pipeline?

**Full-vector questions.** Does class-wise geometric information (GC
per-class distance, or a DAC-style per-class density) add decision value
(accuracy, NLL, net flips) beyond a *capacity-matched, geometry-ablated*
non-geometric control — Temperature Scaling for the 1-parameter fusion,
GLAD-PI's own zero-geometry twin for the learned head-aware model?

---

## 2. Frozen method list

**Phase 0 — zero new GPU work.**
Uncalibrated · Temperature Scaling · PTS · Vector Scaling · Dirichlet/ODIR ·
Top-label Isotonic (verify in Phase 1 whether `Calibrators/isotonic_regression.py`
already targets raw model confidence — if so this is literally zero-code) ·
KCal (full) · Trust Score (diagnostic + switch) · GC-DAC · Published RGCL
(Study A reference row; legacy `C2..C5` draws only, reported as historical,
never as primary corrected-method evidence).

**Phase 1 — cheap new code, no large extraction.**
Native DAC wired as a standalone Study-A row · Mahalanobis confidence
(variant A) · GLAD-PI zero-geometry twin · the fc-exclusion filter (§8a) ·
the unified extraction script (§8b), built and smoke-tested on **one**
checkpoint only (1 fresh forward pass).

**Phase 2 — the designed experiments.**
Run the unified extraction on the remaining 4 checkpoints (4 more fresh
passes, 5 total) · compute Study A's corrected-internal-only RGCL primary
evidence + the checkpoint×draw variance decomposition + the full Study-B 2×2
— all from the same cached outputs.

**Deferred (not in this plan).** Mahalanobis-LDA posterior (variant B) ·
AAR-lite (relabeled proxy) · RGCC (reproduction/ablation only — no per-layer
structure for the DAC factorial, and its coordinate-level pool is too large
to cache under the unified extraction plan) · full-vector DAC-style
class-wise construction · softmax-kNN-blend matched control · official
multi-layer Mahalanobis/AAR · a second dataset/architecture · the
same-representation weakened-readout test for the shift-recoverability
strand (independent of this ID benchmark, reuses the same checkpoints,
worth running in parallel on the user's call).

---

## 3. Exact seed design

- **Checkpoint seeds**: reuse existing `C1..C5`
  (`.../baseline_cross_entropy/cifar100/resnet101/seed{1..5}/best_model.pth`
  per `artifacts/recoverability/seed2/preflight_complete.json`'s checkpoint
  path). No retraining.
- **Common RGC draw seeds**: `D1 = 9101`, `D2 = 9102`, `D3 = 9103` — three
  fixed integers, applied identically to every checkpoint, chosen only to
  avoid collision with the checkpoint identifiers (1–5) or the historical
  per-checkpoint seeds (also 1–5). Arbitrary values, not tied to any special
  meaning — fix and document, never re-roll after seeing results.
- **fc/classifier exclusion**: applies to `D1, D2, D3` on every checkpoint
  (the "corrected" rule). Never applied retroactively to historical draws.
- **Historical artifact roles**:
  - `C1`'s original draw (seed 1, confirmed fc-included via
    `artifacts/recoverability/seed1/preflight_complete.json`) — **never**
    used as ordinary corrected-RGC evidence; citable only as the forensic
    example of the bug.
  - `C2..C5`'s original draws (seed = checkpoint id, confirmed fc-excluded)
    — **auxiliary/legacy only**. Reported separately for continuity with
    prior results; excluded from the primary 15-cell crossed-effects
    analysis and from Study B entirely (§9 artifact-reuse matrix, unchanged
    from the prior round: wrong granularity, already-projected 256-dim
    output, not raw per-layer activations).

---

## 4. Exact representation definitions

| Representation | Mechanism | Layer count | Pool eligibility | Role |
|---|---|---|---|---|
| **Published RGCL** | Unmodified historical `coordinate_extraction.py`, untouched | L=6 (native default) | fc-eligible (may or may not be drawn) | Study A reference only; code never modified |
| **Corrected internal-only RGCL (Study A)** | New fc-exclusion filter (§8a) applied to `layer_map` before planning | L=6 (matches published default for apples-to-apples Study-A comparison) | fc explicitly excluded | Study A primary random-layer method, evaluated on `D1,D2,D3 × C1..C5` |
| **Corrected internal-only random layers (Study B)** | Same fc-exclusion filter | **L=5** (count-matched to DAC, §2 of the prior round) | fc explicitly excluded | Study B rows A/B only |
| **DAC-prescribed layers** | `get_dac_target_layers("resnet101", model)`, verified from `Calibrators/density_aware_calibration.py` against this repo's actual `Net/resnet_cifar.py` (no `maxpool` → PRE-BLOCK resolves to `conv1`) | **L=5**, exactly `["conv1","layer1","layer2","layer3","layer4"]`, deterministic, identical for every checkpoint (architecture-determined, not weight-determined) | N/A — fixed set | Study A (`GC-DAC`, native DAC) and Study B rows C/D |

**Granularity caveat, retained not resolved**: DAC's 5 layers are coarse,
deterministic, *non-leaf* container modules (`layer1..layer4`, each wrapping
dozens of Bottleneck sub-blocks, hooked for their aggregate stage output).
RGC's discovery mechanism hooks **leaf modules only**
(`len(list(module.children()))==0`), so it structurally can never select
`layer1..layer4` as whole units — confirmed directly from
`artifacts/recoverability/seed2/preflight_complete.json`'s `selected_layers`
(`layer2.1.conv1`, `layer1.2.bn3`, etc. — individual leaf submodules). For a
standard 4-stage ResNet there are only 5 stage-level candidates total,
identical to DAC's own set, so constraining the RGC row to that same coarse
grain would make "randomly sample 5 of 5" degenerate. Study B therefore keeps
RGC's native fine-leaf pool, trimmed to `L=5` by count only — the
representation axis (§1, B2) tests "count-matched random-fine-leaf vs.
deterministic-coarse-stage," not a pure apples-to-apples grain match. State
this every time B2's result is reported.

---

## 5. Exact Study-B 2×2

| | GC-derived statistic | DAC-derived density statistic |
|---|---|---|
| **Corrected random internal layers (L=5)** | **A** | **B** |
| **DAC-prescribed layers (L=5)** | **C** | **D** |

- **A, B**: computed per `(checkpoint, draw)` cell — 5×3=15 instances each.
  A and B **share the identical representation** (same cached per-layer
  pooled features for that checkpoint×draw) by construction — both
  statistics are computed from one extraction.
- **C, D**: computed per checkpoint only — 5 instances each (DAC's layer
  selection has no draw seed). C and D **share the identical
  representation** — same cached DAC-layer pooled features per checkpoint.
- **Common pooling** (identical, all 4 cells): spatial-average-pool +
  L2-normalize per layer — reuse the exact pooling function DAC's native
  implementation already uses (`Calibrators/density_aware_calibration.py`),
  don't re-derive it.
- **Common aggregation** (identical, zero learned parameters, all 4 cells):
  rank-normalize each layer's statistic independently against the fitting
  split, then take the **unweighted mean** across the 5 selected layers →
  one scalar `g(x)` per sample. No learned weights anywhere (this, not the
  layer count alone, is what makes "aggregation capacity" equal across
  cells — DAC's own native weighted sum is intentionally not used here).
- **Common mapper** (identical, all 4 cells): rank-normalize `g(x)` (ECDF on
  the fitting split) → isotonic regression (PAV, reuse
  `Calibrators/isotonic_regression.py`) fit against binary correctness →
  assign to the predicted class (same argmax-preserving-by-construction
  family as native GC/RGC).
- **Fitting split**: the existing calibration/fitting split (§7), identical
  across all 4 cells.
- **Objective**: isotonic PAV minimizing squared error against binary
  correctness — identical across all 4 cells, no alternative loss.
- **Hyperparameter budget**: `L=5` fixed, no search. DAC statistic's own
  `k` (kNN density neighbor count) held at DAC's paper-default value,
  applied identically wherever the DAC statistic appears in Study B.
- **Critical distinction to keep visible in every table**: **Cell D is not
  native DAC.** D uses DAC's *statistic* only, through the shared
  rank+isotonic mapper — it is scalar/top-label output, argmax-preserving
  by the GC/RGC mapper convention, **not** DAC's own
  `softmax(z / S(x,w))` full-vector reconstruction. Native DAC (Study A)
  stays a fully separate row with its own mapper and its own metric
  entitlements (§6). Confusing D with native DAC is the single most likely
  reporting error in this plan — a validation test exists specifically for
  it (§10).

---

## 6. Method × metric compatibility (frozen)

Predeclared rule (unchanged from the prior round, restated as frozen): scalar
top-label methods **do not** get multiclass NLL/Brier/classwise-calibration
entries. No post-hoc reconstruction rule is used for them.

| Method | Native output | Bucket | NLL/Brier/classwise? |
|---|---|---|---|
| Uncalibrated | raw softmax | Both | Yes |
| Temperature Scaling, PTS | `softmax(z/T or z/S(x))` | Full vector | Yes |
| Vector Scaling, Dirichlet/ODIR | class-affine + softmax | Full vector | Yes |
| Native DAC | `softmax(z/S(x,w))` | Full vector | Yes |
| Top-label Isotonic, GC-DAC, Published RGCL, Corrected RGCL (Study A) | isotonic + uniform-spread | **Scalar only** | **No** |
| Study-B cells A/B/C/D | common rank+isotonic mapper | **Scalar only** | **No** |
| KCal (full) | KDE posterior | Full vector | Yes |
| Trust Score (diagnostic/switch) | scalar / possible top-2 swap | **Scalar only** (+ argmax-change metrics for switch) | No |
| Mahalanobis confidence (A) | scalar distance | **Scalar only** | **No** |
| Full-vector GC fusion, GLAD-PI (both arms) | `softmax(log p + λS(x))` / per-class MLP correction | Full vector | Yes |

Scalar bucket gets: top-label ECE, Adaptive-ECE, calibration curve/error,
correctness AUROC/AURC, risk-coverage, coverage-at-matched-risk. Full-vector
bucket additionally gets NLL, multiclass Brier, classwise/top-label
calibration, and (where decision-changing) accuracy/argmax-change/flips.

---

## 7. Data/split protocol

- **Train/reference**: full CIFAR-100 training set — used both for the
  already-frozen classifier checkpoints and as the kNN/Gaussian reference
  bank for GC/RGC/DAC/Trust-Score/Mahalanobis. No retraining.
- **Calibration/fitting split**: the existing held-out split already used to
  fit TS/VS/Dirichlet/isotonic/native-RGC mappings. Reused verbatim, not
  redefined, for every method in this plan including the Study-B common
  mapper.
- **Validation split**: the existing split used for hyperparameter grids
  (λ/α/β selection). Reused verbatim.
- **Clean test**: existing CIFAR-100 clean test split.
- **CIFAR-100-C cells**: the existing 12 cells already used by the
  shift-recoverability project (4 corruption types × severities {1,3,5}).
  Reused exactly — no cells added or dropped without separate justification.
- **`sample_id` alignment**: every artifact produced under this plan must
  carry a persistent, dataset-index-based `sample_id` so per-sample outputs
  from different methods can be joined and paired-compared. Verify (don't
  assume) that existing `exports/seed{N}/...` arrays are already index-order
  aligned across methods/splits before joining anything against them — this
  is validation test §10.8, not a free assumption.

---

## 8. Implementation map

### (a) fc-exclusion filter — `utils/layer_utils.py`
- **Add**, do not modify in place: `is_feature_layer_corrected(name, module,
  classifier_attr_names=("fc",))` — calls the existing `is_feature_layer`,
  then additionally excludes exact/prefix matches against the model's known
  classifier attribute name(s).
- **Why a new function, not an edit to `is_feature_layer`**: Published
  RGCL's Study-A reference row must keep calling the original, completely
  unmodified function — CLAUDE.md/RESEARCH_WORKSPACE.md explicitly forbid
  silently changing historical reproduction code.
- **Reuse unchanged**: `discover_coordinate_space`, `plan_coordinate_extraction`,
  `plan_nested_coordinate_extraction` — the correction is applied as a
  filter step on `layer_map` *between* discovery and planning, not inside
  any of these functions.

### (b) Unified single-pass extraction — new file
`Experiments/extract_unified_studyAB_representations.py`
- **`build_combined_hook_set(model, dac_layers, draw_layer_sets)`**: merges
  DAC's 5 deterministic layer names with the union of `D1,D2,D3`'s layer
  names (deduplicated by name), registers forward hooks on all of them.
  Valid because layer identities are fully determined in advance —
  `plan_coordinate_extraction` is a pure function of
  `(total_size, num_coordinates, layer_map, seed)`, and `layer_map` depends
  only on architecture, not trained weights, so the same seed selects the
  same layer names on every checkpoint of the same architecture (this is
  also why "same common draw seeds across checkpoints" is scientifically
  meaningful at all — §10.2 tests it directly).
- **`extract_combined_features(model, loader, combined_hook_set, device)`**:
  one forward pass per batch capturing every hooked raw activation; for each
  captured layer, immediately compute (i) avg-pool+L2-norm (Study B pooling,
  and reusable for native-DAC-in-Study-A on the 5 DAC layers) and (ii), only
  for the 3 draws' layers, SPP-pool then project via that draw's fixed
  Gaussian projection matrix to 256-dim (Study A's native pipeline) — then
  **discard** the large raw/SPP intermediates immediately, keeping only the
  small derived vectors, to bound storage (§14).
- **Reuse, do not reimplement**: `get_dac_target_layers`,
  `plan_coordinate_extraction`, the existing SPP + Gaussian-projection code
  from RGCL's own extraction path, and DAC's own avg-pool+L2-norm function
  from `Calibrators/density_aware_calibration.py` (byte-identical pooling
  convention on both Study-B rows is required by §5).

### (c) Study-B common pipeline — new file
`Calibrators/studyB_common_pipeline.py`
- **`CommonFactorialCalibrator(representation_layers, statistic_fn, k)`**:
  per-layer statistic (`statistic_fn` ∈ {`gc_separation_statistic`,
  `dac_density_statistic`}, both reused from `geometric_calibrator.py` /
  `density_aware_calibration.py`, not reimplemented) → rank-normalize per
  layer → unweighted mean → rank-normalize → isotonic (reuse
  `isotonic_regression.py`).
- **Why new**: this exact shared-mapper combination doesn't exist anywhere
  in the current codebase — every existing pipeline bundles one specific
  statistic with one specific mapper.

### (d) Mahalanobis confidence (A) — new file
`Calibrators/mahalanobis_confidence.py`
- **`MahalanobisConfidenceCalibrator`**: per-class mean + Ledoit-Wolf-shrunk
  shared covariance (`sklearn.covariance.LedoitWolf`) on penultimate
  features, scalar score = `min_c` class Mahalanobis distance, mapped via
  the same isotonic-diagnostic convention Trust-Score-diagnostic already
  uses (reuse that mapping pattern from `Calibrators/trust_score.py`).
- **Reuse check required in Phase 1, not assumed**: verify whether a
  penultimate-feature cache already exists from Trust Score/GC/KCal's own
  runs before assuming this needs its own fresh forward pass (§14 flags
  this explicitly as unresolved).

### (e) GLAD-PI zero-geometry twin — modify `Calibrators/glad_pi.py`
- Add a `zero_geometry: bool` constructor flag that masks `distance_k` to 0
  before the MLP — not a separate subclass, so architecture and parameter
  count are identical by construction (the entire point of the control).

### (f) Native DAC standalone — modify `Experiments/run_unified_benchmark.py`
- Add a new method branch mirroring the existing `gc_dac` branch (~line
  3350), calling `DensityAwareCalibrator` directly with
  `get_dac_target_layers`-derived layers. Integration only, no new math.

### (g) Top-label isotonic — verify first
- Check whether `Calibrators/isotonic_regression.py` already accepts raw
  model confidence as input. If yes: zero new code, one new call site in
  `run_unified_benchmark.py`. If it's hardwired to a geometric-score input
  path: add a thin wrapper, don't modify the existing class.

---

## 9. Artifact schema

One row per `(method, checkpoint_seed, rgc_draw_seed, representation_strategy,
statistic, mapper, dataset, corruption, severity, split)`, JSON or Parquet:

```json
{
  "method": "studyB_cell_A",
  "checkpoint_seed": 3,
  "rgc_draw_seed": 9102,
  "representation_strategy": "corrected_random_internal_L5",
  "statistic": "gc_separation",
  "mapper": "common_rank_isotonic",
  "dataset": "cifar100",
  "corruption": "gaussian_noise",
  "severity": 3,
  "split": "test",
  "sample_ids": "ref:exports/seed3/perclass/gaussian_noise_s3.npz#sample_id",
  "metrics": {
    "top_label_ece": null,
    "adaptive_ece": null,
    "accuracy": null,
    "argmax_change_rate": null,
    "nll": null,
    "brier": null,
    "_metric_bucket": "scalar_only"
  },
  "provenance": {
    "extraction_script": "Experiments/extract_unified_studyAB_representations.py",
    "extraction_job_id": "studyAB_ckpt3_20xx-xx-xx",
    "git_commit": "<sha>",
    "config_hash": "<sha256 of the resolved run config>",
    "plan_version": "BENCHMARK_IMPLEMENTATION_PLAN.md@<git-blob-sha-of-this-file>"
  }
}
```

- `rgc_draw_seed`: `null` for methods with no draw (DAC-prescribed rows,
  standard calibrators, native DAC).
- `metrics._metric_bucket`: `"scalar_only"` or `"full_vector"`, set at
  write time from the frozen §6 table — never inferred after the fact, and
  a `null` NLL/Brier for a scalar-only method must be a real `null` in the
  file, not an omitted key (so downstream aggregation scripts fail loudly
  instead of silently treating a missing key as zero).
- `config_hash`: hash of the fully-resolved run configuration (all fixed
  seeds, split identifiers, L, k, mapper choice) — enables detecting an
  accidental parameter drift between two nominally-identical runs.

---

## 10. Validation tests, run BEFORE any scientific extraction/evaluation

All of these must pass on cheap synthetic or single-checkpoint smoke data
before Phase 2's full run:

1. **fc cannot appear in corrected draws**: for `D1,D2,D3` on every
   checkpoint, assert `"fc" not in sampling_plan.keys()` and
   `"fc" not in {l["name"].split("#")[0] for l in filtered_layer_map}`.
2. **Seed→layer-name consistency across checkpoints**: assert that
   `plan_coordinate_extraction(seed=D1, layer_map=layer_map_for_architecture)`
   yields the identical set of layer *names* (not indices, since only names
   need to match — indices can differ if weights differ layer sizes, which
   they won't here since architecture is fixed) for every checkpoint C1..C5
   — this is the direct test of the "same seed → same corresponding draw
   across checkpoints" claim underlying the whole crossed design.
3. **Study-B A/B share identical representation**: for a given
   `(checkpoint, draw)`, assert the cached pooled per-layer feature array
   feeding cell A is byte-identical to the one feeding cell B (they must be
   read from the same cache entry, not recomputed twice).
4. **Study-B C/D share identical representation**: same assertion for C/D
   per checkpoint.
5. **All four cells use identical mapper/fitting split**: assert the
   `CommonFactorialCalibrator` instances backing A/B/C/D reference the same
   fitting-split sample-ID set and the same isotonic-fit call signature
   (same objective, no cell-specific hyperparameters).
6. **DAC native row stays distinct from Study-B D**: assert the two are
   produced by different code paths (`DensityAwareCalibrator` vs.
   `CommonFactorialCalibrator`) and that their output distributions differ
   (native DAC's output must be a valid full probability vector summing to
   1 across all classes with per-sample variation consistent with
   `softmax`; Study-B D's output must be a scalar assigned-to-predicted-class
   value) — a schema-level check, not just a code-path check.
7. **Scalar methods cannot emit fabricated multiclass NLL/Brier**: assert
   that for every method flagged `scalar_only` in §6, the artifact writer
   raises rather than silently writing a non-null `nll`/`brier` field.
8. **Sample IDs and labels align across methods**: for a shared test split,
   assert every method's per-sample array is joinable on `sample_id` against
   the ground-truth label array with no reordering — explicitly test this on
   at least one existing `exports/seed{N}/...` file before trusting it.
9. **No historical scripts/results are overwritten**: the new extraction
   script and new calibrator files must write only to new, plan-specific
   output paths (e.g. `results/studyAB/...`); a pre-flight check asserts none
   of those paths collide with any existing path under `results/`,
   `exports/`, or `artifacts/`.

---

## 11. Statistical analysis plan

- **B1 (statistic effect, "A vs B", "C vs D")**: paired comparison **within
  the same representation**, at full granularity — 15 paired differences for
  A vs B (one per checkpoint×draw cell), 5 paired differences for C vs D
  (one per checkpoint). No averaging before this comparison.
- **B2 (representation effect, "(A,B) vs (C,D)")**: the random-draw arm is
  **averaged within checkpoint first** (mean over `D1,D2,D3` for a fixed
  statistic), then paired against that checkpoint's single DAC-prescribed
  value — 5 paired differences, one per checkpoint.
- **Checkpoint×draw variance decomposition** (corrected-RGCL, Study A):
  two-way crossed random-effects model,
  `Y_{ij} = μ + checkpoint_i (random, i=1..5) + draw_j (random, j=1..3) + ε_{ij}`,
  fit per metric. With one observation per cell, checkpoint×draw interaction
  is confounded with residual noise — stated as an accepted limitation, not
  resolved by this design.
- **Confidence intervals**: paired bootstrap over the relevant set of paired
  differences (15 for B1-random-row, 5 for B1-DAC-row, 5 for B2) — preferred
  over a parametric CI given no assumption that ECE-like metrics are
  normally distributed at this sample count.
- **No winner from one metric alone**: every headline comparison (Study A
  ranking, B1, B2, full-vector questions) must report the full applicable
  metric bucket from §6 side by side — a method cannot be declared better on
  ECE alone while NLL/accuracy/flips go unreported.

---

## 12. Execution order

**Phase 0**
```
python Experiments/run_unified_benchmark.py \
  --dataset cifar100 --model resnet101 --checkpoints seed1..seed5 \
  --methods uncalibrated,temperature_scaling,pts,vector_scaling,odir_dirichlet,\
top_label_isotonic,kcal_full,trust_score_diagnostic,trust_score_switch,gc_dac,\
published_rgcl \
  --split clean,val,cifar100c_all_cells \
  --output_dir results/studyAB/phase0
```

**Phase 1**
```
# (a) implement fc-exclusion filter, unified extraction script, Mahalanobis
#     confidence, GLAD-PI zero-geometry twin, native-DAC wiring — code only.
# (b) run validation tests (§10) on synthetic/smoke data.
# (c) one smoke extraction, checkpoint C1 only:
python Experiments/extract_unified_studyAB_representations.py \
  --checkpoint seed1 --draw_seeds 9101,9102,9103 \
  --dac_layers auto --output_dir results/studyAB/phase1_smoke

python Experiments/run_unified_benchmark.py \
  --dataset cifar100 --model resnet101 --checkpoints seed1..seed5 \
  --methods native_dac,mahalanobis_confidence,glad_pi,glad_pi_zero_geometry \
  --output_dir results/studyAB/phase1
```

**Phase 2**
```
# Remaining 4 checkpoints, same combined-hook extraction:
for ckpt in seed2 seed3 seed4 seed5; do
  python Experiments/extract_unified_studyAB_representations.py \
    --checkpoint $ckpt --draw_seeds 9101,9102,9103 \
    --dac_layers auto --output_dir results/studyAB/phase2/$ckpt
done

python Experiments/run_unified_benchmark.py \
  --dataset cifar100 --model resnet101 --checkpoints seed1..seed5 \
  --methods corrected_rgcl_studyA,studyB_cell_A,studyB_cell_B,studyB_cell_C,studyB_cell_D \
  --draw_seeds 9101,9102,9103 \
  --output_dir results/studyAB/phase2

python Experiments/aggregate_unified_benchmark_seeds.py \
  --input_dir results/studyAB/phase2 --analysis crossed_random_effects \
  --output results/studyAB/phase2/analysis
```

(CLI flags above follow this repo's existing `run_unified_benchmark.py`
argument conventions; verify exact flag names against current `--help`
output before running — do not assume flag names untested.)

---

## 13. Stop/go checks after each phase

- **After Phase 0**: if standard controls (TS/VS/Dirichlet) already explain
  away most of published RGCL's/GC-DAC's/KCal's apparent gain over
  Uncalibrated, the expensive Study-B factorial work in Phase 2 may not be
  worth running before re-examining whether there's a real effect left to
  decompose.
- **After Phase 1**: if Mahalanobis confidence alone matches or beats
  GC/RGC's scalar performance at far lower implementation/inference
  complexity, that's grounds to reconsider prioritizing further geometric
  complexity (Phase 2 factorial, later full-vector work) ahead of a cheaper
  parametric baseline.
- **After Phase 2, variance decomposition**: if checkpoint-to-checkpoint
  variance dominates draw-to-draw variance by a wide margin, single-draw
  evaluations are defensible for future work (cheaper going forward); if
  draw variance is comparably large, every existing single-seed geometric
  claim in this codebase should be flagged as under-evidenced until
  re-checked against multiple draws.
- **After Phase 2, the 2×2 itself**: if both B1 (statistic effect) and B2
  (representation effect) come back near zero — neither GC vs. DAC statistic
  nor corrected-random vs. DAC-prescribed representation matters — that is a
  real negative result. It would mean this benchmark's two studied axes do
  not explain wherever RGC's apparent gains actually come from, and Phase
  3/4 (full-vector, second dataset) should be re-scoped rather than run on
  the assumption that "more of the same axis" will resolve it.

---

## 14. Estimated compute

- **Fresh forward passes**: **5** — one combined-hook extraction job per
  checkpoint (each internally sweeping the calibration/fitting split,
  validation split, clean test, and all 12 CIFAR-100-C cells once, with the
  full merged hook set attached throughout). This replaces what earlier
  rounds estimated as up to 26 narrower passes — the consolidation in §8b is
  the reason, and is the direct answer to this round's "inspect whether one
  forward pass can hook both" question.
- **+1 possible**, unconfirmed: Mahalanobis confidence may need its own
  penultimate-feature extraction if no existing cache from
  Trust-Score/GC/KCal covers it — verify in Phase 1 before assuming either
  way.
- **Reusable, zero new compute**: all 5 trained checkpoints; existing
  calibration/val/test splits and CIFAR-100-C cells; legacy `C2..C5` draws
  (auxiliary reporting only); `get_dac_target_layers`; existing SPP/JL
  projection code; existing TS/VS/Dirichlet/isotonic/KCal/Trust-Score
  implementations.
- **CPU/GPU**: inference-only (no backprop) on already-trained checkpoints —
  a modest single-GPU job per checkpoint; cheaper per-pass than the
  historical native-RGCL-only extraction (~790s per checkpoint-draw observed
  in `preflight_complete.json`) since Study B's pooling skips SPP/JL
  entirely and the combined pass amortizes the SPP cost for Study A's 3
  draws into one sweep instead of three.
- **Storage**: bounded by discarding raw/SPP intermediates immediately after
  deriving the small final vectors (256-dim per draw for Study A; small
  per-layer pooled vectors for Study B) — order tens of MB per checkpoint,
  not the multi-GB/TB a full raw-activation cache would require. Exact byte
  counts TBD pending final vector dimensions; compute and log actual sizes
  during the Phase 1 smoke test before committing to Phase 2's full run.

---

## 15. ResearchBrain update policy

- **No ResearchBrain writes until results exist and are verified.** This
  plan file is not itself a ResearchBrain artifact — it lives at the
  workspace root by explicit instruction.
- After a phase completes and is verified: add a dated
  `ResearchBrain/05_Experiments/` card citing this plan file's path and git
  blob SHA, exact artifact paths, and seeds (`checkpoint_seed`,
  `rgc_draw_seed`) — never copy raw numbers into prose without that
  citation.
- Promote to `02_Observations/` only with a concrete evidence source and an
  explicit "what this does NOT establish" section, and only after checking
  the seed/draw count actually supports the claim being made (5 checkpoints
  × 3 draws is thin for anything beyond "detected a variance component
  exists," per §11's stated limitation).
- **This plan (§1–14) is the pre-registration and is immutable once
  approved.** Any deviation discovered during implementation (a flag name
  that doesn't match, a cache that turns out reusable/unreusable differently
  than predicted, a metric that needs reclassifying) is appended to §16
  below with a date and reason — never silently edited into the sections
  above.
- **`ResearchBrain/06_Ideas/` is never touched** by this plan or its
  execution, per CLAUDE.md — zero edits, including cross-links.
- **No existing historical script or result is overwritten** — enforced by
  validation test §10.9, not just stated as a policy.

---

## 16. Deviations log

*(Empty at freeze time. Append dated entries here as implementation
proceeds — do not edit §1–14 in place.)*

### 2026-09-16 — Phase 0/1 implementation deviations (does not change §1's scientific questions)

1. **§8a/§4 named the wrong discovery mechanism for RGCL's real layer draw.**
   The plan attributes RGC's layer selection to `discover_coordinate_space`/
   `is_feature_layer`/`plan_coordinate_extraction` (`utils/coordinate_extraction.py`,
   `utils/layer_utils.py`). That is actually **RGCC**'s mechanism (the
   coordinate-level sibling method, explicitly deferred by §2). RGCL's real
   layer draw goes through `Experiments/run_rgc_experiments.py`'s own
   `select_random_rgc_layers` → `discover_model_layers` (`utils/utils.py`,
   hooks *all* modules matching a type allowlist including `Bottleneck`/
   `BasicBlock`, not leaf-only) → its own module-local `filter_non_feature_layers`
   (a separate, independently-implemented pattern list from
   `is_feature_layer`). The historical fc-inclusion bug lives in this second
   function, confirmed empirically: reproducing seed1's draw (L=6, uncorrected,
   real checkpoint) via `select_random_rgc_layers` returns
   `['layer3.1.conv2','fc','layer3.21.conv1','layer3.9.bn1','layer3.4.bn2','layer3.22.conv3']`
   — **byte-identical**, including `fc`'s position, to
   `artifacts/recoverability/seed1/preflight_complete.json`'s `selected_layers`.
   Fix: added `filter_non_feature_layers_corrected`/`select_random_rgc_layers_corrected`
   (additive, `Experiments/run_rgc_experiments.py`) as the actual fc-exclusion
   fix Study A/B's "corrected internal-only" layers use. The originally-planned
   `is_feature_layer_corrected`/`filter_out_classifier_layers`
   (`utils/layer_utils.py`) were still implemented as specified but are not the
   operative fix for this plan's purposes (kept; harmless; correctly tested;
   would matter only if RGCC is ever un-deferred).

2. **§4's granularity caveat is built on the same mistaken premise.**
   Since RGCL's real candidate pool (`discover_model_layers`) is not
   leaf-only, the "only 5 stage-level candidates, identical to DAC's own set"
   reasoning doesn't hold as stated. `discover_model_layers`'s `keep_kinds`
   allowlist does not include `Sequential`, so whole-stage containers
   (`model.layer1` etc.) are *not* selectable at all — the real candidate
   pool is a third granularity (individual `Bottleneck`-instance blocks, e.g.
   `layer2.3`, plus their leaf conv/bn sublayers; confirmed ~270 candidates
   for ResNet-101), structurally disjoint from DAC's 4 whole-stage-container
   layers. This does not change Study B's design or validity (still a
   defensible "count-matched random-fine/block vs. DAC-coarse-stage"
   comparison) — only the stated justification needs correcting.

3. **§8d's Mahalanobis-confidence mapping convention doesn't exist as
   described.** The plan cites reusing "the same isotonic-diagnostic
   convention Trust-Score-diagnostic already uses". Direct inspection of
   `Calibrators/trust_score.py::TrustScoreCalibrator.calibrate_diagnostic`
   and its call site in `Experiments/run_unified_benchmark.py`
   (`trust_score_original_diagnostic`) shows that convention is a genuine
   no-op on probabilities (base_probs returned unchanged; the scalar trust
   score is attached only as diagnostic metadata) — there is no isotonic
   mapping there to reuse. Since §6's frozen bucket table still requires
   Mahalanobis confidence to be scalar-only and argmax-preserving, the actual
   implementation (`Calibrators/mahalanobis_confidence.py`) reuses the real
   existing convention that satisfies that requirement instead: the
   "isotonic + uniform-spread" reconstruction GC-DAC and
   `Calibrators/isotonic_regression.py::TopLabelIsotonicCalibrator` already
   use.

4. **Stale checkpoint path prefix.** `artifacts/recoverability/seed{1..5}/
   preflight_complete.json`'s stored absolute `"checkpoint"` path
   (`/home/itayab/PyCharmProjects/geometric/...`) is missing the `Research/`
   segment and does not resolve; the correct prefix is
   `/home/itayab/PyCharmProjects/Research/geometric/...` (confirmed: `geometric/`
   is a sibling of `GeometricFullCalibration/`, not nested inside it — the
   same off-by-one bug was initially made in this plan's own
   `Experiments/extract_unified_studyAB_representations.py` and
   `Experiments/smoke_studyAB_phase1.py`, fixed before the Phase 1 smoke run).

5. **Integration bug in reusing `extract_and_aggregate_sgc_features` for the
   combined-hook extraction (§8b).** Calling it with `layer_names=[]` (to
   make its own SPP/projection path inert, relying purely on
   `observer_layer_names`/`batch_observer` for the combined hook set) breaks
   its own finalization step: `extract_with_spp`'s `batch_features.size > 0`
   guard silently drops every batch when `layer_names` is empty (each batch's
   own feature has width 0), leaving nothing to concatenate, and the
   function's own final `normalize(np.array([]))` call then crashes on an
   empty 1D array. Fixed by passing one real (harmless, discarded) layer name
   as `layer_names` instead of `[]`; `extract_and_aggregate_sgc_features`
   itself was not modified.

6. **`sample_count` added to the §9 artifact schema beyond the plan's own
   JSON example**, which only carries `sample_ids` as a ref string — added so
   every row is self-describing without dereferencing that ref (requested for
   smoke-test verification; harmless additive field, does not change any
   existing key's meaning).

7. **Phase 1 smoke test executed and verified** (checkpoint seed1, class-
   stratified 2000-sample training reference bank + 300/300 disjoint
   validation-split fit/test slices, git commit `94ec058cce87d4334886906dcce91e540574ff4f`):
   all 15 caller-specified verification items passed, including an
   independent re-check against the on-disk artifacts (not just the
   generating script's self-reported checks). Full report:
   `results/studyAB/phase1_smoke_job21387027/smoke_report.json`. Per plan
   policy, no ResearchBrain conclusions were drawn from this run and its
   numbers must never be cited as calibration-quality evidence.

---

## 17. Implementation checklist for a fresh session

A session with no memory of this conversation can execute this plan by:

1. Read this file in full before touching any code.
2. Confirm environment: `GeometricFullCalibration/` is the working repo;
   checkpoints at `.../baseline_cross_entropy/cifar100/resnet101/seed{1..5}/best_model.pth`
   exist and load.
3. Implement §8(a) — `is_feature_layer_corrected` in `utils/layer_utils.py`,
   as a new function, not an edit to the existing one.
4. Implement §8(g) — check `Calibrators/isotonic_regression.py`'s current
   input contract before writing any new code for top-label isotonic.
5. Implement §8(b) — the unified extraction script. Before running it at
   scale, run validation tests §10.1–10.2 against it on synthetic/dummy data.
6. Implement §8(c), (d), (e), (f) — Study-B common pipeline, Mahalanobis
   confidence, GLAD-PI zero-geometry twin, native-DAC wiring.
7. Run all 9 validation tests in §10 to green before any real extraction.
8. Execute Phase 0 (§12) — no new extraction, existing checkpoints only.
9. Execute Phase 1 (§12) — build + one smoke extraction (checkpoint 1 only),
   confirm storage-per-checkpoint estimate from §14 against actual output
   size, and specifically check whether a reusable penultimate-feature
   cache exists for Mahalanobis before extracting a fresh one.
10. Apply the Phase-0/Phase-1 stop/go check (§13) before proceeding.
11. Execute Phase 2 (§12) — the remaining 4 extraction jobs, full Study-A/B
    computation, variance decomposition, aggregation.
12. Apply the Phase-2 stop/go checks (§13).
13. Write results only into `results/studyAB/...` — never into any existing
    path. Use the artifact schema in §9 exactly, including explicit
    `null`s for inapplicable metrics per §6.
14. Only after verified results: follow §15 to add ResearchBrain
    `05_Experiments/` cards. Do not touch `06_Ideas/`.
15. Log any deviation from §1–14 above in §16 with a date, never by editing
    the frozen sections.

STOP. Do not implement anything beyond writing this file until it is
explicitly approved.
