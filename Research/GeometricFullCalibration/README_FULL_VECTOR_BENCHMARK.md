# Unified Full-Vector Calibration Benchmark

This document covers the research benchmark in
`Experiments/run_unified_benchmark.py`. It is separate from the IJCAI paper
workflow: `README_IJCAI_REVIEWERS.md` remains the stable reproduction guide for
reviewers and is not replaced by this benchmark.

The unified runner loads one trained checkpoint, uses shared train/validation/test
splits, tunes methods on the validation split, and recomputes test metrics through
`utils.unified_metrics`. This keeps checkpoint, split, seed, and evaluation settings
aligned across methods. It does not train the checkpoint.

## Methods and outputs

### Always-run implemented comparisons

Every invocation evaluates:

- the uncalibrated base model;
- temperature, vector, one-vs-rest beta, one-vs-rest isotonic, and ODIR
  Dirichlet calibration;
- RGCL, RGCC, GC-DAC, and GC-TULIP;
- anchored model-tail and anchored rank-geometric-tail reconstructions; and
- the existing full-vector distance-fusion comparison.

These methods are implemented in the unified runner. Their inclusion here does
not make the unified benchmark part of the stable IJCAI reproduction path.

### Default-on research methods

All implemented method families now run by default, including the factorial
study. The historical `--enable_*` flags remain accepted for command
compatibility. Every method has a matching `--disable_*` flag, and
`--disable_all_optional_methods` restores a lightweight always-run-only job.

- full-vector geometric fusion runs the configured geometric score
  modes. The defaults are `neg_distance`, `margin`, `log_trust_ratio`, and
  `rank_log_trust`; override them with `--fvgf_score_modes` and optionally set
  `--fvgf_lambda_grid`.
- `--fusion_feature_source {single_layer,rgcl}` selects the feature source for
  full-vector geometric fusion, kNN blend, and KCal-lite. It has an effect only
  the corresponding methods.
- the softmax-kNN blend baseline runs by default.
- the KCal-lite top-k KDE-lite baseline runs by default;
  `--kcal_k_per_class` controls neighbors per class and defaults to `1`.
- full KCal runs by default. It learns a low-dimensional
  projection from penultimate training embeddings, selects an RBF bandwidth by
  stratified validation cross-validation, and uses every validation embedding as
  the final KDE reference set. Configure it with `--kcal_projection`,
  `--kcal_projection_dim`, `--kcal_projection_epochs`,
  `--kcal_references_per_class`, and `--kcal_bandwidth_folds`.
- RGCL-anchored tail variants run by default. Select tails
  with `--rgcl_tail_sources`.
- post-fusion temperature scaling runs by default after the
  existing full-vector distance fusion.

### Diagnostic and artifact-only outputs

These are analysis data, not additional benchmark methods:

- GC-DAC anchor confidence and top-prediction arrays;
- method ordering and whether each method can change the base argmax;
- per-method probability matrices and available geometry/distance arrays; and
- `--debug_rgcl_tail_hybrid_smoke`, which performs extra consistency checks when
  RGCL tail hybrids are enabled.

## Running the benchmark

All commands require a matching trained checkpoint under `--results_dir`
(`results/models` by default). Use a separate `--output_dir` for each run because
the filenames within a run directory are fixed.

### Base unified benchmark

```bash
python Experiments/run_unified_benchmark.py \
  --dataset cifar10 \
  --model resnet18 \
  --seed 11 \
  --output_dir results/unified_benchmark/cifar10/resnet18/seed11/base
```

### CIFAR-10 quick experiment

There is no dedicated `--quick` flag. For a lighter smoke experiment, reduce the
projection and sampling settings:

```bash
python Experiments/run_unified_benchmark.py \
  --dataset cifar10 \
  --model resnet18 \
  --seed 11 \
  --target_dimension 64 \
  --num_layers 2 \
  --num_coordinates 64 \
  --output_dir results/unified_benchmark/cifar10/resnet18/seed11/quick
```

This configuration is for pipeline verification; compare methods only across
runs that use the same settings.

### CIFAR-100 experiment

The example assumes the corresponding CIFAR-100 checkpoint already exists:

```bash
python Experiments/run_unified_benchmark.py \
  --dataset cifar100 \
  --model resnet18 \
  --seed 12 \
  --output_dir results/unified_benchmark/cifar100/resnet18/seed12/base
```

### Enable full-vector geometric fusion

```bash
python Experiments/run_unified_benchmark.py \
  --dataset cifar10 --model resnet18 --seed 11 \
  --enable_full_vector_geometric_fusion \
  --output_dir results/unified_benchmark/cifar10/resnet18/seed11/fvgf
```

### Compare stability-space metrics

Full-vector distance methods share `--stab_metric`. The whitened-cosine option
fits PCA whitening on the training/reference embeddings only, then evaluates
cosine distance in the whitened space:

```bash
python Experiments/run_unified_benchmark.py \
  --dataset cifar10 --model resnet50 --seed 11 \
  --stab_metric whitened_cosine \
  --whitening_components 128 \
  --output_dir results/unified_benchmark/cifar10/resnet50/seed11/whitened_cosine
```

Use separate output directories when comparing `l2`, `cosine`, and
`whitened_cosine`. `--whitening_eps` controls the PCA variance floor and
defaults to `1e-6`.

The SLURM launcher detects `--stab_metric` inside `--benchmark-extra-args` and
automatically suffixes non-L2 output directories (and job/log names) with the
metric. This preserves the legacy L2 path while preventing cosine variants from
overwriting it. For explicit layouts, use `{stab_metric}` in `--output-template`.
When the matching L2 output directory exists, non-L2 jobs also reuse its completed
metric-invariant/L2-only checkpoints and execute only methods whose outputs depend
on the selected stability metric. The launcher refuses a wasteful full rerun when
that source is missing unless `--allow-non-l2-full-rerun` is explicitly supplied.

### Enable the RGCL feature source

```bash
python Experiments/run_unified_benchmark.py \
  --dataset cifar10 --model resnet18 --seed 11 \
  --enable_full_vector_geometric_fusion \
  --fusion_feature_source rgcl \
  --output_dir results/unified_benchmark/cifar10/resnet18/seed11/fvgf_rgcl
```

The same feature-source flag can be paired with the kNN or KCal-lite opt-ins.

### Enable the kNN blend baseline

```bash
python Experiments/run_unified_benchmark.py \
  --dataset cifar10 --model resnet18 --seed 11 \
  --enable_knn_blend_baseline \
  --output_dir results/unified_benchmark/cifar10/resnet18/seed11/knn_blend
```

### Enable the KCal-lite baseline

```bash
python Experiments/run_unified_benchmark.py \
  --dataset cifar10 --model resnet18 --seed 11 \
  --enable_kcal_lite_baseline \
  --kcal_k_per_class 5 \
  --output_dir results/unified_benchmark/cifar10/resnet18/seed11/kcal_lite_k5
```

### Enable full KCal

```bash
python Experiments/run_unified_benchmark.py \
  --dataset cifar10 --model resnet18 --seed 11 \
  --enable_kcal_baseline \
  --kcal_projection skip_elu \
  --kcal_projection_dim 32 \
  --kcal_projection_epochs 50 \
  --kcal_bandwidth_folds 20 \
  --output_dir results/unified_benchmark/cifar10/resnet18/seed11/kcal
```

Full KCal does not blend its predictions with the base softmax. The learned
projection, selected bandwidth, and projected validation reference set are
checkpointed under `intermediates/kcal/calibrator.pt` for resumable runs.
The implementation follows the core algorithm from
[Lin, Trivedi, and Sun (ICLR 2023)](https://openreview.net/forum?id=p_jIy5QFB7)
and the authors' [reference code](https://github.com/zlin7/KCal).

### Run the controlled KCal factorial study

The factorial family isolates the components that differ between full KCal and
`kcal_lite_rgcl`: representation (learned Pi or RGCL), fusion, full versus
top-k KDE, validation versus training reference bank, and squared-L2 RBF versus
exponential-L2 kernel. The canonical `kcal` and `kcal_lite_rgcl` endpoint rows
are also emitted automatically; the latter uses `--kcal_factorial_k_per_class`.

Start with the six-row core study:

```bash
python Experiments/run_unified_benchmark.py \
  --dataset cifar100 --model resnet18 --seed 21 \
  --enable_kcal_factorial \
  --kcal_factorial_scope core \
  --kcal_projection_dim 32 \
  --output_dir results/unified_benchmark/cifar100/resnet18/seed21/kcal_factorial_core
```

Use `--kcal_factorial_scope full` (the default) for all 48 controlled rows:
16 geometry cells times `replace`, `blend_frozen`, and `blend_joint`.
`blend_frozen` reuses the replacement-selected posterior exactly and therefore
isolates fusion. `blend_joint` jointly selects kernel strength and alpha and is
the practical analogue of KCal-lite tuning.

RGCL embeddings and learned-Pi embeddings are persisted under
`intermediates/kcal_factorial_features/` and reused by resumptions. Completed
factorial rows are checkpointed immediately. Full scope is substantially more
expensive because it includes full KDE against the complete training bank; use
the core scope as a smoke test before submitting full multi-seed jobs.

### Run all methods with RGCL geometry

```bash
python Experiments/run_unified_benchmark.py \
  --dataset cifar10 --model resnet18 --seed 11 \
  --kcal_k_per_class 5 \
  --fusion_feature_source rgcl \
  --output_dir results/unified_benchmark/cifar10/resnet18/seed11/all_methods_rgcl
```

No method-enabling flags are required. Run a second invocation with
`--fusion_feature_source single_layer` and a different output directory to
collect the single-layer variants. For a targeted lightweight run, begin with
`--disable_all_optional_methods` and then use a separate output directory.

## Output artifacts

Each `--output_dir` contains:

- `summary_metrics.json`: benchmark metadata, method settings, and structured
  metrics;
- `summary_metrics.csv`: a tabular summary for cross-method comparison;
- `per_sample/per_sample_arrays.npz`: aligned per-sample arrays, including test
  labels, base probabilities, GC-DAC anchor confidence and top prediction,
  method index/argmax metadata, per-method probability matrices, and geometry or
  distance arrays when available; and
- `per_sample/per_sample_manifest.json`: the companion schema description. It
  records the schema version, array names and shapes, method metadata, feature
  sources, mappings from methods to distance arrays, and geometry fields that are
  unavailable for a method.

Use the manifest rather than inferring optional NPZ keys. Missing geometry entries
mean that the runner did not produce that field for the method; they are not
zero-valued measurements.

## Recommended experiment order

1. Run the reduced CIFAR-10 command to verify the checkpoint and pipeline.
2. Run the base benchmark with the intended dimensions and seed.
3. Disable individual expensive families when running targeted diagnostics.
4. Compare RGCL and single-layer feature sources under otherwise aligned settings.
5. Run the complete default-on command, then repeat on CIFAR-100 or additional seeds.

## Known limitations

- The runner requires a pre-existing checkpoint and does not provide a `--quick`
  preset.
- Reduced dimensions are useful for smoke testing but are not directly comparable
  with default-setting runs.
- RGCL feature extraction and per-class neighbor distances can be costly in time
  and memory, especially on CIFAR-100.
- One feature source is selected per invocation; use separate run directories to
  compare RGCL and single-layer variants.
- Full KCal is substantially more expensive than KCal-lite because projection
  training and bandwidth cross-validation evaluate full-reference kernels.
- KCal-lite remains an adapted top-k/blended baseline and should not be reported
  as the full KCal method.
- Geometry artifacts are method-dependent; consult the manifest for explicit
  mappings and missing fields.

No benchmark results are asserted by this guide; it documents how to produce and
inspect runs.
