# G1-DP specification — Amendment 1 (pre-outcome): tie-aware top-class agreement in the raw-H_3.22 extraction gate

Status: **FROZEN (2026-09-28)**, sidecar `g1_dp_spec_amendment_1.frozen.sha256`. Amends `docs/g1_dp_spec.md` (frozen v1, sha256
`0cbfb62e70125a8bff22f929ab62466b41de5a4d5d9ed8af6454cf859e4da825`, commit `096ed35`), which is preserved unchanged. Where they differ,
this amendment governs; everything not mentioned here is unchanged.

## 1. What changes

Spec §2, raw-tier gate, sub-criterion "argmax agreement ≥ 0.999 over every row of all 13 conditions" is replaced by:

> **Tie-aware top-class agreement ≥ 0.999** in every condition of both checkpoints, where for a row x with stored float16 probe output
> P_cache(x), T(x) = {c : P_cache,c(x) = max_j P_cache,j(x)} (compared as stored float16 values), and the row is top-class-consistent iff
> argmax_c P_rec,c(x) ∈ T(x).

Unchanged: max |P_rec − P_cache| ≤ 0.05; forward-logit consistency max |Δz| ≤ 1e-2; the representation definition (L2-normalized GAP of
`layer3.22`, 1024-d, strict FP32, batch 250); every arm, family, HPO rule, statistic, validity criterion (V1–V6), classification rule,
threshold and verdict rule of G1-DP; the fallback that drops the raw tier if the (amended) gate fails. Plain-argmax agreement is still
computed and reported.

## 2. Why (argmax is not uniquely defined on quantized ties)

P_3.22 is cached as float16. When two or more classes are equal after float16 rounding, "the argmax" of the cached vector is not unique —
numpy's first-index convention picks one arbitrarily — while the exact float64 reconstruction breaks the tie by sub-quantum differences.
A disagreement on such a row is a property of the cache's quantization, not evidence that the re-extracted representation differs.
The original rule therefore mis-scored tied rows; the tie-aware rule accepts any class the cache itself cannot distinguish and still
counts every disagreement on a non-tied row.

## 3. Evidence and timing (pre-outcome)

Trigger: extraction job 21726071 (checkpoint 4) failed the original rule (plain argmax agreement 0.9988 gaussian_noise_s5, 0.9989
defocus_blur_s5); all dependent jobs were cancelled by Slurm before starting. Label-free audit (`results/g1dp/h322/tie_aware_audit.json`):
all 23 disagreements in those cells, and all 82 (checkpoint 2) / 75 (checkpoint 4) plain-argmax disagreements across the 13 conditions,
occur on exact float16 top ties; zero disagreements on non-tied rows; tie-aware agreement 1.000 everywhere; max |ΔP| 0.0078 / 0.0152
(≤ 0.05); max |Δz| = 0. **No label, bundle, fit or G1-DP outcome existed or was inspected when this amendment was written.**
Authorization: researcher decision of 2026-09-28 (option B, tie-aware refinement).

## 4. Execution consequence

The extraction (with the amended gate) is re-run from a new immutable snapshot; only if it passes are the downstream G1-DP jobs
(bundles → six family arrays → aggregation) resubmitted, unchanged in design and resources.
