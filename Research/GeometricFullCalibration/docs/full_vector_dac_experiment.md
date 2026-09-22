# Full-Vector Density-Aware Calibration (FV-DAC) — experiment specification

**Status:** preregistered 2026-09-20; **stopped 2026-09-20 by its own frozen
continuation rule (Outcome E).** See §19 for results. Nothing in §§1–18 was
edited after results were opened.
**Vault:** `ResearchBrain/06_Ideas/Full-Vector Density-Aware Calibration.md`,
`ResearchBrain/04_Hypotheses/H-FVDAC-01 …`,
`ResearchBrain/05_Experiments/2026-09-20 Full-Vector DAC POC.md`.

> This document is a **new study specification**. It does not edit, reinterpret
> or retro-fit `BENCHMARK_IMPLEMENTATION_PLAN.md` or any frozen Phase 0/1
> design, and FV-DAC is not part of the canonical unified benchmark. If and
> only if a formulation is frozen and validated may it later be integrated
> there as a single method.

---

## 1. Question

> Can class-conditioning DAC's representation-density machinery turn a
> calibration-only mechanism into useful full-vector decision correction under
> distribution shift?

Secondary, kept strictly separate throughout:

> If useful class-conditioned reference geometry exists, is internal hidden
> representation geometry necessary, or can output-space geometry explain the
> same effect?

---

## 2. Verified native DAC baseline

Read directly out of the frozen Phase 0/1 state
`results/studyAB/phase0/fitted_method/checkpoint_seed4/native_dac.pkl`
(2026-09-20), not from documentation:

| # | item | verified value |
|---|---|---|
| 1 | selected layers | `conv1`, `layer1`, `layer2`, `layer3`, `layer4` — **5 layers, logits NOT included** |
| 2 | pooling | forward hook does `adaptive_avg_pool2d(feat, (1,1))` for 4-D activations (mean over tokens for 3-D) |
| 3 | L2 normalization | yes, `F.normalize(p=2, dim=-1)`, banks stored already normalized (verified `‖row‖ = 1.000000`) |
| 4 | distance convention | true Euclidean `sqrt(clamp(2 − 2·cos, min=0))` |
| 5 | `k` for CIFAR-100 | 200 (`get_dac_k_value`) |
| 6 | reference bank | clean **train** split, 45 000 rows; dims 64 / 256 / 512 / 1024 / 2048 |
| 7 | FAISS? | **no** — pure PyTorch. The squared-vs-true-distance trap that FAISS's L2 index creates does not arise. Flagged in `normalized_euclidean`'s docstring for any future port. |
| 8 | layer statistic | `s_l(x)` = distance to the k-th nearest bank point (`torch.kthvalue`) |
| 9 | aggregation | `S(x) = max(Σ_l w_l s_l(x) + w_0, 1e-12)` |
| 10 | fitting objective | summed squared error (Brier-style), L-BFGS-B, `tol=1e-12` |
| 11 | fitted weights (seed 4) | `w = [0.244625, 0.181287, 0.412497, 0.000000, 0.123806]` |
| 12 | intercept / positivity | `w_0 = 0.885362`; `w_l ≥ 0` box constraint, `w_0` free; output clipped at `1e-12` |
| 13 | final transform | `softmax(z / S(x,w))` |
| 14 | downstream calibrator `h` | **none appended** in this repository |
| 15 | bank labels preserved | **no** |
| 16 | query representations cached | **no** (only RGCL/penultimate features are cached, not DAC layers) |
| 17 | neighbour IDs retained | **no** — `kthvalue`'s indices are discarded |

Note `w_3 = 0` exactly: `layer3` contributes nothing to the primary arm's
`α`. `fv_dac_shared_layer` is the arm that can revive it.

### 2.1 Discrepancies found — documented, not silently fixed

1. **Layer count — VERIFIED AGAINST THE PAPER (2026-09-20, second audit).**
   Table 6 / Appendix C.1 of the PDF was extracted directly this time (raw
   `FlateDecode` stream extraction; `pdftotext`/`pypdf` are unavailable on
   these nodes). Its CIFAR-100 rows read verbatim:

   > `RESNET18  PRE-BLOCK, BLOCK-1,...,BLOCK-4, LOGITS`
   > `RESNET152 PRE-BLOCK, BLOCK-1,...,BLOCK-4, LOGITS`

   and the caption explicitly defines `LOGITS` as one of the layer sources.
   **The paper's ResNet layer set therefore includes the logits layer, and
   this repository's `native_dac` omits it — 5 sources instead of 6.** That
   is a genuine code-vs-paper divergence, now confirmed rather than inferred.

   **Chosen canonical baseline: the 5-layer benchmark version**, because it
   is what the frozen state contains, what every Phase 0/1 result used, and
   what `β = 0` must reproduce. The original implementation is untouched and
   this study adds no "fix".

   **Consequence for this experiment's framing.** FV-DAC must be described as
   *an extension of the benchmark's frozen five-layer DAC implementation*,
   **not** as an extension of the paper's complete layer set. It also
   complicates the hidden-vs-output contrast: the paper's own DAC already
   uses the logits layer as a density source, so `fv_dac_logit_space` is not
   a representation the published method excludes — it is one this
   repository's DAC happens to have dropped.
2. **Input normalization — my first description of this was WRONG, corrected
   here.** The original text said "CIFAR-100-C uses ImageNet statistics while
   the clean path uses CIFAR statistics". That conflated *train/val* with
   *test*. The actual repository state (`data/cifar100.py`):

   | loader | function | normalization |
   |---|---|---|
   | train + validation (and model training) | `get_train_valid_loader`, line 51 | **CIFAR** 0.4914/0.4822/0.4465, std 0.2023/0.1994/0.2010 |
   | clean **test** | `get_test_loader`, line 145 | **ImageNet** 0.485/0.456/0.406, std 0.229/0.224/0.225 |
   | CIFAR-100-C | `run_unified_benchmark.py` / `_corruption_transform` | **ImageNet** (same as clean test) |

   So the corruption path is *consistent with the benchmark's own clean test
   path*; it is not a corruption-specific quirk. The real inherited issue is
   a **train/test input-normalization mismatch affecting the entire
   benchmark**, clean and corrupted alike, and every method and every prior
   Phase 0/1 number computed in it. The checkpoints were trained through
   `get_train_valid_loader`, i.e. on CIFAR-normalized inputs, and are
   evaluated on ImageNet-normalized inputs (the std differs by ~13%).

   **Why this matters more for FV-DAC than for most methods.** The reference
   bank is built from the *train* split (CIFAR-normalized) while every query
   — clean test and corrupted — is ImageNet-normalized. Every native-DAC
   `s_l(x)` and every FV-DAC `r_{l,k}(x)` in this study is therefore a
   distance computed *across* that mismatch. A method that compares query
   features to a clean reference bank is precisely the kind most exposed to
   it.

   **Not fixed, not re-run.** The comparison here remains internally matched
   (all arms see identical inputs, and our `native_dac` rows reproduce the
   frozen Phase 0/1 rows to 4 dp), so the stopping decision stands for *this*
   benchmark configuration. But the configuration is a **compound shift**
   (corruption + normalization mismatch), not the standard CIFAR-100-C
   protocol, and the conclusion should be read as applying to it.
   Any corrected-normalization run must be a **separately preregistered
   replication of all affected methods**, never an FV-DAC rescue run.
3. **Reference-bank row order, and TF32 non-reproducibility.** The
   benchmark's train loader uses `SubsetRandomSampler`, so the pickled bank's
   row order does **not** correspond to
   `intermediates/splits/train_labels.npy`.

   The first attempt re-extracted its own labelled bank and tried to assert
   set identity by a content hash of the sorted rows. That failed, and the
   failure was informative: the measured max nearest-match distance at
   `conv1` was **9.1e-4**, with 11 of 45 000 rows colliding onto a shared
   nearest neighbour. That magnitude is **TF32 convolution non-reproducibility**
   (`torch.backends.cudnn.allow_tf32` defaults to `True`; 10-bit mantissa,
   plus batch-size-dependent algorithm selection), not a different data set.

   This matters scientifically, not just operationally: a bank perturbed at
   1e-3 moves the k-th-neighbour distance, hence `S(x)`, hence the
   probabilities — so `β = 0` would reproduce native DAC only *approximately*,
   destroying the matched operating point this experiment depends on.

   **Resolution: the frozen bank is used unchanged.** The re-extraction is
   used only to recover *which label belongs to which frozen row*
   (`_label_frozen_bank`), by a sparse global minimum-cost bijection over the
   eight nearest candidates in the equal-weight concatenation of all five
   normalized DAC layers, subject to four fatal
   checks:

   - the map is a **full bijection** — its candidates are the eight nearest
     materialized rows for each frozen row, and equal cardinalities make this
     a global, one-to-one correspondence rather than a greedy match;
   - its max distance is ≤ `--bank_match_tolerance` (default **5e-3**).
     This was revised before any result was opened after the multi-layer
     mapping measured a 3.435e-3 TF32 perturbation; the original 1e-3 number
     came from `conv1` alone and did not transfer to the concatenation;
   - that max is ≤ 10 % of `min_second_nearest_distance`, so "the match
     distance is small" is calibrated against how far apart *distinct* bank
     rows actually are rather than merely asserted;
   - the **same permutation** matches within tolerance at every individual layer,
     which rules out a coincidental re-pairing.

   The logit-space control's own bank is reordered by the same permutation, so
   row `j` of every bank is the same example as `bank_labels[j]`.

   Consequence: `S_DAC` is computed from the frozen bank and is native DAC's
   own temperature, and `β = 0` reproduces native DAC to float precision —
   re-asserted on every cell at evaluation time.

---

## 3. Primary formulation

For every selected DAC layer `l` and class `k`:

```
B_{l,k}    = { h_l(x_i) : y_i = k }
r_{l,k}(x) = K_c-th nearest-neighbour distance( h_l(x), B_{l,k} )
α_l        = ŵ_l / Σ_j ŵ_j            (uniform 1/L iff Σ_j ŵ_j ≤ 0; flagged)
R_k(x)     = Σ_l α_l · r_{l,k}(x)
```

```
q_β(x) = softmax( ( z(x) − β·R(x) ) / S_DAC(x) ),    β ≥ 0
```

Same representations, same pooling, same normalization, same distance
operator, same reference-bank membership, same frozen `S_DAC`. Only the
search domain becomes class-conditioned.

**Sign convention:** smaller class-conditioned distance → less penalty → more
support for that class.

**Required properties** (each has a test):

- `β = 0` reproduces native DAC *numerically*, re-checked on **every** cell at
  evaluation time, not once at implementation time;
- exactly one new free scalar in the primary arm;
- class-symmetric — no per-class bias, no per-class scaling, no MLP, no gate;
- gauge invariant — a common logit shift changes nothing;
- it *can* change the argmax.

### 3.1 Rejected design alternative — per-class temperature

`q_k ∝ exp(z_k / T_k(x))` is deliberately **not** the primary form. It breaks
logit gauge invariance (the test asserts this explicitly), can promote
strongly negative logits as `T_k` grows, makes the result depend on an
arbitrary logit origin, and is hard to interpret. Recorded here and in the
idea note as a rejected alternative, not as an untried option.

---

## 4. Arms

| identity | form | fitted params | role |
|---|---|---|---|
| `fv_dac_nll` | `softmax((z − βR)/S)`, β by NLL | 1 | **primary** |
| `fv_dac_brier` | same, β by squared error | 1 | predeclared objective sensitivity (native DAC is Brier-fitted) |
| `fv_dac_lognorm` | `v_{l,k} = −(log r_{l,k} − μ_{l,k})/σ_l`; `softmax((z + βV)/S)` | 1 (+ frozen μ[L,C], σ[L]) | predeclared conditioning sensitivity A |
| `fv_dac_shared_layer` | `softmax((z − Σ_l b_l r_{l,·})/S)`, `b_l ≥ 0` shared across classes | L = 5 | predeclared sensitivity B |
| `fv_dac_permuted` | primary with seeded permuted bank labels | 1 | negative control |
| `fv_dac_logit_space` | primary operator on centered, L2-normalized logits; `S_DAC` unchanged | 1 | mechanism control |
| `fv_dac_density_only` | `argmin_k R_k(x)` | 0 | diagnostic only |

Sensitivity arms are **predeclared scientific arms**, never candidates to be
promoted to primary after seeing results.

---

## 5. Split roles

```
clean train (45 000)            -> reference bank (features AND labels)
clean validation inner-FIT      -> fit β / b_l / μ / σ
clean validation inner-SELECT   -> select K_c          (disjoint from the above)
clean test (10 000)             -> evaluation
CIFAR-100-C                     -> evaluation only, frozen state
```

Implemented with the repository's existing
`utils.decision_audit.make_inner_validation_split(select_fraction=0.5,
seed=123)`. **No benchmark split is redefined.** There was enough disjoint
clean data to implement the required discipline without touching the
established split semantics, so the "stop and report a conflict" branch of the
brief did not trigger.

**Declared deviation.** Native DAC's own weights were fitted — in the frozen
Phase 0/1 run — on the *whole* validation split, including what is now the
inner-select half. Those weights are frozen and identical across all `K_c`
candidates, so they cannot bias the `K_c` comparison; but the inner-select NLL
is not perfectly independent of them. Recorded in the frozen state under
`split_roles.declared_deviation`.

---

## 6. Fitting objective

```
β̂ = argmin_{β ≥ 0} NLL(q_β, y)      on clean inner-FIT
```

NLL is preferred because it is a proper full-vector scoring rule, β is
one-dimensional, and accuracy stays an *evaluation outcome* rather than a
fitting target. The objective is **never** accuracy and **never** net flips —
`fit_beta`'s body is asserted (by AST inspection, docstring stripped) to
contain neither `accuracy`, `flip` nor `argmax`, and `_objective_fn` rejects
anything but `nll`/`brier`.

`fv_dac_brier` is the predeclared Brier sensitivity, because native DAC itself
is Brier-fitted. It is not an alternative primary.

Search: a deterministic 62-point grid (`0` plus `geomspace(1e-3, 1000, 61)`)
followed by a bounded local refinement in the winning bracket. The frozen
state records `at_lower_boundary` / `at_upper_boundary` so a pinned β is
visible rather than silent.

---

## 7. `K_c` selection

Predeclared set: **`K_c ∈ {5, 20, 200}`** — strongly local / intermediate /
literal same-K extension. CIFAR-100's bank holds ≈ 450 examples per class, so
`K_c = 200` is valid; native DAC's global `K` is also 200. A `K_c` larger than
the *smallest* class bank is a hard error.

Rule: for each `K_c`, fit β on inner-FIT; compare the frozen candidates by NLL
on the disjoint inner-SELECT; take the argmin. **Primary arm only** — every
other arm inherits the selected `K_c`. Corruption data is never consulted.

**This grid will not be expanded if results are poor.** No `{2, 10, 50, 100}`
follow-on without a new scientific justification and a separate experiment
note. There is deliberately no CLI flag that can widen it (asserted by a test).

---

## 8. Controls and comparisons

| comparison | question | arms |
|---|---|---|
| A | does FV-DAC improve decisions? | base / native DAC / FV-DAC |
| B | is it real class geometry? | FV-DAC vs `fv_dac_permuted` |
| C | is hidden representation necessary? | FV-DAC vs `fv_dac_logit_space` |
| D | is it ordinary class-wise calibration? | vs Vector Scaling, ODIR/Dirichlet |
| E | is it nearest-neighbour classification? | vs KCal, vs `fv_dac_density_only` |
| F | are DAC's layer weights sufficient? | primary vs `fv_dac_shared_layer` |

Temperature Scaling / Vector Scaling / ODIR / KCal are **read from the frozen
Phase 0/1 artifacts** for the matching cell (`reused_phase0_baselines` in each
metrics file), never recomputed here — recomputing a method the canonical
benchmark owns risks producing a second, subtly different number for it.

### 8.1 Capacity caveat for comparison C

`fv_dac_logit_space` is a **mechanism control, not a strictly capacity-matched
control**. Every cell's metrics file records, for both arms:
`n_sources`, `source_names`, `dims`, `bank_size`, `kc`, `n_fitted_params`.

| arm | sources | dims | fitted params |
|---|---|---|---|
| hidden | 5 | 64 / 256 / 512 / 1024 / 2048 | 1 |
| logit-space | 1 | 100 | 1 |

`hidden > logit-space` must therefore **not** be read as definitive evidence
for hidden-specific geometry without acknowledging this asymmetry. Conversely,
`hidden ≤ logit-space` kills only H-FVDAC-01-H, never the method-level
hypothesis.

---

## 9. Fit-once / evaluate-many

```
clean train/reference -> clean calibration fit -> clean validation selection
   -> write complete frozen state -> reload frozen state -> clean test
   -> CIFAR-100-C evaluation
```

Corruption evaluation contains **no** `.fit()` path:

- `--stage fit` rejects `--corruption_type` outright;
- `--stage evaluate` installs `_FitGuard` **before** the corrupted loader is
  constructed (asserted by a source-order test), and every fitting entry point
  goes through `guarded_fit_beta` / `guarded_fit_shared_layer`;
- a missing frozen state is a `FileNotFoundError`, never a silent fit;
- the recorded state hash is re-verified on load; a mismatch aborts.

Frozen state (`results/fv_dac/fitted_state/checkpoint_seed{S}/fv_dac_state.json`,
hashed into `fv_dac_state_hash.json`) carries at least: native DAC weights,
intercept, β per arm, `b_l`, `K_c`, layer set, `α`, μ/σ, reference-bank
feature hashes, order-invariant bank set hashes, bank-label hash, permuted-label
seed and hash, logit-space representation definition, preprocessing definition,
checkpoint identity, fitting objective, variant identity, split roles and the
declared deviation.

---

## 10. Metrics

Per cell and method: accuracy, Δaccuracy vs base, NLL, Brier, top-label ECE,
adaptive ECE, classwise ECE, AUROC, AURC (via the benchmark's own
`utils.unified_metrics.evaluate_all`), plus the benchmark's
`utils.decision_audit.decision_audit`.

Flip decomposition, per decision-changing method:

```
W = #(wrong -> correct)       H = #(correct -> wrong)
U = #(wrong -> different wrong)       F = W + H + U
ΔAccuracy = (W − H) / N          <- verified numerically every cell
intervention precision = W / F
decisive precision     = W / (W + H)
fraction of base errors repaired = W / #(base errors)
flip rate by base top-2 margin (5 equal-mass bins)
flip rate by native DAC S(x)   (5 equal-mass bins)
GT rank before / after
destination-class entropy and top-1 share
```

Per-sample arrays (`fv_dac_per_sample.npz`): labels, base logits, `S(x)`, base
prediction, base top-2 margin, GT rank per method, per-method predictions,
per-method probabilities (fp16), density-only prediction, and the `R` summary
(`R_min`, `R_argmin`, `R_at_base_pred`, `R_at_label`).

---

## 11. Continuation rule — frozen 2026-09-20, before any CIFAR-C FV-DAC result

Applied to the predeclared seed-4 12-cell POC (`gaussian_noise`,
`defocus_blur`, `fog`, `jpeg_compression` × severity 1/3/5).

**Continue to full corruption extraction iff ALL hold:**

- **C1 — β alive.** Selected primary `β̂ > 0`, not at the search boundary.
- **C2 — direction.** Mean ΔAcc over the 12 cells `> 0` and `W > H` in
  aggregate.
- **C3 — consistency.** ≥ 7/12 cells with `ΔAcc ≥ 0`, ≥ 5/12 with `ΔAcc > 0`,
  and removing any single corruption family still leaves mean `ΔAcc > 0`.
- **C4 — control.** `fv_dac_permuted` mean ΔAcc ≤ ½ × `fv_dac_nll` mean ΔAcc,
  or nonpositive.
- **C5 — probabilistic health.** Aggregate corruption NLL no worse than
  `native_dac` by more than 0.05, and top-label ECE no worse by more than
  0.02 absolute.

**Shared-layer branch:** if C1–C3 fail for `fv_dac_nll` but hold for
`fv_dac_shared_layer` with C4/C5 satisfied for that arm, continue in that mode
only, report it as Outcome B, and do **not** promote it to primary.

**Otherwise stop** and classify as currently negative / inconclusive — in
particular if `β̂ = 0` for both the primary and shared-layer arms, or if
`W ≤ H` across the 12 cells for every arm.

This rule will not be edited after the 12-cell results are opened.

---

## 12. Kill criteria

**Method-level** (weaken/kill H-FVDAC-01): β collapses to zero across most
checkpoints; `W ≤ H` under shift; mean corruption ΔAcc negligible or
nonpositive; the permuted control reproduces the effect; Vector Scaling /
Dirichlet explain it without geometry; KCal / density-only explain it so
completely that FV-DAC adds nothing distinct; gains confined to one seed or one
corruption; calibration damage substantial relative to native DAC; the fitted
correction is pathological on clean validation; the effect disappears after
frozen-state replication.

**Hidden-representation-specific** (weakens H-FVDAC-01-H *only*):
`fv_dac_logit_space` matches or beats the hidden arm. `FV-DAC_Z > FV-DAC_H >
Base` would falsify the hidden-specific claim while leaving the method-level
hypothesis alive.

No new success criteria may be invented after seeing results.

---

## 13. Outcome categories

- **A** — class-conditioned DAC extension succeeds (positive net flips,
  meaningful positive corruption ΔAcc, calibration largely preserved, permuted
  control near null, ordinary vector calibrators do not explain it, replicates
  across seeds). Interpretation: *class-conditioning DAC's representation-density
  machinery enables useful decision correction unavailable to native
  scalar-temperature DAC.* **Not** "scalar bottleneck confirmed" — the
  extension also introduces reference-label conditioning that native DAC never
  used.
- **B** — DAC machinery useful but its calibration-optimal layer weights are
  not decision-optimal (primary weak/null, shared-layer clearly positive).
- **C** — reference geometry works but the hidden-specific claim fails
  (logit-space matches or beats hidden while one or both beat base).
- **D** — kNN/prototype explanation (density-only and/or KCal match the fused
  method).
- **E** — calibration/decision trade-off (accuracy up, NLL/Brier/ECE collapse):
  decision fusion, not a successful calibration extension.
- **F** — negative.

Increasingly flexible methods must not be added merely to avoid F.

---

## 14. Execution plan

| step | what | artifact |
|---|---|---|
| 1 | `pytest -q`, `git diff --check` | must stay green |
| 2 | seed-4 clean smoke | `results/fv_dac/evaluation/checkpoint_seed4/clean/` |
| 3 | seed-4 clean full table | same |
| 4 | continuation rule frozen | this file §11 + the vault note (both written before any CIFAR-C run) |
| 5 | seed-4 12-cell POC | `scripts/fv_dac_evaluate_corruption.sbatch`, default corruption list |
| 6 | apply §11 | — |
| 7 | remaining 11 corruptions (only if §11 is satisfied) | `FVDAC_CORRUPTIONS="…"` |
| 8 | seeds 1, 2, 3, 5 | identical configuration-selection rules per checkpoint |

Severities 1/3/5 are the prespecified primary reporting grid.

### Slurm

`scripts/fv_dac_fit_clean.sbatch` (array = checkpoint seed) and
`scripts/fv_dac_evaluate_corruption.sbatch` (array indexes
checkpoint × corruption × severity; corruption list and seed list come from
`FVDAC_CORRUPTIONS` / `FVDAC_SEEDS`).

Both are **new files**. `scripts/phase0_1_*.sbatch` are untouched, so
historical Phase 0/1 behaviour is unchanged.

Operational choices, inherited from the working Phase 0/1 scripts rather than
re-derived: partition `rtx4090`, `--gres=gpu:rtx_4090:1`, interpreter
`/home/itayab/.conda/envs/geo_cuda12/bin/python` (Python 3.10.16, PyTorch
2.3.1+cu121). The CUDA/PyTorch environment is **not** changed mid-experiment,
and no GPU extraction runs on a login node. `--device cuda` fails fast rather
than silently degrading to CPU.

---

## 15. Artifact paths

```
results/fv_dac/fitted_state/checkpoint_seed{S}/fv_dac_state.json
results/fv_dac/fitted_state/checkpoint_seed{S}/fv_dac_state_hash.json
results/fv_dac/fitted_state/checkpoint_seed{S}/bank/layer{l}.npy
results/fv_dac/fitted_state/checkpoint_seed{S}/bank/bank_labels.npy
results/fv_dac/fitted_state/checkpoint_seed{S}/bank/bank_labels_permuted.npy
results/fv_dac/fitted_state/checkpoint_seed{S}/bank/bank_logit_repr.npy
results/fv_dac/evaluation/checkpoint_seed{S}/{cell}/fv_dac_metrics.json
results/fv_dac/evaluation/checkpoint_seed{S}/{cell}/fv_dac_per_sample.npz
results/fv_dac/aggregate/                      (Experiments/aggregate_fv_dac.py)
logs/fv_dac/
```

Nothing under `results/studyAB/` is written or overwritten by this study.

---

## 16. Code map

| file | role |
|---|---|
| `Calibrators/full_vector_dac.py` | the method: operator, aggregation, arms, objectives, frozen state |
| `Experiments/run_fv_dac_experiment.py` | standalone runner (`--stage fit` / `--stage evaluate`) |
| `Experiments/aggregate_fv_dac.py` | per-cell → per-seed → cross-seed aggregation |
| `tests/test_full_vector_dac.py` | the 22 preregistered properties |
| `utils/method_metadata.py` | **additive** registry rows for the six FV-DAC arms |
| `scripts/fv_dac_*.sbatch` | Slurm |

`Experiments/run_unified_benchmark.py` is **byte-identical to HEAD** — a test
asserts this, and separately asserts that every pre-existing registry entry's
semantics are unchanged against `git show HEAD:utils/method_metadata.py`. That
is the regression evidence for §25.21/22; no shared-utility refactor was
performed, because none was needed.

---

## 17. Preregistered test list (§25)

1. `β = 0` reproduces native DAC probabilities
2. native DAC remains argmax-invariant
3. FV-DAC can change the argmax on a synthetic example
4. a favourable class distance moves that class in the expected direction
5. a common logit shift does not change FV-DAC's output
6. class-relabelling equivariance
7. permuted bank labels are deterministic by seed (and count-preserving)
8. no reference-label leakage into base / native DAC
9. corruption evaluation cannot call fit
10. frozen-state round-trip reproduces predictions
11. `K_c` bounds valid for every class
12. all probability vectors finite and sum to 1
13. flip accounting identity `ΔAcc = (W − H)/N`
14. method metadata marks FV-DAC full-vector / decision-changing
15. required state/provenance hashes include bank identity and labels
16. the logit-space control uses the intended centered-logit representation
17. the shared-layer variant is class-symmetric
18. no per-class trainable parameters enter the primary method
19. β fitting and `K_c` selection use disjoint split roles
20. corruption evaluation cannot influence `K_c`, β, variant selection or the
    continuation rule
21. FV-DAC reuses shared benchmark utilities rather than forking them
22. canonical benchmark behaviour is unchanged (byte-identical runner +
    unchanged pre-existing registry semantics)

---

## 18. Things this study will not do

Refit on corruptions; choose `K_c` from CIFAR-C; expand the `K_c` grid after
poor results; tune β for accuracy; use the β-fit split to select `K_c`;
adaptively promote a sensitivity variant to primary; redefine repository
splits; invent the continuation rule after seeing CIFAR-C; add an MLP,
per-class learned biases, class-specific layer weights or an adaptive gate;
change the backbone; retrain the classifier; change the CUDA environment;
rewrite historical RGCL/DAC artifacts; overwrite prior result files; delete
negative results; treat this as an RGCL rescue; read `hidden > logit-space` as
definitive hidden-specific evidence without the capacity caveat; read
`hidden ≤ logit-space` as killing the broader hypothesis; or rewrite the
canonical unified benchmark as a second FV-DAC-specific benchmark.


---

## 19. Results and outcome (appended 2026-09-20; §§1–18 unedited)

> **Revised 2026-09-20 after an adversarial re-audit of the artifacts.** The
> first write-up of this section contained three reporting defects, all
> corrected below and listed here so the correction is not silent:
> (i) it omitted `U` (wrong→different-wrong) and therefore quoted *decisive*
> precision where *intervention* precision was the relevant number;
> (ii) it omitted the `fv_dac_brier` arm entirely, although that arm was
> fitted and evaluated in all 12 cells;
> (iii) two means were mis-transcribed (`lognorm` +0.00124 → **+0.00090**,
> `logit_space` +0.00055 → **+0.00060**).
> It also overstated the conclusion as "the mechanism is real, not an
> artefact". See §19.7.

Checkpoint **seed 4 only**. Authoritative artifacts:
`results/fv_dac/aggregate/{per_cell.csv, continuation_rule.json, summary.json}`.
Frozen state hash `351423c41f53b307c09136c1bbd1cb3112367c72b55e47b3720f4ef46f791ae1`.

### 19.1 Structural checks — all passed

| check | result |
|---|---|
| `β = 0` reproduces native DAC | **0.0** exact, re-verified on every cell |
| native DAC reproduced from the labelled frozen bank | 7.9e-7, argmax agreement 1.000 |
| our `native_dac` rows vs frozen Phase 0/1 rows | 4 dp agreement (`gaussian_noise_s3` NLL 3.8860 both) |
| corrupted cell bytes vs benchmark's materialized split | identical |
| flip identity `ΔAcc = (W−H)/N` | holds, every method, every cell |
| corruption runs evaluation-only | `evaluation_only: true` |

### 19.2 Reference-bank labelling — an unplanned dataset finding

Full bijection (max match 3.435e-3 in the concatenated multi-layer space,
every per-layer cross-check inside tolerance), but 14 of 45 000 frozen rows
had a plausible match carrying a **different** label. Pixel diagnosis: those
14 conflict pairs have raw pixel distance **exactly 0** against a random-pair
median of **96.07** — **bit-identical duplicate images with different labels
inside CIFAR-100's own train split**. Impact bounded at ≤2·tolerance in one
member of ~450; gated by a 0.5 % ceiling (actual 0.031 %) plus a requirement
that every conflict pair *be* a duplicate image.

### 19.3 `K_c` selection (clean inner-select NLL, primary arm only)

| `K_c` | β | inner-fit NLL | inner-select NLL | select acc |
|---|---|---|---|---|
| **5** | **6.4324** | 0.87914 | **0.87283** | 0.7696 |
| 20 | 4.6367 | 0.88465 | 0.87780 | 0.7684 |
| 200 | 2.2283 | 0.89483 | 0.88698 | 0.7664 |

(`β = 0` inner-fit NLL 0.90329.) Selected `K_c = 5`, no boundary hits.

### 19.4 Clean test (seed 4, 10 000 samples) — complete, all arms

| method | acc | ΔAcc | NLL | Brier | ECE | W | H | U |
|---|---|---|---|---|---|---|---|---|
| base_model | 0.7631 | — | 0.9378 | 0.3381 | 0.0489 | 0 | 0 | 0 |
| native_dac | 0.7631 | +0.0000 | 0.9472 | 0.3334 | **0.0268** | 0 | 0 | 0 |
| fv_dac_nll | 0.7644 | +0.0013 | 0.9242 | 0.3357 | 0.0515 | 48 | 35 | 64 |
| **fv_dac_brier** | 0.7643 | +0.0012 | 0.9350 | 0.3333 | 0.0299 | 20 | 8 | 19 |
| fv_dac_lognorm | 0.7658 | +0.0027 | 0.9282 | 0.3345 | 0.0441 | 58 | 31 | 74 |
| fv_dac_shared_layer | 0.7666 | +0.0035 | 0.8822 | 0.3314 | 0.0657 | 118 | 83 | 129 |
| fv_dac_permuted | 0.7631 | +0.0000 | 0.9472 | 0.3334 | 0.0268 | 0 | 0 | 0 |
| fv_dac_logit_space | 0.7646 | +0.0015 | 0.9302 | 0.3377 | 0.0583 | 23 | 8 | 15 |

Frozen Phase 0/1 baselines, same cell: `temperature_scaling` 0.7631,
`vector_scaling` 0.7659, `odir_dirichlet` 0.4038, `kcal` 0.7611.
Density-only diagnostic: acc 0.7285, agreement with base 0.844, with FV-DAC
0.854, recoverability among base errors 0.121.

`fv_dac_shared_layer` fitted `b = [0, 0, 0, 26.09, 0]` — all weight on
`layer3`, the one layer native DAC assigned weight exactly **zero**.

### 19.5 Corruption aggregates over the 12 cells — complete flip decomposition

| arm | mean ΔAcc | cells>0 | total flips F | W | H | **U** | net W−H | decisive W/(W+H) | **intervention W/F** | argmax-change | base errors repaired |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **fv_dac_nll** | **+0.00142** | 9/12 | 3518 | 653 | 483 | **2382** | +170 | 0.575 | **0.186** | 2.93 % | 1.08 % |
| fv_dac_brier | +0.00061 | 10/12 | 1136 | 225 | 152 | 759 | +73 | 0.597 | 0.198 | 0.95 % | 0.37 % |
| fv_dac_lognorm | +0.00090 | 9/12 | 3726 | 707 | 599 | 2420 | +108 | 0.541 | 0.190 | 3.11 % | 1.17 % |
| fv_dac_shared_layer | −0.00113 | 6/12 | 9938 | 1558 | 1693 | 6687 | −135 | 0.479 | 0.157 | 8.28 % | 2.59 % |
| fv_dac_logit_space | +0.00060 | 10/12 | 955 | 211 | 139 | 605 | +72 | 0.603 | 0.221 | 0.80 % | 0.35 % |
| **fv_dac_permuted** | **0.00000** | 0/12 | **0** | **0** | **0** | **0** | **0** | n/a | n/a | 0.00 % | 0.00 % |

**The intervention is not targeted in the accuracy sense.** For the primary
arm, 2382 of 3518 flips (**68 %**) are wrong→different-wrong. Only 18.6 % of
interventions are useful, and the method repairs 1.08 % of base errors. The
earlier "57.5 %" figure is *decisive* precision — it conditions on the flip
already having been between a correct and an incorrect label, and it hides
`U` entirely.

It *is* selective about **where** it intervenes: flip rate by base top-2
margin (e.g. `gaussian_noise_s3`) is 0.198 in the lowest-margin quintile and
≈0 in all four others, and flip rate rises monotonically with the DAC density
signal `S(x)` (0.006 → 0.017 → 0.035 → 0.052 → 0.095). So it fires on
near-ties in low-density regions — the mechanism behaves as designed — but
having fired, it mostly moves probability mass between wrong classes.
Mean GT rank 8.93 → 8.43; destination-class entropy 3.87 (vs ln 100 = 4.61),
top-1 destination share 0.079.

### 19.6 Comparisons the first write-up did not make

**D — ordinary class-wise calibration, same 12 cells** (frozen Phase 0/1 rows):

| method | mean ΔAcc | mean NLL | mean ECE |
|---|---|---|---|
| base_model | +0.00000 | 2.3767 | 0.1912 |
| native_dac | +0.00000 | 2.2744 | **0.1268** |
| temperature_scaling | −0.00001 | 2.3028 | 0.1444 |
| **vector_scaling** | **+0.00086** | 2.4469 | 0.1983 |
| odir_dirichlet | −0.22698 | 7.7430 | 0.2563 |
| fv_dac_nll | +0.00142 | 2.3177 | 0.1799 |
| fv_dac_brier | +0.00061 | 2.2797 | 0.1429 |

**Vector Scaling — a purely output-space class-wise affine map with no
geometry at all — gets +0.086 pp**, against FV-DAC's +0.142 pp. The primary
arm beats it by a factor of ~1.6 on one checkpoint; `fv_dac_brier` is *below*
it. Comparison D therefore does **not** cleanly separate FV-DAC from ordinary
non-geometric class-wise calibration.

**KCal is missing from all 12 corruption cells** — it was absent from the
frozen Phase 0/1 corruption runs, so comparison E rests only on the clean
cell plus the density-only diagnostic. Recorded as a gap, not filled.

**Per-corruption and per-severity** (primary / Brier):

| corruption | ΔAcc | | severity | ΔAcc |
|---|---|---|---|---|
| gaussian_noise | +0.00240 / +0.00103 | | **sev 1** | **+0.00198** / +0.00088 |
| jpeg_compression | +0.00253 / +0.00093 | | **sev 3** | **+0.00165** / +0.00055 |
| defocus_blur | +0.00120 / +0.00053 | | **sev 5** | **+0.00063** / +0.00040 |
| fog | **−0.00047** / −0.00007 | | | |

**The effect shrinks monotonically with severity** (+0.198 → +0.165 →
+0.063 pp). §23 predeclared "effect stable or growing with severity" as part
of the strong-positive pattern; the observed trend is the opposite, and it
runs against H-FVDAC-01's own shift sub-claim that the signal should be *more*
useful under stronger corruption.

**Calibration cost, quantified:** native DAC buys ΔECE −0.0644 and ΔNLL
−0.1023 over base. `fv_dac_nll` gives back **82.5 %** of that ECE gain
(+0.0531) and 42 % of the NLL gain (+0.0433); it is also worse than native
DAC on multiclass Brier (+0.0206).

### 19.7 Frozen continuation rule — applied, not re-derived

| criterion | primary | shared-layer |
|---|---|---|
| C1 β alive | **PASS** (6.43, interior) | FAIL |
| C2 direction | **PASS** | FAIL (net −135) |
| C3 consistency | **PASS** (9/12; all leave-one-out means > 0) | FAIL |
| C4 control not the explanation | **PASS** (permuted exactly null) | PASS |
| C5 probabilistic health | **FAIL** (ECE +0.0531 > 0.02) | FAIL |

**Decision: STOP — currently negative / inconclusive.** Steps 7 and 8 were
not launched.

#### The Brier sensitivity would have passed C5 — and is NOT promoted

`fv_dac_brier` scores ΔNLL +0.0053 and ΔECE **+0.0162** vs native DAC, both
inside the C5 allowances, with β = 1.80, net +73, 10/12 cells positive. Had
it been the primary arm, it would have satisfied all five criteria.

It is **not** promoted, for three independent reasons:

1. §18/§26 explicitly forbid adaptively promoting a sensitivity variant to
   primary after seeing results. The rule was written on the primary arm and
   the shared-layer branch; those are what it governs.
2. Its effect is **+0.061 pp** — ~16× below the ~1 pp bar, and *below*
   Vector Scaling's +0.086 pp. Passing a calibration-preservation gate by
   intervening less is not the same as working.
3. It would still be one checkpoint.

Reported here in full because the preregistration required reporting it
regardless of whether it helped. If anyone revisits this family, the honest
starting point is a **new** preregistration in which the Brier objective is
primary *a priori* — not this run reinterpreted.

### 19.8 Uncertainty, with dependence accounted for

Per-cell ΔAcc (primary): mean +0.00142, sd 0.00161 across 12 cells. The 12
cells are **not** 12 independent replications — one checkpoint, one reference
bank, 4 corruption families × 3 severities over the same 10 000 underlying
images.

- Cluster bootstrap over the 4 corruption **families**: 95 % CI
  [+0.00025, +0.00247]. Four clusters is far too few for this to carry
  inferential weight; it is reported to show the width, not to claim
  significance.
- Within-cell paired image bootstrap: 6/12 cells have a CI excluding 0. This
  is a **within-cell** statement only and is *not* evidence of generalization
  across checkpoints. It is explicitly not used as such.
- No between-checkpoint variance can be estimated at all from n = 1 seed.

### 19.9 Outcome category

**E — calibration/decision trade-off**, with the effect small enough that
F's "negligible ΔAccuracy" also applies.

- Not **A**: gains negligible, calibration not preserved.
- Not **B**: `fv_dac_shared_layer` is *worse* (net −135, 8.3 % argmax churn,
  intervention precision 0.157). It did drive all weight onto `layer3` — the
  layer native DAC zeroed — helping `jpeg_compression` (+0.86 pp) while
  hurting `gaussian_noise` (−1.05 pp). That is a **hypothesis-generating
  observation** that calibration-optimal and decision-optimal layers may
  differ. It is **not** evidence for a stable decision layer, and `layer3`
  must not be promoted into a follow-up method on this basis.
- Not **D**: density-only is weaker than the head (0.7285 vs 0.7631 clean)
  and agrees with FV-DAC only 85 %, so FV-DAC is not copying the density
  classifier. This does **not** establish that the correction is specifically
  a *DAC* mechanism rather than a weak kNN-style re-ranking signal — nothing
  here separates those two.
- **C / H-FVDAC-01-H: unanswered.** Hidden +0.142 pp vs logit-space
  +0.060 pp — a 0.082 pp gap, on one checkpoint, between arms that are not
  capacity matched (5 sources, dims 64–2048 vs 1 source, dim 100). The
  paper's own DAC moreover *includes* the logits layer as a density source,
  so this contrast does not map onto a published design distinction.

Scope: **one checkpoint seed, 4 of 15 corruptions, CIFAR-100 / ResNet-101,
under a benchmark with a train/test normalization mismatch (§2.1.2).**

### 19.10 Verdict

> Under one checkpoint and 12 preregistered CIFAR-100-C cells, the minimal
> class-wise extension of DAC produced a small positive paired
> reclassification effect (+0.142 pp; 653 wrong→correct vs 483
> correct→wrong), but failed the preregistered calibration-preservation
> criterion (+0.053 ECE relative to native DAC). We therefore stop this
> formulation. The results indicate a weak label-aligned class signal in the
> DAC reference geometry, but do not establish a robust hidden-representation
> advantage, a practically meaningful accuracy gain, or preservation of DAC's
> calibration benefit.

### 19.11 Next action

None. The frozen rule says stop and is not to be relaxed. Revisiting requires
a **new** preregistration. Do not widen `K_c`, add capacity, tune β for
corruption accuracy, add a post-hoc gate, or relax the ECE threshold.

This conclusion applies to the tested minimal formulation
`q_β = softmax((z − βR)/S_DAC)`. It does **not** establish that no class-wise
representation signal exists, that the broad readout-gap hypothesis is false,
that every source-only full-vector correction must fail, or that
target-adaptive methods cannot exploit these representations.
