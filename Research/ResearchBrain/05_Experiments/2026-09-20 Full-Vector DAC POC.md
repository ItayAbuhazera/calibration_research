---
type: experiment
status: result
date: 2026-09-20
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C, ResNet-101, checkpoint seeds 1-5
preregistered: true
tags: [dac, fv-dac, full-vector, decision-change, shift, preregistration]
---

# 2026-09-20 — Full-Vector DAC POC

Idea: [[Full-Vector Density-Aware Calibration]].
Hypothesis: [[H-FVDAC-01 Class-conditioned DAC density enables decision correction]].

**Everything from "Question" through "Kill criteria" below was written on
2026-09-20 BEFORE any FV-DAC number — clean or corrupted — was produced.**
The `Results` and `Interpretation` sections were empty at that time. The
same content is duplicated in-repo at
`Research/GeometricFullCalibration/docs/full_vector_dac_experiment.md`,
which is the version under git and therefore timestamped by commit history.

## Question

Can class-conditioning DAC's representation-density machinery turn a
calibration-only mechanism into useful full-vector decision correction
under distribution shift? And separately: if useful class-conditioned
reference geometry exists, is internal hidden representation geometry
necessary, or does output-space geometry explain the same effect?

## Hypothesis tested

H-FVDAC-01 (method-level) and H-FVDAC-01-H (hidden-specific), stated in
full in the hypothesis note. They are scored separately.

## Setup

- Data / model: CIFAR-100, ResNet-101, checkpoints
  `baseline/baseline_cross_entropy/cifar100/resnet101/seed{1..5}/best_model.pth`.
- Splits: the frozen Phase 0/1 protocol — train 45000 (reference bank),
  validation 5000 (calibration/fitting), test 10000. Materialized arrays at
  `results/studyAB/phase0/evaluation/checkpoint_seed{S}/clean/intermediates/splits/`.
- Frozen native DAC state:
  `results/studyAB/phase0/fitted_method/checkpoint_seed{S}/native_dac.pkl`.
- Corruption: CIFAR-100-C, 15 standard corruptions, severities 1/3/5 as the
  primary reporting grid; evaluation-only, no `.fit()` reachable.

### Verified native DAC configuration (seed 4, read out of the frozen pickle)

| item | value |
|---|---|
| layers | `conv1`, `layer1`, `layer2`, `layer3`, `layer4` (5, **logits NOT included**) |
| pooling | `adaptive_avg_pool2d` to (1,1) at hook time, then mean over spatial dims |
| normalization | L2, `F.normalize(p=2, dim=-1)` |
| distance | true Euclidean, `sqrt(clamp(2 - 2*cos, min=0))` — PyTorch, **not** FAISS, so no squared-distance convention issue |
| `k` (CIFAR-100) | 200 |
| bank | clean train split, 45000 rows, dims 64/256/512/1024/2048 |
| statistic | `s_l(x)` = distance to the k-th nearest bank point |
| aggregation | `S(x) = max(sum_l w_l s_l(x) + w_0, 1e-12)` |
| objective | summed squared error (Brier-style), L-BFGS-B, `w_l >= 0`, `w_0` free |
| fitted `w` (seed 4) | `[0.244625, 0.181287, 0.412497, 0.000000, 0.123806]`, `w_0 = 0.885362` |
| downstream calibrator `h` | none appended |
| bank labels retained | **no** |
| neighbour IDs retained | **no** |
| query representations cached | **no** |

### Paper / code discrepancies found (documented, not silently fixed)

1. **Layer count — VERIFIED against the paper on 2026-09-20 (second audit).**
   Table 6 / Appendix C.1 extracted directly from the PDF reads
   `RESNET18/RESNET152: PRE-BLOCK, BLOCK-1,...,BLOCK-4, LOGITS` for CIFAR-100,
   with `LOGITS` defined in the caption as a layer source. **The paper does
   include the logits layer; this repository's `native_dac` omits it (5 vs
   6 sources).** FV-DAC must therefore be described as an extension of *the
   benchmark's frozen five-layer DAC implementation*, not of the paper's
   complete layer set. It also weakens the hidden-vs-logit framing: the
   paper's DAC already uses logits as a density source. The canonical baseline for this
   experiment is the 5-layer benchmark version, because that is what the
   frozen state contains and what `beta = 0` must reproduce. The original
   implementation is untouched.
2. **Input normalization — the original description here was WRONG,
   corrected 2026-09-20.** It is not a corruption-specific quirk. Per
   `data/cifar100.py`: train+validation (and model training) use **CIFAR**
   stats (line 51); the **clean test** loader uses **ImageNet** stats
   (line 145); CIFAR-100-C matches the clean test loader. The real inherited
   issue is a **train/test normalization mismatch across the whole
   benchmark**, clean and corrupted alike, affecting every method and every
   prior Phase 0/1 number. It bears on FV-DAC specifically because the
   reference bank is *train* (CIFAR-normalized) while every query is
   *test* (ImageNet-normalized), so every class-conditional distance is
   computed across that mismatch. Not fixed, not re-run: the comparison stays
   internally matched, but this is a **compound shift** (corruption +
   normalization), not the standard CIFAR-100-C protocol. Any corrected run
   must be a separately preregistered replication of all affected methods,
   never an FV-DAC rescue.
3. **Reference-bank row order.** `train_loader` uses `SubsetRandomSampler`,
   so the pickled bank's row order does **not** match
   `intermediates/splits/train_labels.npy`. The bank *set* is identical.
   FV-DAC therefore re-extracts a labelled bank from the materialized
   `train_raw.npy`/`train_labels.npy` and asserts set-identity against the
   frozen bank (sorted row hash + global k-NN distance agreement) before
   using it. During the first seed-4 clean smoke the one-layer greedy match
   had 11 collisions. Before any clean metric or CIFAR-C result was opened,
   the implementation was changed to a sparse minimum-cost one-to-one
   assignment over eight multi-layer candidates per frozen row. Its measured
   maximum multi-layer TF32 perturbation was 0.003435, so the operational
   matching tolerance was raised from 0.001 (a `conv1`-only diagnostic) to
   0.005, while retaining the per-layer and distinct-row-separation checks.
   This is a provenance-validation adjustment, not a method/variant or
   evaluation choice.

## Fixed choices

### Primary formulation (one free scalar)

```text
r_{l,k}(x) = K_c-th NN distance( h_l(x), { h_l(x_i) : y_i = k } )
alpha_l    = w_l_hat / sum_j w_j_hat            (uniform 1/L if degenerate)
R_k(x)     = sum_l alpha_l * r_{l,k}(x)
q_beta(x)  = softmax( ( z(x) - beta * R(x) ) / S_DAC(x) ),   beta >= 0
```

`beta = 0` must reproduce native DAC numerically. No per-class bias, no
per-class scaling, no MLP, no gate.

### Predeclared arms

| identity | description | fitted params |
|---|---|---|
| `fv_dac_nll` | **primary**, NLL-fitted | 1 (`beta`) |
| `fv_dac_brier` | predeclared objective sensitivity (native DAC is Brier-fitted) | 1 |
| `fv_dac_lognorm` | Sensitivity A: `v_{l,k} = -(log r_{l,k} - mu_{l,k}) / sigma_l`, `q = softmax((z + beta*V)/S)` | 1 (+ frozen `mu`, `sigma` statistics) |
| `fv_dac_shared_layer` | Sensitivity B: `q = softmax((z - sum_l b_l r_{l,k})/S)`, `b_l >= 0` shared across classes | 5 |
| `fv_dac_permuted` | negative control: seeded permutation of bank labels | 1 |
| `fv_dac_logit_space` | mechanism control: same operator on centered, L2-normalized logits | 1 |
| `fv_dac_density_only` | diagnostic: `argmin_k R_k(x)`, not fitted | 0 |

### Predeclared `K_c` set

`K_c ∈ {5, 20, 200}` — strongly local / intermediate / literal same-K
extension. CIFAR-100 has ~450 bank examples per class, so `K_c = 200` is
valid, and 200 is also native DAC's global `K`. **This grid will not be
expanded if results are poor.**

### `K_c` selection rule

For every `K_c`, fit `beta` on the inner-**fit** half of clean validation;
compare the frozen candidates by NLL on the disjoint inner-**select** half;
take the argmin. Primary arm only; all other arms inherit the selected
`K_c`. Corruption data is never consulted.

### Split roles

```text
clean train (45000)            -> reference bank only
clean validation inner-fit     -> fit beta / b_l / mu / sigma
clean validation inner-select  -> select K_c
clean test (10000)             -> evaluation
CIFAR-100-C                    -> evaluation only, frozen state
```

`make_inner_validation_split(select_fraction=0.5, seed=123)` — the
repository's existing convention, unchanged; splits are not redefined.

**Declared deviation:** native DAC's own weights were fitted (in the frozen
Phase 0/1 run) on the *whole* validation split, including what is now the
inner-select half. Those weights are frozen and identical across all `K_c`
candidates, so they cannot bias the `K_c` comparison, but the inner-select
NLL is not perfectly clean of them. Recorded rather than papered over.

### Fitting objective

`beta_hat = argmin_{beta >= 0} NLL(q_beta, y)` on inner-fit. Not accuracy,
not net flips, never on CIFAR-C, never after seeing test results. The
Brier-fitted variant is a predeclared sensitivity, not an alternative
primary.

## Primary metrics

Primary outcome: `DeltaAcc = Acc(FV-DAC) - Acc(base)` under corruption.
Secondary: NLL, Brier, ECE, adaptive ECE, classwise ECE, AUROC, AURC, and
the full flip decomposition `W` (wrong->correct), `H` (correct->wrong),
`U` (wrong->different wrong), with the identity `DeltaAcc = (W - H)/N`
verified numerically per cell.

## Baselines

`base_model`, `native_dac`, `temperature_scaling`, `vector_scaling`,
`odir_dirichlet`, `kcal` (reused from the frozen Phase 0/1 artifacts), plus
every FV-DAC control listed above.

## Continuation rule — FROZEN 2026-09-20, BEFORE ANY CIFAR-C FV-DAC RESULT

Applied to the predeclared 12-cell seed-4 POC
(`gaussian_noise`, `defocus_blur`, `fog`, `jpeg_compression` x severity
1/3/5). No hard `1pp` threshold; this is a pattern rule.

**Continue to full corruption extraction if ALL of the following hold:**

- **C1 — beta is alive.** The selected primary `beta_hat > 0` on seed 4 and
  is not pinned at the search boundary.
- **C2 — direction.** Mean `DeltaAcc` across the 12 cells is `> 0`, and
  `W > H` in aggregate across the 12 cells.
- **C3 — consistency.** At least 7 of 12 cells have `DeltaAcc >= 0` and at
  least 5 of 12 have `DeltaAcc > 0`; the effect is not carried by a single
  corruption family (removing any one of the four corruptions leaves mean
  `DeltaAcc > 0`).
- **C4 — the control is not the explanation.** `fv_dac_permuted` mean
  `DeltaAcc` is at most half of `fv_dac_nll` mean `DeltaAcc`, or is
  nonpositive.
- **C5 — probabilistic health.** Aggregate corruption NLL for `fv_dac_nll`
  is no worse than `native_dac` NLL by more than 0.05, and top-label ECE is
  no worse by more than 0.02 absolute.

**Continue in the "shared-layer branch only" mode** if C1/C2/C3 fail for
`fv_dac_nll` but hold for `fv_dac_shared_layer` with C4 and C5 satisfied
for that arm. This is Outcome B and is worth completing; it is reported as
such and does not get promoted to "primary".

**Stop and classify as currently negative / inconclusive otherwise.** In
particular, stop if `beta_hat = 0` for both `fv_dac_nll` and
`fv_dac_shared_layer`, or if `W <= H` across the 12 cells for every arm.

This rule will not be edited after the 12-cell results are opened. If it is
ever revised, the revision is a *new* experiment note with a new date, and
the original rule stays in place here.

## Kill criteria

As recorded in
[[H-FVDAC-01 Class-conditioned DAC density enables decision correction]],
split into method-level criteria and the single hidden-representation-
specific criterion (`FV-DAC-logit-space` matching or beating hidden
FV-DAC), which does **not** by itself kill the method-level hypothesis.

## Artifacts

- Code: `Calibrators/full_vector_dac.py`,
  `Experiments/run_fv_dac_experiment.py`,
  `Experiments/aggregate_fv_dac.py`.
- Design doc: `docs/full_vector_dac_experiment.md`.
- Tests: `tests/test_full_vector_dac.py`.
- Slurm: `scripts/fv_dac_fit_clean.sbatch`,
  `scripts/fv_dac_evaluate_corruption.sbatch`.
- Results root: `results/fv_dac/checkpoint_seed{S}/{cell}/`.
- Frozen state: `results/fv_dac/fitted_state/checkpoint_seed{S}/`.

## Results

Checkpoint **seed 4 only**. Authoritative numbers:
`Research/GeometricFullCalibration/results/fv_dac/aggregate/`
(`per_cell.csv`, `continuation_rule.json`, `summary.json`).
Frozen state hash `351423c41f53b307c09136c1bbd1cb3112367c72b55e47b3720f4ef46f791ae1`.

### Structural checks (all passed)

| check | result |
|---|---|
| `beta = 0` reproduces native DAC | max abs prob diff **0.0** (exact), re-checked on every cell |
| native DAC reproduced from the labelled frozen bank | max abs prob diff 7.9e-7, argmax agreement 1.000 |
| our `native_dac` rows vs frozen Phase 0/1 rows | agree to 4 dp (e.g. `gaussian_noise_s3` NLL 3.8860 both; accuracy 0.2215 vs 0.2214, one sample, TF32) |
| corrupted cell bytes vs benchmark's materialized split | identical |
| flip identity `DeltaAcc = (W-H)/N` | holds for every method, every cell |
| corruption runs evaluation-only | `evaluation_only: true`, fit guard armed before the loader |

### `K_c` selection (clean inner-select NLL, primary arm only)

| `K_c` | beta | inner-fit NLL | inner-select NLL | select acc |
|---|---|---|---|---|
| **5** | **6.4324** | 0.87914 | **0.87283** | 0.7696 |
| 20 | 4.6367 | 0.88465 | 0.87780 | 0.7684 |
| 200 | 2.2283 | 0.89483 | 0.88698 | 0.7664 |

`beta = 0` inner-fit NLL was 0.90329, so beta is genuinely non-degenerate.
Selected `K_c = 5`; no candidate sat at a search boundary.

### Clean test (seed 4, 10 000 samples)

| method | acc | dAcc | NLL | Brier | ECE | W | H | U |
|---|---|---|---|---|---|---|---|---|
| base_model | 0.7631 | — | 0.9378 | 0.3381 | 0.0489 | 0 | 0 | 0 |
| native_dac | 0.7631 | +0.0000 | 0.9472 | 0.3334 | **0.0268** | 0 | 0 | 0 |
| fv_dac_nll | 0.7644 | +0.0013 | 0.9242 | 0.3357 | 0.0515 | 48 | 35 | 64 |
| fv_dac_brier | 0.7643 | +0.0012 | 0.9350 | 0.3333 | 0.0299 | 20 | 8 | 19 |
| fv_dac_lognorm | 0.7658 | +0.0027 | 0.9282 | 0.3345 | 0.0441 | 58 | 31 | 74 |
| fv_dac_shared_layer | 0.7666 | +0.0035 | 0.8822 | 0.3314 | 0.0657 | 118 | 83 | 129 |
| fv_dac_permuted | 0.7631 | +0.0000 | 0.9472 | 0.3334 | 0.0268 | 0 | 0 | 0 |
| fv_dac_logit_space | 0.7646 | +0.0015 | 0.9302 | 0.3377 | 0.0583 | 23 | 8 | 15 |

Reused frozen Phase 0/1 baselines on the same cell: `temperature_scaling`
0.7631, `vector_scaling` 0.7659, `odir_dirichlet` 0.4038, `kcal` 0.7611.
Density-only diagnostic: accuracy 0.7285, agreement with base 0.844, with
FV-DAC 0.854, recoverability among base errors 0.121.

`fv_dac_shared_layer` fitted `b = [0, 0, 0, 26.09, 0]` — it puts **all** of
its weight on `layer3`, the one layer native DAC assigned weight exactly
**zero**. That is the literal Outcome-B signature, and it is why the arm was
predeclared.

### 12-cell corruption POC (seed 4)

Per-cell `dAcc` / net flips `(W-H)`:

| cell | fv_dac_nll | lognorm | shared_layer | permuted | logit_space |
|---|---|---|---|---|---|
| defocus_blur_s1 | +0.0012 / +12 | +0.0008 / +8 | +0.0053 / +53 | 0.0000 / 0 | +0.0010 / +10 |
| defocus_blur_s3 | +0.0025 / +25 | +0.0010 / +10 | -0.0023 / -23 | 0.0000 / 0 | +0.0012 / +12 |
| defocus_blur_s5 | -0.0001 / -1 | -0.0014 / -14 | -0.0087 / -87 | 0.0000 / 0 | +0.0005 / +5 |
| fog_s1 | +0.0014 / +14 | +0.0011 / +11 | +0.0042 / +42 | 0.0000 / 0 | +0.0009 / +9 |
| fog_s3 | -0.0004 / -4 | -0.0024 / -24 | +0.0004 / +4 | 0.0000 / 0 | +0.0011 / +11 |
| fog_s5 | -0.0024 / -24 | -0.0047 / -47 | -0.0069 / -69 | 0.0000 / 0 | +0.0006 / +6 |
| gaussian_noise_s1 | +0.0028 / +28 | +0.0029 / +29 | -0.0102 / -102 | 0.0000 / 0 | +0.0011 / +11 |
| gaussian_noise_s3 | +0.0022 / +22 | +0.0026 / +26 | -0.0127 / -127 | 0.0000 / 0 | 0.0000 / 0 |
| gaussian_noise_s5 | +0.0022 / +22 | +0.0024 / +24 | -0.0085 / -85 | 0.0000 / 0 | +0.0003 / +3 |
| jpeg_compression_s1 | +0.0025 / +25 | +0.0025 / +25 | +0.0100 / +100 | 0.0000 / 0 | +0.0002 / +2 |
| jpeg_compression_s3 | +0.0023 / +23 | +0.0025 / +25 | +0.0068 / +68 | 0.0000 / 0 | -0.0005 / -5 |
| jpeg_compression_s5 | +0.0028 / +28 | +0.0035 / +35 | +0.0091 / +91 | 0.0000 / 0 | +0.0008 / +8 |

Aggregates over the 12 cells:

| arm | mean dAcc | cells > 0 | W | H | net | beta |
|---|---|---|---|---|---|---|
| arm | mean dAcc | cells>0 | F | W | H | **U** | net | decisive | **intervention** |
|---|---|---|---|---|---|---|---|---|---|
| **fv_dac_nll** | **+0.00142** | 9/12 | 3518 | 653 | 483 | **2382** | +170 | 0.575 | **0.186** |
| fv_dac_brier | +0.00061 | 10/12 | 1136 | 225 | 152 | 759 | +73 | 0.597 | 0.198 |
| fv_dac_lognorm | +0.00090 | 9/12 | 3726 | 707 | 599 | 2420 | +108 | 0.541 | 0.190 |
| fv_dac_shared_layer | -0.00113 | 6/12 | 9938 | 1558 | 1693 | 6687 | -135 | 0.479 | 0.157 |
| fv_dac_logit_space | +0.00060 | 10/12 | 955 | 211 | 139 | 605 | +72 | 0.603 | 0.221 |
| **fv_dac_permuted** | **0.00000** | 0/12 | **0** | **0** | **0** | **0** | **0** | n/a | n/a |

**Corrections to the first write-up of this note** (made 2026-09-20 after an
adversarial re-audit): `U` was omitted entirely, so *decisive* precision was
quoted where *intervention* precision belonged; the `fv_dac_brier` arm was
missing although it was fitted and evaluated in all 12 cells; and two means
were mis-transcribed (`lognorm` +0.00124 -> **+0.00090**, `logit_space`
+0.00055 -> **+0.00060**).

68% of the primary arm's flips are wrong->different-wrong. It repairs 1.08%
of base errors. It *is* selective about where it fires (flip rate 0.198 in
the lowest base-margin quintile, ~0 elsewhere; rising monotonically with
`S(x)`), but having fired it mostly moves mass between wrong classes.

**Severity trend runs the wrong way:** sev1 +0.00198, sev3 +0.00165, sev5
+0.00063. The predeclared strong-positive pattern wanted the effect stable or
*growing* with severity, and H-FVDAC-01's shift sub-claim predicted the same.

**Vector Scaling — no geometry at all — gets +0.00086 on the same 12 cells**
(FV-DAC +0.00142, `fv_dac_brier` +0.00061). Comparison D does not cleanly
separate FV-DAC from ordinary class-wise calibration. **KCal is absent from
all 12 corruption cells** (it was missing from the frozen Phase 0/1
corruption runs), so comparison E rests on the clean cell alone.

**The Brier sensitivity would have passed C5** (dNLL +0.0053, dECE +0.0162,
both inside the allowances) and is **not** promoted: promoting a sensitivity
after seeing results is forbidden, its effect (+0.061 pp) is below Vector
Scaling's, and it is still one checkpoint.

### The calibration cost (why C5 failed)

FV-DAC's ECE lands almost exactly back on the **base model's** ECE, i.e. it
spends essentially all of native DAC's calibration gain:

| cell | base ECE | native_dac ECE | fv_dac_nll ECE | FV − nDAC |
|---|---|---|---|---|
| gaussian_noise_s1 | 0.1993 | 0.1231 | 0.1826 | +0.0595 |
| gaussian_noise_s5 | 0.4009 | 0.2902 | 0.3546 | +0.0644 |
| fog_s5 | 0.2784 | 0.2153 | 0.2748 | +0.0595 |
| jpeg_compression_s5 | 0.1715 | 0.1079 | 0.1650 | +0.0571 |
| defocus_blur_s3 | 0.1216 | 0.0596 | 0.1150 | +0.0553 |

Mean over the 12 cells: NLL **+0.0433** vs native DAC (inside the 0.05
allowance), top-label ECE **+0.0531** (far outside the 0.02 allowance).

### Frozen continuation rule — applied, not re-derived

| criterion | primary | shared-layer |
|---|---|---|
| C1 beta alive | **PASS** (6.43, interior) | FAIL |
| C2 direction | **PASS** (mean dAcc > 0, W > H) | FAIL (net -135) |
| C3 consistency | **PASS** (9/12 > 0; every leave-one-corruption-out mean > 0) | FAIL |
| C4 control not the explanation | **PASS** (permuted exactly null) | PASS |
| C5 probabilistic health | **FAIL** (ECE +0.0531 > 0.02) | FAIL |

**Decision: STOP — classify as currently negative / inconclusive.**

The rule was not edited after these results were opened. Four of five
criteria passed; the rule requires all five, so the run stops here. Steps 7
and 8 of the execution plan (the remaining 11 corruptions and checkpoint
seeds 1, 2, 3, 5) were therefore **not** launched.

## Protocol deviations

1. Native DAC's own weights saw the inner-select half (see "Declared
   deviation" above).
2. The canonical native DAC baseline is the repository's 5-layer version,
   not the 6-layer paper description.
3. CIFAR-100-C keeps the benchmark's ImageNet normalization.

## Interpretation

> **Revised 2026-09-20 after an adversarial re-audit.** The first version of
> this section said "the mechanism is real, not an artefact". That was an
> overclaim and is withdrawn. What follows replaces it.

**Outcome E — calibration/decision trade-off — with the effect size small
enough that F's "negligible DeltaAccuracy" also applies.**

### What the permuted control does and does not establish

The control is genuinely informative, and more so than a null result usually
is: the permuted-label arm fitted `beta = 0` *exactly* and flipped **zero**
samples in all 12 cells. Permuted reference geometry offered the optimizer no
clean-NLL-improving direction at all, while real labels produced
`beta = 6.43`. That asymmetry is real.

But it establishes only:

- the optimizer is not *forced* to introduce a nonzero correction;
- real class assignments carry more clean-NLL-relevant structure than this
  particular permutation;
- the +0.142 pp is not an automatic consequence of adding one free parameter.

It does **not** establish that the corruption gain is robust. One checkpoint;
one permutation seed; 12 cells sharing a checkpoint, a bank and the same
10 000 underlying images, so not 12 independent replications; and the
hidden-vs-logit contrast is not capacity matched.

**Defensible statement:** *a weak label-aligned semantic signal is present in
this realization, but the experiment does not establish a robust or
practically meaningful decision improvement.*

### Claim-by-claim

| # | claim | evidence | status | scope limit |
|---|---|---|---|---|
| 1 | The implementation can change argmax | synthetic unit test; 3518 flips over 12 cells | **supported** | structural |
| 2 | Real class assignments produce a nonzero fitted correction | `beta = 6.43` vs permuted `beta = 0` exactly | **weakly supported** | 1 seed, 1 permutation, clean-NLL criterion |
| 3 | The correction makes some useful flips | `W > H` in 9/12 cells, net +170 | **weakly supported** | intervention precision 0.186; 68% of flips are wrong->wrong |
| 4 | The accuracy gain is stable across models and shifts | +0.142 pp mean; **falls** with severity; fog negative; VS gets +0.086 pp | **not established** | 1 checkpoint, 4/15 corruptions |
| 5 | Hidden representations add unique information beyond logits | 0.142 vs 0.060 pp (gap 0.082) | **unanswered** | not capacity matched; paper's DAC already uses logits |
| 6 | The method preserves native DAC's calibration benefit | gives back 82.5% of DAC's ECE gain; Brier also worse | **contradicted** | for this formulation |
| 7 | The broad representation readout-gap hypothesis | — | **not tested** | out of scope of this design |

### The ablations, read conservatively

- **shared-layer**: mean negative (-0.113 pp), 8.3% argmax churn,
  intervention precision 0.157. That it placed all weight on `layer3` — the
  layer native DAC zeroed — while helping JPEG (+0.86 pp) and hurting
  Gaussian noise (-1.05 pp) is a **hypothesis-generating observation** that
  calibration-optimal and decision-optimal layers may differ. It is **not**
  evidence for a stable decision layer, and `layer3` must not be promoted
  into a follow-up method.
- **density-only**: weaker than the head (0.7285 vs 0.7631) with 85%
  agreement, so FV-DAC is not merely copying it. This does **not** show the
  correction is specifically a *DAC* mechanism rather than a weak kNN-style
  re-ranking signal; nothing here separates those.
- **hidden vs logit-space**: treat as unanswered, not as weak support.

### Scope

One checkpoint seed, 4 of 15 corruptions, one architecture, one dataset, and
a benchmark carrying a train/test normalization mismatch (see Protocol
deviations). Nothing here revises any prior RGCL / full-vector / GC-DAC
result; this was never an attempt to rescue them.

## Next action

None. The frozen continuation rule says stop, and it is not to be relaxed.
H-FVDAC-01 is closed as a negative experiment.

If this direction is ever revisited it needs a **new** experiment note and a
new preregistration, not an edit to this one. The specific thing that would
have to change first is the calibration cost: any future FV-DAC variant has
to keep native DAC's ECE, because paying 0.05 ECE for 0.0014 accuracy is a
bad trade at any effect size. Do **not** respond to this result by widening
the `K_c` grid or adding capacity — both are explicitly forbidden by §26 of
the specification, and neither addresses the actual failure.

## Addendum 2026-09-21 — limited scope of this verdict (appended; nothing above edited)

* The experiment ran under the **legacy** preprocessing protocol (`legacy_v1_mixed_norm`): test and CIFAR-100-C inputs
  ImageNet-normalized, reference bank/fitting CIFAR-normalized — confirmed in [[2026-09-21 Normalization Audit and Corrected Protocol]].
  The closed verdict and its historical outputs stand as a legacy-protocol result; it is **not** re-run or rescued here.
* It tested one combination: five DAC sources, GAP pooling, class-conditional K_c-th neighbour distances, DAC-inherited layer weights
  (one of which is exactly 0), one additive coefficient. It does **not** show that internal representations hold no additional decision information.
  The follow-up [[2026-09-21 Layer-Selection Pilot]] separates layer choice, pooling, distances vs probes, readout, transfer and preprocessing.
* Flip interpretation, corrected: 86.3 % of flips (W+U = 3035 of 3518) were on examples the base model already got wrong; that share is **not**
  evidence of geometric error detection without comparison to base-error prevalence and a matched-margin control (the pilot adds that control; enrichment ≈ 1.06–1.09).
  wrong→different-wrong (U) flips are **neutral** for top-1 accuracy, not automatically harmful; low intervention precision alone does not decide utility —
  the operational quantity is (W−H)/N alongside probability-quality metrics (here net +170 flips ≈ +0.142 pp, ECE +0.053 vs native DAC).
