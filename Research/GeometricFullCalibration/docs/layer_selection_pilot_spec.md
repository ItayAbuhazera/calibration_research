# Layer-selection pilot (L ∈ {1, 4, 6, 8}) — frozen specification

**Frozen:** 2026-09-21, before any corrected-protocol corruption cell was
evaluated and before any layer-study number was computed. §§1–11 are not to be
edited after results are opened; results go in a separate section appended at
the end (as `docs/full_vector_dac_experiment.md` §19 did).
**Vault:** `ResearchBrain/05_Experiments/2026-09-21 Layer-Selection Pilot.md`,
`ResearchBrain/04_Hypotheses/H-LAYER-01 …`.
**Relation to the closed FV-DAC pilot:** that experiment stays closed under its
own stopping rule and its historical outputs and verdict are untouched. This is
a **new** bounded study, not a rescue run and not a relaxation of that rule.

---

## 0. Research question

> Does access to additional, appropriately *selected* internal layers improve
> clean-fitted decision correction and probability quality under corruption
> shift, and is any benefit specific to geometric (distance) statistics?

The FV-DAC pilot confounded six things. This study is built to separate them:

| tag | candidate explanation for FV-DAC's small effect | how this pilot isolates it |
|---|---|---|
| A | inadequate layer choice | greedy clean-selected sets vs depth-spaced sets vs the old inherited native-DAC set; L = 1/4/6/8 |
| B | information lost by pooling | **not resolved by the primary run.** Global average pooling is fixed; one bounded 2×2-grid sensitivity is pre-declared (§4.3). A null under GAP says nothing about the un-pooled tensor. |
| C | information lost when representations are reduced to distances | Family A (distances) vs Family B (linear probes on the same pooled features) |
| D | inadequate decision readout | Family B and the full-logit probe control; additive-β vs probe readout |
| E | no clean-to-shift transfer | every arm is clean-fitted/clean-selected and evaluated under 12 corruption cells; clean-test column reported alongside |
| F | preprocessing artifacts | corrected protocol `corrected_v2_train_norm`; legacy numbers are never mixed in |

A null result here can rule out only **the tested combination**; it is not an
information-theoretic statement (a null linear probe does not prove no useful
information exists).

## 1. Preprocessing (resolves F)

`corrected_v2_train_norm` (`utils/preprocessing_protocol.py`): clean test and
CIFAR-100-C are normalized with the CIFAR-100 statistics the checkpoints were
trained with. Evidence and the legacy-vs-corrected audit are in
`docs/normalization_audit.md`. Legacy Phase 0/1 numbers are legacy-labelled and
are **not** reused as baselines.

## 2. Checkpoints and conditions

* ResNet-101, CIFAR-100, `baseline_cross_entropy`, **checkpoint seeds 2 and 4**,
  selected before any corrected shift evaluation. Seed 4 = the seed of the
  closed FV-DAC pilot (continuity); seed 2 = the lowest-numbered other
  Phase 0/1 seed that is not seed 1 (seed 1 is the documented kNN-regime outlier in
  `02_Observations/Seed 1 occupies a qualitatively different kNN regime.md`).
  The choice used only that note and the seed number, not any result.
  **Two seeds do not support a cross-seed robustness claim.**
* Conditions: clean test (10 000) + the previously defined 12 CIFAR-100-C cells
  (`gaussian_noise`, `defocus_blur`, `fog`, `jpeg_compression` × severity 1/3/5).
  **These 12 cells are previously inspected development conditions, not
  pristine confirmation data**; the closed pilot's legacy-protocol results on
  seed 4 were seen. Any later "primary configuration" claim needs held-out
  conditions (other corruptions, other severities, other seeds).
* Checkpoint paths: `…/aaai_full_experiments/results/baseline/baseline_cross_entropy/cifar100/resnet101/seed{2,4}/best_model.pth`.

## 3. Split roles (new plan; versioned `layer_pilot_split_v1`)

| role | size | used for |
|---|---|---|
| train (benchmark split for the seed) | 45 000 | Family A reference bank (features + labels); Family B probe weights (see note) |
| validation inner-FIT | 2 500 | β; probe regularizer λ and probe temperature T |
| validation inner-SELECT | 2 500 | layer-set scoring (greedy selection criterion) |
| clean test | 10 000 | evaluation |
| CIFAR-100-C cells | 10 000 each | evaluation only; frozen state |

Inner split = `utils.decision_audit.make_inner_validation_split(select_fraction=0.5, seed=123)`
(identical to the FV-DAC pilot). No existing split is redefined.

* **No independent role for a later formal risk calibration is consumed.** The
  clean test split is *not* borrowed for anything. If a risk-control
  follow-up is run it needs its own role (e.g. splitting inner-SELECT again or
  new clean data); this pilot deliberately leaves clean test untouched.
* **Declared deviation 1.** Native DAC's `S_DAC` weights are fitted by the
  benchmark on the whole validation split (including inner-SELECT). `S_DAC` is
  identical for every Family-A arm, so it cannot bias the comparison between
  layer sets, but the inner-SELECT NLL of a Family-A arm is not fully
  independent of it.
* **Declared deviation 2.** Family-B layer probes are fitted on **train** (the
  checkpoint's own training data), because 2 500 validation examples cannot
  identify a 100-class probe on 2 048-d features. Train features are the
  network's training set (memorized to a degree); the probe temperature,
  regularizer and all selection use held-out validation roles only. The
  **full-logit probe cannot be fitted on train** (train logits are degenerate:
  near one-hot), so it is fitted on validation inner-FIT with the same family
  and the same λ grid. The two data sizes differ; this asymmetry disadvantages
  the logit probe and is stated wherever the two are compared.

## 4. Representations (resolves the layer-choice axis)

### 4.1 Candidate layers (declared before corrected corruption results)

ResNet-101 (`Net/resnet_cifar.py`): 3×3 stem `conv1`→`bn1`→ReLU (64 ch, 32×32);
`layer1` 3 Bottleneck (256 ch, 32×32); `layer2` 4 (512 ch, 16×16); `layer3` 23
(1024 ch, 8×8); `layer4` 3 (2048 ch, 4×4); `avg_pool2d(4)` → `fc`. 33 residual
blocks. Candidates are **post-ReLU outputs of Bottleneck blocks** (module output
tensor of `layerS.B`), 12 distinct sources spread over depth:

| idx | module | shape (C×H×W) | pooled d |
|---|---|---|---|
| 0 | `layer1.0` | 256×32×32 | 256 |
| 1 | `layer1.2` | 256×32×32 | 256 |
| 2 | `layer2.1` | 512×16×16 | 512 |
| 3 | `layer2.3` | 512×16×16 | 512 |
| 4 | `layer3.2` | 1024×8×8 | 1024 |
| 5 | `layer3.7` | 1024×8×8 | 1024 |
| 6 | `layer3.12` | 1024×8×8 | 1024 |
| 7 | `layer3.17` | 1024×8×8 | 1024 |
| 8 | `layer3.22` | 1024×8×8 | 1024 |
| 9 | `layer4.0` | 2048×4×4 | 2048 |
| 10 | `layer4.1` | 2048×4×4 | 2048 |
| 11 | `layer4.2` | 2048×4×4 | 2048 |

Aliases are avoided: `layer1.2`, `layer2.3`, `layer3.22`, `layer4.2` **are** the
tensors native DAC calls `layer1`…`layer4` (the `nn.Sequential` output is its
last block's output), so native DAC's non-stem layers are candidates 1, 3, 8, 11
and are extracted once. The native-DAC stem source (`conv1`, *pre*-BN conv
output, 64-d) is not a residual-block output and is used **only** to compute
`S_DAC`, never as a candidate. Logits are **not** a layer; they enter as separate
controls (§5, §6).

Depth caveat that is part of the design, not hidden: 5 of the 12 candidates are
`layer3` (23 blocks), 3 are `layer4`. Sets are therefore not depth-uniform.

### 4.2 Extraction rule (fixed, primary)

Forward hook on the block output → **global average pool over (H, W)** → L2
normalize (`F.normalize(p=2)`), exactly the operation native DAC's hooks and
`LayerKNNScorer._preprocess` apply. No projection, compression, JL, whitening or
dimension matching: dimensionality varies with the layer (256–2048) and is
**reported, not equalized**. Every conclusion applies to this representation.
Strict fp32 (TF32 disabled) for the pilot's own forward passes.

### 4.3 Bounded spatial sensitivity (pre-declared, Family A only, budget permitting)

`grid2`: replace GAP by a fixed 2×2 adaptive average pool, flatten (d → 4d),
L2 normalize. Applied to the **layers of the GAP-selected nested set A₈ only**,
with the same equal-weight aggregation, β refitted on inner-FIT, selection **not**
re-run (no search over pooling). No other pooling configuration will be tried.

## 5. Family A — DAC-style class-distance correction

For a layer set A (|A| = L), class k, query x:

* `r_{l,k}(x)` = distance from `h_l(x)` to its **K_c-th nearest neighbour among
  the reference-bank rows of class k**; true Euclidean distance of unit vectors
  (`√max(2−2⟨a,b⟩,0)`); bank = the 45 000 train rows of the seed; **K_c = 5**.
  K_c = 5 is **inherited from the closed FV-DAC pilot** (selected there on clean
  inner-SELECT under the *legacy* protocol from {5, 20, 200}); it is **not**
  re-selected here and the grid is not widened. With ≈450 examples/class it is
  valid for every class.
* `R_{A,k}(x) = (1/L) Σ_{l∈A} r_{l,k}(x)` — equal weights, **no learned
  per-layer weights, no zero weights inherited from native DAC**, **no distance
  rescaling** (raw unit-vector Euclidean distances, as native DAC's `s_l`).
  Layers differ in typical distance scale, so equal weighting is not
  scale-neutral; β is refitted per set.
* `q_{A,β}(x) = softmax((z(x) − β R_A(x)) / S_DAC(x))`, β ≥ 0, one scalar fitted
  by minimizing NLL on inner-FIT (grid `{0}∪geomspace(1e-3,1e3,61)` + bounded
  refinement — `Calibrators/full_vector_dac.py::fit_beta`, accuracy/flips are
  never the objective).
* `S_DAC(x) = max(Σ_l w_l s_l(x) + w_0, 1e-12)` is the **corrected-protocol
  native DAC component**: weights and intercept are read from the corrected
  benchmark's fitted `native_dac.pkl`; `s_l(x)` is recomputed in the pilot's own
  pass (k = 200, native layers `conv1, layer1..layer4`). It is **identical across
  all layer-count arms**.
* β = 0 reproduces native DAC: `q = softmax(z/S_DAC)` — asserted per cell against
  the pilot's own recomputation, and cross-checked against the benchmark's stored
  `native_dac` probabilities (numerical tolerance reported; the two differ only by
  TF32 vs fp32 forward passes).
* **Scalar-vs-vector distinction (explicit).** `S_DAC` is a positive scalar per
  sample; dividing a fixed logit vector by it cannot change the argmax. Any
  decision change in Family A comes from the term `−βR_A`, i.e. class `j` beats
  `i` iff `β(R_i − R_j) > z_i − z_j`.
* Controls: **β = 0** (= native DAC); **permuted-label bank** (bank labels
  permuted with a fixed seed; same distances, wrong class conditioning) at every
  reported set; **logit-space source** at L = 1 (operator applied to centered,
  L2-normalized logits of the 45 000 train rows, dim 100); **depth-spaced sets**
  (§7).

## 6. Family B — clean-fitted linear-probe readout (diagnostic comparator)

**Not an extension of DAC.** It replaces the decision, it does not correct the
base head.

* Probe family (identical for every layer and for logits): multinomial logistic
  regression on **z-scored** pooled-L2-normalized features (train mean/std);
  loss = mean cross-entropy + (λ/2)‖W‖²_F; full-batch L-BFGS (strong-Wolfe,
  history 20, ≤ 200 iterations, zero init, deterministic); λ ∈ {1e-4, 1e-3, 1e-2}
  chosen per probe by inner-FIT NLL; then a scalar temperature T by 1-D NLL
  minimization on inner-FIT. Scalar calibration rule is the same for every probe
  and every L.
* Multi-layer combination: **equal-weight average of the per-layer calibrated
  probability vectors** `q_B,A = (1/L) Σ_l softmax(probe_l(x)/T_l)`. No learned
  fusion.
* Controls / comparators: **full-logit probe** (input = the 100-d logit vector,
  same family, fitted on inner-FIT — see §3 deviation 2); **final-representation
  probe** (candidate 11, `layer4.2` GAP, which is exactly the vector `fc` reads;
  on train it re-learns the head); depth-spaced sets (§7).
* Parameter counts are reported per arm (`d·100 + 100 + 1` per layer probe; the
  logit probe has 10 101). Increasing L increases the number of fitted probes, so
  a Family-B gain is **not** attributable to unique geometric information.

## 7. Selection (clean data only) and controls

* **Greedy forward selection**, separately per family, on **inner-SELECT NLL**
  of the fully specified arm (Family A: β refitted on inner-FIT for each
  candidate set; Family B: per-layer probes/T from inner-FIT, average probability
  scored on inner-SELECT). Start with the best single layer; at each step add the
  candidate giving the lowest inner-SELECT NLL **even if that NLL is worse than
  the previous step** (so the trace shows whether more layers help). Ties within
  1e-9 → smaller candidate index. Nested sets `A₁ ⊂ A₄ ⊂ A₆ ⊂ A₈` = prefixes of
  the greedy order. The complete trace, including every rejected candidate's
  score at every step, is saved (`selection_trace.json`).
* Family-A and Family-B sets are chosen separately and the distinction is
  recorded. Same search procedure, aggregation rule and preprocessing for every L.
* **Depth-spaced control** (no search): candidate indices
  `round(linspace(0, 11, L))` with `np.round` (half-to-even), L=1 → `[11]`
  (final block): L=4 → {0,4,7,11}; L=6 → {0,2,4,7,9,11}; L=8 → {0,2,3,5,6,8,9,11}.
* **All four counts are a predefined comparison.** No count is retrospectively
  promoted to "the method"; any later primary configuration must be selected on
  clean data or evaluated on genuinely held-out conditions.
* Baselines (corrected-protocol artifacts only): base model, Temperature
  Scaling, Vector Scaling, corrected native DAC — the frozen fitted state written
  by the corrected benchmark run (`results/studyAB/phase0_corrected_v2/`) applied to
  the pilot's logits; plus the Family-B full-logit and final-representation probes
  and the depth-spaced sets above.
* Not added automatically (conditional follow-ups only if this pilot leaves a
  concrete unresolved question): target-labelled oracle suite; BN-Adapt suite.

## 8. Metrics and reporting

Per (seed, cell, arm): accuracy, ΔAccuracy vs base, NLL, multiclass Brier,
fixed-bin (15) top-label ECE, adaptive ECE (`utils.unified_metrics.evaluate_all`);
flip decomposition W (wrong→correct), H (correct→wrong), U (wrong→different-
wrong), total flips F=W+H+U, net W−H, decisive precision W/(W+H), intervention
precision W/F, fraction of base errors repaired W/#base-errors; ground-truth rank
before/after; results by corruption and severity; **base-error enrichment**
of the flipped set against a **top-2-margin-matched** expectation (so a share
of flips on base-wrong examples is compared with base error prevalence at matched
margin rather than read as geometric error detection); cost: dims per set, bank
bytes, reference-search FLOP proxy `N_bank·Σd`, measured reference-search wall
time, probe parameter count.

Interpretation rules fixed here: wrong→wrong (U) flips are **neutral for top-1
accuracy**, not harmful; low intervention precision alone does not determine net
utility; the operational quantity is `(W−H)/N` alongside probability-quality
metrics.

## 9. Continuation thresholds (engineering decisions, not statistical thresholds)

The closed pilot used a ≈1 percentage-point practical bar. It informs, but is not
a universal threshold. Let ΔAcc denote the mean over the 12 cells of
`acc(arm) − acc(base)`, averaged over the 2 seeds unless stated. The predeclared
candidate arms are the greedy arms at L ∈ {4, 6, 8} of both families (6 arms);
L = 1 arms and all controls are comparators, not candidates.

An arm is **promising** iff ALL hold:
* P1 ΔAcc ≥ +1.0 pp seed-averaged **and** ≥ +0.5 pp in each seed;
* P2 it exceeds both Vector Scaling and the Family-B full-logit probe by ≥ +0.25 pp
  ΔAcc (so the effect is not ordinary class-wise logit calibration);
* P3 W > H in ≥ 20 of the 24 (seed × cell) pairs;
* P4 probabilistic health vs corrected native DAC: mean top-label ECE worse by
  ≤ 0.02 absolute **and** mean NLL worse by ≤ 0.05;
* P5 (Family A only) the permuted-label arm's ΔAcc ≤ ½ of the arm's.

Verdicts on the pilot as a whole:
* **A promising diagnostic result requiring replication** — ≥ 1 candidate arm is
  promising. With six candidate arms and development conditions, this is a
  flag for replication on fresh seeds and conditions, **not** evidence.
* **No material evidence to continue this tested family** — no candidate arm has
  seed-averaged ΔAcc ≥ +0.5 pp with ≥ +0.25 pp in each seed, **or** every arm
  reaching that level fails P4 or P5.
* **Inconclusive** — everything else (e.g. ΔAcc in [0.5, 1.0) pp with healthy
  calibration; positive in one seed only; sign disagreement between seeds; beats
  base but not Vector Scaling / the logit probe).

No informal sequential testing, no early stopping on an attractive intermediate
result, no significance claims from two seeds.

## 10. Resource budget

One rtx4090 GPU job per seed (≤ 3 h wall, ≤ 48 GB RAM); ≈ 180 000 forward passes
of ResNet-101 per seed plus per-layer neighbour search; stored: per-cell distance
arrays (`N×13×100` fp16) and per-arm probabilities. Corrected baselines: one
`run_unified_benchmark.py` clean fit + 12 evaluation-only cells per seed with the
reduced method set (`scripts/corrected_v2_*.sbatch`).

## 11. Code map

| file | role |
|---|---|
| `utils/preprocessing_protocol.py` | authoritative preprocessing specification + compatibility guard |
| `Experiments/preprocessing_diagnostic.py` | paired legacy-vs-corrected clean-input diagnostic |
| `Experiments/layer_selection_pilot.py` (`Calibrators/layer_readouts.py` for readout math) | pilot runner |
| `Experiments/aggregate_layer_pilot.py` | per-cell → per-seed aggregation, verdict per §9 |
| `tests/test_layer_selection_pilot.py` | properties in §12 |
| `scripts/corrected_v2_*.sbatch`, `scripts/layer_pilot_*.sbatch` | Slurm |

## 12. Properties asserted by tests

β=0 ⇒ native-DAC probabilities; equal-weight aggregation is a plain mean; greedy
traces are nested, deterministic and record every rejected candidate; the
depth-spaced sets are as declared; candidate list has ≥ 8 distinct non-logit
sources and no aliases; `fit_*` are unreachable from the evaluation path;
permuted labels preserve class counts; Family-B average is a valid probability
vector; probe parameter counts; corruption evaluation never influences a selected
set, β, λ or T (asserted by source-order/AST checks as in the FV-DAC tests).

---

## 13. Results (appended 2026-09-21; §§0–12 unedited)

Authoritative artifacts: `results/layer_pilot/checkpoint_seed{2,4}/{frozen_state.json,
selection_trace.json,<cell>/cell_metrics.json,<cell>/per_sample.npz}`,
`results/layer_pilot/aggregate/{summary.json,verdict.json,per_cell.csv,
per_seed_arm.csv,by_corruption.csv,by_severity.csv}`, sensitivity in
`results/layer_pilot/spatial_grid2/`. Frozen-state hashes: seed 2
`209405a4…d1596b`, seed 4 `03e4572f…b25c7`. Every cell asserted `β=0 ⇒ native
DAC` exactly (max deviation 0.0) and `evaluation_only: true`.
Wall time ≈ 90 s fit + ≈ 10 s per cell per seed; reference search ≈ 0.08–0.12 s per
layer per 10 000 queries (both label tables, 4090).

**Verdict (frozen §9 rule, `aggregate/verdict.json`):
`no_material_evidence_to_continue_tested_family`.** No candidate arm (greedy L ∈
{4,6,8}, either family) reaches seed-averaged ΔAcc ≥ +0.5 pp with ≥ +0.25 pp in each
seed; the best candidate is `B_greedy_L4` at **+0.095 pp** and `A_greedy_L4` at **+0.076 pp**.
This is a statement about the tested combination only (GAP-pooled, unscaled
equal-weight distance mean, K_c=5, these two readouts, two checkpoints, four
corruption families); it is not an information-theoretic null.

Mean over the 12 corruption cells, seeds 2 / 4 and their average (ΔAcc in pp vs base;
NLL/ECE are seed averages; W/H/U summed over both seeds × 12 cells):

| arm | ΔAcc s2 / s4 | avg | NLL | ECE | W / H / U | cells W>H (of 24) | clean-test ΔAcc |
|---|---|---|---|---|---|---|---|
| base | – | 0 | 2.348 | 0.191 | – | – | – |
| Temperature Scaling | 0 / 0 | 0 | 2.275 | 0.145 | 0 / 0 / 0 | – | 0 |
| **corrected native DAC** | 0 / 0 | 0 | **2.255** | **0.132** | 0 / 0 / 0 | – | 0 |
| Vector Scaling | +0.098 / +0.282 | +0.190 | 2.407 | 0.196 | 4170 / 3714 / 13257 | 19 | +0.32 |
| A greedy L=1 | −0.026 / +0.075 | +0.025 | 2.403 | 0.209 | 3207 / 3148 / 12697 | 13 | +0.54 |
| A greedy L=4 | +0.043 / +0.109 | +0.076 | 2.393 | 0.214 | 2526 / 2344 / 9699 | 14 | +0.56 |
| A greedy L=6 | +0.007 / −0.023 | −0.008 | 2.393 | 0.208 | 2712 / 2732 / 10823 | 12 | +0.45 |
| A greedy L=8 | +0.008 / −0.057 | −0.024 | 2.363 | 0.200 | 2615 / 2673 / 10553 | 10 | +0.35 |
| A depth-spaced L=1/4/6/8 | | +0.043 / +0.089 / +0.094 / +0.091 | 2.331 / 2.338 / 2.341 / 2.346 | 0.187 / 0.196 / 0.200 / 0.203 | 327/225/830 ; 699/486/2321 ; 878/653/3007 ; 1051/833/3710 | 16 / 18 / 16 / 17 | +0.01 / +0.08 / +0.18 / +0.20 |
| A permuted-label (greedy sets) | ≈0 | ≈0 | 2.255 | 0.132 | ≈0 | – | ≈0 |
| A logit-space source (L=1) | +0.031 / +0.060 | +0.045 | 2.339 | 0.193 | 394 / 285 / 1253 | 16 | +0.07 |
| B greedy L=1 | −2.413 / −0.318 | −1.366 | 2.369 | 0.156 | 6455 / 9733 / 28837 | 2 | −0.29 |
| B greedy L=4 | +0.088 / +0.102 | +0.095 | **2.201** | **0.115** | 6142 / 5914 / 22333 | 14 | +0.96 |
| B greedy L=6 | −0.085 / −0.162 | −0.124 | 2.207 | 0.102 | 8742 / 9039 / 32196 | 14 | +1.22 |
| B greedy L=8 | −0.735 / −1.033 | −0.884 | 2.276 | 0.114 | 9774 / 11896 / 38886 | 11 | +0.74 |
| B depth-spaced L=1/4/6/8 | | −0.066 / −1.377 / −1.579 / −2.224 | 2.393 / 2.412 / 2.420 / 2.421 | 0.201 / 0.165 / 0.168 / 0.163 | 3468/3627/13430 ; 8818/12124/39722 ; 8549/12339/40546 ; 10260/15598/45466 | 10 / 10 / 9 / 2 | +0.10 / +0.24 / +0.49 / −0.24 |
| B final-representation probe (`layer4.2`) | −0.081 / −0.052 | −0.066 | 2.393 | 0.201 | 3468 / 3627 / 13430 | 10 | +0.10 |
| B full-logit probe | −0.109 / +0.188 | +0.039 | 2.648 | 0.240 | 5317 / 5223 / 19359 | 11 | +0.08 |

(W/H/U for each depth-spaced L are listed in order L=1/4/6/8, W/H/U per group.)

**Selected sets (candidate indices, see §4.1).** Family A seed 2: `[8,9,7,6,5,1,3,0]`,
seed 4: `[8,7,6,9,5,4,3,2]`, i.e. L=1 → `layer3.22` in both seeds; L=4 → `{layer3.22, layer4.0,
layer3.17, layer3.12}` (s2) / `{layer3.22, layer3.17, layer3.12, layer4.0}` (s4).
Family B seed 2: `[9,10,8,11,7,6,5,4]`, seed 4: `[10,8,9,11,7,6,5,4]`, L=4 → `{layer4.0,
layer4.1, layer3.22, layer4.2}` (both seeds, order differs). Full trace (every rejected
candidate's score at every step): `selection_trace.json`.
**Inner-SELECT NLL along the greedy path is not monotone.** Family A: s2
0.8085→0.8066→0.8051→0.8055→0.8077→0.8098→0.8123→0.8145 (minimum at L=3); s4
0.8302→0.8308→0.8327→0.8334→0.8333→0.8346→0.8368→0.8393 (minimum at L=1). Family B: s2
minimum at L=3 (0.7446), s4 at L=4 (0.7796) and rising to 0.819/0.857 at L=8. On the
selection criterion itself, adding layers beyond 3–4 **worsens** NLL.

**Sensitivity — 2×2 spatial pooling, Family A (pre-declared, selection not re-run;
`spatial_grid2/`).** Seed-averaged ΔAcc: L=1 **+0.443**, L=4 +0.368, L=6 +0.312, L=8
+0.327 pp (seed 2: 0.374/0.353/0.307/0.358; seed 4: 0.513/0.383/0.318/0.296); cells
W>H 19/17/16/14 of 24; NLL 2.383/2.375/2.373/2.341, ECE 0.211/0.217/0.211/0.204
(vs native DAC 2.255 / 0.132). Compared with the same layers under GAP (ΔAcc
+0.025/+0.076/−0.008/−0.024) the accuracy effect is larger by ≈0.3–0.4 pp at every L, but
it still misses the +0.5 pp floor of §9 and does **not** preserve native DAC's calibration.
This sensitivity is **not** part of the §9 verdict rule.

## 14. Interpretation (appended 2026-09-21; separate from §13's measurements)

Labels: **measured** = in the tables above; **interpretation** = my reading, not
tested further; **unresolved** = the pilot cannot say.

1. *Layer choice (A).* **Measured:** NLL-greedy selection picked `layer3.22` first in both
   seeds and produced sets with β ≈ 19–27, whereas the depth-spaced control had β ≈ 1–9.
   On accuracy the greedy arms (+0.076 / −0.008 / −0.024 pp at L=4/6/8) are not better than
   depth-spaced (+0.089 / +0.094 / +0.091); on NLL and ECE depth-spaced is better
   (2.34 vs 2.39; 0.19–0.20 vs 0.20–0.21). **Interpretation:** clean-NLL selection did not
   transfer to shift; more layers gave no monotone gain in either family (Family A greedy
   +0.025 → +0.076 → −0.008 → −0.024; Family B greedy +0.095 at L=4 then falls to −0.88 at L=8;
   the inner-SELECT NLL itself is minimal at L = 1–4). This matches the T3 mechanism
   (`Theory Plan` §T3): within-class correlation of layer statistics is ρ ≈ 0.6
   (exploratory, post-hoc, `Experiments/layer_pilot_exploratory_diagnostics.py`), so an added layer
   must be at least ≈ 0.8–0.95 as informative as the current mean to help.
2. *Aggregation / readout (C, D).* **Measured:** the only arm with a probability-quality gain is
   Family B L=4 (NLL 2.201, ECE 0.115 vs native DAC 2.255 / 0.132) at ΔAcc +0.095 pp;
   Vector Scaling — a logit-only class-wise map with no representation access — has the
   highest decision effect (+0.190 pp, W>H in 19/24 cells). **Interpretation:** no accuracy
   effect in this pilot is separable from what a logit-only readout achieves (P2 fails for every
   candidate). The Family-B gain is a calibration-of-an-average effect of four deep
   probes, not evidence for distance geometry. The full-logit probe is a weak control
   (10 101 parameters, 2 500 fitting rows; NLL 2.65).
3. *Geometry specificity.* **Measured:** the permuted-label arm is ≈ 0 everywhere
   (β = 0 in 5 of 8 seed×L fits, tiny otherwise); the real-label Family-A arms move
   6–8 % of argmaxes at β ≈ 20 but repair 2.4–2.9 % of base errors (L=4, 8), with ≈ 67 % of flips
   wrong→different-wrong; base-error enrichment of flips vs a top-2-margin-matched
   expectation is only 1.06–1.09 (share on base errors 0.82 vs 0.78 expected).
   **Interpretation:** the correction is label-aligned but mostly explained by base margin
   and prevalence; it is not evidence of "geometric error detection". **Unresolved:** whether
   a *DAC-specific* mechanism exists separately from a generic weak class-conditional kNN signal.
4. *Transfer (E).* **Measured:** the sign of the effect depends on the corruption family:
   greedy Family A L=4 is −0.68 pp on `gaussian_noise` and +0.68 pp on `jpeg_compression`;
   Family-B L=8 is −3.4 pp on noise; effects shrink with severity (Family A L=4:
   +0.20 / +0.11 / −0.08 pp at severities 1/3/5). Clean-test gains (+0.35–0.56 pp for greedy
   Family A) do not predict corruption gains. **Interpretation:** conditional utility is not
   stable across shift types (`Theory Plan` §T5 P5.1).
5. *Pooling (B).* **Measured:** the pre-declared 2×2 sensitivity raises Family-A ΔAcc by
   ≈ 0.3–0.4 pp at every L (best +0.443 pp at L=1; W>H in 19/24 cells) without reaching
   +0.5 pp and without preserving native DAC's calibration (ECE 0.21 vs 0.13).
   **Interpretation / hypothesis-generating:** GAP may discard decision-relevant spatial
   structure in the class-conditional distances. One alternative pooling was tested on GAP-selected
   layers; this is *not* evidence that the un-pooled tensor helps and it must not be promoted
   to a primary method on this run.
6. *Preprocessing (F).* Corrected protocol throughout. Not a differentiator between
   these arms; the closed FV-DAC verdict stands as a legacy-protocol result.

**Scope:** 2 checkpoint seeds, 4 of 15 corruption families (previously inspected
development conditions), one architecture/dataset, GAP (+ one 2×2 sensitivity),
K_c = 5 inherited from a legacy-protocol pilot, unscaled equal-weight distance mean,
linear probes only. No significance test was run (24 seed × cell pairs are not
independent). Nothing here shows that layers or distances cannot help; it shows that this
tested family gave no material accuracy gain over a logit-only readout and paid
a calibration cost.

**Next bounded decision (for the researcher, not taken here).** Either (i) close this
family under GAP and spend the next slot on the T1 headroom measurement (cross-fitted
logit vs logit+geometry probe accuracy) — cheap, uses stored arrays; or (ii) a *new*
preregistration with the 2×2 (or one fixed spatial) pooling as primary a priori, one layer
(L = 1, `layer3.22`), a calibration-preserving objective declared up front, held-out
seeds and held-out corruption families, and thresholds written before running. Do not
widen K_c, add capacity, or rescore this pilot with new criteria.

**Cross-check vs the corrected benchmark (informational).** `Experiments/crosscheck_layer_pilot_vs_benchmark.py`
compares the pilot's base / native DAC / Temperature Scaling / Vector Scaling rows with the benchmark's own
`summary_metrics.json` for the same checkpoint and cell. On the two **clean** cells (8 method rows): max |Δacc| = 1.0e-4
(one sample), max |ΔNLL| = 3.2e-5, max |ΔECE| = 5.6e-4 (TF32 vs strict-fp32 forward passes). The 12 corruption cells per seed of the
benchmark (`scripts/corrected_v2_evaluate_corruption.sbatch`, job array 21533078) were still pending behind the benchmark's clean fit
(21532856, which also runs the always-on RGCL/GC-DAC paths) when this section was written; the pilot itself does not depend on them
(it builds each cell from CIFAR-100-C with the corrected transform, and records pixel/label identity against the benchmark cell when that
cell exists). Re-run the cross-check script after they complete; results are written to `results/layer_pilot/aggregate/crosscheck_vs_benchmark.json`.

## 15. Corrections (appended 2026-09-21, second pass; §§0–14 unedited)

Interpretation corrections are recorded in `docs/residual_evidence_study_spec.md` §0 and in the Research Brain theory plan
("Corrections appended 2026-09-21"): the practical (not literal-zero) meaning of "no material evidence"; probe-averaging attribution untested;
marginal-AUC statements do not bound incremental information; the layer-addition threshold is model-dependent; JL sufficient-dimension vacuity
is not an impossibility; the Bayes-accuracy identity requires a multiclass action set. The §13 measurements and the §9 verdict are unchanged.

**§15 addendum (2026-09-21).** The benchmark fits Temperature Scaling and Vector Scaling on the *whole* validation split; the pilot applied those frozen pickles, so the pilot's TS/VS baseline rows saw the pilot's inner-SELECT rows (a baseline-only role incompatibility; no evaluation labels involved; verdict unaffected). `docs/residual_evidence_study_spec.md` §16.6 refits TS/VS on FIT rows only.
