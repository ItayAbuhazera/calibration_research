# Baseline fairness and the incremental-geometry test

Companion to `utils/method_metadata.py` (the canonical semantics registry) and
`Experiments/audit_baseline_fairness.py` (the machine-checked table). This
file records the judgements that a table cannot express.

The scientific question the Phase 0/1 benchmark must answer:

> Given the same frozen classifier and the same calibration data, does adding
> representation-space geometric information improve confidence
> ranking/calibration/selective risk beyond what can be extracted from
> ordinary model outputs alone?

A positive answer is only interesting if the non-geometric comparison is hard
to beat. The three sub-questions must not be conflated:

| # | Question | Primary metrics | Comparator |
|---|---|---|---|
| 1 | Calibration | top-label ECE, adaptive ECE | `top_label_isotonic` |
| 2 | Confidence ranking / selective prediction | correctness AUROC, AURC, excess AURC, coverage at matched risk | `top_label_isotonic` |
| 3 | Decision improvement | accuracy, changed-to-correct, changed-to-wrong, net flips | `glad_pi_zero_geometry` |

Question 2 is the one that matters most and the one the benchmark was
previously unable to answer at all: correctness AUROC, AURC and
risk-coverage did not exist in the codebase before
`utils/selective_metrics.py`.

---

## 1. Why `top_label_isotonic` is the right non-geometric reference for Q1/Q2

Every scalar geometric method (`gc_dac`, `rgcl`, `rgcc`, `gc_tulip`,
`mahalanobis_confidence`, the Study-B cells) has the same shape: a per-sample
scalar statistic, rank/isotonic-mapped against binary correctness on the
calibration split, assigned to the base predicted class.

`top_label_isotonic` is that identical pipeline with the geometry removed —
the statistic is the model's own max softmax probability. Same mapper, same
fitting split, same objective, same (zero) hyperparameter budget, same metric
bucket, same effective prediction. The only difference is the input
statistic, which is exactly the ablation Q1/Q2 require.

Comparing a geometric scalar method against **raw MSP** (`base_model`) instead
would be the weak-baseline mistake: it would credit geometry for the isotonic
recalibration that any method gets for free.

## 2. Is `glad_pi` vs `glad_pi_zero_geometry` a genuine capacity-matched test?

Audited directly against `Calibrators/glad_pi.py`. **Matched:**

- **Architecture.** `_CorrectionNet.INPUT_DIM` is 5 in both arms
  (`[logit_k, distance_k, max_prob, entropy, top_margin]`). The
  zero-geometry arm does not remove the distance input; it zeroes it, so the
  layer shapes and the **trainable parameter count are identical**.
- **Mechanism of the ablation.** `zero_geometry` has exactly one behavioural
  use, in `_to_tensors`: `dist_t = torch.zeros_like(dist_t)`. Its only other
  occurrences are the constructor assignment and `get_params` reporting.
  Enforced by a test.
- **Optimization.** Same `lr`, `weight_decay`, `epochs`, `patience`, `margin`,
  optimizer (Adam), full-batch schedule and early-stopping rule.
- **Data.** Same `inner_fit` / `inner_select` deterministic split, from the
  same `inner_val_fraction` and `inner_val_seed`.
- **Selection.** Same `beta_grid`, same NLL-tolerance budget, same
  `net_flips >= 0` filter, same tie-break order, same beta=0 fallback rule.
- **No distributional artifact from zeroing.** Distances are standardized on
  the fit split *before* `_to_tensors` zeroes them, so the geometry arm's
  distance channel is approximately zero-mean/unit-variance and the control's
  constant 0 sits at that channel's mean. Zeroing does not introduce an
  out-of-range input.

**Two caveats that weaken it, neither fatal:**

1. **No seeding of network initialization.** Neither `glad_pi.py` nor the
   runner calls `torch.manual_seed`, and each arm trains `len(beta_grid)`
   fresh `_CorrectionNet`s from the ambient RNG. The two arms therefore get
   *different* random initializations, decided by call order. With one run per
   arm per checkpoint, part of any observed arm difference is initialization
   luck. Mitigation: five checkpoint seeds, and treat a difference smaller
   than the across-seed spread as noise. A proper fix is to seed each arm
   identically per beta; that changes fitted state, so it is **not** applied
   here.
2. **Early stopping is on training loss**, not the select split
   (`_train_one_beta` tracks `loss.item()` on the fit split). Identical in
   both arms, so it does not bias the comparison, but it means neither arm is
   trained to its best achievable select-split performance.

**Verdict: yes for question 3 (decision improvement).** It is a defensible,
capacity-matched isolation of geometry for the decision-correction task and
should be the primary incremental-geometry comparison there.

**Verdict: no for question 2 (confidence ranking).** See below.

## 3. The remaining gap: no capacity-matched regular-only *confidence* model

`glad_pi_zero_geometry` is a matched control for a **decision corrector**, not
for a **confidence predictor**. Two reasons it does not close question 2:

1. **Its objective is not confidence ranking.** Both arms optimize
   NLL + an asymmetric margin term on base-wrong samples, and beta is selected
   on `net_flips`. Neither arm is trained or selected for AUROC/AURC.
2. **Its regular-signal inputs are impoverished.** The shared per-class MLP
   sees `logit_k` per class, but the only cross-class context is
   `[max_prob, entropy, top_margin]`, all computed from the softmax. Softmax
   features are invariant to adding a constant to every logit, so this arm
   **cannot represent energy / log-sum-exp or logit norm** — two of the
   strongest known output-space correctness and shift signals. A shared
   per-class MLP cannot recover them either, since it cannot sum over classes.

So the current strongest non-geometric confidence signal in the benchmark is
`top_label_isotonic`, i.e. isotonic-calibrated MSP alone. Beating MSP is a low
bar for a confidence-ranking claim.

**Minimal recommended addition (one method, not a zoo):** a *regular-only
scalar confidence baseline* that is capacity-matched to the geometric scalar
methods' pipeline — the same isotonic mapper against binary correctness on the
same calibration split, fed a small fixed feature vector of ordinary
output-space signals:

```
[ MSP, entropy, logit margin, probability margin,
  energy (log-sum-exp of logits), logit L2 norm, top-k probability mass ]
```

aggregated by a low-capacity learned combiner (e.g. logistic regression or a
tiny MLP) whose parameter count is stated and whose hyperparameter budget is
declared in the registry. Handcrafted signals beyond these are unnecessary:
full logits plus energy and logit norm subsume the usual list, and the point
is a *strong* baseline, not an exhaustive one.

Without it, a geometric win on AUROC/AURC is only a win over calibrated MSP,
and a reviewer can reasonably object that an ordinary-signal model of equal
capacity was never tried. This addition changes the frozen method set, so it
needs an explicit decision before Phase 2.

## 4. Information-budget checks that are machine-enforced

`Experiments/audit_baseline_fairness.py` fails if any of these break:

- a geometric method fits on a split no non-geometric method uses;
- any fitting or selection split mentions test or corruption data;
- the two GLAD-PI arms disagree on split, objective, hyperparameter budget or
  input access on any axis other than `uses_internal_representation`;
- a method in a run has no registry entry.

As of this writing all checks pass for the declared Phase 0/1 method set.

## 5. What would count as geometry winning

Not "lower ECE". Concretely, on the matched comparator for each question:

- **Q2 (the main claim):** lower AURC *and* higher correctness AUROC than
  `top_label_isotonic`, with positive `delta_coverage` in
  `coverage_at_matched_risk`, consistent in sign across all five checkpoint
  seeds, and persisting under CIFAR-C shift.
- **Q3:** positive net flips against `glad_pi_zero_geometry` at equal or better
  NLL, again consistent across seeds.
- **Q1 alone is not sufficient.** A method can improve ECE while destroying
  the confidence ranking — and in the seed 4 clean rescoring, `gc_dac` and
  `rgcl` do exactly that (ECE 0.0087 / 0.0078, the best in the run, with
  correctness AUROC 0.596 / 0.428 against 0.865 for calibrated MSP; `rgcl` is
  below chance). ECE is invariant to a monotone-preserving squash of
  confidence toward the accuracy; it does not reward discrimination.

## 6. Cross-method comparability warning for AUROC/AURC

For a method that changes the effective prediction, `correct` is computed
against *its own* decisions, so its AUROC/AURC are not directly comparable
with a method of different accuracy. `odir_dirichlet` in the seed 4 clean
rescoring is the cautionary example: correctness AUROC 0.937 — the highest in
the run — with accuracy 0.404, because ranking errors is easy when there are
many of them. Use `excess_aurc` (AURC minus the AURC of a perfect ranking at
the same error count) for cross-method comparison, and restrict paired
per-sample complementarity to methods that share an effective prediction.
