---
type: idea
status: tested-negative
novelty: unknown
top_tier_potential: unknown
benchmark: CIFAR-100 / CIFAR-100-C, ResNet-101, checkpoint seeds 1-5
project: Full-Vector Geometric Calibration
tags: [dac, density, hidden-representations, full-vector, decision-change, shift]
---

# Full-Vector Density-Aware Calibration (FV-DAC)

> Created 2026-09-20 at the researcher's explicit request (see
> `Research/CLAUDE.md`: `06_Ideas/` is never auto-created). This is a **new
> research branch**, not a rescue attempt for RGCL or for any prior
> full-vector geometric fusion result. Prior negative results in
> [[Full-Vector Geometric Calibration]] stand unchanged.

## One-sentence contribution

Test whether **class-conditioning** the reference-bank search of
[[Beyond In-Domain Scenarios - Robust Density-Aware Calibration]] turns its
provably accuracy-preserving scalar-temperature mechanism into a
low-capacity full-vector correction that can change the decision under
corruption shift.

## Phenomenon

Native DAC produces a sample-dependent **scalar** temperature from
hidden-layer neighbourhood density,

```text
S(x,w) = w_0 + sum_l w_l * s_l(x)
q_DAC(x) = softmax( z(x) / S(x,w) )
```

and for `S > 0` this is order-preserving, so

```text
argmax_k q_DAC(x) = argmax_k z(x).
```

That is structural, not empirical — it is the same algebra recorded in
[[Sample-dependent scalar temperature cannot change the predicted class]].
It is confirmed numerically in this repository: on
`results/studyAB/phase0/evaluation/checkpoint_seed4/`, `native_dac` accuracy
equals `base_model` accuracy exactly on clean (0.7631) and on every
corruption cell inspected (e.g. `gaussian_noise_s3` 0.2214,
`fog_s5` 0.3693), while its ECE improves substantially
(0.3770 -> 0.2752 and 0.2783 -> 0.2152 respectively).

So DAC's density signal demonstrably carries information that is useful
under shift, and that information is currently spent entirely on a scalar.

```text
NATIVE DAC                          PROPOSED EXTENSION
representation density              same DAC representation machinery
        |                                   |
    scalar S(x)                     class-conditioned distance vector
        |                                   |
  temperature scaling               minimal class-wise correction
        |                                   |
   argmax fixed                     argmax may change
```

## Core hypothesis

See [[H-FVDAC-01 Class-conditioned DAC density enables decision correction]].

## Important qualification (do not overstate)

Native DAC's statistic `s_l(x)` is **class-agnostic**: it is a k-th
nearest-neighbour distance to the whole reference bank, and this
repository's frozen DAC state does not even retain bank labels. The
proposed extension **introduces reference labels**. Therefore the only
claim this experiment can support is:

> class-conditioning DAC's representation-density machinery yields useful
> decision corrections

and **not**

> DAC's original scalar statistic secretly contained class-specific
> information.

Those are different statements and must stay separate in every write-up.

## Why this may be new

DAC is explicitly accuracy-preserving and is presented as such; its own
audit note records "Can change argmax? No in its stated accuracy-preserving
formulation." The published follow-up space around DAC is about better
density estimators and better calibrators, not about removing the scalar
bottleneck. What is genuinely uncertain is whether a **one-parameter,
class-symmetric** correction on top of the *unchanged* DAC geometry is
enough to move decisions usefully — most prior attempts in this codebase
that moved decisions added far more capacity.

## Closest prior art

- [[Beyond In-Domain Scenarios - Robust Density-Aware Calibration]] — the
  direct parent. Same representations, same bank, same layer structure.
- [[Taking a Step Back with KCal]] — reference-based, class-conditional
  kernel density posterior. The strongest "this is just non-parametric
  class-conditional density" competitor; it is already a benchmark method.
- [[Beyond Temperature Scaling - Dirichlet Calibration]] — ordinary
  non-geometric class-wise calibration (Vector Scaling / ODIR). The
  "ordinary class-wise calibration explains it" competitor.
- [[Deep k-Nearest Neighbors]], [[To Trust Or Not To Trust A Classifier]] —
  the nearest-neighbour-classification explanation.
- [[Full-Vector Geometric Calibration]] — this repository's own prior
  full-vector attempts. **Related but distinct**: those fused RGCL/GC-DAC
  geometry into the probability vector; this one keeps DAC's exact operator
  and changes only the search domain.

## Benchmark

CIFAR-100 / CIFAR-100-C, ResNet-101, the five Phase 0/1 checkpoint seeds,
reusing the frozen Phase 0/1 split protocol and the frozen native-DAC state
at `results/studyAB/phase0/fitted_method/checkpoint_seed{S}/native_dac.pkl`.

## Matched operating point

`beta = 0` must reproduce native DAC **numerically**, not approximately.
Every arm shares native DAC's representations, pooling, L2 normalization,
distance convention, reference-bank membership and frozen `S(x)`.
Only the search domain becomes class-conditioned.

## Decisive POC

12 corruption cells (`gaussian_noise`, `defocus_blur`, `fog`,
`jpeg_compression` x severities 1/3/5) on checkpoint seed 4, evaluated
under evaluation-only frozen state, against a continuation rule written
down **before** those results are opened. See
[[2026-09-20 Full-Vector DAC POC]].

## Kill criterion

Recorded in full in [[2026-09-20 Full-Vector DAC POC]] and in
[[H-FVDAC-01 Class-conditioned DAC density enables decision correction]].
Summary: `beta -> 0`; `W <= H` under shift; negligible/nonpositive mean
corruption `DeltaAcc`; the permuted-bank-label control reproduces the
effect; Vector Scaling / ODIR or KCal / density-only explain it; gains
confined to one seed or one corruption; calibration damage large relative
to native DAC.

## Rejected design alternative — per-class temperature

`q_k ∝ exp(z_k / T_k(x))` is **deliberately not** the primary extension. It
breaks logit gauge invariance (adding a constant to all logits changes the
output), can promote strongly negative logits as `T_k` grows, and therefore
makes the result depend on an arbitrary logit origin. The chosen additive
form `softmax((z - beta*R(x))/S(x))` is invariant to a common logit shift
and has a single sign convention: smaller class-conditioned distance ->
less penalty -> more support.

## Reviewer #2 attack

1. "You added reference labels — of course a labelled kNN signal helps."
   Answered by the permuted-bank-label control and by the KCal /
   density-only comparisons, not by assertion.
2. "This is just Vector Scaling with extra steps." Answered by the
   non-geometric class-wise calibrators already in the benchmark.
3. "Nothing here needs hidden layers." Answered — partially — by
   `FV-DAC-logit-space`, which is a **mechanism control, not a strictly
   capacity-matched control** (5 hidden sources vs 1 output-space source).
4. "You tuned it until it worked." Answered by the predeclared `K_c` set,
   the NLL-only fitting objective, the disjoint fit/select splits and the
   pre-registered continuation rule.

## Current verdict

**Tested and stopped on 2026-09-20 by its own frozen continuation rule —
Outcome E.** See [[2026-09-20 Full-Vector DAC POC]].

The decisive POC ran on checkpoint seed 4. Four of five continuation
criteria passed: beta was alive (6.43), the direction was positive
(+0.14 pp mean corruption accuracy, W=653 vs H=483), the effect was
consistent (9/12 cells positive), and the permuted-bank-label control was
*exactly* null (beta=0, zero flips in all 12 cells). C5 failed: top-label
ECE came in +0.053 worse than native DAC against a 0.02 allowance, i.e. the
method spends essentially all of native DAC's calibration gain to buy an
effect roughly 10x smaller than the predeclared "interesting" scale.

Re-audited the same day: the first write-up's "the mechanism is real" was an
overclaim and is withdrawn. The defensible statement is that **a weak
label-aligned semantic signal is present in this realization**, on one
checkpoint, with intervention precision 0.186 (68% of flips go
wrong->different-wrong), an effect that *shrinks* with severity, and a
geometry-free Vector Scaling baseline reaching +0.086 pp against FV-DAC's
+0.142 pp. The scalar bottleneck **can** be removed this cheaply; what you
get for it is not worth the calibration, and the experiment does not
establish that the gain would survive another checkpoint.

`novelty: unknown` — never vetted for novelty, and there is now no reason
to.

## Update 2026-09-21 (appended at the researcher's instruction; verdict fields above untouched)
Follow-ups on this idea, all provenance-linked: normalization repair [[2026-09-21 Normalization Audit and Corrected Protocol]] (the original pilot ran under the legacy mixed-normalization protocol);
layer/pooling pilot [[2026-09-21 Layer-Selection Pilot]] (no material evidence for the tested family; exploratory 2×2 signal); prospective common-readout test
[[2026-09-21 Residual Evidence Study]] (**stopped by its frozen gate**: hidden-evidence procedure +0.102 pp vs output-evidence control +0.118 pp; no spatial or class-radius advantage; clean selection prefers a near-zero residual at n=2 500).
Net status of the idea: still no supported mechanism beyond logit-only recalibration; the diagnostic classifier used in the last study is a known readout construction and is not a novelty claim.
