---
type: hypothesis
status: not-supported-under-this-protocol
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C, ResNet-101, checkpoint seeds 1-5
novelty: unknown
tags: [dac, density, full-vector, decision-change, shift, H-FVDAC]
---

# H-FVDAC-01 — Class-conditioning DAC's representation-density operator enables decision correction

Parent idea: [[Full-Vector Density-Aware Calibration]].
Decisive experiment: [[2026-09-20 Full-Vector DAC POC]].

## Formal statement

Let `h_l(x)` be the DAC-preprocessed representation at selected layer `l`
(spatially averaged, L2-normalized), let `B_l` be the frozen DAC reference
bank with labels `y_i`, and let

```text
B_{l,k} = { h_l(x_i) : y_i = k }
r_{l,k}(x) = K_c-th nearest-neighbour distance( h_l(x), B_{l,k} )
alpha_l    = w_l_hat / sum_j w_j_hat          (frozen native DAC layer weights)
R_k(x)     = sum_l alpha_l * r_{l,k}(x)
```

Then with native DAC's frozen scalar `S_DAC(x)` unchanged,

```text
q_beta(x) = softmax( ( z(x) - beta * R(x) ) / S_DAC(x) ),   beta >= 0
```

**H-FVDAC-01.** There exists `beta > 0`, fitted on clean calibration data by
NLL alone, such that under CIFAR-100-C corruption shift

```text
Acc(q_beta) > Acc(base) = Acc(q_DAC)
```

with positive net useful flips `W - H > 0`, while native DAC's calibration
improvement is largely preserved.

### Shift sub-claim

The class-conditioned signal is expected to be **more** useful under
corruption than on clean data, because that is precisely the regime where
native DAC's class-agnostic density is already known (in this repository's
own Phase 0/1 artifacts) to carry usable information.

### Separate, weaker-supported sub-claim — do not merge with the above

**H-FVDAC-01-H (hidden-specific).** The useful class-conditioned geometry
requires the classifier's *internal* representations, i.e. hidden-space
FV-DAC beats the output-space analogue `FV-DAC-logit-space` built on
centered logits with the same bank, labels, `K_c`, beta-fitting and
correction equation.

This is a **separate** hypothesis. Failing it does not falsify H-FVDAC-01.

## Why it follows from evidence

Not deductively — this is a genuine open question. What the evidence
supports is that the question is worth asking:

- Native DAC is argmax-invariant by construction for `S > 0`
  ([[Sample-dependent scalar temperature cannot change the predicted class]]),
  so *any* accuracy gain from DAC's density signal is structurally
  unreachable in its published form.
- In this repository's frozen Phase 0/1 artifacts the DAC density signal
  measurably improves calibration under corruption
  (`checkpoint_seed4/gaussian_noise_s3`: ECE 0.3770 -> 0.2752), so the
  signal is not vacuous under shift.
- Whether that signal becomes *decision-relevant* once conditioned on
  reference labels is not established by either observation.

## Competing explanations (each has a control in the experiment)

1. **The label information, not the geometry, does the work.** Control:
   permuted-bank-label negative control (same features, same distances,
   same `K_c`, same beta protocol, semantics destroyed).
2. **Ordinary class-wise calibration explains it.** Controls: Vector
   Scaling, ODIR/Dirichlet.
3. **It is nearest-neighbour classification in disguise.** Controls: KCal,
   and the `argmin_k R_k(x)` density-only diagnostic.
4. **Output-space geometry suffices.** Control: `FV-DAC-logit-space`
   (mechanism control; see the capacity caveat below).
5. **DAC's calibration-optimal layer weights are not decision-optimal.**
   Arm: `FV-DAC-shared-layer` with `b_l >= 0` shared across classes.

## Benchmark

CIFAR-100 clean + CIFAR-100-C (15 standard corruptions), severities 1/3/5
as the prespecified primary grid, ResNet-101, checkpoint seeds 1-5, frozen
Phase 0/1 split protocol and frozen native-DAC state.

## Baselines

`base_model`, `native_dac`, `temperature_scaling`, `vector_scaling`,
`odir_dirichlet`, `kcal`, plus the FV-DAC controls above.

## Decisive experiment

[[2026-09-20 Full-Vector DAC POC]].

## Go criterion

All of:

- primary `beta_hat > 0` and not at the search boundary;
- mean corruption `DeltaAcc = Acc(FV-DAC) - Acc(base)` clearly positive
  with `W > H`;
- majority of corruption cells positive, stable or growing with severity;
- permuted-label control near null;
- FV-DAC above Vector Scaling and ODIR, and competitive with or better than
  KCal;
- NLL/Brier/ECE remain acceptable relative to native DAC;
- replicates across checkpoint seeds.

## Kill criterion

### Method-level (kills / weakens H-FVDAC-01 itself)

- primary `beta` collapses to zero across most checkpoints;
- `W <= H` under shift;
- mean corruption `DeltaAcc` negligible or nonpositive;
- the permuted-bank-label control reproduces the effect;
- Vector Scaling / ODIR explain the effect without geometry;
- KCal / density-only explain it so completely that FV-DAC adds nothing
  distinct;
- gains appear in only one seed or one isolated corruption;
- calibration damage is substantial relative to native DAC;
- the fitted correction is pathological on clean validation;
- the effect disappears after proper frozen-state replication.

### Hidden-representation-specific (kills / weakens H-FVDAC-01-H only)

- `FV-DAC-logit-space` matches or beats hidden FV-DAC.

This is **not** automatically a method-level kill. The ordering
`FV-DAC_Z > FV-DAC_H > Base` would falsify H-FVDAC-01-H while leaving
H-FVDAC-01 alive.

## Prior-art threats

KCal already provides a class-conditional reference-based posterior; if
FV-DAC cannot be distinguished from it, the contribution collapses to "a
worse KCal expressed in DAC's notation". Dirichlet/Vector Scaling provide
the non-geometric class-wise null. Both are already in the benchmark, so
the threat is testable rather than rhetorical.

## Comparison-asymmetry caveat (for H-FVDAC-01-H)

Hidden FV-DAC aggregates 5 representation sources (dims 64/256/512/1024/
2048); the logit-space arm has 1 source of dim 100. Both fit the same
number of free parameters (one `beta`), but they are **not** matched on
representation capacity. A positive hidden-vs-logit result must be
reported with that asymmetry stated, and must not be called definitive
evidence for hidden-specific geometry on its own.


---

## Verdict (2026-09-20, seed 4 only) — appended, nothing above revised

**H-FVDAC-01: not supported under this protocol.** See
[[2026-09-20 Full-Vector DAC POC]] for the numbers.

The go criterion required, among other things, "meaningful positive
corruption DeltaAcc" and "NLL/Brier/ECE remain acceptable relative to native
DAC". The measured effect was +0.14 pp mean corruption accuracy with
top-label ECE +0.053 worse than native DAC. The frozen continuation rule
returned STOP on criterion C5.

What survived, stated at the strength the evidence supports (this wording
replaces an earlier "the mechanism is real, not an artefact", which was an
overclaim and is withdrawn):

- **A weak label-aligned semantic signal is present in this realization.**
  The permuted-label control fitted `beta = 0` *exactly* and flipped zero
  samples in all 12 cells, while real labels gave `beta = 6.43` — a real
  asymmetry. But that shows the correction is not an automatic consequence
  of adding a parameter; it does **not** show the corruption gain is robust.
  One checkpoint, one permutation seed, and 12 cells that share a
  checkpoint, a bank and the same underlying images are not 12 independent
  replications.
- **The intervention is not accurate.** 68% of the primary arm's 3518 flips
  are wrong->different-wrong; intervention precision is **0.186**, not the
  0.575 decisive precision quoted in the first write-up. It repairs 1.08% of
  base errors.
- **The effect shrinks with severity** (+0.198 / +0.165 / +0.063 pp at
  sev 1/3/5), contradicting this hypothesis's own shift sub-claim.
- **Vector Scaling, with no geometry at all, reaches +0.086 pp** on the same
  cells against FV-DAC's +0.142 pp, so the geometric contribution is not
  cleanly separated from ordinary class-wise calibration.
- It is **too small to be useful**, and it trades away 82.5% of native DAC's
  ECE gain to achieve even that.

**H-FVDAC-01-H (hidden-specific) was NOT answered.** Hidden FV-DAC
(+0.142 pp) did exceed the logit-space mechanism control (+0.060 pp), so the
hidden-specific kill criterion did not trigger — but with both effects
negligible, one checkpoint, and no capacity matching (5 hidden sources of
dim 64-2048 vs 1 output-space source of dim 100), the comparison carries no
weight in either direction. Recorded as unanswered, not as weak support.

Evidence scope: **one checkpoint seed (4), 4 of 15 corruptions, severities
1/3/5, CIFAR-100 / ResNet-101 only.**

This does not revise [[Sample-dependent scalar temperature cannot change the predicted class]]
(which remains structurally true and is what motivated the experiment), and
it does not bear on any prior RGCL or full-vector geometric result.


### Additional limitation recorded on re-audit (2026-09-20)

Table 6 of the DAC paper was extracted and verified: the paper's ResNet layer
set for CIFAR-100 is `PRE-BLOCK, BLOCK-1..BLOCK-4, LOGITS` — it **includes
the logits layer**, which this repository's `native_dac` omits. This work is
therefore an extension of *the benchmark's frozen five-layer DAC
implementation*, not of the paper's complete layer set, and the
hidden-vs-logit contrast does not map onto a published design distinction.

The benchmark also carries a **train/test input-normalization mismatch**
(train/val CIFAR stats, clean *and* corrupted test ImageNet stats), so every
distance in this study is computed between a CIFAR-normalized reference bank
and ImageNet-normalized queries. The conclusion applies to that compound
configuration.
