---
type: hypothesis
status: proposed
project: rgc-shift
benchmark: CIFAR-100-C
novelty: unknown
tags: [shift, reliability]
---

# H-RGC-01 Reliability mapping shifts under corruption

## Formal statement

Let R be the current label-free reliability feature vector and G indicate
whether geometry is preferable to the head.

The mapping

P(G=1 | R)

estimated on the validation cell is not invariant under corruption.
(The primary fit was validation, not clean; earlier notes said "clean-fitted".)

## Why it follows from evidence

Substantial oracle headroom exists, yet a validation-fitted gate captures
essentially none of it.

This hypothesis is currently **blocking** two notes: until its in-domain
discrimination numbers exist,
[[Validation-fitted neighbourhood reliability features fail under corruption]]
may not be restated as a *transfer* failure, and the kill in
[[Simple neighbourhood-statistics gate for geometric correction]] stays
protocol-contingent. Run the diagnostic at the pre-registered `k_vote=50`.

## Competing explanation

R simply contains too little information even in-domain. Note that
[[Beyond In-Domain Scenarios - Robust Density-Aware Calibration]] reports that
in-domain-fitted hidden-layer kNN density *does* help calibration under CIFAR-C,
which argues against a blanket "neighbourhood features do not survive corruption"
reading - though DAC's target is calibration and it never moves the argmax.

## Decisive experiment

Measure gate discrimination / calibration separately on validation and each
corruption cell using labels for diagnosis only. Compare:
- AUROC / AUPRC for geometry-better-than-head;
- calibration curves;
- conditional prevalence;
- feature distribution shift.

## Kill criterion

If discrimination is already near chance on validation, this is not primarily a
shift problem; the observables are simply weak.

## Go criterion

Good validation discrimination followed by systematic corruption collapse.
