---
type: failure_mode
status: open
project: rgc-shift
scope: corruption_only_in_domain_unmeasured
integrity_flags: [protocol-contingent-k_vote, gate-code-not-installed-locally]
tags: [reliability, distribution-shift, gating]
---

# Validation-fitted neighbourhood reliability features fail under corruption

> Renamed 2026-09-15. The previous title ("clean-fitted ... do not transfer")
> claimed two things the evidence does not support: (a) the primary fit was on
> the **validation** cell, not the clean cell (clean was used only for mechanism
> diagnostics); (b) *transfer* failure requires in-domain discrimination to have
> been measured and to have been adequate. It was never measured. Restore a
> transfer claim only after [[H-RGC-01 Reliability mapping shifts under corruption]]
> reports in-domain AUROC/AUPRC.

## Failure

A gate using:
- head-neighbour agreement,
- kNN radius,
- neighbour margin,
- neighbourhood concentration

fitted on the validation cell, fails to identify the samples where geometry
should override the head under CIFAR-100-C.

## Evidence

[[2026-09-15 RGC Shift Recoverability]]

Oracle headroom exists, but gate capture is approximately zero or negative
(-0.1% per-class, -2.3% global kNN; gate - head = +0.002 pp, CI [-0.010, +0.014]).

## Scope limits (do not over-read)

1. **In-domain discrimination was never measured.** This note records a failure
   *under corruption*, not a demonstrated failure *of transfer*. Weak observables
   in-domain is an equally live explanation.
2. **Protocol-contingent.** The run recorded `k_vote=200` where the
   pre-registration pinned `k_vote=50`. Averaging head-neighbour agreement and
   concentration over 200 neighbours may wash out the local signal the gate
   needed. **[inference]** - this mechanism is reasoning, not a measured result.
3. One benchmark, one architecture (ResNet-101 / CIFAR-100-C), five seeds.

## Published counter-evidence

[[Beyond In-Domain Scenarios - Robust Density-Aware Calibration]] (DAC) reports
that hidden-layer kNN density fitted **in-domain** does improve calibration under
CIFAR-C. That is the closest published counterexample to a strong reading of this
note.

It is not a direct contradiction: DAC's target is the calibration of the head's
own confidence and it never changes the predicted class, whereas this note is
about *routing* between head and geometry. But it does mean "neighbourhood
statistics stop working under corruption" is too broad a statement of the
failure. The failure here is specific: these four statistics, fitted on
validation, at k_vote=200, for the routing decision.

## Competing explanations

1. Covariate shift: feature distributions move.
2. Concept shift: P(geometry correct | features) changes.
3. Missing observables: current features omit the relevant signal (not excluded).
4. Too few decisive examples in the fitting split.
5. Representation instability across seeds.
6. Protocol deviation (`k_vote`) destroyed the locality of two of the four features.

## What would falsify this failure mode

A validation-fitted gate using the same four observables, run at the
pre-registered `k_vote=50`, that consistently captures meaningful headroom on a
wider corruption set / architecture set at a matched pre-registered operating
point.

## Addendum 2026-09-15 — gate-fitting code is missing from the canonical repo

While scoping the G3 k_vote=50 audit (see [[2026-09-15 G3 Headroom Null]] "Next
action"), the code that fitted this gate is not reproducible: `results/
seed{N}/recoverability_{perclass,globalknn}.json`'s `gate` block (features
`head_neighbour_agreement`, `knn_radius`, `neighbour_margin`,
`neighbourhood_concentration`; a fitted `threshold`; `n_train_decisive`) has
no corresponding computation anywhere in the canonical
`GeometricFullCalibration` repo. Checked: `scripts/export_recoverability.py`
(no gate-fitting code), `scripts/aggregate_recoverability.py` (only reads
`result.get("gate")`, does not compute it), and `git log --all -S
"neighbourhood_concentration"` across full history (zero commits — this
string was never committed to this repo). Integrity label:
~~verified implementation issue~~ **CORRECTED 2026-09-15, same day**: this
was wrong. The gate-fitting code is not missing from the repository — it
lives in `GeometricFullCalibration/rgc-shift.zip`, a gitignored (`*.zip`)
package `rgc_shift` (pinned `git ff81032f...`) that
`scripts/aggregate_recoverability.py` and `scripts/export_recoverability.py`
both import directly. `git log -S "neighbourhood_concentration"` found
nothing because that string lives in `rgc_shift`'s own separate embedded git
repo, not in `GeometricFullCalibration`'s history — a search-scope miss, not
a missing-code finding. See the correction section of
[[2026-09-15 G3 Controlled Complementarity]] for the full account, including
that `rgc_shift.recoverability.neighbour_statistics()` takes k implicitly
from the input array width, so a literal k=50 re-run of two of the four real
gate features (`head_neighbour_agreement`, `neighbourhood_concentration`) is
possible with the *exact* production formula — not attempted yet. The real
remaining gap is smaller: `rgc_shift` is not installed in any available
conda environment on this machine, so it cannot currently be run end-to-end
without unzipping and installing it first.

A fresh (explicitly non-reproducing) univariate audit of the recomputable
vote-based features at k=50 vs k=200 was run instead — see
[[2026-09-15 G3 k_vote Audit]]. Result: small, mixed effect (margin and a
newly-defined vote-concentration feature improve slightly at k=50, ~1-2
AUROC points; the agreement feature gets very slightly worse). `knn_radius`
could not be audited (raw per-neighbour distances aren't cached). This does
not strongly support or refute the k_vote=200-washed-out-signal mechanism
listed in "Competing explanations" below — treat that explanation as
weakly, not strongly, supported. **Caveat added 2026-09-15**: that audit's
`agreement` feature used a binary majority-vote-match definition, which
turns out to differ from the real `head_neighbour_agreement` (a continuous
neighbour-membership fraction) — so its negative-delta finding measures a
different feature than the real gate used, not an approximation of it. The
`margin` and `vote_concentration` findings are unaffected (formulas
confirmed correct against the real source).

## Related

- [[Simple neighbourhood-statistics gate for geometric correction]]
- [[H-RGC-01 Reliability mapping shifts under corruption]]
- [[H-RGC-02 Richer representation features predict geometric reliability]]

## Historical precedents (related lineage, not the same mechanism)

Two earlier projects in the same research lineage found related but
mechanistically distinct failures of "trust the geometric signal here":

- [[Clean-fitted geometric confidence mappings can reverse under synthetic corruption]]
  (original geometric-separation project, raw-pixel geometry, custom
  synthetic corruptions).
- [[Composite geometry scores do not reliably select calibration layers]]
  (RGC project, an unsupervised layer-selection score, backup-sourced
  evidence).

Do not read these three as one phenomenon — see [[Research Lineage]] for the
explicit statement that this is a recurring *question*, not an established
common mechanism.
