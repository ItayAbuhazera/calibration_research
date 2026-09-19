---
type: experiment
status: completed
date: 2026-09-15
project: rgc-shift
benchmark: CIFAR-100-C
model: ResNet-101
seeds: [1, 2, 3, 4, 5]
preregistered: true
protocol_deviations: [k_vote, fitting_split]
deviation_status: unresolved
aliases: ["2026-09-15 RGC Shift Recoverability — ResNet-101 / CIFAR-100-C"]
tags: [geometry, ood, recoverability, negative-result]
---

# RGC Shift Recoverability — ResNet-101 / CIFAR-100-C

## Question

Under corruption, does the geometric predictor contain accuracy-relevant
information the head lacks, and can a clean/validation-fitted label-free rule
recover a useful fraction of that information?

## Primary result

- Per-class oracle-union headroom: **4.385 pp**
- Global-kNN oracle-union headroom: **3.798 pp**
- Per-class gate captured: **-0.1%**
- Global-kNN gate captured: **-2.3%**
- No gate produced a reliable gain over the head or the published beta=30 vector arm.
- All five blend sweeps selected alpha=0 (head-only).
- Harness outcomes: per-class = reformulate for all five seeds; global-kNN =
  stop for seed 1, reformulate for seeds 2–5.

## Not measured in this run (limits every claim above)

- **No null baseline for the oracle union.** The union of the head with *any*
  second imperfect, disagreeing predictor is positive by construction. Without a
  null (a second independently trained head, or a noise-matched predictor), the
  4.385 pp cannot be read as "geometry carries complementary information" rather
  than "two imperfect predictors disagree". This blocks interpretation of both
  headroom numbers and of the per-class vs global gap.
- **No standalone geometry quality numbers.** Geometry accuracy and head-geometry
  disagreement rate were not recorded, so the per-class advantage could simply
  reflect per-class geometry disagreeing more often.
- **No per-corruption / per-severity breakdown.**
- **No matched-risk or matched-coverage operating point** (README rule 6). All
  numbers are raw accuracy / ECE / NLL.
- **The fixed beta=30 arm's absolute gain over the head is not reported.** The
  often-quoted "roughly +0.02 pp" is obtained by subtracting two paired CIs
  (gate-head and gate-beta30) and is **[inference]**, not a measured value.

## Paired seed-level effects

- Per-class gate minus head: +0.002 pp, 95% CI [-0.010, +0.014]
- Per-class gate minus beta=30: -0.021 pp, 95% CI [-0.085, +0.044]
- Global-kNN gate minus head: -0.021 pp, 95% CI [-0.078, +0.036]
- Global-kNN gate minus beta=30: -0.044 pp, 95% CI [-0.116, +0.029]

## Positive signal inside the negative result

Per-class geometry exposes **0.587 pp more oracle headroom** than global-kNN,
with a paired 95% CI of approximately [0.057, 1.117] pp.

This means the per-class representation appears to contain more complementary
accuracy-relevant information, even though the pre-registered label-free gate
does not know when to use it.

## Strong interpretation

This experiment rejects the simple operational hypothesis:

> radius + neighbour agreement + neighbour margin + neighbourhood concentration,
> fitted on clean/validation data, are sufficient to recover geometric
> complementarity under CIFAR-100-C shift.

It does **not** reject the existence of useful geometric information.

## Important protocol deviations / integrity checks

### k_vote discrepancy - UNRESOLVED, and the negative result may depend on it
The original planned protocol pinned `k_vote=50` and `k_radius=200`. The recorded
experiment card reports `k_vote=200` and `k_radius=200`.

This is not only a bookkeeping problem. With `k_vote=200`, two of the four gate
features (head-neighbour agreement, neighbourhood concentration) are averaged over
200 neighbours, which may wash out exactly the local signal the gate needed.
Seed 1's median top-1 neighbour count of 200/200 [[Seed 1 occupies a qualitatively different kNN regime]]
is consistent with that reading. **[inference]** - mechanism, not measurement.

Consequence: **the gate kill is protocol-contingent.** Every downstream note that
cites this kill carries the flag `protocol-contingent-k_vote` until the run is
repeated at `k_vote=50`.

Resolution step: check the run config in
`research_log/experiments/2026-09-15_rgc-shift-recoverability/` for a documented
approved change. If none exists, re-run the gate arm at `k_vote=50` before the
kill is treated as final.

### fitting split discrepancy
The original prompt stated that everything fitted should be fitted on the
clean cell. The recorded run used a validation cell for primary fitting and
used clean for mechanism diagnostics.

Methodologically that may be preferable, but it is a deviation from the stated
prompt and must remain explicit.

## Next action

Do NOT immediately add a more powerful supervised gate.

First ask whether the failure is due to:
1. covariate shift in the reliability features,
2. concept shift in the mapping from reliability features to "geometry is right",
3. missing observables,
4. insufficient decisive training examples,
5. seed / representation regime instability.

See linked notes.
