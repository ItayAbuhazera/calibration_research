---
type: hypothesis
status: proposed
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100 / CIFAR-100-C (12 exposed development cells), ResNet-101, checkpoints 2 and 4
novelty: unknown (territory audit 2026-09-29: WATCH — settings intersection, not a new problem)
tags: [H-SELREP, selective-repair, repair-vs-harm, internal-evidence, T3]
---

# H-SELREP-01 — Multi-depth internal evidence discriminates repair from harm beyond output evidence under held-out corruption

## Formal statement
Claim scope: end-to-end ResNet-101 (checkpoints 2, 4); held-out CIFAR-100-C corruption family (leave-one-family-out × image folds);
inputs = native logits + cached clean-trained probe logits at 12 depths; target Δ = 1[c(x)=y] − 1[head(x)=y] for ONE pre-specified
label-free internal candidate c; decision KEEP vs APPLY. A selector using internal evidence achieves higher realized (W − H)/N than the
matched output-only selector (N1a feature set F_Z).

## Why it follows from evidence
Internal candidates carry correct alternatives on some errors (Atlas ≈13 %) and target-fitted combinations gain (Stage 0); earlier gates
were clean-fitted, and leave-family-out supervision is untested for this candidate type.

## Competing explanation
Output-only evidence already captures whatever is selectable (N1a: output selectors capture 10–19 % of routing headroom); internal
candidates' harm dominates (H:W ≈ 3.8:1, fixed-gate study; SelfChecker's CIFAR-100 decrease). Prediction overlap: both predict small
gains; they separate only if internal − output ≥ practical threshold.

## Benchmark
Existing layer-pilot cache (`results/layer_pilot/checkpoint_seed{2,4}/<cond>/per_sample.npz`), Stage-0 folds. Read-only, CPU.

## Baselines
Never/always switch; output-only selector (F_Z); runner-up-of-head candidate; focal-paper-style margin gate.

## Decisive experiment
Primary type: operational recoverability / action quality. Primary contrast: realized ΔAcc(internal selector) − ΔAcc(output-only),
family-macro, both checkpoints. Controls: dimension-matched shuffled probe block; runner-up candidate; oracle-gated headroom reported only.
Uncertainty tested: action quality, transfer. Exposure: exposed development cells only (see [[Representation Correction Exposure Ledger]]);
target labels of held-out family used for evaluation only.

## Outcome-to-decision matrix

| Possible outcome | Explanation supported/weakened | Still unresolved | Next decision |
|---|---|---|---|
| internal − output ≥ +0.25 pp, lower bound > 0, ≥ 3/4 families, both checkpoints; shuffle ≤ +0.10 | internal evidence carries selectable repair-vs-harm info | reserved families, causal mechanism | preregistered confirmation + T4 |
| upper bound < +0.25 pp in both checkpoints | output-sufficiency / harm dominance | whether other candidates differ (not to be searched) | stop this line on this substrate |
| mixed across checkpoints/families | — | precision | inconclusive; no variant search |

## Kill criterion
Upper interval bound of (internal − output-only) < +0.25 pp in both checkpoints, or internal selector ≤ never-switch on held-out families.

## Go criterion
As row 1 above. Practical threshold +0.25 pp family-macro (N1a/N1a-DP scale); interval is image-group bootstrap conditional on fitted
selectors — no universal CI rule.

## Prior-art threats
[[To Adapt or Not to Adapt - Selective Adaptation for VLMs]]; [[Self-Checking Deep Neural Networks in Deployment]]; ALTAS; CALRD;
[[Representation Trajectories Matters]] margin gate. Not authorized; decide jointly with the PAUSED N1a-DP
([[Internal-Evidence Recoverability and Selective Correction]]).

## Correction 2026-09-29 — red-team: POC-T3 DO NOT RUN; hypothesis dormant (status unchanged: proposed)

Source: `GeometricFullCalibration/docs/internal_computation_recoverability_redteam_2026-09-29.md` §11. Not tested; no data opened.
* Central contrast already run under clean-fitting: fixed-gate Z1 − Z0 null in both checkpoints ([[2026-09-21 Fixed Deep Candidate Gate Study]]).
* Confounds in the spec as written: (a) the internal selector sees the candidate's confidence, the F_Z selector does not; (b) the 12 × 100
  probe block is much wider than F_Z (G1-DP width/shuffle issues); (c) no layer4.2-only arm, so depth is not separated from a family-fitted
  representation readout; (d) the candidate depth band is informed by exposed results (Stage-0 ablation), and clean-only selection is
  impossible because all clean probes are below the head.
* +0.25 pp is a detectability threshold, not a scientific effect size. Primary quantities, if ever run: fraction of oracle-gated headroom
  captured; repair-vs-harm AUROC among disagreements.
* Reopen only if: (i) a primary source shows an internal gate beating output + candidate-confidence for label-changing corrections under
  held-out natural shift in vision; (ii) N1a-DP is authorized (then only as a preregistered secondary arm with controls (a)–(d)); (iii) a
  substrate where some clean depth probe beats the head.
