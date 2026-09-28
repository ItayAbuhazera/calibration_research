---
type: experiment
status: completed_development_inconclusive_precision
date: 2026-09-28
project: Full-Vector Geometric Calibration
benchmark: CIFAR-100-C (12 development cells), ResNet-101 checkpoints 2 and 4 (independent training seeds)
preregistered: true
experiment_type: descriptive / operational recoverability (existence of an action-selection problem; no internal evidence)
parent_question: Is there per-example route-vs-keep ambiguity that strong output-only evidence cannot resolve?
exposure_ledger: "[[Representation Correction Exposure Ledger]]"
tags: [N1a, action-advantage, routing, ambiguity, identifiability]
---

# 2026-09-28 — N1a: action ambiguity audit

**Frozen specification:** `GeometricFullCalibration/docs/n1a_action_ambiguity_spec.md`, sha256 `3d50a806…be618`, committed `3b2ab7b` before any N1a code or output. Code `3d5ac7b`, `5cefe65`; snapshot `snapshots/n1a_0c501a1eb60a`. Report: `GeometricFullCalibration/docs/n1a_action_ambiguity_audit_2026-09-28.md`.

## Question
Claim scope: checkpoints 2 and 4 routed symmetrically; 12 exposed cells; target = Δ_route = 1[other correct] − 1[base correct]; pre-action evidence = full base logits and deterministic functions of them only (206 features); reference channel (post-action, not deployable) adds the other model's output features.

## Competing explanations and predictions
- No action problem (repairs and harms rare or one-sided) → heterogeneity headroom h < 1.0 pp.
- Full Z resolves the action → ambiguity mass M_A (regret any Z-bin-constant policy must pay) < 1.0 pp, or no observable channel reduces it (Q < 0.5 pp).
- Residual, resolvable action ambiguity → M_A ≥ 1.0 (lower ≥ 0.5), M_B ≥ 1.0, Q ≥ 0.5 with lower > 0, in both bases.

## Experiment and primary contrast
Leave-one-family-out × Stage-0 image folds (family and image-identity holdout); 3-way multinomial selector; policy route iff P(+1) − P(−1) > 0; realized policy utility, not AUROC. Outcome matrix and thresholds: spec §5–6.

## Results (2026-09-28; `GeometricFullCalibration/results/n1a/report/n1a_aggregate.json`; full report `GeometricFullCalibration/docs/n1a_action_ambiguity_audit_2026-09-28.md`)

**Frozen decision: INCONCLUSIVE — insufficient precision.** Base 2 meets every GO criterion; base 4 meets all except the resolvability gap Q ≥ 0.5 pp (Q = +0.41 [+0.28, +0.55]).

| family-macro, pp | base 2 → 4 | base 4 → 2 |
|---|---|---|
| repair / harm rate (%) | 9.20 / 8.28 | 8.28 / 9.20 |
| headroom h | 8.28 [8.05, 8.49] | 8.28 [8.06, 8.48] |
| G_Z strong output-only policy | 2.03 [1.78, 2.29] | 1.25 [1.00, 1.47] |
| M_A output-policy ambiguity | 7.12 [6.90, 7.28] | 6.91 [6.73, 7.08] |
| M_B local-Z ambiguity | 3.51 [3.46, 3.55] | 3.44 [3.40, 3.49] |
| Q = G_ZZo (post-action reference) − G_Z | +0.54 [+0.41, +0.68] | +0.41 [+0.28, +0.55] |

Validity: 80/80 fits converged; family + image holdout and Z_o-exclusion assertions held; support 2,113–3,347 repairs/harms per (base, family). Limitation: 77/80 fits selected λ = 1e-5 (weak edge; the spec had no edge rule).

## Post-mortem and allocation
- An action-selection problem exists (balanced repair/harm ≈ 8–9 % each; oracle 8.3–9.2 pp vs best constant 0–0.9 pp) and strong full-logit evidence captures only ≈ 10–19 % of it.
- Most of the residual is not resolved by any output channel tested, including the other model's own output (+0.4–0.5 pp); whether a *resolvable* residual of practical size exists was not certified in both bases.
- Not tested: internal representations. Not established: irreducibility, non-identifiability, reserved-family or other-model behaviour.
- Allocation: not GO → **N1b not designed or run.** Continuing on this substrate needs a stated reason why a base-model internal channel should resolve more than the other model's own output does.
- Review status: single-author, agent-assisted; decision computed by the pre-frozen, unit-tested rule.
