# N1a action ambiguity audit — 2026-09-28

Development diagnostic on exposed CIFAR-100-C cells. Not confirmation. No internal representation is used anywhere in N1a.
Artifacts: `results/n1a/report/n1a_aggregate.json`; fits `results/n1a/fits/base{2,4}/<family>/{Z,ZZo}/fold{0-4}.npz`; job ledger
`results/n1a/ledger.json`; logs `results/n1a/logs/`. Spec `docs/n1a_action_ambiguity_spec.md` (sha256 `3d50a806812aff9cd47cc20688a38f4166634b16e9469a6b9a09cb45171be618`,
frozen in `3b2ab7b` before any N1a code or output); code `3d5ac7b`, `5cefe65`; snapshot `snapshots/n1a_0c501a1eb60a` (an earlier
snapshot `n1a_6dcf7e63f585` was superseded by an engineering fix to log/ledger paths before any job ran and was never used).

# 1. G1 closure

G1 (frozen Outcome C, both checkpoints): Stage 0 revealed accessible compact internal evidence; G1 did not support a depth-specific
interpretation and exposed substantial capacity / parameterization sensitivity in full-rank H_L readouts; it does not establish absence
of the information from H_L. The Stage-0 branch is closed. Vault (pushed to `origin/main` in `dfd137b`):
`ResearchBrain/05_Experiments/2026-09-28 G1 Conditional Accessibility Gatekeeper.md`,
`ResearchBrain/10_Projects/Current Evidence - Representation-Based Correction Program.md` (§ "G1 update — 2026-09-28"),
`ResearchBrain/02_Observations/A full-rank penultimate target readout does not reproduce the compact layer3 probe increment in ResNet-101.md`,
`ResearchBrain/03_Failure_Modes/Equivalent feature spans can differ materially under finite-sample regularized readouts.md`,
`ResearchBrain/01_Papers/Improving LLM Final Representations with Inter-Layer Geometry.md`,
`ResearchBrain/01_Papers/When Is Test-Time Adaptation Identifiable From Unlabeled Evidence.md`, `ResearchBrain/08_Weekly_Synthesis/2026-W40.md`.

# 2. N1a frozen design

- **Estimand:** per-example route advantage Δ_route = 1[argmax Z_o = y] − 1[argmax Z_b = y] ∈ {−1, 0, +1}, for base b ∈ {2, 4} and
  other o = the other checkpoint (independent training seeds), run symmetrically. Ensemble: descriptive only. Abstain: not analysed.
- **Evidence boundary:** pre-action output evidence F_Z(b) = standardized full base logits (100) + entropy, p1−p2, p1−p3, p1−p5, max
  softmax, max logit + one-hot argmax (206 features); Z_o never enters F_Z (unit-tested). Post-action *reference* channel ZZo =
  F_Z(b) + F_Z(o), not deployable evidence; it measures whether any observable output resolves the residual.
- **Holdout:** leave-one-corruption-family-out (all 3 severities) × Stage-0 image folds: train on 3 families × training-fold images,
  evaluate on the held-out family × held-out images (image-identity disjointness asserted; each image evaluated once per family).
- **Selector:** 3-way penalized multinomial logistic regression (Stage-0 fitter unchanged; λ ∈ {1e-1..1e-5} by inner-val NLL); policy
  route iff P(+1) − P(−1) > 0; realized policy utility is the endpoint.
- **Quantities:** headroom h = min(P(+1), P(−1)); output-policy ambiguity M_A = Σ_decile-bins w·min(p+, p−) (regret any policy constant
  within e_Z deciles must pay); local-Z ambiguity M_B (k = 20 nearest held-out rows in standardized full-logit space, other images
  only); resolvable gap Q = G_ZZo − G_Z. Intervals: 95 % image-group bootstrap, B = 2000.
- **Frozen rules:** STOP-heterogeneity if h < 1.0 pp in either base; STOP-Z if in both bases M_A upper < 1.0 or Q upper < 0.5; GO if in
  both bases M_A ≥ 1.0 (lower ≥ 0.5, ≥ 3/4 families), M_B ≥ 1.0, Q ≥ 0.5 with lower > 0; otherwise INCONCLUSIVE. Support ≥ 300
  repairs and ≥ 300 harms per (base, family).

# 3. Oracle payoff structure (Phase 1; % of rows, 30,000 rows per family per base)

| base → other | family | both correct | both wrong | **repair (+1)** | **harm (−1)** |
|---|---|---|---|---|---|
| 2 → 4 | gaussian_noise | 20.20 | 64.49 | 8.26 | 7.04 |
| 2 → 4 | defocus_blur | 49.51 | 33.59 | 8.56 | 8.34 |
| 2 → 4 | fog | 52.33 | 30.72 | 8.83 | 8.12 |
| 2 → 4 | jpeg_compression | 45.20 | 34.02 | 11.16 | 9.63 |
| 2 → 4 | all 12 cells | 41.81 | 40.71 | **9.20** | **8.28** |
| 4 → 2 | all 12 cells | 41.81 | 40.71 | **8.28** | **9.20** |

(4 → 2 is the mirror image: repairs and harms swap.) Support: 2,113–3,347 repairs and harms per (base, family), far above 300.
Heterogeneity headroom h = 8.28 pp in both bases [8.05, 8.49]; oracle route gain 9.20 / 8.28 pp vs best constant policy 0.92 / 0.00 pp.
Repair and harm rates are stable across image folds (base 2: repair 8.85–9.42 %, harm 7.97–8.53 %).
**Ensemble (secondary, descriptive):** repair 6.04 / 5.58 %, harm 2.80 / 3.26 % (bases 2 / 4) — ensembling is net positive on average
(+3.2 / +2.3 pp) but still harms ≈ 3 % of rows.
Model disagreement alone would not have shown this: the payoff is nearly balanced between repair and harm, which is exactly the regime
where a per-example selector (not a constant policy) is needed.

# 4. Output-only ambiguity (Phase 2; family-macro, pp, 95 % image-group bootstrap)

| quantity | base 2 (→4) | base 4 (→2) |
|---|---|---|
| oracle gain | 9.20 | 8.28 |
| best constant (eval) / train-chosen constant | 0.92 / 0.92 | 0.00 / 0.00 |
| **G_Z** (strong output-only policy) | **2.03 [1.78, 2.29]** | **1.25 [1.00, 1.47]** |
| headroom captured by Z | 10–17 % by family | 10–19 % by family |
| **M_A** output-policy ambiguity | **7.12 [6.90, 7.28]** | **6.91 [6.73, 7.08]** |
| **M_B** local-Z ambiguity (k = 20) | **3.51 [3.46, 3.55]** | **3.44 [3.40, 3.49]** |
| G_ZZo (post-action reference) | 2.57 | 1.66 |
| **Q = G_ZZo − G_Z** | **+0.54 [+0.41, +0.68]** | **+0.41 [+0.28, +0.55]** |

Budget-matched curves (descriptive; top-k % routed by predicted advantage), e.g. base 2 fog: Z top-10 % +1.10, ZZo +1.95 pp;
base 4 fog: Z +1.01, ZZo +1.54 pp (all families in the artifact, `families.<f>.budget_curves`).

# 5. Cross-family / cross-seed stability

| family | base 2: M_A / M_B / Q | base 4: M_A / M_B / Q |
|---|---|---|
| gaussian_noise | 6.19 / 3.15 / +0.10 | 6.18 / 3.32 / +0.25 |
| defocus_blur | 6.98 / 3.26 / +0.75 | 6.76 / 3.11 / +0.42 |
| fog | 6.73 / 3.18 / +0.81 | 6.59 / 3.04 / +0.64 |
| jpeg_compression | 8.57 / 4.44 / +0.51 | 8.12 / 4.30 / +0.32 |

Headroom and ambiguity (M_A, M_B) are large and stable across all four held-out families and both bases. The **resolvable** part Q is
small and heterogeneous: near zero for gaussian noise, 0.3–0.8 pp elsewhere, and below +0.5 pp family-macro for base 4.

# 6. N1a decision

**INCONCLUSIVE — insufficient precision** (frozen rule, `decision` in `n1a_aggregate.json`): base 2 satisfies every GO criterion; base 4
satisfies all except Q ≥ 0.5 (Q = +0.41 [+0.28, +0.55]); STOP-Z does not fire (Q upper bounds exceed 0.5). Validity: all 80 fits
converged; image/family holdout and the Z_o-exclusion assertions held; support far above the floor. Tests: 7/7 N1a tests pass.

**Limitation recorded (not a validity criterion in the frozen spec):** 77 of 80 selector fits selected λ = 1e-5 (3 selected 1e-4), the weak-regularization
edge of the grid; the spec had no grid-edge rule for N1a. Selectors may be marginally under-fitted; this affects Z and ZZo alike and is
not corrected post hoc.

# 7. Scientific interpretation

Established (development, exposed cells, two checkpoints):
- A per-example **action-selection problem exists**: routing repairs ≈ 8–11 % and harms ≈ 7–11 % of rows, nearly balanced, so a constant
  policy gains ≈ 0–0.9 pp while the oracle gains 8.3–9.2 pp.
- **Strong output-only evidence resolves little of it:** the full-logit selector captures ≈ 10–19 % of the headroom (G_Z 1.2–2.0 pp);
  ≈ 7 pp of regret remains that no policy constant within the output-predicted bins can avoid (M_A), and ≈ 3.4 pp remains within local
  full-logit neighbourhoods (M_B).
- **Most of that residual is not resolved by any output channel tested**, including the action's own output: adding the other model's
  output features raises realized utility by only +0.41 / +0.54 pp. The resolvable fraction is the unresolved question; the frozen rule
  could not certify it at the 0.5 pp scale in both bases.

Not established: that internal representations resolve any of the ambiguity (not tested); that the residual is irreducible (N1a only
bounds output channels); mathematical non-identifiability (this is a finite-sample empirical audit); anything about reserved families,
other models or deployment.

Allocation per the frozen rule: **not GO — N1b is not designed or run.** Whether to spend further effort on this substrate is a
researcher decision; the numbers suggest that the ceiling for any richer pre-action channel is likely small relative to the headroom
(the post-action reference itself recovers only ≈ 0.4–0.5 pp), which argues against N1b on this substrate unless a reason is stated why an
internal channel of the *base* model should resolve more than the other model's own output does.

# 8. N1b preregistration draft

Not produced: N1a did not return GO.
