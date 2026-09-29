# N1a-DP — Output Decoder Panel action-ambiguity audit (frozen specification v1)

Status: **DRAFT r2 (2026-09-29, revised after G1-DP; not frozen)** until committed with `n1a_dp_spec.frozen.sha256`, which happens only after the G1-DP report is complete and before any N1a-DP
fit exists. Instrument: Decoder Panel v1 (`docs/decoder_panel_v1_spec.md`, frozen). Code: `atlas/n1adp.py`, `atlas/n1adp_aggregate.py`,
`atlas/n1adp_rules.py`. Journal: `docs/decoder_panel_v1_master_report_2026-09-28.md`.

A NEW experiment. **Historical N1a (`docs/n1a_action_ambiguity_spec.md`) remains INCONCLUSIVE — insufficient precision**; nothing here
re-scores it. Development diagnostic on exposed cells; not confirmation. **No internal representation is used (this is not N1b).**

## 0. Question

Does the large residual action ambiguity observed in N1a persist when the full output evidence Z is given to a diverse, properly optimized
decoder panel under held-out-family transfer? Motivation: N1a's output-only selector (penalized multinomial logistic) captured 10–19 %
of the per-example headroom (G_Z = 2.03 / 1.25 pp vs oracle 9.20 / 8.28 pp), left M_A = 7.12 / 6.91 pp of within-score-bin regret and
M_B = 3.51 / 3.44 pp of local-Z ambiguity — but **77 / 80 of its fits selected the weakest λ on its grid**, so the selector may have been
decoder-limited. N1a-DP asks whether the OUTPUT DECODER was the problem.

## 1. Action problem (unchanged from N1a)

Base b ∈ {2, 4}, other o = 6 − b (independently trained ResNet-101 CIFAR-100 checkpoints), both directions. For row r = (image, cell):
U_keep = 1[argmax Z_b = y], U_route = 1[argmax Z_o = y], Δ_route = U_route − U_keep ∈ {−1, 0, +1} (+1 repair, −1 harm). Primary action menu:
keep vs route. Ensemble and abstention are not analysed here. Target for every decoder: 3-class y = Δ_route + 1; decision score
e(r) = P(+1 | r) − P(−1 | r); **policy: route iff e(r) > 0.**

## 2. Evidence boundary

Pre-action evidence = F_Z(b), the frozen N1a feature map of the BASE logits only (`atlas.n1a.output_features`: standardized-by-decoder full
logits z (100), softmax entropy, p1 − p2, p1 − p3, p1 − p5, max softmax, max logit, one-hot argmax (100); 206 columns). **Z_o is never
admissible** (computing the other model is the action's cost); asserted in code and tests. The historical ZZo result (Q = +0.54 / +0.41 pp)
is cited as a reference comparator only; it is not a ceiling and is not re-run.

## 3. Data and holdout (hard invariants)

Canonical strict-FP32 atlas logits `results/atlas/seed{2,4}/u0/<cell>.npz`, labels via `atlas.stage0_data.load_cell`, the 12 exposed cells
(4 families × severities 1/3/5). Not accessed: reserved families, checkpoints 1/3/5, new images, internal representations.
Outer unit = (base, held-out family f, Stage-0 outer image fold k): **training rows** = 3 other families × 3 severities × fold-k outer-train
images; **evaluation rows** = family f × 3 severities × fold-k outer-test images. Both family holdout AND image-identity disjointness are
enforced (asserted); every (base, f) evaluation covers the 10,000 images × 3 severities exactly once across folds. 2 × 4 × 5 = 40 units.

## 4. Decoders and HPO (SHIFT-TRANSFER regime)

All six Decoder Panel v1 families (unanchored 3-class classification), frozen grids/spaces/budgets. HPO regime `shift_transfer`:
inner model selection = leave-one-TRAINING-family-out: for each of the three training families g, fit rows = the other two training
families × inner-fit images (Stage-0 `inner_fit_mask`), validation rows = family g × inner-val images (image-disjoint); **selection objective
= mean over the three inner environments of the negative realized policy utility −mean(Δ · 1[e > 0])** (not AUROC, not NLL). LightGBM / MLP
early stopping uses each inner split's validation rows (native 3-class log-loss); refit on all training rows (round(mean best iteration)).
One model per outer unit; no per-severity tuning. The held-out family never enters HPO, early stopping, preprocessing or refit.
Study seeds from ids {base, heldout, fold, arm = Z}.

## 4b. Continuity control (original N1a linear selector) — not a panel family

C0 = the historical N1a output-only selector, reused exactly: its cross-fitted predictions `results/n1a/fits/base{2,4}/<family>/Z/fold{0-4}.npz`
(snapshot `snapshots/n1a_0c501a1eb60a`; penalized 3-way multinomial logistic, λ ∈ {1e-1..1e-5} by inner-val NLL, image-only inner split) —
same outer family × image-fold holdout and same F_Z(b). It is re-scored with every §5 quantity and reported beside the six families, but it
**never counts toward any §7 outcome**. It is already exposed (N1a report). Validity check V-C0: the re-scored family-macro G reproduces the
historical G_Z (2.03 / 1.25 pp) within ±0.01 pp (checks the new aggregation code, not the science).

## 5. Quantities (per family m, per base; per held-out family = pooled over folds and severities; family-macro = mean over 4 held-out families)

- Oracle gain G_or = P(Δ = +1); best fixed action G_c = max(0, mean Δ) (evaluation rows); train-chosen constant (reference);
  heterogeneity headroom h = G_or − G_c.
- Realized utility G_m = mean(Δ · 1[e_m > 0]); gain over best fixed action G_m − G_c; **regret to oracle U_m = G_or − G_m**;
  **fraction of opportunity recovered φ_m = (G_m − G_c) / h**; repair capture = routed repairs / repairs; harmful-routing rate = routed harms /
  rows (and harm capture = routed harms / harms); route rate; calibration of e (10 quantile bins: mean e vs mean Δ, weighted absolute error);
  all by held-out family and by severity, separately for base 2 and base 4.
- **A. Output-score ambiguity M_A,m** = Σ_bins w_b · min(p+_b, p−_b) over 10 quantile bins of the cross-fitted e_m (as N1a).
- **B. Local-Z ambiguity M_B** (decoder-independent; N1a definition, k = 20 nearest evaluation rows in train-standardized full-logit space
  within (held-out family, fold), other images only) — recomputed once per base.
- Intervals: 95 % image-group bootstrap, B = 2000, **one resample array per base shared by all families** (seed 20261010 + 70000 + base).
  φ_m interval = ratio of resampled family-macro means.

Empirical ambiguity audit only; no claim of mathematical non-identifiability.

## 6. Thresholds, informativeness and stochasticity (frozen before any N1a-DP outcome)

### 6.1 Outcome thresholds (unchanged from draft r1; justified from exposed N1a results)
h ≈ 8.3 pp (N1a, both bases); material utility scale 0.5 pp; N1a interval half-widths were 0.2–0.3 pp.
- **RESOLVED (m, b):** φ lower 95 % ≥ 0.50 AND M_A,m upper 95 % ≤ 3.5 pp.
- **SUBSTANTIAL (m, b):** U_m lower 95 % ≥ 4.0 pp AND M_A,m lower 95 % ≥ 2.0 pp.
- Otherwise INTERMEDIATE. Local-Z precondition: M_B ≥ 1.0 pp in both bases.

### 6.2 Informativeness (new; motivated by G1-DP, where kNN selected the anchor-only candidate in 97/110 studies and therefore measured
nothing while still counting as a valid family)
A family that effectively implements a constant policy cannot be evidence that the output channel lacks usable action information.
Per family m and base b, over its 20 outer units (4 held-out families × 5 folds), using only training-side quantities and label-free
evaluation decisions (no evaluation labels):
- **I1 non-degenerate decisions:** the refitted selector's evaluation route rate lies in [1 %, 99 %] in ≥ 16 of 20 units.
- **I2 inner sensitivity:** the selected configuration's inner-validation utility exceeds that of the inner best-constant policy by
  ≥ τ in ≥ 16 of 20 units. Inner best-constant policy: on each inner split, route-all if mean Δ on that split's fit rows > 0, else keep-all,
  scored on that split's validation rows; macro over the three inner splits (same rows as model selection; training labels only).
  τ = max(0.10 pp, 2 · s_inner(m)), with s_inner from the stochasticity audit (§6.3); for deterministic families s_inner = 0.
- m is **INFORMATIVE for b** iff I1 and I2; otherwise **NON-INFORMATIVE**. NON-INFORMATIVE families are reported in full but cannot count
  toward PANEL-ROBUST OUTPUT AMBIGUITY (§7 rule 2). Informativeness does not use evaluation outcomes, so it cannot be tuned to the verdict.

### 6.3 Stochasticity (new)
LightGBM and MLP are stochastic given data (trial seeds, bagging/feature sampling, initialization, mini-batch order); the other four
families are deterministic given data and their fixed panel seeds. Before freezing, an engineering-only audit
(`docs/n1a_dp_stochasticity_audit_plan.md`; synthetic targets; no N1a-DP outcome) estimates per-family seed standard deviations
s_G, s_φ, s_U, s_MA, s_inner (family-macro scale). They are recorded in this spec at freeze (§6.4) and used as follows:
- **Seed-robust labels:** for m ∈ {lgbm, mlp}, RESOLVED requires φ_lo − 2 s_φ ≥ 0.50 and MA_hi + 2 s_MA ≤ 3.5; SUBSTANTIAL requires
  U_lo − 2 s_U ≥ 4.0 and MA_lo − 2 s_MA ≥ 2.0. A label that holds only without the margin is **SEED-FRAGILE** and is treated as
  INTERMEDIATE. (Deterministic families: s = 0.)
- The production run uses the single frozen panel seed per study; audit replicates are never used for selection (no best-of-seed).

### 6.4 Audited noise scale (filled from the audit before freezing)
{{AUDIT_VALUES}}

## 7. Decision (first matching row)

0. **INCONCLUSIVE (validity)** if fewer than 5 families are valid, V-C0 fails, or any holdout / image / Z_o assertion fails. A family is
   valid iff, in both bases: all 40 units complete (validated markers), every D1/D2/D6 refit converged, the strongest λ (1e1) was selected
   in < 2 of 20 units (D1/D2/D6), support ≥ 300 repairs and ≥ 300 harms per (base, held-out family). (Weak-edge λ = 1e-7 is reported, not
   invalidating: with 72,000 standardized rows it is numerically the unpenalized fit.)
1. **OUTPUT-DECODER LIMITATION**: ≥ 2 valid families RESOLVED (seed-robust) in both bases. → the N1a residual was substantially decoder-
   caused; do NOT proceed to N1b on this substrate.
2. **PANEL-ROBUST OUTPUT AMBIGUITY**: ≥ 4 valid families are INFORMATIVE in both bases; every valid INFORMATIVE family is SUBSTANTIAL
   (seed-robust) in both bases; no valid family is RESOLVED in either base; M_B ≥ 1.0 in both bases. → residual action ambiguity is not
   explained by the tested decoder families; N1b is scientifically justified **for design only**.
3. **FAMILY-SPECIFIC**: some valid family RESOLVED in at least one base but rule 1 does not hold. → decoder-sensitive measurement; no broad
   output-insufficiency claim; no N1b.
4. **INCONCLUSIVE**: otherwise (including < 4 informative families, seed-fragile labels, intermediate recoveries, base disagreement). → no N1b.

Always reported: the full six-family vector per base plus C0, each family's informativeness (I1, I2 counts) and seed-robustness; never the
best family alone.

| outcome | supported / weakened | still unresolved | next decision |
|---|---|---|---|
| OUTPUT-DECODER LIMITATION | N1a residual largely a decoder limitation | deployability, reserved families | stop N1 on this substrate (no N1b) |
| PANEL-ROBUST OUTPUT AMBIGUITY | informative output-only decoders leave robust action ambiguity | whether ANY pre-action channel resolves it | design N1b (draft only; not run) |
| FAMILY-SPECIFIC | decoder-sensitive measurement | which inductive bias matters | record; no N1b |
| INCONCLUSIVE | none | precision / validity / informativeness | record; no N1b |

## 7b. Relation to Decoder Panel v1 and G1-DP

These informativeness, continuity and stochasticity rules are prospective N1a-DP validity refinements motivated by a limitation observed in
G1-DP. Decoder Panel v1 (spec, families, grids, budgets) is unchanged, and G1-DP (frozen outcome INCONCLUSIVE (validity)) is not reopened.

## 8. Execution

One Slurm array per family over the 40-unit frozen mapping (`atlas.n1adp fit_index`), CPU, from an immutable snapshot
(`python -m atlas.snapshot n1adp`); aggregation afterok (`atlas.n1adp_aggregate`, refuses incomplete arrays). Resources: resource plan.
Engineering recovery allowed; scientific changes after outcomes forbidden.
