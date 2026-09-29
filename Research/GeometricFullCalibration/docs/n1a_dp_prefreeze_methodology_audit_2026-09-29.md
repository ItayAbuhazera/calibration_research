# N1a-DP pre-freeze methodological audit — five-family prospective panel (2026-09-29)

**Status of N1a-DP: DRAFT / PAUSED / NOT AUTHORIZED.** No N1a-DP scientific outcome has been computed or inspected. This report contains
no scientific N1a-DP verdict. It summarizes engineering / methodological audit outputs only.

Sources: frozen audit protocol `docs/n1a_dp_stochasticity_audit_plan.md` (sha256 `0b35471336766c00038a3110673c7a271b68a1c5d97f437d968b5d617216c36b`,
commit `07db6ed`); runner `atlas/n1adp_audit.py`; snapshot `snapshots/n1adp_audit_242c55939ba4`; completed outputs
`results/n1adp_audit/{A,B,null}/<family>/u<k>.json`; five-family summary `results/n1adp_audit/summary_5family.json` produced by
`python -m atlas.n1adp_audit_report5` (same formulas as the frozen `summary`, applied to the five remaining families; the frozen runner is
unmodified). Draft spec: `docs/n1a_dp_spec.md` (DRAFT r4). Journal: `docs/decoder_panel_v1_master_report_2026-09-28.md`.

## 1. Executive summary

- The frozen six-family pre-freeze audit **did not complete for poly2**: its null-control runs were cancelled at 2 h 41 min because their
  runtime was judged disproportionate to their scientific value for this prospective experiment. poly2 is removed from the **prospective**
  N1a-DP panel only (not from Decoder Panel v1, not from G1-DP). This is a resource/design decision before freezing, not a scientific result.
- Five prospective families: linear, rff, LightGBM, MLP, kNN; structural groups {linear, rff}, {LightGBM}, {MLP}, {kNN}.
- **Shuffled-target null: the STOP rule (≥ 2 of 4 null units passing I1 and I2) does not fire for any of the five families.** LightGBM
  passes jointly on 1 of 4 null units (a near-constant 97.6 % route-all policy with a 0.167 pp inner margin); all others pass on 0 of 4.
- Noise constants (max per-unit full-HPO SD): LightGBM s(I2) 0.0672 pp → τ = 0.1344 pp; MLP s(I2) 0.0614 pp → τ = 0.1228 pp; deterministic
  families τ = 0.10 pp. φ/U/M_A margins in §8.
- Historical continuity control (original N1a linear selector) reproduces its reported family-macro gains **exactly** (|Δ| = 0.0 pp).
- Open validity concerns for researcher review (§11): the I1 bound [1 %, 99 %] admits near-constant policies under the null; MLP
  final-fit-only SDs exceed full-HPO SDs; rff/linear tend toward regularization / smoothness grid edges; kNN shows random-like routing
  with negative null margins (likely non-informative).

## 2. Original six-family audit status (record, unchanged)

The frozen protocol specified six families. Completed: A and B for LightGBM and MLP (4 units each), null for linear, rff, LightGBM, MLP,
kNN (4 units each). **Not completed: null for poly2** (21732556, cancelled at 2:41:37 of 6 h, no output written; hedge 21733698 cancelled
earlier; dependent six-family summary 21732561 cancelled, never ran). The completed outputs are immutable records. The six-family
`results/n1adp_audit/summary.json` was never produced; nothing here is presented as the six-family audit.

## 3. Reason for prospective poly2 removal

"The original six-family N1a-DP pre-freeze audit did not complete for poly2 because its null-control runtime was judged disproportionate
to its scientific value for this prospective experiment." poly2's N1a-scale engineering unit took 134 min on synthetic targets; under
shuffled targets rff (the most similar grid family: L2 head on ~1k constructed features) ran ≈ 4.3× its benchmark, projecting poly2 null to
≈ 9–10 h. This is NOT a scientific failure of poly2, NOT evidence poly2 would not work, and NOT a reinterpretation of any result.

## 4. Five-family prospective panel

| family | Decoder Panel v1 definition (unchanged) | group | HPO |
|---|---|---|---|
| linear | L2 multinomial logistic, λ ∈ {1e1…1e-7} | L2-linear-head | grid |
| rff | RBF random Fourier features (1024) → L2 head; γ ∈ {0.25…4}×γ_med × λ | L2-linear-head | grid |
| LightGBM | raw coordinates, max_delta_step 2.0 | trees | Optuna, 50 trials |
| MLP | one hidden layer, CPU | neural | Optuna, 50 trials |
| kNN | standardize → (JL if needed) → kNN; k × weighting | local | grid |

## 5. Per-family null-control table (shuffled Δ within each training environment; production procedure, panel seed)

I1 = route fraction in [1 %, 99 %] (on pseudo-eval rows' features, label-free proxy); I2 = inner margin ≥ τ_family; joint = I1 and I2.

| family | τ (pp) | unit | route fraction | inner margin (pp) | I1 | I2 | joint |
|---|---|---|---|---|---|---|---|
| linear | 0.1000 | u0 | 1.0000 | +0.0000 | fail | fail | **fail** |
| linear | 0.1000 | u1 | 0.0000 | -0.0000 | fail | fail | **fail** |
| linear | 0.1000 | u2 | 1.0000 | +0.0056 | fail | fail | **fail** |
| linear | 0.1000 | u3 | 0.0000 | +0.0500 | fail | fail | **fail** |
| rff | 0.1000 | u0 | 1.0000 | +0.0000 | fail | fail | **fail** |
| rff | 0.1000 | u1 | 0.0000 | -0.0000 | fail | fail | **fail** |
| rff | 0.1000 | u2 | 0.9948 | +0.0833 | fail | fail | **fail** |
| rff | 0.1000 | u3 | 0.0000 | -0.0000 | fail | fail | **fail** |
| lgbm | 0.1344 | u0 | 0.9972 | +0.0222 | fail | fail | **fail** |
| lgbm | 0.1344 | u1 | 0.0370 | +0.0833 | pass | fail | **fail** |
| lgbm | 0.1344 | u2 | 0.9758 | +0.1667 | pass | pass | **PASS** |
| lgbm | 0.1344 | u3 | 0.0063 | +0.1221 | fail | fail | **fail** |
| mlp | 0.1228 | u0 | 0.7657 | -0.0389 | pass | fail | **fail** |
| mlp | 0.1228 | u1 | 0.0210 | +0.0389 | pass | fail | **fail** |
| mlp | 0.1228 | u2 | 0.6588 | +0.0889 | pass | fail | **fail** |
| mlp | 0.1228 | u3 | 0.0844 | +0.1110 | pass | fail | **fail** |
| knn | 0.1000 | u0 | 0.5333 | -0.3833 | pass | fail | **fail** |
| knn | 0.1000 | u1 | 0.2845 | -0.5889 | pass | fail | **fail** |
| knn | 0.1000 | u2 | 0.5408 | -0.2056 | pass | fail | **fail** |
| knn | 0.1000 | u3 | 0.2411 | -0.1998 | pass | fail | **fail** |
| family | null units passing jointly | max apparent inner gain under shuffled labels (pp) | triggers STOP (≥ 2/4)? |
|---|---|---|---|
| linear | 0 / 4 | +0.050 | no |
| rff | 0 / 4 | +0.083 | no |
| LightGBM | **1 / 4** (u2) | +0.167 | no |
| MLP | 0 / 4 | +0.111 | no |
| kNN | 0 / 4 | −0.200 | no |

**STOP rule does not fire for any remaining family.** (The rule was defined for the six-family audit; it is applied unchanged, per family,
to the five completed families.)

## 6. LightGBM stochasticity audit (4 units; 5 full-HPO replicates and 5 final-fit-only replicates; variability only)

Procedure A = real-unit I2 quantity; procedure B = pseudo-unit φ, U, M_A (pseudo-held-out families: u0 defocus_blur, u1 fog,
u2 jpeg_compression, u3 gaussian_noise). Centered deviations are in `summary_5family.json`.

| quantity | unit | full-HPO SD | full-HPO range | final-fit-only SD | final-fit-only range |
|---|---|---|---|---|---|
| I2 inner margin (pp) | u0 | 0.0405 | 0.0889 | 0.0658 | 0.1778 |
| I2 inner margin (pp) | u1 | 0.0672 | 0.1611 | 0.0484 | 0.1278 |
| I2 inner margin (pp) | u2 | 0.0154 | 0.0389 | 0.0658 | 0.1556 |
| I2 inner margin (pp) | u3 | 0.0292 | 0.0611 | 0.0280 | 0.0555 |
| φ (fraction) | u0 | 0.0139 | 0.0331 | 0.0064 | 0.0177 |
| φ (fraction) | u1 | 0.0240 | 0.0574 | 0.0141 | 0.0340 |
| φ (fraction) | u2 | 0.0097 | 0.0212 | 0.0182 | 0.0425 |
| φ (fraction) | u3 | 0.0119 | 0.0299 | 0.0077 | 0.0199 |
| U (pp) | u0 | 0.1047 | 0.2500 | 0.0480 | 0.1333 |
| U (pp) | u1 | 0.1880 | 0.4500 | 0.1108 | 0.2667 |
| U (pp) | u2 | 0.0838 | 0.1833 | 0.1571 | 0.3667 |
| U (pp) | u3 | 0.0793 | 0.1998 | 0.0513 | 0.1332 |
| M_A (pp) | u0 | 0.0500 | 0.1333 | 0.0560 | 0.1333 |
| M_A (pp) | u1 | 0.1718 | 0.3833 | 0.1051 | 0.2667 |
| M_A (pp) | u2 | 0.1464 | 0.3667 | 0.1070 | 0.2667 |
| M_A (pp) | u3 | 0.0630 | 0.1499 | 0.1132 | 0.3164 |
## 7. MLP stochasticity audit (same design)

| quantity | unit | full-HPO SD | full-HPO range | final-fit-only SD | final-fit-only range |
|---|---|---|---|---|---|
| I2 inner margin (pp) | u0 | 0.0556 | 0.1278 | 0.0724 | 0.1667 |
| I2 inner margin (pp) | u1 | 0.0445 | 0.0944 | 0.1056 | 0.2333 |
| I2 inner margin (pp) | u2 | 0.0371 | 0.0944 | 0.1341 | 0.3611 |
| I2 inner margin (pp) | u3 | 0.0614 | 0.1721 | 0.0925 | 0.2165 |
| φ (fraction) | u0 | 0.0197 | 0.0486 | 0.0088 | 0.0221 |
| φ (fraction) | u1 | 0.0420 | 0.1149 | 0.0386 | 0.0957 |
| φ (fraction) | u2 | 0.0313 | 0.0849 | 0.0731 | 0.2027 |
| φ (fraction) | u3 | 0.0149 | 0.0373 | 0.0269 | 0.0647 |
| U (pp) | u0 | 0.1484 | 0.3667 | 0.0662 | 0.1667 |
| U (pp) | u1 | 0.3288 | 0.9000 | 0.3023 | 0.7500 |
| U (pp) | u2 | 0.2705 | 0.7333 | 0.6313 | 1.7500 |
| U (pp) | u3 | 0.0996 | 0.2498 | 0.1803 | 0.4329 |
| M_A (pp) | u0 | 0.1063 | 0.2667 | 0.0874 | 0.2167 |
| M_A (pp) | u1 | 0.1931 | 0.4833 | 0.2402 | 0.5833 |
| M_A (pp) | u2 | 0.0535 | 0.1333 | 0.2419 | 0.5833 |
| M_A (pp) | u3 | 0.2404 | 0.5828 | 0.1478 | 0.3497 |
## 8. Derived noise constants and τ (s = max per-unit full-HPO SD; no √N division)

| family | s(I2 inner margin) pp | τ = max(0.10, 2s) pp | s(φ) | s(U) pp | s(M_A) pp |
|---|---|---|---|---|---|
| LightGBM | 0.0672 | **0.1344** | 0.0240 | 0.1880 | 0.1718 |
| MLP | 0.0614 | **0.1228** | 0.0420 | 0.3288 | 0.2404 |
| linear, rff, kNN (deterministic) | 0 | **0.10** | 0 | 0 | 0 |

Which scale applies where (exact threshold metrics):
- I2: inner margin ≥ τ_family.
- RESOLVED: φ_lo − 2 s(φ) ≥ 0.50 (LightGBM: φ_lo ≥ 0.548; MLP: φ_lo ≥ 0.584) AND MA_hi + 2 s(M_A) ≤ 3.5 pp (LightGBM: MA_hi ≤ 3.156;
  MLP: MA_hi ≤ 3.019).
- SUBSTANTIAL: U_lo − 2 s(U) ≥ 4.0 pp (LightGBM: U_lo ≥ 4.376; MLP: U_lo ≥ 4.658) AND MA_lo − 2 s(M_A) ≥ 2.0 pp (LightGBM: MA_lo ≥ 2.344;
  MLP: MA_lo ≥ 2.481).
- Deterministic families: thresholds without margin (φ_lo ≥ 0.50, MA_hi ≤ 3.5, U_lo ≥ 4.0, MA_lo ≥ 2.0).
Final-fit-only maxima (reported, not used by the frozen formula): LightGBM I2 0.0658, φ 0.0182, U 0.1571, M_A 0.1132; MLP I2 **0.1341**,
φ **0.0731**, U **0.6313**, M_A **0.2419** — see §11.

## 9. Continuity-control result

C0 (historical N1a linear selector; `results/n1a/fits/base{2,4}/<family>/Z/fold{0-4}.npz`, reused exactly) re-scored by the N1a-DP
aggregator: base 2 family-macro G = 2.0316666666666663 pp vs historical 2.0316666666666663 (|Δ| = 0.0); base 4 = 1.2475 vs 1.2475 (|Δ| = 0.0).
**Reproduces exactly** (tolerance 1e-9 pp).

## 10. Family-by-family methodological assessment (valid ≠ informative)

Validity (spec §7 rule 0) is about completeness, convergence, grid-edge and support; informativeness (§6.2) is about whether the family
actually implements a non-constant policy that beats the best constant on inner validation. A valid family may be non-informative.
- **linear** — deterministic; under the null it selects the strongest λ (1e1 or 1e0) and collapses to a constant policy (route fraction
  exactly 0 or 1): I1 correctly rejects. Known sensitivity: historical N1a (λ ≤ 1e-1 grid) sat at its weak edge in 77/80 fits; the panel grid
  extends to 1e-7 (weak edge reported, not invalidating). Engineering benchmark (synthetic) selected λ = 1e-7 (edge). Expected to be the most
  stable reference.
- **rff** — deterministic; under the null it selects γ = 0.25 × γ_med (smoothest grid value) with λ = 1e1 in 3/4 units (constant policy) and
  γ = 1, λ = 1 once (route 99.5 %, just outside I1). Boundary tendency: γ at the 0.25 edge in 52/110 G1-DP studies and in the N1a engineering
  benchmark. Most expensive family here (≈ 1.7–1.9 h per null unit).
- **LightGBM** — stochastic; non-trivial hyperparameters under the null; the only null joint pass (u2: route 97.6 %, margin 0.167 pp just
  above τ = 0.134 pp) is a near-constant route-all policy — selection optimism of a 50-trial search. Seed SD of I2 ≤ 0.067 pp; φ ≤ 0.024.
- **MLP** — stochastic; no null joint pass (max margin 0.111 < τ 0.123). Final-fit-only SDs exceed full-HPO SDs in several units (e.g. u2:
  U 0.63 vs 0.27 pp), i.e. training noise at a fixed configuration is not smaller than HPO-path noise; with 5 replicates each SD is itself
  uncertain (roughly ±35 %).
- **kNN** — deterministic; under the null it routes 24–54 % of rows with strongly negative inner margins (−0.20 to −0.59 pp): non-degenerate
  but worse than constant, so I2 rejects. In G1-DP (anchored) it selected the anchor-only candidate in 97/110 studies; in unanchored N1a-DP
  there is no anchor candidate, but the null behavior suggests it may be NON-INFORMATIVE on real targets (a methodological expectation,
  not an outcome).

## 11. Remaining validity concerns (for researcher decision before any freeze)

1. **I1 bound permissiveness.** Under shuffled targets, near-constant policies pass I1 and are excluded only by I2 (MLP u1 route 2.1 %,
   LightGBM u1 3.7 %, MLP u3 8.4 %); LightGBM's single joint null pass routes 97.6 % (LightGBM u3 at 0.63 % failed I1). The joint criterion held (STOP does not fire), so per the instruction the
   [1 %, 99 %] bound is kept in draft r4; tightening (e.g. [5 %, 95 %], which would have rejected the only null joint pass) is an open option.
2. **Noise definition.** The frozen formula uses full-HPO SDs; MLP final-fit-only SDs are larger (I2 0.134 vs 0.061 pp; U 0.631 vs 0.329 pp).
   Using s = max(full-HPO, final-fit-only) would raise MLP τ to 0.268 pp and its margins accordingly. Not adopted without approval.
3. **Replicate count.** SDs from 5 replicates are imprecise; the "max over 4 units" rule partially compensates.
4. **Pseudo-unit scale.** φ/U/M_A variability was measured on pseudo-units (one family × inner-val images, half-size training) — conservative
   for family-macro production quantities, but not an exact match.
5. **Grid edges.** rff γ and linear λ edge tendencies (reported, not invalidating under the current rules).
6. **Minimum valid families.** With five families, draft r4 sets INCONCLUSIVE (validity) at < 4 valid (tolerating one invalid family, as the
   six-family rule did); flagged for confirmation.

## 12. Exact proposed five-family N1a-DP decision rules (draft r4; not frozen)

0. INCONCLUSIVE (validity): < 4 of 5 families valid, V-C0 fails, or any holdout / image / Z_o assertion fails.
1. OUTPUT-DECODER LIMITATION: ≥ 2 valid families RESOLVED (seed-robust) in both bases from **distinct groups** among {linear, rff},
   {LightGBM}, {MLP}, {kNN}.
2. PANEL-ROBUST OUTPUT AMBIGUITY: ≥ 4 of the 5 families valid AND INFORMATIVE in both bases; every counted informative family SUBSTANTIAL
   (seed-robust) in both bases; none RESOLVED in either base; M_B ≥ 1.0 pp in both bases (unchanged).
3. FAMILY-SPECIFIC: some valid family RESOLVED (seed-robust) in at least one base, rule 1 not met.
4. INCONCLUSIVE: otherwise.
INFORMATIVE (per family, per base): I1 and I2 pass in ≥ 16 of 20 units overall AND ≥ 3 of 5 folds within every held-out family; τ as in §8.
Thresholds unchanged: φ 0.50, M_A 3.5 / 2.0 pp, U 4.0 pp, M_B 1.0 pp.

## 13. Resource implications

Measured per-unit costs (8 cores): linear ≈ 3–7 min; rff ≈ 25 min (synthetic benchmark) to ≈ 1.7–1.9 h (shuffled); LightGBM ≈ 2–17 min per
50-trial study; MLP ≈ 17–20 min per study; kNN ≈ 3–4 min. Audit wall-clock per job: A LightGBM 13–18 min, A MLP 53–79 min, B LightGBM 3–5 min,
B MLP 12–21 min. Projected five-family N1a-DP: 5 × 40 units ≈ 300–800 core-hours, a few hours elapsed at %20 per family (rff dominant).
Removing poly2 removes the largest projected cost (≈ 90 unit-hours at ≥ 2.2 h per unit).

## 14. Current N1a-DP status

**DRAFT / PAUSED / NOT AUTHORIZED.** Not frozen (no sidecar), not submitted, no scientific output. N1b not begun. The N1a/N1b line is paused
pending research-direction selection (internal computation trajectories → recoverability → selective internal repair / intervention), to be
handled in a fresh session.
