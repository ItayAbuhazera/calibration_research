# N1a-DP — stochasticity and sensitivity audit plan (engineering only; DRAFT for researcher review)

Purpose: before freezing `docs/n1a_dp_spec.md` (§6.2–6.4), estimate how much the realized N1a-DP metrics of the two stochastic Decoder
Panel v1 families (LightGBM, MLP) move under changes of algorithmic randomness alone, and check that the informativeness criteria behave
sensibly. Not a search: replicates are never used to select anything; production N1a-DP uses the single frozen panel seed per study.
Decoder Panel v1 and G1-DP are not modified.

## 1. What is varied

- **HPO + training seed (primary):** R = 5 replicate audit master seeds (e.g. 910001…910005, never the panel seed 20260928), each
  running the full frozen shift-transfer procedure (50 Optuna TPE trials, three inner leave-one-training-family-out splits, early stopping,
  refit, evaluation). Varies TPE sampling, trial seeds and refit seed together.
- **Training seed only (decomposition):** at the configuration selected in replicate 1, 5 refit-only seeds (no HPO). Separates
  optimizer/initialization noise from HPO-path noise.
- Fixed: data, splits, frozen spaces, budget, objective, features, all other code.

## 2. Where (bounded representative subset)

Two real N1a-DP outer units (real row layout, real F_Z(b) features, real inner splits):
(a) base 2, held-out fog, fold 0; (b) base 4, held-out jpeg_compression, fold 3 (the other base and the family with the largest N1a
ambiguity). **Targets are synthetic**, not the real Δ_route: a fixed random teacher on F_Z(b) produces Δ ∈ {−1, 0, +1} with repair / harm
rates ≈ 9 % / 8 % (as in the existing engineering benchmark, `atlas/dp_engineering.py`, mode n1a). Rationale: running replicates on the
real targets would expose N1a-DP outcomes (utility, φ, M_A) before the spec is frozen. Limitation: the synthetic signal-to-noise differs
from the real one, so the audit measures algorithmic variability under matched data geometry, not the real-target variance itself.
(Alternative for the researcher to choose instead: run the same replicates on real targets but report only the between-seed SDs to the
spec, keeping means blinded. Not proposed as default because partial blinding is hard to verify.)

## 3. Measured per replicate (on the synthetic held-out rows)

Realized utility G (pp), φ (vs the synthetic headroom), U = G_or − G (pp), M_A (10 score bins), evaluation route rate, selected inner
objective, inner margin over the inner best-constant policy (I2 quantity), selected hyperparameters, runtime.

## 4. Outputs → spec §6.4

For each of LightGBM and MLP: s_X = standard deviation over the 5 HPO replicates of each metric X ∈ {G, φ, U, M_A, inner margin}, taken as
the maximum over the two units, converted to the family-macro scale by dividing unit-level SDs by √4 (family-macro averages four held-out
families; conservative alternative √1 if replicate correlations across units cannot be assumed away — **proposed: use the unit-level SD
without division, i.e. conservative**). Also reported: the refit-only SDs and the fraction of variance attributable to HPO path.
These values are written into `n1a_dp_spec.md` §6.4 and determine τ (I2) and the seed-robust margins (§6.3).

Sanity checks on informativeness (engineering): I1/I2 evaluated on the synthetic units for all six families (single frozen seed) to
confirm the criteria are neither trivially satisfied nor trivially failed when real (synthetic) signal exists; plus one **null-target
control** (Δ permuted within inner environments) where every family should come out NON-INFORMATIVE by I2.

## 5. Cost and execution

LightGBM ≈ 17 min and MLP ≈ 19 min per full N1a-scale study on 8 cores (Stage-B benchmark). 2 units × 2 families × (5 HPO + 5 refit-only)
plus 6 families × 2 units (informativeness check) plus the null control ≈ 60 short jobs ≈ 60 core-hours, < 1 h elapsed. `cpu` partition,
8 cores, 16G, 2 h limits, array throttle %20. Outputs `results/n1adp_audit/` (engineering), journal entries in the master report.

## 6. Implementation needed before the audit (small, tested)

1. `atlas/n1adp.py`: store per unit the evaluation route rate and the inner best-constant objective (for I1/I2).
2. `atlas/n1adp_aggregate.py`: informativeness (I1/I2), seed-robust labels, continuity control C0 (re-score `results/n1a/fits/*/Z`),
   V-C0 reproduction check; decision per revised §7.
3. `atlas/n1adp_rules.py`: revised pure decision function + tests.
4. `atlas/dp_engineering.py`: `--mode n1a_seed` replicate runner (audit master seed override, second unit, null-target control).
