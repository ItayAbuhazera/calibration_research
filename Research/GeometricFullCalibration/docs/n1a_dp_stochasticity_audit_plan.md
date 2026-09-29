# N1a-DP — pre-freeze stochasticity and sensitivity audit (FROZEN audit protocol)

Status: **FROZEN (2026-09-29) before execution**, sidecar `n1a_dp_stochasticity_audit_plan.frozen.sha256`. Engineering audit for
`docs/n1a_dp_spec.md` (§6.2–6.4). Runner: `atlas/n1adp_audit.py` (tests in `tests/test_decoder_panel.py`). Decoder Panel v1 and G1-DP
are not modified. Researcher instruction of 2026-09-29 (supersedes the earlier synthetic-target draft of this plan).

## 1. Access boundary (hard; asserted in code and tests)

For every audit unit (base b, outer held-out family f, outer fold k) the runner builds ONLY the unit's training rows: the three training
families × fold-k outer-train images, loading only those cells. No row of family f, no fold-k outer-test image, and no outer held-out label
or outcome is ever read. Real Δ_route targets are used only on those training rows. Outputs contain **only across-seed variability**
(SD with ddof = 1, max − min, and deviations from the replicate mean) — never a mean or a per-replicate level of any performance quantity.

## 2. Units (preregistered; one per held-out family, balanced across bases and folds)

| unit | base | outer held-out family | fold | pseudo-held-out family (cyclic successor) |
|---|---|---|---|---|
| u0 | 2 | gaussian_noise | 0 | defocus_blur |
| u1 | 4 | defocus_blur | 1 | fog |
| u2 | 2 | fog | 2 | jpeg_compression |
| u3 | 4 | jpeg_compression | 3 | gaussian_noise |

## 3. Procedures (LightGBM and MLP, the stochastic families)

**A — I2 quantity on the real unit.** Production shift-transfer HPO on the unit's training rows (3 inner leave-one-training-family-out
splits; inner-fit / inner-val images from the Stage-0 mask; objective = negative realized policy utility). Quantity = inner margin (pp) =
inner utility of the selected configuration − inner best-constant utility (`atlas.n1adp.inner_const_utility`: per split, route-all iff mean
Δ on the split's fit rows > 0, scored on its validation rows; macro). Exactly the production I2 quantity.
- 5 full-HPO replicates: audit master seeds 910001…910005 (vary TPE sampling, trial seeds, refit seed).
- 5 final-fit-only replicates: replicate 0's selected configuration and best iterations, re-trained per split with 5 new seeds.

**B — φ, U, M_A on a pseudo-unit inside the training rows.** Pseudo-held-out family g (table above). Pseudo-train = the two other training
families × inner-fit images; pseudo-eval = g × inner-val images (image-disjoint). Pseudo HPO = shift-transfer with 2 inner splits (each of the
two pseudo-train families held out once) over a fixed duplicate-group-aware 75/25 sub-split of the inner-fit images (seed 910100).
Quantities on pseudo-eval, defined exactly as in the spec: φ = (G − G_c)/(G_or − G_c); U = G_or − G (pp); M_A = Σ over 10 quantile bins of the
decision score of min(repairs, harms) / rows (pp); G (pp) reported for reference.
- 5 full-HPO replicates (same audit seeds); 5 final-fit-only refits of replicate 0's selection (new refit seeds).

## 4. Noise scales and derived constants (computed by `python -m atlas.n1adp_audit summary`)

For each stochastic family m and each threshold-driving quantity q ∈ {inner margin, φ, U, M_A}:
**s_m(q) = the maximum over the four audit units of the per-unit full-HPO replicate SD** (conservative; no division by √(number of held-out
families) or √(units)). τ_m = max(0.10 pp, 2 · s_m(inner margin)). Deterministic families (linear, poly2, knn, rff): s = 0, τ = 0.10 pp.
Final-fit-only SDs are reported to decompose HPO-path vs training noise; they are not used for thresholds.
Known conservatism: pseudo-units train on ~half the rows of production units and evaluate on one family × one-quarter of the images, and
production quantities average 20 independently fitted units; the per-unit SD therefore overstates the family-macro seed SD.

## 5. Shuffled-target null (all six families)

Per audit unit: Δ permuted within each training environment (seed 910200 + unit); production procedure with the frozen panel seed on the
unit's training rows. I2 = inner margin ≥ τ_m; I1 = route rate of the refitted selector on the pseudo-eval rows' features (label-free proxy
for the production evaluation route rate) within [1 %, 99 %]. **STOP rule: if any family passes I1 and I2 in ≥ 2 of the 4 null units, STOP
and revise the informativeness criterion before freezing N1a-DP** (stricter than the full 16/20 criterion scaled to 4 units).

## 6. Execution

`cpu` partition, 8 cores / 24G: A jobs (8) 6 h limit, B jobs (8) 4 h, null jobs (24) 6 h for poly2 and 4 h otherwise, all from an
immutable snapshot `snapshots/n1adp_audit_*`; summary afterok all. Outputs `results/n1adp_audit/{A,B,null}/<family>/u<k>.json`,
`results/n1adp_audit/summary.json`; logs `results/n1adp_audit/logs/`. ≈ 150–200 core-hours.
