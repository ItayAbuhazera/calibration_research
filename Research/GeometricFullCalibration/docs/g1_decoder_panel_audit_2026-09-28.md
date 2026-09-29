# G1-DP — Decoder Panel accessibility audit of G1 (report)

Development diagnostic on exposed CIFAR-100-C cells. **TARGET-POOLED HPO — ACCESSIBILITY DIAGNOSTIC ONLY** (target-label access; not
deployment, source-trained generalization or unseen-shift transfer). Completed 2026-09-29.

**Official frozen outcome: INCONCLUSIVE (validity) for all three frozen contrasts (B−C primary, H−C raw tier, D−C conditional).**
Everything in §8 below the verdict is descriptive and carries no verdict weight.

Artifacts: `results/g1dp/report/g1dp_aggregate.json` (sha256 `0b2c6a4899356c76979521bd7fec5f96e56a7cb71de86323b8d2841dd6fe7921`),
`results/g1dp/report/g1dp_table.md`; per-unit outputs `results/g1dp/fits/<family>/seed<s>/fold<k>/<arm>.{npz,study.json,done.json}`;
bundles `results/g1dp/bundles/`; raw H_3.22 `results/g1dp/h322/` (+ `tie_aware_audit.json`); logs `results/g1dp/logs/`; ledger
`results/g1dp/ledger.json`. Specs: `docs/decoder_panel_v1_spec.md` (sha256 `c27a1df4…`, `0dc2041`), `docs/g1_dp_spec.md`
(`0cbfb62e…`, `096ed35`), `docs/g1_dp_spec_amendment_1.md` (`6e3930cb…`, `d399af3`). Snapshot `snapshots/g1dp_a1_5f363f77476c`.
Journal: `docs/decoder_panel_v1_master_report_2026-09-28.md`.

# 1. Historical G1 result (unchanged)

G1 (anchored linear readout, T-8k×1; frozen **Outcome C — capacity / estimation confound**, both checkpoints): B − A = +1.88 / +1.80 pp,
C − A = +0.27 / +0.39 pp, conditional D − C = +0.24 / +0.24 pp, equivalent-span sensitivity |G − C| = 0.72 / 0.50 pp, D − B = −1.38 / −1.16 pp.
G1 did not establish absence of the information from H_L or depth-specific evidence. G1-DP does not alter that verdict.

# 2. Why G1-DP exists

G1's parameterization sensitivity (0.5–0.7 pp) exceeded its conditional contrast, and N1a's selector sat at its weak-regularization grid
edge in 77/80 fits. We therefore could not separate "the evidence channel lacks usable information" from "this readout did not exploit
it". G1-DP asks: does the accessibility difference between compact P_3.22 and penultimate H_L survive changes in decoder family?

# 3. Frozen Decoder Panel v1

Six families, closed set: linear (L2 anchored logistic, λ grid 1e1…1e-7), poly2 ([x̃, TensorSketch₂] → L2 head; dims 512/1024/2048),
LightGBM (raw coordinates, Optuna 50 trials, max_delta_step 2.0), one-hidden-layer MLP (Optuna 50 trials, CPU), JL+kNN (k × weighting × β
grid, JL eps 0.5), RBF random features (1024 comps, γ ∈ {0.25…4}×γ_med × λ). All anchored at z; anchor-only candidate always selectable.

# 4. HPO protocol

Target-pooled regime (preserves G1's fitting unit — one fit per checkpoint × outer fold × arm, pooled over the 12 cells; the task
prompt's per-cell assumption was contradicted by `atlas/g1_fit.py`, recorded in the journal). Selection on the Stage-0 image-grouped inner
split (6,000 / 2,000 rows) by inner-validation NLL; refit on all 8,000 outer-train rows; single evaluation on outer-test images × 13
conditions. Identical spaces/budgets across arms; deterministic study seeds; no outer-test label reaches selection (tested).
Arms A, B, C, D, E, F (G1 meanings), H = z + H_3.22_raw, I = z + H_L + H_3.22_raw, shuffles Cs, Hs, Ds (G1 D-shuf rule).
Raw tier: H_3.22_raw = L2-normalized GAP(layer3.22), re-extracted strict FP32; Amendment 1 (pre-outcome) made the extraction gate's
top-class agreement tie-aware (all 157 plain-argmax disagreements were exact float16 ties; 0 on non-tied rows); gate passed both checkpoints.

# 5. SLURM execution / resource use

660 HPO studies (2 checkpoints × 5 folds × 11 arms × 6 families) in 460 array tasks + 2 GPU extractions + 10 bundles + 1 aggregation.
≈ 2,600 CPU core-hours, 0.02 GPU-hours (estimate was ≈ 5,600). Zero failed, OOM or timed-out tasks; no resubmissions. Scheduling-only
changes: lgbm TimeLimit 24 h → 6 h and array throttle 20 → 40 (researcher-authorized; longest unit 5 h 14 min). Runtime scaled with arm
input width (LightGBM 7 min for 100-d to ~5 h for 3,172-d).

# 6. Results for every decoder family (12-cell macro top-1 accuracy, pp; 95 % image-group bootstrap, B = 2000, shared resamples)

Class labels per the frozen rule (m = 0.5 pp): POS / REV / NULL (interval inside ±0.5) / UNC.

### Checkpoint 2

| contrast (pp) | linear | poly2 | lgbm | mlp | knn | rff |
|---|---|---|---|---|---|---|
| B-A | +1.88 [+1.73, +2.03] POS | +2.30 [+2.12, +2.47] POS | +1.68 [+1.51, +1.84] POS | +1.99 [+1.79, +2.19] POS | +0.00 [+0.00, +0.00] NULL | +3.13 [+2.92, +3.33] POS |
| C-A | +0.27 [+0.19, +0.35] NULL | +0.33 [+0.25, +0.42] NULL | +0.55 [+0.42, +0.70] POS | -0.07 [-0.17, +0.03] NULL | +0.00 [+0.00, +0.00] NULL | +0.01 [-0.05, +0.07] NULL |
| D-C | +0.24 [+0.20, +0.28] NULL | +0.36 [+0.29, +0.42] NULL | +0.63 [+0.49, +0.76] POS | +0.32 [+0.20, +0.43] NULL | +0.00 [+0.00, +0.00] NULL | +0.35 [+0.26, +0.43] NULL |
| D-B | -1.38 [-1.52, -1.24] REV | -1.60 [-1.76, -1.44] REV | -0.50 [-0.66, -0.34] UNC | -1.74 [-1.94, -1.54] REV | +0.00 [+0.00, +0.00] NULL | -2.77 [-2.98, -2.58] REV |
| E-C | +0.02 [+0.00, +0.04] NULL | -0.03 [-0.08, +0.02] NULL | -0.08 [-0.20, +0.03] NULL | +0.06 [-0.05, +0.17] NULL | +0.00 [+0.00, +0.00] NULL | +0.01 [-0.04, +0.05] NULL |
| F-C | +0.51 [+0.45, +0.56] POS | +0.79 [+0.71, +0.87] POS | +1.42 [+1.27, +1.56] POS | +0.33 [+0.21, +0.45] NULL | +0.00 [+0.00, +0.00] NULL | +2.19 [+2.03, +2.35] POS |
| B-C | +1.62 [+1.47, +1.77] POS | +1.96 [+1.79, +2.13] POS | +1.12 [+0.96, +1.29] POS | +2.06 [+1.86, +2.27] POS | +0.00 [+0.00, +0.00] NULL | +3.12 [+2.91, +3.33] POS |
| H-A | +4.57 [+4.33, +4.82] POS | +4.39 [+4.14, +4.63] POS | +3.04 [+2.84, +3.25] POS | +3.32 [+3.08, +3.56] POS | +0.48 [+0.41, +0.55] UNC | +4.04 [+3.81, +4.27] POS |
| H-C | +4.31 [+4.07, +4.54] POS | +4.06 [+3.82, +4.29] POS | +2.49 [+2.30, +2.69] POS | +3.39 [+3.14, +3.65] POS | +0.48 [+0.41, +0.55] UNC | +4.03 [+3.80, +4.26] POS |
| I-C | +1.99 [+1.85, +2.11] POS | +2.23 [+2.08, +2.38] POS | +1.83 [+1.67, +1.99] POS | +2.15 [+1.95, +2.35] POS | +0.00 [+0.00, +0.00] NULL | +3.21 [+3.00, +3.41] POS |
| I-H | -2.32 [-2.50, -2.13] REV | -1.82 [-2.00, -1.64] REV | -0.67 [-0.82, -0.51] REV | -1.24 [-1.45, -1.03] REV | -0.48 [-0.55, -0.41] UNC | -0.82 [-0.99, -0.65] REV |
| Cs-A | -0.73 [-0.84, -0.62] REV | -0.84 [-0.98, -0.70] REV | -0.48 [-0.63, -0.32] UNC | -1.00 [-1.18, -0.84] REV | +0.01 [-0.01, +0.03] NULL | -0.72 [-0.85, -0.59] REV |
| Hs-A | +0.26 [+0.13, +0.38] NULL | +0.08 [-0.07, +0.22] NULL | +0.18 [+0.03, +0.34] NULL | -0.10 [-0.27, +0.05] NULL | +0.01 [-0.00, +0.03] NULL | +0.03 [-0.10, +0.15] NULL |
| Ds-C | +0.13 [+0.08, +0.19] NULL | +0.12 [+0.05, +0.19] NULL | +0.12 [+0.01, +0.24] NULL | +0.20 [+0.09, +0.32] NULL | +0.00 [+0.00, +0.00] NULL | +0.01 [-0.04, +0.07] NULL |
| arm accuracies A/B/C/H | 52.24/54.12/52.51/56.81 | 52.38/54.67/52.71/56.77 | 52.31/53.98/52.86/55.35 | 52.76/54.75/52.69/56.08 | 50.09/50.09/50.09/50.57 | 52.28/55.41/52.29/56.32 |
| valid | False | True | True | False | True | True |

### Checkpoint 4

| contrast (pp) | linear | poly2 | lgbm | mlp | knn | rff |
|---|---|---|---|---|---|---|
| B-A | +1.80 [+1.64, +1.96] POS | +2.23 [+2.05, +2.40] POS | +1.70 [+1.53, +1.86] POS | +2.04 [+1.85, +2.25] POS | +0.00 [+0.00, +0.00] NULL | +2.89 [+2.68, +3.11] POS |
| C-A | +0.39 [+0.31, +0.47] NULL | +0.42 [+0.34, +0.51] UNC | +0.45 [+0.31, +0.60] UNC | +0.03 [-0.06, +0.12] NULL | +0.00 [+0.00, +0.00] NULL | -0.04 [-0.10, +0.03] NULL |
| D-C | +0.24 [+0.20, +0.28] NULL | +0.32 [+0.26, +0.38] NULL | +0.67 [+0.54, +0.79] POS | +0.28 [+0.17, +0.38] NULL | +0.00 [+0.00, +0.00] NULL | +0.13 [+0.08, +0.18] NULL |
| D-B | -1.16 [-1.31, -1.02] REV | -1.49 [-1.65, -1.33] REV | -0.58 [-0.73, -0.42] REV | -1.73 [-1.93, -1.56] REV | +0.00 [+0.00, +0.00] NULL | -2.80 [-3.02, -2.59] REV |
| E-C | +0.02 [+0.00, +0.04] NULL | -0.00 [-0.05, +0.04] NULL | -0.01 [-0.12, +0.10] NULL | -0.01 [-0.10, +0.10] NULL | +0.00 [+0.00, +0.00] NULL | +0.02 [-0.03, +0.07] NULL |
| F-C | +0.36 [+0.31, +0.42] NULL | +0.68 [+0.61, +0.76] POS | +1.08 [+0.95, +1.21] POS | +0.16 [+0.06, +0.25] NULL | +0.00 [+0.00, +0.00] NULL | +1.72 [+1.56, +1.89] POS |
| B-C | +1.41 [+1.26, +1.55] POS | +1.81 [+1.63, +1.97] POS | +1.25 [+1.08, +1.41] POS | +2.01 [+1.83, +2.22] POS | +0.00 [+0.00, +0.00] NULL | +2.93 [+2.71, +3.14] POS |
| H-A | +4.29 [+4.05, +4.55] POS | +4.25 [+4.02, +4.52] POS | +3.00 [+2.80, +3.18] POS | +3.28 [+3.04, +3.53] POS | +0.58 [+0.50, +0.66] POS | +3.74 [+3.51, +3.97] POS |
| H-C | +3.90 [+3.67, +4.15] POS | +3.83 [+3.60, +4.09] POS | +2.54 [+2.35, +2.74] POS | +3.25 [+3.02, +3.52] POS | +0.58 [+0.50, +0.66] POS | +3.77 [+3.54, +4.01] POS |
| I-C | +1.93 [+1.81, +2.06] POS | +2.25 [+2.10, +2.40] POS | +1.86 [+1.69, +2.02] POS | +1.75 [+1.59, +1.92] POS | +0.00 [+0.00, +0.00] NULL | +3.10 [+2.89, +3.31] POS |
| I-H | -1.97 [-2.16, -1.80] REV | -1.59 [-1.77, -1.41] REV | -0.68 [-0.84, -0.53] REV | -1.50 [-1.71, -1.28] REV | -0.58 [-0.66, -0.50] REV | -0.68 [-0.86, -0.50] REV |
| Cs-A | -0.80 [-0.91, -0.69] REV | -0.93 [-1.08, -0.78] REV | -0.52 [-0.69, -0.36] REV | -0.95 [-1.13, -0.78] REV | +0.00 [+0.00, +0.00] NULL | -0.89 [-1.02, -0.74] REV |
| Hs-A | +0.09 [-0.03, +0.21] NULL | -0.03 [-0.18, +0.12] NULL | +0.03 [-0.13, +0.18] NULL | -0.08 [-0.25, +0.08] NULL | +0.00 [+0.00, +0.00] NULL | -0.14 [-0.26, -0.02] NULL |
| Ds-C | +0.11 [+0.06, +0.16] NULL | +0.18 [+0.11, +0.25] NULL | +0.07 [-0.04, +0.19] NULL | +0.20 [+0.10, +0.30] NULL | +0.00 [+0.00, +0.00] NULL | +0.09 [+0.04, +0.14] NULL |
| arm accuracies A/B/C/H | 53.25/55.04/53.64/57.54 | 53.37/55.60/53.79/57.62 | 53.37/55.07/53.82/56.36 | 53.73/55.78/53.76/57.02 | 51.01/51.01/51.01/51.59 | 53.38/56.27/53.34/57.12 |
| valid | True | True | True | True | True | True |

Verdicts: {"B-C": "INCONCLUSIVE (validity)", "H-C": "INCONCLUSIVE (validity)", "D-C": "INCONCLUSIVE (validity)"}
Selection diagnostics (110 studies per family): anchor-only candidate selected — linear 0, poly2 0, lgbm 0, mlp 0, **knn 97/110** (every
fold of arms A, B, C, D, E, F and most others; kNN only departed from the anchor for the raw-H_3.22 arms); strongest-λ edge (1e1) never
selected (V4 passed). Grid-edge flags (reported, not validity criteria): poly2 sketch dimension at 512 or 2048 in 77/110 (both ends;
λ interior); rff γ multiplier at 0.25 (smoothest) in 52/110; linear λ interior in all 110. JL audit (53 projected fits): 5th pct ≥ 0.933,
95th pct ≤ 1.064, median of medians 0.9997.

# 7. Raw H_3.22 tier

Available and valid (Amendment 1 gate passed both checkpoints). Arms H, I, Hs were fitted in all six families; H − C is reported above
and in the verdict below.

# 8. Cross-family verdict (frozen rule, g1_dp_spec §6–8 + Amendment 1)

Validity (per family; must hold in both checkpoints): **valid = poly2, LightGBM, kNN, rff (4 families) < required 5.**
- linear — V5 shuffle control failed at checkpoint 2: Hs − A = **+0.259 pp** (≥ +0.2). (Checkpoint 4: +0.086.)
- MLP — V5 failed at checkpoint 2: Ds − C = **+0.202 pp** (≥ +0.2). (Checkpoint 4: +0.196.)
- All other checks passed for all families: completeness, convergence, no strongest-λ edge, JL band, both extraction gates.

| frozen contrast | outcome |
|---|---|
| **B − C** (compact P_3.22 vs H_L; primary) | **INCONCLUSIVE (validity)** |
| **H − C** (raw H_3.22 vs H_L; raw tier) | **INCONCLUSIVE (validity)** |
| **D − C** (G1 conditional increment) | **INCONCLUSIVE (validity)** |

Per the frozen matrix: INCONCLUSIVE → record; no follow-up by default. The frozen rule is not re-scored; the 0.2 pp shuffle threshold and
the 5-family minimum stand as written.

**Descriptive only (no verdict weight):**
- B − C was POS in 5 of 6 families in both checkpoints (+1.12 to +3.12 pp); the sixth (kNN) is identically 0 because kNN selected the
  anchor in those arms. No family was REV.
- H − C was POS in 5 of 6 families in both checkpoints (+2.49 to +4.31 pp); kNN +0.48 (UNC) / +0.58 (POS).
- D − C was NULL in linear, poly2, MLP, rff (+0.13 to +0.36 pp) and POS only in LightGBM (+0.63 / +0.67 pp); kNN 0.
- Width/capacity cost is visible: Cs − A (shuffled 2,048-d block) = −0.5 to −1.0 pp in five families; I − H (adding H_L to H_3.22) =
  −0.7 to −2.3 pp; so "C − A small" partly reflects the cost of fitting a wide block at 8k labels, while real H_L beats a matched noise
  block by ≈ +1 pp (C − Cs).
- Had linear and MLP passed V5, the frozen rule would have returned CROSS-FAMILY ROBUST (+) for B − C and H − C. They did not; this
  counterfactual is not a result.

# 9. What changed / did not change relative to G1

Did not change: G1's historical Outcome C; Stage 0 remains closed; no claim about information content, depth mechanism, deployment or
source-trained transfer. Changed (descriptively only): the G1 B − C gap no longer looks like an artefact of the linear readout alone —
the same sign and material size appear in four other families — but the preregistered validity rule prevents a CROSS-FAMILY ROBUST
claim. The raw layer3.22 representation was the most accessible evidence tested, but "more accessible" ≠ "more informative".

# 10. Limitations

- **Failure across Decoder Panel v1 is not evidence of information-theoretic absence**; neither is any NULL here. Accessibility is
  specific to the target-pooled supervision, 8k-label budget, compute and HPO budgets tested.
- The shuffle-control threshold (+0.2 pp, Stage-0 convention) is a fallible fault detector: small positive shuffle gains can arise from
  regularization/selection noise rather than leakage; the rule was applied as frozen regardless.
- kNN was valid but effectively non-informative for most arms (anchor-only selected 97/110); it contributes little evidence either way.
  This limitation motivates a prospective informativeness requirement for N1a-DP (not a retrospective change to G1-DP).
- rff often preferred the smoothest γ and poly2 the extreme sketch dimensions; the frozen grids were not widened.
- Two checkpoints of one architecture; exposed development cells; intervals conditional on fitted CV predictions (no training-seed or
  HPO-seed variance).
- Amendment 1 changed only the extraction gate's tie handling, before any label or outcome existed.
