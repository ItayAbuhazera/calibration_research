# Regime-map follow-up — results record, Stages 1–2 (ORACLE DIAGNOSTIC; append-only; not interpretive)

Design: `docs/regime_map_followup_spec_v1.md` (sha256 `9214ab74…a6986`, unedited) as amended by `docs/regime_map_followup_amendment_1.md` (sha256 `31ee68be…56d2f`, unedited; its §9 is not used, so both hashes still verify).
All quantities are anchored, target-fitted (target labels: ORACLE DIAGNOSTIC), T-8k×1, 12-cell macro, pp, 95% image-bootstrap intervals with one shared resample array per state; intervals contain no training-seed variance.
Artifacts: `results/regime_map_followup/report/{stage1_anchored,stage2_anchored}.json` and `*_table.md`; fits `results/regime_map_followup/stage{1,2}/`; extraction `results/regime_map_followup/hL/`.
Provenance: Stage 1 snapshot `followup_s1_ea2dfacff7fe` (git `7ab8afc`), jobs 21698229–21698236; Stage 2 snapshot `followup_s2_be595d5404ce` (git `5d3df94`), jobs 21698326–21698340. Nothing was run from the live tree.

## Validation
* Stage 1: 35/35 tasks completed; 0 unconverged, 0 retried; Gate 1 passed (D = +0.79 / +0.84 pp in b10_s1 / b10_s2; C1b fraction −0.469 / −0.463).
* Stage 2 extraction: consistency checks passed in all 7 states (state (a) refit head: max |Δz| 0, argmax agreement 1.0; fine-tuned states: max |z_stored − (W h_L + b)| ≤ 1.8e-05, threshold 1e-2); `consistency.json` per state.
* Stage 2 fits: 35/35 completed; 0 unconverged, 0 retried in every arm and state; the anchored Z-only re-fit reproduced Stage 1 to 0 differing predictions in six states and 1 in b1_s2.
* Measured cost: extraction 0.30 GPU-h in total (mean 153 s per state); Stage 2 fits mean 86 min per (state, fold) task, ≈ 200 four-core hours in total (the extrapolated estimate was ≈ 160 CPU-hours).

## Stage 1 table (from `stage1_anchored_table.md`)
| state | anchored Z-only − base (macro / clean) | anchored (Z,P) − base (macro / clean) | D = Δ_T(Z,P) | C1b − Z-only (Δ_T) | C1b fraction of D |
|---|---|---|---|---|---|
| a | +2.96 [+2.69, +3.23] / -2.08 [-2.61, -1.58] | +5.04 [+4.75, +5.32] / -0.98 [-1.53, -0.47] | +2.08 [+1.93, +2.22] | +0.27 [+0.16, +0.37] | 0.129 [0.076, 0.177] |
| b1_s1 | +13.51 [+13.02, +13.97] / +7.81 [+6.97, +8.67] | +15.97 [+15.46, +16.47] / +8.52 [+7.70, +9.36] | +2.46 [+2.22, +2.69] | +0.10 [-0.07, +0.26] | 0.039 [-0.028, 0.103] |
| b3_s1 | +9.77 [+9.41, +10.15] / +1.10 [+0.46, +1.74] | +11.27 [+10.87, +11.67] / +1.29 [+0.64, +1.97] | +1.50 [+1.29, +1.72] | +0.11 [-0.03, +0.26] | 0.075 [-0.018, 0.168] |
| b10_s1 | +5.98 [+5.68, +6.29] / -1.88 [-2.43, -1.32] | +6.77 [+6.45, +7.11] / -2.40 [-2.93, -1.86] | +0.79 [+0.62, +0.96] | -0.37 [-0.51, -0.22] | -0.469 [-0.743, -0.253] |
| b1_s2 | +11.17 [+10.73, +11.61] / +4.87 [+4.11, +5.66] | +13.00 [+12.54, +13.46] / +5.55 [+4.75, +6.31] | +1.83 [+1.60, +2.07] | -0.01 [-0.16, +0.15] | -0.005 [-0.095, 0.079] |
| b3_s2 | +11.62 [+11.22, +12.04] / +3.37 [+2.63, +4.10] | +13.21 [+12.77, +13.64] / +3.40 [+2.70, +4.10] | +1.59 [+1.39, +1.79] | +0.02 [-0.14, +0.17] | 0.011 [-0.090, 0.101] |
| b10_s2 | +6.60 [+6.30, +6.92] / -1.84 [-2.33, -1.33] | +7.44 [+7.12, +7.78] / -2.38 [-2.93, -1.85] | +0.84 [+0.66, +1.02] | -0.39 [-0.53, -0.24] | -0.463 [-0.719, -0.269] |

## Stage 2 table (from `stage2_anchored_table.md`; C1b fraction from Stage 1)
| state | D = Δ_T(Z,P) | C1b frac | C1e (Z,P_L) Δ_T / frac of D | C1c (h_L) Δ_T | C1d (Z,h_L) Δ_T | K Δ_T | Z+K Δ_T | D − Δ_T(C1d) |
|---|---|---|---|---|---|---|---|---|
| a | +2.08 [+1.93, +2.22] | 0.129 | +0.30 [+0.21, +0.40] / 0.146 [0.103, 0.189] | +0.84 [+0.67, +1.00] | +0.89 [+0.72, +1.05] | +1.29 [+1.09, +1.47] | +1.35 [+1.16, +1.54] | +1.19 [+1.02, +1.38] |
| b1_s1 | +2.46 [+2.22, +2.69] | 0.039 | +0.57 [+0.40, +0.73] / 0.233 [0.167, 0.296] | +1.96 [+1.75, +2.17] | +1.95 [+1.74, +2.16] | +2.02 [+1.76, +2.26] | +2.07 [+1.81, +2.31] | +0.51 [+0.27, +0.75] |
| b3_s1 | +1.50 [+1.29, +1.72] | 0.075 | +0.13 [-0.01, +0.28] / 0.088 [-0.005, 0.179] | +1.48 [+1.30, +1.67] | +1.45 [+1.27, +1.64] | +1.69 [+1.46, +1.93] | +1.73 [+1.51, +1.96] | +0.05 [-0.16, +0.26] |
| b10_s1 | +0.79 [+0.62, +0.96] | -0.469 | -0.32 [-0.45, -0.20] / -0.411 [-0.651, -0.223] | +0.69 [+0.50, +0.88] | +0.65 [+0.46, +0.84] | +0.91 [+0.69, +1.14] | +0.97 [+0.76, +1.21] | +0.14 [-0.06, +0.32] |
| b1_s2 | +1.83 [+1.60, +2.07] | -0.005 | -0.01 [-0.17, +0.15] / -0.007 [-0.098, 0.083] | +1.74 [+1.53, +1.95] | +1.74 [+1.53, +1.95] | +1.82 [+1.56, +2.07] | +1.91 [+1.66, +2.16] | +0.09 [-0.14, +0.33] |
| b3_s2 | +1.59 [+1.39, +1.79] | 0.011 | -0.19 [-0.34, -0.02] / -0.118 [-0.228, -0.013] | +1.56 [+1.36, +1.76] | +1.54 [+1.35, +1.73] | +1.59 [+1.37, +1.83] | +1.70 [+1.47, +1.93] | +0.05 [-0.17, +0.25] |
| b10_s2 | +0.84 [+0.66, +1.02] | -0.463 | -0.32 [-0.44, -0.19] / -0.380 [-0.597, -0.205] | +0.85 [+0.67, +1.03] | +0.80 [+0.62, +0.97] | +1.01 [+0.79, +1.22] | +1.08 [+0.86, +1.29] | +0.04 [-0.15, +0.24] |

`Prow` sanity (reported, not gated), macro accuracy minus anchored Z-only and mean absolute probability difference:

| state | pp | mean abs prob diff |
|---|---|---|
| a | +0.561 | 0.00143 |
| b1_s1 | +0.078 | 0.00219 |
| b3_s1 | -0.613 | 0.00195 |
| b10_s1 | -0.453 | 0.00209 |
| b1_s2 | -0.739 | 0.00188 |
| b3_s2 | -0.450 | 0.00199 |
| b10_s2 | -0.320 | 0.00211 |

## Gates and decision (pre-declared, applied to b10_s1 and b10_s2)
* Gate 1: passed. Gate 2: not triggered (C1e explains -0.411 and -0.380 of D; the stop needs ≥ 0.5 in both).
* HEAD-DISCARD: not met (Δ_T(C1d) = 0.65 and 0.80 are below D = 0.79 and 0.84).
* INTERMEDIATE-SPECIFIC: not met (D − Δ_T(C1d) intervals [-0.05671967726878704, 0.32258535040341396] and [-0.1542079249183171, 0.2350574281522971] include 0).
* **Outcome (`decision` in `stage2_anchored.json`): INCONCLUSIVE — stop.**
* Stage 3 (label budget) has not been run; see the report to the researcher.

## Stage 3 — not run (researcher's decision, 2026-09-27)
The INCONCLUSIVE stop of the pre-declared decision table covers Stage 3. Stage 3 (label budget, amendment §5) was meant to size a phenomenon for continuation; with the decision at stop it cannot change anything, so it was not run. No further runs on this line.
