# Decoder Panel v1 — durable master report / execution journal (2026-09-28)

This file is the persistent source of truth for the Decoder Panel v1 program (panel spec, G1-DP, N1a-DP, conditional N1b design).
Together with SLURM accounting (`sacct -j <ids>`) and the result manifests listed below, a new session must be able to reconstruct
the full state. Watchers are convenience only.

## 0. Status header (updated at every stage)

| Field | Value |
|---|---|
| Current stage | D — G1-DP running under Amendment 1 (snapshot g1dp_a1_5f363f77476c; jobs 21726203–21726210) |
| Git HEAD at start | `ec4bf50` (N1a results) |
| Push status | 4 N1a commits (3b2ab7b, 3d5ac7b, 5cefe65, ec4bf50) **pushed** to origin/main (fast-forward dfd137b..ec4bf50) |
| Frozen specs | decoder_panel_v1_spec.md (c27a1df4…, 0dc2041); g1_dp_spec.md (0cbfb62e…, 096ed35); g1_dp_spec_amendment_1.md (6e3930cb…, d399af3) |
| Implementation commit | b58db89 |
| Snapshot | snapshots/g1dp_a1_5f363f77476c (supersedes g1dp_531dd80dcfa0) |
| Jobs running | G1-DP 21726203–21726210 + engineering audits |
| Next action | see HANDOFF section at end |

## 1. Git verification log (Stage 0)

- 2026-09-28: `git status` clean; `git fetch origin`; origin/main = dfd137b; HEAD = ec4bf50; `origin/main..HEAD` = exactly the
  four expected N1a commits; `HEAD..origin/main` empty (no divergence); diff touches only N1a files plus the small submit-script
  log-directory fix in `atlas/g1_submit.py` (part of 5cefe65, expected). Pushed: `dfd137b..ec4bf50 main -> main`.

## 2. Discrepancies between the task prompt and the repository (§61 log)

| # | Assumed by prompt | Repository shows | Class |
|---|---|---|---|
| 1 | G1 fits a separate target-supervised readout per corruption family x severity cell | `atlas/g1_fit.py` + spec §3: ONE fit per (checkpoint, regime, arm, outer fold) pooling all 12 cells (T-8k x 1: every outer-train image appears once at a preassigned cell, `cell_assignment_local`); evaluation on all 13 conditions; 12-cell macro accuracy | Engineering-level for G1-DP (the prompt explicitly says: preserve the original unit). G1-DP HPO unit = checkpoint x outer fold x evidence arm x decoder family (pooled over cells). Label: TARGET-POOLED HPO — ACCESSIBILITY DIAGNOSTIC ONLY. |

## 3. Journal

- Stage A started 2026-09-28.

### Stage A findings (2026-09-28)

- Cluster (live `sinfo`, 2026-09-28): partitions `cpu` (77 nodes; cpu128 nodes with ~100+ idle cores each; several cpu256 nodes in
  maint), `rtx4090` (31 nodes, 3 GPUs each; cs-4090-04/05 invalid, cs-4090-03/07 maint), 7-day `cpu` limit. Account `cliron`,
  QoS `normal`, no per-user TRES limit shown by `sacctmgr`. One unrelated interactive user job (`pycharm_`, 21725577) running.
- **No `$SLURM_TMPDIR` on this cluster.** Node-local `/tmp` (~190 GB free, /dev/sda4) is used as scratch: every task uses
  `/tmp/dp_<jobid>_<taskid>` and removes it on exit; only results/logs are written to the persistent tree.
- Dependencies in `/home/itayab/.conda/envs/geo_cuda12`: scikit-learn 1.6.1, lightgbm 4.6.0, optuna 4.2.1, torch 2.3.1+cu121,
  numpy 1.26.4, scipy 1.15.2 — all six families implementable without new installs.
- G1 fitting granularity: see discrepancy #1 (pooled over the 12 cells; one fit per checkpoint x regime x arm x fold).
- Raw H_3.22: P_3.22 = `raw__probe_logits[:, 8, :]` = saved layer-pilot probe (`results/layer_pilot/checkpoint_seed{s}/probe_weights.pt`,
  `((x - mean)/std) @ W + b`, pre-temperature) applied to x = L2-normalized global-average-pooled `layer3.22` output (1024-d), strict FP32,
  corrected_v2 protocol (`Experiments/layer_selection_pilot.py` FeatureTap). Raw features were NOT cached, but are **exactly reproducible**
  with an exact label-free consistency gate (reconstruct cached P_3.22 from re-extracted features with the saved probe). Raw tier planned.
- G1 anchored-linear fit cost (from G1 fits): A/B ~1 min; C/D/E/F ~25–40 min per (arm, fold) on 4 cores (lambda path + refit).

### Stage B — engineering benchmarks (synthetic teacher targets, NO scientific labels)

Code: `atlas/decoder_panel.py` (panel implementation, draft), `atlas/dp_engineering.py`, `atlas/dp_submit.py` (uncommitted draft at
submission time; engineering only, run from the live tree, not a snapshot). Units: G1-DP seed 2 / fold 0 rows (arms A 100-d,
C 2148-d, I-standin 3172-d), N1a-DP base 2 / fog held out / fold 0 (72,000 x 206 training rows, 3 inner env splits).
Intended command: `python -m atlas.dp_submit engineering` (16 jobs, `cpu` 8 cores / 32G except one rtx4090 MLP job).
Outputs: `results/decoder_panel/engineering/*.json`; ledger `results/decoder_panel/ledger.json`; logs `results/decoder_panel/logs/`.
Submitted 2026-09-28 (no dependencies): eng_g1_linear 21725636, eng_n1a_linear 21725637, eng_g1_poly2 21725638, eng_n1a_poly2 21725639,
eng_g1_knn 21725640, eng_n1a_knn 21725641, eng_g1_rff 21725642, eng_n1a_rff 21725643, eng_g1_lgbm_C50 21725644, eng_g1_lgbm_AI 21725645,
eng_g1_mlp_cpu 21725646, eng_g1_mlp_gpu 21725647 (rtx4090), eng_n1a_lgbm 21725648, eng_n1a_mlp 21725649.
Recover state: `sacct -j 21725636,21725637,21725638,21725639,21725640,21725641,21725642,21725643,21725644,21725645,21725646,21725647,21725648,21725649`.

Cancelled 2026-09-28 (never started; pending on Priority; 1–3-day limits blocked backfill): all engineering jobs except 21725647 (GPU MLP,
COMPLETED). Resubmitted with shorter limits (6–16 h) and per-trial progress logging (engineering change only):
  eng_g1_linear 21725689
  eng_n1a_linear 21725690
  eng_g1_poly2 21725691
  eng_n1a_poly2 21725692
  eng_g1_knn 21725693
  eng_n1a_knn 21725694
  eng_g1_rff 21725695
  eng_n1a_rff 21725696
  eng_g1_lgbm_C50 21725697
  eng_g1_lgbm_AI 21725698
  eng_g1_mlp_cpu 21725699
  eng_n1a_lgbm 21725700
  eng_n1a_mlp 21725701

**Engineering finding (2026-09-28, synthetic targets):** anchored LightGBM (init_score = z) produced exploding first-round leaves
(max |leaf| ≈ 8,645; inner NLL 229 vs 7.1 for the anchor alone) because saturated softmax anchors give near-zero hessians and Newton
leaf values blow up. Fix frozen into the family definition before any scientific run: `max_delta_step = 2.0` (with the cap: best NLL 2.21;
cap 1.0: 2.32). Cancelled the uncapped LightGBM audits 21725697, 21725698, 21725700; resubmitted with the cap:
eng_g1_lgbm_C50_mds2 21725772, eng_g1_lgbm_AI_mds2 21725773, eng_n1a_lgbm_mds2 21725774.
Also found/fixed (engineering): `stage0_fit` sets torch's default dtype to float64 at import → MLP now pinned to float32;
`fit_decoder` default `n_trials` was bound at definition time → now read at call time.
Replacement kNN audits (NumPy kNN): eng_g1_knn_np 21725842, eng_n1a_knn_np 21725843 (COMPLETED). Test runs: tests_without_dp 21725891,
tests_with_dp 21725892 (COMPLETED).

### Stage B result (2026-09-28)

- Budget: **N_TRIALS = 50** (rule: raise if any audited study improved > 0.25 % from 20 → 30; G1-scale MLP improved 0.37 % CPU / 1.47 % GPU).
- MLP on **CPU** (GPU 2–7× faster per study but ≤ 19 min on CPU with far more free slots).
- Engineering defects fixed before freezing: LightGBM leaf explosion under saturated anchors (`max_delta_step = 2.0`), float64 global
  default dtype, `n_trials` default binding, sklearn kNN out-of-range indices (replaced by exact NumPy kNN).
- Resource estimate: G1-DP ≈ 5,600 core-h + < 0.5 GPU-h, elapsed ≈ 12–20 h; N1a-DP ≈ 900 core-h. Judged acceptable (not unexpectedly
  large for 660 + 240 nested HPO studies); nothing scientific dropped. `docs/decoder_panel_v1_resource_plan.md`.
- **Frozen panel spec:** `docs/decoder_panel_v1_spec.md`, sha256 `c27a1df454889bb706e9ded852275683f7be2d3316a5d05603ca0b3246c5d50d`, commit `0dc2041`.
  Resource plan commit `7c4b463`.

### Tests (requirement 20)

- `tests/test_decoder_panel.py`: 32 passed (items 1–19 + rules + end-to-end task path).
- Full suite: 526 passed, 26 failed, 5 errors WITH the new module; **identical 26 failed / 5 errors WITHOUT it** (495 passed), and all 31
  of those pass when run in isolation → **pre-existing, order-dependent failures unrelated to this work** (test_glad_pi_shapes,
  test_integration_smoke, test_kcal ["mixed dtype" — consistent with `stage0_fit`'s global float64 default leaking across test modules],
  test_layer_selection_pilot, test_pts_argmax_invariance, test_recoverability_export, test_reliability_experiment,
  test_scalar_confidence_semantics, test_studyAB_benchmark_validation errors). Not fixed (unrelated research code).
  Logs: `results/decoder_panel/engineering/full_suite_with_dp.txt`, `full_suite_without_dp.txt`, `preexisting_failures_isolated.txt`.

## Stage D — G1-DP submission (2026-09-28)

§63 constraint re-check before launch: N1a commits pushed (ec4bf50) ✓; G1 remains closed (Outcome C untouched) ✓; N1a remains
INCONCLUSIVE ✓; master report exists ✓; exactly six families ✓; Optuna only for lgbm/mlp, grids elsewhere ✓; identical HPO opportunity
across arms (tested) ✓; G1 fitting unit preserved (pooled; discrepancy #1) ✓; N1a-DP not yet frozen/run ✓; no N1b ✓; parallelism across
units, Optuna n_jobs = 1 ✓; node-local /tmp staging, results persistent ✓; job IDs recorded below ✓.

- Frozen G1-DP spec: `docs/g1_dp_spec.md`, sha256 `0cbfb62e70125a8bff22f929ab62466b41de5a4d5d9ed8af6454cf859e4da825`, commit `096ed35`.
- Implementation commit `b58db89`; **snapshot `snapshots/g1dp_531dd80dcfa0`** (SNAPSHOT.json git_head 096ed35, clean tracked diff).
- Command (run from repo root): `python -m atlas.dp_submit g1dp snapshots/g1dp_531dd80dcfa0`
  → extract_s2, extract_s4 (rtx4090) → bundle array 0-9 (afterok) → fit_<family> arrays (afterok bundle; mapping manifests
  `results/g1dp/array_manifest_<family>.json`; linear/poly2/rff/lgbm 110 tasks %20, mlp/knn 10 tasks %10) → aggregate (afterok all fits).
- Outputs: `results/g1dp/h322/`, `results/g1dp/bundles/`, `results/g1dp/fits/<family>/seed<s>/fold<k>/<arm>.{npz,study.json,done.json}`,
  `results/g1dp/report/`; logs `results/g1dp/logs/`; ledger `results/g1dp/ledger.json`.

Submitted 2026-09-28T22:50: extract_s2 21726070, extract_s4 21726071 (no deps); bundle 21726072 (afterok 21726070:21726071);
fit_linear 21726073, fit_poly2 21726074, fit_rff 21726075, fit_lgbm 21726076, fit_mlp 21726077, fit_knn 21726078 (each afterok 21726072);
aggregate 21726079 (afterok all six fit arrays).
Recover: `sacct -j 21726070,21726071,21726072,21726073,21726074,21726075,21726076,21726077,21726078,21726079 -o JobID,JobName%25,State,Elapsed`.

## HANDOFF / exact next actions (for a new session)

1. Monitor with sacct (IDs above). If extraction fails its gate (exit 2): the raw tier is dropped per spec §2 — but then the bundle
   stage (afterok) will not start; engineering decision needed: rebuild bundles without h3 is NOT implemented → STOP and report.
2. Failed fit indices (time/memory/node): resubmit only those indices from the SAME snapshot (`sbatch --array=<idx>` of the same
   command via `atlas.dp_submit.sb`); completed units are skipped automatically (checksum-validated `.done.json`). Then run
   `python -m atlas.g1dp_aggregate` from the snapshot (it refuses incomplete arrays).
3. After aggregation: write `docs/g1_decoder_panel_audit_2026-09-28.md` (structure: prompt §45), update this report, experiment card.
4. Only then: fill `N1ADP_RES` in `atlas/dp_submit.py` (resource plan §5), freeze `docs/n1a_dp_spec.md` (draft present, not frozen),
   commit + sidecar, snapshot `n1adp`, submit `python -m atlas.dp_submit n1adp <snapshot>`.
5. Still running (engineering only, descriptive): eng_g1_linear 21725689, eng_g1_poly2 21725691, eng_n1a_poly2 21725692,
   eng_g1_rff 21725695, eng_g1_lgbm_C50_mds2 21725772, eng_g1_lgbm_AI_mds2 21725773 — append their timings/curves here when done.

## STOP — raw-H_3.22 extraction gate (2026-09-28 ~22:55) — AWAITING RESEARCHER DECISION

- extract_s2 21726070 COMPLETED: gate passed (max |P_rec − P_cache| 0.0078 = float16 half-ulp; argmax agreement ≥ 0.9991 all cells).
- extract_s4 21726071 FAILED (exit 2): max |ΔP| ≤ 0.0152 everywhere (limit 0.05) and max |Δz| = 0, but argmax agreement
  0.9988 (gaussian_noise_s5) and 0.9989 (defocus_blur_s5) < frozen 0.999. Consequently bundle 21726072, all six fit arrays
  21726073–21726078 and aggregate 21726079 were CANCELLED by Slurm (afterok never satisfied). **No G1-DP fit or outcome exists.**
- Diagnosis (label-free): all 12 + 11 disagreeing rows are EXACT ties in the float16 cache (cached top-1 − top-2 gap = 0.0; reconstructed
  gap ≤ 0.0064 < float16 spacing). The re-extracted H_3.22 reproduces P_3.22 to cache precision; the argmax sub-criterion was
  miscalibrated for float16 ties. Extracted arrays for both seeds are on disk: `results/g1dp/h322/seed{2,4}/` (+ seed-2 consistency.json).
- Options: (A) follow the frozen spec literally: drop the raw tier (arms H, I, Hs) and state that raw layer3-vs-layer4 accessibility is
  not tested — requires an engineering change so bundles/aggregation run without h3; (B) formally documented pre-outcome amendment of the
  gate: compute argmax agreement only over rows whose cached top-1/top-2 are not tied at float16 precision (keep all other thresholds),
  re-run extraction from a new snapshot, proceed with the full frozen arm set. Recommendation: (B) — it is label-free, pre-outcome,
  and the defect is in the check, not the representation.
- Engineering audits still running (descriptive only): 21725689, 21725691, 21725692, 21725695, 21725772, 21725773.

### Late engineering audit results (descriptive; cannot change frozen values) — 2026-09-28 ~23:10

Measured on G1 arm C (2,148-d; 8 cores): linear 38 min, poly2 48 min (below the ~2 h extrapolation in the resource plan), rff 27 min.
Arm I and the N1a poly2 audit were still running (jobs 21725689/91/92/95, 21725772/73). The 16 h poly2 limit and 8 h linear/rff limits
remain conservative. The G1-DP arrays were not started (see STOP above); nothing is scheduled to run until the gate decision.

## Researcher decision (2026-09-28): option B with a tie-aware rule — PRE-AMENDMENT RECORD

Decision: amend the raw-H_3.22 gate's argmax sub-criterion to a tie-aware top-class rule (not a denominator exclusion): with
T(x) = {c : P_cache_c = max_j P_cache_j} on the stored float16 probe outputs, a row is top-class-consistent iff argmax(P_rec) ∈ T(x).
Option A only if any disagreement remained unexplained.

Pre-amendment verification (label-free; `results/g1dp/h322/tie_aware_audit.json`; reads only re-extracted H_3.22, the saved probe and
cached float16 probe logits):
1. The 23 disagreements in the two failing seed-4 cells (gaussian_noise_s5: 12, defocus_blur_s5: 11) all occur on exact stored-float16
   top ties. Over ALL 13 conditions: seed 2 has 82 plain-argmax disagreements, seed 4 has 75 — every one on a tied row.
2. Disagreements on non-tied rows: **0** (both seeds, all conditions). Tie-aware agreement = 1.000 in every cell.
3. Original numeric thresholds still pass: max |P_rec − P_cache| = 0.0078 (seed 2) / 0.0152 (seed 4) ≤ 0.05; max |Δz| = 0 ≤ 1e-2.
4. No labels and no G1-DP outcome have been inspected: no bundle was built and no fit ran (jobs 21726072–79 were cancelled
   by Slurm before starting); the extraction and audit read no labels.
All four hold → proceeding with a documented pre-outcome amendment.
Original frozen G1-DP spec: `docs/g1_dp_spec.md`, sha256 `0cbfb62e70125a8bff22f929ab62466b41de5a4d5d9ed8af6454cf859e4da825` (unchanged).
Amendment 1 frozen: `docs/g1_dp_spec_amendment_1.md`, sha256 `6e3930cb04f78a0e0f1d821c2ac972d428fbf46f13bfe641af4f6852b70b7ded` (original spec hash unchanged).

## Stage C/D resubmission under Amendment 1 (2026-09-28)

- Amendment commit `d399af3`; new snapshot **`snapshots/g1dp_a1_5f363f77476c`** (git_head d399af3, clean tracked diff). Previous snapshot `snapshots/g1dp_531dd80dcfa0`
  is superseded (its extraction gate is the unamended rule); its seed-2/seed-4 extraction jobs (21726070/71) are superseded.
- Step 1 (validation only): extract_s2, extract_s4 from the new snapshot (re-extracts H_3.22 and re-evaluates the amended gate).
- Step 2 (only if BOTH pass): downstream chain UNCHANGED in design/resources: `atlas.dp_submit` g1dp chain minus the extraction step
  (bundle array 0-9 → six family arrays → aggregate), all from `snapshots/g1dp_a1_5f363f77476c`.
  Submitted: extract_a1_s2 21726199, extract_a1_s4 21726200 (rtx4090, no deps).
- Step 1 result: extract_a1_s2 21726199 COMPLETED, extract_a1_s4 21726200 COMPLETED — **amended gate PASSED in both checkpoints**
  (tie-aware agreement 1.000 all cells; plain argmax min 0.999 / 0.9988 reported; max |ΔP| 0.0078 / 0.0152 ≤ 0.05; max |Δz| 0).
  `results/g1dp/h322/seed{2,4}/consistency.json` (field `amendment: g1_dp_spec_amendment_1`). Raw tier valid.
- Step 2 submitted 2026-09-28 (from `snapshots/g1dp_a1_5f363f77476c`): bundle 21726203 (array 0-9, no deps — extraction already
  passed); fit_linear 21726204, fit_poly2 21726205, fit_rff 21726206, fit_lgbm 21726207 (110 tasks each, %20), fit_mlp 21726208,
  fit_knn 21726209 (10 tasks each, %10) — each afterok 21726203; aggregate 21726210 (afterok all six).
  Recover: `sacct -j 21726203,21726204,21726205,21726206,21726207,21726208,21726209,21726210 -o JobID,JobName%25,State,Elapsed`.
  The HANDOFF steps above apply with these IDs and this snapshot (resubmit failed indices from `snapshots/g1dp_a1_5f363f77476c`).
- Bundles 21726203_[0-9] all COMPLETED (0.75–38 min; slow tasks on old/busy nodes); 27 blocks each, ~1 GB, manifests with sha256. Fit arrays released.
- Monitoring 2026-09-29 ~00:05: fit_linear 21726204 — 5 units COMPLETED (seed 2 / fold 0: A, B, H, Hs, Cs), 8 RUNNING; poly2/rff/lgbm/mlp/knn
  arrays pending on Priority. No Traceback/Error/Killed in any `results/g1dp/logs/fit_*`. Engineering check of the finished units
  (study JSON only): all refits converged, all selected λ interior (A/B/H 1e-2, Hs 1e-1, Cs 1e0; no grid edge), HPO 2–5 min per arm.
  Disclosure: that check displayed the inner-validation NLL of the selected and anchor-only candidates for this one fold; no
  test-set accuracy or contrast was computed, and nothing is decided from partial results (spec §5: aggregation only after all arrays).
- Monitoring 2026-09-29 ~01:15: arm-units complete — linear 93/110, poly2 22/110, rff 26/110, mlp 110/110 (10/10 tasks), knn 110/110 (10/10 tasks), lgbm 0/110 (array 21726207 still pending on Priority); aggregate 21726210 pending. No Traceback/Error/OOM/TIMEOUT in any fit log; no resubmission needed.
- Monitoring 2026-09-29 ~02:20: linear 110/110 COMPLETED; rff 77/110; poly2 56/110; mlp/knn complete; lgbm 21726207 still PENDING (Priority) — if still pending at the next check, consider an engineering-only resubmission of the same array with a shorter limit (e.g. 12 h; per-arm units are restart-safe). No errors; no resubmission.
- Monitoring 2026-09-29 ~03:20: rff 110/110 COMPLETED; poly2 88/110 (20 running); linear/mlp/knn complete; lgbm 21726207 still fully
  PENDING (Priority) after ~3.5 h. `sbatch --test-only` (cpu, 8 cores, 24G) start estimates: 24 h limit → 2026-10-01 05:00;
  12 h or 8 h limit → 2026-09-30 23:15. A shorter limit gains only ~6 h (estimates are coarse) while risking timeouts on the
  3,172-d arms; the bottleneck is fair-share priority on a busy cluster. **Decision: keep 21726207 as submitted (no churn).**
  Expect G1-DP completion ~1–2 days out, dominated by lgbm. No errors; no resubmission.
- Monitoring 2026-09-29 ~04:20: poly2 108/110 (2 running); linear/rff/mlp/knn complete; lgbm 21726207 still fully PENDING (Priority); aggregate pending. No errors; no resubmission.
