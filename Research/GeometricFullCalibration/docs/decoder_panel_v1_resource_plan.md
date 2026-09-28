# Decoder Panel v1 — HPO budget audit and resource plan (engineering; frozen with the panel spec)

All numbers come from Stage-B engineering jobs on **synthetic teacher targets** (a fixed random network applied to label-free inputs
z / H_L / F_Z; `atlas/dp_engineering.py`). No scientific label or outcome was read. Artifacts: `results/decoder_panel/engineering/*.json`,
logs `results/decoder_panel/logs/eng_*`, ledger `results/decoder_panel/ledger.json`. Representative units: **G1-DP** seed 2 / fold 0
(8,000 training rows, 6,000 inner-fit / 2,000 inner-val, K = 100 anchored; arms A 100-d, C 2,148-d, I-stand-in 3,172-d; prediction on
26,000 rows = 2,000 images × 13 conditions); **N1a-DP** base 2 / fog held out / fold 0 (72,000 × 206 training rows, three inner
leave-one-training-family-out splits of 36,000 / 6,000 rows, K = 3). CPU jobs: `cpu` partition, 8 cores, mixed Xeon generations
(E5-2660 v4 … Gold 6140), which dominates run-to-run timing variance.

## 1. Optuna budget convergence (best inner objective after n trials; lower is better)

| study | 5 | 10 | 20 | 30 | 40 | 50 |
|---|---|---|---|---|---|---|
| G1 MLP (CPU) arm A | 1.4821 | 1.4821 | 1.4777 | 1.4777 | 1.4641 | 1.4641 |
| G1 MLP (CPU) arm C | 1.5558 | 1.5046 | 1.4946 | 1.4920 | 1.4802 | 1.4802 |
| G1 MLP (CPU) arm I | 1.5523 | 1.5052 | 1.4962 | 1.4907 | 1.4855 | 1.4855 |
| G1 MLP (GPU) arm I | 1.5575 | 1.5036 | 1.5036 | 1.4815 | 1.4815 | 1.4815 |
| G1 LightGBM arm A (30-trial study) | 1.7557 | 1.7229 | 1.6385 | 1.6368 | — | — |
| N1a LightGBM (−utility) | −0.0314 | −0.0314 | −0.0314 | −0.0314 | −0.0318 | −0.0318 |
| N1a MLP (−utility) | −0.0375 | −0.0387 | −0.0389 | −0.0389 | −0.0390 | −0.0390 |

Frozen audit rule: *raise the global budget from 30 to 50 if any audited study still improves by > 0.25 % (relative) from 20 → 30 trials.*
G1-scale MLP arm I improved 0.37 % (CPU) / 1.47 % (GPU) from 20 → 30 and 0.3–0.9 % from 30 → 50; LightGBM arm A improved 5.0 % from
10 → 20 and 0.10 % from 20 → 30. **Decision: N_TRIALS = 50 for both Optuna families, all experiments** (one panel-wide constant).
N1a-scale studies are flat after 10–40 trials. (The 50-trial G1 LightGBM arm-C audit, job 21725772, was still running at freeze time;
its curve is appended below when complete as descriptive evidence only — it cannot change the frozen budget.)

## 2. Measured cost per HPO study (selection + refit + prediction), 8 cores

| family | G1-DP arm A (100-d) | G1-DP arm C (2,148-d) | G1-DP arm I (3,172-d) | N1a-DP unit | peak RSS |
|---|---|---|---|---|---|
| linear (9-λ path) | 4.7 min | 38 min | ~60 min (extrap.) | 7 min | ≤ 3 GB |
| poly2 (3 sketch dims × 9 λ) | 22 min | ~2 h (extrap.) | ~3–5 h (extrap.) | > 1 h (running) | ≤ 6 GB |
| lgbm (50 trials) | ~0.9 h (64 s/trial) | ~3 h (213 s/trial) | ~3–6 h | 17 min (12 s/trial) | ≤ 4 GB |
| mlp (50 trials, CPU) | 3.3 min | 5.8 min | 7.2 min | 18.5 min | ≤ 3 GB |
| mlp (50 trials, GPU rtx4090) | 1.5 min | 0.9 min | 0.9 min | — | — |
| knn (50 grid cells) | 0.7 min | 1.8 min | 1.9 min | 3.8 min | ≤ 3 GB |
| rff (5 γ × 9 λ) | 37 min | ~40 min (head is 1,024-d for every arm) | ~40 min | 25 min | ≤ 3 GB |

Local-scratch staging: copying the 12 H_L cell arrays (≈ 1 GB) to node `/tmp` took ≈ 5–10 s; a G1-DP (seed, fold) bundle is ≈ 1 GB.
JL audit (D5, G1 arms C/I): target dim 418 for 6,000 rows; distortion median 0.99–1.00, 5th pct 0.94, 95th pct 1.05–1.06.

**MLP CPU vs GPU:** GPU is 2–7× faster per study, but a CPU study costs ≤ 7 min (G1) / 19 min (N1a), the `cpu` partition has far
more free slots than `rtx4090`, and CPU avoids GPU queueing and host↔device transfers. **Frozen: MLP runs on CPU** (all experiments).

## 3. Engineering defects found and fixed before freezing (no scientific data involved)

1. Anchored LightGBM exploded (first-round |leaf| ≈ 8,645; NLL 229 vs 7.1 for the anchor alone): Newton steps under saturated softmax
   anchors. Fix: fixed `max_delta_step = 2.0` in the family definition (NLL 2.21 with the cap).
2. `atlas.stage0_fit` sets torch's global default dtype to float64 at import; MLP pinned to float32.
3. `fit_decoder`'s default `n_trials` was bound at definition time; now read at call time.
4. sklearn `NearestNeighbors` (threaded brute force) returned out-of-range neighbour indices when OpenMP runtimes were mixed in one
   process (seen in the full test run); replaced by an exact blockwise NumPy kNN (index-identical to sklearn on clean inputs) with range
   assertions.

## 4. G1-DP execution plan and estimate

Units: 2 checkpoints × 5 folds × 11 arms × 6 families = 660 HPO studies. Array granularity: **one task per (seed, fold, arm)** for linear,
poly2, rff, lgbm (110 tasks each; long single studies) and **one task per (seed, fold) with all 11 arms sequential** for mlp and knn
(10 tasks each). Mapping: `atlas.g1dp.units(split_arms)` (lexicographic seed → fold → arm), written to
`results/g1dp/array_manifest_<family>.json` before submission.

| family | tasks | cpus / mem / limit | concurrency | est. core-h |
|---|---|---|---|---|
| linear | 110 | 8 / 24G / 8 h | %20 | ≈ 400 |
| poly2 | 110 | 8 / 32G / 16 h | %20 | ≈ 1,800 |
| rff | 110 | 8 / 24G / 8 h | %20 | ≈ 600 |
| lgbm | 110 | 8 / 24G / 24 h | %20 | ≈ 2,700 |
| mlp | 10 | 8 / 24G / 6 h | %10 | ≈ 80 |
| knn | 10 | 8 / 32G / 2 h | %10 | ≈ 20 |
| extraction | 2 | rtx4090 ×1, 4 cpus / 48G / 1.5 h | — | < 0.5 GPU-h |
| bundles | 10 | 2 / 32G / 2 h | — | ≈ 5 |

Total ≈ **5,600 CPU core-hours + < 0.5 GPU-hours**; expected elapsed ≈ 12–20 h at the chosen concurrency (≤ 20 × 8 = 160 cores per heavy
family, ≤ ~660 cores in total at peak). Concurrency rationale: live `sinfo` showed ~1,500 idle `cpu` cores but ~1,000 pending jobs of other
users; 20-way arrays keep this program at well under half the idle capacity. Time limits are kept ≤ 24 h because multi-day limits blocked
backfill (observed 2026-09-28). This is large relative to G1 (≈ 70 core-h) but is the expected consequence of 660 nested HPO studies over
six families; it was judged acceptable, not unexpectedly large, and nothing scientific was dropped to reduce it.

## 5. N1a-DP plan (finalized in `docs/n1a_dp_spec.md` before its freeze)

40 units × 6 families = 240 tasks (one task per unit; family arrays %20). Estimated ≈ 40 × (0.12 + ≥1.5 + 0.4 + 0.3 + 0.3 + 0.06) h × 8 ≈
900 core-hours (poly2 dominant; its N1a timing is appended when the audit finishes).

## 6. Late audit results

Recorded in the master report (descriptive only; they cannot change frozen values).
