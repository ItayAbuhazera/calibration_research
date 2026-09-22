# Slurm job-decision table — representation-atlas program (2026-09-21)

Queue snapshot re-queried with `squeue` / `sacct` / `scontrol show job` before any action.

| job / tasks | purpose (from `Command`, `WorkDir`, `StdOut`) | valid outputs | remaining necessity | action | reason |
|---|---|---|---|---|---|
| `21470116` `pycharm_srv` | the researcher's PyCharm server (`/home/itayab/pycharm-server.sbatch`) | n/a | outside this program | **retain, untouched** | not an experiment job; explicitly outside the cancellation authorization |
| `21533078_0–5` | corrected-v2 benchmark evaluation-only cells (`scripts/corrected_v2_evaluate_corruption.sbatch`, seed 2, 6 of 12 dev cells) | `results/studyAB/phase0_corrected_v2/evaluation/checkpoint_seed2/<cell>/` | complete | none (COMPLETED) | outputs used for baseline reconciliation |
| `21533078_6–11` | same array, seed 2 remaining cells + | in progress | needed for reconciliation of the residual/atlas baselines with the benchmark | **retain** | corrected-baseline reconciliation is outstanding work; not obsolete or duplicate |
| `21533078_12–23` | same array, seed 4 (12 cells) | not yet produced | needed for seed-4 reconciliation | **retain; array throttle lowered `%4 → %3` (`scontrol update ArrayTaskThrottle=3`)** | keeps the array alive but leaves quota headroom for the atlas jobs; no task cancelled; reversible |
| `21535488–21535490` (mine) | first GPU engineering smoke tests of atlas stages | none | superseded | **cancelled (3 tasks by exact ID)** | they were stuck behind the GRES quota; replaced by CPU-partition smoke jobs `21535538/21535539` (env `ATLAS_ALLOW_CPU_SMOKE=1`, engineering only); no outputs lost |
| `21535538`, `21535539` (mine) | CPU smoke of `stage_u0` and `stage_a` (300/100 queries, 8 000-image bank) | `results/atlas/_smoke/` (engineering; not scientific) | done | completed | verified code path and output schema |

`QOSMaxGRESPerUser` is a per-user GRES quota, not a fault: at least six 4090 slots were observed simultaneously in use by the legacy array; the exact limit is not inferred. No other account/partition/oversubscription is used. GPU work is submitted as arrays with a concurrency limit; downstream consumers use `afterok`.

Atlas submission (all from the immutable snapshot recorded in `results/atlas/ledger.json`): see the ledger for exact job IDs, dependencies and command lines.
