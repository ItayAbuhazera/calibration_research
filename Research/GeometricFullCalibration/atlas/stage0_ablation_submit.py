"""Submit the Stage 0 evidence-ablation DAG (docs/stage0_evidence_ablation_spec.md) from an immutable snapshot.

  python -m atlas.stage0_ablation_submit snapshots/<snapshot_dir>

One array per (evidence, regime): 10 tasks = 2 checkpoints x 5 folds (task id -> seed, fold).
Aggregation is submitted with afterok on every array. CPU only.
"""
import json
import os
import subprocess
import sys
import time

PY = "/home/itayab/.conda/envs/geo_cuda12/bin/python"
LEDGER = "results/stage0_ablation/ledger.json"
FULL = ("L11", "xckpt")
LAYERS_OTHER = ("L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L9", "L10")
FULL_REGIMES = ("T-8k1", "S-8k1", "T-2.5k1", "S-2.5k1")
LAYER_REGIMES = ("T-8k1", "S-8k1")


def ledger_add(e):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    L = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    L.append(e)
    json.dump(L, open(LEDGER, "w"), indent=1)


def sb(snap, stage, cmd, deps=(), mem="16G", hours="02:00:00", cpus=4, array=None):
    name = os.path.basename(snap)
    logdir = os.path.abspath("results/stage0_ablation/logs"); os.makedirs(logdir, exist_ok=True)
    args = ["sbatch", "--parsable", f"--job-name=abl-{stage}", f"--mem={mem}", f"--time={hours}", f"--cpus-per-task={cpus}",
            "--partition=cpu", f"--output={logdir}/{stage}_%A_%a.out" if array else f"--output={logdir}/{stage}_%j.out"]
    if array:
        args.append(f"--array={array}")
    if deps:
        args.append("--dependency=afterok:" + ":".join(str(d) for d in deps))
    args += ["--wrap", f"cd {os.path.abspath(snap)} && ABL_SNAPSHOT={name} PYTHONPATH={os.path.abspath(snap)} PYTHONDONTWRITEBYTECODE=1 "
                       f"OMP_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS={cpus} {cmd}"]
    jid = subprocess.check_output(args, text=True).strip().split(";")[0]
    ledger_add({"job_id": jid, "stage": stage, "cmd": cmd, "deps": list(deps), "array": array, "snapshot": name,
                "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(stage, jid, flush=True)
    return jid


def main(snap):
    ids = []
    plan = [(ev, FULL_REGIMES) for ev in FULL] + [(ev, LAYER_REGIMES) for ev in LAYERS_OTHER]
    for ev, regimes in plan:
        for rg in regimes:
            cmd = (f"{PY} -m atlas.stage0c_run --evidence {ev} --regime {rg} "
                   f"--seed $((2 + 2*(SLURM_ARRAY_TASK_ID/5))) --fold $((SLURM_ARRAY_TASK_ID%5))")
            ids.append(sb(snap, f"{ev}_{rg}", cmd, array="0-9%5"))
    agg = sb(snap, "aggregate", f"{PY} -m atlas.stage0_ablation_aggregate", deps=ids, hours="01:00:00", mem="24G")
    json.dump({"arrays": ids, "aggregate": agg}, open("results/stage0_ablation/last_submit_ids.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
