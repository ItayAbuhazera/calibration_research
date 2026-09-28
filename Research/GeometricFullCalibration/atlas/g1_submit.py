"""Submit G1 (docs/g1_conditional_access_spec.md) from an immutable snapshot: GPU extraction per checkpoint, CPU fit arrays per
(checkpoint, regime, arm) over folds (D-shuf: fold 0 only), then aggregation (afterok).

    python -m atlas.g1_submit snapshots/<dir>
"""
import json
import os
import subprocess
import sys
import time

from .g1_fit import REGIME_ARMS

PY = "/home/itayab/.conda/envs/geo_cuda12/bin/python"
LEDGER = "results/g1/ledger.json"


def sb(snap, name, cmd, deps=(), mem="24G", hours="03:00:00", cpus=4, array=None, gpu=False, study="g1"):
    logdir = os.path.abspath(f"results/{study}/logs"); os.makedirs(logdir, exist_ok=True)
    args = ["sbatch", "--parsable", f"--job-name={study}-{name}", f"--mem={mem}", f"--time={hours}", f"--cpus-per-task={cpus}",
            *(["--partition=rtx4090", "--gres=gpu:rtx_4090:1", "--exclude=cs-4090-09,cs-4090-05"] if gpu else ["--partition=cpu"]),
            f"--output={logdir}/{name}_%A_%a.out" if array else f"--output={logdir}/{name}_%j.out"]
    if array:
        args.append(f"--array={array}")
    if deps:
        args.append("--dependency=afterok:" + ":".join(str(d) for d in deps))
    args += ["--wrap", f"cd {os.path.abspath(snap)} && hostname && lscpu | grep 'Model name' && PYTHONPATH={os.path.abspath(snap)} PYTHONDONTWRITEBYTECODE=1 "
                       f"OMP_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS={cpus} {cmd}"]
    jid = subprocess.check_output(args, text=True).strip().split(";")[0]
    ledger = f"results/{study}/ledger.json"
    L = json.load(open(ledger)) if os.path.exists(ledger) else []
    L.append({"job_id": jid, "name": name, "cmd": cmd, "deps": list(deps), "array": array, "snapshot": os.path.basename(snap), "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")})
    os.makedirs(os.path.dirname(ledger), exist_ok=True); json.dump(L, open(ledger, "w"), indent=1)
    print(name, jid, flush=True)
    return jid


def main(snap):
    fits = []
    for seed in (2, 4):
        ext = sb(snap, f"x{seed}", f"{PY} -m atlas.g1_extract --seed {seed}", gpu=True, hours="01:00:00", mem="48G")
        for regime, arms in REGIME_ARMS.items():
            for arm in arms:
                arr = "0" if arm == "Dshuf" else "0-4"
                fits.append(sb(snap, f"s{seed}_{regime}_{arm}", f"{PY} -m atlas.g1_fit --seed {seed} --regime {regime} --arm {arm} --fold $SLURM_ARRAY_TASK_ID",
                               deps=[ext], array=arr))
    sb(snap, "aggregate", f"{PY} -m atlas.g1_aggregate", deps=fits, hours="01:00:00", mem="32G")


if __name__ == "__main__":
    main(sys.argv[1])
