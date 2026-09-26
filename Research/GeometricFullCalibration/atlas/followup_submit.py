"""Submit a stage of the regime-map follow-up (docs/regime_map_followup_amendment_1.md) from an immutable snapshot.

  python -m atlas.followup_submit stage1 snapshots/<dir>

Stage 1: CPU array over (state, fold) = 35 tasks + aggregation (afterok). Later stages are added only after the previous gate passes.
"""
import json
import os
import subprocess
import sys
import time

PY = "/home/itayab/.conda/envs/geo_cuda12/bin/python"
LEDGER = "results/regime_map_followup/ledger.json"
STATES = ("a", "b1_s1", "b3_s1", "b10_s1", "b1_s2", "b3_s2", "b10_s2")


def sb(snap, stage, cmd, deps=(), mem="16G", hours="02:00:00", cpus=4, array=None):
    name = os.path.basename(snap)
    logdir = os.path.abspath("results/regime_map_followup/logs"); os.makedirs(logdir, exist_ok=True)
    args = ["sbatch", "--parsable", f"--job-name=fu-{stage}", f"--mem={mem}", f"--time={hours}", f"--cpus-per-task={cpus}", "--partition=cpu",
            f"--output={logdir}/{stage}_%A_%a.out" if array else f"--output={logdir}/{stage}_%j.out"]
    if array:
        args.append(f"--array={array}")
    if deps:
        args.append("--dependency=afterok:" + ":".join(str(d) for d in deps))
    args += ["--wrap", f"cd {os.path.abspath(snap)} && hostname && lscpu | grep 'Model name' && PYTHONPATH={os.path.abspath(snap)} PYTHONDONTWRITEBYTECODE=1 "
                       f"OMP_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS={cpus} {cmd}"]
    jid = subprocess.check_output(args, text=True).strip().split(";")[0]
    L = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    L.append({"job_id": jid, "stage": stage, "cmd": cmd, "deps": list(deps), "array": array, "snapshot": name, "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")})
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True); json.dump(L, open(LEDGER, "w"), indent=1)
    print(stage, jid, flush=True)
    return jid


def main(stage, snap):
    if stage == "stage1":
        ids = []
        for i, s in enumerate(STATES):
            ids.append(sb(snap, f"s1_{s}", f"{PY} -m atlas.followup_stage1 --state {s} --fold $SLURM_ARRAY_TASK_ID", array="0-4"))
        sb(snap, "s1_aggregate", f"{PY} -m atlas.followup_stage1_aggregate", deps=ids, hours="01:00:00", mem="24G")
    else:
        raise SystemExit("only stage1 is defined until Gate 1 passes")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
