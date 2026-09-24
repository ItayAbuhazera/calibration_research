"""PREPARED, NOT LAUNCHED (docs/regime_map_pilot_spec.md section 8): submit the regime-map pilot DAG from an immutable snapshot.

  python -m atlas.regime_submit snapshots/<snapshot_dir>      # only after the researcher's go

fine-tune (GPU) -> state extraction b1,b3,b10 (GPU, afterok); state a extraction (GPU, independent) ->
Stage 0 fits per state (CPU arrays, afterok) -> aggregation (afterok on all fits).
"""
import json
import os
import subprocess
import sys
import time

PY = "/home/itayab/.conda/envs/geo_cuda12/bin/python"
LEDGER = "results/regime_map/ledger.json"
REGIMES = ("T-8k1", "S-8k1", "T-2.5k1", "S-2.5k1")


def sb(snap, stage, cmd, deps=(), gpu=False, mem="32G", hours="03:00:00", cpus=4, array=None):
    name = os.path.basename(snap)
    logdir = os.path.abspath("results/regime_map/logs"); os.makedirs(logdir, exist_ok=True)
    args = ["sbatch", "--parsable", f"--job-name=rmap-{stage}", f"--mem={mem}", f"--time={hours}", f"--cpus-per-task={cpus}",
            f"--output={logdir}/{stage}_%A_%a.out" if array else f"--output={logdir}/{stage}_%j.out"]
    args += ["--partition=rtx4090", "--gres=gpu:rtx_4090:1", "--exclude=cs-4090-09,cs-4090-05"] if gpu else ["--partition=cpu"]
    if array:
        args.append(f"--array={array}")
    if deps:
        args.append("--dependency=afterok:" + ":".join(str(d) for d in deps))
    args += ["--wrap", f"cd {os.path.abspath(snap)} && PYTHONPATH={os.path.abspath(snap)} PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS={cpus} MKL_NUM_THREADS={cpus} OPENBLAS_NUM_THREADS={cpus} {cmd}"]
    jid = subprocess.check_output(args, text=True).strip().split(";")[0]
    L = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    L.append({"job_id": jid, "stage": stage, "cmd": cmd, "deps": list(deps), "gpu": gpu, "array": array, "snapshot": name, "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")})
    json.dump(L, open(LEDGER, "w"), indent=1)
    print(stage, jid, flush=True)
    return jid


def main(snap):
    ft = sb(snap, "finetune", f"{PY} -m atlas.regime_run finetune", gpu=True, hours="02:00:00", mem="48G")
    st = {"a": sb(snap, "state_a", f"{PY} -m atlas.regime_run state --state a", gpu=True, hours="02:00:00", mem="48G")}
    for s in ("b1", "b3", "b10"):
        st[s] = sb(snap, f"state_{s}", f"{PY} -m atlas.regime_run state --state {s}", deps=[ft], gpu=True, hours="02:00:00", mem="48G")
    fits = []
    for s, jid in st.items():
        for rg in REGIMES:
            fits.append(sb(snap, f"fit_{s}_{rg}", f"{PY} -m atlas.stage0c_run --seed 0 --regime {rg} --fold $SLURM_ARRAY_TASK_ID --regime-state {s}",
                           deps=[jid], hours="02:00:00", mem="16G", array="0-4%5"))
    sb(snap, "aggregate", f"{PY} -m atlas.regime_aggregate", deps=fits, hours="02:00:00", mem="24G")


if __name__ == "__main__":
    main(sys.argv[1])
