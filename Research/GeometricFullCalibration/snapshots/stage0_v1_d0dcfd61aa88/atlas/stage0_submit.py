"""Submit the Stage 0 DAG to Slurm from an immutable snapshot, recording every job in
results/stage0/ledger.json (mirrors atlas/submit.py's convention).

  python -m atlas.snapshot stage0_v1                       # first, from the live tree
  python -m atlas.stage0_submit snapshots/stage0_v1_<hash>  # then, from here

CPU-only: the Section-3 benchmark (see docs/stage0_execution_spec.md Section 7 results) showed
one full-lambda-grid two-arm fit for the largest regime (T-8k12) completes in ~10-11 CPU-node
minutes; no GPU is used anywhere in Stage 0.
"""
import json
import os
import subprocess
import sys
import time

PY = "/home/itayab/.conda/envs/geo_cuda12/bin/python"
LEDGER = "results/stage0/ledger.json"
REGIMES = ("T-8k12", "T-8k1", "T-2.5k12", "T-2.5k1", "S-8k1", "S-2.5k1")
SEEDS = (2, 4)
FOLDS = range(5)
SHUFFLE_REGIMES = ("T-8k12", "T-2.5k12", "T-8k1")


def ledger_add(entry):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    L = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    L.append(entry)
    json.dump(L, open(LEDGER, "w"), indent=1)


def sb(snap, stage, cmd, deps=(), mem="16G", hours="03:00:00", cpus=4, array=None):
    name = os.path.basename(snap)
    logdir = os.path.abspath("results/stage0/logs"); os.makedirs(logdir, exist_ok=True)
    args = ["sbatch", "--parsable", f"--job-name=stage0-{stage}", f"--mem={mem}", f"--time={hours}",
            f"--cpus-per-task={cpus}", "--partition=cpu",
            f"--output={logdir}/{stage}_%A_%a.out" if array else f"--output={logdir}/{stage}_%j.out"]
    if array:
        args.append(f"--array={array}")
    if deps:
        args.append("--dependency=afterok:" + ":".join(str(d) for d in deps))
    wrap = f"cd {os.path.abspath(snap)} && STAGE0_SNAPSHOT={name} PYTHONPATH={os.path.abspath(snap)} PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS={cpus} {cmd}"
    args += ["--wrap", wrap]
    jid = subprocess.check_output(args, text=True).strip().split(";")[0]
    ledger_add({"job_id": jid, "stage": stage, "cmd": cmd, "deps": list(deps), "array": array, "snapshot": name,
                "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(stage, jid)
    return jid


def main(snap: str):
    ids = {}

    ids["folds"] = sb(snap, "folds", f"{PY} -m atlas.stage0_folds", hours="00:20:00", mem="8G", cpus=2)

    for seed in SEEDS:
        for regime in REGIMES:
            arr = f"0-4%3"
            ids[f"c_{seed}_{regime}"] = sb(
                snap, f"c_s{seed}_{regime}",
                f"{PY} -m atlas.stage0c_run --seed {seed} --regime {regime} --fold $SLURM_ARRAY_TASK_ID",
                deps=[ids["folds"]], hours="02:00:00", mem="16G", cpus=4, array=arr)

    for seed in SEEDS:
        for regime in SHUFFLE_REGIMES:
            ids[f"shuf_{seed}_{regime}"] = sb(
                snap, f"shuf_s{seed}_{regime}",
                f"{PY} -m atlas.stage0_shuffle --seed {seed} --regime {regime}",
                deps=[ids["folds"], ids[f"c_{seed}_{regime}"]], hours="02:00:00", mem="16G", cpus=4)

    ids["a"] = sb(snap, "0a", f"{PY} -m atlas.stage0a", deps=[ids["folds"]], hours="00:30:00", mem="8G", cpus=2)
    ids["b"] = sb(snap, "0b", f"{PY} -m atlas.stage0b", deps=[ids["folds"]], hours="00:30:00", mem="8G", cpus=2)

    all_c = [ids[f"c_{s}_{r}"] for s in SEEDS for r in REGIMES]
    all_shuf = [ids[f"shuf_{s}_{r}"] for s in SEEDS for r in SHUFFLE_REGIMES]
    ids["aggregate"] = sb(snap, "aggregate", f"{PY} -m atlas.stage0_aggregate", deps=all_c, hours="00:30:00", mem="16G", cpus=4)
    ids["verify"] = sb(snap, "verify", f"{PY} -m pytest {os.path.abspath(snap)}/tests/test_stage0.py -q",
                        deps=all_c + all_shuf, hours="00:30:00", mem="8G", cpus=2)

    json.dump({k: v for k, v in ids.items()}, open("results/stage0/last_submit_ids.json", "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
