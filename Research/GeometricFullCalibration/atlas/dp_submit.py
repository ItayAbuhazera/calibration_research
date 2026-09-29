"""Slurm submission helpers for Decoder Panel v1 studies (engineering benchmarks, G1-DP, N1a-DP).

Every submission is appended to results/<study>/ledger.json; logs go to results/<study>/logs (persistent). Thread counts of every
BLAS/OpenMP library are aligned with --cpus-per-task (no nested parallelism; Optuna n_jobs = 1; Slurm arrays give the parallelism).

    python -m atlas.dp_submit engineering
"""
import json
import os
import subprocess
import sys
import time

PY = "/home/itayab/.conda/envs/geo_cuda12/bin/python"
BAD_GPU_NODES = "cs-4090-09,cs-4090-05"


def sb(study, name, cmd, cwd=".", deps=(), mem="16G", hours="12:00:00", cpus=8, array=None, gpu=False, dep_type="afterok"):
    logdir = os.path.abspath(f"results/{study}/logs"); os.makedirs(logdir, exist_ok=True)
    args = ["sbatch", "--parsable", f"--job-name={study}-{name}", f"--mem={mem}", f"--time={hours}", f"--cpus-per-task={cpus}",
            *(["--partition=rtx4090", "--gres=gpu:rtx_4090:1", f"--exclude={BAD_GPU_NODES}"] if gpu else ["--partition=cpu"]),
            f"--output={logdir}/{name}_%A_%a.out" if array else f"--output={logdir}/{name}_%j.out"]
    if array:
        args.append(f"--array={array}")
    if deps:
        args.append(f"--dependency={dep_type}:" + ":".join(str(d) for d in deps))
    thr = " ".join(f"{v}={cpus}" for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"))
    cwd = os.path.abspath(cwd)
    args += ["--wrap", f"cd {cwd} && hostname && lscpu | grep 'Model name' && PYTHONPATH={cwd} PYTHONDONTWRITEBYTECODE=1 {thr} {cmd}"]
    jid = subprocess.check_output(args, text=True).strip().split(";")[0]
    ledger = f"results/{study}/ledger.json"
    L = json.load(open(ledger)) if os.path.exists(ledger) else []
    L.append({"job_id": jid, "name": name, "cmd": cmd, "cwd": cwd, "deps": list(deps), "array": array, "cpus": cpus, "mem": mem, "time": hours,
              "gpu": gpu, "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")})
    json.dump(L, open(ledger, "w"), indent=1)
    print(name, jid, flush=True)
    return jid


def engineering():
    st = "decoder_panel"
    m = f"{PY} -m atlas.dp_engineering"
    for fam in ("linear", "poly2", "knn", "rff"):
        sb(st, f"eng_g1_{fam}", f"{m} --mode g1 --family {fam}", mem="32G", hours="08:00:00")
        sb(st, f"eng_n1a_{fam}", f"{m} --mode n1a --family {fam}", mem="32G", hours="06:00:00")
    sb(st, "eng_g1_lgbm_C50", f"{m} --mode g1 --family lgbm --trials 50 --arms C", mem="32G", hours="16:00:00")
    sb(st, "eng_g1_lgbm_AI", f"{m} --mode g1 --family lgbm --trials 30 --arms A,I", mem="32G", hours="16:00:00")
    sb(st, "eng_g1_mlp_cpu", f"{m} --mode g1 --family mlp --trials 50", mem="32G", hours="06:00:00")
    # done: 21725647
    # sb(st, "eng_g1_mlp_gpu", f"{m} --mode g1 --family mlp --trials 50 --device cuda", mem="32G", hours="1-00:00:00", gpu=True, cpus=4)
    sb(st, "eng_n1a_lgbm", f"{m} --mode n1a --family lgbm --trials 50", mem="32G", hours="08:00:00")
    sb(st, "eng_n1a_mlp", f"{m} --mode n1a --family mlp --trials 50", mem="32G", hours="08:00:00")


# ---- G1-DP / N1a-DP (resource classes: docs/decoder_panel_v1_resource_plan.md) ------------------------------------------------------
# family -> (split_arms, cpus, mem, time, concurrency)
G1DP_RES = {"linear": (True, 8, "24G", "08:00:00", 20), "poly2": (True, 8, "32G", "16:00:00", 20), "rff": (True, 8, "24G", "08:00:00", 20),
            "lgbm": (True, 8, "24G", "24:00:00", 20), "mlp": (False, 8, "24G", "06:00:00", 10), "knn": (False, 8, "32G", "02:00:00", 10)}
# PROSPECTIVE five-family N1a-DP (DRAFT r4; not authorized): family -> (cpus, mem, time, concurrency); poly2 removed prospectively
N1ADP_RES = {"linear": (8, "16G", "01:00:00", 20), "rff": (8, "16G", "06:00:00", 20), "lgbm": (8, "16G", "02:00:00", 20),
             "mlp": (8, "16G", "02:00:00", 20), "knn": (8, "16G", "01:00:00", 20)}


def write_manifest(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(rows, open(path, "w"), indent=1)


def g1dp(snap):
    from . import g1dp as G
    st = "g1dp"
    ext = [sb(st, f"extract_s{s}", f"{PY} -m atlas.g1dp_extract --seed {s}", cwd=snap, gpu=True, cpus=4, mem="48G", hours="01:30:00") for s in G.SEEDS]
    bun = sb(st, "bundle", f"{PY} -m atlas.g1dp bundle --seed $(( SLURM_ARRAY_TASK_ID / 5 == 0 ? 2 : 4 )) --fold $(( SLURM_ARRAY_TASK_ID % 5 ))",
             cwd=snap, deps=ext, array="0-9", cpus=2, mem="32G", hours="02:00:00")
    fits = []
    for fam, (split, cpus, mem, hours, conc) in G1DP_RES.items():
        U = G.units(split)
        write_manifest(f"results/{st}/array_manifest_{fam}.json", [{"index": i, "seed": s, "fold": f, "arm": a or "ALL"} for i, (s, f, a) in enumerate(U)])
        fits.append(sb(st, f"fit_{fam}", f"{PY} -m atlas.g1dp fit_index --family {fam} --index $SLURM_ARRAY_TASK_ID" + (" --split_arms" if split else ""),
                       cwd=snap, deps=[bun], array=f"0-{len(U) - 1}%{conc}", cpus=cpus, mem=mem, hours=hours))
    sb(st, "aggregate", f"{PY} -m atlas.g1dp_aggregate", cwd=snap, deps=fits, cpus=4, mem="64G", hours="04:00:00")


def n1adp(snap):
    from . import n1adp as N
    st = "n1adp"; U = N.units(); fits = []
    for fam, (cpus, mem, hours, conc) in N1ADP_RES.items():
        write_manifest(f"results/{st}/array_manifest_{fam}.json", [{"index": i, "base": b, "heldout": h, "fold": k} for i, (b, h, k) in enumerate(U)])
        fits.append(sb(st, f"fit_{fam}", f"{PY} -m atlas.n1adp fit_index --family {fam} --index $SLURM_ARRAY_TASK_ID", cwd=snap,
                       array=f"0-{len(U) - 1}%{conc}", cpus=cpus, mem=mem, hours=hours))
    sb(st, "aggregate", f"{PY} -m atlas.n1adp_aggregate", cwd=snap, deps=fits, cpus=4, mem="64G", hours="06:00:00")


def n1adp_audit(snap):
    """Frozen pre-freeze audit (docs/n1a_dp_stochasticity_audit_plan.md)."""
    from . import decoder_panel as dp
    st = "n1adp_audit"; m = f"{PY} -m atlas.n1adp_audit"; jobs = []
    for fam in ("lgbm", "mlp"):
        jobs.append(sb(st, f"A_{fam}", f"{m} A --family {fam} --unit $SLURM_ARRAY_TASK_ID", cwd=snap, array="0-3", cpus=8, mem="24G", hours="06:00:00"))
        jobs.append(sb(st, f"B_{fam}", f"{m} B --family {fam} --unit $SLURM_ARRAY_TASK_ID", cwd=snap, array="0-3", cpus=8, mem="24G", hours="04:00:00"))
    for fam in dp.FAMILIES:
        jobs.append(sb(st, f"null_{fam}", f"{m} null --family {fam} --unit $SLURM_ARRAY_TASK_ID", cwd=snap, array="0-3", cpus=8, mem="24G",
                       hours="06:00:00" if fam == "poly2" else "04:00:00"))
    sb(st, "summary", f"{m} summary", cwd=snap, deps=jobs, cpus=2, mem="8G", hours="00:30:00")


if __name__ == "__main__":
    fn = {"engineering": engineering, "g1dp": g1dp, "n1adp": n1adp, "n1adp_audit": n1adp_audit}[sys.argv[1]]
    fn(*sys.argv[2:])
