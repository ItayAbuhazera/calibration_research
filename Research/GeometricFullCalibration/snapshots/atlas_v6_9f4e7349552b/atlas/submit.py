"""Submit the atlas DAG to Slurm from an immutable snapshot, recording every job in results/atlas/ledger.json.

  python -m atlas.submit <snapshot_dir> [--stages u0,a,...]
Dependencies use ``afterok`` on the producers (a failed producer blocks its consumers); every stage additionally validates its
inputs and exits non-zero if they are missing or incompatible. Stage scripts are idempotent (completion markers), so a bounded retry
of an infrastructure failure is safe. No worker selects anything with corruption labels: the selection stages read validation rows only.
"""
import json, os, subprocess, sys, time

PY = "/home/itayab/.conda/envs/geo_cuda12/bin/python"
LEDGER = "results/atlas/ledger.json"
POOLS = ("gap", "grid2", "spp")


def ledger_add(entry):
    L = json.load(open(LEDGER)) if os.path.exists(LEDGER) else []
    L.append(entry)
    json.dump(L, open(LEDGER, "w"), indent=1)


def sb(snap, stage, cmd, deps=(), gpu=False, mem="60G", hours="02:00:00", array=None, cpus=4):
    name = os.path.basename(snap)
    logdir = os.path.abspath("results/atlas/logs"); os.makedirs(logdir, exist_ok=True)
    args = ["sbatch", "--parsable", f"--job-name=atlas-{stage}", f"--mem={mem}", f"--time={hours}", f"--cpus-per-task={cpus}",
            f"--output={logdir}/{stage}_%A_%a.out" if array else f"--output={logdir}/{stage}_%j.out"]
    args += ["--partition=rtx4090", "--gres=gpu:rtx_4090:1"] if gpu else ["--partition=cpu"]
    if array:
        args.append(f"--array={array}")
    if deps:
        args.append("--dependency=afterok:" + ":".join(str(d) for d in deps))
    wrap = f"cd {os.path.abspath(snap)} && ATLAS_SNAPSHOT={name} PYTHONPATH={os.path.abspath(snap)} PYTHONDONTWRITEBYTECODE=1 {cmd}"
    args += ["--wrap", wrap]
    jid = subprocess.check_output(args, text=True).strip().split(";")[0]
    ledger_add({"job_id": jid, "stage": stage, "cmd": cmd, "deps": list(deps), "gpu": gpu, "array": array, "snapshot": name,
                "submitted": time.strftime("%Y-%m-%dT%H:%M:%S")})
    print(stage, jid); return jid


def js(seed, key):
    return f"$({PY} -c \"import json;print(json.load(open('results/atlas/seed{seed}/{key[0]}'))['{key[1]}']['{key[2]}']{key[3]})\")"


def main(snap, stages="all"):
    want = None if stages == "all" else set(stages.split(","))
    ok = lambda s: want is None or s in want  # noqa: E731
    ids = {}
    for s in (2, 4):
        if ok("u0"):
            ids[("u0", s)] = sb(snap, f"u0_s{s}", f"{PY} -m atlas.stage_u0 --seed {s}", gpu=True, hours="02:00:00")
        if ok("a"):
            ids[("a", s)] = sb(snap, f"A_s{s}", f"{PY} -m atlas.stage_a --seed {s} --group $SLURM_ARRAY_TASK_ID", gpu=True, array="0-15%4", hours="03:00:00")
        if ok("sel"):
            deps = [ids[k] for k in (("a", s), ("u0", s)) if k in ids]
            ids[("sel", s)] = sb(snap, f"sel_s{s}", f"{PY} -m atlas.stage_stats --seed {s} --what clean && {PY} -m atlas.stage_stats --seed {s} --what select", deps=deps)
        if ok("b"):
            for p in POOLS:
                site = f"$({PY} -c \"import json;print(json.load(open('results/atlas/seed{s}/shortlist.json'))['chosen']['{p}']['site'])\")"
                ids[("b", s, p)] = sb(snap, f"B_s{s}_{p}", f"{PY} -m atlas.stage_b --seed {s} --site {site} --pool {p} --mode atlas", deps=[ids[("sel", s)]] if ("sel", s) in ids else [], gpu=True, hours="03:00:00")
        if ok("bsel"):
            deps = [ids[k] for k in ids if k[0] == "b" and k[1] == s] + ([ids[("u0", s)]] if ("u0", s) in ids else [])
            ids[("bsel", s)] = sb(snap, f"bsel_s{s}", f"{PY} -m atlas.stage_bsel --seed {s}", deps=deps)
        if ok("f"):
            for p in POOLS:
                site = f"$({PY} -c \"import json;print(json.load(open('results/atlas/seed{s}/shortlist.json'))['chosen']['{p}']['site'])\")"
                met = f"$({PY} -c \"import json;print(json.load(open('results/atlas/seed{s}/metric_selection.json'))['hidden']['{p}']['metric'])\")"
                ids[("f", s, p)] = sb(snap, f"F_s{s}_{p}", f"{PY} -m atlas.stage_b --seed {s} --site {site} --pool {p} --mode F --metric {met}", deps=[ids[("bsel", s)]] if ("bsel", s) in ids else [], gpu=True, hours="04:00:00")
        if ok("cd"):
            deps = [ids[k] for k in ids if k[0] == "f" and k[1] == s] + [ids[("u0", s)]] if ("u0", s) in ids else []
            ids[("cd", s)] = sb(snap, f"CD_s{s}", f"{PY} -m atlas.stage_cd --seed {s}", deps=deps, hours="03:00:00")
        if ok("v"):
            ids[("v", s)] = sb(snap, f"V_s{s}", f"{PY} -m atlas.stage_v --seed {s}", deps=[ids[("sel", s)]] if ("sel", s) in ids else [], gpu=True, hours="03:00:00")
        if ok("lat"):
            ids[("lat", s)] = sb(snap, f"L_s{s}", f"{PY} -m atlas.stage_latency --seed {s}", deps=[ids[("bsel", s)]] if ("bsel", s) in ids else [], gpu=True, hours="01:00:00")
        if ok("target"):
            deps = [ids[k] for k in (("sel", s),) if k in ids]
            ids[("target", s)] = sb(snap, f"target_s{s}", f"{PY} -m atlas.stage_stats --seed {s} --what target", deps=deps)
    json.dump({str(k): v for k, v in ids.items()}, open("results/atlas/last_submit_ids.json", "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
