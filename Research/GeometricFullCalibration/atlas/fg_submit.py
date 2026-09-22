"""Submit stages of the fixed-gate study from an immutable snapshot: python -m atlas.fg_submit <snapshot> <stages>
Reuses atlas.submit.sb; ledger: results/fixed_gate/ledger.json."""
import json, os, sys
from . import submit as S

S.LEDGER = "results/fixed_gate/ledger.json"
PY = S.PY
IDS = "results/fixed_gate/last_submit_ids.json"


def load():
    return json.load(open(IDS)) if os.path.exists(IDS) else {}


def main(snap, stages):
    want = set(stages.split(",")); ids = load()
    for s in (2, 4):
        if "ext" in want:
            ids[f"ext_{s}"] = S.sb(snap, f"fgext_s{s}", f"{PY} -m atlas.stage_b --seed {s} --site layer3.22 --pool grid2 --mode F --metric unit_l2 --out_root results/fixed_gate/seed{s}", gpu=True, hours="02:00:00")
        if "num" in want:
            ids[f"num_{s}"] = S.sb(snap, f"fgnum_s{s}", f"{PY} -m atlas.fg_numerics --seed {s}", gpu=True, hours="02:00:00")
        if "lat" in want:
            ids[f"lat_{s}"] = S.sb(snap, f"fglat_s{s}", f"{PY} -m atlas.fg_latency --seed {s}", gpu=True, hours="01:00:00", deps=[ids[f"freeze_{s}"]] if f"freeze_{s}" in ids else [])
        if "freeze" in want:
            deps = [ids[f"ext_{s}"]] if f"ext_{s}" in ids and os.environ.get("FG_DEP", "1") == "1" else []
            ids[f"freeze_{s}"] = S.sb(snap, f"fgfreeze_s{s}", f"{PY} -m atlas.fg_run --seed {s} --stage freeze", deps=deps, hours="02:00:00")
        if "audit" in want:
            deps = [ids[f"ext_{s}"]] if f"ext_{s}" in ids else []
            ids[f"audit_{s}"] = S.sb(snap, f"fgaudit_s{s}", f"{PY} -m atlas.fg_audit --seed {s}", deps=deps, hours="01:00:00")
        if "eval" in want:
            ids[f"eval_{s}"] = S.sb(snap, f"fgeval_s{s}", f"{PY} -m atlas.fg_run --seed {s} --stage eval", deps=[ids[f"freeze_{s}"]], hours="03:00:00")
    if "report" in want:
        deps = [ids[f"eval_{s}"] for s in (2, 4) if f"eval_{s}" in ids]
        ids["report"] = S.sb(snap, "fgreport", f"{PY} -m atlas.fg_report", deps=deps, hours="01:00:00")
    json.dump(ids, open(IDS, "w"), indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
