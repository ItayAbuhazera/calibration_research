"""Read-only report tables from existing Stage 0 outputs (no new fits).

    python -m atlas.stage0_report
"""
from __future__ import annotations

import json

import numpy as np

from . import spec
from .stage0_aggregate import (BOOT_SEED, group_id_for_bootstrap, grouped_bootstrap, per_image_correct, pool_oof, OUT)

R = f"{OUT}"


def main():
    agg = json.load(open(f"{R}/stage0c_aggregate.json"))
    b = json.load(open(f"{R}/stage0b_layer_probe_eval.json"))
    elig = json.load(open(f"{R}/stage0b_eligibility_flag.json"))
    gid = group_id_for_bootstrap()
    out = {"note": "computed from existing outputs; no new fits"}

    # (a) Delta_S and Delta_T at 8k x 1 and 2.5k x 1, macro-12, with CIs; gap CIs from aggregate
    ps = agg["primary_secondary_by_seed"]
    sc = agg["secondary_contrasts"]
    out["delta_single_view"] = {
        s: {r: {"delta_pp": ps[s]["macro12"][r]["delta_acc_pp"], "ci95_pp": ps[s]["macro12"][r]["bootstrap_pp"]["ci95_pp"]}
            for r in ("T-8k1", "S-8k1", "T-2.5k1", "S-2.5k1")} for s in ("2", "4")}
    out["recoverability_gap"] = {s: {k: sc[s][k] for k in ("recoverability_gap_8k", "recoverability_gap_2.5k")} for s in ("2", "4")}

    # (b) Delta_T per cell (T-8k12), per-cell grouped bootstrap
    per_cell = {}
    for seed in spec.DEV_SEEDS:
        p = pool_oof(seed, "T-8k12")
        rows = {}
        for i, c in enumerate(spec.CELLS):
            d = per_image_correct(p, "q_ZP", c) - per_image_correct(p, "q_Z", c)
            bs = grouped_bootstrap(d, gid, seed=BOOT_SEED + 1000 + 10 * seed + i)
            rows[c] = {"q_Z_acc": float(per_image_correct(p, "q_Z", c).mean()), "q_ZP_acc": float(per_image_correct(p, "q_ZP", c).mean()),
                       "delta_pp": bs["estimate"] * 100, "ci95_pp": [x * 100 for x in bs["ci95"]]}
        per_cell[str(seed)] = rows
    out["delta_T_per_cell_T8k12"] = per_cell

    # (c) 0b: layer3.22 probe alone vs base, per cell
    l322 = {}
    for seed in spec.DEV_SEEDS:
        rows = [r for r in b if r["seed"] == seed and r["layer_name"] == "layer3.22"]
        l322[str(seed)] = {r["cell"]: {"acc": r["accuracy"], "base_acc": r["base_accuracy"], "delta_pp": r["delta_accuracy_pp_vs_base"],
                                       "nll": r["nll"], "base_nll": r["base_nll"], "brier": r["brier"], "base_brier": r["base_brier"]} for r in rows}
    out["layer3.22_probe_alone_vs_base_0b"] = l322
    out["eligibility_flags_0b"] = elig

    # (d) clean-copy accuracy/NLL: T-8k12 q_ZP vs S-8k1 q_ZP (and q_Z for reference)
    out["clean_copy"] = {s: {r: {arm: {"acc": ps[s]["per_regime_per_cell"][r]["clean"][arm]["accuracy"], "nll": ps[s]["per_regime_per_cell"][r]["clean"][arm]["nll"]}
                                 for arm in ("q_Z", "q_ZP")} for r in ("T-8k12", "S-8k1")} for s in ("2", "4")}
    json.dump(out, open(f"{R}/stage0c_requested_tables.json", "w"), indent=1)

    f = lambda x: f"{x:+.2f}"
    print("== single-view Delta (macro12, pp [CI]) ==")
    for s in ("2", "4"):
        for r in ("T-8k1", "S-8k1", "T-2.5k1", "S-2.5k1"):
            d = out["delta_single_view"][s][r]
            print(f"seed{s} {r:8s} {f(d['delta_pp'])} [{f(d['ci95_pp'][0])}, {f(d['ci95_pp'][1])}]")
        for k, v in out["recoverability_gap"][s].items():
            print(f"seed{s} {k} {f(v['estimate_pp'])} [{f(v['ci95_pp'][0])}, {f(v['ci95_pp'][1])}]")
    print("== Delta_T per cell (T-8k12, pp [CI]) ==")
    for c in spec.CELLS:
        print(f"{c:22s} " + "  ".join(f"s{s}: {f(per_cell[s][c]['delta_pp'])} [{f(per_cell[s][c]['ci95_pp'][0])},{f(per_cell[s][c]['ci95_pp'][1])}]" for s in ("2", "4")))
    print("== 0b layer3.22 probe alone minus base (pp) / acc ==")
    for c in spec.CONDITIONS:
        print(f"{c:22s} " + "  ".join(f"s{s}: {l322[s][c]['delta_pp']:+.2f} (probe {100*l322[s][c]['acc']:.1f} vs base {100*l322[s][c]['base_acc']:.1f}; nll {l322[s][c]['nll']:.2f} vs {l322[s][c]['base_nll']:.2f})" for s in ("2", "4")))
    print("eligible layers:", [k for k, v in elig.items() if v["eligible_descriptive_only"]] or "none")
    print("== clean copy: acc / nll ==")
    for s in ("2", "4"):
        for r in ("T-8k12", "S-8k1"):
            d = out["clean_copy"][s][r]
            print(f"seed{s} {r:7s} q_Z {100*d['q_Z']['acc']:.2f}/{d['q_Z']['nll']:.3f}  q_ZP {100*d['q_ZP']['acc']:.2f}/{d['q_ZP']['nll']:.3f}")


if __name__ == "__main__":
    main()
