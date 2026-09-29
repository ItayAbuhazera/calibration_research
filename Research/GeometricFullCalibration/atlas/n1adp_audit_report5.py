"""Five-family pre-freeze methodological summary for the PROSPECTIVE N1a-DP panel (docs/n1a_dp_prefreeze_methodology_audit_2026-09-29.md).

Reads ONLY already-completed outputs of the frozen six-family audit (docs/n1a_dp_stochasticity_audit_plan.md; runner atlas/n1adp_audit.py,
snapshot snapshots/n1adp_audit_242c55939ba4). The poly2 null portion of that audit was cancelled (runtime) and is not used. The frozen
runner and its `summary` are NOT modified; this script applies the same formulas to the five remaining families:
    s_m(q)  = max over the 4 audit units of the per-unit full-HPO replicate SD (no sqrt division)
    tau_m   = max(0.10 pp, 2 * s_m(inner margin)); deterministic families: s = 0, tau = 0.10 pp
    null    : unit passes iff route rate in [1 %, 99 %] (I1) and inner margin >= tau_m (I2); STOP iff any family passes in >= 2 of 4 units
Emits only variability statistics and permuted-label null quantities (no mean performance).

    python -m atlas.n1adp_audit_report5
"""
import json

from . import common

OUT = "results/n1adp_audit"
FAMILIES5 = ("linear", "rff", "lgbm", "mlp", "knn")
STOCHASTIC = ("lgbm", "mlp")
UNITS = 4
TAU_FLOOR, ROUTE_LO, ROUTE_HI, STOP_UNITS = 0.10, 0.01, 0.99, 2


def main():
    res = {"scope": "five-family prospective N1a-DP panel; completed outputs of the frozen six-family audit (poly2 null cancelled, unused)",
           "stochastic": {}, "null": {}}
    for fam in STOCHASTIC:
        A = [json.load(open(f"{OUT}/A/{fam}/u{u}.json")) for u in range(UNITS)]
        B = [json.load(open(f"{OUT}/B/{fam}/u{u}.json")) for u in range(UNITS)]
        per_unit = {"inner_margin_pp": [{k: a["inner_margin_pp"][k] for k in ("full_hpo", "final_fit_only")} for a in A]}
        for q in ("phi", "U", "MA"):
            per_unit[q] = [{k: b[q][k] for k in ("full_hpo", "final_fit_only")} for b in B]
        s = {q: max(x["full_hpo"]["sd"] for x in v) for q, v in per_unit.items()}
        s_fit = {q: max(x["final_fit_only"]["sd"] for x in v) for q, v in per_unit.items()}
        res["stochastic"][fam] = {"s": s, "final_fit_only_sd_max": s_fit, "tau_pp": max(TAU_FLOOR, 2 * s["inner_margin_pp"]),
                                  "per_unit": per_unit, "pseudo_families": [b["pseudo_family"] for b in B],
                                  "minutes": {"A": [round(a["seconds"] / 60, 1) for a in A], "B": [round(b["seconds"] / 60, 1) for b in B]}}
    stop = []
    for fam in FAMILIES5:
        tau = res["stochastic"].get(fam, {}).get("tau_pp", TAU_FLOOR)
        rows = []
        for u in range(UNITS):
            r = json.load(open(f"{OUT}/null/{fam}/u{u}.json"))
            i1 = ROUTE_LO <= r["route_rate"] <= ROUTE_HI; i2 = r["inner_margin_pp"] >= tau
            rows.append({"unit": u, "route_rate": r["route_rate"], "inner_margin_pp": r["inner_margin_pp"], "I1": i1, "I2": i2, "joint": i1 and i2,
                         "selected": r["selected"], "minutes": round(r["seconds"] / 60, 1)})
        n = sum(x["joint"] for x in rows)
        res["null"][fam] = {"tau_pp": tau, "units": rows, "n_joint_pass": n, "max_inner_margin_pp": max(x["inner_margin_pp"] for x in rows),
                            "triggers_STOP": n >= STOP_UNITS}
        if n >= STOP_UNITS:
            stop.append(fam)
    res["STOP"] = bool(stop); res["stop_families"] = stop
    common.atomic_json(f"{OUT}/summary_5family.json", res)
    for fam in FAMILIES5:
        v = res["null"][fam]
        print(fam, "tau", round(v["tau_pp"], 4), "joint", v["n_joint_pass"], "/4 max margin", round(v["max_inner_margin_pp"], 4),
              [(round(x["route_rate"], 3), round(x["inner_margin_pp"], 3), x["I1"], x["I2"]) for x in v["units"]])
    for fam, v in res["stochastic"].items():
        print(fam, "s", {k: round(x, 4) for k, x in v["s"].items()}, "fit-only max", {k: round(x, 4) for k, x in v["final_fit_only_sd_max"].items()})
    print("STOP", res["STOP"], res["stop_families"])


if __name__ == "__main__":
    main()
