"""G1-DP aggregation, bootstrap and frozen verdicts (docs/g1_dp_spec.md sec. 5-8). Refuses incomplete arrays.
TARGET-POOLED HPO — ACCESSIBILITY DIAGNOSTIC ONLY. Anchored decoders; 12-cell macro accuracy; image-group bootstrap with one resample-index
array per checkpoint shared by all arms and all families.

    python -m atlas.g1dp_aggregate
"""
import json
import os

import numpy as np

from . import decoder_panel as dp, g1dp, spec
from .g1dp_rules import SHUF, classify, verdict
from .stage0_aggregate import BOOT_SEED, group_id_for_bootstrap

REPORT = "results/g1dp/report"
B = 2000
CONTRASTS = {"B-A": ("B", "A"), "C-A": ("C", "A"), "D-C": ("D", "C"), "D-B": ("D", "B"), "E-C": ("E", "C"), "F-C": ("F", "C"),
             "B-C": ("B", "C"), "H-A": ("H", "A"), "H-C": ("H", "C"), "I-C": ("I", "C"), "I-H": ("I", "H"),
             "Cs-A": ("Cs", "A"), "Hs-A": ("Hs", "A"), "Ds-C": ("Ds", "C")}
PRIMARY = {"B-C": "primary: compact P_3.22 vs H_L", "H-C": "raw tier: H_3.22_raw vs H_L", "D-C": "G1 conditional increment"}
RAW_ARMS = ("H", "I", "Hs")


def pool(family, seed, arm):
    cor = np.full((10000, len(spec.CONDITIONS)), np.nan); nll = np.full((10000, len(spec.CONDITIONS)), np.nan); studies = []
    for f in g1dp.FOLDS:
        if not g1dp.is_complete(family, seed, f, arm):
            raise SystemExit(f"incomplete: {family} seed{seed} fold{f} {arm} — refusing to aggregate")
        p = g1dp.out_paths(family, seed, f, arm); x = np.load(p["npz"], allow_pickle=True)
        cor[x["test_idx"]] = x["correct"].T; nll[x["test_idx"]] = x["nll"].T; studies.append(json.load(open(p["study"])))
    assert not np.isnan(cor).any()
    return cor, nll, studies


def fold_meta(st):
    rm = st.get("refit_meta", {})
    m = {"selected": st["selected"], "selected_inner_nll": st["selected_objective"], "null_inner_nll": (st.get("null_candidate") or {}).get("objective"),
         "grid_edge": st.get("grid_edge"), "hpo_min": st["hpo_seconds"] / 60, "converged": rm.get("converged", True)}
    if "jl_audit" in rm:
        m["jl_audit_refit"] = rm["jl_audit"]
    jl_inner = (st.get("study_meta") or {}).get("jl")
    if jl_inner:
        m["jl_audit_inner"] = [j.get("jl_audit") for j in jl_inner]
    return m


def main():
    os.makedirs(REPORT, exist_ok=True)
    gid = group_id_for_bootstrap(); _, inv = np.unique(gid, return_inverse=True); ng = inv.max() + 1; cnt = np.bincount(inv)
    gmean = lambda v: np.bincount(inv, weights=v, minlength=ng) / cnt  # noqa: E731
    raw_ok = {}
    for s in g1dp.SEEDS:
        c = json.load(open(f"{g1dp.H3_ROOT}/seed{s}/consistency.json")); raw_ok[s] = bool(c["consistency_passed"])
    hl_ok = all(json.load(open(f"{g1dp.HL_ROOT}/seed{s}/consistency.json"))["consistency_passed"] for s in g1dp.SEEDS)
    raw_tier = all(raw_ok.values())
    res = {"label": "G1-DP TARGET-POOLED HPO — ACCESSIBILITY DIAGNOSTIC ONLY; Decoder Panel v1; anchored decoders; 12-cell macro accuracy (pp); "
                    "95% image-group bootstrap intervals conditional on the fitted CV predictions (shared resamples across arms and families)",
           "raw_tier_valid": raw_tier, "raw_gate": raw_ok, "hl_gate": hl_ok, "families": {}}
    labels = {k: {} for k in PRIMARY}; validity = {}
    for fam in dp.FAMILIES:
        res["families"][fam] = {}; validity[fam] = {}
        for k in PRIMARY:
            labels[k][fam] = {}
        for seed in g1dp.SEEDS:
            rng = np.random.default_rng(BOOT_SEED + 60000 + seed); idx = rng.integers(0, ng, (B, ng))   # identical for every family
            macro, arms_meta, nllm = {}, {}, {}
            for arm in g1dp.ARM_ORDER:
                c, nl, st = pool(fam, seed, arm)
                macro[arm] = c[:, 1:].mean(1); nllm[arm] = float(nl[:, 1:].mean())
                arms_meta[arm] = {"macro12_acc": 100 * float(macro[arm].mean()), "clean_acc_descriptive": 100 * float(c[:, 0].mean()),
                                  "macro12_test_nll": nllm[arm], "folds": [fold_meta(x) for x in st]}
            G = {k: gmean(v) for k, v in macro.items()}
            q = {}
            for name, (a, b) in CONTRASTS.items():
                q[name] = 100 * float((macro[a] - macro[b]).mean())
                q[name + "_ci"] = [100 * float(np.quantile((G[a] - G[b])[idx].mean(1), p)) for p in (.025, .975)]
                q[name + "_class"] = classify(q[name], q[name + "_ci"])
            # validity (spec sec. 6)
            key = ("A", "B", "C", "D", "H", "I")
            v = {"V1_hl_gate": hl_ok, "V2_complete": True}
            if fam in ("linear", "poly2", "rff"):
                v["V3_converged"] = all(m["converged"] for a in key for m in arms_meta[a]["folds"])
                v["V4_no_strong_edge"] = all(sum(m["selected"].get("lambda") == dp.LAMBDA_GRID[0] for m in arms_meta[a]["folds"]) < 2 for a in ("A", "B", "C", "H"))
            v["V5_shuffle"] = q["Cs-A"] < SHUF and q["Ds-C"] < SHUF and (q["Hs-A"] < SHUF if raw_tier else True)
            if fam == "knn":
                aud = [j for a in g1dp.ARM_ORDER for m in arms_meta[a]["folds"] for j in ([m.get("jl_audit_refit")] + (m.get("jl_audit_inner") or [])) if j]
                v["V6_jl"] = all(j["p05"] >= 0.5 and j["p95"] <= 1.5 for j in aud)
                q["jl_audit_summary"] = {"n": len(aud), "min_p05": min(j["p05"] for j in aud) if aud else None,
                                         "max_p95": max(j["p95"] for j in aud) if aud else None,
                                         "median_of_medians": float(np.median([j["median"] for j in aud])) if aud else None}
            validity[fam][seed] = v
            for k in PRIMARY:
                labels[k][fam][seed] = q[k + "_class"]
            res["families"][fam][str(seed)] = {"contrasts": q, "validity": v, "arms": arms_meta}
    valid = {f for f in dp.FAMILIES if all(all(v.values()) for v in validity[f].values())}
    res["valid_families"] = sorted(valid)
    res["verdicts"] = {}
    for k, desc in PRIMARY.items():
        if k == "H-C" and not raw_tier:
            res["verdicts"][k] = {"outcome": "NOT_EVALUATED", "name": "raw tier invalid (H_3.22 gate failed)", "description": desc}; continue
        res["verdicts"][k] = dict(verdict(labels[k], valid), description=desc, labels=labels[k])
    json.dump(res, open(f"{REPORT}/g1dp_aggregate.json", "w"), indent=1, default=float)
    f = lambda v, c: "%+.2f [%+.2f, %+.2f]" % (v, *c)  # noqa: E731
    L = []
    for seed in g1dp.SEEDS:
        L += [f"### Checkpoint {seed}", "", "| contrast (pp) | " + " | ".join(dp.FAMILIES) + " |", "|---|" + "---|" * len(dp.FAMILIES)]
        for name in CONTRASTS:
            row = [f"{f(res['families'][fm][str(seed)]['contrasts'][name], res['families'][fm][str(seed)]['contrasts'][name + '_ci'])} {res['families'][fm][str(seed)]['contrasts'][name + '_class']}" for fm in dp.FAMILIES]
            L.append(f"| {name} | " + " | ".join(row) + " |")
        L.append("| arm accuracies A/B/C/H | " + " | ".join("/".join("%.2f" % res["families"][fm][str(seed)]["arms"][a]["macro12_acc"] for a in "ABCH") for fm in dp.FAMILIES) + " |")
        L.append("| valid | " + " | ".join(str(all(validity[fm][seed].values())) for fm in dp.FAMILIES) + " |"); L.append("")
    L.append("Verdicts: " + json.dumps({k: v["name"] for k, v in res["verdicts"].items()}))
    open(f"{REPORT}/g1dp_table.md", "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
