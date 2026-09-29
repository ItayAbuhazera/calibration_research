"""N1a-DP aggregation, image-group bootstrap and frozen decision (docs/n1a_dp_spec.md sec. 5-7). Refuses incomplete arrays.
Output evidence only; shift-transfer HPO; family + image-identity holdout.

    python -m atlas.n1adp_aggregate
"""
import json
import os

import numpy as np

from . import decoder_panel as dp, n1a, n1adp, spec
from .n1a_aggregate import knn_ambiguity
from . import n1adp_rules as rules
from .stage0_aggregate import BOOT_SEED, group_id_for_bootstrap

REPORT = "results/n1adp/report"
B = 2000
NBIN = 10


C0_FITS = "results/n1a/fits"
N1A_REPORT = "results/n1a/report/n1a_aggregate.json"


def pool(family, base, heldout):
    xs = []
    if family == "C0":                  # continuity control: historical N1a selector outputs, reused exactly (spec sec. 4b)
        for k in range(5):
            x = np.load(f"{C0_FITS}/base{base}/{heldout}/Z/fold{k}.npz", allow_pickle=True)
            xs.append((dict(x, eval_route_rate=float((x["e"] > 0).mean()), inner_margin_pp=np.nan),
                       {"selected": {"lambda": float(x["selected_lambda"])}, "selected_objective": np.nan, "hpo_seconds": 0.0,
                        "refit_meta": {"converged": bool(x["converged"])}}))
    for k in range(5 if family != "C0" else 0):
        if not n1adp.is_complete(family, base, heldout, k):
            raise SystemExit(f"incomplete: {family} base{base} {heldout} fold{k} — refusing to aggregate")
        p = n1adp.out_paths(family, base, heldout, k); xs.append((np.load(p["npz"], allow_pickle=True), json.load(open(p["study"]))))
    cat = lambda key: np.concatenate([x[key] for x, _ in xs])  # noqa: E731
    P = {"img": cat("image_id"), "cell": cat("cell_index"), "d": cat("d_route").astype(np.int64), "e": cat("e"), "z": cat("zstd_eval"),
         "fold": np.concatenate([np.full(len(x["d_route"]), int(x["fold"])) for x, _ in xs]),
         "tmd": np.concatenate([np.full(len(x["d_route"]), float(x["train_mean_delta"])) for x, _ in xs])}
    o = np.lexsort((P["cell"], P["img"]))
    P = {k: v[o] for k, v in P.items()}
    meta = [{"fold": k, "selected": st["selected"], "selected_inner_objective": st["selected_objective"], "grid_edge": st.get("grid_edge"),
             "converged": st.get("refit_meta", {}).get("converged", True), "hpo_min": st["hpo_seconds"] / 60,
             "eval_route_rate": float(x["eval_route_rate"]), "inner_margin_pp": float(x["inner_margin_pp"])} for k, (x, st) in enumerate(xs)]
    return P, meta


def main():
    os.makedirs(REPORT, exist_ok=True)
    gid = group_id_for_bootstrap(); _, ginv = np.unique(gid, return_inverse=True); ng = ginv.max() + 1
    res = {"label": "N1a-DP development audit on exposed cells; Decoder Panel v1 output-only selectors (F_Z(b), 206 features); shift-transfer "
                    "HPO (inner leave-one-training-family-out, objective = realized policy utility); family + image-identity holdout; "
                    "95% image-group bootstrap (one resample array per base shared by all families)", "bases": {}}
    FAMS = tuple(rules.FAMILIES5) + ("C0",)
    per = {f: {} for f in FAMS}; MB = {}; validity = {f: {} for f in rules.FAMILIES5}; info = {f: {} for f in rules.FAMILIES5}; unit_pass = {}
    for base in n1a.BASES:
        rng = np.random.default_rng(BOOT_SEED + 70000 + base); idx = rng.integers(0, ng, (B, ng))
        R = np.zeros((B, ng), np.float32)
        for bi in range(B):
            R[bi] = np.bincount(idx[bi], minlength=ng)
        base_out = {"families": {}}; mb_boot = np.zeros((B, len(n1a.FAMILIES))); mb_pt = []
        ref = {}
        for fam in FAMS:
            boot = {k: np.zeros((B, len(n1a.FAMILIES))) for k in ("G", "h", "Gc", "MA", "U")}
            fs, conv, metas = {}, True, {}
            for hi, ho in enumerate(n1a.FAMILIES):
                P, meta = pool(fam, base, ho); metas[ho] = meta; conv &= all(m["converged"] for m in meta)
                if ho in ref:
                    assert np.array_equal(ref[ho]["img"], P["img"]) and np.array_equal(ref[ho]["d"], P["d"])
                else:
                    ref[ho] = P
                d, g = P["d"], ginv[P["img"]]; route = P["e"] > 0
                edges = np.quantile(P["e"], np.linspace(0, 1, NBIN + 1)[1:-1]); bins = np.digitize(P["e"], edges)
                cols = {"n": np.ones_like(d, float), "pos": (d == 1).astype(float), "neg": (d == -1).astype(float), "g": d * route,
                        "dsum": d.astype(float)}
                S = {k: np.bincount(g, weights=v, minlength=ng) for k, v in cols.items()}
                Sb = {s: np.stack([np.bincount(g[bins == j], weights=(d[bins == j] == (1 if s == "pos" else -1)).astype(float), minlength=ng) for j in range(NBIN)], 1) for s in ("pos", "neg")}
                N = R @ S["n"]; pos, neg = R @ S["pos"] / N, R @ S["neg"] / N; Gc = np.maximum(0, R @ S["dsum"] / N)
                boot["h"][:, hi] = pos - Gc; boot["G"][:, hi] = R @ S["g"] / N; boot["Gc"][:, hi] = Gc
                boot["U"][:, hi] = pos - boot["G"][:, hi]                        # oracle gain - realized gain = unrecovered opportunity
                boot["MA"][:, hi] = np.minimum(R @ Sb["pos"], R @ Sb["neg"]).sum(1) / N
                n = len(d); P1, M1 = (d == 1).mean(), (d == -1).mean(); Gcv = max(0.0, float(d.mean())); Gm = float(np.mean(d * route))
                ma = sum(min((d[bins == j] == 1).sum(), (d[bins == j] == -1).sum()) for j in range(NBIN)) / n
                cal = [{"bin": j, "mean_e": float(P["e"][bins == j].mean()), "mean_delta": float(d[bins == j].mean()), "n": int((bins == j).sum())} for j in range(NBIN)]
                by_sev = {}
                for s in spec.DEV_SEVERITIES:
                    m = P["cell"] == spec.CELLS.index(f"{ho}_s{s}"); dm, rm = d[m], route[m]
                    by_sev[int(s)] = {"G": 100 * float(np.mean(dm * rm)), "G_oracle": 100 * float((dm == 1).mean()), "G_const": 100 * max(0.0, float(dm.mean())),
                                      "repair_capture": float(((dm == 1) & rm).sum() / max(1, (dm == 1).sum())), "route_rate": 100 * float(rm.mean())}
                fs[ho] = {"n_eval_rows": n, "support_repair": int((d == 1).sum()), "support_harm": int((d == -1).sum()), "G_oracle": 100 * P1,
                          "G_const_eval": 100 * Gcv, "G_const_train": 100 * float(np.mean(np.where(P["tmd"] > 0, d, 0))), "h": 100 * (P1 - Gcv),
                          "G": 100 * Gm, "gain_over_best_fixed": 100 * (Gm - Gcv), "regret_to_oracle": 100 * (P1 - Gm),
                          "phi_recovered": (Gm - Gcv) / (P1 - Gcv) if P1 > Gcv else None, "repair_capture": float(((d == 1) & route).sum() / max(1, (d == 1).sum())),
                          "harmful_routing_rate": 100 * float(((d == -1) & route).mean()), "harm_capture": float(((d == -1) & route).sum() / max(1, (d == -1).sum())),
                          "route_rate": 100 * float(route.mean()), "MA": 100 * ma,
                          "calibration_bins": cal, "calibration_error_pp": 100 * float(sum(c["n"] * abs(c["mean_e"] - c["mean_delta"]) for c in cal) / n),
                          "by_severity": by_sev, "fold_meta": meta}
                if fam == rules.FAMILIES5[0]:
                    mb = knn_ambiguity(P); mb_pt.append(100 * float(mb.mean())); Sm = np.bincount(g, weights=mb, minlength=ng); mb_boot[:, hi] = R @ Sm / N
            ci = lambda a: [float(np.quantile(a, q)) for q in (.025, .975)]  # noqa: E731
            pt = lambda k: float(np.mean([fs[h][k] for h in n1a.FAMILIES]))  # noqa: E731
            phi_b = (boot["G"].mean(1) - boot["Gc"].mean(1)) / boot["h"].mean(1)
            hmac = pt("h")
            q = {"G": pt("G"), "G_ci": ci(100 * boot["G"].mean(1)), "G_oracle": pt("G_oracle"), "G_const_eval": pt("G_const_eval"),
                 "G_const_train": pt("G_const_train"), "h": hmac, "h_ci": ci(100 * boot["h"].mean(1)),
                 "gain_over_best_fixed": pt("gain_over_best_fixed"), "regret_to_oracle": pt("regret_to_oracle"),
                 "phi": (pt("G") - pt("G_const_eval")) / hmac, "phi_ci": ci(phi_b),
                 "U": pt("G_oracle") - pt("G"), "U_ci": ci(100 * boot["U"].mean(1)), "MA": pt("MA"), "MA_ci": ci(100 * boot["MA"].mean(1)),
                 "MA_by_family": {h: fs[h]["MA"] for h in n1a.FAMILIES}, "repair_capture": pt("repair_capture"), "harmful_routing_rate": pt("harmful_routing_rate"),
                 "calibration_error_pp": pt("calibration_error_pp"), "converged": bool(conv),
                 "support_ok": all(fs[h]["support_repair"] >= 300 and fs[h]["support_harm"] >= 300 for h in n1a.FAMILIES)}
            per[fam][base] = q
            if fam == "C0":
                base_out["C0_continuity_control"] = {"family_macro": q, "by_heldout_family": fs}; continue
            up = {h: [rules.unit_passes(m["eval_route_rate"], m["inner_margin_pp"], fam) for m in metas[h]] for h in n1a.FAMILIES}
            info[fam][base] = rules.informative(up); unit_pass[(fam, base)] = up
            validity[fam][base] = {"complete": True, "converged": bool(conv), "support": q["support_ok"]}
            if fam in ("linear", "poly2", "rff"):
                validity[fam][base]["no_strong_edge"] = sum(m["selected"].get("lambda") == dp.LAMBDA_GRID[0] for h in n1a.FAMILIES for m in metas[h]) < 2
            base_out["families"][fam] = {"family_macro": q, "by_heldout_family": fs, "validity": validity[fam][base],
                                         "informative": info[fam][base], "unit_pass_I1_I2": up, "tau_pp": rules.tau(fam)}
        MB[base] = float(np.mean(mb_pt)); base_out["MB_local_Z"] = MB[base]; base_out["MB_ci"] = [100 * float(np.quantile(mb_boot.mean(1), q)) for q in (.025, .975)]
        res["bases"][str(base)] = base_out
    valid = {f for f in rules.FAMILIES5 if all(all(v.values()) for v in validity[f].values())}
    res["valid_families"] = sorted(valid)
    hist = json.load(open(N1A_REPORT))["bases"]
    vc0 = {b: abs(per["C0"][b]["G"] - hist[str(b)]["family_macro"]["G_Z"]) for b in n1a.BASES}
    res["V_C0"] = {"abs_diff_pp": vc0, "tolerance_pp": 1e-9, "passed": all(v <= 1e-9 for v in vc0.values())}
    res["decision"] = rules.decide({f: per[f] for f in rules.FAMILIES5}, MB, valid, info, validity_extra_ok=res["V_C0"]["passed"])
    res["historical_reference"] = {"N1a_G_Z_pp": {"2": 2.03, "4": 1.25}, "N1a_MA_pp": {"2": 7.12, "4": 6.91}, "N1a_MB_pp": {"2": 3.51, "4": 3.44},
                                   "N1a_Q_Zother_reference_pp": {"2": 0.54, "4": 0.41}, "note": "reference only; Z_other is not admissible pre-action evidence"}
    json.dump(res, open(f"{REPORT}/n1adp_aggregate.json", "w"), indent=1, default=float)
    f2 = lambda v, c: "%.2f [%.2f, %.2f]" % (v, *c)  # noqa: E731
    L = []
    for base in n1a.BASES:
        L += [f"### Base {base} -> {n1a.other(base)} (family-macro; M_B local-Z = {MB[base]:.2f} pp)", "",
              "| family | G (pp) | gain over best fixed | phi recovered | U unrecovered (pp) | M_A (pp) | repair capture | harmful routing (pp) | calib. err (pp) | label |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for fam in FAMS:
            q = per[fam][base]
            L.append(f"| {fam} | {f2(q['G'], q['G_ci'])} | {q['gain_over_best_fixed']:.2f} | {f2(q['phi'], q['phi_ci'])} | {f2(q['U'], q['U_ci'])} | "
                     f"{f2(q['MA'], q['MA_ci'])} | {q['repair_capture']:.3f} | {q['harmful_routing_rate']:.2f} | {q['calibration_error_pp']:.2f} | "
                     f"{res['decision']['labels'][fam][base] if fam != 'C0' else 'continuity control'} |")
        L.append("")
    L.append("Decision: " + json.dumps({k: v for k, v in res["decision"].items() if k != "labels"}))
    open(f"{REPORT}/n1adp_table.md", "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
