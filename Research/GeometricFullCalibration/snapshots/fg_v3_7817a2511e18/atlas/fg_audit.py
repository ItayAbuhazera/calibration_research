"""CPU audit tables (spec sec. 2 of the fixed-gate study): matched candidate statistics, matched repair/harm rates, oracle-union scope.
  python -m atlas.fg_audit --seed 2
Reads existing atlas artifacts, the new fixed-candidate neighbours and atlas cd per-sample predictions; writes
results/fixed_gate/seed{S}/audit.json. Everything here is a descriptive recomputation; nothing is fitted.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from . import common, data, fg_data, spec, stats
from .stage_stats import QBLOCK


def D_counts(bp, j, y):
    D = (j == y).astype(int) - (bp == y).astype(int); dis = j != bp
    return {"D_pos_all": int((D > 0).sum()), "D_neg_all": int((D < 0).sum()), "D_zero_all": int((D == 0).sum()),
            "n_disagree": int(dis.sum()), "disagree_rate": float(dis.mean()), "gate_D_pos_W": int((dis & (D > 0)).sum()), "gate_D_neg_H": int((dis & (D < 0)).sum()),
            "gate_D_zero_U": int((dis & (D == 0)).sum())}


def row(bp, j, y):
    r = stats.wh_stats(bp, j, y)
    r.update(D_counts(bp, j, y))
    r["denominators"] = {"W/base_errors": r["n_base_errors"], "H/base_correct": r["n_base_correct"], "(W-H)/N": r["n"], "W/(W+H+U)": r["total_flips"], "W/(W+H)": r["W"] + r["H"]}
    return r


def main(seed: int):
    ctx = fg_data.load_ctx(seed); root = ctx["root"]
    out = {"seed": seed, "candidates": {}, "definitions": {}}
    for c, d in ctx["defs"].items():
        out["definitions"][c] = {"site": d["site"], "pool": d["pool"], "metric": d["metric"], "bank_rows": 45000, "k": spec.K_NN,
                                 "feature_dim": (spec.pool_dim(d["site"], d["pool"]) if d["kind"] == "hidden" else 100), "smoothing": "(n_c + pi_c)/(k+1)",
                                 "neighbour_file_dir": d["dir"]}
    sub = np.load(f"{data.SHARED}/subset_ids.npy")
    roles = ctx["roles"]
    for c in fg_data.CANDS:
        cand = fg_data.load_candidate(ctx, c)
        t = {}
        for role, rows in roles.items():
            t[f"role:{role}"] = row(ctx["base_pred"]["val"][rows], cand["val"]["j"][rows], ctx["val_y"][rows])
        for s in spec.CONDITIONS:
            t[f"full10k:{s}"] = row(ctx["base_pred"][s], cand[s]["j"], ctx["Y"][s])
            t[f"atlas_subset2000:{s}"] = row(ctx["base_pred"][s][sub], cand[s]["j"][sub], ctx["Y"][s][sub])
        # macro over the 12 cells, both row sets: repair rate, harm rate, net
        for tag in ("full10k", "atlas_subset2000"):
            cells = [t[f"{tag}:{s}"] for s in spec.CELLS]
            t[f"{tag}:macro12"] = {k: float(np.mean([r[k] for r in cells])) for k in ("candidate_acc_among_base_errors", "harm_rate_among_base_correct", "net_utility", "base_acc", "cand_acc", "disagree_rate")}
            t[f"{tag}:pooled12"] = {"W": sum(r["W"] for r in cells), "H": sum(r["H"] for r in cells), "U": sum(r["U"] for r in cells), "n_base_errors": sum(r["n_base_errors"] for r in cells),
                                    "n_base_correct": sum(r["n_base_correct"] for r in cells), "n": sum(r["n"] for r in cells)}
            p = t[f"{tag}:pooled12"]; p["W/base_errors"] = p["W"] / p["n_base_errors"]; p["H/base_correct"] = p["H"] / p["n_base_correct"]
        out["candidates"][c] = t
    # ---- matched repair/harm rates on the atlas subset for the deep-layer3 atlas sites vs the selected layer4 sites and the logit control
    st = json.load(open(f"{root}/stats_target_unit_l2.json"))
    tab = {}
    for site in ("layer3.22", "layer3.17", "layer3.12", "layer4.0", "layer4.1", "layer4.2"):
        for p in spec.POOLS:
            cells = [st[site][p][s] for s in spec.CELLS]
            tab[f"{site}|{p}"] = {"repair_rate_macro12": float(np.mean([r["candidate_acc_among_base_errors"] for r in cells])),
                                  "harm_rate_macro12": float(np.mean([r["harm_rate_among_base_correct"] for r in cells])), "net_macro12": float(np.mean([r["net_utility"] for r in cells])),
                                  "clean_repair": st[site][p]["clean"]["candidate_acc_among_base_errors"], "clean_harm": st[site][p]["clean"]["harm_rate_among_base_correct"]}
    peak = {}
    for p in spec.POOLS:
        best = max(spec.SITE_NAMES, key=lambda s: np.mean([st[s][p][c]["candidate_acc_among_base_errors"] for c in spec.CELLS]))
        peak[p] = {"site_with_max_macro_repair_by_target_labels": best, "repair_rate_macro12": tab.get(f"{best}|{p}", {}).get("repair_rate_macro12") or float(np.mean([st[best][p][c]["candidate_acc_among_base_errors"] for c in spec.CELLS])),
                   "harm_rate_macro12": float(np.mean([st[best][p][c]["harm_rate_among_base_correct"] for c in spec.CELLS]))}
    oc = out["candidates"]["out"]["atlas_subset2000:macro12"]
    out["matched_rates_atlas_subset"] = {"note": "same seed, same 2 000 class-stratified test images per cell, same 12 cells, same base predictions (u0), same macro-over-cells aggregation; ratios are per-cell then averaged",
                                         "atlas_sites": tab, "target_label_peak_sites": peak,
                                         "output_space_logit_control": {"repair_rate_macro12": oc["candidate_acc_among_base_errors"], "harm_rate_macro12": oc["harm_rate_among_base_correct"]},
                                         "caveat": "the 'peak' sites are chosen by TARGET labels among 34 sites (an optimistic maximum); they are not a clean-selected candidate"}
    # ---- oracle unions (evaluation-only), old set unchanged plus this study's fixed candidates, on full 10k with cd per-sample predictions
    ps = {s: np.load(f"{root}/cd/per_sample_{s}.npz") for s in spec.CONDITIONS}
    cand = {c: fg_data.load_candidate(ctx, c, sets=spec.CONDITIONS) for c in fg_data.CANDS}
    def anyc(get):
        return 100 * float(np.mean([np.any([g(s) == ctx["Y"][s] for g in get], axis=0) for s in spec.CELLS]))
    base_acc = 100 * float(np.mean([ctx["base_pred"][s] == ctx["Y"][s] for s in spec.CELLS]))
    P = lambda name: (lambda s: ps[s][f"pred__{name}"].astype(int))                       # noqa: E731
    Bs = lambda s: ctx["base_pred"][s]                                                      # noqa: E731
    deep = lambda s: cand["deep"][s]["j"]                                                   # noqa: E731
    old = [Bs, P("out_alone"), P("vs"), P("ms")] + [P(f"{p}_alone") for p in spec.POOLS]
    sl = json.load(open(f"{root}/shortlist.json"))["chosen"]
    out["oracle_union"] = {
        "label": "evaluation-only label oracle over a frozen prediction set; NOT a bound on any feasible gate; increments over an output ORACLE are not ceilings on gains over a feasible output-only system",
        "old_set_unchanged": {"members": ["base", "out_alone", "vs", "ms"] + [f"{p}_alone@{sl[p]['site']}" for p in spec.POOLS], "deep_layer3_spatial_included": False,
                              "base_acc_macro12": base_acc, "union_gain_pp": anyc(old) - base_acc, "output_only_oracle_gain_pp": anyc([Bs, P("out_alone"), P("vs"), P("ms")]) - base_acc,
                              "hidden_increment_pp": anyc(old) - anyc([Bs, P("out_alone"), P("vs"), P("ms")])},
        "this_study_fixed_candidates": {"members_deep": "base + deep(layer3.22/grid2) alone", "base+deep_pp": anyc([Bs, deep]) - base_acc,
                                        "base+out_alone+vs+ms_pp": anyc([Bs, P("out_alone"), P("vs"), P("ms")]) - base_acc,
                                        "base+out_alone+vs+ms+deep_pp": anyc([Bs, P("out_alone"), P("vs"), P("ms"), deep]) - base_acc,
                                        "deep_increment_over_output_oracle_pp": anyc([Bs, P("out_alone"), P("vs"), P("ms"), deep]) - anyc([Bs, P("out_alone"), P("vs"), P("ms")]),
                                        "base+deep+layer4gap_pp": anyc([Bs, deep, P("gap_alone")]) - base_acc}}
    # ---- storage vs resident size (bytes actually on disk vs analytic bank size)
    def du(p):
        return int(sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(p) for f in fs)) if os.path.exists(p) else 0
    out["storage_vs_resident"] = {"stored_bytes": {d: du(f"{root}/{d}") for d in ("knn_unit", "knn_F", "knn_B", "u0", "cd", "viz")} | {"fixed_gate_knn_F": du(f"{fg_data.FG}/seed{seed}/knn_F")},
                                  "note": "stored = neighbour indices/distances (int32+float32 x 50 per query) and logits, NOT feature banks; resident = 45 000 x dim x 4 bytes GPU bank",
                                  "resident_bank_bytes": {"deep layer3.22 grid2 (4096-d)": 45000 * 4096 * 4, "layer4.1 GAP (2048-d)": 45000 * 2048 * 4, "layer4.2 SPP (43008-d)": 45000 * 43008 * 4},
                                  "feature_shapes": {"layer3.22 map": [1024, 8, 8], "after adaptive_avg_pool2d(2)+flatten": [4096], "projection": "none"}}
    common.atomic_json(f"{fg_data.FG}/seed{seed}/audit.json", out)
    print("audit written")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, required=True); main(ap.parse_args().seed)
