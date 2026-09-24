"""Phase 1B (no fitting): reconciliation with the Uselis & Oh (ICLR 2025) claim as stated in the task,
from the Stage 0b per-layer probe evaluation and the layer-pilot frozen state.

  1. clean-selected layer: the layer probe with the best CLEAN accuracy; its 12-cell accuracy vs the native head
  2. target-selected layer per corruption family (best mean accuracy over the family's 3 severities);
     labelled target-selected (not deployable)
  3. the "retrained last-layer probe" analogue (layer4.2 probe) versus the native head and the best intermediate probes
  4. cached numbers on the probe recipe (selected lambda, grid edge)

    python -m atlas.stage0_reconcile_uo
"""
import json
import numpy as np
from . import spec, stage0_data

R = "results/stage0/report"
FAMS = spec.DEV_CORRUPTIONS


def main():
    b = json.load(open(f"{R}/stage0b_layer_probe_eval.json"))
    names = stage0_data.candidate_layer_names(2)
    out = {"label": "descriptive, cached outputs only, no fitting"}
    for seed in spec.DEV_SEEDS:
        A = {(r["layer_name"], r["cell"]): r["accuracy"] for r in b if r["seed"] == seed}
        base = {r["cell"]: r["base_accuracy"] for r in b if r["seed"] == seed}
        fam_cells = {f: [c for c in spec.CELLS if c.startswith(f)] for f in FAMS}
        macro = lambda L: 100 * float(np.mean([A[(L, c)] for c in spec.CELLS]))
        head_macro = 100 * float(np.mean([base[c] for c in spec.CELLS]))
        head_fam = {f: 100 * float(np.mean([base[c] for c in cs])) for f, cs in fam_cells.items()}
        res = {"head_clean": 100 * base["clean"], "head_macro12": head_macro, "head_by_family": head_fam}

        # 1. clean-selected (ties -> deeper layer)
        cs = max(range(len(names)), key=lambda i: (round(A[(names[i], "clean")], 6), i))
        L = names[cs]
        res["clean_selected"] = {"layer": L, "clean_acc": 100 * A[(L, "clean")], "macro12": macro(L), "delta_vs_head_pp": macro(L) - head_macro,
                                 "by_family": {f: 100 * float(np.mean([A[(L, c)] for c in cs_])) for f, cs_ in fam_cells.items()}}
        # clean-selected among INTERMEDIATE layers only (exclude the final layer4.2)
        ci = max(range(len(names) - 1), key=lambda i: (round(A[(names[i], "clean")], 6), i)); Li = names[ci]
        res["clean_selected_intermediate_only"] = {"layer": Li, "clean_acc": 100 * A[(Li, "clean")], "macro12": macro(Li), "delta_vs_head_pp": macro(Li) - head_macro}
        # 2. target-selected per family (not deployable)
        ts = {}
        for f, cells in fam_cells.items():
            fm = lambda L_: 100 * float(np.mean([A[(L_, c)] for c in cells]))
            best = max(names, key=fm)
            besti = max(names[:-1], key=fm)
            ts[f] = {"best_layer": best, "acc": fm(best), "head": head_fam[f], "delta_pp": fm(best) - head_fam[f],
                     "best_intermediate_layer": besti, "acc_intermediate": fm(besti), "delta_intermediate_pp": fm(besti) - head_fam[f]}
        res["target_selected_by_family__NOT_DEPLOYABLE"] = ts
        res["target_selected_macro_over_families__NOT_DEPLOYABLE"] = float(np.mean([ts[f]["acc"] for f in FAMS]))
        res["target_selected_intermediate_macro_over_families__NOT_DEPLOYABLE"] = float(np.mean([ts[f]["acc_intermediate"] for f in FAMS]))
        # 3. retrained last-layer probe analogue
        last = names[-1]
        res["retrained_last_layer_probe_layer4.2"] = {"clean": 100 * A[(last, "clean")], "macro12": macro(last), "delta_vs_head_macro12_pp": macro(last) - head_macro,
                                                       "max_abs_delta_vs_head_over_13_conditions_pp": 100 * max(abs(A[(last, c)] - base[c]) for c in spec.CONDITIONS)}
        res["per_layer_macro12_and_clean"] = {n: {"macro12": macro(n), "clean": 100 * A[(n, "clean")]} for n in names}
        # 4. recipe: selected lambda and grid edge (frozen state)
        fs = stage0_data.frozen_state(seed)
        res["probe_recipe"] = {n: {"selected_lambda": fs["probes"]["layers"][n]["lambda"], "inner_fit_nll_by_lambda": fs["probes"]["layers"][n]["lambda_grid_inner_fit_nll"]} for n in names}
        res["lambda_grid"] = sorted(float(k) for k in fs["probes"]["layers"][names[0]]["lambda_grid_inner_fit_nll"])
        out[str(seed)] = res
    json.dump(out, open(f"{R}/stage0_reconcile_uo.json", "w"), indent=1)
    for s, r in ((k, v) for k, v in out.items() if k in ("2", "4")):
        print("seed", s, "head clean %.2f macro12 %.2f" % (r["head_clean"], r["head_macro12"]))
        c = r["clean_selected"]; print("  clean-selected:", c["layer"], "clean %.2f macro12 %.2f (%+.2f pp vs head)" % (c["clean_acc"], c["macro12"], c["delta_vs_head_pp"]))
        c = r["clean_selected_intermediate_only"]; print("  clean-selected intermediate:", c["layer"], "clean %.2f macro12 %.2f (%+.2f)" % (c["clean_acc"], c["macro12"], c["delta_vs_head_pp"]))
        for f, t in r["target_selected_by_family__NOT_DEPLOYABLE"].items():
            print("  target-selected %-16s best %-9s %.2f vs head %.2f (%+.2f) | best intermediate %-9s %.2f (%+.2f)" % (f, t["best_layer"], t["acc"], t["head"], t["delta_pp"], t["best_intermediate_layer"], t["acc_intermediate"], t["delta_intermediate_pp"]))
        print("  target-selected macro over families %.2f ; intermediate-only %.2f" % (r["target_selected_macro_over_families__NOT_DEPLOYABLE"], r["target_selected_intermediate_macro_over_families__NOT_DEPLOYABLE"]))
        l = r["retrained_last_layer_probe_layer4.2"]; print("  layer4.2 probe: clean %.2f macro12 %.2f (%+.2f vs head); max |delta| any condition %.2f pp" % (l["clean"], l["macro12"], l["delta_vs_head_macro12_pp"], l["max_abs_delta_vs_head_over_13_conditions_pp"]))
        print("  lambdas:", {n: v["selected_lambda"] for n, v in r["probe_recipe"].items()}, "grid", r["lambda_grid"])
        l = r["probe_recipe"]["layer4.2"]["inner_fit_nll_by_lambda"]; print("  layer4.2 inner-fit NLL by lambda:", {k: round(v, 3) for k, v in l.items()})


if __name__ == "__main__":
    main()
