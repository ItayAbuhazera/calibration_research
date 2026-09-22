"""CPU: paired image-level bootstrap descriptions and evaluation-only oracle unions (per checkpoint).

* Bootstrap: resample the 10 000 original test-image IDs (shared by every condition and every pipeline of that checkpoint; corruption
  copies of an image move together), 2 000 resamples; statistic = macro mean over the 12 development cells (and, separately, clean) of
  the accuracy difference of two pipelines. DESCRIPTIVE: conditional on the checkpoint, not selection-adjusted for 102+ configurations,
  not a population-seed statement, two checkpoints only.
* Oracle unions: an evaluation-only label oracle that picks, per image, the correct one among already-fixed predictions. Reported
  SEPARATELY from learned-policy quantities and never as a bound on a feasible gate's advantage over a feasible output-only method.
"""
import json
import numpy as np

from . import data, spec

CELLS = spec.CELLS
B = 2000


def load(seed):
    root = data.seed_dir(seed)
    ps = {c: np.load(f"{root}/cd/per_sample_{c}.npz") for c in spec.CONDITIONS}
    return root, ps, json.load(open(f"{root}/shortlist.json"))["chosen"]


def correct(ps, name, cells):
    return np.mean([ps[c][f"pred__{name}"].astype(int) == ps[c]["labels"].astype(int) for c in cells], axis=0)   # per-image mean over cells


def boot(diff_per_image, seed=0):
    rng = np.random.default_rng(20260930 + seed)
    n = len(diff_per_image)
    b = np.array([diff_per_image[rng.integers(0, n, n)].mean() for _ in range(B)])
    return {"estimate_pp": 100 * float(diff_per_image.mean()), "ci95_pp": [100 * float(np.quantile(b, .025)), 100 * float(np.quantile(b, .975))]}


def main():
    out = {}
    for seed in spec.DEV_SEEDS:
        root, ps, sl = load(seed)
        res = json.load(open(f"{root}/cd/results.json"))
        best = max(spec.POOLS, key=lambda p: sl[p]["net_utility_selection"])
        outs = ("vs", "ms", "out_F0", "out_F1")
        strongest = max(outs, key=lambda n: np.mean([res[n]["T_nll"][c]["accuracy"] for c in CELLS]))
        contrasts = {f"{best}_F1 - base": (f"{best}_F1", "base"), f"{best}_F1 - {strongest}": (f"{best}_F1", strongest), f"{best}_F1 - {best}_F0": (f"{best}_F1", f"{best}_F0"),
                     f"{best}_F1 - out_F1": (f"{best}_F1", "out_F1"), f"{best}_alone - out_alone": (f"{best}_alone", "out_alone"), "vs - base": ("vs", "base"),
                     "ms - base": ("ms", "base"), "native_dac - base": ("native_dac", "base")}
        for p in spec.POOLS:
            contrasts[f"{p}_alone - base"] = (f"{p}_alone", "base")
        out[seed] = {"best_hidden_pool_clean_selected": best, "strongest_output_control_target_informed": strongest, "bootstrap": {}, "bootstrap_clean": {}, "oracle_unions": {}}
        for k, (a, b) in contrasts.items():
            out[seed]["bootstrap"][k] = boot(correct(ps, a, CELLS) - correct(ps, b, CELLS), seed)
            out[seed]["bootstrap_clean"][k] = boot(correct(ps, a, ["clean"]) - correct(ps, b, ["clean"]), seed)
        sets = {"base": correct(ps, "base", CELLS)}
        # oracle unions on the 12 cells: per (image, cell) any-correct
        def anyc(names):
            return float(np.mean([np.any([ps[c][f"pred__{n}"].astype(int) == ps[c]["labels"].astype(int) for n in names], axis=0) for c in CELLS])) * 100
        base_acc = 100 * float(np.mean([ps[c]["base_pred"].astype(int) == ps[c]["labels"].astype(int) for c in CELLS]))
        o = {"base_acc": base_acc}
        for p in spec.POOLS:
            o[f"base+{p}_alone_pp"] = anyc(["base", f"{p}_alone"]) - base_acc
        o["base+out_alone_pp"] = anyc(["base", "out_alone"]) - base_acc
        o["base+vs_pp"] = anyc(["base", "vs"]) - base_acc
        o["base+ms_pp"] = anyc(["base", "ms"]) - base_acc
        o["base+all_hidden_alone_pp"] = anyc(["base"] + [f"{p}_alone" for p in spec.POOLS]) - base_acc
        o["base+out_alone+vs+ms_pp"] = anyc(["base", "out_alone", "vs", "ms"]) - base_acc
        o["base+hidden+output_pp"] = anyc(["base", "out_alone", "vs", "ms"] + [f"{p}_alone" for p in spec.POOLS]) - base_acc
        o["hidden_increment_over_output_oracle_pp"] = o["base+hidden+output_pp"] - o["base+out_alone+vs+ms_pp"]
        o["note"] = "evaluation-only label oracle over the specified frozen candidate set; NOT a bound on a feasible hidden-informed gate's advantage over a feasible output-only method"
        out[seed]["oracle_unions"] = o
    json.dump(out, open("results/atlas/report/analysis_extra.json", "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
