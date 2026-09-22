"""Stage 0b -- descriptive evaluation of the 12 already-frozen clean-trained GAP probes
(docs/stage0_execution_spec.md, controlling prompt Section 8/0b). No new fitting: applies each
probe's own frozen temperature exactly once to its cached pre-temperature logits.

    python -m atlas.stage0b
"""
from __future__ import annotations

import json
import os

import numpy as np

from utils.unified_metrics import evaluate_all
from . import spec, stage0_data

OUT = "results/stage0/report"


def np_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = []
    for seed in spec.DEV_SEEDS:
        temps = stage0_data.probe_temperatures_all12(seed)
        layer_names = stage0_data.candidate_layer_names(seed)
        for cell in spec.CONDITIONS:
            d = stage0_data.load_cell(seed, cell)
            labels = d["labels"]
            base_probs = np_softmax(d["z"])
            base_m = evaluate_all(base_probs, labels)

            pilot = np.load(f"results/layer_pilot/checkpoint_seed{seed}/{cell}/per_sample.npz")
            raw = pilot["raw__probe_logits"].astype(np.float64)  # (N,12,100), pre-temperature, float16-quantized at save time
            for li, lname in enumerate(layer_names):
                logits_l = raw[:, li, :]
                probs_l = np_softmax(logits_l / temps[li])
                m = evaluate_all(probs_l, labels)
                rows.append({
                    "seed": seed, "cell": cell, "layer_index": li, "layer_name": lname,
                    "temperature": float(temps[li]),
                    "accuracy": m["accuracy"], "delta_accuracy_pp_vs_base": 100.0 * (m["accuracy"] - base_m["accuracy"]),
                    "nll": m["nll"], "brier": m["brier"], "top_label_ece": m["top_label_ece"],
                    "base_accuracy": base_m["accuracy"], "base_nll": base_m["nll"], "base_brier": base_m["brier"],
                })

    json.dump(rows, open(f"{OUT}/stage0b_layer_probe_eval.json", "w"), indent=1)

    # macro means over the 12 corruption cells (clean reported separately), by seed/layer
    macro = {}
    for seed in spec.DEV_SEEDS:
        by_layer = {}
        for li in range(12):
            cell_rows = [r for r in rows if r["seed"] == seed and r["layer_index"] == li and r["cell"] in spec.CELLS]
            clean_row = next(r for r in rows if r["seed"] == seed and r["layer_index"] == li and r["cell"] == "clean")
            by_layer[cell_rows[0]["layer_name"]] = {
                "macro12_delta_acc_pp": float(np.mean([r["delta_accuracy_pp_vs_base"] for r in cell_rows])),
                "macro12_nll": float(np.mean([r["nll"] for r in cell_rows])),
                "macro12_brier": float(np.mean([r["brier"] for r in cell_rows])),
                "clean_delta_acc_pp": clean_row["delta_accuracy_pp_vs_base"],
            }
        macro[str(seed)] = by_layer
    json.dump(macro, open(f"{OUT}/stage0b_macro_by_layer.json", "w"), indent=1)

    # descriptive eligibility flag (NOT a Direction-2 authorization): >1pp on >=2 families in both checkpoints
    families = spec.DEV_CORRUPTIONS
    eligibility = {}
    for li in range(12):
        lname = None
        per_seed_family_hits = {}
        for seed in spec.DEV_SEEDS:
            hits = []
            for fam in families:
                fam_rows = [r for r in rows if r["seed"] == seed and r["layer_index"] == li and r["cell"].startswith(fam)]
                lname = fam_rows[0]["layer_name"]
                fam_mean = float(np.mean([r["delta_accuracy_pp_vs_base"] for r in fam_rows]))
                if fam_mean > 1.0:
                    hits.append(fam)
            per_seed_family_hits[str(seed)] = hits
        eligible = all(len(per_seed_family_hits[str(s)]) >= 2 for s in spec.DEV_SEEDS)
        eligibility[lname] = {"eligible_descriptive_only": eligible, "per_seed_families_over_1pp": per_seed_family_hits}
    json.dump(eligibility, open(f"{OUT}/stage0b_eligibility_flag.json", "w"), indent=1)

    print(json.dumps(macro, indent=1))
    print("eligibility (descriptive only, not a continuation authorization):")
    print(json.dumps(eligibility, indent=1))
    print("wrote", f"{OUT}/stage0b_layer_probe_eval.json", f"{OUT}/stage0b_macro_by_layer.json", f"{OUT}/stage0b_eligibility_flag.json")


if __name__ == "__main__":
    main()
