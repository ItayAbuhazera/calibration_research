"""Audit of the five frozen atlas hypotheses: exact statements, thresholds, mechanical outcome, and a descriptive interval where one can be computed
(paired image-level bootstrap over the 10 000 test IDs on the atlas cd per-sample predictions; NOT the frozen test rows, which were the 2 000-image atlas subset).
Classification: threshold_failed (mechanical rule not met) / estimate_uncertain (interval includes both the threshold-relevant and null values, or seed-dependent sign) /
evidence_against_threshold_magnitude (descriptive 95% interval excludes the frozen magnitude)."""
import json
import numpy as np
from . import common, data, spec

SPEC_LINES = {}


def macro_acc(ps, name):
    return np.mean([ps[c][f"pred__{name}"].astype(int) == ps[c]["labels"].astype(int) for c in spec.CELLS], axis=0)


def boot(diff, seed, B=2000):
    rng = np.random.default_rng(20261011 + seed); n = len(diff); b = diff[rng.integers(0, n, (B, n))].mean(1)
    return {"estimate_pp": 100 * float(diff.mean()), "ci95_pp": [100 * float(np.quantile(b, .025)), 100 * float(np.quantile(b, .975))]}


def main():
    lines = [l.strip() for l in open("docs/atlas_program_spec.md") if l.startswith("* **H-")]
    hyp = {l.split("**")[1].split(" ")[0]: l for l in lines[:5]}
    ax = json.load(open("results/atlas/report/analysis_extra.json"))
    out = {"statements_quoted_from_docs/atlas_program_spec.md": hyp, "per_hypothesis": {}}
    for seed in spec.DEV_SEEDS:
        root = data.seed_dir(seed)
        ps = {c: np.load(f"{root}/cd/per_sample_{c}.npz") for c in spec.CELLS}
        g, gr, sp = macro_acc(ps, "gap_alone"), macro_acc(ps, "grid2_alone"), macro_acc(ps, "spp_alone")
        out["per_hypothesis"].setdefault("H-A", {})[seed] = {"threshold_pp": 0.5, 
                                                         "descriptive_full10k_grid2_minus_gap": boot(gr - g, seed), "descriptive_full10k_spp_minus_gap": boot(sp - g, seed)}
        e = ax[str(seed)]["bootstrap"]
        out["per_hypothesis"].setdefault("H-D", {})[seed] = {"threshold_pp": 0.10, "F1_minus_F0_pp": e["gap_F1 - gap_F0"], "hidden_F1_minus_out_F1_pp": e["gap_F1 - out_F1"]}
        out["per_hypothesis"].setdefault("H-E", {})[seed] = {"thresholds_pp": {"over_base": 0.5, "over_strongest_output_control": 0.25}, "hidden_F1_minus_base": e["gap_F1 - base"],
                                                         "hidden_F1_minus_strongest_output_control": e.get(f"gap_F1 - {ax[str(seed)]['strongest_output_control_target_informed']}", e.get("gap_F1 - out_F1")),
                                                         "note": "strongest output control was chosen with target information in the atlas report; interval is conditional on that choice"}
    hy = json.load(open("results/atlas/report/summary.json"))["hypotheses"]
    for k in ("H-A", "H-B", "H-C", "H-D", "H-E"):
        out["per_hypothesis"].setdefault(k, {})["frozen_mechanical_result"] = hy[k]
    out["classification"] = {
        "H-A": "threshold_failed; frozen point estimates +0.11/-0.03 pp against a 0.5 pp requirement; descriptive full-10k intervals (see per_hypothesis) exclude 0.5 pp => evidence against the frozen magnitude for the CLEAN-SELECTED layer4 candidates only (says nothing about deep-layer3 spatial pooling, which was not a candidate)",
        "H-B": "threshold_failed but estimate_uncertain: unique repairs minus harms is -70 in seed 2 and +80 in seed 4 (opposite signs; no interval computed) => seed-dependent, not evidence against complementarity",
        "H-C": "threshold_failed at the margin (raw_l2 grid2 +0.48 pp vs 0.5 pp in seed 4, <=0 elsewhere) => estimate_uncertain for raw_l2; the frozen Mahalanobis ESTIMATOR was strongly harmful on high-dimensional hidden features (evidence against that estimator, not against covariance structure)",
        "H-D": "threshold_failed; seed 2 F1-F0 interval upper bound below 0.10 pp => evidence against the frozen magnitude at that candidate/budget; seed 4 gates made identical decisions (F1-F0 = 0 exactly) => no information",
        "H-E": "threshold_failed; descriptive intervals for both targets lie below the frozen magnitudes for the layer4 clean-selected candidate; conditional on the target-informed choice of the strongest output control"}
    common.atomic_json("results/fixed_gate/report/hypothesis_audit.json", out)
    print(json.dumps(out["per_hypothesis"]["H-A"], indent=1))


if __name__ == "__main__":
    main()
