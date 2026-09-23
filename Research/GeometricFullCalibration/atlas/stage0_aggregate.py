"""Stage 0c aggregation: pools out-of-fold predictions, computes the primary/secondary
contrasts, grouped paired bootstrap, and the interpretation table
(docs/stage0_execution_spec.md Sections 4-6).

    python -m atlas.stage0_aggregate
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import numpy as np

from utils.unified_metrics import evaluate_all
from . import spec, stage0_data, stage0_fit
from .stage0c_run import REGIMES, OUT_ROOT
from .stage0_shuffle import REGIMES as SHUFFLE_REGIMES, OUT_ROOT as SHUFFLE_ROOT, FOLD as SHUFFLE_FOLD

B_BOOT = 2000
BOOT_SEED = 20261010
OUT = "results/stage0/report"
LAMBDA_EDGES = (stage0_fit.LAMBDA_GRID[0], stage0_fit.LAMBDA_GRID[-1])


def group_id_for_bootstrap() -> np.ndarray:
    n = 10000
    gid = np.arange(n)
    for entry in stage0_data.test_duplicate_groups().values():
        idxs = sorted(entry["indices"])
        gid[idxs] = idxs[0]
    return gid


def pool_oof(seed: int, regime: str) -> Dict[str, np.ndarray]:
    """Pool the 5 folds' held-out predictions into one (10000,) / (10000,100) array per key."""
    per_fold = []
    for f in range(5):
        path = f"{OUT_ROOT}/seed{seed}/{regime}/fold{f}.npz"
        if not os.path.exists(path):
            return None
        per_fold.append(np.load(path, allow_pickle=True))

    n = 10000
    pooled: Dict[str, np.ndarray] = {}
    covered = np.zeros(n, dtype=bool)
    for cond in spec.CONDITIONS:
        labels_full = np.full(n, -1, dtype=np.int64)
        probs_qz_full = np.full((n, spec.NUM_CLASSES), np.nan, dtype=np.float64)
        probs_qzp_full = np.full((n, spec.NUM_CLASSES), np.nan, dtype=np.float64)
        for d in per_fold:
            idx = d["test_idx"]
            labels_full[idx] = d[f"labels__{cond}"]
            probs_qz_full[idx] = d[f"probs__q_Z__{cond}"]
            probs_qzp_full[idx] = d[f"probs__q_ZP__{cond}"]
            covered[idx] = True
        pooled[f"labels__{cond}"] = labels_full
        pooled[f"probs__q_Z__{cond}"] = probs_qz_full
        pooled[f"probs__q_ZP__{cond}"] = probs_qzp_full
    assert covered.all(), f"seed{seed}/{regime}: {(~covered).sum()} images never held out across the 5 folds"
    return pooled


def per_image_correct(pooled, arm, cond) -> np.ndarray:
    probs = pooled[f"probs__{arm}__{cond}"]
    labels = pooled[f"labels__{cond}"]
    return (probs.argmax(1) == labels).astype(np.float64)


def per_image_negloglik(pooled, arm, cond, eps=1e-12) -> np.ndarray:
    probs = pooled[f"probs__{arm}__{cond}"]
    labels = pooled[f"labels__{cond}"]
    idx = np.arange(len(labels))
    return -np.log(np.clip(probs[idx, labels], eps, 1.0))


def grouped_bootstrap(diff_per_image: np.ndarray, gid: np.ndarray, seed: int, valid_mask: np.ndarray = None) -> Dict:
    if valid_mask is not None:
        diff_per_image = diff_per_image.copy()
        diff_per_image[~valid_mask] = np.nan
    reps = np.unique(gid)
    group_val = np.array([np.nanmean(diff_per_image[gid == g]) for g in reps])
    group_val = group_val[~np.isnan(group_val)]
    rng = np.random.default_rng(seed)
    n = len(group_val)
    idx = rng.integers(0, n, (B_BOOT, n))
    b = group_val[idx].mean(1)
    return {"estimate": float(np.nanmean(diff_per_image)), "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))]}


def cell_metrics(pooled, arm, cond) -> Dict:
    probs = pooled[f"probs__{arm}__{cond}"]
    labels = pooled[f"labels__{cond}"]
    m = evaluate_all(probs, labels)
    return {"accuracy": m["accuracy"], "nll": m["nll"], "brier": m["brier"]}


def grouped_bootstrap_diff_of_diffs(diff_a: np.ndarray, diff_b: np.ndarray, gid: np.ndarray, seed: int) -> Dict:
    """CI for (regime A's per-image effect) - (regime B's per-image effect), via ONE shared
    resampling draw over image groups -- the "joint paired resamples" the frozen spec (Section 5)
    requires for derived gaps, rather than subtracting two independently-bootstrapped CIs."""
    return grouped_bootstrap(diff_a - diff_b, gid, seed=seed)


def primary_and_secondary(gid: np.ndarray) -> Dict:
    out = {}
    diff_arrays: Dict[int, Dict[str, Dict[str, np.ndarray]]] = {}
    for seed in spec.DEV_SEEDS:
        pooled = {r: pool_oof(seed, r) for r in REGIMES}
        missing = [r for r in REGIMES if pooled[r] is None]
        if missing:
            out[str(seed)] = {"status": "incomplete", "missing_regimes": missing}
            continue

        seed_out = {"per_regime_per_cell": {}, "macro12": {}}
        diff_arrays[seed] = {}
        for regime in REGIMES:
            p = pooled[regime]
            per_cell = {}
            for cond in spec.CELLS:
                per_cell[cond] = {"q_Z": cell_metrics(p, "q_Z", cond), "q_ZP": cell_metrics(p, "q_ZP", cond)}
            per_cell["clean"] = {"q_Z": cell_metrics(p, "q_Z", "clean"), "q_ZP": cell_metrics(p, "q_ZP", "clean")}
            seed_out["per_regime_per_cell"][regime] = per_cell

            diff_macro12 = np.mean([per_image_correct(p, "q_ZP", c) - per_image_correct(p, "q_Z", c) for c in spec.CELLS], axis=0)
            diff_clean = per_image_correct(p, "q_ZP", "clean") - per_image_correct(p, "q_Z", "clean")
            diff_arrays[seed][regime] = {"macro12": diff_macro12, "clean": diff_clean}

            macro_delta_acc_pp = float(np.mean([per_cell[c]["q_ZP"]["accuracy"] - per_cell[c]["q_Z"]["accuracy"] for c in spec.CELLS])) * 100
            boot = grouped_bootstrap(diff_macro12, gid, seed=BOOT_SEED + seed)
            seed_out["macro12"][regime] = {"delta_acc_pp": macro_delta_acc_pp, "bootstrap_pp": {"estimate": boot["estimate"] * 100, "ci95_pp": [x * 100 for x in boot["ci95"]]}}

        out[str(seed)] = seed_out

    # secondary contrasts: 8k vs 2.5k, 12-view vs 1-view, recoverability gap, clean increment.
    # Each derived gap is bootstrapped jointly (one shared resampling draw over image groups on
    # the two regimes' per-image diff arrays), not by subtracting two independent CIs.
    secondary = {}
    for seed in spec.DEV_SEEDS:
        if out[str(seed)].get("status") == "incomplete":
            secondary[str(seed)] = {"status": "incomplete"}
            continue
        da = diff_arrays[seed]

        def gap(name_a, key_a, name_b, key_b, salt):
            point = out[str(seed)]["macro12"][name_a]["delta_acc_pp"] - out[str(seed)]["macro12"][name_b]["delta_acc_pp"] \
                if key_a == "macro12" and key_b == "macro12" else \
                float(np.mean(da[name_a][key_a]) - np.mean(da[name_b][key_b])) * 100
            boot = grouped_bootstrap_diff_of_diffs(da[name_a][key_a], da[name_b][key_b], gid, seed=BOOT_SEED + salt + seed)
            return {"estimate_pp": point, "ci95_pp": [x * 100 for x in boot["ci95"]]}

        secondary[str(seed)] = {
            "independent_image_effect_T12": gap("T-8k12", "macro12", "T-2.5k12", "macro12", 100),
            "independent_image_effect_T1": gap("T-8k1", "macro12", "T-2.5k1", "macro12", 200),
            "view_effect_8k": gap("T-8k12", "macro12", "T-8k1", "macro12", 300),
            "view_effect_2.5k": gap("T-2.5k12", "macro12", "T-2.5k1", "macro12", 400),
            "recoverability_gap_8k": gap("T-8k1", "macro12", "S-8k1", "macro12", 500),
            "recoverability_gap_2.5k": gap("T-2.5k1", "macro12", "S-2.5k1", "macro12", 600),
            "S_clean_increment_pp_8k": {"estimate_pp": float(np.mean(da["S-8k1"]["clean"])) * 100,
                 "ci95_pp": [x * 100 for x in grouped_bootstrap(da["S-8k1"]["clean"], gid, seed=BOOT_SEED + 700 + seed)["ci95"]]},
            "S_clean_increment_pp_2.5k": {"estimate_pp": float(np.mean(da["S-2.5k1"]["clean"])) * 100,
                 "ci95_pp": [x * 100 for x in grouped_bootstrap(da["S-2.5k1"]["clean"], gid, seed=BOOT_SEED + 800 + seed)["ci95"]]},
        }
    return {"primary_secondary_by_seed": out, "secondary_contrasts": secondary}


def interpretation(result: Dict) -> Dict:
    seeds = [str(s) for s in spec.DEV_SEEDS]
    if any(result["primary_secondary_by_seed"][s].get("status") == "incomplete" for s in seeds):
        return {"status": "incomplete", "note": "not all regimes/folds are written yet"}
    deltas = {s: result["primary_secondary_by_seed"][s]["macro12"]["T-8k12"]["delta_acc_pp"] for s in seeds}
    cis = {s: result["primary_secondary_by_seed"][s]["macro12"]["T-8k12"]["bootstrap_pp"]["ci95_pp"] for s in seeds}
    material = all(deltas[s] >= 0.5 for s in seeds) and all(not (cis[s][0] <= 0 <= cis[s][1]) for s in seeds)
    below_scale = all(cis[s][1] < 0.2 for s in seeds)
    if material:
        verdict = "material_development_increment"
    elif below_scale:
        verdict = "below_practical_scale"
    else:
        verdict = "small_uncertain_or_checkpoint_dependent"
    return {"verdict": verdict, "delta_T_pp_by_seed": deltas, "ci95_pp_by_seed": cis}


def convergence_and_lambda_edge_report() -> Dict:
    """Every fit's convergence/retry flag and whether the selected lambda hit a grid edge
    (requested check: a lambda pinned at the grid boundary was the residual study's failure
    signature -- see docs/stage0_execution_spec.md Section 3)."""
    rows = []
    for seed in spec.DEV_SEEDS:
        for regime in REGIMES:
            for fold in range(5):
                path = f"{OUT_ROOT}/seed{seed}/{regime}/fold{fold}.npz"
                if not os.path.exists(path):
                    continue
                d = np.load(path, allow_pickle=True)
                for arm in ("q_Z", "q_ZP"):
                    lam = float(d[f"meta__{arm}__selected_lambda"])
                    rows.append({
                        "seed": seed, "regime": regime, "fold": fold, "arm": arm,
                        "selected_lambda": lam, "at_grid_edge": lam in LAMBDA_EDGES,
                        "converged": bool(d[f"meta__{arm}__converged"]), "retried": bool(d[f"meta__{arm}__retried"]),
                    })
    n = len(rows)
    n_edge = sum(r["at_grid_edge"] for r in rows)
    n_unconverged = sum(not r["converged"] for r in rows)
    n_retried = sum(r["retried"] for r in rows)
    by_lambda = {}
    for r in rows:
        by_lambda[r["selected_lambda"]] = by_lambda.get(r["selected_lambda"], 0) + 1
    return {"n_fits": n, "n_at_grid_edge": n_edge, "frac_at_grid_edge": n_edge / n if n else None,
            "n_unconverged": n_unconverged, "n_retried": n_retried,
            "selected_lambda_histogram": by_lambda, "rows": rows}


MEMO_SANITY_REGIME = "T-8k12"  # the memo's Section 4 sanity control: T-8k12 only, fold 0, per checkpoint


def shuffle_control_report(gid: np.ndarray) -> Dict:
    """Shuffled-P control vs the REAL (unshuffled) q_Z, matched to the same fold-0 held-out
    rows. Reported and printed BEFORE the primary contrast, per review feedback (docs/
    stage0_execution_spec.md Section 5): a positive gain here is an audit trigger, not
    automatic evidence of leakage.

    Deviation 1 (docs/stage0_execution_spec.md Section 7): the memo's own sanity control is
    T-8k12 only, fold 0, per checkpoint (2 real fits). This also ran T-2.5k12/T-8k1 as extra,
    non-memo-specified shuffled checks -- labelled separately below so the memo-matching gate
    isn't diluted by the additional ones."""
    out = {}
    for seed in spec.DEV_SEEDS:
        for regime in SHUFFLE_REGIMES:
            shuf_path = f"{SHUFFLE_ROOT}/seed{seed}_{regime}.npz"
            real_path = f"{OUT_ROOT}/seed{seed}/{regime}/fold{SHUFFLE_FOLD}.npz"
            if not (os.path.exists(shuf_path) and os.path.exists(real_path)):
                out[f"{seed}_{regime}"] = {"status": "incomplete"}
                continue
            shuf = np.load(shuf_path, allow_pickle=True)
            real = np.load(real_path, allow_pickle=True)
            assert np.array_equal(shuf["test_idx"], real["test_idx"]), "shuffle control / real fold-0 test rows misaligned"

            diffs = []
            for cond in spec.CELLS:
                y = shuf[f"labels__{cond}"]
                acc_shuf = (shuf[f"probs__q_ZP_shuffled__{cond}"].argmax(1) == y).astype(np.float64)
                acc_real_qz = (real[f"probs__q_Z__{cond}"].argmax(1) == real[f"labels__{cond}"]).astype(np.float64)
                diffs.append(acc_shuf - acc_real_qz)
            diff_per_image_local = np.mean(diffs, axis=0)
            gid_local = gid[shuf["test_idx"]]
            full_diff = np.full(10000, np.nan)
            full_diff[shuf["test_idx"]] = diff_per_image_local
            boot = grouped_bootstrap(full_diff, gid, seed=BOOT_SEED + 500 + seed, valid_mask=~np.isnan(full_diff))
            gain_pp = boot["estimate"] * 100
            out[f"{seed}_{regime}"] = {
                "shuffled_qZP_minus_real_qZ_macro12_pp": gain_pp,
                "ci95_pp": [x * 100 for x in boot["ci95"]],
                "audit_trigger_ge_0.2pp": gain_pp >= 0.2,
                "converged": bool(shuf["converged"]), "retried": bool(shuf["retried"]),
                "selected_lambda": float(shuf["selected_lambda"]),
                "memo_specified_sanity_gate": regime == MEMO_SANITY_REGIME,
            }
    memo_gate_entries = {k: v for k, v in out.items() if v.get("memo_specified_sanity_gate")}
    memo_gate_passed = all(not v["audit_trigger_ge_0.2pp"] for v in memo_gate_entries.values()) if memo_gate_entries else None
    return {"per_seed_regime": out, "memo_specified_gate_regime": MEMO_SANITY_REGIME,
            "memo_specified_gate_passed": memo_gate_passed}


def evaluation_only_exclusion_sensitivity(gid: np.ndarray) -> Dict:
    """Primary Delta_T (T-8k12) recomputed with the test-vs-train pixel-duplicate images dropped
    from EVALUATION only -- no refit (docs/stage0_execution_spec.md Section 2/7; memo: "the
    primary keeps them, and a sensitivity analysis drops them from evaluation... needs no extra
    fits")."""
    excluded = set(stage0_data.test_vs_train_duplicate_test_indices())
    out = {"excluded_test_indices": sorted(excluded), "n_excluded": len(excluded)}
    for seed in spec.DEV_SEEDS:
        pooled = pool_oof(seed, "T-8k12")
        if pooled is None:
            out[str(seed)] = {"status": "incomplete"}
            continue
        diff = np.mean([per_image_correct(pooled, "q_ZP", c) - per_image_correct(pooled, "q_Z", c) for c in spec.CELLS], axis=0)
        valid_mask = np.ones(10000, dtype=bool)
        valid_mask[list(excluded)] = False
        boot = grouped_bootstrap(diff, gid, seed=BOOT_SEED + 900 + seed, valid_mask=valid_mask)
        out[str(seed)] = {"delta_acc_pp": boot["estimate"] * 100, "ci95_pp": [x * 100 for x in boot["ci95"]],
                           "n_evaluated_images": int(valid_mask.sum())}
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    gid = group_id_for_bootstrap()

    print("=== 1. Shuffled-P control (read FIRST, before the primary contrast) ===")
    shuffle_report = shuffle_control_report(gid)
    print(json.dumps(shuffle_report, indent=1))

    print("=== 2. Convergence / lambda-grid-edge audit ===")
    conv_report = convergence_and_lambda_edge_report()
    print(json.dumps({k: v for k, v in conv_report.items() if k != "rows"}, indent=1))

    print("=== 3. Primary + secondary contrasts ===")
    result = primary_and_secondary(gid)
    result["interpretation"] = interpretation(result)
    result["shuffle_control"] = shuffle_report
    result["convergence_and_lambda_edge_audit"] = conv_report
    result["evaluation_only_exclusion_sensitivity"] = evaluation_only_exclusion_sensitivity(gid)
    json.dump(result, open(f"{OUT}/stage0c_aggregate.json", "w"), indent=1)
    print(json.dumps(result["interpretation"], indent=1))
    print("wrote", f"{OUT}/stage0c_aggregate.json")


if __name__ == "__main__":
    main()
