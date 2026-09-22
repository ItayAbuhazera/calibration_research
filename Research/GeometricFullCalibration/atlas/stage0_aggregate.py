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


def primary_and_secondary(gid: np.ndarray) -> Dict:
    out = {}
    for seed in spec.DEV_SEEDS:
        pooled = {r: pool_oof(seed, r) for r in REGIMES}
        missing = [r for r in REGIMES if pooled[r] is None]
        if missing:
            out[str(seed)] = {"status": "incomplete", "missing_regimes": missing}
            continue

        seed_out = {"per_regime_per_cell": {}, "macro12": {}}
        for regime in REGIMES:
            p = pooled[regime]
            per_cell = {}
            for cond in spec.CELLS:
                per_cell[cond] = {"q_Z": cell_metrics(p, "q_Z", cond), "q_ZP": cell_metrics(p, "q_ZP", cond)}
            per_cell["clean"] = {"q_Z": cell_metrics(p, "q_Z", "clean"), "q_ZP": cell_metrics(p, "q_ZP", "clean")}
            seed_out["per_regime_per_cell"][regime] = per_cell

            diff_acc_by_cell = np.mean([[per_image_correct(p, "q_ZP", c) - per_image_correct(p, "q_Z", c)] for c in spec.CELLS], axis=0)[0]
            macro_delta_acc_pp = float(np.mean([per_cell[c]["q_ZP"]["accuracy"] - per_cell[c]["q_Z"]["accuracy"] for c in spec.CELLS])) * 100
            boot = grouped_bootstrap(
                np.mean([per_image_correct(p, "q_ZP", c) - per_image_correct(p, "q_Z", c) for c in spec.CELLS], axis=0),
                gid, seed=BOOT_SEED + seed)
            seed_out["macro12"][regime] = {"delta_acc_pp": macro_delta_acc_pp, "bootstrap_pp": {"estimate": boot["estimate"] * 100, "ci95_pp": [x * 100 for x in boot["ci95"]]}}

        out[str(seed)] = seed_out

    # secondary contrasts: 8k vs 2.5k, 12-view vs 1-view, recoverability gap, clean increment
    secondary = {}
    for seed in spec.DEV_SEEDS:
        if out[str(seed)].get("status") == "incomplete":
            secondary[str(seed)] = {"status": "incomplete"}
            continue
        m = out[str(seed)]["macro12"]
        secondary[str(seed)] = {
            "independent_image_effect_T12": m["T-8k12"]["delta_acc_pp"] - m["T-2.5k12"]["delta_acc_pp"],
            "independent_image_effect_T1": m["T-8k1"]["delta_acc_pp"] - m["T-2.5k1"]["delta_acc_pp"],
            "view_effect_8k": m["T-8k12"]["delta_acc_pp"] - m["T-8k1"]["delta_acc_pp"],
            "view_effect_2.5k": m["T-2.5k12"]["delta_acc_pp"] - m["T-2.5k1"]["delta_acc_pp"],
            "recoverability_gap_8k": m["T-8k1"]["delta_acc_pp"] - m["S-8k1"]["delta_acc_pp"],
            "recoverability_gap_2.5k": m["T-2.5k1"]["delta_acc_pp"] - m["S-2.5k1"]["delta_acc_pp"],
            "S_clean_increment_pp_8k": out[str(seed)]["per_regime_per_cell"]["S-8k1"]["clean"]["q_ZP"]["accuracy"] * 100
                                       - out[str(seed)]["per_regime_per_cell"]["S-8k1"]["clean"]["q_Z"]["accuracy"] * 100,
            "S_clean_increment_pp_2.5k": out[str(seed)]["per_regime_per_cell"]["S-2.5k1"]["clean"]["q_ZP"]["accuracy"] * 100
                                         - out[str(seed)]["per_regime_per_cell"]["S-2.5k1"]["clean"]["q_Z"]["accuracy"] * 100,
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


def shuffle_control_report(gid: np.ndarray) -> Dict:
    """Shuffled-P control vs the REAL (unshuffled) q_Z, matched to the same fold-0 held-out
    rows. Reported and printed BEFORE the primary contrast, per review feedback (docs/
    stage0_execution_spec.md Section 5): a positive gain here is an audit trigger, not
    automatic evidence of leakage."""
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
            }
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
    json.dump(result, open(f"{OUT}/stage0c_aggregate.json", "w"), indent=1)
    print(json.dumps(result["interpretation"], indent=1))
    print("wrote", f"{OUT}/stage0c_aggregate.json")


if __name__ == "__main__":
    main()
