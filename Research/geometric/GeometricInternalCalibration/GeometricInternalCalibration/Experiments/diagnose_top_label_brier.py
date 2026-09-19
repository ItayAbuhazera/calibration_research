"""Diagnostic: top-label Brier from existing per-sample data, for RGCL and RGCC only.

Reads npz files under calibration_comparison_mce_adaptive/, computes per-seed
top-label Brier, aggregates with 95% CI, and joins against the ECE / multi-class
Brier already stored in per-seed JSON results. No experiments are run.

Top-label Brier = mean((max_prob - correct)^2), reported in percentage points.

Outputs:
  top_label_brier_long.csv  per-(dataset, model, method) aggregates
  top_label_brier_diag.csv  joined with ECE and multi-class Brier for comparison
  stdout                    ranking agreement summary and inventory
"""

from __future__ import annotations

import glob
import json
import os
import re

import numpy as np
import pandas as pd

ROOT = "calibration_comparison_mce_adaptive"
TRAINING_METHOD = "baseline_cross_entropy"

# Maps the per-sample saved method_name to display name and JSON section.
# Both fields come from the source code in run_global_random_calibration and
# run_coordinate_calibration.
METHOD_MAP = {
    "geometric_(combined_1_main_blocks)": {
        "display": "RGCL",
        "json_section": "global_random_separation",
    },
    "coordinate": {
        "display": "RGCC",
        "json_section": "coordinate_sampling_separation",
    },
}

BASELINE_METHOD_MAP = {
    "standard_baselines.uncalibrated": "Uncal",
    "standard_baselines.temperature_scaling": "TS",
    "standard_baselines.platt_scaling": "Platt",
    "standard_baselines.isotonic_toplabel": "Isotonic",
    "standard_baselines.beta_calibration": "Beta",
    "standard_baselines.density_aware_calibration": "DAC",
}

DISPLAY_TO_JSON_SECTION = {
    **{v["display"]: v["json_section"] for v in METHOD_MAP.values()},
    **{display: section for section, display in BASELINE_METHOD_MAP.items()},
}

# Filename written by save_per_sample_data:
# {model}_{dataset}_{training_method}_seed{seed}_{method}_per_sample.npz
# Note: cifar100 must come before cifar10 in the alternation so the longer
# match wins. Method names can contain parens and underscores, hence .+ at end.
FILENAME_RE = re.compile(
    r"^(?P<model>.+?)_"
    r"(?P<dataset>cifar100|cifar10|tiny_imagenet)_"
    r"(?P<training_method>baseline_cross_entropy|augmix)_"
    r"seed(?P<seed>\d+)_"
    r"(?P<method>.+)_per_sample\.npz$"
)

JSON_FILENAME_RE = re.compile(
    r"^ablation_(?P<training_method>.+?)_"
    r"(?P<dataset>cifar100|cifar10|tiny_imagenet)_"
    r"(?P<model>.+)_"
    r"seed(?P<seed>\d+)\.json$"
)


def discover_npz_files(root: str):
    pattern = os.path.join(root, "**", "per_sample_data", "*_per_sample.npz")
    for path in glob.iglob(pattern, recursive=True):
        fname = os.path.basename(path)
        m = FILENAME_RE.match(fname)
        if m is None:
            continue
        meta = m.groupdict()
        meta["seed"] = int(meta["seed"])
        meta["path"] = path
        yield meta


def discover_json_files(root: str):
    pattern = os.path.join(root, "ablation_*.json")
    for path in glob.iglob(pattern):
        fname = os.path.basename(path)
        m = JSON_FILENAME_RE.match(fname)
        if m is None:
            continue
        meta = m.groupdict()
        if meta["training_method"] != TRAINING_METHOD:
            continue
        meta["seed"] = int(meta["seed"])
        meta["path"] = path
        yield meta


def compute_top_label_brier_pct(npz_path: str) -> float:
    data = np.load(npz_path)
    max_probs = data["max_probs"].astype(np.float64)
    correct = data["correct"].astype(np.float64)
    if max_probs.shape != correct.shape:
        raise ValueError(
            f"shape mismatch in {npz_path}: "
            f"max_probs {max_probs.shape} vs correct {correct.shape}"
        )
    return float(np.mean((max_probs - correct) ** 2)) * 100.0


def aggregate(group: pd.DataFrame) -> pd.Series:
    vals = group["top_label_brier_pct"].values
    n = int(len(vals))
    mean = float(np.mean(vals))
    if n > 1:
        ci95 = 1.96 * float(np.std(vals, ddof=1)) / np.sqrt(n)
    else:
        ci95 = float("nan")
    return pd.Series({"n_seeds": n, "tlb_mean_pct": mean, "tlb_ci95_pct": ci95})


def load_json(root: str, dataset: str, model: str, seed: int) -> dict | None:
    pattern = os.path.join(
        root, f"ablation_{TRAINING_METHOD}_{dataset}_{model}_seed{seed}.json"
    )
    matches = glob.glob(pattern)
    if not matches:
        return None
    try:
        with open(matches[0]) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def get_nested(data: dict, dotted_path: str):
    cur = data
    for part in dotted_path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def get_section_value(data: dict, section: str, key: str):
    if not data:
        return None
    sec = get_nested(data, section)
    if not isinstance(sec, dict):
        return None
    val = sec.get(key)
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def main() -> None:
    if not os.path.isdir(ROOT):
        print(f"ROOT not found: {ROOT}")
        return

    # 1. Inventory and compute top-label Brier per file
    rows = []
    skipped = {}
    for meta in discover_npz_files(ROOT):
        method = meta["method"]
        if method not in METHOD_MAP:
            skipped[method] = skipped.get(method, 0) + 1
            continue
        try:
            tlb = compute_top_label_brier_pct(meta["path"])
        except Exception as e:
            print(f"WARN: skipping {meta['path']}: {e}")
            continue
        rows.append(
            {
                "dataset": meta["dataset"],
                "model": meta["model"],
                "method": METHOD_MAP[method]["display"],
                "seed": meta["seed"],
                "top_label_brier_pct": tlb,
                "source": "npz",
            }
        )

    # Baseline top-label Brier is written directly to JSON by
    # --baselines-only --compute-brier-top-label jobs.
    for meta in discover_json_files(ROOT):
        data = load_json(ROOT, meta["dataset"], meta["model"], meta["seed"])
        if data is None:
            continue
        for section, display in BASELINE_METHOD_MAP.items():
            tlb = get_section_value(data, section, "top_label_brier")
            if tlb is None:
                continue
            rows.append(
                {
                    "dataset": meta["dataset"],
                    "model": meta["model"],
                    "method": display,
                    "seed": meta["seed"],
                    "top_label_brier_pct": tlb * 100.0,
                    "source": "json",
                }
            )

    if not rows:
        print("No valid per-sample files found under ROOT.")
        if skipped:
            print(f"Unmapped methods seen: {skipped}")
        return

    per_seed = pd.DataFrame(rows)

    # Sanity check: top-label Brier must be in [0, 25] in percentage points.
    # 25 corresponds to p_hat=0.5 always wrong, which is a hard ceiling for any
    # reasonable calibrator. Anything outside this range indicates a data bug.
    bad = per_seed[
        (per_seed["top_label_brier_pct"] < 0)
        | (per_seed["top_label_brier_pct"] > 25)
    ]
    if len(bad):
        print("WARN: implausible top-label Brier values:")
        print(bad.to_string(index=False))

    # 2. Aggregate per (dataset, model, method)
    agg = (
        per_seed.groupby(["dataset", "model", "method"], as_index=False)
        .apply(aggregate, include_groups=False)
        .reset_index(drop=True)
    )
    agg.to_csv("top_label_brier_long.csv", index=False)
    print(f"Wrote top_label_brier_long.csv ({len(agg)} rows)")

    # 3. For each (dataset, model, seed, method), join in ECE and multi-class
    # Brier from the JSONs so we can compare rankings.
    diag_rows = []
    for _, r in per_seed.iterrows():
        section = DISPLAY_TO_JSON_SECTION.get(r["method"])
        if section is None:
            continue
        data = load_json(ROOT, r["dataset"], r["model"], r["seed"])
        ece = get_section_value(data, section, "ece")
        mcb = get_section_value(data, section, "brier")
        diag_rows.append(
            {
                "dataset": r["dataset"],
                "model": r["model"],
                "method": r["method"],
                "seed": r["seed"],
                "top_label_brier_pct": r["top_label_brier_pct"],
                "ece_pct": ece * 100.0 if ece is not None else None,
                "multi_class_brier_pct": mcb * 100.0 if mcb is not None else None,
            }
        )
    diag = pd.DataFrame(diag_rows)
    diag_agg = (
        diag.groupby(["dataset", "model", "method"], as_index=False)
        .agg(
            n_seeds=("seed", "count"),
            tlb_mean_pct=("top_label_brier_pct", "mean"),
            ece_mean_pct=("ece_pct", "mean"),
            mcb_mean_pct=("multi_class_brier_pct", "mean"),
        )
    )
    diag_agg.to_csv("top_label_brier_diag.csv", index=False)
    print(f"Wrote top_label_brier_diag.csv ({len(diag_agg)} rows)")

    # 4. Print a readable summary
    print()
    print("=" * 80)
    print("PER-CONFIG COMPARISON: top-label Brier (TLB) vs ECE vs multi-class Brier (MCB)")
    print("All values in percentage points. Lower is better for all three.")
    print("=" * 80)
    pretty = diag_agg.copy()
    for col in ("tlb_mean_pct", "ece_mean_pct", "mcb_mean_pct"):
        pretty[col] = pretty[col].apply(lambda v: f"{v:6.2f}" if pd.notna(v) else "  ----")
    print(pretty.to_string(index=False))

    # 5. Ranking agreement: per (dataset, model), does TLB rank RGCL vs RGCC
    # the same as ECE does?
    print()
    print("=" * 80)
    print("RGCL-vs-RGCC RANKING AGREEMENT")
    print("=" * 80)
    pivot_tlb = diag_agg.pivot_table(
        index=["dataset", "model"], columns="method", values="tlb_mean_pct"
    )
    pivot_ece = diag_agg.pivot_table(
        index=["dataset", "model"], columns="method", values="ece_mean_pct"
    )
    pivot_mcb = diag_agg.pivot_table(
        index=["dataset", "model"], columns="method", values="mcb_mean_pct"
    )
    matches = 0
    mismatches = 0
    for idx in pivot_tlb.index:
        if "RGCL" not in pivot_tlb.columns or "RGCC" not in pivot_tlb.columns:
            continue
        l_tlb = pivot_tlb.loc[idx].get("RGCL", np.nan)
        c_tlb = pivot_tlb.loc[idx].get("RGCC", np.nan)
        l_ece = pivot_ece.loc[idx].get("RGCL", np.nan)
        c_ece = pivot_ece.loc[idx].get("RGCC", np.nan)
        if any(pd.isna(x) for x in (l_tlb, c_tlb, l_ece, c_ece)):
            print(f"  {idx}: incomplete data, skipping")
            continue
        tlb_winner = "RGCL" if l_tlb < c_tlb else "RGCC"
        ece_winner = "RGCL" if l_ece < c_ece else "RGCC"
        verdict = "AGREE" if tlb_winner == ece_winner else "MISMATCH"
        if tlb_winner == ece_winner:
            matches += 1
        else:
            mismatches += 1
        print(
            f"  {idx}: TLB winner={tlb_winner} ({l_tlb:.2f} vs {c_tlb:.2f}), "
            f"ECE winner={ece_winner} ({l_ece:.2f} vs {c_ece:.2f}) -> {verdict}"
        )
    total = matches + mismatches
    if total > 0:
        print(f"\nAgreement: {matches}/{total} configs ({100 * matches / total:.0f}%)")

    # 6. Sanity check: is multi-class Brier always larger than top-label Brier?
    # It should be, by construction. If not, something is broken.
    print()
    print("=" * 80)
    print("SANITY: multi-class Brier should be >= top-label Brier per config")
    print("=" * 80)
    for _, r in diag_agg.iterrows():
        if pd.isna(r["mcb_mean_pct"]) or pd.isna(r["tlb_mean_pct"]):
            continue
        diff = r["mcb_mean_pct"] - r["tlb_mean_pct"]
        flag = "" if diff >= 0 else "  <-- VIOLATION"
        print(
            f"  {r['dataset']} {r['model']} {r['method']}: "
            f"MCB={r['mcb_mean_pct']:.2f}, TLB={r['tlb_mean_pct']:.2f}, "
            f"gap={diff:+.2f}{flag}"
        )

    if skipped:
        print()
        print("Unmapped methods skipped (informational):")
        for k, v in sorted(skipped.items(), key=lambda kv: -kv[1]):
            print(f"  {k}: {v} files")


if __name__ == "__main__":
    main()
