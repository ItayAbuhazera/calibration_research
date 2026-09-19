#!/usr/bin/env python3
"""
Aggregate multi-seed unified benchmark results from summary_metrics.json files.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
REQUIRED_STAGES = {
    "stage1_splits",
    "stage2_early",
    "stage3_features_fusion",
    "stage4_late_outputs",
}

DEFAULT_METHODS: List[str] = [
    "base_model",
    "temperature_scaling",
    "gc_dac",
    "anchored_model_tail",
    "full_vector_distance_fusion",
    "full_vector_distance_fusion_post_temperature",
]

MAIN_METRICS: List[str] = [
    "accuracy",
    "nll",
    "brier",
    "top_label_ece",
    "classwise_ece",
    "adaptive_ece",
]

METHOD_METADATA_COLUMNS: List[str] = [
    "method_family",
    "can_change_argmax",
    "tuning_procedure",
    "tuning_split",
    "tuning_objective",
]

PRESERVED_NON_NUMERIC_COLUMNS: List[str] = [
    "dataset",
    "model",
    "training_method",
    "shared_metric_module",
    "method_family",
    "can_change_argmax",
    "tuning_procedure",
    "tuning_split",
    "tuning_objective",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate unified benchmark summary_metrics.json files across seeds."
    )
    parser.add_argument(
        "--input-glob",
        type=str,
        default="results/unified_benchmark_c100_seed*_rankgeom_mix_posttemp/summary_metrics.json",
        help="Glob for summary_metrics.json files.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=DEFAULT_METHODS,
        help="Method names to include.",
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Aggregate every method found in the input summaries.",
    )
    parser.add_argument(
        "--output-per-seed",
        type=str,
        default="results/aggregated/posttemp_validation_per_seed.csv",
        help="Output CSV path for per-seed long-form rows.",
    )
    parser.add_argument(
        "--output-aggregate",
        type=str,
        default="results/aggregated/c100_rankgeom_mix_posttemp_mean_ci95.csv",
        help="Output CSV path for aggregated rows.",
    )
    parser.add_argument(
        "--ci-z",
        type=float,
        default=None,
        help="Optional fixed critical value. By default, use Student's t for each metric's seed count.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Only include summaries whose four stages and final per-sample artifacts are complete.",
    )
    parser.add_argument(
        "--identity-tolerance",
        type=float,
        default=1e-7,
        help="Tolerance for identity diagnostics on aggregated means.",
    )
    parser.add_argument(
        "--min-base-accuracy",
        type=float,
        default=0.0,
        help="Drop full seeds whose base_model accuracy is below this threshold (range [0, 1]).",
    )
    parser.add_argument(
        "--per-architecture",
        action="store_true",
        help=(
            "Group input summaries by model (architecture) and write one pair of output files per "
            "architecture. Output paths are derived from --output-per-seed and --output-aggregate "
            "by inserting the model name before the file extension."
        ),
    )
    return parser.parse_args()


def _extract_seed_from_name(text: str) -> int | None:
    match = re.search(r"seed(\d+)", text)
    return int(match.group(1)) if match else None


def _normalize_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True)
    return value


def _extract_base_model_accuracy(methods: List[Any]) -> float | None:
    for method_entry in methods:
        if not isinstance(method_entry, dict):
            continue
        if method_entry.get("method_name") != "base_model":
            continue
        method_metrics = method_entry.get("metrics")
        if not isinstance(method_metrics, dict):
            return None
        accuracy_value = method_metrics.get("accuracy")
        if accuracy_value is None:
            return None
        try:
            return float(accuracy_value)
        except (TypeError, ValueError):
            return None
    return None


def _flatten_one_summary(json_path: Path, selected_methods: set[str] | None) -> Dict[str, Any]:
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    metadata = payload.get("metadata")
    methods = payload.get("methods")
    if not isinstance(metadata, dict):
        raise ValueError(f"'metadata' object missing in {json_path}")
    if not isinstance(methods, list):
        raise ValueError(f"'methods' list missing in {json_path}")

    source_result_dir = str(json_path.parent)
    metadata_seed = metadata.get("seed")
    seed = metadata_seed if metadata_seed is not None else _extract_seed_from_name(source_result_dir)

    base_row = {
        "source_summary_file": str(json_path),
        "source_result_dir": source_result_dir,
        "seed": seed,
        "dataset": metadata.get("dataset"),
        "model": metadata.get("model"),
        "training_method": metadata.get("method"),
        "checkpoint_path": metadata.get("checkpoint_path"),
        "shared_metric_module": metadata.get("shared_metric_module"),
        "split_sizes": _normalize_value(metadata.get("split_sizes")),
    }

    rows: List[Dict[str, Any]] = []
    for method_entry in methods:
        if not isinstance(method_entry, dict):
            continue
        method_name = method_entry.get("method_name")
        if selected_methods is not None and method_name not in selected_methods:
            continue

        row = dict(base_row)
        for col in METHOD_METADATA_COLUMNS:
            row[col] = method_entry.get(col)
        row["method_name"] = method_name

        method_metrics = method_entry.get("metrics", {})
        if not isinstance(method_metrics, dict):
            raise ValueError(f"'metrics' must be an object for method '{method_name}' in {json_path}")
        for metric_name, metric_value in method_metrics.items():
            row[metric_name] = _normalize_value(metric_value)

        for key, value in method_entry.items():
            if key in {"method_name", "method_family", "can_change_argmax", "tuning_procedure", "tuning_split", "tuning_objective", "metrics"}:
                continue
            row[key] = _normalize_value(value)

        rows.append(row)

    return {
        "json_path": json_path,
        "seed": seed,
        "dataset": metadata.get("dataset"),
        "model": metadata.get("model"),
        "base_model_accuracy": _extract_base_model_accuracy(methods),
        "rows": rows,
    }


def _is_complete_summary(json_path: Path) -> bool:
    result_dir = json_path.parent
    progress_path = result_dir / "intermediates" / "progress.json"
    manifest_path = result_dir / "per_sample" / "per_sample_manifest.json"
    arrays_path = result_dir / "per_sample" / "per_sample_arrays.npz"
    if not progress_path.is_file() or not manifest_path.is_file() or not arrays_path.is_file():
        return False
    if manifest_path.stat().st_size == 0 or arrays_path.stat().st_size == 0:
        return False
    try:
        with progress_path.open("r", encoding="utf-8") as f:
            progress = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    completed_stages = set(progress.get("completed_stages", []))
    return REQUIRED_STAGES.issubset(completed_stages)


def _mixed_or_single(series: pd.Series) -> Any:
    unique = [v for v in series.dropna().unique().tolist() if v != ""]
    if not unique:
        return pd.NA
    if len(unique) == 1:
        return unique[0]
    return "__MIXED__"


def _prepare_numeric_columns(per_seed_df: pd.DataFrame) -> List[str]:
    non_numeric_force = {
        "seed",
        "source_summary_file",
        "source_result_dir",
        "checkpoint_path",
        "split_sizes",
        "method_name",
        "dataset",
        "model",
        "training_method",
        "shared_metric_module",
        "method_family",
        "can_change_argmax",
        "tuning_procedure",
        "tuning_split",
        "tuning_objective",
        "anchor_source_method",
        "post_fusion_calibration_type",
        "beta_selection",
        "validation_gates",
    }

    numeric_columns: List[str] = []
    for col in per_seed_df.columns:
        if col in non_numeric_force:
            continue
        numeric_candidate = pd.to_numeric(per_seed_df[col], errors="coerce")
        if numeric_candidate.notna().any():
            per_seed_df[col] = numeric_candidate
            numeric_columns.append(col)
    return numeric_columns


def _t_critical_95(sample_count: int) -> float:
    """Return the two-sided 95% Student-t critical value for n observations."""
    critical_by_df = {
        1: 12.7062047364,
        2: 4.3026527297,
        3: 3.1824463053,
        4: 2.7764451052,
        5: 2.5705818356,
        6: 2.4469118511,
        7: 2.3646242510,
        8: 2.3060041352,
        9: 2.2621571629,
        10: 2.2281388520,
        11: 2.2009851601,
        12: 2.1788128297,
        13: 2.1603686565,
        14: 2.1447866879,
        15: 2.1314495456,
        16: 2.1199052992,
        17: 2.1098155778,
        18: 2.1009220402,
        19: 2.0930240544,
        20: 2.0859634473,
        21: 2.0796138447,
        22: 2.0738730679,
        23: 2.0686576104,
        24: 2.0638985616,
        25: 2.0595385528,
        26: 2.0555294386,
        27: 2.0518305165,
        28: 2.0484071418,
        29: 2.0452296421,
        30: 2.0422724563,
    }
    if sample_count < 2:
        return float("nan")
    degrees_of_freedom = sample_count - 1
    return critical_by_df.get(degrees_of_freedom, 1.9599639845)


def _build_aggregate(
    per_seed_df: pd.DataFrame,
    numeric_columns: Sequence[str],
    ci_z: float | None,
    seed_count_total_by_method: pd.Series,
) -> pd.DataFrame:
    grouped = per_seed_df.groupby("method_name", dropna=False, sort=False)

    aggregate_parts: List[pd.DataFrame] = []
    for col in numeric_columns:
        stats = grouped[col].agg(["mean", "std", "count"])
        sem = stats["std"] / stats["count"].pow(0.5)
        if ci_z is None:
            critical_value = stats["count"].map(lambda n: _t_critical_95(int(n)))
        else:
            critical_value = ci_z
        ci_halfwidth = critical_value * sem
        ci_low = stats["mean"] - ci_halfwidth
        ci_high = stats["mean"] + ci_halfwidth
        stats = stats.rename(
            columns={
                "mean": f"{col}_mean",
                "count": f"{col}_seed_count",
            }
        )
        stats[f"{col}_std"] = stats["std"]
        stats[f"{col}_sem"] = sem
        stats[f"{col}_ci95_low"] = ci_low
        stats[f"{col}_ci95_high"] = ci_high
        stats[f"{col}_ci95_halfwidth"] = ci_halfwidth
        stats = stats.drop(columns=["std"])
        aggregate_parts.append(stats)

    if aggregate_parts:
        aggregate_df = pd.concat(aggregate_parts, axis=1)
    else:
        aggregate_df = pd.DataFrame(index=grouped.size().index)

    supplemental_columns: Dict[str, pd.Series] = {
        "seed_count_total": seed_count_total_by_method.reindex(aggregate_df.index).fillna(0).astype(int),
        "seed_count_after_filter": grouped["seed"].nunique(dropna=True),
        "row_count_total": grouped.size(),
    }
    for col in PRESERVED_NON_NUMERIC_COLUMNS:
        if col in per_seed_df.columns:
            supplemental_columns[col] = grouped[col].apply(_mixed_or_single)

    aggregate_df = pd.concat([aggregate_df, pd.DataFrame(supplemental_columns)], axis=1)
    aggregate_df = aggregate_df.reset_index()
    return aggregate_df


def _format_metric_line(row: pd.Series, metrics: Sequence[str]) -> str:
    parts: List[str] = []
    for metric in metrics:
        mean_col = f"{metric}_mean"
        ci_halfwidth_col = f"{metric}_ci95_halfwidth"
        count_col = f"{metric}_seed_count"
        if mean_col not in row.index or pd.isna(row[mean_col]):
            continue
        mean_val = float(row[mean_col])
        ci_halfwidth_val = row[ci_halfwidth_col] if ci_halfwidth_col in row.index else pd.NA
        count_val = int(row[count_col]) if count_col in row.index and pd.notna(row[count_col]) else 0
        if pd.notna(ci_halfwidth_val):
            parts.append(f"{metric}={mean_val:.6f}±{float(ci_halfwidth_val):.6f} (n={count_val}, 95% CI)")
        else:
            parts.append(f"{metric}={mean_val:.6f} (n={count_val})")
    return ", ".join(parts)


def _print_method_summary(aggregate_df: pd.DataFrame, methods: Sequence[str]) -> None:
    print("\nMain method summary (mean ± 95% CI half-width across seeds):")
    for method_name in methods:
        match = aggregate_df[aggregate_df["method_name"] == method_name]
        if match.empty:
            print(f"  - {method_name}: not found")
            continue
        row = match.iloc[0]
        line = _format_metric_line(row, MAIN_METRICS)
        print(f"  - {method_name}: {line if line else 'no numeric metrics'}")


def _print_identity_diagnostic(aggregate_df: pd.DataFrame, tol: float) -> None:
    pairs = [
        ("full_vector_distance_fusion", "base_model"),
        ("full_vector_distance_fusion_post_temperature", "temperature_scaling"),
    ]

    print(f"\nIdentity diagnostic (tolerance={tol:g} on aggregated means):")
    for left, right in pairs:
        left_row = aggregate_df[aggregate_df["method_name"] == left]
        right_row = aggregate_df[aggregate_df["method_name"] == right]
        if left_row.empty or right_row.empty:
            print(f"  - {left} vs {right}: skipped (missing method)")
            continue

        left_row = left_row.iloc[0]
        right_row = right_row.iloc[0]
        diffs: List[str] = []
        identical = True
        for metric in MAIN_METRICS:
            left_col = f"{metric}_mean"
            right_col = f"{metric}_mean"
            if left_col not in left_row.index or pd.isna(left_row[left_col]) or pd.isna(right_row[right_col]):
                identical = False
                diffs.append(f"{metric}=NA")
                continue
            delta = abs(float(left_row[left_col]) - float(right_row[right_col]))
            diffs.append(f"{metric} Δ={delta:.3e}")
            if delta > tol:
                identical = False

        print(f"  - {left} vs {right}: " + ", ".join(diffs))
        if identical:
            print("    WARNING: numerically identical within tolerance; fusion may be acting like identity on these runs.")


def _load_and_filter_seed_records(
    input_files: List[Path],
    selected_methods: set[str] | None,
    min_base_accuracy: float,
    logger: logging.Logger,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], pd.DataFrame]:
    """Load summaries, apply accuracy filter, return (retained, dropped, all_rows_df)."""
    seed_records: List[Dict[str, Any]] = []
    for json_path in input_files:
        seed_records.append(_flatten_one_summary(json_path, selected_methods))

    rows_all: List[Dict[str, Any]] = []
    for seed_record in seed_records:
        rows_all.extend(seed_record["rows"])

    per_seed_df_all = pd.DataFrame(rows_all) if rows_all else pd.DataFrame()
    if "seed" in per_seed_df_all.columns:
        per_seed_df_all["seed"] = pd.to_numeric(per_seed_df_all["seed"], errors="coerce").astype("Int64")

    dropped_seeds: List[Dict[str, Any]] = []
    retained_seed_records: List[Dict[str, Any]] = []
    for seed_record in seed_records:
        base_acc = seed_record["base_model_accuracy"]
        if base_acc is None:
            dropped_seeds.append(
                {
                    "source_summary_file": str(seed_record["json_path"]),
                    "base_model_accuracy": "missing",
                    "reason": "no_base_model_row",
                }
            )
            continue
        if base_acc < min_base_accuracy:
            dropped_seeds.append(
                {
                    "source_summary_file": str(seed_record["json_path"]),
                    "base_model_accuracy": base_acc,
                    "reason": "below_threshold",
                }
            )
            continue
        retained_seed_records.append(seed_record)

    logger.info("Total seeds discovered: %d", len(seed_records))
    logger.info("Seeds dropped by filter: %d (threshold=%.6f)", len(dropped_seeds), min_base_accuracy)
    for dropped in dropped_seeds:
        logger.info(
            "Dropped seed: source_summary_file=%s, base_model_accuracy=%s, reason=%s",
            dropped["source_summary_file"],
            dropped["base_model_accuracy"],
            dropped["reason"],
        )
    return retained_seed_records, dropped_seeds, per_seed_df_all


def _run_one_aggregation(
    retained_seed_records: List[Dict[str, Any]],
    per_seed_df_all: pd.DataFrame,
    ci_z: float | None,
    out_per_seed: Path,
    out_aggregate: Path,
    identity_tolerance: float,
    label: str,
    logger: logging.Logger,
) -> None:
    """Build and write per-seed + aggregate CSVs for one group of seed records."""
    rows_retained: List[Dict[str, Any]] = []
    for seed_record in retained_seed_records:
        rows_retained.extend(seed_record["rows"])

    if not rows_retained:
        logger.warning("No method rows remained after filtering for %s; skipping.", label)
        return

    per_seed_df = pd.DataFrame(rows_retained)
    if "seed" in per_seed_df.columns:
        per_seed_df["seed"] = pd.to_numeric(per_seed_df["seed"], errors="coerce").astype("Int64")

    missing_main_metrics = [m for m in MAIN_METRICS if m not in per_seed_df.columns]
    if missing_main_metrics:
        raise ValueError(f"Missing required main metrics for {label}: {missing_main_metrics}")

    if not per_seed_df_all.empty and "method_name" in per_seed_df_all.columns and "seed" in per_seed_df_all.columns:
        seed_count_total_by_method = per_seed_df_all.groupby("method_name", dropna=False, sort=False)["seed"].nunique(dropna=True)
    else:
        seed_count_total_by_method = per_seed_df.groupby("method_name", dropna=False, sort=False)["seed"].nunique(dropna=True)

    numeric_columns = _prepare_numeric_columns(per_seed_df)
    aggregate_df = _build_aggregate(
        per_seed_df,
        numeric_columns,
        ci_z=ci_z,
        seed_count_total_by_method=seed_count_total_by_method,
    )

    out_per_seed.parent.mkdir(parents=True, exist_ok=True)
    out_aggregate.parent.mkdir(parents=True, exist_ok=True)
    per_seed_df.to_csv(out_per_seed, index=False)
    aggregate_df.to_csv(out_aggregate, index=False)

    retained_tuples = [(s["dataset"], s["model"], s["seed"]) for s in retained_seed_records]
    print(f"\n[{label}] Seeds retained: {len(retained_seed_records)} — {retained_tuples}")
    print(f"[{label}] Wrote per-seed table : {out_per_seed}")
    print(f"[{label}] Wrote aggregated table: {out_aggregate}")
    _print_method_summary(aggregate_df, DEFAULT_METHODS)
    _print_identity_diagnostic(aggregate_df, identity_tolerance)


def _derive_arch_output_path(base_path: str, model_name: str) -> Path:
    p = Path(base_path)
    return p.parent / f"{p.stem}_{model_name}{p.suffix}"


def main() -> None:
    args = _parse_args()
    if not 0.0 <= args.min_base_accuracy <= 1.0:
        raise ValueError("--min-base-accuracy must be in [0, 1].")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    selected_methods = None if args.all_methods else set(args.methods)

    input_files = sorted(Path(".").glob(args.input_glob))
    input_files = [p for p in input_files if p.is_file()]
    if not input_files:
        raise FileNotFoundError(f"No files matched --input-glob: {args.input_glob}")
    if args.require_complete:
        incomplete_files = [p for p in input_files if not _is_complete_summary(p)]
        input_files = [p for p in input_files if _is_complete_summary(p)]
        for incomplete_file in incomplete_files:
            logger.info("Excluded incomplete summary: %s", incomplete_file)
        if not input_files:
            raise ValueError("No complete summary files remained after --require-complete filtering.")

    print(f"Loaded {len(input_files)} summary JSON files.")

    if args.per_architecture:
        # Group files by model, then aggregate each group separately.
        from collections import defaultdict
        files_by_model: Dict[str, List[Path]] = defaultdict(list)
        for json_path in input_files:
            try:
                with json_path.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                model_name = payload.get("metadata", {}).get("model") or "unknown"
            except Exception:
                model_name = "unknown"
            files_by_model[model_name].append(json_path)

        print(f"Architectures found: {sorted(files_by_model)}")
        for model_name, model_files in sorted(files_by_model.items()):
            logger.info("--- Architecture: %s (%d files) ---", model_name, len(model_files))
            retained, dropped, all_df = _load_and_filter_seed_records(
                model_files, selected_methods, args.min_base_accuracy, logger
            )
            if not retained:
                logger.warning("All seeds dropped for %s; skipping.", model_name)
                continue
            _run_one_aggregation(
                retained_seed_records=retained,
                per_seed_df_all=all_df,
                ci_z=args.ci_z,
                out_per_seed=_derive_arch_output_path(args.output_per_seed, model_name),
                out_aggregate=_derive_arch_output_path(args.output_aggregate, model_name),
                identity_tolerance=args.identity_tolerance,
                label=model_name,
                logger=logger,
            )
    else:
        retained, dropped, all_df = _load_and_filter_seed_records(
            input_files, selected_methods, args.min_base_accuracy, logger
        )
        if not retained:
            raise ValueError("All seeds were dropped by --min-base-accuracy filtering; no output files were written.")
        _run_one_aggregation(
            retained_seed_records=retained,
            per_seed_df_all=all_df,
            ci_z=args.ci_z,
            out_per_seed=Path(args.output_per_seed),
            out_aggregate=Path(args.output_aggregate),
            identity_tolerance=args.identity_tolerance,
            label="all",
            logger=logger,
        )


if __name__ == "__main__":
    main()
