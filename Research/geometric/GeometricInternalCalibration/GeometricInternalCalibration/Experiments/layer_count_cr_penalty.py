#!/usr/bin/env python3
"""
Aggregate layer-count ablation by compression ratio.

Reads the global-random layer count comparison (document 6: layer_count_global_random.tex
is produced from this CSV) and produces a compact LaTeX table:
  - Rows: layer counts (L=2,4,6,8)
  - Columns: compression ratios
  - Each cell: mean ECE across all configs for that layer/CR
  - Bold = best CR for that layer
  - Penalty column = worst - best ECE for that layer
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
from utils.logging_config import get_logger
logger = get_logger(__name__)


def _format_cr(cr: float) -> str:
    """Pretty-print compression ratio (integer-ish → no decimals)."""
    if float(cr).is_integer():
        return f"{int(cr)}x"
    return f"{cr:g}x"


def load_global_random(path: Path, strategy: str = "global_random") -> pd.DataFrame:
    df = pd.read_csv(path)
    if strategy:
        df = df[df["strategy"] == strategy]
    if df.empty:
        raise ValueError(f"No rows for strategy={strategy} in {path}")
    return df


def compute_experiment_overview(df: pd.DataFrame) -> Dict[str, Any]:
    """Compute experiment overview statistics."""
    overview: Dict[str, Any] = {}
    
    if df.empty:
        return overview
    
    # Count unique configurations (training_method, dataset, model)
    config_cols = ['training_method', 'dataset', 'model']
    if all(col in df.columns for col in config_cols):
        unique_configs = df[config_cols].drop_duplicates()
        overview["total_configurations"] = len(unique_configs)
    else:
        overview["total_configurations"] = 0
    
    # Count total experiments
    overview["total_experiments"] = len(df)
    
    # Extract unique values
    if 'training_method' in df.columns:
        overview["training_methods"] = sorted(df["training_method"].unique().tolist())
    else:
        overview["training_methods"] = []
    
    if 'dataset' in df.columns:
        overview["datasets"] = sorted(df["dataset"].unique().tolist())
    else:
        overview["datasets"] = []
    
    if 'model' in df.columns:
        overview["models"] = sorted(df["model"].unique().tolist())
    else:
        overview["models"] = []
    
    # Extract compression ratios
    if 'compression_ratio' in df.columns:
        compression_ratios = sorted(df["compression_ratio"].dropna().unique().tolist())
        overview["compression_ratios"] = [f"{int(cr)}x" if cr == int(cr) else f"{cr:.1f}x" for cr in compression_ratios]
    else:
        overview["compression_ratios"] = []
    
    # Extract layer counts (from column names like layers_2_ece)
    layer_counts = []
    for col in df.columns:
        if col.startswith('layers_') and col.endswith('_ece'):
            try:
                layer_num = int(col.split('_')[1])
                if layer_num not in layer_counts:
                    layer_counts.append(layer_num)
            except (ValueError, IndexError):
                continue
    overview["layer_counts"] = sorted(layer_counts)
    
    return overview


def print_experiment_overview(overview: Dict[str, Any]):
    """Print experiment overview statistics."""
    if not overview:
        logger.warning("No overview data to display.")
        return
    
    print("\nEXPERIMENT OVERVIEW")
    print("-" * 80)
    print(f"Total configurations: {overview.get('total_configurations', 0)}")
    print(f"Total experiments (including seeds): {overview.get('total_experiments', 0)}")
    
    training_methods = overview.get("training_methods", [])
    if training_methods:
        methods_str = ", ".join(training_methods)
        print(f"Training methods: {methods_str} ({len(training_methods)} total)")
    
    datasets = overview.get("datasets", [])
    if datasets:
        datasets_str = ", ".join(datasets)
        print(f"Datasets: {datasets_str} ({len(datasets)} total)")
    
    models = overview.get("models", [])
    if models:
        models_str = ", ".join(models)
        print(f"Models: {models_str} ({len(models)} total)")
    
    compression_ratios = overview.get("compression_ratios", [])
    if compression_ratios:
        ratios_str = ", ".join(compression_ratios)
        print(f"Compression ratios: {ratios_str} ({len(compression_ratios)} total)")
    
    layer_counts = overview.get("layer_counts", [])
    if layer_counts:
        counts_str = ", ".join(str(lc) for lc in layer_counts)
        print(f"Layer counts: {counts_str} ({len(layer_counts)} total)")


def aggregate_by_layer(df: pd.DataFrame, layer_counts: List[int]) -> Dict[int, Dict]:
    """Compute mean ECE per CR for each layer count and best/worst stats."""
    crs = sorted(df["compression_ratio"].unique())
    results: Dict[int, Dict] = {}

    for lc in layer_counts:
        col = f"layers_{lc}_ece"
        if col not in df.columns:
            logger.warning("Missing column %s, skipping", col)
            continue

        agg = (
            df[["compression_ratio", col]]
            .dropna()
            .groupby("compression_ratio")[col]
            .mean()
            .reindex(crs)
        )

        valid = {cr: val for cr, val in agg.items() if not np.isnan(val)}
        if not valid:
            logger.warning("No data for L=%s", lc)
            continue

        best_cr = min(valid, key=valid.get)
        worst_cr = max(valid, key=valid.get)
        results[lc] = {
            "crs": crs,
            "values": agg,
            "best_cr": best_cr,
            "penalty": valid[worst_cr] - valid[best_cr],
        }
    return results


def latex_table(
    aggregated: Dict[int, Dict], output_path: Path, caption: str, label: str
) -> None:
    """Write LaTeX table with rows=L and columns=CR plus penalty."""
    # Collect all CRs that appear anywhere (preserve sorted order)
    all_crs: List[float] = []
    for info in aggregated.values():
        for cr in info["crs"]:
            if cr not in all_crs:
                all_crs.append(cr)
    all_crs = sorted(all_crs)

    header_cols = " & ".join(_format_cr(cr) for cr in all_crs)
    col_spec = "l" + "r" * len(all_crs) + "r" + "r"

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule",
        f"Layer & {header_cols} & Best CR & Penalty \\\\",
        "\\midrule",
    ]

    for lc, info in sorted(aggregated.items()):
        row_vals = []
        values = info["values"]
        best_cr = info["best_cr"]
        for cr in all_crs:
            val = values.get(cr, np.nan)
            if np.isnan(val):
                row_vals.append("---")
            else:
                sval = f"{val:.2f}"
                if cr == best_cr:
                    sval = f"\\textbf{{{sval}}}"
                row_vals.append(sval)
        penalty = f"{info['penalty']:.2f}"
        best_cr_str = _format_cr(best_cr)
        lines.append(f"L={lc} & " + " & ".join(row_vals) + f" & {best_cr_str} & {penalty} \\\\")

    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    logger.info("Wrote LaTeX table to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Summarize layer-count ablation by compression ratio (global random)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "calibration_comparison_results_full/layer_count_analysis/global_random_comparison.csv"
        ),
        help="CSV with per-config layer ECE columns (layers_{L}_ece).",
    )
    parser.add_argument(
        "--output-tex",
        type=Path,
        default=Path(
            "calibration_comparison_results_full/layer_count_analysis/latex_tables/layer_count_cr_by_layer.tex"
        ),
        help="Path to write the LaTeX table.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[2, 4, 6, 8],
        help="Layer counts to include.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="global_random",
        help="Filter rows by strategy column.",
    )
    parser.add_argument(
        "--caption",
        type=str,
        default="Global Random: mean ECE by compression ratio for each layer count (best CR bolded).",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="tab:layer_cr_penalty",
    )

    args = parser.parse_args()

    df = load_global_random(args.input, args.strategy)
    
    # Compute and print experiment overview
    overview = compute_experiment_overview(df)
    print_experiment_overview(overview)
    
    aggregated = aggregate_by_layer(df, args.layers)
    latex_table(aggregated, args.output_tex, args.caption, args.label)


if __name__ == "__main__":
    main()





