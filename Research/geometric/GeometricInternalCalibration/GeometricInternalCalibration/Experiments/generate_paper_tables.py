#!/usr/bin/env python3
"""
Generate publication-ready tables from aggregated calibration results.

This script relies on `aggregate_calibration_results.py` for JSON parsing and
method extraction. It regenerates aggregation on demand instead of reading
intermediate artifacts to keep the pipeline simple and robust.

Tables produced:
- Table 1 (Main Results): Uncal, TS, DAC, SGC, GeoComb, RandCoord
  grouped by (dataset, model), aggregating across training methods.
- Table 2 (Ablation - Feature Sources): Geo+DACFeat, DAC+SGCFeat, DAC+RandL
  grouped by (dataset, model).
- Table 3 (Coordinate Methods): RandCoord, CoordSPP, CoordSPP+DAC, DAC+Coord
  grouped by (dataset, model).
- Table 4 (By Training Method): Uncal, TS, DAC, TS+DAC, SGC, GeoComb, RandCoord,
  NormPostAgg grouped by (training, dataset, model).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import pandas as pd

from .aggregate_calibration_results import (
    METHODS,
    calculate_statistics,
    format_statistic,
    collect_results_from_dir,
    logger,
)


def _write_latex_table(df: pd.DataFrame, output_tex: Path) -> None:
    """Write a simple booktabs LaTeX table without pandas' optional jinja2 dependency."""
    col_spec = " ".join(["l"] * min(3, len(df.columns)) + ["c"] * max(0, len(df.columns) - 3))
    lines = [
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        " & ".join(str(col) for col in df.columns) + " \\\\",
        "\\midrule",
    ]

    for _, row in df.iterrows():
        lines.append(" & ".join(str(row[col]) for col in df.columns) + " \\\\")

    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
    ])
    output_tex.write_text("\n".join(lines))


def _aggregate_by_key(
    config_data: Dict[Tuple[str, str, str], Dict[str, List[float]]],
    key_fn,
    methods: Iterable[str],
) -> pd.DataFrame:
    """
    Generic helper for aggregating over configurations and returning a table.

    key_fn:  (training, dataset, model) -> tuple used as grouping key
    methods: iterable of method codes to include
    """
    rows = []
    for (training, dataset, model), method_data in sorted(config_data.items()):
        group_key = key_fn(training, dataset, model)
        row_base = {}
        # group_key is either (dataset, model) or (training, dataset, model)
        if len(group_key) == 2:
            d, m = group_key
            row_base.update({"Dataset": d, "Model": m})
        else:
            t, d, m = group_key
            row_base.update({"Training": t, "Dataset": d, "Model": m})

        # Collect all values for this group across configs
        if len(group_key) == 2:
            # group across all trainings
            # we'll accumulate later outside loop
            pass

        rows.append((group_key, (training, dataset, model), method_data))

    # First stage: accumulate values at group key level
    grouped: Dict[Tuple, Dict[str, List[float]]] = {}
    for group_key, _cfg_key, method_data in rows:
        if group_key not in grouped:
            grouped[group_key] = {code: [] for code in METHODS.keys()}
        for code in METHODS.keys():
            if code in method_data:
                grouped[group_key][code].extend(method_data[code])

    # Second stage: compute stats and build DataFrame rows
    out_rows = []
    for group_key, method_values in sorted(grouped.items()):
        if len(group_key) == 2:
            dataset, model = group_key
            row = {"Dataset": dataset, "Model": model}
        else:
            training, dataset, model = group_key
            row = {"Training": training, "Dataset": dataset, "Model": model}

        # N = max samples across included methods
        n_max = 0
        for code in methods:
            vals = method_values.get(code, [])
            if vals:
                n_max = max(n_max, len(vals))
        row["N"] = n_max

        for code in methods:
            vals = method_values.get(code, [])
            n, mean, ci = calculate_statistics(vals)
            spec = METHODS[code]
            row[spec.display] = format_statistic(mean, ci, n)

        out_rows.append(row)

    df = pd.DataFrame(out_rows)
    # Stable column order: keys + N + methods (by display name)
    key_cols = ["Training", "Dataset", "Model"] if "Training" in df.columns else ["Dataset", "Model"]
    method_displays = [METHODS[c].display for c in methods]
    cols = key_cols + ["N"] + method_displays
    df = df.reindex(columns=cols)
    return df


def generate_main_table(
    config_data: Dict[Tuple[str, str, str], Dict[str, List[float]]],
    output_csv: Path,
    output_tex: Path,
) -> None:
    """Generate main comparison table (Table 1)."""
    methods = ["uncal", "ts", "dac", "sgc", "geo_comb", "rand_coord"]
    df = _aggregate_by_key(
        config_data,
        key_fn=lambda _t, d, m: (d, m),
        methods=methods,
    )
    df.to_csv(output_csv, index=False)
    logger.info(f"Table 1 (main results) CSV saved to {output_csv}")
    _write_latex_table(df, output_tex)
    logger.info(f"Table 1 (main results) LaTeX saved to {output_tex}")


def generate_ablation_table(
    config_data: Dict[Tuple[str, str, str], Dict[str, List[float]]],
    output_csv: Path,
    output_tex: Path,
) -> None:
    """Generate ablation study table (Table 2): feature sources."""
    methods = ["geo_dac_feat", "dac_sgc_feat", "dac_rand_l"]
    df = _aggregate_by_key(
        config_data,
        key_fn=lambda _t, d, m: (d, m),
        methods=methods,
    )
    df.to_csv(output_csv, index=False)
    logger.info(f"Table 2 (ablation) CSV saved to {output_csv}")
    _write_latex_table(df, output_tex)
    logger.info(f"Table 2 (ablation) LaTeX saved to {output_tex}")


def generate_coordinate_table(
    config_data: Dict[Tuple[str, str, str], Dict[str, List[float]]],
    output_csv: Path,
    output_tex: Path,
) -> None:
    """Generate coordinate methods table (Table 3)."""
    methods = ["rand_coord", "coord_spp", "coord_spp_dac", "dac_coord"]
    df = _aggregate_by_key(
        config_data,
        key_fn=lambda _t, d, m: (d, m),
        methods=methods,
    )
    df.to_csv(output_csv, index=False)
    logger.info(f"Table 3 (coordinate methods) CSV saved to {output_csv}")
    _write_latex_table(df, output_tex)
    logger.info(f"Table 3 (coordinate methods) LaTeX saved to {output_tex}")


def generate_training_method_table(
    config_data: Dict[Tuple[str, str, str], Dict[str, List[float]]],
    output_csv: Path,
    output_tex: Path,
) -> None:
    """
    Generate per-training-method breakdown (Table 4).

    Columns: Uncal, TS, DAC, TS+DAC, SGC, GeoComb, RandCoord, NormPostAgg.
    Grouping key: (training_method, dataset, model).
    """
    methods = ["uncal", "ts", "dac", "ts_dac", "sgc", "geo_comb", "rand_coord", "norm_post"]
    df = _aggregate_by_key(
        config_data,
        key_fn=lambda t, d, m: (t, d, m),
        methods=methods,
    )
    df.to_csv(output_csv, index=False)
    logger.info(f"Table 4 (training-method breakdown) CSV saved to {output_csv}")
    _write_latex_table(df, output_tex)
    logger.info(f"Table 4 (training-method breakdown) LaTeX saved to {output_tex}")


def _parse_mean_from_formatted(formatted_str: str) -> Optional[float]:
    """Extract mean value from formatted string like '3.66 ± 0.10' or 'N/A'."""
    if pd.isna(formatted_str) or formatted_str == "N/A":
        return None
    # Match pattern like "3.66 ± 0.10" or "3.66"
    match = re.match(r'([\d.]+)', str(formatted_str))
    if match:
        return float(match.group(1))
    return None


def _find_best_methods(row: pd.Series, method_columns: List[str]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Find 1st, 2nd, 3rd best methods (lowest mean ECE) in a row."""
    method_values = []
    for col in method_columns:
        mean_val = _parse_mean_from_formatted(row[col])
        if mean_val is not None:
            method_values.append((mean_val, col))
    
    if not method_values:
        return None, None, None
    
    # Sort by mean value (lower is better)
    method_values.sort(key=lambda x: x[0])
    
    first = method_values[0][1] if len(method_values) > 0 else None
    second = method_values[1][1] if len(method_values) > 1 else None
    third = method_values[2][1] if len(method_values) > 2 else None
    
    return first, second, third


def _apply_formatting(formatted_str: str, is_bold: bool, is_underline: bool) -> str:
    """Apply LaTeX formatting to a string. Bold takes precedence over underline."""
    if pd.isna(formatted_str) or formatted_str == "N/A":
        return "N/A"
    
    result = str(formatted_str)
    # Apply formatting: bold for 1st, underline for 2nd/3rd (mutually exclusive)
    if is_bold:
        result = f"\\textbf{{{result}}}"
    elif is_underline:
        result = f"\\underline{{{result}}}"
    return result


def generate_comprehensive_table(
    config_data: Dict[Tuple[str, str, str], Dict[str, List[float]]],
    output_csv: Path,
    output_tex: Path,
    metric_label: str = "ECE",
) -> None:
    """
    Generate comprehensive comparison table matching comprehensive_analysis.tex.
    
    Columns: Uncalibrated, TS, DAC, SGC, SGC-Concat, Coord-Geo, Coord-SPP-Geo,
    DAC (SPP feat), DAC (Coord feat).
    Grouping key: (training, dataset, model).
    Marks: 1st best = bold, 2nd and 3rd best = underline.
    """
    # Method codes in the order they appear in the table
    methods = ["uncal", "ts", "dac", "sgc", "geo_comb", "rand_coord", "coord_spp", "dac_sgc_feat", "dac_coord"]
    
    # Custom display names matching the tex file
    display_name_map = {
        "uncal": "Uncalibrated",
        "ts": "TS",
        "dac": "DAC",
        "sgc": "SGC",
        "geo_comb": "SGC-Concat",
        "rand_coord": "Coord-Geo",
        "coord_spp": "Coord-SPP-Geo",
        "dac_sgc_feat": "DAC (SPP feat)",
        "dac_coord": "DAC (Coord feat)",
    }
    
    # Aggregate data grouped by (training, dataset, model)
    rows = []
    for (training, dataset, model), method_data in sorted(config_data.items()):
        rows.append(((training, dataset, model), method_data))
    
    # Accumulate values at group key level
    grouped: Dict[Tuple, Dict[str, List[float]]] = {}
    for group_key, method_data in rows:
        if group_key not in grouped:
            grouped[group_key] = {code: [] for code in METHODS.keys()}
        for code in METHODS.keys():
            if code in method_data:
                grouped[group_key][code].extend(method_data[code])
    
    # Build DataFrame rows with statistics
    out_rows = []
    for (training, dataset, model), method_values in sorted(grouped.items()):
        row = {"Training": training, "Dataset": dataset, "Model": model}
        
        # N = max samples across included methods
        n_max = 0
        for code in methods:
            vals = method_values.get(code, [])
            if vals:
                n_max = max(n_max, len(vals))
        row["N"] = n_max
        
        # Compute statistics for each method
        for code in methods:
            vals = method_values.get(code, [])
            n, mean, ci = calculate_statistics(vals)
            display_name = display_name_map[code]
            row[display_name] = format_statistic(mean, ci, n)
        
        out_rows.append(row)
    
    df = pd.DataFrame(out_rows)
    
    # Column order matching the tex file
    key_cols = ["Training", "Dataset", "Model"]
    method_displays = [display_name_map[c] for c in methods]
    cols = key_cols + ["N"] + method_displays
    df = df.reindex(columns=cols)
    
    # Save CSV
    df.to_csv(output_csv, index=False)
    logger.info(f"Comprehensive table CSV saved to {output_csv}")
    
    # Apply formatting for LaTeX: bold for 1st, underline for 2nd and 3rd
    df_formatted = df.copy()
    method_columns = method_displays
    
    for idx in df.index:
        first, second, third = _find_best_methods(df.loc[idx], method_columns)
        
        for col in method_columns:
            original = df.loc[idx, col]
            is_bold = (col == first)
            is_underline = (col == second or col == third)
            df_formatted.loc[idx, col] = _apply_formatting(original, is_bold, is_underline)
    
    # Generate LaTeX manually to match exact format
    # Column spec: l l l c c c c c c c c c c (3 l's for Training/Dataset/Model, then c's)
    num_cols = len(cols)  # Should be 13: Training, Dataset, Model, N, and 9 methods
    col_spec = "l " * 3 + "c " * (num_cols - 3)
    
    latex_lines = [
        "\\begin{table*}[htbp]",
        "\\centering",
        "\\small",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\toprule"
    ]
    
    # Header row
    header = " & ".join(cols) + " \\\\"
    latex_lines.append(header)
    latex_lines.append("\\midrule")
    
    # Data rows
    for idx in df_formatted.index:
        row_data = []
        for col in cols:
            val = df_formatted.loc[idx, col]
            row_data.append(str(val))
        row_str = " & ".join(row_data) + " \\\\"
        latex_lines.append(row_str)
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append(f"\\caption{{Comprehensive calibration comparison across methods. Values show Mean {metric_label} (\\%) ± 95\\% CI. Best method per row is bolded, 2nd and 3rd best are underlined.}}")
    latex_lines.append("\\label{tab:comprehensive_comparison}")
    latex_lines.append("\\end{table*}")
    
    latex_str = "\n".join(latex_lines)
    
    # Write LaTeX file
    output_tex.write_text(latex_str)
    logger.info(f"Comprehensive table LaTeX saved to {output_tex}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready tables from calibration JSON results.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing ablation_*.json files (e.g. calibration_comparison/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_tables",
        help="Directory where tables (CSV/TeX) will be written.",
    )
    parser.add_argument(
        "--metric-field",
        type=str,
        default="ece",
        choices=["ece", "adaptive_ece", "calibration_mce"],
        help="Metric field to read from each method result.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metric_labels = {
        "ece": "ECE",
        "adaptive_ece": "Adaptive ECE",
        "calibration_mce": "MCE",
    }

    config_data, _raw_rows = collect_results_from_dir(input_dir, metric_field=args.metric_field)
    if not config_data:
        raise SystemExit("No configurations found; check input directory and JSON structure.")

    # Table 1
    generate_main_table(
        config_data,
        output_csv=out_dir / "table1_main_results.csv",
        output_tex=out_dir / "table1_main_results.tex",
    )

    # Table 2
    generate_ablation_table(
        config_data,
        output_csv=out_dir / "table2_ablation_features.csv",
        output_tex=out_dir / "table2_ablation_features.tex",
    )

    # Table 3
    generate_coordinate_table(
        config_data,
        output_csv=out_dir / "table3_coordinate_methods.csv",
        output_tex=out_dir / "table3_coordinate_methods.tex",
    )

    # Table 4
    generate_training_method_table(
        config_data,
        output_csv=out_dir / "table4_by_training_method.csv",
        output_tex=out_dir / "table4_by_training_method.tex",
    )

    # Comprehensive table (matching comprehensive_analysis.tex)
    generate_comprehensive_table(
        config_data,
        output_csv=out_dir / "comprehensive_analysis.csv",
        output_tex=out_dir / "comprehensive_analysis.tex",
        metric_label=metric_labels[args.metric_field],
    )

    logger.info(f"All {args.metric_field} tables written to {out_dir}")


if __name__ == "__main__":
    main()
