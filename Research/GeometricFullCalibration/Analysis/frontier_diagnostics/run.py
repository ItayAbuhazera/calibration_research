#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

EPS = 1e-12

METHOD_QUANTILE_COLUMNS = [
    "run_id",
    "method_name",
    "quantile_bin",
    "quantile_lo",
    "quantile_hi",
    "edge_lo",
    "edge_hi",
    "count",
    "accuracy",
    "mean_c_t",
    "nll",
    "brier_top_label",
    "brier_full",
    "mean_q_y",
    "delta_nll_vs_base",
]

PAIRWISE_COLUMNS = [
    "run_id",
    "reference_method",
    "comparison_method",
    "quantile_bin",
    "quantile_lo",
    "quantile_hi",
    "edge_lo",
    "edge_hi",
    "count",
    "nll_reference",
    "nll_comparison",
    "signed_nll_gap_method_minus_reference",
    "frac_top_equals_truth_reference",
    "frac_top_equals_truth_comparison",
]

DECOMP_COLUMNS = [
    "run_id",
    "method_name",
    "quantile_bin",
    "quantile_lo",
    "quantile_hi",
    "edge_lo",
    "edge_hi",
    "base_correct_flag",
    "count",
    "mean_c_t",
    "mean_p_t_base_top_prob",
    "mean_q_y",
    "mean_neg_log_q_y",
]


def _parse_quantiles(raw: str) -> List[float]:
    values = [float(v.strip()) for v in raw.split(",") if v.strip()]
    if len(values) < 2:
        raise ValueError("--quantiles must contain at least two values")
    if values[0] != 0.0 or values[-1] != 1.0:
        raise ValueError("--quantiles must start at 0.0 and end at 1.0")
    if any(values[i] > values[i + 1] for i in range(len(values) - 1)):
        raise ValueError("--quantiles must be non-decreasing")
    return values


def _mean_or_none(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.mean(values))


def _build_quantile_bins(c_t: np.ndarray, quantiles: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    n = int(c_t.shape[0])
    if n == 0:
        raise ValueError("Empty c_t array is not supported")
    edges = np.quantile(c_t, quantiles).astype(np.float64)

    # Rank-based quantile assignment with deterministic tie-breaking by original index.
    order = np.lexsort((np.arange(n, dtype=np.int64), c_t))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n, dtype=np.int64)

    cut_positions = np.array([int(round(q * n)) for q in quantiles], dtype=np.int64)
    cut_positions[0] = 0
    cut_positions[-1] = n
    cut_positions = np.maximum.accumulate(cut_positions)
    cut_positions = np.clip(cut_positions, 0, n)

    bin_idx = np.searchsorted(cut_positions[1:-1], rank, side="right").astype(np.int64)
    return bin_idx, edges


def _quantile_label(q_lo: float, q_hi: float) -> str:
    return f"q{int(round(q_lo * 100))}_q{int(round(q_hi * 100))}"


def _write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(path, index=False)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _to_float_or_none(v: float | None) -> float | None:
    return None if v is None else float(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Frontier diagnostics from per-sample benchmark arrays")
    parser.add_argument("run_dir", type=str, help="Unified benchmark run directory")
    parser.add_argument("--quantiles", type=str, default="0,0.25,0.5,0.75,1.0")
    parser.add_argument("--reference_method", type=str, default="vector_scaling")
    parser.add_argument(
        "--comparison_methods",
        type=str,
        default="anchored_model_tail,rgcl_tail_dirichlet,rgcl,base_model,odir_dirichlet",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    run_id = run_dir.name
    npz_path = run_dir / "per_sample" / "per_sample_arrays.npz"
    frontier_dir = run_dir / "frontier"
    frontier_dir.mkdir(parents=True, exist_ok=True)

    quantiles = _parse_quantiles(args.quantiles)
    comparison_methods = [m.strip() for m in args.comparison_methods.split(",") if m.strip()]

    data = np.load(npz_path, allow_pickle=False)
    labels = data["labels_test"].astype(np.int64)
    base_probs = data["base_probs_test"].astype(np.float64)
    c_t = data["gc_dac_anchor_ct_test"].astype(np.float64)
    method_index = [str(m) for m in data["method_index"].tolist()]
    method_can_change_argmax = data["method_can_change_argmax"].astype(np.bool_)
    can_change_map = {m: bool(v) for m, v in zip(method_index, method_can_change_argmax.tolist())}

    method_probs: Dict[str, np.ndarray] = {}
    for m in method_index:
        key = f"method_probs__{m}"
        if key not in data:
            continue
        method_probs[m] = data[key].astype(np.float64)

    if "base_model" not in method_probs:
        raise ValueError("Missing method_probs__base_model in per_sample_arrays.npz")
    if args.reference_method not in method_probs:
        raise ValueError(f"Reference method missing from NPZ: {args.reference_method}")

    n_test = int(labels.shape[0])
    idx = np.arange(n_test, dtype=np.int64)
    base_preds = np.argmax(base_probs, axis=1)
    base_correct = base_preds == labels
    base_top_prob = np.max(base_probs, axis=1)

    bin_idx, edges = _build_quantile_bins(c_t, quantiles)
    n_bins = len(quantiles) - 1
    bin_labels = [_quantile_label(quantiles[b], quantiles[b + 1]) for b in range(n_bins)]

    _write_json(
        frontier_dir / "quantile_edges.json",
        {
            "run_id": run_id,
            "quantiles": [float(q) for q in quantiles],
            "edges": [float(e) for e in edges.tolist()],
            "bin_labels": bin_labels,
            "bin_index_per_sample": [int(v) for v in bin_idx.tolist()],
        },
    )

    base_true_probs = np.clip(method_probs["base_model"][idx, labels], EPS, 1.0)
    base_nll = -np.log(base_true_probs)
    base_acc_by_bin: Dict[int, float | None] = {}

    method_quantile_rows: List[Dict[str, Any]] = []
    counts_by_method_bin: Dict[str, List[int]] = {}

    for m in method_index:
        if m not in method_probs:
            continue
        probs = method_probs[m]
        preds = np.argmax(probs, axis=1)
        is_correct = (preds == labels).astype(np.float64)
        q_y = np.clip(probs[idx, labels], EPS, 1.0)
        nll = -np.log(q_y)
        top_conf = np.max(probs, axis=1)
        top_correct = (preds == labels).astype(np.float64)
        brier_top = (top_conf - top_correct) ** 2
        one_hot = np.eye(probs.shape[1], dtype=np.float64)[labels]
        brier_full = np.sum((probs - one_hot) ** 2, axis=1)

        counts_by_method_bin[m] = []
        for b in range(n_bins):
            mask = bin_idx == b
            count = int(np.sum(mask))
            counts_by_method_bin[m].append(count)
            q_lo = float(quantiles[b])
            q_hi = float(quantiles[b + 1])
            e_lo = float(edges[b])
            e_hi = float(edges[b + 1])

            if count == 0:
                row = {
                    "run_id": run_id,
                    "method_name": m,
                    "quantile_bin": bin_labels[b],
                    "quantile_lo": q_lo,
                    "quantile_hi": q_hi,
                    "edge_lo": e_lo,
                    "edge_hi": e_hi,
                    "count": 0,
                    "accuracy": None,
                    "mean_c_t": None,
                    "nll": None,
                    "brier_top_label": None,
                    "brier_full": None,
                    "mean_q_y": None,
                    "delta_nll_vs_base": None,
                }
            else:
                mean_nll = float(np.mean(nll[mask]))
                base_bin_nll = float(np.mean(base_nll[mask]))
                row = {
                    "run_id": run_id,
                    "method_name": m,
                    "quantile_bin": bin_labels[b],
                    "quantile_lo": q_lo,
                    "quantile_hi": q_hi,
                    "edge_lo": e_lo,
                    "edge_hi": e_hi,
                    "count": count,
                    "accuracy": float(np.mean(is_correct[mask])),
                    "mean_c_t": float(np.mean(c_t[mask])),
                    "nll": mean_nll,
                    "brier_top_label": float(np.mean(brier_top[mask])),
                    "brier_full": float(np.mean(brier_full[mask])),
                    "mean_q_y": float(np.mean(q_y[mask])),
                    "delta_nll_vs_base": float(mean_nll - base_bin_nll),
                }
                if m == "base_model":
                    base_acc_by_bin[b] = float(np.mean(is_correct[mask]))
            method_quantile_rows.append(row)

    # Invariants 1 and 2.
    for m, counts in counts_by_method_bin.items():
        if int(np.sum(counts)) != n_test:
            raise AssertionError(
                f"Invariant failed: per-quantile counts do not sum to N_test for method={m}."
            )
    reference_counts = None
    reference_method_for_counts = None
    for m in method_index:
        if m in counts_by_method_bin:
            reference_counts = counts_by_method_bin[m]
            reference_method_for_counts = m
            break
    if reference_counts is None:
        raise AssertionError("Invariant failed: no methods available for bin count checks.")
    for m, counts in counts_by_method_bin.items():
        for b in range(n_bins):
            if counts[b] != reference_counts[b]:
                raise AssertionError(
                    f"Invariant failed: bin count mismatch across methods at quantile={bin_labels[b]} "
                    f"(method_a={reference_method_for_counts}, method_b={m})."
                )

    # Invariant 3.
    for row in method_quantile_rows:
        if row["method_name"] == "base_model" and row["count"] > 0:
            if float(row["delta_nll_vs_base"]) != 0.0:
                raise AssertionError(
                    f"Invariant failed: base_model delta_nll_vs_base is non-zero at quantile={row['quantile_bin']}."
                )

    method_quantile_csv = frontier_dir / "method_quantile_metrics.csv"
    method_quantile_json = frontier_dir / "method_quantile_metrics.json"
    _write_csv(method_quantile_csv, method_quantile_rows, METHOD_QUANTILE_COLUMNS)
    _write_json(
        method_quantile_json,
        {"run_id": run_id, "columns": METHOD_QUANTILE_COLUMNS, "rows": method_quantile_rows},
    )

    # Pairwise table.
    pairwise_rows: List[Dict[str, Any]] = []
    pairwise_comparisons = [args.reference_method] + [m for m in comparison_methods if m != args.reference_method]
    for comp in pairwise_comparisons:
        if comp not in method_probs:
            continue
        ref_probs = method_probs[args.reference_method]
        cmp_probs = method_probs[comp]
        ref_qy = np.clip(ref_probs[idx, labels], EPS, 1.0)
        cmp_qy = np.clip(cmp_probs[idx, labels], EPS, 1.0)
        ref_nll = -np.log(ref_qy)
        cmp_nll = -np.log(cmp_qy)
        ref_acc = (np.argmax(ref_probs, axis=1) == labels).astype(np.float64)
        cmp_acc = (np.argmax(cmp_probs, axis=1) == labels).astype(np.float64)
        for b in range(n_bins):
            mask = bin_idx == b
            count = int(np.sum(mask))
            q_lo = float(quantiles[b])
            q_hi = float(quantiles[b + 1])
            e_lo = float(edges[b])
            e_hi = float(edges[b + 1])
            if count == 0:
                row = {
                    "run_id": run_id,
                    "reference_method": args.reference_method,
                    "comparison_method": comp,
                    "quantile_bin": bin_labels[b],
                    "quantile_lo": q_lo,
                    "quantile_hi": q_hi,
                    "edge_lo": e_lo,
                    "edge_hi": e_hi,
                    "count": 0,
                    "nll_reference": None,
                    "nll_comparison": None,
                    "signed_nll_gap_method_minus_reference": None,
                    "frac_top_equals_truth_reference": None,
                    "frac_top_equals_truth_comparison": None,
                }
            else:
                ref_nll_bin = float(np.mean(ref_nll[mask]))
                cmp_nll_bin = float(np.mean(cmp_nll[mask]))
                row = {
                    "run_id": run_id,
                    "reference_method": args.reference_method,
                    "comparison_method": comp,
                    "quantile_bin": bin_labels[b],
                    "quantile_lo": q_lo,
                    "quantile_hi": q_hi,
                    "edge_lo": e_lo,
                    "edge_hi": e_hi,
                    "count": count,
                    "nll_reference": ref_nll_bin,
                    "nll_comparison": cmp_nll_bin,
                    "signed_nll_gap_method_minus_reference": float(cmp_nll_bin - ref_nll_bin),
                    "frac_top_equals_truth_reference": float(np.mean(ref_acc[mask])),
                    "frac_top_equals_truth_comparison": float(np.mean(cmp_acc[mask])),
                }
            pairwise_rows.append(row)

    # Invariant 5.
    for row in pairwise_rows:
        if (
            row["reference_method"] == row["comparison_method"]
            and row["count"] > 0
            and float(row["signed_nll_gap_method_minus_reference"]) != 0.0
        ):
            raise AssertionError(
                f"Invariant failed: reference self-gap is non-zero at quantile={row['quantile_bin']} "
                f"for reference_method={args.reference_method}."
            )

    pairwise_csv = frontier_dir / "reference_pairwise_quantile_comparison.csv"
    pairwise_json = frontier_dir / "reference_pairwise_quantile_comparison.json"
    _write_csv(pairwise_csv, pairwise_rows, PAIRWISE_COLUMNS)
    _write_json(pairwise_json, {"run_id": run_id, "columns": PAIRWISE_COLUMNS, "rows": pairwise_rows})

    # Base-correct decomposition.
    decomposition_rows: List[Dict[str, Any]] = []
    for m in method_index:
        if m not in method_probs:
            continue
        probs = method_probs[m]
        q_y = np.clip(probs[idx, labels], EPS, 1.0)
        neg_log_qy = -np.log(q_y)
        for b in range(n_bins):
            quant_mask = bin_idx == b
            q_lo = float(quantiles[b])
            q_hi = float(quantiles[b + 1])
            e_lo = float(edges[b])
            e_hi = float(edges[b + 1])
            for flag in [True, False]:
                mask = quant_mask & (base_correct == flag)
                count = int(np.sum(mask))
                row = {
                    "run_id": run_id,
                    "method_name": m,
                    "quantile_bin": bin_labels[b],
                    "quantile_lo": q_lo,
                    "quantile_hi": q_hi,
                    "edge_lo": e_lo,
                    "edge_hi": e_hi,
                    "base_correct_flag": bool(flag),
                    "count": count,
                    "mean_c_t": _to_float_or_none(_mean_or_none(c_t[mask])),
                    "mean_p_t_base_top_prob": _to_float_or_none(_mean_or_none(base_top_prob[mask])),
                    "mean_q_y": _to_float_or_none(_mean_or_none(q_y[mask])),
                    "mean_neg_log_q_y": _to_float_or_none(_mean_or_none(neg_log_qy[mask])),
                }
                decomposition_rows.append(row)

    decomp_csv = frontier_dir / "base_correctness_quantile_decomposition.csv"
    decomp_json = frontier_dir / "base_correctness_quantile_decomposition.json"
    _write_csv(decomp_csv, decomposition_rows, DECOMP_COLUMNS)
    _write_json(decomp_json, {"run_id": run_id, "columns": DECOMP_COLUMNS, "rows": decomposition_rows})

    # Invariant 4.
    for row in pairwise_rows:
        if row["count"] == 0:
            continue
        m = row["comparison_method"]
        q_label = row["quantile_bin"]
        q_bin = bin_labels.index(q_label)
        if not can_change_map.get(m, True):
            base_acc = base_acc_by_bin.get(q_bin)
            cmp_acc = row["frac_top_equals_truth_comparison"]
            if base_acc is None or cmp_acc is None:
                continue
            if float(cmp_acc) != float(base_acc):
                raise AssertionError(
                    f"Invariant failed: can_change_argmax=False method disagrees with base quantile accuracy "
                    f"(method={m}, quantile={q_label})."
                )


if __name__ == "__main__":
    main()
