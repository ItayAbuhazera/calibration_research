#!/usr/bin/env python3
"""Aggregate the preregistered five-seed recoverability experiment."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rgc_shift.recoverability import proper_scores, select_blend_alpha

from scripts.export_recoverability import (
    BETA,
    CHECKPOINT_SHA256,
    CORRUPTIONS,
    K_RADIUS,
    K_VOTE,
    SEEDS,
    SEVERITIES,
    TEMPERATURE,
    _assert_arm_equivalence,
    _validate_export,
)


CORRUPTED_CELLS = tuple(
    f"{corruption}_s{severity}"
    for corruption in CORRUPTIONS
    for severity in SEVERITIES
)
ARMS = ("perclass", "globalknn")
BLEND_ALPHAS = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
CAVEAT = (
    "These five seeds vary the trained checkpoint (and, where the same seed drives it, "
    "RGC's internal randomisation) with the corruption set held fixed. The resulting "
    "interval measures variation across training runs, NOT sampling uncertainty over a "
    "population of corruption types. It is a valid internal go/no-go criterion because "
    "it is pre-registered here, and it must not later be presented as five independent "
    "samples from the CIFAR-C distribution."
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _cell_map(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["name"]: row for row in result["cells"]}


def _t_ci(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    if len(array) < 2:
        low = high = mean
    else:
        sem = float(stats.sem(array))
        if sem == 0.0:
            low = high = mean
        else:
            low, high = stats.t.interval(
                0.95, df=len(array) - 1, loc=mean, scale=sem
            )
    return {
        "n": int(len(array)),
        "mean": mean,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "values": array.tolist(),
    }


def _macro(values: list[float]) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(array))


def _vector_scores(export_root: Path, seed: int, cell_name: str) -> dict[str, float]:
    path = export_root / f"seed{seed}" / "globalknn" / f"{cell_name}.npz"
    with np.load(path, allow_pickle=False) as payload:
        # Call the package function directly so clipping (1e-7), classwise
        # Brier definition, and accuracy convention are identical to harness
        # scores rather than merely numerically similar.
        return proper_scores(payload["vector_probabilities"], payload["y_true"])


def _audit_exports(export_root: Path) -> dict[str, Any]:
    expected_cells = {"val", "clean", *CORRUPTED_CELLS}
    audited = 0
    for seed in SEEDS:
        arm_cells = {
            arm: {path.stem for path in (export_root / f"seed{seed}" / arm).glob("*.npz")}
            for arm in ARMS
        }
        if any(cells != expected_cells for cells in arm_cells.values()):
            raise RuntimeError(f"seed {seed}: unexpected export cell sets {arm_cells}")
        for cell_name in sorted(expected_cells):
            payloads: dict[str, dict[str, np.ndarray]] = {}
            for arm in ARMS:
                path = export_root / f"seed{seed}" / arm / f"{cell_name}.npz"
                with np.load(path, allow_pickle=False) as stored:
                    payload = {key: np.asarray(stored[key]) for key in stored.files}
                _validate_export(payload, arm, cell_name)
                n = 5_000 if cell_name == "val" else 10_000
                exact_shapes = {
                    "sample_id": (n,),
                    "sample_id_logits": (n,),
                    "sample_id_geometry": (n,),
                    "sample_id_labels": (n,),
                    "y_true": (n,),
                    "y_pred_head": (n,),
                    "y_pred_knn": (n,),
                    "y_pred_vector": (n,),
                    "neighbour_labels": (n, K_VOTE),
                    "neighbours": (n, K_VOTE),
                    "knn_counts": (n, 100),
                    "contrast": (n,),
                    "logits": (n, 100),
                    "knn_radius": (n,),
                    "neighbour_margin": (n,),
                    "temperature": (n,),
                    "vector_probabilities": (n, 100),
                }
                if arm == "globalknn":
                    exact_shapes.update(
                        {
                            "head_probabilities": (n, 100),
                            "knn_distribution": (n, 100),
                        }
                    )
                for key, shape in exact_shapes.items():
                    if payload[key].shape != shape:
                        raise RuntimeError(
                            f"seed {seed}/{cell_name}/{arm}: {key} shape "
                            f"{payload[key].shape} != {shape}"
                        )
                payloads[arm] = payload
            _assert_arm_equivalence(
                payloads["perclass"], payloads["globalknn"], f"seed{seed}/{cell_name}"
            )
            audited += 1
    return {
        "status": "pass",
        "seed_cell_pairs": audited,
        "files": 2 * audited,
        "expected_cells_per_arm": len(expected_cells),
        "shared_array_equality": "all shared keys equal; y_pred_knn exempted",
        "global_only_keys": ["head_probabilities", "knn_distribution"],
    }


def aggregate(args: argparse.Namespace) -> dict[str, Any]:
    export_audit = _audit_exports(args.export_root)
    primary: dict[tuple[int, str], dict[str, Any]] = {}
    mechanism: dict[tuple[int, str], dict[str, Any]] = {}
    for seed in SEEDS:
        for arm in ARMS:
            primary[(seed, arm)] = _load_json(
                args.results_root / f"seed{seed}" / f"recoverability_{arm}.json"
            )
            mechanism[(seed, arm)] = _load_json(
                args.results_root
                / f"seed{seed}"
                / f"recoverability_{arm}_mechanism_cleanfit.json"
            )

    seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        for arm in ARMS:
            result = primary[(seed, arm)]
            cells = _cell_map(result)
            missing = set(CORRUPTED_CELLS) - set(cells)
            if missing:
                raise RuntimeError(f"seed {seed}/{arm}: missing cells {sorted(missing)}")
            gated = result.get("accuracy_by_arm", {})
            rows = [cells[name] for name in CORRUPTED_CELLS]
            gate_acc = [gated[name]["gated"] for name in CORRUPTED_CELLS]
            head_acc = [row["acc_head"] for row in rows]
            vector_acc = [row["acc_vector"] for row in rows]
            headroom = [row["oracle_headroom_pp"] for row in rows]
            captured = [
                (gate - head) / (room / 100.0) if room > 1e-9 else np.nan
                for gate, head, room in zip(gate_acc, head_acc, headroom)
            ]
            row = {
                "seed": seed,
                "arm": arm,
                "head_accuracy": _macro(head_acc),
                "knn_accuracy": _macro([item["acc_knn"] for item in rows]),
                "vector_accuracy": _macro(vector_acc),
                "gate_accuracy": _macro(gate_acc),
                "oracle_union_accuracy": _macro(
                    [item["acc_oracle_union"] for item in rows]
                ),
                "oracle_headroom_pp": _macro(headroom),
                "headroom_captured": _macro(captured),
                "gate_minus_head_pp": 100.0 * _macro(
                    [gate - head for gate, head in zip(gate_acc, head_acc)]
                ),
                "gate_minus_vector_pp": 100.0 * _macro(
                    [gate - vector for gate, vector in zip(gate_acc, vector_acc)]
                ),
                "harness_decision": result["decision"]["decision"],
                "harness_scope": result["decision"]["scope"],
                "blend_alpha": result.get("blend_alpha"),
                "gate_features": (result.get("gate") or {}).get("features"),
                "gate_threshold": (result.get("gate") or {}).get("threshold"),
                "n_train_decisive": (result.get("gate") or {}).get(
                    "n_train_decisive"
                ),
            }
            seed_rows.append(row)

    paired: dict[str, Any] = {}
    for arm in ARMS:
        arm_rows = [row for row in seed_rows if row["arm"] == arm]
        for metric in ("gate_minus_head_pp", "gate_minus_vector_pp"):
            paired[f"{arm}_{metric}"] = _t_ci(
                [float(row[metric]) for row in arm_rows]
            )

    by_seed_arm = {(row["seed"], row["arm"]): row for row in seed_rows}
    for metric, output_name, scale in (
        ("gate_accuracy", "gate_accuracy_pp", 100.0),
        ("oracle_headroom_pp", "oracle_headroom_pp", 1.0),
        ("headroom_captured", "headroom_captured", 1.0),
        ("knn_accuracy", "knn_accuracy_pp", 100.0),
    ):
        paired[f"globalknn_minus_perclass_{output_name}"] = _t_ci(
            [
                scale
                * (
                    float(by_seed_arm[(seed, "globalknn")][metric])
                    - float(by_seed_arm[(seed, "perclass")][metric])
                )
                for seed in SEEDS
            ]
        )

    per_cell: list[dict[str, Any]] = []
    for arm in ARMS:
        for cell_name in ("clean", *CORRUPTED_CELLS):
            rows = [_cell_map(primary[(seed, arm)])[cell_name] for seed in SEEDS]
            gates = [
                primary[(seed, arm)]["accuracy_by_arm"][cell_name]["gated"]
                for seed in SEEDS
            ]
            for metric, values in (
                ("acc_head", [row["acc_head"] for row in rows]),
                ("acc_knn", [row["acc_knn"] for row in rows]),
                ("acc_vector", [row["acc_vector"] for row in rows]),
                ("acc_oracle_union", [row["acc_oracle_union"] for row in rows]),
                ("oracle_headroom_pp", [row["oracle_headroom_pp"] for row in rows]),
                ("acc_gated", gates),
            ):
                per_cell.append(
                    {"arm": arm, "cell": cell_name, "metric": metric, **_t_ci(values)}
                )

    proper: list[dict[str, Any]] = []
    for seed in SEEDS:
        result = primary[(seed, "globalknn")]
        for cell_name in ("val", "clean", *CORRUPTED_CELLS):
            for method, values in result.get("proper_scores", {}).get(
                cell_name, {}
            ).items():
                proper.append(
                    {
                        "seed": seed,
                        "cell": cell_name,
                        "method": method,
                        **values,
                    }
                )
            proper.append(
                {
                    "seed": seed,
                    "cell": cell_name,
                    "method": "published_full_vector_single_layer_beta30",
                    **_vector_scores(args.export_root, seed, cell_name),
                }
            )

    proper_macro: list[dict[str, Any]] = []
    proper_lookup = {
        (row["seed"], row["cell"], row["method"]): row for row in proper
    }
    proper_methods = sorted({row["method"] for row in proper})
    for seed in SEEDS:
        for method in proper_methods:
            selected = [
                proper_lookup[(seed, cell_name, method)]
                for cell_name in CORRUPTED_CELLS
                if (seed, cell_name, method) in proper_lookup
            ]
            if len(selected) != len(CORRUPTED_CELLS):
                continue
            proper_macro.append(
                {
                    "seed": seed,
                    "source_arm": "globalknn",
                    "method": method,
                    "accuracy": _macro([row["accuracy"] for row in selected]),
                    "nll": _macro([row["nll"] for row in selected]),
                    "brier": _macro([row["brier"] for row in selected]),
                    "n_cells": len(selected),
                }
            )

    proper_per_cell: list[dict[str, Any]] = []
    for cell_name in ("clean", *CORRUPTED_CELLS):
        for method in proper_methods:
            rows = [
                proper_lookup[(seed, cell_name, method)]
                for seed in SEEDS
                if (seed, cell_name, method) in proper_lookup
            ]
            if len(rows) != len(SEEDS):
                continue
            for metric in ("accuracy", "nll", "brier"):
                proper_per_cell.append(
                    {
                        "source_arm": "globalknn",
                        "cell": cell_name,
                        "method": method,
                        "metric": metric,
                        **_t_ci([row[metric] for row in rows]),
                    }
                )

    mechanism_rows = []
    for seed in SEEDS:
        for arm in ARMS:
            for row in mechanism[(seed, arm)]["cells"]:
                if row["name"] in CORRUPTED_CELLS:
                    mechanism_rows.append(
                        {
                            "seed": seed,
                            "arm": arm,
                            "cell": row["name"],
                            "neighbour_jaccard_vs_clean": row[
                                "neighbour_jaccard_vs_clean"
                            ],
                            "contrast_spearman_vs_clean": row[
                                "contrast_spearman_vs_clean"
                            ],
                        }
                    )

    mechanism_descriptive: list[dict[str, Any]] = []
    for arm in ARMS:
        for cell_name in CORRUPTED_CELLS:
            rows = [
                row
                for row in mechanism_rows
                if row["arm"] == arm and row["cell"] == cell_name
            ]
            for metric in (
                "neighbour_jaccard_vs_clean",
                "contrast_spearman_vs_clean",
            ):
                mechanism_descriptive.append(
                    {
                        "arm": arm,
                        "cell": cell_name,
                        "metric": metric,
                        **_t_ci([row[metric] for row in rows]),
                    }
                )

    clean_sweeps: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        val_path = args.export_root / f"seed{seed}" / "globalknn" / "val.npz"
        with np.load(val_path, allow_pickle=False) as payload:
            selected, sweep = select_blend_alpha(
                payload["head_probabilities"],
                payload["knn_distribution"],
                payload["y_true"],
                BLEND_ALPHAS,
                metric="nll",
            )
        clean_sweeps[seed] = {
            "selected_alpha": selected,
            "validation_sweep": {str(alpha): values for alpha, values in sweep.items()},
        }

    tie_statistics = {
        seed: {
            path.stem: _load_json(path)
            for path in sorted(
                (args.artifact_root / f"seed{seed}" / "tie_stats").glob("*.json")
            )
        }
        for seed in SEEDS
    }

    return {
        "primary_clean_name": "val",
        "export_audit": export_audit,
        "primary_cells": list(CORRUPTED_CELLS),
        "primary_excludes": ["val", "clean"],
        "mechanism_clean_name": "clean",
        "seed_macro_averages": seed_rows,
        "paired_t_confidence_intervals": paired,
        "per_cell_descriptive": per_cell,
        "proper_scores_globalknn_and_vector": proper,
        "proper_score_seed_macro_averages": proper_macro,
        "proper_score_per_cell_descriptive": proper_per_cell,
        "mechanism_diagnostics": mechanism_rows,
        "mechanism_per_cell_descriptive": mechanism_descriptive,
        "clean_blend_sweeps": clean_sweeps,
        "tie_statistics": tie_statistics,
        "caveat": CAVEAT,
        "harness_clean_name_limitation": (
            "The CLI fitted on --clean-name val, but packaged decide() excludes only a cell "
            "literally named clean. Literal per-seed harness verdicts therefore include val; "
            "the preregistered cross-seed analysis above explicitly uses only 12 corruptions."
        ),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt_ci(row: dict[str, Any], scale: float = 1.0) -> str:
    return (
        f"{scale * row['mean']:.4f} "
        f"[{scale * row['ci95_low']:.4f}, {scale * row['ci95_high']:.4f}]"
    )


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def _write_reports(args: argparse.Namespace, aggregate: dict[str, Any]) -> None:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    _write_csv(
        args.out.with_name("recoverability_seed_macro.csv"),
        aggregate["seed_macro_averages"],
    )
    _write_csv(
        args.out.with_name("recoverability_per_cell.csv"),
        aggregate["per_cell_descriptive"],
    )
    _write_csv(
        args.out.with_name("recoverability_proper_scores.csv"),
        aggregate["proper_scores_globalknn_and_vector"],
    )
    _write_csv(
        args.out.with_name("recoverability_proper_score_seed_macro.csv"),
        aggregate["proper_score_seed_macro_averages"],
    )
    _write_csv(
        args.out.with_name("recoverability_proper_score_per_cell.csv"),
        aggregate["proper_score_per_cell_descriptive"],
    )
    _write_csv(
        args.out.with_name("recoverability_mechanism_diagnostics.csv"),
        aggregate["mechanism_diagnostics"],
    )
    _write_csv(
        args.out.with_name("recoverability_mechanism_per_cell.csv"),
        aggregate["mechanism_per_cell_descriptive"],
    )
    tie_rows = [
        {"seed": seed, "cell": cell, **values}
        for seed, cells in aggregate["tie_statistics"].items()
        for cell, values in cells.items()
    ]
    _write_csv(args.out.with_name("recoverability_tie_statistics.csv"), tie_rows)

    paired = aggregate["paired_t_confidence_intervals"]
    lines = [
        "# Accuracy-recoverability summary",
        "",
        "Primary estimates fit the rule on each seed's validation split and macro-average "
        "only the 12 corruptions. Clean-test-fitted runs are mechanism-only.",
        "",
        "## Honest primary readout",
        "",
    ]
    for arm in ARMS:
        rows = [
            row for row in aggregate["seed_macro_averages"] if row["arm"] == arm
        ]
        mean_headroom = float(np.mean([row["oracle_headroom_pp"] for row in rows]))
        mean_captured = float(np.mean([row["headroom_captured"] for row in rows]))
        vector_ci = paired[f"{arm}_gate_minus_vector_pp"]
        beats_vector = vector_ci["ci95_low"] > 0.0
        decisions = sorted({row["harness_decision"] for row in rows})
        lines.append(
            f"- `{arm}`: oracle-union headroom {mean_headroom:.3f}pp; gate captured "
            f"{mean_captured:.1%}; gate−published-vector {_fmt_ci(vector_ci)} pp "
            f"({'paired CI excludes 0' if beats_vector else 'paired CI includes 0'}); "
            f"literal harness decision(s): {', '.join(decisions)}."
        )
    gate_comparison = paired["globalknn_minus_perclass_gate_accuracy_pp"]
    headroom_comparison = paired[
        "globalknn_minus_perclass_oracle_headroom_pp"
    ]
    lines.append(
        f"- Primary geometry comparison, globalknn−perclass: gate accuracy "
        f"{_fmt_ci(gate_comparison)} pp; oracle headroom "
        f"{_fmt_ci(headroom_comparison)} pp."
    )
    lines.extend(
        [
            "",
            "| Paired seed-level difference | Mean [95% t CI] |",
            "|---|---:|",
        ]
    )
    for name, row in paired.items():
        lines.append(f"| `{name}` | {_fmt_ci(row)} |")
    lines.extend(["", "## Per-seed harness decisions", ""])
    for row in aggregate["seed_macro_averages"]:
        lines.append(
            f"- Seed {row['seed']} `{row['arm']}`: **{row['harness_decision']}**, "
            f"oracle headroom {row['oracle_headroom_pp']:.3f}pp, captured "
            f"{row['headroom_captured']:.1%}, gate−vector "
            f"{row['gate_minus_vector_pp']:+.3f}pp."
        )
    lines.extend(
        [
            "",
            "## Seed-level proper-score macro-averages (12 corruptions)",
            "",
            "| Seed | Probability arm | Accuracy | NLL | Brier |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for row in aggregate["proper_score_seed_macro_averages"]:
        lines.append(
            f"| {row['seed']} | `{row['method']}` | {row['accuracy']:.4f} | "
            f"{row['nll']:.4f} | {row['brier']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Package limitation: " + aggregate["harness_clean_name_limitation"],
            "",
            CAVEAT,
            "",
        ]
    )
    args.out.with_name("recoverability_summary.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

    preflights = {
        seed: _load_json(
            args.artifact_root / f"seed{seed}" / "preflight_complete.json"
        )
        for seed in SEEDS
    }
    tie_stats = aggregate["tie_statistics"]
    under_resolved = [
        seed
        for seed in SEEDS
        if tie_stats[seed]["clean"][
            "fraction_top1_count_tied_before_distance_break"
        ]
        > 0.25
    ]
    card = [
        "# ResNet-101 / CIFAR-100-C accuracy-recoverability experiment card",
        "",
        f"- RGCL base git commit: `{_git_sha()}`; the approved exporter adapter ran from "
        "the current uncommitted working tree.",
        "- rgc-shift: `0.1.0`, git `ff81032fbf646b6e7c4d34eef906e1b49855a182`",
        "- rgc-shift pre-run gate: `56 passed in 19.84s`; console log "
        "`rgc_shift_tests.log`.",
        f"- Python used for aggregation: `{platform.python_version()}`",
        "- Seed source: user-pinned `1,2,3,4,5`, matching "
        "`results/unified_benchmark_c100_resnet101_seed<S>_rankgeom_mix/summary_metrics.json`.",
        f"- Seeds: `{list(SEEDS)}`; the seed selects checkpoint, clean split, RGC layers, and projection.",
        f"- Corruptions: `{list(CORRUPTIONS)}`; severities: `{list(SEVERITIES)}`",
        "- CIFAR-100-C source: `/home/itayab/PyCharmProjects/geometric/"
        "GeometricInternalCalibration/GeometricInternalCalibration/data/cifar100-c`; "
        "for every corruption, all five severity slices `[(severity-1)*10000:severity*10000]` "
        "were checked elementwise against the canonical 10,000 CIFAR-100 test labels and matched.",
        f"- `k_vote={K_VOTE}`, `k_radius={K_RADIUS}`; true Euclidean L2.",
        "- Vote ties: modal classes are resolved by smallest within-class mean neighbour distance.",
        "- kNN probability rule: fixed Laplace smoothing `(count + 1) / (200 + 100)`.",
        "- Tie/probability convention: `y_pred_knn` uses the pinned distance tie-break, "
        "whereas the exact Laplace distribution remains count-symmetric across tied classes. "
        "Accuracy claims use `y_pred_knn`/the gate's hard choice; NLL and Brier use the "
        "unchanged Laplace probabilities (and their package-reported argmax accuracy is "
        "listed only with the proper-score table).",
        f"- Temperature: `{TEMPERATURE}`, inherited and validation-selected in each summary's `temperature_scaling` row.",
        f"- Constant arm: published full-vector arm, single-layer source, beta={BETA:g}; validation-selected in each summary's `full_vector_distance_fusion` row.",
        "- Pinned-config confirmation: all five established summaries share method-config "
        "SHA-256 `4504d9b302986475730b4853905092522ee5f8d2061a3f63d8feb29c0b497cc2`.",
        "- Selection-split evidence: `Experiments/run_unified_benchmark.py` materializes "
        "`val_loader` as validation (stage 1), fits temperature scaling from `logits_val`/"
        "`val_labels`, and records `tuning_split=validation`; `_run_full_vector_fusion` "
        "passes `val_features`, `val_labels`, and `val_base_probs` to `fusion.fit`, while "
        "the stored full-vector row records `tuning_split=validation`.",
        "- Expected-accuracy confirmation: each established summary records a 10,000-row "
        "test split, and its base/argmax-invariant accuracy equals the corresponding pin.",
        "- Pass A / primary: `--clean-name val`; paired statistics use only 12 corruptions.",
        "- Pass B / mechanism: `--clean-name clean`; used only for Jaccard and Spearman diagnostics.",
        f"- Post-run export audit: `{aggregate['export_audit']}`.",
        "",
        "## Existing method components reused",
        "",
        "- Feature extraction, SPP, projection, equal layer aggregation, and sklearn L2 "
        "normalisation: `Experiments/run_rgc_experiments.py::"
        "extract_and_aggregate_sgc_features`.",
        "- Seeded layer draw: `Experiments/run_rgc_experiments.py::select_random_rgc_layers`, "
        "factored without changing the original selection block.",
        "- Per-class distances and contrast: `utils/stability_space.py::"
        "StabilitySpace.calc_per_class_1nn_distances` and `StabilitySpace.calc_stab`.",
        "- Head logits: the checkpoint model's existing `forward`, observed during the same "
        "forward pass used for RGCL feature extraction.",
        "- Published constant arm: `Calibrators/geometric_calibrator.py::"
        "FullVectorDistanceFusionCalibrator.calibrate`, using its published single `layer3` "
        "global-average feature source and frozen beta 30.",
        "- Measurement-only global index: `sklearn.neighbors.NearestNeighbors`, fitted once "
        "per seed to the already-returned RGCL embedding with `metric=l2` and queried once "
        "at k=200 per cell.",
        "",
        "## Primary results (validation-fitted; 12-corruption macro-averages)",
        "",
        "| Seed | Arm | Head | Geometry | Published vector | Gate | Oracle union | Headroom pp | Captured | Harness |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in aggregate["seed_macro_averages"]:
        card.append(
            f"| {row['seed']} | `{row['arm']}` | {row['head_accuracy']:.4f} | "
            f"{row['knn_accuracy']:.4f} | {row['vector_accuracy']:.4f} | "
            f"{row['gate_accuracy']:.4f} | {row['oracle_union_accuracy']:.4f} | "
            f"{row['oracle_headroom_pp']:.3f} | {row['headroom_captured']:.1%} | "
            f"{row['harness_decision']} |"
        )
    card.extend(
        [
            "",
            "Paired 95% t confidence intervals across the five seed-level corruption "
            "macro-averages:",
            "",
        ]
    )
    for name, interval in aggregate["paired_t_confidence_intervals"].items():
        card.append(f"- `{name}`: {_fmt_ci(interval)}")
    card.extend([
        "",
        "## Checkpoints and clean sanity",
        "",
        "| Seed | Checkpoint path | SHA-256 | Expected | Measured |",
        "|---:|---|---|---:|---:|",
    ])
    for seed in SEEDS:
        meta = preflights[seed]
        card.append(
            f"| {seed} | `{meta['checkpoint']}` | `{CHECKPOINT_SHA256[seed]}` | "
            f"{meta['expected_clean_accuracy']:.4f} | {meta['measured_clean_accuracy']:.4f} |"
        )
    card.extend(["", "## Clean tie resolution", ""])
    if under_resolved:
        card.append(
            "**PROMINENT WARNING:** clean tie fraction exceeds 0.25 for seed(s) "
            + ", ".join(map(str, under_resolved))
            + "; the 200-neighbour vote remains under-resolved."
        )
    for seed in SEEDS:
        row = tie_stats[seed]["clean"]
        card.append(
            f"- Seed {seed}: mean top-1 count {row['mean_top1_vote_count']:.3f}, "
            f"median {row['median_top1_vote_count']:.3f}, pre-break tie fraction "
            f"{row['fraction_top1_count_tied_before_distance_break']:.3f}."
        )
    card.extend(["", "## Clean-fitted parameters and controls", ""])
    seed_rows = {
        (row["seed"], row["arm"]): row for row in aggregate["seed_macro_averages"]
    }
    for seed in SEEDS:
        global_row = seed_rows[(seed, "globalknn")]
        sweep = aggregate["clean_blend_sweeps"][seed]
        card.append(
            f"- Seed {seed} `globalknn` blend: alpha={global_row['blend_alpha']}; "
            f"validation alpha sweep="
            f"`{sweep['validation_sweep']}`."
        )
        for arm in ARMS:
            row = seed_rows[(seed, arm)]
            card.append(
                f"- Seed {seed} `{arm}` gate: threshold={row['gate_threshold']}, "
                f"features={row['gate_features']}, "
                f"n_train_decisive={row['n_train_decisive']}."
            )
    for seed in SEEDS:
        controls = preflights[seed]["controls"]
        perclass_control = controls["control1_geometry"]["perclass"]
        global_control = controls["control1_geometry"]["globalknn"]
        card.append(
            f"- Seed {seed}: Control 1 logits PASS (harness rc "
            f"{controls['control1_logits']['harness_returncode']}); geometry PASS — "
            f"perclass intact/permuted={perclass_control['intact_head_geometry_agreement']:.4f}/"
            f"{perclass_control['permuted_head_geometry_agreement']:.4f}, globalknn="
            f"{global_control['intact_head_geometry_agreement']:.4f}/"
            f"{global_control['permuted_head_geometry_agreement']:.4f}; Control 2 "
            f"order-only labels/logits/geometry for both arms PASS."
        )
    card.extend(["", "## Runtime environments", ""])
    for seed in SEEDS:
        for phase in ("preflight", "corruptions"):
            environment_log = (
                PROJECT_ROOT
                / "logs"
                / "recoverability"
                / f"seed{seed}"
                / f"{phase}_environment.log"
            )
            if environment_log.exists():
                card.extend(
                    [
                        f"### Seed {seed}, {phase}",
                        "",
                        "```text",
                        environment_log.read_text(encoding="utf-8").strip(),
                        "```",
                        "",
                    ]
                )
    card.extend(
        [
            "",
            "## Interpretation and known harness behavior",
            "",
            aggregate["harness_clean_name_limitation"],
            "",
            CAVEAT,
            "",
        ]
    )
    args.out.with_name("recoverability_experiment_card.md").write_text(
        "\n".join(card), encoding="utf-8"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=PROJECT_ROOT / "results")
    parser.add_argument("--export-root", type=Path, default=PROJECT_ROOT / "exports")
    parser.add_argument(
        "--artifact-root", type=Path, default=PROJECT_ROOT / "artifacts" / "recoverability"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "results" / "recoverability_aggregate.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = aggregate(args)
    _write_reports(args, result)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
