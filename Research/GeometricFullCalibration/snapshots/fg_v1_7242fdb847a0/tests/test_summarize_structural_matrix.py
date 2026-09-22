"""
Tests for Experiments/summarize_structural_matrix.py.

Runs the script on a synthetic summary_metrics.json and verifies the output CSV.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SCRIPT = str(Path(__file__).resolve().parents[1] / "Experiments" / "summarize_structural_matrix.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _make_synthetic_summary(methods_list: list) -> dict:
    return {"methods": methods_list, "run_metadata": {"dataset": "synthetic"}}


def _synthetic_method_entry(
    method_name: str,
    sample_dep: bool,
    class_dep: bool,
    joint: bool,
    argmax_inv: bool,
    can_change: bool,
    net_flips: int = 5,
    accuracy: float = 0.75,
    nll: float = 1.0,
) -> dict:
    return {
        "method_name": method_name,
        "can_change_argmax": can_change,
        "metrics": {
            "accuracy": accuracy,
            "nll": nll,
            "top_label_ece": 0.05,
        },
        "structural_axes": {
            "sample_dependent": sample_dep,
            "class_dependent": class_dep,
            "joint_sample_class_dependent": joint,
            "argmax_invariant_by_construction": argmax_inv,
            "can_change_argmax": can_change,
            "notes": f"test entry for {method_name}",
        },
        "decision_audit": {
            "net_flips": net_flips,
            "net_flip_rate": net_flips / 1000.0,
            "argmax_change_rate": 0.1,
            "changed_to_correct_count": max(net_flips, 0),
            "changed_to_wrong_count": max(-net_flips, 0),
            "accuracy_delta": 0.001,
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_summary(tmp_path):
    methods = [
        _synthetic_method_entry("temperature_scaling", False, False, False, True, False,
                                net_flips=0, accuracy=0.73),
        _synthetic_method_entry("parameterized_temperature_scaling", True, False, False, True, False,
                                net_flips=0, accuracy=0.73),
        _synthetic_method_entry("glad_pi", True, True, True, False, True,
                                net_flips=42, accuracy=0.75),
        _synthetic_method_entry("aar_lightweight", True, True, False, False, True,
                                net_flips=8, accuracy=0.74),
        _synthetic_method_entry("trust_score_original_switch", True, True, False, False, True,
                                net_flips=3, accuracy=0.731),
    ]
    summary = _make_synthetic_summary(methods)
    json_path = tmp_path / "summary_metrics.json"
    with open(json_path, "w") as f:
        json.dump(summary, f)
    return json_path, tmp_path


# ---------------------------------------------------------------------------
# Tests: script invocation
# ---------------------------------------------------------------------------

class TestSummarizeStructuralMatrix:
    def test_script_runs_successfully(self, synthetic_summary):
        json_path, tmp_path = synthetic_summary
        output_csv = tmp_path / "structural_matrix.csv"

        result = subprocess.run(
            [sys.executable, SCRIPT,
             "--summary_json", str(json_path),
             "--output_csv", str(output_csv)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert output_csv.exists(), "Output CSV was not created"

    def test_output_csv_has_correct_columns(self, synthetic_summary):
        json_path, tmp_path = synthetic_summary
        output_csv = tmp_path / "structural_matrix.csv"

        subprocess.run(
            [sys.executable, SCRIPT,
             "--summary_json", str(json_path),
             "--output_csv", str(output_csv)],
            capture_output=True, text=True, timeout=30,
        )

        with open(output_csv, newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []

        required = [
            "method_name", "sample_dependent", "class_dependent",
            "joint_sample_class_dependent", "argmax_invariant_by_construction",
            "can_change_argmax", "net_flips", "accuracy",
        ]
        for col in required:
            assert col in cols, f"Output CSV missing column: {col}"

    def test_output_csv_has_correct_row_count(self, synthetic_summary):
        json_path, tmp_path = synthetic_summary
        output_csv = tmp_path / "structural_matrix.csv"

        subprocess.run(
            [sys.executable, SCRIPT,
             "--summary_json", str(json_path),
             "--output_csv", str(output_csv)],
            capture_output=True, text=True, timeout=30,
        )

        with open(output_csv, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 5, f"Expected 5 rows, got {len(rows)}"

    def test_argmax_invariant_methods_sorted_first(self, synthetic_summary):
        json_path, tmp_path = synthetic_summary
        output_csv = tmp_path / "structural_matrix.csv"

        subprocess.run(
            [sys.executable, SCRIPT,
             "--summary_json", str(json_path),
             "--output_csv", str(output_csv)],
            capture_output=True, text=True, timeout=30,
        )

        with open(output_csv, newline="") as f:
            rows = list(csv.DictReader(f))

        # argmax_invariant_by_construction=True methods should come first
        inv_methods = [r for r in rows if r["argmax_invariant_by_construction"] == "True"]
        non_inv_methods = [r for r in rows if r["argmax_invariant_by_construction"] == "False"]
        if inv_methods and non_inv_methods:
            # All invariant rows must appear before any non-invariant row (by sort order)
            last_inv_idx = max(rows.index(r) for r in inv_methods)
            first_non_inv_idx = min(rows.index(r) for r in non_inv_methods)
            assert last_inv_idx < first_non_inv_idx, "Argmax-invariant methods should sort first"

    def test_glad_pi_net_flips_value_preserved(self, synthetic_summary):
        json_path, tmp_path = synthetic_summary
        output_csv = tmp_path / "structural_matrix.csv"

        subprocess.run(
            [sys.executable, SCRIPT,
             "--summary_json", str(json_path),
             "--output_csv", str(output_csv)],
            capture_output=True, text=True, timeout=30,
        )

        with open(output_csv, newline="") as f:
            rows = {r["method_name"]: r for r in csv.DictReader(f)}

        assert rows["glad_pi"]["net_flips"] == "42"
        assert rows["temperature_scaling"]["net_flips"] == "0"

    def test_empty_summary_produces_header_only(self, tmp_path):
        """An empty methods list should produce a CSV with only the header row."""
        json_path = tmp_path / "empty_summary.json"
        output_csv = tmp_path / "empty_structural_matrix.csv"
        with open(json_path, "w") as f:
            json.dump({"methods": []}, f)

        result = subprocess.run(
            [sys.executable, SCRIPT,
             "--summary_json", str(json_path),
             "--output_csv", str(output_csv)],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0
        assert output_csv.exists()

        with open(output_csv, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 0

    def test_method_missing_structural_axes_still_emits_row(self, tmp_path):
        """Methods without structural_axes should still appear, with None values."""
        methods = [
            {
                "method_name": "some_external_method",
                "can_change_argmax": True,
                "metrics": {"accuracy": 0.8, "nll": 0.9},
                "decision_audit": {"net_flips": 10, "accuracy_delta": 0.02,
                                   "changed_to_correct_count": 15, "changed_to_wrong_count": 5},
            }
        ]
        json_path = tmp_path / "partial_summary.json"
        output_csv = tmp_path / "partial_structural_matrix.csv"
        with open(json_path, "w") as f:
            json.dump({"methods": methods}, f)

        subprocess.run(
            [sys.executable, SCRIPT,
             "--summary_json", str(json_path),
             "--output_csv", str(output_csv)],
            capture_output=True, text=True, timeout=30,
        )

        with open(output_csv, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 1
        assert rows[0]["method_name"] == "some_external_method"
        assert rows[0]["net_flips"] == "10"
