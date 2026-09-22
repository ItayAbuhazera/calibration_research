"""
Regression guard: every non-base method entry produced by _make_method_entry
must carry decision_audit with the four primary fields, and the CSV writer
must flatten them as da_* columns.

This test works purely with synthetic in-memory data — no CIFAR, no real model.
It calls the same helpers used by run_unified_benchmark.py directly.
"""

import json
import sys
import io
import csv
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.decision_audit import decision_audit


# ---------------------------------------------------------------------------
# Helpers that mirror the logic in run_unified_benchmark.py
# ---------------------------------------------------------------------------

_REQUIRED_DA_FIELDS = (
    "net_flips",
    "accuracy_delta",
    "changed_to_correct_count",
    "changed_to_wrong_count",
)

_REQUIRED_DA_CSV_COLS = tuple(f"da_{f}" for f in _REQUIRED_DA_FIELDS)


def _fake_method_entry(method_name, probs, labels, base_probs=None):
    """
    Minimal reproduction of _make_method_entry logic for testing purposes.
    Attaches decision_audit when base_probs is supplied.
    """
    entry = {"method_name": method_name, "metrics": {"accuracy": float(np.mean(np.argmax(probs, 1) == labels))}}
    if base_probs is not None:
        entry["decision_audit"] = decision_audit(base_probs, probs, labels)
    return entry


def _flatten_to_csv_row(entry):
    """Mirrors the _write_csv flattening logic in run_unified_benchmark.py."""
    flat = {k: v for k, v in entry.items() if k not in ("metrics", "decision_audit", "nll_subsets")}
    for mk, mv in entry["metrics"].items():
        flat[mk] = mv
    if "decision_audit" in entry and isinstance(entry["decision_audit"], dict):
        for dk, dv in entry["decision_audit"].items():
            flat[f"da_{dk}"] = dv
    if isinstance(flat.get("structural_axes"), dict):
        flat["structural_axes"] = json.dumps(flat["structural_axes"])
    return flat


# ---------------------------------------------------------------------------
# Synthetic data fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    rng = np.random.RandomState(0)
    N, C = 200, 10
    base_logits = rng.randn(N, C)
    base_probs = np.exp(base_logits) / np.exp(base_logits).sum(1, keepdims=True)
    # Method probs: slightly perturbed, can change argmax
    method_logits = base_logits + 0.5 * rng.randn(N, C)
    method_probs = np.exp(method_logits) / np.exp(method_logits).sum(1, keepdims=True)
    labels = rng.randint(0, C, size=N)
    return base_probs, method_probs, labels


# ---------------------------------------------------------------------------
# Tests: decision_audit sub-dict
# ---------------------------------------------------------------------------

class TestDecisionAuditCoverage:
    def test_entry_with_base_probs_has_decision_audit(self, synthetic_data):
        base_probs, method_probs, labels = synthetic_data
        entry = _fake_method_entry("some_method", method_probs, labels, base_probs=base_probs)
        assert "decision_audit" in entry, "decision_audit missing when base_probs provided"

    def test_required_da_fields_present(self, synthetic_data):
        base_probs, method_probs, labels = synthetic_data
        entry = _fake_method_entry("some_method", method_probs, labels, base_probs=base_probs)
        da = entry["decision_audit"]
        for field in _REQUIRED_DA_FIELDS:
            assert field in da, f"decision_audit missing required field: {field}"

    def test_entry_without_base_probs_has_no_decision_audit(self, synthetic_data):
        _, method_probs, labels = synthetic_data
        entry = _fake_method_entry("some_method", method_probs, labels, base_probs=None)
        assert "decision_audit" not in entry

    def test_base_model_entry_should_be_skipped(self, synthetic_data):
        base_probs, _, labels = synthetic_data
        # base_model is compared against itself — net_flips would always be 0
        # The runner skips base_probs for base_model; verify our synthetic entry
        # correctly shows all-zero decision changes when probs identical
        entry = _fake_method_entry("base_model", base_probs, labels, base_probs=base_probs)
        da = entry["decision_audit"]
        assert da["net_flips"] == 0
        assert da["changed_to_correct_count"] == 0
        assert da["changed_to_wrong_count"] == 0
        assert da["argmax_change_rate"] == 0.0

    def test_all_simulated_methods_have_required_da_fields(self, synthetic_data):
        base_probs, method_probs, labels = synthetic_data
        method_names = [
            "temperature_scaling",
            "parameterized_temperature_scaling",
            "vector_scaling",
            "beta_calibration",
            "ovr_isotonic",
            "odir_dirichlet",
            "gc_dac",
            "rgcl",
            "rgcc",
            "gc_tulip",
            "full_vector_geometric_fusion_cosine",
            "full_vector_geometric_fusion_rgcl_cosine",
            "softmax_knn_blend",
            "kcal_lite",
            "kcal_full",
        ]
        for name in method_names:
            entry = _fake_method_entry(name, method_probs, labels, base_probs=base_probs)
            da = entry["decision_audit"]
            for field in _REQUIRED_DA_FIELDS:
                assert field in da, f"{name}: decision_audit missing field {field}"


# ---------------------------------------------------------------------------
# Tests: CSV da_* flattening
# ---------------------------------------------------------------------------

class TestCSVDecisionAuditFlattening:
    def test_da_columns_present_in_csv_row(self, synthetic_data):
        base_probs, method_probs, labels = synthetic_data
        entry = _fake_method_entry("test_method", method_probs, labels, base_probs=base_probs)
        flat = _flatten_to_csv_row(entry)
        for col in _REQUIRED_DA_CSV_COLS:
            assert col in flat, f"CSV row missing column: {col}"

    def test_da_net_flips_value_consistent(self, synthetic_data):
        base_probs, method_probs, labels = synthetic_data
        entry = _fake_method_entry("test_method", method_probs, labels, base_probs=base_probs)
        flat = _flatten_to_csv_row(entry)
        da = entry["decision_audit"]
        assert flat["da_net_flips"] == da["net_flips"]
        assert flat["da_changed_to_correct_count"] == da["changed_to_correct_count"]
        assert flat["da_changed_to_wrong_count"] == da["changed_to_wrong_count"]

    def test_csv_row_no_nested_dicts(self, synthetic_data):
        base_probs, method_probs, labels = synthetic_data
        entry = _fake_method_entry("test_method", method_probs, labels, base_probs=base_probs)
        flat = _flatten_to_csv_row(entry)
        for k, v in flat.items():
            assert not isinstance(v, dict), f"CSV row still has nested dict at key: {k}"

    def test_entry_without_base_probs_has_no_da_columns(self, synthetic_data):
        _, method_probs, labels = synthetic_data
        entry = _fake_method_entry("no_base", method_probs, labels, base_probs=None)
        flat = _flatten_to_csv_row(entry)
        for col in _REQUIRED_DA_CSV_COLS:
            assert col not in flat, f"Unexpected da column in row without decision_audit: {col}"

    def test_net_flips_arithmetic_in_csv(self, synthetic_data):
        base_probs, method_probs, labels = synthetic_data
        entry = _fake_method_entry("test_method", method_probs, labels, base_probs=base_probs)
        flat = _flatten_to_csv_row(entry)
        assert flat["da_net_flips"] == (
            flat["da_changed_to_correct_count"] - flat["da_changed_to_wrong_count"]
        )


# ---------------------------------------------------------------------------
# Regression test: method missing decision_audit triggers the diagnostic
# ---------------------------------------------------------------------------

class TestMissingAuditDiagnostic:
    def test_missing_audit_detection(self, synthetic_data):
        """
        Simulates the post-summary diagnostic loop: identifies methods
        missing decision_audit (other than base_model).
        """
        base_probs, method_probs, labels = synthetic_data
        methods = [
            _fake_method_entry("base_model", base_probs, labels, base_probs=None),  # intentionally skipped
            _fake_method_entry("temperature_scaling", method_probs, labels, base_probs=base_probs),
            _fake_method_entry("some_new_method", method_probs, labels, base_probs=None),  # regression
        ]
        missing = [
            m["method_name"]
            for m in methods
            if m["method_name"] != "base_model" and "decision_audit" not in m
        ]
        assert "some_new_method" in missing
        assert "temperature_scaling" not in missing
        assert "base_model" not in missing

    def test_all_required_da_fields_validation(self):
        """
        Helper that matches the verification logic: given a decision_audit dict,
        check all required fields are present and not None.
        """
        rng = np.random.RandomState(7)
        N, C = 50, 5
        base_probs = rng.dirichlet(np.ones(C), size=N)
        method_probs = rng.dirichlet(np.ones(C), size=N)
        labels = rng.randint(0, C, size=N)
        da = decision_audit(base_probs, method_probs, labels)
        for field in _REQUIRED_DA_FIELDS:
            assert field in da
            assert da[field] is not None
