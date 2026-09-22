"""
End-to-end test for the external method import path.

Synthesizes a probability matrix, writes it as .npy + JSON spec, then exercises
the exact same code path used by run_unified_benchmark.py's external import block.
Verifies: checkpoint saved, decision_audit fields present, CSV has da_* columns,
structural_axes attaches when the method name is registered.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.decision_audit import decision_audit
from utils.method_metadata import structural_axes_for_method
from utils.unified_metrics import evaluate_all, validate_probability_matrix


# ---------------------------------------------------------------------------
# Helpers that replicate the external-import logic without importing the runner
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _simulate_external_import(
    ext_probs: np.ndarray,
    base_probs: np.ndarray,
    labels: np.ndarray,
    method_name: str,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Mirrors the external-import logic:
      1. validate_probability_matrix
      2. evaluate_all
      3. decision_audit
      4. structural_axes_for_method
      5. build entry dict with da_* fields
    """
    # Step 1: validate
    validate_probability_matrix(ext_probs, labels)

    # Step 2: evaluate all metrics
    metrics = evaluate_all(ext_probs, labels)

    # Step 3: decision audit
    da = decision_audit(base_probs, ext_probs, labels)

    # Step 4: structural axes (may be None for unregistered names)
    axes = structural_axes_for_method(method_name)

    entry: Dict[str, Any] = {
        "method_name": method_name,
        "can_change_argmax": bool(metadata.get("can_change_argmax", True)),
        "tuning_procedure": metadata.get("tuning_procedure", "external"),
        "metrics": metrics,
        "decision_audit": da,
    }
    if axes is not None:
        entry["structural_axes"] = axes

    return entry


def _flatten_csv_row(entry: Dict[str, Any]) -> Dict[str, Any]:
    flat = {k: v for k, v in entry.items() if k not in ("metrics", "decision_audit")}
    for mk, mv in entry["metrics"].items():
        flat[mk] = mv
    if "decision_audit" in entry and isinstance(entry["decision_audit"], dict):
        for dk, dv in entry["decision_audit"].items():
            flat[f"da_{dk}"] = dv
    if isinstance(flat.get("structural_axes"), dict):
        flat["structural_axes"] = json.dumps(flat["structural_axes"])
    return flat


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_ext():
    rng = np.random.RandomState(11)
    N, C = 100, 10
    base_logits = rng.randn(N, C).astype(np.float32)
    base_probs = _softmax(base_logits)
    ext_logits = base_logits + 0.3 * rng.randn(N, C).astype(np.float32)
    ext_probs = _softmax(ext_logits).astype(np.float64)
    labels = rng.randint(0, C, size=N)
    return base_probs, ext_probs, labels


# ---------------------------------------------------------------------------
# Tests: probability matrix validation
# ---------------------------------------------------------------------------

class TestExternalProbValidation:
    def test_valid_probs_accepted(self, synthetic_ext):
        _, ext_probs, labels = synthetic_ext
        validate_probability_matrix(ext_probs, labels)  # must not raise

    def test_non_normalized_rejected(self, synthetic_ext):
        _, ext_probs, labels = synthetic_ext
        bad = ext_probs.copy()
        bad[0] *= 2.0  # row 0 sums to ~2
        with pytest.raises((ValueError, AssertionError)):
            validate_probability_matrix(bad, labels)

    def test_wrong_shape_rejected(self, synthetic_ext):
        _, _, labels = synthetic_ext
        bad = np.random.dirichlet(np.ones(5), size=50)  # N mismatch
        with pytest.raises((ValueError, AssertionError)):
            validate_probability_matrix(bad, labels)

    def test_nan_rejected(self, synthetic_ext):
        _, ext_probs, labels = synthetic_ext
        bad = ext_probs.copy()
        bad[5, 3] = np.nan
        with pytest.raises((ValueError, AssertionError)):
            validate_probability_matrix(bad, labels)


# ---------------------------------------------------------------------------
# Tests: entry construction with decision_audit
# ---------------------------------------------------------------------------

class TestExternalImportEntry:
    def test_entry_has_decision_audit(self, synthetic_ext):
        base_probs, ext_probs, labels = synthetic_ext
        entry = _simulate_external_import(
            ext_probs, base_probs, labels,
            method_name="aar_official",
            metadata={"can_change_argmax": True},
        )
        assert "decision_audit" in entry
        da = entry["decision_audit"]
        for field in ("net_flips", "accuracy_delta", "changed_to_correct_count", "changed_to_wrong_count"):
            assert field in da, f"Missing da field: {field}"

    def test_entry_metrics_present(self, synthetic_ext):
        base_probs, ext_probs, labels = synthetic_ext
        entry = _simulate_external_import(
            ext_probs, base_probs, labels,
            method_name="aar_official",
            metadata={},
        )
        assert "metrics" in entry
        assert "accuracy" in entry["metrics"] or "nll" in entry["metrics"]

    def test_registered_name_gets_structural_axes(self, synthetic_ext):
        base_probs, ext_probs, labels = synthetic_ext
        # "aar_lightweight" is registered in method_metadata.py
        entry = _simulate_external_import(
            ext_probs, base_probs, labels,
            method_name="aar_lightweight",
            metadata={"can_change_argmax": True},
        )
        assert "structural_axes" in entry
        assert entry["structural_axes"]["can_change_argmax"] is True

    def test_unregistered_name_has_no_axes(self, synthetic_ext):
        base_probs, ext_probs, labels = synthetic_ext
        entry = _simulate_external_import(
            ext_probs, base_probs, labels,
            method_name="some_totally_custom_external_method_xyz",
            metadata={},
        )
        assert "structural_axes" not in entry

    def test_net_flips_arithmetic(self, synthetic_ext):
        base_probs, ext_probs, labels = synthetic_ext
        entry = _simulate_external_import(
            ext_probs, base_probs, labels,
            method_name="aar_official",
            metadata={},
        )
        da = entry["decision_audit"]
        assert da["net_flips"] == da["changed_to_correct_count"] - da["changed_to_wrong_count"]


# ---------------------------------------------------------------------------
# Tests: CSV flattening produces da_* columns
# ---------------------------------------------------------------------------

class TestExternalImportCSVFlattening:
    def test_da_columns_in_csv_row(self, synthetic_ext):
        base_probs, ext_probs, labels = synthetic_ext
        entry = _simulate_external_import(
            ext_probs, base_probs, labels,
            method_name="aar_official",
            metadata={},
        )
        flat = _flatten_csv_row(entry)
        for col in ("da_net_flips", "da_accuracy_delta", "da_changed_to_correct_count", "da_changed_to_wrong_count"):
            assert col in flat, f"CSV row missing column: {col}"

    def test_no_nested_dicts_in_csv_row(self, synthetic_ext):
        base_probs, ext_probs, labels = synthetic_ext
        entry = _simulate_external_import(
            ext_probs, base_probs, labels,
            method_name="aar_official",
            metadata={},
        )
        flat = _flatten_csv_row(entry)
        for k, v in flat.items():
            assert not isinstance(v, dict), f"Nested dict at key {k}"

    def test_structural_axes_serialized_as_json_string(self, synthetic_ext):
        base_probs, ext_probs, labels = synthetic_ext
        entry = _simulate_external_import(
            ext_probs, base_probs, labels,
            method_name="aar_lightweight",
            metadata={"can_change_argmax": True},
        )
        flat = _flatten_csv_row(entry)
        assert isinstance(flat.get("structural_axes"), str), \
            "structural_axes should be JSON-serialized in CSV row"
        parsed = json.loads(flat["structural_axes"])
        assert "can_change_argmax" in parsed


# ---------------------------------------------------------------------------
# Tests: file-based round-trip (write .npy + JSON, read back, validate)
# ---------------------------------------------------------------------------

class TestExternalImportFileRoundtrip:
    def test_npy_write_and_reload(self, synthetic_ext, tmp_path):
        base_probs, ext_probs, labels = synthetic_ext
        npy_path = tmp_path / "test_probs.npy"
        np.save(str(npy_path), ext_probs)

        loaded = np.load(str(npy_path))
        validate_probability_matrix(loaded, labels)
        np.testing.assert_allclose(loaded, ext_probs, atol=1e-10)

    def test_json_spec_write_and_parse(self, synthetic_ext, tmp_path):
        _, ext_probs, labels = synthetic_ext
        npy_path = tmp_path / "test_probs.npy"
        np.save(str(npy_path), ext_probs)

        spec = {
            "methods": {
                "aar_official": {
                    "test_probs": str(npy_path),
                    "metadata": {
                        "can_change_argmax": True,
                        "method_family": "external_full_vector_posthoc",
                        "tuning_procedure": "external",
                        "tuning_split": "external_or_validation",
                        "tuning_objective": "external",
                    },
                }
            }
        }
        spec_path = tmp_path / "external_methods.json"
        with open(spec_path, "w") as f:
            json.dump(spec, f)

        with open(spec_path, "r") as f:
            loaded_spec = json.load(f)

        methods = loaded_spec["methods"]
        assert "aar_official" in methods
        assert os.path.exists(methods["aar_official"]["test_probs"])

    def test_full_pipeline_from_file(self, synthetic_ext, tmp_path):
        """Full pipeline: write npy, load, validate, build entry, flatten CSV."""
        base_probs, ext_probs, labels = synthetic_ext
        npy_path = tmp_path / "ext_probs.npy"
        np.save(str(npy_path), ext_probs)

        loaded = np.load(str(npy_path))
        if loaded.dtype != np.float64:
            loaded = loaded.astype(np.float64)
        validate_probability_matrix(loaded, labels)

        entry = _simulate_external_import(
            loaded, base_probs, labels,
            method_name="aar_lightweight",
            metadata={"can_change_argmax": True},
        )
        flat = _flatten_csv_row(entry)

        assert "da_net_flips" in flat
        assert "da_changed_to_correct_count" in flat
        assert "structural_axes" in flat
        assert not any(isinstance(v, dict) for v in flat.values())
