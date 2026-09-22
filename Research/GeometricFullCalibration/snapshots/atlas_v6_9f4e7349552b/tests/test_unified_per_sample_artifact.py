"""Tests for the extended per-sample NPZ artifact (schema version 2).

Constraints verified:
  1. All legacy NPZ keys and method-probability keys preserved unchanged.
  2. New keys: sample_idx, base_pred, method_pred__*, flip diagnostics.
  3. Selected parameters stored as float32 scalars per method.
  4. Distance arrays stored as float32 ([N,C] and [N,C,k] tensors).
  5. Manifest written with artifact_schema_version=2.
  6. Missing geometry fields recorded without failing artifact creation.
  7. No raw-image or raw-sample-tensor keys written.
  8. Method names are sanitized before use as NPZ keys.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Experiments.run_unified_benchmark import _sanitize_npz_key, _write_per_sample_npz

# ---------------------------------------------------------------------------
# Shared synthetic data
# ---------------------------------------------------------------------------

N, C = 40, 3


def _make_probs(n: int, c: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    p = np.abs(rng.randn(n, c)) + 0.1
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float64)


@pytest.fixture
def base_data():
    rng = np.random.RandomState(7)
    labels = rng.randint(0, C, N).astype(np.int64)
    base_probs = _make_probs(N, C, 0)
    anchor = rng.uniform(0.3, 0.9, N).astype(np.float64)
    top_pred = np.argmax(base_probs, axis=1).astype(np.int64)
    return labels, base_probs, anchor, top_pred


@pytest.fixture
def methods(base_data):
    labels, base_probs, *_ = base_data
    return {
        "base_model": base_probs.copy(),
        "temperature_scaling": _make_probs(N, C, 1),
        "vector_scaling": _make_probs(N, C, 2),
    }


@pytest.fixture
def can_change():
    return {"base_model": False, "temperature_scaling": False, "vector_scaling": True}


@pytest.fixture
def artifact_ctx():
    rng = np.random.RandomState(9)
    dm = rng.uniform(0, 5, (N, C)).astype(np.float32)
    knn = rng.uniform(0, 5, (N, C, 2)).astype(np.float32)
    return {
        "distance_arrays": {
            "distance_matrix": dm,
            "knn_distances__kcal_lite": knn,
        },
        "method_artifact_metadata": {
            "vector_scaling": {
                "score_mode": "neg_distance",
                "distance_array_key": "distance_matrix",
                "formula": "FullVectorGeometricFusionCalibrator",
            }
        },
        "selected_params": {
            "temperature_scaling": {"temperature": 1.5},
            "vector_scaling": {"lambda": 0.5, "alpha": 2.0},
        },
        "missing_fields": [{"field": "distance_array:foo", "reason": "test missing"}],
        "run_meta": {
            "dataset": "cifar10",
            "model": "resnet18",
            "seed": 42,
            "checkpoint_path": "/tmp/model.pt",
            "split_sizes": {"train": 1000, "validation": 200, "test": N},
        },
    }


def _call(tmp_path: Path, base_data, methods, can_change, ctx=None) -> Path:
    labels, base_probs, anchor, top_pred = base_data
    _write_per_sample_npz(
        str(tmp_path), labels, base_probs, anchor, top_pred, methods, can_change, ctx
    )
    return tmp_path / "per_sample"


def _load_npz(per_sample_dir: Path):
    return np.load(str(per_sample_dir / "per_sample_arrays.npz"), allow_pickle=True)


def _load_manifest(per_sample_dir: Path) -> Dict[str, Any]:
    return json.loads((per_sample_dir / "per_sample_manifest.json").read_text())


# ---------------------------------------------------------------------------
# 1. Legacy keys preserved
# ---------------------------------------------------------------------------

class TestLegacyKeysPreserved:

    def test_core_legacy_keys_present(self, tmp_path, base_data, methods, can_change):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change))
        for key in [
            "labels_test",
            "base_probs_test",
            "gc_dac_anchor_ct_test",
            "gc_dac_top_pred_test",
            "method_index",
            "method_can_change_argmax",
        ]:
            assert key in d.files, f"legacy key '{key}' missing"

    def test_method_prob_keys_present(self, tmp_path, base_data, methods, can_change):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change))
        for m in methods:
            assert f"method_probs__{m}" in d.files

    def test_legacy_dtypes_unchanged(self, tmp_path, base_data, methods, can_change):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change))
        assert d["labels_test"].dtype == np.int64
        assert d["base_probs_test"].dtype == np.float64
        assert d["gc_dac_anchor_ct_test"].dtype == np.float64
        assert d["gc_dac_top_pred_test"].dtype == np.int64
        for m in methods:
            assert d[f"method_probs__{m}"].dtype == np.float64

    def test_legacy_values_unchanged(self, tmp_path, base_data, methods, can_change):
        labels, base_probs, anchor, top_pred = base_data
        d = _load_npz(_call(tmp_path, base_data, methods, can_change))
        np.testing.assert_array_equal(d["labels_test"], labels)
        np.testing.assert_array_equal(d["base_probs_test"], base_probs)
        np.testing.assert_array_equal(d["gc_dac_anchor_ct_test"], anchor)
        np.testing.assert_array_equal(d["gc_dac_top_pred_test"], top_pred)


# ---------------------------------------------------------------------------
# 2. New prediction and flip-diagnostic keys
# ---------------------------------------------------------------------------

class TestNewPredictionKeys:

    def test_sample_idx(self, tmp_path, base_data, methods, can_change):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change))
        assert "sample_idx" in d.files
        np.testing.assert_array_equal(d["sample_idx"], np.arange(N))
        assert d["sample_idx"].dtype == np.int64

    def test_base_pred(self, tmp_path, base_data, methods, can_change):
        labels, base_probs, anchor, top_pred = base_data
        d = _load_npz(_call(tmp_path, base_data, methods, can_change))
        assert "base_pred" in d.files
        np.testing.assert_array_equal(d["base_pred"], np.argmax(base_probs, axis=1))
        assert d["base_pred"].dtype == np.int64

    def test_method_pred_keys_present(self, tmp_path, base_data, methods, can_change):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change))
        for m in methods:
            skey = _sanitize_npz_key(m)
            assert f"method_pred__{skey}" in d.files

    def test_flip_keys_for_decision_changing(self, tmp_path, base_data, methods, can_change):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change))
        for key in [
            "argmax_changed__vector_scaling",
            "flip_to_correct__vector_scaling",
            "flip_to_wrong__vector_scaling",
        ]:
            assert key in d.files, f"flip key '{key}' missing"

    def test_no_flip_keys_for_non_decision_changing(self, tmp_path, base_data, methods, can_change):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change))
        assert "argmax_changed__temperature_scaling" not in d.files
        assert "argmax_changed__base_model" not in d.files

    def test_flip_values_correct(self, tmp_path, base_data, methods, can_change):
        labels, base_probs, *_ = base_data
        d = _load_npz(_call(tmp_path, base_data, methods, can_change))
        base_pred = np.argmax(base_probs, axis=1)
        vs_pred = np.argmax(methods["vector_scaling"], axis=1)
        changed = vs_pred != base_pred
        np.testing.assert_array_equal(d["argmax_changed__vector_scaling"], changed)
        np.testing.assert_array_equal(
            d["flip_to_correct__vector_scaling"],
            changed & (base_pred != labels) & (vs_pred == labels),
        )
        np.testing.assert_array_equal(
            d["flip_to_wrong__vector_scaling"],
            changed & (base_pred == labels) & (vs_pred != labels),
        )


# ---------------------------------------------------------------------------
# 3. Selected parameters
# ---------------------------------------------------------------------------

class TestSelectedParams:

    def test_scalar_param_stored(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change, artifact_ctx))
        assert "selected_temperature__temperature_scaling" in d.files
        np.testing.assert_allclose(
            float(d["selected_temperature__temperature_scaling"]), 1.5, atol=1e-5
        )

    def test_param_dtype_float32(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change, artifact_ctx))
        assert d["selected_temperature__temperature_scaling"].dtype == np.float32

    def test_multiple_params_per_method(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change, artifact_ctx))
        assert "selected_lambda__vector_scaling" in d.files
        assert "selected_alpha__vector_scaling" in d.files
        np.testing.assert_allclose(float(d["selected_lambda__vector_scaling"]), 0.5, atol=1e-5)
        np.testing.assert_allclose(float(d["selected_alpha__vector_scaling"]), 2.0, atol=1e-5)

    def test_no_params_without_ctx(self, tmp_path, base_data, methods, can_change):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change))
        param_keys = [k for k in d.files if k.startswith("selected_")]
        assert param_keys == [], "no selected_* keys expected when artifact_ctx is None"


# ---------------------------------------------------------------------------
# 4. Distance arrays
# ---------------------------------------------------------------------------

class TestDistanceArrays:

    def test_2d_distance_matrix_stored(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change, artifact_ctx))
        assert "distance_matrix" in d.files
        assert d["distance_matrix"].dtype == np.float32
        assert d["distance_matrix"].shape == (N, C)

    def test_3d_knn_tensor_stored(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change, artifact_ctx))
        assert "knn_distances__kcal_lite" in d.files
        assert d["knn_distances__kcal_lite"].dtype == np.float32
        assert d["knn_distances__kcal_lite"].shape == (N, C, 2)

    def test_float64_input_stored_as_float32(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        ctx = dict(artifact_ctx)
        ctx["distance_arrays"] = {"distance_matrix": np.ones((N, C), dtype=np.float64)}
        d = _load_npz(_call(tmp_path, base_data, methods, can_change, ctx))
        assert d["distance_matrix"].dtype == np.float32

    def test_none_array_skipped_and_recorded(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        ctx = dict(artifact_ctx)
        ctx["distance_arrays"] = {"distance_matrix": None}
        ctx["missing_fields"] = []
        d = _load_npz(_call(tmp_path, base_data, methods, can_change, ctx))
        assert "distance_matrix" not in d.files
        m = _load_manifest(tmp_path / "per_sample")
        missing_fields = [f["field"] for f in m["missing_fields"]]
        assert "distance_array:distance_matrix" in missing_fields

    def test_no_distance_arrays_without_ctx(self, tmp_path, base_data, methods, can_change):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change))
        dist_keys = [k for k in d.files if k.startswith("distance_") or k.startswith("knn_")]
        assert dist_keys == []


# ---------------------------------------------------------------------------
# 5. Manifest
# ---------------------------------------------------------------------------

class TestManifest:

    def test_schema_version(self, tmp_path, base_data, methods, can_change):
        _call(tmp_path, base_data, methods, can_change)
        m = _load_manifest(tmp_path / "per_sample")
        assert m["artifact_schema_version"] == 2

    def test_manifest_array_keys_match_npz(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        pdir = _call(tmp_path, base_data, methods, can_change, artifact_ctx)
        d = _load_npz(pdir)
        m = _load_manifest(pdir)
        assert {a["name"] for a in m["arrays"]} == set(d.files)

    def test_manifest_array_shapes_correct(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        pdir = _call(tmp_path, base_data, methods, can_change, artifact_ctx)
        d = _load_npz(pdir)
        m = _load_manifest(pdir)
        for entry in m["arrays"]:
            np.testing.assert_array_equal(
                entry["shape"], list(d[entry["name"]].shape),
                err_msg=f"shape mismatch for key '{entry['name']}'",
            )

    def test_distance_array_keys_in_manifest(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        m = _load_manifest(_call(tmp_path, base_data, methods, can_change, artifact_ctx))
        assert "distance_matrix" in m["distance_array_keys"]
        assert "knn_distances__kcal_lite" in m["distance_array_keys"]

    def test_method_names_in_manifest(self, tmp_path, base_data, methods, can_change):
        m = _load_manifest(_call(tmp_path, base_data, methods, can_change))
        assert set(m["method_names"]) == set(methods.keys())

    def test_method_can_change_argmax_in_manifest(self, tmp_path, base_data, methods, can_change):
        m = _load_manifest(_call(tmp_path, base_data, methods, can_change))
        assert m["method_can_change_argmax"]["vector_scaling"] is True
        assert m["method_can_change_argmax"]["temperature_scaling"] is False

    def test_missing_fields_propagated(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        m = _load_manifest(_call(tmp_path, base_data, methods, can_change, artifact_ctx))
        assert any(f["field"] == "distance_array:foo" for f in m["missing_fields"])

    def test_score_reconstruction_in_manifest(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        m = _load_manifest(_call(tmp_path, base_data, methods, can_change, artifact_ctx))
        assert "vector_scaling" in m["score_reconstruction"]
        rec = m["score_reconstruction"]["vector_scaling"]
        assert rec["score_mode"] == "neg_distance"
        assert rec["distance_array_key"] == "distance_matrix"

    def test_selected_params_in_manifest(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        m = _load_manifest(_call(tmp_path, base_data, methods, can_change, artifact_ctx))
        assert "temperature_scaling" in m["selected_params"]
        assert m["selected_params"]["temperature_scaling"]["temperature"] == pytest.approx(1.5)

    def test_run_meta_in_manifest(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        m = _load_manifest(_call(tmp_path, base_data, methods, can_change, artifact_ctx))
        assert m.get("dataset") == "cifar10"
        assert m.get("model") == "resnet18"
        assert m.get("seed") == 42


# ---------------------------------------------------------------------------
# 6. Missing geometry — NPZ and manifest still written
# ---------------------------------------------------------------------------

class TestMissingGeometry:

    def _ctx_no_geometry(self):
        return {
            "distance_arrays": {},
            "method_artifact_metadata": {},
            "selected_params": {},
            "missing_fields": [
                {"field": "distance_array:distance_matrix", "reason": "geometry unavailable"}
            ],
            "run_meta": {},
        }

    def test_npz_written_without_geometry(self, tmp_path, base_data, methods, can_change):
        pdir = _call(tmp_path, base_data, methods, can_change, self._ctx_no_geometry())
        npz = pdir / "per_sample_arrays.npz"
        assert npz.exists()
        d = np.load(str(npz), allow_pickle=True)
        assert "labels_test" in d.files
        assert "base_probs_test" in d.files

    def test_manifest_written_without_geometry(self, tmp_path, base_data, methods, can_change):
        pdir = _call(tmp_path, base_data, methods, can_change, self._ctx_no_geometry())
        manifest_path = pdir / "per_sample_manifest.json"
        assert manifest_path.exists()
        m = json.loads(manifest_path.read_text())
        assert m["artifact_schema_version"] == 2
        missing = [f["field"] for f in m["missing_fields"]]
        assert "distance_array:distance_matrix" in missing

    def test_new_diagnostic_keys_present_without_geometry(
        self, tmp_path, base_data, methods, can_change
    ):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change, self._ctx_no_geometry()))
        assert "sample_idx" in d.files
        assert "base_pred" in d.files
        assert "method_pred__vector_scaling" in d.files


# ---------------------------------------------------------------------------
# 7. No raw images or sample tensors
# ---------------------------------------------------------------------------

class TestNoRawData:

    FORBIDDEN_PREFIXES = (
        "image", "img", "raw", "x_test", "x_val", "x_train", "pixel", "input",
    )

    def test_no_raw_image_keys(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change, artifact_ctx))
        for key in d.files:
            low = key.lower()
            for prefix in self.FORBIDDEN_PREFIXES:
                assert not low.startswith(prefix), (
                    f"Array '{key}' looks like raw image/sample data (starts with '{prefix}')"
                )

    def test_no_4d_arrays(self, tmp_path, base_data, methods, can_change, artifact_ctx):
        d = _load_npz(_call(tmp_path, base_data, methods, can_change, artifact_ctx))
        for key in d.files:
            assert d[key].ndim <= 3, (
                f"Array '{key}' has ndim={d[key].ndim} — 4-D arrays are not expected"
            )


# ---------------------------------------------------------------------------
# 8. Sanitize method names
# ---------------------------------------------------------------------------

class TestSanitizeKey:

    def test_clean_name_unchanged(self):
        assert _sanitize_npz_key("temperature_scaling") == "temperature_scaling"
        assert _sanitize_npz_key("vector_scaling") == "vector_scaling"

    def test_hyphen_replaced(self):
        assert _sanitize_npz_key("method-name") == "method_name"

    def test_dot_replaced(self):
        assert _sanitize_npz_key("method.v2") == "method_v2"

    def test_space_replaced(self):
        assert _sanitize_npz_key("my method") == "my_method"

    def test_mixed_special_chars(self):
        assert _sanitize_npz_key("a.b-c d") == "a_b_c_d"

    def test_alphanumeric_and_underscore_preserved(self):
        name = "abc_DEF_123"
        assert _sanitize_npz_key(name) == name

    def test_sanitized_key_used_in_npz(self, tmp_path, base_data, can_change):
        labels, base_probs, anchor, top_pred = base_data
        special_method = "method-with.special chars"
        methods_special = {special_method: base_probs.copy()}
        can_change_special = {special_method: False}
        _write_per_sample_npz(
            str(tmp_path),
            labels, base_probs, anchor, top_pred,
            methods_special, can_change_special,
        )
        d = np.load(str(tmp_path / "per_sample" / "per_sample_arrays.npz"), allow_pickle=True)
        skey = _sanitize_npz_key(special_method)
        assert f"method_pred__{skey}" in d.files
        assert f"method_probs__{special_method}" in d.files
