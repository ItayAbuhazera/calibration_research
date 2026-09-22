"""
End-to-end synthetic integration smoke test.

Exercises PTS, Trust Score (diagnostic + switch), AAR-lightweight, and GLAD-PI
on synthetic logits, labels, features, and distance matrices.

No CIFAR-100, no real model. Must complete in under 2 minutes on CPU.

For each method: checks output shape, row sums, finite values, and decision_audit.
Also verifies structural_axes attachment and CSV flattening.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.decision_audit import decision_audit
from utils.method_metadata import structural_axes_for_method


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _make_data(N: int, C: int, seed: int = 0):
    rng = np.random.RandomState(seed)
    logits = rng.randn(N, C).astype(np.float32)
    labels = rng.randint(0, C, size=N)
    base_probs = _softmax(logits)
    features = rng.randn(N, 32).astype(np.float32)
    dist_matrix = np.abs(rng.randn(N, C)).astype(np.float32) + 0.1
    return logits, labels, base_probs, features, dist_matrix


# ---------------------------------------------------------------------------
# Helper: assert valid probability matrix
# ---------------------------------------------------------------------------

def _assert_valid_probs(probs: np.ndarray, N: int, C: int, name: str) -> None:
    assert probs.shape == (N, C), f"{name}: expected ({N},{C}), got {probs.shape}"
    assert np.all(np.isfinite(probs)), f"{name}: non-finite values"
    assert np.all(probs >= 0), f"{name}: negative values"
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5,
                               err_msg=f"{name}: rows do not sum to 1")


# ---------------------------------------------------------------------------
# Helper: make a method entry (mirrors benchmark runner logic)
# ---------------------------------------------------------------------------

def _make_entry(
    method_name: str,
    probs: np.ndarray,
    labels: np.ndarray,
    base_probs: Optional[np.ndarray] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "method_name": method_name,
        "metrics": {"accuracy": float(np.mean(np.argmax(probs, 1) == labels))},
    }
    if extra:
        entry.update(extra)
    axes = structural_axes_for_method(method_name)
    if axes is not None:
        entry["structural_axes"] = axes
    if base_probs is not None:
        entry["decision_audit"] = decision_audit(base_probs, probs, labels)
    return entry


def _flatten(entry: Dict[str, Any]) -> Dict[str, Any]:
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
# PTS smoke test
# ---------------------------------------------------------------------------

class TestPTSSmoke:
    def test_pts_end_to_end(self):
        N, C = 120, 10
        logits, labels, base_probs, _, _ = _make_data(N, C, seed=1)
        logits_val, labels_val = logits[:80], labels[:80]
        logits_test, labels_test = logits[80:], labels[80:]
        base_probs_test = base_probs[80:]

        from Calibrators.parameterized_temperature_scaling import ParameterizedTemperatureScaling
        pts = ParameterizedTemperatureScaling(hidden_dim=8, epochs=3, patience=2)
        pts.fit(logits_val, labels_val)
        probs = pts.calibrate(logits_test)

        _assert_valid_probs(probs, 40, C, "PTS")

        entry = _make_entry("parameterized_temperature_scaling", probs, labels_test, base_probs=base_probs_test)
        assert "decision_audit" in entry
        assert "structural_axes" in entry
        assert entry["structural_axes"]["argmax_invariant_by_construction"] is True
        assert entry["decision_audit"]["net_flips"] == 0  # PTS cannot flip argmax

        flat = _flatten(entry)
        assert "da_net_flips" in flat
        assert flat["da_net_flips"] == 0


# ---------------------------------------------------------------------------
# Trust Score smoke test
# ---------------------------------------------------------------------------

class TestTrustScoreSmoke:
    def test_trust_diagnostic_no_prob_change(self):
        N, C = 120, 5
        logits, labels, base_probs, features, _ = _make_data(N, C, seed=2)
        train_feat, train_labels = features[:60], labels[:60]
        test_feat = features[60:]
        test_probs = base_probs[60:]
        test_labels = labels[60:]

        from Calibrators.trust_score import TrustScoreCalibrator
        ts = TrustScoreCalibrator()
        ts.fit(train_feat, train_labels)
        calibrated, meta = ts.calibrate_diagnostic(test_feat, test_probs)

        _assert_valid_probs(calibrated, 60, C, "trust_score_diagnostic")
        np.testing.assert_array_equal(calibrated, test_probs,
                                      err_msg="Diagnostic variant must not change probs")
        assert "mean_trust_score" in meta

        entry = _make_entry("trust_score_original_diagnostic", calibrated, test_labels, base_probs=test_probs)
        assert entry["decision_audit"]["net_flips"] == 0
        assert entry["structural_axes"]["can_change_argmax"] is False

    def test_trust_switch_outputs_valid(self):
        N, C = 150, 5
        logits, labels, base_probs, features, _ = _make_data(N, C, seed=3)
        train_feat, train_labels = features[:50], labels[:50]
        val_feat = features[50:100]
        val_labels = labels[50:100]
        val_probs = base_probs[50:100]
        test_feat = features[100:]
        test_probs = base_probs[100:]
        test_labels = labels[100:]

        from Calibrators.trust_score import TrustScoreCalibrator
        ts = TrustScoreCalibrator()
        ts.fit(train_feat, train_labels)
        calibrated, meta = ts.calibrate_switch(
            test_feat, test_probs, val_feat, val_probs, val_labels
        )

        _assert_valid_probs(calibrated, 50, C, "trust_score_switch")
        assert "selected_threshold" in meta

        entry = _make_entry("trust_score_original_switch", calibrated, test_labels, base_probs=test_probs)
        assert "decision_audit" in entry
        assert entry["structural_axes"]["can_change_argmax"] is True

    def test_trust_score_alpha_nonzero_raises(self):
        from Calibrators.trust_score import TrustScoreCalibrator
        with pytest.raises(NotImplementedError, match="alpha > 0"):
            TrustScoreCalibrator(alpha=0.1)


# ---------------------------------------------------------------------------
# AAR-Lightweight smoke test
# ---------------------------------------------------------------------------

class TestAARLightweightSmoke:
    def test_aar_end_to_end(self):
        N, C = 120, 8
        logits, labels, _, _, dist = _make_data(N, C, seed=4)
        logits_val, labels_val, dist_val = logits[:80], labels[:80], dist[:80]
        logits_test, labels_test, dist_test = logits[80:], labels[80:], dist[80:]
        base_probs_test = _softmax(logits_test)

        from Calibrators.aar_calibration import AARLightweightCalibrator, METADATA
        aar = AARLightweightCalibrator()
        result = aar.fit(logits_val, labels_val, dist_val)

        assert "fit_nll" in result
        assert result["fit_nll"] < 100

        probs = aar.calibrate(logits_test, dist_test)
        _assert_valid_probs(probs, 40, C, "aar_lightweight")

        entry = _make_entry("aar_lightweight", probs, labels_test, base_probs=base_probs_test)
        assert entry["structural_axes"]["can_change_argmax"] is True
        assert "decision_audit" in entry

        params = aar.get_params()
        assert params["official_implementation"] is False
        assert params["atypicality_estimator"] == "nn_distance_proxy"

    def test_aar_not_fitted_raises(self):
        from Calibrators.aar_calibration import AARLightweightCalibrator
        aar = AARLightweightCalibrator()
        with pytest.raises(RuntimeError, match="fit"):
            aar.calibrate(np.random.randn(10, 5), np.ones((10, 5)))


# ---------------------------------------------------------------------------
# GLAD-PI smoke test
# ---------------------------------------------------------------------------

class TestGLADPISmoke:
    def test_glad_pi_end_to_end(self):
        N, C = 120, 6
        logits, labels, _, _, dist = _make_data(N, C, seed=5)
        logits_fit, labels_fit, dist_fit = logits[:60], labels[:60], dist[:60]
        logits_sel, labels_sel, dist_sel = logits[60:90], labels[60:90], dist[60:90]
        logits_test, labels_test, dist_test = logits[90:], labels[90:], dist[90:]
        base_probs_test = _softmax(logits_test)

        from Calibrators.glad_pi import GLADPICalibrator
        glad = GLADPICalibrator(
            hidden_dim=8, epochs=3, patience=3,
            beta_grid=[0.0, 0.1, 1.0], nll_tolerance=0.5,
        )
        fit_result = glad.fit(logits_fit, labels_fit, dist_fit, logits_sel, labels_sel, dist_sel)

        assert "selected_beta" in fit_result
        assert "selected_fallback" in fit_result
        assert fit_result["selected_beta"] in glad.beta_grid

        probs = glad.calibrate(logits_test, dist_test)
        _assert_valid_probs(probs, 30, C, "glad_pi")

        entry = _make_entry("glad_pi", probs, labels_test, base_probs=base_probs_test)
        assert entry["structural_axes"]["can_change_argmax"] is True
        assert entry["structural_axes"]["joint_sample_class_dependent"] is True
        assert "decision_audit" in entry

        flat = _flatten(entry)
        assert "da_net_flips" in flat
        assert "da_changed_to_correct_count" in flat

    def test_glad_pi_single_beta_zero(self):
        from Calibrators.glad_pi import GLADPICalibrator
        N, C = 60, 4
        logits, labels, _, _, dist = _make_data(N, C, seed=6)
        glad = GLADPICalibrator(hidden_dim=8, epochs=2, beta_grid=[0.0])
        glad.fit(logits[:40], labels[:40], dist[:40],
                 logits[40:], labels[40:], dist[40:])
        probs = glad.calibrate(logits[40:], dist[40:])
        assert probs.shape == (20, C)
        assert glad._selected_beta == 0.0
        assert glad._selected_fallback is False


# ---------------------------------------------------------------------------
# Cross-method consistency check
# ---------------------------------------------------------------------------

class TestCrossMethodConsistency:
    def test_no_key_collisions_in_method_entries(self):
        """All method entries can be flattened to CSV rows without key conflicts."""
        N, C = 80, 5
        logits, labels, base_probs, features, dist = _make_data(N, C, seed=99)
        probs_test = base_probs  # reuse base as fake calibrated output

        method_names = [
            "temperature_scaling",
            "parameterized_temperature_scaling",
            "vector_scaling",
            "trust_score_original_diagnostic",
            "trust_score_original_switch",
            "aar_lightweight",
            "glad_pi",
            "full_vector_geometric_fusion_cosine",
        ]
        entries = [_make_entry(n, probs_test, labels, base_probs=base_probs) for n in method_names]
        rows = [_flatten(e) for e in entries]

        for row, name in zip(rows, method_names):
            # No nested dicts
            for k, v in row.items():
                assert not isinstance(v, dict), f"{name}: nested dict at {k}"
            # Required da_ columns present (we passed base_probs)
            for col in ("da_net_flips", "da_accuracy_delta", "da_changed_to_correct_count"):
                assert col in row, f"{name}: missing CSV column {col}"

    def test_all_registered_methods_have_structural_axes(self):
        """All methods used in the smoke test have registered structural axes."""
        method_names = [
            "temperature_scaling",
            "parameterized_temperature_scaling",
            "trust_score_original_diagnostic",
            "trust_score_original_switch",
            "aar_lightweight",
            "glad_pi",
        ]
        for name in method_names:
            axes = structural_axes_for_method(name)
            assert axes is not None, f"{name} is not registered in method_metadata.py"
            assert "can_change_argmax" in axes
            assert "argmax_invariant_by_construction" in axes

    def test_smoke_test_completes_within_time_limit(self):
        """Full suite should complete in < 120 seconds on CPU."""
        start = time.time()
        N, C = 60, 4
        logits, labels, base_probs, features, dist = _make_data(N, C, seed=7)

        from Calibrators.parameterized_temperature_scaling import ParameterizedTemperatureScaling
        from Calibrators.aar_calibration import AARLightweightCalibrator
        from Calibrators.glad_pi import GLADPICalibrator

        # PTS
        pts = ParameterizedTemperatureScaling(hidden_dim=8, epochs=3)
        pts.fit(logits[:40], labels[:40])
        pts.calibrate(logits[40:])

        # AAR
        aar = AARLightweightCalibrator()
        aar.fit(logits[:40], labels[:40], dist[:40])
        aar.calibrate(logits[40:], dist[40:])

        # GLAD-PI
        glad = GLADPICalibrator(hidden_dim=8, epochs=2, beta_grid=[0.0, 0.1])
        glad.fit(logits[:30], labels[:30], dist[:30], logits[30:40], labels[30:40], dist[30:40])
        glad.calibrate(logits[40:], dist[40:])

        elapsed = time.time() - start
        assert elapsed < 120, f"Smoke test took {elapsed:.1f}s (limit: 120s)"
