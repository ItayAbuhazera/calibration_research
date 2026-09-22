"""
Tests for Calibrators/parameterized_temperature_scaling.py.

Empirically verifies the scalar-temperature structural lemma:

    Lemma: For any positive scalar function T: R^K -> R_{>0},
    argmax_k z_k = argmax_k (z_k / T(z)).
    Proof: dividing all logits by a positive constant preserves relative order.

The test is an empirical check, not a proof — but it should always pass
because the implementation explicitly asserts this invariant in calibrate().
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Calibrators.parameterized_temperature_scaling import ParameterizedTemperatureScaling


class TestPTSArgmaxInvariance:
    def _random_logits(self, n: int, c: int, seed: int = 0) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.randn(n, c).astype(np.float64)

    def _random_labels(self, n: int, c: int, seed: int = 1) -> np.ndarray:
        rng = np.random.RandomState(seed)
        return rng.randint(0, c, size=n)

    def test_random_positive_temperature_preserves_argmax(self):
        """
        Scalar-temperature lemma: argmax(z) == argmax(z / T) for any T > 0.
        Verified directly with random logits and random positive temperatures.
        """
        rng = np.random.RandomState(42)
        n, c = 200, 10
        logits = rng.randn(n, c)
        T = rng.uniform(0.01, 10.0, size=n)  # random positive temperatures

        base_preds = np.argmax(logits, axis=1)
        scaled_preds = np.argmax(logits / T[:, None], axis=1)
        assert np.array_equal(base_preds, scaled_preds), (
            "Scalar-temperature lemma violated: argmax changed after dividing by positive T"
        )

    def test_pts_fit_and_calibrate_argmax_invariant(self):
        """PTS calibrate() asserts argmax invariance; fit then calibrate must not raise."""
        n, c = 80, 5
        logits_val = self._random_logits(n, c, seed=0)
        labels_val = self._random_labels(n, c, seed=1)
        logits_test = self._random_logits(60, c, seed=2)

        pts = ParameterizedTemperatureScaling(hidden_dim=16, lr=1e-2, epochs=5, patience=3)
        pts.fit(logits_val, labels_val)
        probs = pts.calibrate(logits_test)

        assert probs.shape == (60, c)
        assert np.all(np.isfinite(probs)), "PTS output contains non-finite values"
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)

    def test_pts_temperature_is_positive(self):
        """All per-sample temperatures must be strictly positive."""
        import torch
        import torch.nn.functional as F
        from Calibrators.parameterized_temperature_scaling import (
            _extract_logit_features,
            _TemperatureNet,
        )

        rng = np.random.RandomState(99)
        logits = rng.randn(50, 8).astype(np.float32)
        pts = ParameterizedTemperatureScaling(hidden_dim=16, epochs=3)
        pts.fit(logits, rng.randint(0, 8, size=50))

        feats = _extract_logit_features(logits)
        feats_norm = (feats - pts._feature_mean) / pts._feature_std
        device = next(pts._net.parameters()).device
        with torch.no_grad():
            T = pts._net(torch.tensor(feats_norm, device=device))
        assert torch.all(T > 0), "Temperature must always be strictly positive"

    def test_pts_get_params(self):
        """get_params() must return a dict with the expected keys."""
        pts = ParameterizedTemperatureScaling(epochs=3)
        pts.fit(
            np.random.randn(30, 4),
            np.random.randint(0, 4, size=30),
        )
        params = pts.get_params()
        assert params["fitted"] is True
        assert "training_history" in params
        assert len(params["training_history"]) > 0

    def test_pts_not_fitted_raises(self):
        """calibrate() before fit() must raise."""
        pts = ParameterizedTemperatureScaling()
        with pytest.raises(RuntimeError, match="fit"):
            pts.calibrate(np.random.randn(10, 5))
