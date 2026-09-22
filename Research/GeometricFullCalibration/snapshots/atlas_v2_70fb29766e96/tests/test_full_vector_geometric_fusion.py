"""
Tests for distance_matrix_to_geometry_scores, FullVectorGeometricFusionCalibrator,
and the legacy FullVectorDistanceFusionCalibrator wrapper.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Calibrators.geometric_calibrator import (
    distance_matrix_to_geometry_scores,
    FullVectorGeometricFusionCalibrator,
    FullVectorDistanceFusionCalibrator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockModel:
    """Minimal model that returns pre-set probabilities."""
    def __init__(self, probs: np.ndarray):
        self._probs = probs

    def predict_proba(self, X):
        return self._probs[: len(X)]


def _make_probs(n: int, K: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    p = np.abs(rng.randn(n, K)) + 0.1
    return p / p.sum(axis=1, keepdims=True)


def _make_synthetic(seed: int = 0, n_per_class: int = 30, n_val: int = 20, dim: int = 4, K: int = 3):
    rng = np.random.RandomState(seed)
    centers = rng.randn(K, dim) * 4.0
    parts = [rng.randn(n_per_class, dim) + centers[k] for k in range(K)]
    X_train = np.vstack(parts).astype(np.float32)
    y_train = np.repeat(np.arange(K), n_per_class)
    X_val = (rng.randn(n_val, dim) + centers[0]).astype(np.float32)
    y_val = np.zeros(n_val, dtype=int)
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def D_3x4():
    """3 samples × 4 classes, no ties in per-row minimum."""
    return np.array([
        [1.0, 4.0, 2.0, 3.0],
        [0.5, 2.0, 3.0, 1.0],
        [3.0, 1.0, 4.0, 2.0],
    ], dtype=np.float64)


@pytest.fixture
def synth():
    return _make_synthetic(seed=7)


# ---------------------------------------------------------------------------
# distance_matrix_to_geometry_scores
# ---------------------------------------------------------------------------

class TestDistanceMatrixToGeometryScores:

    # --- output shape and dtype ---

    @pytest.mark.parametrize("mode", [
        "neg_distance", "margin", "log_trust_ratio", "rank_log_trust"
    ])
    def test_shape_preserved(self, D_3x4, mode):
        S = distance_matrix_to_geometry_scores(D_3x4, mode)
        assert S.shape == D_3x4.shape

    @pytest.mark.parametrize("mode", [
        "neg_distance", "margin", "log_trust_ratio", "rank_log_trust"
    ])
    def test_returns_float64(self, D_3x4, mode):
        S = distance_matrix_to_geometry_scores(D_3x4, mode)
        assert S.dtype == np.float64, f"mode={mode}: expected float64, got {S.dtype}"

    # --- neg_distance ---

    def test_neg_distance_exact_values(self, D_3x4):
        S = distance_matrix_to_geometry_scores(D_3x4, "neg_distance")
        np.testing.assert_array_equal(S, -D_3x4)

    # --- margin ---

    def test_margin_manual(self):
        D = np.array([[1.0, 4.0, 2.0, 3.0]])
        S = distance_matrix_to_geometry_scores(D, "margin")
        # class 0: min(4,2,3)-1 = 1,  class 1: min(1,2,3)-4 = -3
        # class 2: min(1,4,3)-2 = -1, class 3: min(1,4,2)-3 = -2
        np.testing.assert_allclose(S[0], [1.0, -3.0, -1.0, -2.0])

    def test_margin_nearest_class_has_highest_score(self, D_3x4):
        S = distance_matrix_to_geometry_scores(D_3x4, "margin")
        for i in range(D_3x4.shape[0]):
            assert np.argmin(D_3x4[i]) == np.argmax(S[i])

    # --- log_trust_ratio ---

    def test_log_trust_ratio_finite(self, D_3x4):
        S = distance_matrix_to_geometry_scores(D_3x4, "log_trust_ratio")
        assert np.all(np.isfinite(S)), "log_trust_ratio produced non-finite values"

    def test_log_trust_ratio_finite_with_zero_distance(self):
        D = np.array([[0.0, 1.0, 2.0]])
        S = distance_matrix_to_geometry_scores(D, "log_trust_ratio", eps=1e-12)
        assert np.all(np.isfinite(S))

    def test_log_trust_ratio_nearest_class_has_highest_score(self, D_3x4):
        S = distance_matrix_to_geometry_scores(D_3x4, "log_trust_ratio")
        for i in range(D_3x4.shape[0]):
            assert np.argmin(D_3x4[i]) == np.argmax(S[i])

    # --- rank_log_trust ---

    def test_rank_log_trust_range(self, D_3x4):
        S = distance_matrix_to_geometry_scores(D_3x4, "rank_log_trust")
        assert np.all(S >= -1e-9) and np.all(S <= 1.0 + 1e-9)

    def test_rank_log_trust_row_extremes(self, D_3x4):
        """Each row must include 0 (lowest trust) and 1 (highest trust)."""
        S = distance_matrix_to_geometry_scores(D_3x4, "rank_log_trust")
        for i in range(S.shape[0]):
            assert S[i].min() == pytest.approx(0.0, abs=1e-9), f"row {i} min != 0"
            assert S[i].max() == pytest.approx(1.0, abs=1e-9), f"row {i} max != 1"

    def test_rank_log_trust_ties_use_average_ranks(self):
        # D[0,0]==D[0,1] and D[0,2]==D[0,3] → symmetric ties in log_trust_ratio
        D = np.array([[1.0, 1.0, 4.0, 4.0]])
        S = distance_matrix_to_geometry_scores(D, "rank_log_trust")
        assert S[0, 0] == pytest.approx(S[0, 1], abs=1e-9), "tied pair 0/1 should share rank"
        assert S[0, 2] == pytest.approx(S[0, 3], abs=1e-9), "tied pair 2/3 should share rank"
        # Near classes (0,1) have higher trust → higher rank_log_trust than far classes (2,3)
        assert S[0, 0] > S[0, 2] + 1e-9, "near classes should have higher rank_log_trust than far classes"

    # --- error handling ---

    def test_invalid_mode_raises(self, D_3x4):
        with pytest.raises(ValueError, match="Unknown score mode"):
            distance_matrix_to_geometry_scores(D_3x4, "bad_mode")


# ---------------------------------------------------------------------------
# FullVectorGeometricFusionCalibrator
# ---------------------------------------------------------------------------

class TestFullVectorGeometricFusionCalibrator:

    def test_fit_sets_is_fitted(self, synth):
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=1)
        cal = FullVectorGeometricFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2",
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        assert cal.is_fitted
        assert cal.best_lambda is not None

    def test_output_shape(self, synth):
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=2)
        cal = FullVectorGeometricFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2",
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        out = cal.calibrate(X_test_embed=X_val, model_probs=model_probs)
        assert out.shape == (len(X_val), 3)

    def test_output_valid_probabilities(self, synth):
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=3)
        cal = FullVectorGeometricFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2",
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        out = cal.calibrate(X_test_embed=X_val, model_probs=model_probs)
        assert np.all(out >= 0), "negative probabilities"
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)

    def test_lambda_zero_identity(self, synth):
        """lambda=0 must reproduce model_probs up to floating-point."""
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=4)
        cal = FullVectorGeometricFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2",
            lambda_grid=np.array([0.0]),   # force selection of lambda=0
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        out = cal.calibrate(X_test_embed=X_val, model_probs=model_probs)
        np.testing.assert_allclose(out, model_probs, atol=1e-6,
                                   err_msg="lambda=0 must reproduce model_probs")

    def test_lambda0_max_abs_diff_near_zero(self, synth):
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=5)
        cal = FullVectorGeometricFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2",
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        assert cal.lambda0_max_abs_diff is not None
        assert cal.lambda0_max_abs_diff < 1e-5

    def test_return_details_keys(self, synth):
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=6)
        cal = FullVectorGeometricFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2",
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        _, details = cal.calibrate(X_test_embed=X_val, model_probs=model_probs, return_details=True)
        for key in ("base_probs", "distance_matrix", "score_mode", "lambda"):
            assert key in details, f"missing key {key!r} in return_details"

    def test_get_params_keys(self, synth):
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=7)
        cal = FullVectorGeometricFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2",
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        params = cal.get_params()
        for key in ("score_mode", "best_lambda", "lambda_grid", "lambda_selection", "lambda0_max_abs_diff"):
            assert key in params, f"missing key {key!r} in get_params()"

    @pytest.mark.parametrize("mode", [
        "neg_distance", "margin", "log_trust_ratio", "rank_log_trust"
    ])
    def test_all_modes_produce_valid_probs(self, synth, mode):
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=8)
        cal = FullVectorGeometricFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2", score_mode=mode,
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        out = cal.calibrate(X_test_embed=X_val, model_probs=model_probs)
        assert np.all(out >= 0), f"mode={mode}: negative probability"
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6,
                                   err_msg=f"mode={mode}: rows don't sum to 1")

    def test_invalid_score_mode_raises(self, synth):
        X_train, y_train, _, _ = synth
        with pytest.raises(ValueError, match="score_mode must be one of"):
            FullVectorGeometricFusionCalibrator(
                model=None, X_train_embed=X_train, y_train=y_train,
                score_mode="bad",
            )

    def test_log_trust_ratio_prefers_nearest_class(self):
        """log_trust_ratio with sufficient lambda should assign highest prob to nearest class."""
        # 1-D: class 0 at x=0, class 1 at x=5, class 2 at x=10
        X_train = np.array(
            [[0.0], [0.1], [0.2], [5.0], [5.1], [5.2], [10.0], [10.1], [10.2]],
            dtype=np.float32,
        )
        y_train = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        # Val point very near class 1
        X_val = np.array([[5.05]], dtype=np.float32)
        y_val = np.array([1])
        uniform = np.array([[1 / 3, 1 / 3, 1 / 3]])

        cal = FullVectorGeometricFusionCalibrator(
            model=MockModel(uniform),
            X_train_embed=X_train,
            y_train=y_train,
            library="fast_separation",
            metric="l2",
            score_mode="log_trust_ratio",
            lambda_grid=np.array([0.0, 0.1, 0.5, 1.0, 5.0, 10.0]),
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=uniform)

        # Test also near class 1
        X_test = np.array([[4.9]], dtype=np.float32)
        out = cal.calibrate(X_test_embed=X_test, model_probs=uniform)
        assert np.argmax(out[0]) == 1, (
            f"Expected class 1 to be most confident; got class {np.argmax(out[0])}, probs={out[0]}"
        )


# ---------------------------------------------------------------------------
# FullVectorDistanceFusionCalibrator (legacy wrapper)
# ---------------------------------------------------------------------------

class TestFullVectorDistanceFusionCalibrator:

    def test_beta_attributes_after_fit(self, synth):
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=10)
        cal = FullVectorDistanceFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2",
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        assert cal.best_beta is not None
        assert isinstance(cal.beta_grid, np.ndarray)
        assert isinstance(cal.beta_selection, dict)
        assert cal.beta0_max_abs_diff is not None

    def test_get_params_legacy_keys(self, synth):
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=11)
        cal = FullVectorDistanceFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2",
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        params = cal.get_params()
        for key in ("best_beta", "beta_grid", "beta_selection", "beta0_max_abs_diff"):
            assert key in params, f"missing legacy key {key!r}"

    def test_return_details_beta_key(self, synth):
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=12)
        cal = FullVectorDistanceFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2",
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        _, details = cal.calibrate(X_test_embed=X_val, model_probs=model_probs, return_details=True)
        assert "beta" in details, "return_details must use 'beta' key, not 'lambda'"
        assert "base_probs" in details
        assert "distance_matrix" in details

    def test_legacy_formula_equivalence(self, synth):
        """Output must match softmax(log(p) - best_beta * D) exactly."""
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=13)
        cal = FullVectorDistanceFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2",
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)

        fused, details = cal.calibrate(
            X_test_embed=X_val, model_probs=model_probs, return_details=True
        )
        D = details["distance_matrix"]
        beta = cal.best_beta
        eps = cal.eps

        logits = np.log(np.clip(model_probs, eps, 1.0)) - beta * D
        logits -= logits.max(axis=1, keepdims=True)
        exp_l = np.exp(logits)
        expected = exp_l / exp_l.sum(axis=1, keepdims=True)

        np.testing.assert_allclose(fused, expected, atol=1e-6,
                                   err_msg="Legacy formula equivalence failed")

    def test_valid_probability_output(self, synth):
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=14)
        cal = FullVectorDistanceFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2",
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        out = cal.calibrate(X_test_embed=X_val, model_probs=model_probs)
        assert np.all(out >= 0)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)

    def test_beta0_max_abs_diff_near_zero(self, synth):
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=15)
        cal = FullVectorDistanceFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2",
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        assert cal.beta0_max_abs_diff < 1e-5

    def test_custom_beta_grid_respected(self, synth):
        X_train, y_train, X_val, y_val = synth
        model_probs = _make_probs(len(X_val), 3, seed=16)
        custom_grid = np.array([0.0, 0.5, 1.0])
        cal = FullVectorDistanceFusionCalibrator(
            model=MockModel(model_probs), X_train_embed=X_train, y_train=y_train,
            library="fast_separation", metric="l2", beta_grid=custom_grid,
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        assert cal.best_beta in [0.0, 0.5, 1.0]
        np.testing.assert_array_equal(cal.beta_grid, custom_grid)
