"""
Tests for KCalLiteCalibrator and the top-k KDE-lite formula.

Covers:
  - [N, C, k] distance shape, ascending order, k=1 backward compat
  - Top-k mean-kernel formula against manual calculation
  - k=1 posterior == softmax(-gamma * D_1nn)
  - Blend endpoints (alpha=0 -> p_kcal, alpha=1 -> p_model)
  - Validation-grid selection
  - Metadata fields
  - Invalid-input rejection (k<1, k>min class count, unfitted calibrate)
  - Probability normalization
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.stability_space import StabilitySpace
from Calibrators.kcal_lite import KCalLiteCalibrator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_data():
    """Three well-separated 2-D clusters, 20 points each."""
    rng = np.random.RandomState(7)
    X0 = rng.randn(20, 2) + np.array([0.0, 0.0])
    X1 = rng.randn(20, 2) + np.array([8.0, 0.0])
    X2 = rng.randn(20, 2) + np.array([4.0, 8.0])
    X_train = np.vstack([X0, X1, X2]).astype(np.float32)
    y_train = np.array([0] * 20 + [1] * 20 + [2] * 20)
    X_val = np.array([[0.1, 0.1], [8.1, 0.1], [4.1, 8.1]], dtype=np.float32)
    y_val = np.array([0, 1, 2])
    # Uniform model probs as a neutral baseline.
    model_probs = np.full((3, 3), 1.0 / 3.0, dtype=np.float64)
    return X_train, y_train, X_val, y_val, model_probs


@pytest.fixture
def stability_space(synthetic_data):
    X_train, y_train, *_ = synthetic_data
    return StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)


@pytest.fixture
def fitted_calibrator(synthetic_data, stability_space):
    X_train, y_train, X_val, y_val, model_probs = synthetic_data
    cal = KCalLiteCalibrator(
        model=None,
        y_train=y_train,
        k_per_class=1,
        stability_space=stability_space,
    )
    cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
    return cal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _manual_kcal_probs(knn_distances: np.ndarray, gamma: float) -> np.ndarray:
    """Reference implementation: normalize(sum_r exp(-gamma*d_r)) per class."""
    N, C, k = knn_distances.shape
    gamma = float(gamma)
    if gamma == 0.0:
        return np.full((N, C), 1.0 / C, dtype=np.float64)
    a = np.sum(np.exp(-gamma * knn_distances.astype(np.float64)), axis=2)  # [N, C]
    row_sums = a.sum(axis=1, keepdims=True)
    return a / np.clip(row_sums, 1e-300, None)


# ---------------------------------------------------------------------------
# Formula correctness
# ---------------------------------------------------------------------------

class TestKCalLiteFormula:

    def test_k1_posterior_equals_softmax_1nn(self, synthetic_data, stability_space):
        """At k=1, KCal-lite posterior must equal softmax(-gamma * D_1nn) exactly."""
        _, _, X_val, _, _ = synthetic_data
        gamma = 2.0
        D_1nn = stability_space.calc_per_class_1nn_distances(X_val)   # [N, C]
        knn_d = stability_space.calc_per_class_knn_distances(X_val, k=1)  # [N, C, 1]

        p_ref = _softmax(-gamma * D_1nn, axis=1)

        cal = KCalLiteCalibrator(model=None, y_train=np.array([0]*20+[1]*20+[2]*20),
                                 k_per_class=1, stability_space=stability_space)
        p_kcal = cal._kcal_probs(knn_d, gamma)

        np.testing.assert_allclose(p_kcal, p_ref, rtol=1e-5, atol=1e-6)

    def test_k2_manual_formula(self, synthetic_data, stability_space):
        """Top-k mean-kernel formula verified against manual sum."""
        _, _, X_val, _, _ = synthetic_data
        gamma = 1.5
        knn_d = stability_space.calc_per_class_knn_distances(X_val, k=2)  # [N, C, 2]

        p_manual = _manual_kcal_probs(knn_d, gamma)

        cal = KCalLiteCalibrator(model=None, y_train=np.array([0]*20+[1]*20+[2]*20),
                                 k_per_class=2, stability_space=stability_space)
        p_kcal = cal._kcal_probs(knn_d, gamma)

        np.testing.assert_allclose(p_kcal, p_manual, rtol=1e-5, atol=1e-7)

    def test_gamma_zero_gives_uniform(self, synthetic_data, stability_space):
        _, _, X_val, _, _ = synthetic_data
        knn_d = stability_space.calc_per_class_knn_distances(X_val, k=1)
        cal = KCalLiteCalibrator(model=None, y_train=np.array([0]*20+[1]*20+[2]*20),
                                 k_per_class=1, stability_space=stability_space)
        p = cal._kcal_probs(knn_d, gamma=0.0)
        expected = np.full((len(X_val), 3), 1.0 / 3.0)
        np.testing.assert_allclose(p, expected, atol=1e-10)

    def test_kcal_probs_sum_to_one(self, synthetic_data, stability_space):
        _, _, X_val, _, _ = synthetic_data
        knn_d = stability_space.calc_per_class_knn_distances(X_val, k=3)
        cal = KCalLiteCalibrator(model=None, y_train=np.array([0]*20+[1]*20+[2]*20),
                                 k_per_class=3, stability_space=stability_space)
        for gamma in [0.0, 0.1, 1.0, 10.0]:
            p = cal._kcal_probs(knn_d, gamma)
            row_sums = p.sum(axis=1)
            np.testing.assert_allclose(row_sums, np.ones(len(X_val)), atol=1e-10)


# ---------------------------------------------------------------------------
# Blend endpoints
# ---------------------------------------------------------------------------

class TestBlendEndpoints:

    def test_alpha_one_returns_model_probs(self, synthetic_data, stability_space):
        """alpha=1 -> output == p_model."""
        _, y_train, X_val, y_val, model_probs = synthetic_data
        knn_d = stability_space.calc_per_class_knn_distances(X_val, k=1)
        cal = KCalLiteCalibrator(model=None, y_train=y_train, k_per_class=1,
                                 stability_space=stability_space)
        result = cal._blend(model_probs, knn_d, alpha=1.0, gamma=2.0)
        np.testing.assert_allclose(result, model_probs, atol=1e-10)

    def test_alpha_zero_returns_kcal_probs(self, synthetic_data, stability_space):
        """alpha=0 -> output == p_kcal."""
        _, y_train, X_val, y_val, model_probs = synthetic_data
        knn_d = stability_space.calc_per_class_knn_distances(X_val, k=1)
        cal = KCalLiteCalibrator(model=None, y_train=y_train, k_per_class=1,
                                 stability_space=stability_space)
        gamma = 2.0
        result = cal._blend(model_probs, knn_d, alpha=0.0, gamma=gamma)
        expected = cal._kcal_probs(knn_d, gamma)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_blend_output_rows_sum_to_one(self, synthetic_data, stability_space):
        _, y_train, X_val, y_val, model_probs = synthetic_data
        knn_d = stability_space.calc_per_class_knn_distances(X_val, k=1)
        cal = KCalLiteCalibrator(model=None, y_train=y_train, k_per_class=1,
                                 stability_space=stability_space)
        for alpha in [0.0, 0.5, 1.0]:
            result = cal._blend(model_probs, knn_d, alpha=alpha, gamma=1.0)
            row_sums = result.sum(axis=1)
            np.testing.assert_allclose(row_sums, np.ones(len(X_val)), atol=1e-10)


# ---------------------------------------------------------------------------
# Fit and calibrate
# ---------------------------------------------------------------------------

class TestFitCalibrate:

    def test_fit_sets_best_alpha_gamma(self, fitted_calibrator):
        assert fitted_calibrator.best_alpha is not None
        assert fitted_calibrator.best_gamma is not None

    def test_fit_sets_validation_nll_grid(self, fitted_calibrator):
        cal = fitted_calibrator
        n_alpha = len(cal.alpha_grid)
        n_gamma = len(cal.gamma_grid)
        assert cal.validation_nll_grid is not None
        assert cal.validation_nll_grid.shape == (n_alpha, n_gamma)
        assert np.all(np.isfinite(cal.validation_nll_grid))

    def test_calibrate_output_shape(self, synthetic_data, fitted_calibrator):
        _, _, X_val, _, model_probs = synthetic_data
        probs = fitted_calibrator.calibrate(X_test_embed=X_val, model_probs=model_probs)
        assert probs.shape == (len(X_val), 3)

    def test_calibrate_rows_sum_to_one(self, synthetic_data, fitted_calibrator):
        _, _, X_val, _, model_probs = synthetic_data
        probs = fitted_calibrator.calibrate(X_test_embed=X_val, model_probs=model_probs)
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, np.ones(len(X_val)), atol=1e-10)

    def test_calibrate_nonnegative(self, synthetic_data, fitted_calibrator):
        _, _, X_val, _, model_probs = synthetic_data
        probs = fitted_calibrator.calibrate(X_test_embed=X_val, model_probs=model_probs)
        assert np.all(probs >= 0.0)

    def test_selected_alpha_gamma_minimise_nll(self, synthetic_data, stability_space):
        """Verify the selected pair achieves the minimum NLL on the grid."""
        _, y_train, X_val, y_val, model_probs = synthetic_data
        cal = KCalLiteCalibrator(
            model=None, y_train=y_train, k_per_class=1, stability_space=stability_space
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        best_nll = cal.validation_nll_grid.min()
        grid_nll = cal.validation_nll_grid[
            np.where(cal.alpha_grid == cal.best_alpha)[0][0],
            np.where(cal.gamma_grid == cal.best_gamma)[0][0],
        ]
        assert abs(grid_nll - best_nll) < 1e-10

    def test_precomputed_distances_accepted(self, synthetic_data, stability_space):
        _, y_train, X_val, y_val, model_probs = synthetic_data
        knn_d = stability_space.calc_per_class_knn_distances(X_val, k=1)
        cal = KCalLiteCalibrator(
            model=None, y_train=y_train, k_per_class=1, stability_space=stability_space
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs,
                val_knn_distances=knn_d)
        probs = cal.calibrate(X_test_embed=X_val, model_probs=model_probs,
                              test_knn_distances=knn_d)
        assert probs.shape == (len(X_val), 3)


# ---------------------------------------------------------------------------
# Metadata / get_params
# ---------------------------------------------------------------------------

class TestMetadata:

    def test_get_params_fields(self, fitted_calibrator):
        p = fitted_calibrator.get_params()
        for key in [
            "metric", "library", "k_per_class", "num_labels", "is_fitted",
            "best_alpha", "best_gamma", "alpha_grid", "gamma_grid",
            "validation_nll_grid",
        ]:
            assert key in p, f"Missing key: {key}"

    def test_get_params_k_per_class(self, synthetic_data, stability_space):
        _, y_train, X_val, y_val, model_probs = synthetic_data
        cal = KCalLiteCalibrator(
            model=None, y_train=y_train, k_per_class=3, stability_space=stability_space
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=model_probs)
        assert cal.get_params()["k_per_class"] == 3

    def test_is_fitted_false_before_fit(self, synthetic_data, stability_space):
        _, y_train, *_ = synthetic_data
        cal = KCalLiteCalibrator(
            model=None, y_train=y_train, k_per_class=1, stability_space=stability_space
        )
        assert not cal.is_fitted


# ---------------------------------------------------------------------------
# Invalid input rejection
# ---------------------------------------------------------------------------

class TestInvalidInputs:

    def test_calibrate_before_fit_raises(self, synthetic_data, stability_space):
        _, y_train, X_val, _, model_probs = synthetic_data
        cal = KCalLiteCalibrator(
            model=None, y_train=y_train, k_per_class=1, stability_space=stability_space
        )
        with pytest.raises(ValueError, match="fit"):
            cal.calibrate(X_test_embed=X_val, model_probs=model_probs)

    def test_fit_without_model_probs_or_original_raises(self, synthetic_data, stability_space):
        _, y_train, X_val, y_val, _ = synthetic_data
        cal = KCalLiteCalibrator(
            model=None, y_train=y_train, k_per_class=1, stability_space=stability_space
        )
        with pytest.raises(ValueError):
            cal.fit(X_val_embed=X_val, y_val=y_val)

    def test_k_zero_raises(self, synthetic_data):
        X_train, y_train, X_val, _, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        with pytest.raises(ValueError, match="k must be >= 1"):
            ss.calc_per_class_knn_distances(X_val, k=0)

    def test_k_exceeds_class_count_raises(self, synthetic_data):
        X_train, y_train, X_val, _, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        # Each class has 20 samples.
        with pytest.raises(ValueError, match="k=25"):
            ss.calc_per_class_knn_distances(X_val, k=25)

    def test_missing_x_train_embed_raises(self, synthetic_data):
        _, y_train, *_ = synthetic_data
        with pytest.raises(ValueError, match="X_train_embed"):
            KCalLiteCalibrator(model=None, y_train=y_train, k_per_class=1)


# ---------------------------------------------------------------------------
# k=1 agreement with SoftmaxKNNBlendCalibrator kernel
# ---------------------------------------------------------------------------

class TestK1AgreeWithSoftmaxKNN:

    def test_kcal_lite_k1_matches_knn_blend_kernel(self, synthetic_data, stability_space):
        """
        The KCal-lite p_kcal at k=1 must equal the SoftmaxKNNBlendCalibrator
        _knn_probs output, since both compute softmax(-gamma * D_1nn).
        """
        from Calibrators.geometric_calibrator import SoftmaxKNNBlendCalibrator

        _, y_train, X_val, _, _ = synthetic_data
        gamma = 3.0

        D_1nn = stability_space.calc_per_class_1nn_distances(X_val)   # [N, C]
        knn_d_3d = stability_space.calc_per_class_knn_distances(X_val, k=1)  # [N, C, 1]

        knn_blend = SoftmaxKNNBlendCalibrator(
            model=None, y_train=y_train, stability_space=stability_space
        )
        p_knn_blend = knn_blend._knn_probs(D_1nn, gamma)  # [N, C]

        kcal = KCalLiteCalibrator(
            model=None, y_train=y_train, k_per_class=1, stability_space=stability_space
        )
        p_kcal = kcal._kcal_probs(knn_d_3d, gamma)  # [N, C]

        np.testing.assert_allclose(p_kcal, p_knn_blend, rtol=1e-5, atol=1e-6)
