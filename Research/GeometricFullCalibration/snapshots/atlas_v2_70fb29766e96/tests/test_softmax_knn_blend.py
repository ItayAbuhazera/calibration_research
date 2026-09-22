"""Tests for SoftmaxKNNBlendCalibrator."""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Calibrators.geometric_calibrator import SoftmaxKNNBlendCalibrator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class MockModel:
    def __init__(self, probs: np.ndarray):
        self._probs = np.asarray(probs, dtype=np.float64)

    def predict_proba(self, X):
        return self._probs[: len(X)]


def _make_probs(n: int, K: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    p = np.abs(rng.randn(n, K)) + 0.1
    return (p / p.sum(axis=1, keepdims=True)).astype(np.float64)


def _make_synthetic(seed: int = 0, n_per_class: int = 30, n_val: int = 20,
                    dim: int = 4, K: int = 3):
    rng = np.random.RandomState(seed)
    centers = rng.randn(K, dim) * 4.0
    parts = [rng.randn(n_per_class, dim) + centers[k] for k in range(K)]
    X_train = np.vstack(parts).astype(np.float32)
    y_train = np.repeat(np.arange(K), n_per_class)
    X_val = (rng.randn(n_val, dim) + centers[0]).astype(np.float32)
    y_val = np.zeros(n_val, dtype=int)
    return X_train, y_train, X_val, y_val


def _make_calibrator(synth, seed=1, **kwargs):
    X_train, y_train, X_val, y_val = synth
    probs = _make_probs(len(X_val), 3, seed=seed)
    cal = SoftmaxKNNBlendCalibrator(
        model=MockModel(probs),
        y_train=y_train,
        X_train_embed=X_train,
        library="fast_separation",
        metric="l2",
        **kwargs,
    )
    return cal, probs, X_val, y_val


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def synth():
    return _make_synthetic(seed=42)


# ---------------------------------------------------------------------------
# kNN posterior stability
# ---------------------------------------------------------------------------

class TestKNNPosterior:

    def test_softmax_knn_valid_probs(self, synth):
        """softmax(-gamma*D) must be non-negative and sum to 1 for any gamma."""
        X_train, y_train, X_val, _ = synth
        from utils.stability_space import StabilitySpace
        stab = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2")
        dm = stab.calc_per_class_1nn_distances(X_val)

        cal = SoftmaxKNNBlendCalibrator(
            model=MockModel(_make_probs(len(X_val), 3, seed=0)),
            y_train=y_train,
            stability_space=stab,
        )
        for gamma in SoftmaxKNNBlendCalibrator.DEFAULT_GAMMA_GRID:
            p_knn = cal._knn_probs(dm, float(gamma))
            assert np.all(np.isfinite(p_knn)), f"gamma={gamma}: non-finite kNN probs"
            assert np.all(p_knn >= 0), f"gamma={gamma}: negative kNN probs"
            np.testing.assert_allclose(
                p_knn.sum(axis=1), 1.0, atol=1e-6,
                err_msg=f"gamma={gamma}: kNN probs don't sum to 1",
            )

    def test_gamma_zero_gives_uniform(self, synth):
        """gamma=0 must produce a uniform distribution (softmax of zeros)."""
        X_train, y_train, X_val, _ = synth
        from utils.stability_space import StabilitySpace
        stab = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2")
        dm = stab.calc_per_class_1nn_distances(X_val)
        K = len(np.unique(y_train))

        cal = SoftmaxKNNBlendCalibrator(
            model=MockModel(_make_probs(len(X_val), K, seed=0)),
            y_train=y_train,
            stability_space=stab,
        )
        p_knn = cal._knn_probs(dm, gamma=0.0)
        expected = np.full((len(X_val), K), 1.0 / K)
        np.testing.assert_allclose(p_knn, expected, atol=1e-12,
                                   err_msg="gamma=0 must give uniform kNN probs")


# ---------------------------------------------------------------------------
# Blend endpoints
# ---------------------------------------------------------------------------

class TestBlendEndpoints:

    def test_alpha_one_reproduces_base_model_all_gammas(self, synth):
        """alpha=1 must exactly reproduce base-model probabilities for every gamma."""
        X_train, y_train, X_val, y_val = synth
        probs = _make_probs(len(X_val), 3, seed=10)
        from utils.stability_space import StabilitySpace
        stab = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2")
        dm = stab.calc_per_class_1nn_distances(X_val)

        cal = SoftmaxKNNBlendCalibrator(
            model=MockModel(probs),
            y_train=y_train,
            stability_space=stab,
        )
        for gamma in SoftmaxKNNBlendCalibrator.DEFAULT_GAMMA_GRID:
            blended = cal._blend(probs, dm, alpha=1.0, gamma=float(gamma))
            np.testing.assert_allclose(
                blended, probs, atol=1e-12,
                err_msg=f"alpha=1, gamma={gamma}: blend must equal base-model probs exactly",
            )

    def test_alpha_one_via_fit_and_calibrate(self, synth):
        """When alpha_grid=[1.0], calibrate output must match base-model probs."""
        cal, probs, X_val, y_val = _make_calibrator(
            synth, seed=11, alpha_grid=np.array([1.0])
        )
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        out = cal.calibrate(X_test_embed=X_val, model_probs=probs)
        np.testing.assert_allclose(out, probs, atol=1e-12,
                                   err_msg="alpha=1 path: calibrate must reproduce base-model probs")

    def test_alpha_zero_gives_knn_posterior(self, synth):
        """alpha=0 blend must equal softmax(-gamma*D) for the selected gamma."""
        X_train, y_train, X_val, _ = synth
        probs = _make_probs(len(X_val), 3, seed=12)
        from utils.stability_space import StabilitySpace
        stab = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2")
        dm = stab.calc_per_class_1nn_distances(X_val)

        cal = SoftmaxKNNBlendCalibrator(
            model=MockModel(probs),
            y_train=y_train,
            stability_space=stab,
        )
        gamma = 1.0
        blended = cal._blend(probs, dm, alpha=0.0, gamma=gamma)
        p_knn = cal._knn_probs(dm, gamma)
        np.testing.assert_allclose(blended, p_knn, atol=1e-12,
                                   err_msg="alpha=0 must equal p_knn")


# ---------------------------------------------------------------------------
# Validation-NLL selection
# ---------------------------------------------------------------------------

class TestNLLSelection:

    def test_deterministic(self, synth):
        """Fitting twice must select the same alpha and gamma."""
        cal1, probs, X_val, y_val = _make_calibrator(synth, seed=20)
        cal2, _, _, _ = _make_calibrator(synth, seed=20)
        cal1.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        cal2.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        assert cal1.best_alpha == cal2.best_alpha
        assert cal1.best_gamma == cal2.best_gamma

    def test_selected_alpha_in_grid(self, synth):
        cal, probs, X_val, y_val = _make_calibrator(synth, seed=21)
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        assert cal.best_alpha in cal.alpha_grid.tolist()

    def test_selected_gamma_in_grid(self, synth):
        cal, probs, X_val, y_val = _make_calibrator(synth, seed=22)
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        assert cal.best_gamma in cal.gamma_grid.tolist()

    def test_selected_pair_minimises_nll(self, synth):
        """The selected (alpha, gamma) must achieve the minimum NLL in the grid."""
        cal, probs, X_val, y_val = _make_calibrator(synth, seed=23)
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        grid = cal.validation_nll_grid
        i_best = np.unravel_index(np.argmin(grid), grid.shape)
        assert cal.best_alpha == pytest.approx(cal.alpha_grid[i_best[0]])
        assert cal.best_gamma == pytest.approx(cal.gamma_grid[i_best[1]])


# ---------------------------------------------------------------------------
# Metadata and grid shape
# ---------------------------------------------------------------------------

class TestMetadataAndGridShape:

    def test_nll_grid_shape(self, synth):
        cal, probs, X_val, y_val = _make_calibrator(synth, seed=30)
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        expected = (
            len(SoftmaxKNNBlendCalibrator.DEFAULT_ALPHA_GRID),
            len(SoftmaxKNNBlendCalibrator.DEFAULT_GAMMA_GRID),
        )
        assert cal.validation_nll_grid.shape == expected

    def test_nll_grid_finite(self, synth):
        cal, probs, X_val, y_val = _make_calibrator(synth, seed=31)
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        assert np.all(np.isfinite(cal.validation_nll_grid))

    def test_get_params_keys(self, synth):
        required = {
            "selected_alpha", "selected_gamma",
            "alpha_grid", "gamma_grid", "validation_nll_grid",
        }
        cal, probs, X_val, y_val = _make_calibrator(synth, seed=32)
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        params = cal.get_params()
        for key in required:
            assert key in params, f"missing key {key!r} in get_params()"

    def test_get_params_grids_match_defaults(self, synth):
        cal, probs, X_val, y_val = _make_calibrator(synth, seed=33)
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        params = cal.get_params()
        np.testing.assert_array_equal(
            params["alpha_grid"], SoftmaxKNNBlendCalibrator.DEFAULT_ALPHA_GRID.tolist()
        )
        np.testing.assert_array_equal(
            params["gamma_grid"], SoftmaxKNNBlendCalibrator.DEFAULT_GAMMA_GRID.tolist()
        )

    def test_is_fitted_flag(self, synth):
        cal, probs, X_val, y_val = _make_calibrator(synth, seed=34)
        assert not cal.is_fitted
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        assert cal.is_fitted


# ---------------------------------------------------------------------------
# Invalid-input rejection
# ---------------------------------------------------------------------------

class TestInvalidInputs:

    def test_calibrate_before_fit_raises(self, synth):
        cal, probs, X_val, _ = _make_calibrator(synth, seed=40)
        with pytest.raises(ValueError, match="fit"):
            cal.calibrate(X_test_embed=X_val, model_probs=probs)

    def test_fit_without_probs_or_original_raises(self, synth):
        cal, _, X_val, y_val = _make_calibrator(synth, seed=41)
        with pytest.raises(ValueError):
            cal.fit(X_val_embed=X_val, y_val=y_val)  # neither model_probs nor X_val_original

    def test_calibrate_without_probs_or_original_raises(self, synth):
        cal, probs, X_val, y_val = _make_calibrator(synth, seed=42)
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        with pytest.raises(ValueError):
            cal.calibrate(X_test_embed=X_val)  # neither model_probs nor X_test_original

    def test_no_train_embed_without_stability_space_raises(self, synth):
        X_train, y_train, X_val, y_val = synth
        probs = _make_probs(len(X_val), 3, seed=43)
        with pytest.raises(ValueError, match="X_train_embed"):
            SoftmaxKNNBlendCalibrator(
                model=MockModel(probs),
                y_train=y_train,
                # X_train_embed omitted and stability_space=None
            )


# ---------------------------------------------------------------------------
# Normalized output probabilities
# ---------------------------------------------------------------------------

class TestNormalizedOutput:

    def test_output_non_negative(self, synth):
        cal, probs, X_val, y_val = _make_calibrator(synth, seed=50)
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        out = cal.calibrate(X_test_embed=X_val, model_probs=probs)
        assert np.all(out >= 0), "negative probability in output"

    def test_output_rows_sum_to_one(self, synth):
        cal, probs, X_val, y_val = _make_calibrator(synth, seed=51)
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        out = cal.calibrate(X_test_embed=X_val, model_probs=probs)
        np.testing.assert_allclose(out.sum(axis=1), 1.0, atol=1e-6)

    def test_output_finite(self, synth):
        cal, probs, X_val, y_val = _make_calibrator(synth, seed=52)
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        out = cal.calibrate(X_test_embed=X_val, model_probs=probs)
        assert np.all(np.isfinite(out)), "non-finite values in output"

    def test_output_shape(self, synth):
        X_train, y_train, X_val, y_val = synth
        K = len(np.unique(y_train))
        cal, probs, X_val, y_val = _make_calibrator(synth, seed=53)
        cal.fit(X_val_embed=X_val, y_val=y_val, model_probs=probs)
        out = cal.calibrate(X_test_embed=X_val, model_probs=probs)
        assert out.shape == (len(X_val), K)
