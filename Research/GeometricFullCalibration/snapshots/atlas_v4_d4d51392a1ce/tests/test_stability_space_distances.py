"""
Tests for per-class and predicted-class 1-NN distance correctness across backends.

CPU-only tests run unconditionally (fast_separation, kdtree).
FAISS tests are skipped when the faiss package is unavailable.
"""
import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.stability_space import StabilitySpace


@pytest.fixture
def synthetic_data():
    """2D, three-class data with three well-separated clusters."""
    rng = np.random.RandomState(42)
    X0 = rng.randn(20, 2) + np.array([0.0, 0.0])
    X1 = rng.randn(20, 2) + np.array([5.0, 0.0])
    X2 = rng.randn(20, 2) + np.array([2.5, 5.0])
    X_train = np.vstack([X0, X1, X2]).astype(np.float32)
    y_train = np.array([0] * 20 + [1] * 20 + [2] * 20)

    # One query point near each cluster; predicted label matches cluster.
    X_val = np.array([[0.1, 0.1], [5.1, 0.1], [2.6, 5.1]], dtype=np.float32)
    y_val_pred = np.array([0, 1, 2])

    return X_train, y_train, X_val, y_val_pred


# ---------------------------------------------------------------------------
# CPU tests: fast_separation and kdtree
# ---------------------------------------------------------------------------

class TestCpuPerClassDistances:

    def test_shape_fast_separation(self, synthetic_data):
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        D = ss.calc_per_class_1nn_distances(X_val)
        assert D.shape == (len(X_val), 3)

    def test_shape_kdtree(self, synthetic_data):
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="kdtree", metric="l2", use_cuda=False)
        D = ss.calc_per_class_1nn_distances(X_val)
        assert D.shape == (len(X_val), 3)

    def test_fast_separation_and_kdtree_agree(self, synthetic_data):
        X_train, y_train, X_val, _ = synthetic_data
        ss_fs = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        ss_kd = StabilitySpace(X_train, y_train, library="kdtree", metric="l2", use_cuda=False)
        D_fs = ss_fs.calc_per_class_1nn_distances(X_val)
        D_kd = ss_kd.calc_per_class_1nn_distances(X_val)
        np.testing.assert_allclose(D_fs, D_kd, rtol=1e-4, atol=1e-4)

    def test_nearest_class_is_predicted_class(self, synthetic_data):
        """With well-separated clusters each query's nearest class equals its predicted label."""
        X_train, y_train, X_val, y_val_pred = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        D = ss.calc_per_class_1nn_distances(X_val)
        for i, pred in enumerate(y_val_pred):
            assert D[i, pred] == pytest.approx(D[i, :].min(), abs=1e-4)

    def test_predicted_class_distances_match_per_class_columns(self, synthetic_data):
        """calc_predicted_class_same_distances must equal the per-class column for fast_separation."""
        X_train, y_train, X_val, y_val_pred = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        D = ss.calc_per_class_1nn_distances(X_val)
        same_d = ss.calc_predicted_class_same_distances(X_val, y_val_pred)
        expected = D[np.arange(len(y_val_pred)), y_val_pred]
        np.testing.assert_allclose(same_d, expected, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# FAISS tests — skipped when faiss is not installed
# ---------------------------------------------------------------------------

class TestFaissDistances:

    def test_per_class_distances_shape(self, synthetic_data):
        pytest.importorskip("faiss")
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="faiss", metric="l2", use_cuda=False)
        D = ss.calc_per_class_1nn_distances(X_val)
        assert D.shape == (len(X_val), 3)

    def test_per_class_distances_agree_with_fast_separation(self, synthetic_data):
        pytest.importorskip("faiss")
        X_train, y_train, X_val, _ = synthetic_data
        ss_fs = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        ss_fa = StabilitySpace(X_train, y_train, library="faiss", metric="l2", use_cuda=False)
        D_fs = ss_fs.calc_per_class_1nn_distances(X_val)
        D_fa = ss_fa.calc_per_class_1nn_distances(X_val)
        np.testing.assert_allclose(D_fs, D_fa, rtol=1e-3, atol=1e-3)

    def test_predicted_class_distances_agree_with_fast_separation(self, synthetic_data):
        pytest.importorskip("faiss")
        X_train, y_train, X_val, y_val_pred = synthetic_data
        ss_fs = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        ss_fa = StabilitySpace(X_train, y_train, library="faiss", metric="l2", use_cuda=False)
        d_fs = ss_fs.calc_predicted_class_same_distances(X_val, y_val_pred)
        d_fa = ss_fa.calc_predicted_class_same_distances(X_val, y_val_pred)
        np.testing.assert_allclose(d_fs, d_fa, rtol=1e-3, atol=1e-3)

    def test_stability_agrees_with_fast_separation(self, synthetic_data):
        pytest.importorskip("faiss")
        X_train, y_train, X_val, y_val_pred = synthetic_data
        ss_fs = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        ss_fa = StabilitySpace(X_train, y_train, library="faiss", metric="l2", use_cuda=False)
        stab_fs = ss_fs.calc_stab(X_val, y_val_pred)
        stab_fa = ss_fa.calc_stab(X_val, y_val_pred)
        np.testing.assert_allclose(stab_fs, stab_fa, rtol=1e-2, atol=1e-2)

    def test_trust_score_agrees_with_fast_separation(self, synthetic_data):
        pytest.importorskip("faiss")
        X_train, y_train, X_val, y_val_pred = synthetic_data
        ss_fs = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        ss_fa = StabilitySpace(X_train, y_train, library="faiss", metric="l2", use_cuda=False)
        ts_fs = ss_fs.calc_trust_score(X_val, y_val_pred)
        ts_fa = ss_fa.calc_trust_score(X_val, y_val_pred)
        np.testing.assert_allclose(ts_fs, ts_fa, rtol=0.05, atol=0.05)


# ---------------------------------------------------------------------------
# kNN distance tests (k > 1)
# ---------------------------------------------------------------------------

class TestKnnPerClassDistances:
    """Tests for calc_per_class_knn_distances across CPU backends."""

    def test_shape_k1_fast_separation(self, synthetic_data):
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        D = ss.calc_per_class_knn_distances(X_val, k=1)
        assert D.shape == (len(X_val), 3, 1)

    def test_shape_k3_fast_separation(self, synthetic_data):
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        D = ss.calc_per_class_knn_distances(X_val, k=3)
        assert D.shape == (len(X_val), 3, 3)

    def test_shape_k1_kdtree(self, synthetic_data):
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="kdtree", metric="l2", use_cuda=False)
        D = ss.calc_per_class_knn_distances(X_val, k=1)
        assert D.shape == (len(X_val), 3, 1)

    def test_shape_k3_kdtree(self, synthetic_data):
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="kdtree", metric="l2", use_cuda=False)
        D = ss.calc_per_class_knn_distances(X_val, k=3)
        assert D.shape == (len(X_val), 3, 3)

    def test_distances_ascending_fast_separation(self, synthetic_data):
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        D = ss.calc_per_class_knn_distances(X_val, k=5)
        # Each row of distances within a class must be non-decreasing.
        for i in range(len(X_val)):
            for c in range(3):
                diffs = np.diff(D[i, c, :])
                assert np.all(diffs >= -1e-6), (
                    f"Distances not ascending for sample {i}, class {c}: {D[i, c, :]}"
                )

    def test_distances_ascending_kdtree(self, synthetic_data):
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="kdtree", metric="l2", use_cuda=False)
        D = ss.calc_per_class_knn_distances(X_val, k=5)
        for i in range(len(X_val)):
            for c in range(3):
                diffs = np.diff(D[i, c, :])
                assert np.all(diffs >= -1e-6), (
                    f"Distances not ascending for sample {i}, class {c}: {D[i, c, :]}"
                )

    def test_k1_matches_1nn_distances_fast_separation(self, synthetic_data):
        """calc_per_class_knn_distances(X, 1)[:, :, 0] must equal calc_per_class_1nn_distances(X)."""
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        D_knn = ss.calc_per_class_knn_distances(X_val, k=1)
        D_1nn = ss.calc_per_class_1nn_distances(X_val)
        np.testing.assert_array_equal(D_knn[:, :, 0], D_1nn)

    def test_k1_matches_1nn_distances_kdtree(self, synthetic_data):
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="kdtree", metric="l2", use_cuda=False)
        D_knn = ss.calc_per_class_knn_distances(X_val, k=1)
        D_1nn = ss.calc_per_class_1nn_distances(X_val)
        np.testing.assert_array_equal(D_knn[:, :, 0], D_1nn)

    def test_fast_separation_kdtree_agree_k3(self, synthetic_data):
        X_train, y_train, X_val, _ = synthetic_data
        ss_fs = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        ss_kd = StabilitySpace(X_train, y_train, library="kdtree", metric="l2", use_cuda=False)
        D_fs = ss_fs.calc_per_class_knn_distances(X_val, k=3)
        D_kd = ss_kd.calc_per_class_knn_distances(X_val, k=3)
        np.testing.assert_allclose(D_fs, D_kd, rtol=1e-4, atol=1e-4)

    def test_excessive_k_raises_value_error(self, synthetic_data):
        """k > min class count must raise a clear ValueError."""
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        # Each class has 20 samples; k=21 should fail.
        with pytest.raises(ValueError, match="k=21"):
            ss.calc_per_class_knn_distances(X_val, k=21)

    def test_k_less_than_1_raises_value_error(self, synthetic_data):
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        with pytest.raises(ValueError, match="k must be >= 1"):
            ss.calc_per_class_knn_distances(X_val, k=0)


class TestKnnFaissDistances:
    """FAISS-specific kNN distance tests (skipped when faiss is unavailable)."""

    def test_shape_k3(self, synthetic_data):
        pytest.importorskip("faiss")
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="faiss", metric="l2", use_cuda=False)
        D = ss.calc_per_class_knn_distances(X_val, k=3)
        assert D.shape == (len(X_val), 3, 3)

    def test_k1_matches_1nn_distances(self, synthetic_data):
        pytest.importorskip("faiss")
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="faiss", metric="l2", use_cuda=False)
        D_knn = ss.calc_per_class_knn_distances(X_val, k=1)
        D_1nn = ss.calc_per_class_1nn_distances(X_val)
        np.testing.assert_allclose(D_knn[:, :, 0], D_1nn, rtol=1e-5, atol=1e-5)

    def test_k3_agrees_with_fast_separation(self, synthetic_data):
        pytest.importorskip("faiss")
        X_train, y_train, X_val, _ = synthetic_data
        ss_fs = StabilitySpace(X_train, y_train, library="fast_separation", metric="l2", use_cuda=False)
        ss_fa = StabilitySpace(X_train, y_train, library="faiss", metric="l2", use_cuda=False)
        D_fs = ss_fs.calc_per_class_knn_distances(X_val, k=3)
        D_fa = ss_fa.calc_per_class_knn_distances(X_val, k=3)
        np.testing.assert_allclose(D_fs, D_fa, rtol=1e-3, atol=1e-3)

    def test_distances_ascending(self, synthetic_data):
        pytest.importorskip("faiss")
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(X_train, y_train, library="faiss", metric="l2", use_cuda=False)
        D = ss.calc_per_class_knn_distances(X_val, k=5)
        for i in range(len(X_val)):
            for c in range(3):
                diffs = np.diff(D[i, c, :])
                assert np.all(diffs >= -1e-5), (
                    f"FAISS distances not ascending for sample {i}, class {c}: {D[i, c, :]}"
                )


# ---------------------------------------------------------------------------
# Whitened-cosine metric
# ---------------------------------------------------------------------------

class TestWhitenedCosineDistances:
    """Backend-independent invariants for PCA-whitened cosine geometry."""

    def test_knn_distances_are_sorted_and_k1_matches_1nn(self, synthetic_data):
        X_train, y_train, X_val, _ = synthetic_data
        ss = StabilitySpace(
            X_train,
            y_train,
            library="fast_separation",
            metric="whitened_cosine",
            whitening_components=2,
            use_cuda=False,
        )

        D = ss.calc_per_class_knn_distances(X_val, k=5)
        assert D.shape == (len(X_val), 3, 5)
        assert np.all(np.diff(D, axis=2) >= -1e-6)
        np.testing.assert_array_equal(
            ss.calc_per_class_knn_distances(X_val, k=1)[:, :, 0],
            ss.calc_per_class_1nn_distances(X_val),
        )

    def test_predicted_class_column_consistency(self, synthetic_data):
        X_train, y_train, X_val, y_val_pred = synthetic_data
        ss = StabilitySpace(
            X_train,
            y_train,
            library="fast_separation",
            metric="whitened_cosine",
            whitening_components=2,
            use_cuda=False,
        )
        D = ss.calc_per_class_1nn_distances(X_val)
        same_d = ss.calc_predicted_class_same_distances(X_val, y_val_pred)
        np.testing.assert_allclose(
            same_d,
            D[np.arange(len(y_val_pred)), y_val_pred],
            rtol=1e-5,
            atol=1e-6,
        )

    def test_isotropic_full_rank_data_matches_cosine(self):
        # This centered reference set has covariance proportional to identity.
        # Full-rank whitening is therefore only a scaled orthogonal transform,
        # which leaves cosine distances unchanged.
        X_train = np.vstack([np.eye(4), -np.eye(4)]).astype(np.float32)
        y_train = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        X_val = np.array(
            [[0.5, -0.25, 0.75, 0.1], [-0.2, 0.6, 0.3, -0.7]],
            dtype=np.float32,
        )
        cosine = StabilitySpace(
            X_train, y_train, library="fast_separation", metric="cosine", use_cuda=False
        )
        whitened = StabilitySpace(
            X_train,
            y_train,
            library="fast_separation",
            metric="whitened_cosine",
            whitening_components=4,
            use_cuda=False,
        )
        np.testing.assert_allclose(
            whitened.calc_per_class_knn_distances(X_val, k=2),
            cosine.calc_per_class_knn_distances(X_val, k=2),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_whitening_params_and_component_cap_are_logged(self, synthetic_data):
        X_train, y_train, _, _ = synthetic_data
        ss = StabilitySpace(
            X_train,
            y_train,
            library="fast_separation",
            metric="whitened_cosine",
            whitening_components=128,
            whitening_eps=1e-5,
            use_cuda=False,
        )
        params = ss.get_params()
        assert params["whitening_components"] == 128
        assert params["whitening_components_fitted"] == X_train.shape[1]
        assert params["whitening_eps"] == pytest.approx(1e-5)


class TestWhitenedCosineFaissDistances:
    def test_faiss_agrees_with_sklearn(self, synthetic_data):
        pytest.importorskip("faiss")
        X_train, y_train, X_val, _ = synthetic_data
        kwargs = dict(
            metric="whitened_cosine", whitening_components=2, use_cuda=False
        )
        sklearn_space = StabilitySpace(
            X_train, y_train, library="fast_separation", **kwargs
        )
        faiss_space = StabilitySpace(X_train, y_train, library="faiss", **kwargs)
        np.testing.assert_allclose(
            faiss_space.calc_per_class_knn_distances(X_val, k=5),
            sklearn_space.calc_per_class_knn_distances(X_val, k=5),
            rtol=1e-4,
            atol=1e-5,
        )
