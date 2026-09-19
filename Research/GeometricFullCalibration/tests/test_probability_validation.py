"""
Tests for external method probability import validation.

Verifies that the validation logic (from utils/unified_metrics.py) correctly
rejects malformed probability matrices that could come from external sources
like official AAR outputs or other external calibrators.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.unified_metrics import validate_probability_matrix


def _valid_probs(n=50, c=10, seed=0):
    rng = np.random.RandomState(seed)
    raw = rng.dirichlet(np.ones(c), size=n)
    return raw.astype(np.float64)


class TestValidateProbabilityMatrix:
    def test_valid_matrix_accepted(self):
        """A properly normalized [N, C] matrix should pass without error."""
        probs = _valid_probs(50, 10)
        labels = np.random.randint(0, 10, size=50)
        validate_probability_matrix(probs, labels)  # should not raise

    def test_3d_array_rejected(self):
        """3D array must be rejected."""
        probs = np.ones((5, 3, 2)) / 6
        labels = np.zeros(5, dtype=int)
        with pytest.raises(ValueError, match=r"\[N, C\]|shape"):
            validate_probability_matrix(probs, labels)

    def test_n_mismatch_rejected(self):
        """Probability matrix with wrong N must be rejected."""
        probs = _valid_probs(50, 10)
        labels = np.zeros(40, dtype=int)  # wrong length
        with pytest.raises(ValueError, match="differ|align"):
            validate_probability_matrix(probs, labels)

    def test_non_normalized_rows_rejected(self):
        """Rows that do not sum to 1 must be rejected."""
        probs = np.ones((5, 3), dtype=np.float64) * 0.5  # rows sum to 1.5
        labels = np.zeros(5, dtype=int)
        with pytest.raises(ValueError, match="sum to 1|row_sum"):
            validate_probability_matrix(probs, labels)

    def test_negative_values_rejected(self):
        """Negative probability values must be rejected."""
        probs = _valid_probs(10, 5)
        probs[0, 0] = -0.1
        probs[0, 1] += 0.1  # keep sum ~1
        labels = np.zeros(10, dtype=int)
        with pytest.raises(ValueError):
            validate_probability_matrix(probs, labels)

    def test_nan_rejected(self):
        probs = _valid_probs(10, 5)
        probs[2, 3] = np.nan
        labels = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="NaN|finite"):
            validate_probability_matrix(probs, labels)

    def test_inf_rejected(self):
        probs = _valid_probs(10, 5)
        probs[1, 0] = np.inf
        labels = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="Inf|finite"):
            validate_probability_matrix(probs, labels)

    def test_empty_matrix_rejected(self):
        with pytest.raises(ValueError, match="Empty|empty"):
            validate_probability_matrix(np.zeros((0, 5)), np.zeros(0, dtype=int))

    def test_out_of_range_label_rejected(self):
        """Labels outside [0, C-1] must be rejected."""
        probs = _valid_probs(10, 5)
        labels = np.array([0, 1, 2, 3, 4, 5, 4, 3, 2, 1])  # label 5 >= C=5
        with pytest.raises(ValueError, match="Label|range"):
            validate_probability_matrix(probs, labels)

    def test_tolerance_boundary(self):
        """Row sums exactly at tolerance boundary should be accepted."""
        probs = _valid_probs(10, 5)
        probs[0] /= probs[0].sum()
        probs[0, 0] += 5e-7  # within default 1e-6 tolerance
        labels = np.zeros(10, dtype=int)
        # Should not raise; at the boundary of tolerance
        validate_probability_matrix(probs, labels, sum_tolerance=1e-6)


class TestExternalMethodImportSimulation:
    """
    Simulate the external method import workflow end-to-end.

    This does not test run_unified_benchmark.py directly (that requires a real model),
    but validates that the probability-level entry point behaves correctly for
    a synthetic external probability matrix.
    """

    def test_external_probs_evaluate_all(self):
        """Valid external probs should run through evaluate_all without error."""
        from utils.unified_metrics import evaluate_all
        from utils.decision_audit import decision_audit

        n, c = 100, 10
        base_probs = _valid_probs(n, c, seed=0)
        ext_probs = _valid_probs(n, c, seed=1)
        labels = np.random.randint(0, c, size=n)

        metrics = evaluate_all(ext_probs, labels)
        assert "accuracy" in metrics
        assert "nll" in metrics

        audit = decision_audit(base_probs, ext_probs, labels)
        assert "net_flips" in audit
        assert "changed_to_correct_count" in audit

    def test_external_probs_structural_axes(self):
        """Known external method names should resolve structural axes."""
        from utils.method_metadata import structural_axes_for_method
        axes = structural_axes_for_method("aar_lightweight")
        assert axes is not None
        assert axes["can_change_argmax"] is True
