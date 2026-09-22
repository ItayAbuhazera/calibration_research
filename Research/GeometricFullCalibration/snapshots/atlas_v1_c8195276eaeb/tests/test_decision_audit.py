"""
Tests for utils/decision_audit.py.

Uses hand-constructed arrays where flip counts are known exactly.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.decision_audit import decision_audit, make_inner_validation_split


class TestDecisionAudit:
    def _make_probs(self, preds, n_classes):
        """Build one-hot-like probability arrays from a list of predicted class indices."""
        n = len(preds)
        p = np.zeros((n, n_classes), dtype=np.float64)
        for i, c in enumerate(preds):
            p[i, c] = 1.0
        # Add tiny noise to non-argmax to avoid degenerate softmax edge cases
        noise = np.ones((n, n_classes), dtype=np.float64) * 1e-9
        noise[np.arange(n), preds] = 0.0
        p = p + noise
        p /= p.sum(axis=1, keepdims=True)
        return p

    def test_known_counts(self):
        """
        Construct arrays where we know exactly:
          2 samples: base wrong, method correct (changed_to_correct)
          1 sample: base correct, method wrong (changed_to_wrong)
          1 sample: base wrong, method wrong (wrong_to_wrong, different class)
          2 samples: unchanged, base correct (correct_to_correct)
        Total N = 6.
        """
        labels = np.array([0, 1, 2, 3, 4, 5])

        # Base preds: [0, 1, 2, 3, 4, 5] — all correct initially
        # Then we'll modify
        # Sample 0: base=0, method=0 — base correct, unchanged → correct_to_correct
        # Sample 1: base=0, method=1 — base wrong (base=0, label=1), method correct → changed_to_correct
        # Sample 2: base=0, method=2 — base wrong (base=0, label=2), method correct → changed_to_correct
        # Sample 3: base=3, method=0 — base correct (base=3, label=3), method wrong → changed_to_wrong
        # Sample 4: base=0, method=5 — base wrong (label=4), method still wrong → wrong_to_wrong
        # Sample 5: base=5, method=5 — base correct, unchanged → correct_to_correct

        base_preds = np.array([0, 0, 0, 3, 0, 5])
        method_preds = np.array([0, 1, 2, 0, 5, 5])

        n_classes = 6
        base_probs = self._make_probs(base_preds, n_classes)
        method_probs = self._make_probs(method_preds, n_classes)

        result = decision_audit(base_probs, method_probs, labels)

        assert result["changed_to_correct_count"] == 2
        assert result["changed_to_wrong_count"] == 1
        assert result["wrong_to_wrong_count"] == 1
        assert result["correct_to_correct_count"] == 2
        assert result["net_flips"] == 1  # 2 - 1
        assert result["net_flip_rate"] == pytest.approx(1 / 6)
        assert result["argmax_change_rate"] == pytest.approx(4 / 6)  # samples 1,2,3,4 changed

    def test_no_changes(self):
        """When method == base, all decision fields should be zero."""
        labels = np.array([0, 1, 2])
        base_preds = np.array([0, 1, 2])
        probs = self._make_probs(base_preds, 3)
        result = decision_audit(probs, probs.copy(), labels)

        assert result["changed_to_correct_count"] == 0
        assert result["changed_to_wrong_count"] == 0
        assert result["net_flips"] == 0
        assert result["argmax_change_rate"] == pytest.approx(0.0)
        assert result["base_accuracy"] == pytest.approx(1.0)
        assert result["method_accuracy"] == pytest.approx(1.0)
        assert result["accuracy_delta"] == pytest.approx(0.0)

    def test_aliases_match_canonical(self):
        """Compatibility aliases must be identical to canonical fields."""
        labels = np.array([0, 1, 2, 3])
        base_preds = np.array([0, 0, 2, 3])
        method_preds = np.array([0, 1, 0, 3])
        base_probs = self._make_probs(base_preds, 4)
        method_probs = self._make_probs(method_preds, 4)

        result = decision_audit(base_probs, method_probs, labels)

        assert result["flip_to_correct_count"] == result["changed_to_correct_count"]
        assert result["flip_to_wrong_count"] == result["changed_to_wrong_count"]
        assert result["flip_to_correct_rate"] == result["changed_to_correct_rate"]
        assert result["flip_to_wrong_rate"] == result["changed_to_wrong_rate"]

    def test_net_flips_formula(self):
        """net_flips must equal changed_to_correct - changed_to_wrong."""
        labels = np.array([0, 1, 2, 3, 4])
        base_preds = np.array([0, 0, 2, 0, 4])
        method_preds = np.array([1, 1, 0, 3, 0])
        base_probs = self._make_probs(base_preds, 5)
        method_probs = self._make_probs(method_preds, 5)

        result = decision_audit(base_probs, method_probs, labels)
        assert result["net_flips"] == result["changed_to_correct_count"] - result["changed_to_wrong_count"]

    def test_accuracy_delta(self):
        """accuracy_delta == method_accuracy - base_accuracy."""
        labels = np.array([0, 1, 2])
        base_preds = np.array([0, 0, 2])  # 2/3 correct
        method_preds = np.array([0, 1, 2])  # 3/3 correct
        base_probs = self._make_probs(base_preds, 3)
        method_probs = self._make_probs(method_preds, 3)

        result = decision_audit(base_probs, method_probs, labels)
        assert result["base_accuracy"] == pytest.approx(2 / 3)
        assert result["method_accuracy"] == pytest.approx(1.0)
        assert result["accuracy_delta"] == pytest.approx(1 / 3)

    def test_shape_mismatch_raises(self):
        """Different shapes should raise ValueError."""
        with pytest.raises(ValueError, match="identical shape"):
            decision_audit(
                np.ones((5, 3)),
                np.ones((5, 4)),
                np.zeros(5, dtype=int),
            )

    def test_label_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="label count"):
            decision_audit(
                np.ones((5, 3)) / 3,
                np.ones((5, 3)) / 3,
                np.zeros(4, dtype=int),
            )

    def test_wrong_ndim_raises(self):
        with pytest.raises(ValueError, match="2D"):
            decision_audit(
                np.ones((5, 3, 2)),
                np.ones((5, 3, 2)),
                np.zeros(5, dtype=int),
            )

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="Empty"):
            decision_audit(
                np.zeros((0, 3)),
                np.zeros((0, 3)),
                np.zeros(0, dtype=int),
            )


class TestMakeInnerValidationSplit:
    def test_fraction_sizes(self):
        """select_fraction=0.3 should give ~30% select samples."""
        labels = np.array([0] * 50 + [1] * 50)
        fit_idx, select_idx = make_inner_validation_split(labels, select_fraction=0.3, seed=0)
        assert len(fit_idx) + len(select_idx) == len(labels)
        assert abs(len(select_idx) - 30) <= 2  # within rounding

    def test_disjoint(self):
        labels = np.array([0] * 50 + [1] * 50)
        fit_idx, select_idx = make_inner_validation_split(labels, select_fraction=0.5, seed=42)
        assert len(set(fit_idx) & set(select_idx)) == 0

    def test_default_fraction(self):
        """Default fraction is 0.5 — equal split."""
        labels = np.array([0] * 100 + [1] * 100)
        fit_idx, select_idx = make_inner_validation_split(labels, seed=0)
        assert abs(len(select_idx) - 100) <= 2

    def test_invalid_fraction_raises(self):
        labels = np.zeros(10, dtype=int)
        with pytest.raises(ValueError, match="select_fraction"):
            make_inner_validation_split(labels, select_fraction=1.5)

    def test_reproducible_with_seed(self):
        labels = np.array([0] * 30 + [1] * 30 + [2] * 30)
        fi1, si1 = make_inner_validation_split(labels, seed=7)
        fi2, si2 = make_inner_validation_split(labels, seed=7)
        np.testing.assert_array_equal(fi1, fi2)
        np.testing.assert_array_equal(si1, si2)
