"""
Tests for utils/method_metadata.py.

Verifies that registered methods return expected structural axes and that
unknown methods return None without error.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from utils.method_metadata import structural_axes_for_method


REQUIRED_KEYS = {
    "sample_dependent",
    "class_dependent",
    "joint_sample_class_dependent",
    "argmax_invariant_by_construction",
    "can_change_argmax",
    "notes",
}


def _check_axes(axes):
    assert axes is not None
    assert REQUIRED_KEYS.issubset(set(axes.keys()))
    for key in REQUIRED_KEYS - {"notes"}:
        assert isinstance(axes[key], bool), f"{key} must be bool, got {type(axes[key])}"
    assert isinstance(axes["notes"], str)


class TestStructuralAxesForMethod:
    def test_temperature_scaling_argmax_invariant(self):
        axes = structural_axes_for_method("temperature_scaling")
        _check_axes(axes)
        assert axes["argmax_invariant_by_construction"] is True
        assert axes["can_change_argmax"] is False
        assert axes["sample_dependent"] is False
        assert axes["class_dependent"] is False

    def test_pts_argmax_invariant_sample_dependent(self):
        axes = structural_axes_for_method("parameterized_temperature_scaling")
        _check_axes(axes)
        assert axes["argmax_invariant_by_construction"] is True
        assert axes["can_change_argmax"] is False
        assert axes["sample_dependent"] is True
        assert axes["class_dependent"] is False

    def test_vector_scaling_can_change_argmax(self):
        axes = structural_axes_for_method("vector_scaling")
        _check_axes(axes)
        assert axes["can_change_argmax"] is True
        assert axes["argmax_invariant_by_construction"] is False
        assert axes["class_dependent"] is True

    def test_glad_pi_joint_sample_class(self):
        axes = structural_axes_for_method("glad_pi")
        _check_axes(axes)
        assert axes["can_change_argmax"] is True
        assert axes["sample_dependent"] is True
        assert axes["class_dependent"] is True
        assert axes["joint_sample_class_dependent"] is True
        assert axes["argmax_invariant_by_construction"] is False

    def test_rgcl_argmax_invariant(self):
        axes = structural_axes_for_method("rgcl")
        _check_axes(axes)
        assert axes["argmax_invariant_by_construction"] is True
        assert axes["can_change_argmax"] is False

    def test_full_vector_geometric_fusion_prefix_match(self):
        # Prefix match for variants like full_vector_geometric_fusion_neg_distance
        axes = structural_axes_for_method("full_vector_geometric_fusion_neg_distance")
        _check_axes(axes)
        assert axes["can_change_argmax"] is True
        assert axes["joint_sample_class_dependent"] is True

    def test_kcal_lite_prefix_match(self):
        axes = structural_axes_for_method("kcal_lite_rgcl")
        _check_axes(axes)
        assert axes["can_change_argmax"] is True
        assert axes["sample_dependent"] is True

    def test_kcal_factorial_prefix_match(self):
        axes = structural_axes_for_method(
            "kcal_factorial_rgcl_blend_frozen_full_cal_rbf_sq"
        )
        _check_axes(axes)
        assert axes["can_change_argmax"] is True
        assert axes["joint_sample_class_dependent"] is True

    def test_rgcl_neighbor_correction_prefix_match(self):
        axes = structural_axes_for_method("rgcl_neighbor_correction_trust")
        _check_axes(axes)
        assert axes["can_change_argmax"] is True
        assert axes["joint_sample_class_dependent"] is True

    def test_unknown_method_returns_none(self):
        axes = structural_axes_for_method("some_completely_unknown_method_xyz")
        assert axes is None

    def test_aar_lightweight_not_joint(self):
        axes = structural_axes_for_method("aar_lightweight")
        _check_axes(axes)
        assert axes["sample_dependent"] is True
        assert axes["class_dependent"] is True
        # Not jointly sample×class (temperature is sample-only, additive is class-only)
        assert axes["joint_sample_class_dependent"] is False

    def test_trust_score_diagnostic_no_change(self):
        axes = structural_axes_for_method("trust_score_original_diagnostic")
        _check_axes(axes)
        assert axes["can_change_argmax"] is False
        assert axes["argmax_invariant_by_construction"] is True

    def test_trust_score_switch_can_change(self):
        axes = structural_axes_for_method("trust_score_original_switch")
        _check_axes(axes)
        assert axes["can_change_argmax"] is True

    def test_odir_dirichlet(self):
        axes = structural_axes_for_method("odir_dirichlet")
        _check_axes(axes)
        assert axes["can_change_argmax"] is True
        assert axes["class_dependent"] is True

    def test_base_model(self):
        axes = structural_axes_for_method("base_model")
        _check_axes(axes)
        assert axes["can_change_argmax"] is False
        assert axes["argmax_invariant_by_construction"] is True

    def test_return_is_copy(self):
        """structural_axes_for_method should return a copy, not a reference to the registry."""
        axes1 = structural_axes_for_method("temperature_scaling")
        axes2 = structural_axes_for_method("temperature_scaling")
        axes1["can_change_argmax"] = True  # mutate copy
        axes3 = structural_axes_for_method("temperature_scaling")
        assert axes3["can_change_argmax"] is False  # registry untouched
