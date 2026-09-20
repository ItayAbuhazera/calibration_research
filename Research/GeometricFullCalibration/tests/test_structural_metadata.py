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


PHASE0_1_RUNNER_METHODS = (
    "base_model", "temperature_scaling", "parameterized_temperature_scaling",
    "vector_scaling", "beta_calibration", "ovr_isotonic", "odir_dirichlet",
    "native_dac", "gc_dac", "top_label_isotonic", "rgcl",
    "anchored_model_tail", "anchored_rankgeom_tail_mixture",
    "full_vector_distance_fusion", "post_fusion_topiso",
    "trust_score_original_diagnostic", "trust_score_original_switch",
    "glad_pi", "glad_pi_zero_geometry", "mahalanobis_confidence", "kcal",
)


def test_every_phase0_1_method_has_structural_axes():
    from utils.method_metadata import structural_axes_for_method

    missing = [m for m in PHASE0_1_RUNNER_METHODS if structural_axes_for_method(m) is None]
    assert not missing, f"Phase 0/1 methods without structural axes: {missing}"


def test_registry_semantics_for_key_methods():
    """Pins the semantics the scientific question depends on."""
    from utils.method_metadata import method_semantics

    def sem(m):
        a = method_semantics(m)
        return (
            a["output_semantics"], a["effective_prediction_source"],
            a["can_change_argmax"], a["metric_bucket"],
            a["sample_dependent"], a["representation_geometry_dependent"],
        )

    # Scalar confidence, prediction stays the base prediction.
    assert sem("top_label_isotonic") == ("scalar_confidence", "base", False, "scalar_only", False, False)
    assert sem("gc_dac") == ("scalar_confidence", "base", False, "scalar_only", True, True)
    assert sem("rgcl") == ("scalar_confidence", "base", False, "scalar_only", True, True)
    assert sem("gc_tulip") == ("scalar_confidence", "base", False, "scalar_only", True, True)
    assert sem("mahalanobis_confidence") == ("scalar_confidence", "base", False, "scalar_only", True, True)
    assert sem("trust_score_original_diagnostic") == ("scalar_confidence", "base", False, "scalar_only", True, True)

    # Declared decision intervention: the one scalar method that may switch.
    assert sem("trust_score_original_switch") == (
        "scalar_confidence", "explicit_switch", True, "scalar_only", True, True,
    )

    # Full vector, argmax-invariant by a positive-scalar-temperature proof.
    assert sem("temperature_scaling") == ("full_vector", "calibrated", False, "full_vector", False, False)
    assert sem("parameterized_temperature_scaling") == ("full_vector", "calibrated", False, "full_vector", True, False)
    assert sem("native_dac") == ("full_vector", "calibrated", False, "full_vector", True, True)

    # Full vector, genuinely decision-changing.
    assert sem("vector_scaling") == ("full_vector", "calibrated", True, "full_vector", False, False)
    assert sem("odir_dirichlet") == ("full_vector", "calibrated", True, "full_vector", False, False)

    # The matched geometry ablation differs ONLY on the geometry axis.
    g, z = method_semantics("glad_pi"), method_semantics("glad_pi_zero_geometry")
    assert g["representation_geometry_dependent"] is True
    assert z["representation_geometry_dependent"] is False
    for k in ("output_semantics", "effective_prediction_source", "metric_bucket",
              "can_change_argmax", "sample_dependent", "class_dependent",
              "joint_sample_class_dependent"):
        assert g[k] == z[k], f"GLAD-PI arms differ on {k}, which is not the geometry axis"


def test_argmax_invariance_and_can_change_are_always_opposites():
    from utils.method_metadata import method_semantics, registered_method_names

    exact, prefixes = registered_method_names()
    for name in exact + tuple(p + "x" for p in prefixes):
        a = method_semantics(name)
        assert a is not None, name
        assert a["argmax_invariant_by_construction"] != a["can_change_argmax"], name


def test_base_prediction_source_implies_cannot_change_argmax():
    from utils.method_metadata import method_semantics, registered_method_names

    exact, _ = registered_method_names()
    for name in exact:
        a = method_semantics(name)
        if a["effective_prediction_source"] == "base":
            assert a["can_change_argmax"] is False, name
            assert a["argmax_invariant_by_construction"] is True, name


def test_scalar_methods_document_their_surrogate_reconstruction():
    from utils.method_metadata import method_semantics, registered_method_names

    exact, _ = registered_method_names()
    for name in exact:
        a = method_semantics(name)
        if a["output_semantics"] == "scalar_confidence":
            assert a["surrogate_reconstruction"], name
            assert a["metric_bucket"] == "scalar_only", name


def test_unregistered_method_fails_loudly():
    import pytest as _pytest
    from utils.method_metadata import metric_bucket_for_method, require_method_semantics

    with _pytest.raises(ValueError, match="not in the canonical semantics registry"):
        require_method_semantics("totally_made_up_method")
    with _pytest.raises(ValueError, match="not in the canonical semantics registry"):
        metric_bucket_for_method("totally_made_up_method")
