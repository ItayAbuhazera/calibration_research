"""
Structural metadata registry for calibration methods.

Provides `structural_axes_for_method` which returns the 2x2 structural-argument
axes for any registered method. The axes encode the key properties needed for
the decision-improvement paper's structural classification:

  sample_dependent            — output depends on per-sample geometry/features
  class_dependent             — output has per-class components (not scalar)
  joint_sample_class_dependent — correction is a joint function of (sample, class)
  argmax_invariant_by_construction — cannot change the predicted class by design
  can_change_argmax           — is structurally capable of changing the predicted class

These are structural properties of the method, not empirical observations.
"""

from __future__ import annotations

from typing import Dict, Optional


def _axes(
    sample_dep: bool,
    class_dep: bool,
    joint: bool,
    argmax_inv: bool,
    can_change: bool,
    notes: str = "",
) -> Dict[str, object]:
    return {
        "sample_dependent": sample_dep,
        "class_dependent": class_dep,
        "joint_sample_class_dependent": joint,
        "argmax_invariant_by_construction": argmax_inv,
        "can_change_argmax": can_change,
        "notes": notes,
    }


# Registry of exact method names.
_EXACT: Dict[str, Dict[str, object]] = {
    "base_model": _axes(
        True, True, True, True, False,
        "Reference; argmax-invariant relative to itself by definition.",
    ),
    "temperature_scaling": _axes(
        False, False, False, True, False,
        "Global scalar temperature. Argmax-invariant by the scalar-temperature lemma.",
    ),
    "parameterized_temperature_scaling": _axes(
        True, False, False, True, False,
        "Sample-adaptive positive scalar temperature. Argmax-invariant by the scalar-temperature lemma: "
        "dividing all logits by T(x) > 0 preserves relative order.",
    ),
    "vector_scaling": _axes(
        False, True, False, False, True,
        "Global per-class scaling. Class-dependent but not sample-dependent.",
    ),
    "beta_calibration": _axes(
        False, True, False, False, True,
        "OvR beta calibration. Per-class transform, can reorder classes.",
    ),
    "ovr_isotonic": _axes(
        False, True, False, False, True,
        "OvR isotonic regression per class.",
    ),
    "odir_dirichlet": _axes(
        False, True, False, False, True,
        "ODIR Dirichlet calibration. Global class-dependent linear transform on log-probs.",
    ),
    "rgcl": _axes(
        False, False, False, True, False,
        "Top-label RGCL calibration. Rescales max-probability only; argmax-invariant.",
    ),
    "rgcc": _axes(
        False, False, False, True, False,
        "Top-label coordinate calibration. Argmax-invariant.",
    ),
    "gc_dac": _axes(
        False, False, False, True, False,
        "Top-label GC-DAC anchor calibration. Argmax-invariant.",
    ),
    "gc_tulip": _axes(
        False, False, False, True, False,
        "Top-label GC-TULIP calibration. Argmax-invariant.",
    ),
    "anchored_model_tail": _axes(
        False, True, False, False, True,
        "Anchored tail redistribution using GC-DAC top-label confidence. Can reorder non-top classes.",
    ),
    "trust_score_original_diagnostic": _axes(
        True, False, False, True, False,
        "Trust Score computed but predictions unchanged. Diagnostic only.",
    ),
    "trust_score_original_switch": _axes(
        True, True, False, False, True,
        "Trust Score threshold-based argmax switch to nearest other class.",
    ),
    "aar_lightweight": _axes(
        True, True, False, False, True,
        "Lightweight taxonomic fallback. Atypicality via per-sample mean 1-NN distance (not official AAR). "
        "Sample-dependent scalar temperature × class additive correction; joint=False because the "
        "temperature is sample-only and the additive bias is class-only (not jointly a function of both).",
    ),
    "glad_pi": _axes(
        True, True, True, False, True,
        "Geometric Logit Additive Decision-calibrator with Permutation Invariance. "
        "Jointly sample×class-dependent correction via shared per-class MLP with "
        "permutation-invariant global context.",
    ),
}

# Prefix mapping: method names starting with these prefixes inherit the listed axes,
# unless a more specific exact match exists.
_PREFIXES: list[tuple[str, Dict[str, object]]] = [
    ("full_vector_geometric_fusion", _axes(
        True, True, True, False, True,
        "Full-vector distance-geometric fusion. Jointly sample×class-dependent.",
    )),
    ("full_vector_distance_fusion", _axes(
        True, True, True, False, True,
        "Full-vector distance fusion.",
    )),
    ("softmax_knn_blend", _axes(
        True, True, True, False, True,
        "Softmax-kNN blend. Jointly sample×class-dependent via per-class kNN distances.",
    )),
    ("kcal_lite", _axes(
        True, True, True, False, True,
        "KCal-lite top-k KDE baseline. Jointly sample×class-dependent.",
    )),
    ("kcal_factorial", _axes(
        True, True, True, False, True,
        "Controlled KCal factorial variant. Per-sample, per-class kernel posterior, "
        "optionally blended with base probabilities.",
    )),
    ("kcal", _axes(
        True, True, True, False, True,
        "Full KCal with learned projection and RBF bandwidth. Jointly sample×class-dependent.",
    )),
    ("rgcl_neighbor_correction", _axes(
        True, True, True, False, True,
        "RGCL nearest-neighbor argmax switch. Jointly sample×class-dependent hard correction.",
    )),
    ("rgcl_tail", _axes(
        False, True, False, True, False,
        "RGCL tail hybrid. Top-label argmax-invariant, tail from RGCL-derived probabilities.",
    )),
    ("anchored_rankgeom", _axes(
        True, True, True, False, True,
        "Anchored rank-geometric tail mixture. Jointly sample×class-dependent non-top redistribution.",
    )),
    ("contrastive_beta", _axes(
        True, True, True, False, True,
        "Contrastive per-class nearest-neighbor distance correction added to scaled logits.",
    )),
]


def structural_axes_for_method(method_name: str) -> Optional[Dict[str, object]]:
    """
    Return the structural axes for a registered method name, or None if unknown.

    Exact names are checked first; prefix matches are used as fallback.
    """
    if method_name in _EXACT:
        return dict(_EXACT[method_name])

    for prefix, axes in _PREFIXES:
        if method_name.startswith(prefix):
            axes_copy = dict(axes)
            axes_copy["notes"] = f"[prefix match: {prefix}] " + axes_copy["notes"]
            return axes_copy

    return None
