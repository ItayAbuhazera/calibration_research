"""
Canonical method-semantics registry for the Phase 0/1 benchmark.

This module is the SINGLE source of truth for every method's evaluation
semantics. Nothing else may define these values independently:

  * Experiments/run_unified_benchmark.py::_make_method_entry derives
    `can_change_argmax`, the metric bucket and the scalar/full-vector
    evaluation path from here (call sites pass no independent flag that is
    not asserted against this registry).
  * Experiments/studyAB_artifact_schema.py::metric_bucket_for_method
    delegates to `metric_bucket_for_method` below.
  * `method_can_change_argmax_map` / the per-sample NPZ metadata are built
    from here.

A method that is missing from this registry is a hard error at artifact-write
time -- the bucket is never inferred after the fact (BENCHMARK_IMPLEMENTATION_PLAN.md
§6, §9).

--------------------------------------------------------------------------
AXIS DEFINITIONS (exact meanings -- a reviewer must not have to guess)
--------------------------------------------------------------------------

output_semantics
    "scalar_confidence"
        The method's NATIVE object is a single confidence value attached to
        the BASE MODEL's predicted class. Any NxC matrix it produces is a
        reconstruction for storage/compatibility only (see
        `surrogate_reconstruction`) and must never define the method's
        prediction or confidence. Per §6 these methods get top-label ECE,
        adaptive ECE, correctness AUROC/AURC, risk-coverage and
        coverage-at-matched-risk; NLL, multiclass Brier and classwise ECE
        are a real `null`.
    "full_vector"
        The method's native object IS the NxC distribution. Its argmax and
        max are meaningful, and it additionally gets NLL / multiclass Brier /
        classwise calibration.

effective_prediction_source
    Which label the method is scored against.
    "base"            -- argmax(base_probs); the method only re-scores
                         confidence and never overrides the decision.
    "calibrated"      -- argmax of the method's own full-vector output.
    "explicit_switch" -- the method is a declared decision intervention that
                         may replace the base prediction by an explicit rule
                         (not as a side effect of a reconstruction).

argmax_invariant_by_construction
    True iff the method's EFFECTIVE prediction provably equals the base
    prediction for every possible input, by the structure of the transform --
    not because it empirically happened not to flip on one dataset.
    Two distinct reasons produce True:
      (a) scalar_confidence + effective_prediction_source="base": the
          prediction is the base prediction by definition; and
      (b) a full-vector transform that is order-preserving by a proof, e.g.
          dividing all logits by a strictly positive per-sample scalar
          (temperature_scaling, parameterized_temperature_scaling,
          native_dac's softmax(z / S(x,w)) with S > 0).

can_change_argmax
    Whether the EFFECTIVE prediction can differ from the base prediction.
    This is NOT derived from effective_prediction_source alone: a
    "calibrated" full-vector method may still be argmax-invariant by
    construction (temperature scaling, native DAC), in which case it is
    False. It is the negation of `argmax_invariant_by_construction`, and both
    are stored explicitly so a disagreement is a loud test failure rather
    than a silent derivation.

sample_dependent
    The TRANSFORM applied to a sample varies with that sample -- i.e. the
    method is not one fixed global map applied identically to every sample's
    output vector. Examples: parameterized_temperature_scaling (T(x)) and
    native_dac (S(x,w)) are True; a single global temperature, a global
    class-affine map, and a single global monotone map of the top
    probability (top_label_isotonic) are False.
    NOTE: this axis says nothing about WHERE the per-sample information comes
    from. Geometry is a separate axis below, because "sample_dependent" alone
    was previously being read as "uses representation geometry", which it is
    not.

representation_geometry_dependent
    The method reads the classifier's INTERNAL representation space (hidden
    features, distances, densities, neighbourhoods) rather than only the
    output-space logits/probabilities. This is the axis the benchmark's
    scientific question turns on: it separates "geometric" from
    "ordinary output-signal" methods.

class_dependent
    The output has per-class components that are not a single shared scalar
    applied to every class.

joint_sample_class_dependent
    The correction is a joint function of (sample, class) -- i.e. it does not
    factor into a per-sample scalar times a per-class term.

metric_bucket
    "scalar_only" or "full_vector", the frozen §6 table.

surrogate_reconstruction
    For scalar_confidence methods only: how the stored NxC matrix is built
    from the scalar. Documented because the reconstruction is NOT the
    scientific object and, for the proportional-tail variants, its argmax can
    differ from the base prediction even though the method's effective
    prediction cannot. See `surrogate_argmax_change_rate` in the artifacts.

information_budget
    Fairness bookkeeping used by Experiments/audit_baseline_fairness.py, so
    an unequal comparison (more labels, more tuning, bigger head) is visible
    at a glance rather than buried in the runner.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

SCALAR_CONFIDENCE = "scalar_confidence"
FULL_VECTOR = "full_vector"

SCALAR_ONLY_BUCKET = "scalar_only"
FULL_VECTOR_BUCKET = "full_vector"

PRED_BASE = "base"
PRED_CALIBRATED = "calibrated"
PRED_EXPLICIT_SWITCH = "explicit_switch"

# Schema version for the semantics recorded in artifacts. Bump whenever the
# meaning of an emitted field changes so old and new rows can never be
# silently mixed.
SEMANTIC_SCHEMA_VERSION = "scalar_semantics_v2"

# Surrogate reconstruction conventions (scalar methods only).
PROPORTIONAL_TAIL = (
    "proportional_tail: q[t]=c_t, q[j!=t] = (1-c_t) * p_j / (1-p_t). Can place a "
    "rescaled rival above c_t, so the SURROGATE argmax may differ from the base "
    "prediction; the effective prediction is unaffected."
)
UNIFORM_TAIL = (
    "uniform_tail: q[t]=c_t, q[j!=t] = (1-c_t)/(C-1). Surrogate argmax differs from "
    "the base prediction only if c_t < 1/C."
)
NO_RECONSTRUCTION = "none: the stored matrix is the base model's own probability vector."


def _budget(
    *,
    logits: bool,
    softmax: bool,
    representation: bool,
    labels_in_fitting: bool,
    fitting_split: str,
    selection_split: Optional[str],
    objective: str,
    trainable_params: str,
    hp_search_budget: str,
) -> Dict[str, Any]:
    return {
        "uses_logits": logits,
        "uses_softmax": softmax,
        "uses_internal_representation": representation,
        "uses_labels_in_fitting": labels_in_fitting,
        "fitting_split": fitting_split,
        "selection_split": selection_split,
        "objective": objective,
        "trainable_params": trainable_params,
        "hp_search_budget": hp_search_budget,
    }


def _m(
    *,
    output_semantics: str,
    effective_prediction_source: str,
    argmax_invariant: bool,
    can_change_argmax: bool,
    metric_bucket: str,
    sample_dep: bool,
    geometry_dep: bool,
    class_dep: bool,
    joint: bool,
    notes: str,
    surrogate_reconstruction: Optional[str] = None,
    information_budget: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if argmax_invariant == can_change_argmax:
        raise ValueError(
            "argmax_invariant_by_construction and can_change_argmax must be "
            f"opposites; got {argmax_invariant!r} / {can_change_argmax!r} for: {notes[:60]}"
        )
    if output_semantics == SCALAR_CONFIDENCE and surrogate_reconstruction is None:
        raise ValueError(
            "scalar_confidence methods must document surrogate_reconstruction"
        )
    if effective_prediction_source == PRED_BASE and can_change_argmax:
        raise ValueError(
            "effective_prediction_source='base' cannot also change the argmax"
        )
    return {
        "output_semantics": output_semantics,
        "effective_prediction_source": effective_prediction_source,
        "argmax_invariant_by_construction": argmax_invariant,
        "can_change_argmax": can_change_argmax,
        "metric_bucket": metric_bucket,
        "sample_dependent": sample_dep,
        "representation_geometry_dependent": geometry_dep,
        "class_dependent": class_dep,
        "joint_sample_class_dependent": joint,
        "surrogate_reconstruction": surrogate_reconstruction,
        "information_budget": information_budget,
        "notes": notes,
    }


# Shared information budgets. The fitting split is the frozen §7
# calibration/validation split for every post-hoc method; "inner_fit" /
# "inner_select" are the deterministic sub-split of it used by GLAD-PI.
_POSTHOC_OUTPUT_BUDGET = dict(
    logits=True, softmax=True, representation=False, labels_in_fitting=True,
    fitting_split="calibration_validation", selection_split=None,
)
_POSTHOC_GEOMETRY_BUDGET = dict(
    logits=True, softmax=True, representation=True, labels_in_fitting=True,
    fitting_split="calibration_validation", selection_split=None,
)


_EXACT: Dict[str, Dict[str, Any]] = {
    # ---------------------------------------------------------------- base
    "base_model": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=False, geometry_dep=False, class_dep=True, joint=False,
        notes="Uncalibrated reference. Argmax-invariant relative to itself by definition.",
        information_budget=_budget(
            logits=True, softmax=True, representation=False, labels_in_fitting=False,
            fitting_split="none", selection_split=None,
            objective="none (reference)", trainable_params="0", hp_search_budget="0",
        ),
    ),
    # -------------------------------------------- output-space calibrators
    "temperature_scaling": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=False, geometry_dep=False, class_dep=False, joint=False,
        notes="softmax(z/T), one global scalar T>0. Argmax-invariant by the "
              "scalar-temperature lemma (dividing all logits by T>0 preserves order).",
        information_budget=_budget(
            **_POSTHOC_OUTPUT_BUDGET,
            objective="validation NLL", trainable_params="1", hp_search_budget="1D line search",
        ),
    ),
    "parameterized_temperature_scaling": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=False, class_dep=False, joint=False,
        notes="softmax(z/T(x)) with a sample-adaptive strictly positive scalar T(x) "
              "predicted from the sorted logits. Sample-dependent but class-independent, "
              "so argmax-invariant by the same lemma as temperature_scaling.",
        information_budget=_budget(
            **_POSTHOC_OUTPUT_BUDGET,
            objective="validation NLL", trainable_params="small MLP over sorted logits",
            hp_search_budget="fixed architecture, no grid",
        ),
    ),
    "vector_scaling": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=False, geometry_dep=False, class_dep=True, joint=False,
        notes="Global per-class affine rescaling of logits. Class-dependent, not "
              "sample-dependent; reorders classes.",
        information_budget=_budget(
            **_POSTHOC_OUTPUT_BUDGET,
            objective="validation NLL", trainable_params="2C", hp_search_budget="0",
        ),
    ),
    "beta_calibration": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=False, geometry_dep=False, class_dep=True, joint=False,
        notes="One-vs-rest beta calibration; per-class transform, can reorder classes.",
        information_budget=_budget(
            **_POSTHOC_OUTPUT_BUDGET,
            objective="per-class likelihood", trainable_params="3C", hp_search_budget="0",
        ),
    ),
    "ovr_isotonic": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=False, geometry_dep=False, class_dep=True, joint=False,
        notes="One-vs-rest isotonic regression per class, then renormalization.",
        information_budget=_budget(
            **_POSTHOC_OUTPUT_BUDGET,
            objective="per-class squared error vs one-hot",
            trainable_params="C isotonic step functions", hp_search_budget="0",
        ),
    ),
    "odir_dirichlet": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=False, geometry_dep=False, class_dep=True, joint=False,
        notes="ODIR Dirichlet calibration: global class-dependent linear map on log-probs.",
        information_budget=_budget(
            **_POSTHOC_OUTPUT_BUDGET,
            objective="validation NLL + ODIR regularizer",
            trainable_params="C^2 + C", hp_search_budget="regularization grid",
        ),
    ),
    "top_label_isotonic": _m(
        output_semantics=SCALAR_CONFIDENCE,
        effective_prediction_source=PRED_BASE,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=SCALAR_ONLY_BUCKET,
        sample_dep=False, geometry_dep=False, class_dep=False, joint=False,
        notes="THE primary non-geometric scalar-confidence baseline: one global monotone "
              "(PAV isotonic) map of the base model's max softmax probability onto binary "
              "correctness. Not sample-dependent (a single global map) and not "
              "geometry-dependent -- it sees only MSP.",
        surrogate_reconstruction=PROPORTIONAL_TAIL,
        information_budget=_budget(
            **_POSTHOC_OUTPUT_BUDGET,
            objective="isotonic PAV squared error vs binary correctness",
            trainable_params="1 isotonic step function", hp_search_budget="0",
        ),
    ),
    # ------------------------------------------------- geometric (scalar)
    "gc_dac": _m(
        output_semantics=SCALAR_CONFIDENCE,
        effective_prediction_source=PRED_BASE,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=SCALAR_ONLY_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=False, joint=False,
        notes="Geometric separation statistic computed on DAC's target layers, mapped by "
              "isotonic regression onto binary correctness and assigned to the base "
              "predicted class. Sample-dependent AND geometry-dependent: the statistic is "
              "per-sample representation-space geometry. NOT native DAC (different mapper, "
              "different metric bucket).",
        surrogate_reconstruction=UNIFORM_TAIL,
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="isotonic PAV squared error vs binary correctness",
            trainable_params="1 isotonic step function", hp_search_budget="0",
        ),
    ),
    "rgcl": _m(
        output_semantics=SCALAR_CONFIDENCE,
        effective_prediction_source=PRED_BASE,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=SCALAR_ONLY_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=False, joint=False,
        notes="Published Random Geometric Calibration: per-sample stability/separation "
              "statistic over randomly drawn internal layers, isotonic-mapped to the base "
              "predicted class. Sample- and geometry-dependent.",
        surrogate_reconstruction=UNIFORM_TAIL,
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="isotonic PAV squared error vs binary correctness",
            trainable_params="1 isotonic step function",
            hp_search_budget="0 (layer draw is a fixed seeded plan, not tuned)",
        ),
    ),
    "rgcc": _m(
        output_semantics=SCALAR_CONFIDENCE,
        effective_prediction_source=PRED_BASE,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=SCALAR_ONLY_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=False, joint=False,
        notes="Top-label geometric coordinate calibration. Same scalar family as rgcl. "
              "EXTENSION BEYOND the frozen §6 table (which names Top-label Isotonic, GC-DAC "
              "and published/corrected RGCL explicitly); bucketed scalar_only by the same "
              "reasoning. Disabled in Phase 0/1.",
        surrogate_reconstruction=UNIFORM_TAIL,
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="isotonic PAV squared error vs binary correctness",
            trainable_params="1 isotonic step function", hp_search_budget="0",
        ),
    ),
    "gc_tulip": _m(
        output_semantics=SCALAR_CONFIDENCE,
        effective_prediction_source=PRED_BASE,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=SCALAR_ONLY_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=False, joint=False,
        notes="Top-label GC-TULIP calibration; per-sample representation geometry through "
              "the same isotonic top-label mapper. EXTENSION BEYOND the frozen §6 table. "
              "Disabled in Phase 0/1.",
        surrogate_reconstruction=UNIFORM_TAIL,
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="isotonic PAV squared error vs binary correctness",
            trainable_params="1 isotonic step function", hp_search_budget="0",
        ),
    ),
    "mahalanobis_confidence": _m(
        output_semantics=SCALAR_CONFIDENCE,
        effective_prediction_source=PRED_BASE,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=SCALAR_ONLY_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=False, joint=False,
        notes="Variant A: per-class means + Ledoit-Wolf shrunk shared covariance on "
              "penultimate features; scalar score = -min_c Mahalanobis distance, isotonic-"
              "mapped onto the base predicted class. Sample- and geometry-dependent, "
              "class-independent (one scalar per sample).",
        surrogate_reconstruction=PROPORTIONAL_TAIL,
        information_budget=_budget(
            logits=False, softmax=True, representation=True, labels_in_fitting=True,
            fitting_split="calibration_validation (isotonic) + train (class means/covariance)",
            selection_split=None,
            objective="isotonic PAV squared error vs binary correctness",
            trainable_params="1 isotonic step function + C means + 1 shared covariance "
                             "(closed-form, not gradient-trained)",
            hp_search_budget="0 (Ledoit-Wolf shrinkage is estimated, not tuned)",
        ),
    ),
    # ----------------------------------------------------- trust score
    "trust_score_original_diagnostic": _m(
        output_semantics=SCALAR_CONFIDENCE,
        effective_prediction_source=PRED_BASE,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=SCALAR_ONLY_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=False, joint=False,
        notes="Trust Score computed and reported; probabilities are returned unchanged. "
              "Diagnostic only -- it never alters the prediction. NOTE: the stored matrix "
              "is the base model's own vector, so its scalar confidence equals base MSP, "
              "NOT the trust score itself (the trust score is recorded in the entry's "
              "extra fields as mean/median/fraction_below_1).",
        surrogate_reconstruction=NO_RECONSTRUCTION,
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="none (alpha=0 nearest-neighbour ratio, no fit)",
            trainable_params="0", hp_search_budget="0",
        ),
    ),
    "trust_score_original_switch": _m(
        output_semantics=SCALAR_CONFIDENCE,
        effective_prediction_source=PRED_EXPLICIT_SWITCH,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=SCALAR_ONLY_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=False, joint=False,
        notes="Declared decision intervention: when the trust ratio falls below a "
              "validation-tuned threshold the prediction is switched to the nearest other "
              "class. The switch is an explicit rule, not a reconstruction side effect, so "
              "its effective prediction is its OWN argmax and it earns argmax-change "
              "metrics (frozen §6 grants these explicitly for the switch arm).",
        surrogate_reconstruction=(
            "explicit_switch: the stored matrix encodes the switched decision; its argmax "
            "IS the effective prediction."
        ),
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="validation accuracy over a threshold sweep",
            trainable_params="1 threshold", hp_search_budget="1D threshold sweep",
        ),
    ),
    # ------------------------------------------------ full-vector methods
    "native_dac": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=False, joint=False,
        notes="Native Density-Aware Calibration: softmax(z / S(x,w)) where S(x,w) is a "
              "per-sample density temperature from layer-wise kNN scores, clipped to "
              ">= 1e-12 and therefore strictly positive. ARGMAX-INVARIANT BY CONSTRUCTION "
              "by the scalar-temperature lemma -- the runner previously recorded "
              "can_change_argmax=True, which was wrong (seed 4 clean observed 0 flips). "
              "Full-vector bucket per §6 because softmax(z/S) is a genuine distribution.",
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="DAC paper squared-error objective",
            trainable_params="L+1 (per-layer weights + bias)",
            hp_search_budget="0 (k fixed at the paper default)",
        ),
    ),
    "anchored_model_tail": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=True,
        notes="Anchors the base top class to GC-DAC's c_t and renormalizes the model's own "
              "tail. A deliberate full-vector construction (not a scalar surrogate): the "
              "tail shape is the scientific object, so it is evaluated on its own argmax.",
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="inherits gc_dac's isotonic fit (no separate objective)",
            trainable_params="0 beyond gc_dac", hp_search_budget="0",
        ),
    ),
    "anchored_rankgeom_tail_mixture": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=True,
        notes="Anchored rank-geometric tail mixture; jointly sample x class dependent "
              "non-top redistribution driven by the distance matrix.",
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="validation NLL over a (lambda, alpha) grid",
            trainable_params="2 (lambda, alpha)", hp_search_budget="3 x 6 = 18 grid points",
        ),
    ),
    "full_vector_distance_fusion": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=True,
        notes="softmax(log p + beta * S(x)) full-vector distance fusion.",
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="validation NLL over a beta grid",
            trainable_params="1 (beta)", hp_search_budget="1D beta grid",
        ),
    ),
    "post_fusion_topiso": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=True,
        notes="Top-label isotonic repair applied on top of the fused full-vector output. "
              "Full-vector (not scalar) because its parent's tail is the object being "
              "repaired and its prediction is the fused/repaired argmax.",
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="isotonic PAV on fused top probability vs correctness",
            trainable_params="1 isotonic step function beyond the fusion",
            hp_search_budget="0 beyond the fusion's beta grid",
        ),
    ),
    "kcal": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=True,
        notes="Full KCal: learned penultimate projection + RBF-kernel KDE posterior over a "
              "reference bank. Jointly sample x class dependent.",
        information_budget=_budget(
            logits=False, softmax=True, representation=True, labels_in_fitting=True,
            fitting_split="calibration_validation (+ train reference bank)",
            selection_split="cross-validated bandwidth folds",
            objective="KDE posterior likelihood + learned projection",
            trainable_params="projection matrix (penultimate_dim x proj_dim)",
            hp_search_budget="bandwidth CV folds",
        ),
    ),
    "glad_pi": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=True,
        notes="GEOMETRY ARM of the matched ablation. Shared per-class MLP over "
              "[logit_k, distance_k, max_prob, entropy, top_margin] producing a centered "
              "additive logit correction. Its ONLY input difference from "
              "glad_pi_zero_geometry is the distance_k channel.",
        information_budget=_budget(
            logits=True, softmax=True, representation=True, labels_in_fitting=True,
            fitting_split="inner_fit (deterministic sub-split of calibration_validation)",
            selection_split="inner_select (same deterministic split, same seed)",
            objective="NLL + beta * asymmetric margin loss on base-wrong samples",
            trainable_params="shared per-class MLP: 5*H + H + H + 1, plus correction_weight",
            hp_search_budget="len(beta_grid) trained models",
        ),
    ),
    "glad_pi_zero_geometry": _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=False, class_dep=True, joint=True,
        notes="REGULAR-ONLY ARM of the matched ablation: identical architecture, parameter "
              "count, optimizer, epochs, splits, objective and beta grid, with distance_k "
              "forced to exactly 0. Its remaining inputs are ordinary output signals only "
              "(per-class logits + max_prob, entropy, top_margin), so "
              "representation_geometry_dependent=False. NOTE: because the context features "
              "are softmax-derived they are shift-invariant, so this arm cannot represent "
              "energy/logsumexp or logit-norm; see docs/baseline_fairness.md.",
        information_budget=_budget(
            logits=True, softmax=True, representation=False, labels_in_fitting=True,
            fitting_split="inner_fit (deterministic sub-split of calibration_validation)",
            selection_split="inner_select (same deterministic split, same seed)",
            objective="NLL + beta * asymmetric margin loss on base-wrong samples",
            trainable_params="shared per-class MLP: 5*H + H + H + 1, plus correction_weight "
                             "(identical count to glad_pi; the distance input is a dead channel)",
            hp_search_budget="len(beta_grid) trained models",
        ),
    ),
}


# Prefix fallbacks for method families with per-variant name suffixes.
_PREFIXES: List[Tuple[str, Dict[str, Any]]] = [
    ("studyB_cell_", _m(
        output_semantics=SCALAR_CONFIDENCE,
        effective_prediction_source=PRED_BASE,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=SCALAR_ONLY_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=False, joint=False,
        notes="Study-B common rank+isotonic scalar mapper (cells A/B/C/D). Scalar "
              "confidence assigned to the base predicted class, per the frozen §5 mapper "
              "definition. Cell D is NOT native DAC.",
        surrogate_reconstruction=PROPORTIONAL_TAIL,
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="isotonic PAV squared error vs binary correctness",
            trainable_params="1 isotonic step function (zero learned aggregation weights)",
            hp_search_budget="0 (L=5 fixed, k at DAC default)",
        ),
    )),
    ("corrected_rgcl_studyA", _m(
        output_semantics=SCALAR_CONFIDENCE,
        effective_prediction_source=PRED_BASE,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=SCALAR_ONLY_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=False, joint=False,
        notes="Corrected RGCL (Study A): fc-excluded layer draws, same scalar top-label "
              "mapper as published RGCL.",
        surrogate_reconstruction=UNIFORM_TAIL,
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="isotonic PAV squared error vs binary correctness",
            trainable_params="1 isotonic step function", hp_search_budget="0",
        ),
    )),
    ("published_rgcl", _m(
        output_semantics=SCALAR_CONFIDENCE,
        effective_prediction_source=PRED_BASE,
        argmax_invariant=True, can_change_argmax=False,
        metric_bucket=SCALAR_ONLY_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=False, joint=False,
        notes="Published RGCL row, kept as separate provenance from corrected RGCL.",
        surrogate_reconstruction=UNIFORM_TAIL,
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="isotonic PAV squared error vs binary correctness",
            trainable_params="1 isotonic step function", hp_search_budget="0",
        ),
    )),
    ("trust_score", _EXACT["trust_score_original_diagnostic"]),
    ("full_vector_geometric_fusion", _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=True,
        notes="Full-vector distance-geometric fusion family.",
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="validation NLL over a lambda grid",
            trainable_params="1 (lambda)", hp_search_budget="1D lambda grid",
        ),
    )),
    ("full_vector_distance_fusion", _EXACT["full_vector_distance_fusion"]),
    ("softmax_knn_blend", _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=True,
        notes="Softmax-kNN blend; jointly sample x class dependent via per-class kNN distances.",
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="validation NLL over an alpha grid",
            trainable_params="1 (alpha)", hp_search_budget="1D alpha grid",
        ),
    )),
    ("kcal_lite", _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=True,
        notes="KCal-lite top-k KDE baseline.",
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="KDE posterior likelihood",
            trainable_params="0 learned projection", hp_search_budget="bandwidth grid",
        ),
    )),
    ("kcal_factorial", _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=True,
        notes="Controlled KCal factorial variants.",
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="KDE posterior likelihood, optional blend",
            trainable_params="varies by cell", hp_search_budget="factorial cell grid",
        ),
    )),
    ("kcal", _EXACT["kcal"]),
    ("rgcl_neighbor_correction", _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_EXPLICIT_SWITCH,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=True,
        notes="RGCL nearest-neighbour argmax switch: an explicit hard decision correction.",
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="validation accuracy over a threshold sweep",
            trainable_params="1 threshold", hp_search_budget="1D threshold sweep",
        ),
    )),
    ("rgcl_tail", _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=True,
        notes="RGCL tail hybrid: explicit full-vector tail construction from RGCL-derived "
              "probabilities.",
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="inherits rgcl's isotonic fit",
            trainable_params="0 beyond rgcl", hp_search_budget="0",
        ),
    )),
    ("anchored_rankgeom", _EXACT["anchored_rankgeom_tail_mixture"]),
    ("contrastive_beta", _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=True,
        notes="Contrastive per-class nearest-neighbour distance correction on scaled logits.",
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="validation NLL under a gate",
            trainable_params="1 (beta)", hp_search_budget="1D beta grid",
        ),
    )),
    ("aar_lightweight", _m(
        output_semantics=FULL_VECTOR,
        effective_prediction_source=PRED_CALIBRATED,
        argmax_invariant=False, can_change_argmax=True,
        metric_bucket=FULL_VECTOR_BUCKET,
        sample_dep=True, geometry_dep=True, class_dep=True, joint=False,
        notes="Lightweight taxonomic fallback (not official AAR). Sample-dependent scalar "
              "temperature x class additive correction; joint=False because the temperature "
              "is sample-only and the additive bias is class-only.",
        information_budget=_budget(
            **_POSTHOC_GEOMETRY_BUDGET,
            objective="validation NLL",
            trainable_params="C + 1", hp_search_budget="small grid",
        ),
    )),
    ("uncalibrated", _EXACT["base_model"]),
]


def method_semantics(method_name: str) -> Optional[Dict[str, Any]]:
    """Full canonical semantics for a method, or None if unregistered."""
    if method_name in _EXACT:
        return dict(_EXACT[method_name])
    for prefix, sem in _PREFIXES:
        if method_name.startswith(prefix):
            out = dict(sem)
            out["notes"] = f"[prefix match: {prefix}] " + str(out["notes"])
            return out
    return None


def require_method_semantics(method_name: str) -> Dict[str, Any]:
    """Like `method_semantics` but raises for an unregistered method.

    Artifact writers use this so an unregistered method fails loudly rather
    than silently losing its bucket/semantics (§6, §9).
    """
    sem = method_semantics(method_name)
    if sem is None:
        raise ValueError(
            f"Method '{method_name}' is not in the canonical semantics registry "
            "(utils/method_metadata.py). Add it -- output semantics, effective "
            "prediction source and metric bucket are never inferred after the fact."
        )
    return sem


def metric_bucket_for_method(method_name: str) -> str:
    """Frozen §6 metric bucket, derived from the canonical registry."""
    return str(require_method_semantics(method_name)["metric_bucket"])


def is_scalar_confidence_method(method_name: str) -> bool:
    return require_method_semantics(method_name)["output_semantics"] == SCALAR_CONFIDENCE


def effective_prediction_source(method_name: str) -> str:
    return str(require_method_semantics(method_name)["effective_prediction_source"])


def can_change_argmax(method_name: str) -> bool:
    return bool(require_method_semantics(method_name)["can_change_argmax"])


# Structural-axis subset, kept for the existing artifact field name.
_STRUCTURAL_KEYS = (
    "sample_dependent",
    "representation_geometry_dependent",
    "class_dependent",
    "joint_sample_class_dependent",
    "argmax_invariant_by_construction",
    "can_change_argmax",
    "notes",
)


def structural_axes_for_method(method_name: str) -> Optional[Dict[str, Any]]:
    """Structural axes for a registered method, or None if unknown.

    Retained for the `structural_axes` artifact field; the authoritative
    record is `method_semantics`.
    """
    sem = method_semantics(method_name)
    if sem is None:
        return None
    return {k: sem[k] for k in _STRUCTURAL_KEYS}


def registered_method_names() -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """(exact names, prefix patterns) -- used by the fairness audit and tests."""
    return tuple(sorted(_EXACT)), tuple(p for p, _ in _PREFIXES)
