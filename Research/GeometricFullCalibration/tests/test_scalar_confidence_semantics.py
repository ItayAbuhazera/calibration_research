"""
Adversarial tests for scalar-confidence semantics and selective-prediction metrics.

These are deliberately hostile: each one constructs the input that would make
the OLD (surrogate-argmax) evaluation produce a wrong answer, and asserts the
corrected semantics is unmoved.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Experiments.run_unified_benchmark import _make_method_entry  # noqa: E402
from utils.method_metadata import (  # noqa: E402
    method_semantics,
    metric_bucket_for_method,
    registered_method_names,
)
from utils.selective_metrics import (  # noqa: E402
    aurc,
    correctness_auroc,
    coverage_at_risk,
    risk_at_coverage,
    risk_coverage_curve,
)
from utils.unified_metrics import (  # noqa: E402
    confidence_adaptive_ece,
    confidence_ece,
    evaluate_scalar_confidence,
    scalar_confidence_from_surrogate,
)

SCALAR_BASE_METHODS = (
    "top_label_isotonic",
    "gc_dac",
    "rgcl",
    "mahalanobis_confidence",
    "trust_score_original_diagnostic",
)


def _adversarial_surrogate(n=200, c=5, seed=0):
    """Base probs plus a surrogate whose argmax is forced AWAY from the base
    prediction on most rows: the calibrated top confidence is driven near zero
    and the tail is rescaled, which is exactly the proportional-tail failure."""
    rng = np.random.default_rng(seed)
    base = rng.dirichlet(np.ones(c) * 0.4, n)
    labels = rng.integers(0, c, n)
    base_pred = np.argmax(base, axis=1)
    rows = np.arange(n)

    surrogate = base.copy()
    # Force a very low calibrated top confidence on 70% of rows.
    low = rng.random(n) < 0.7
    c_t = np.where(low, 1e-4, base[rows, base_pred])
    p_top_old = base[rows, base_pred]
    scale = (1.0 - c_t) / np.clip(1.0 - p_top_old, 1e-6, 1.0)
    surrogate *= scale[:, None]
    surrogate[rows, base_pred] = c_t
    surrogate /= surrogate.sum(axis=1, keepdims=True)
    return base, surrogate, labels, base_pred


# ---------------------------------------------------------------- K1
def test_scalar_surrogate_forced_to_flip_keeps_base_prediction():
    base, surrogate, labels, base_pred = _adversarial_surrogate()
    # The adversarial construction really does move the surrogate argmax.
    assert np.mean(np.argmax(surrogate, axis=1) != base_pred) > 0.5

    for name in SCALAR_BASE_METHODS:
        entry = _make_method_entry(
            name, surrogate, labels, "fam", False, "t", "validation", "o",
            base_probs=base,
        )
        assert entry["effective_prediction_source"] == "base"
        assert entry["effective_argmax_change_rate"] == 0.0
        assert entry["surrogate_only"] is True
        assert entry["surrogate_argmax_change_rate"] > 0.5
        assert entry["metrics"]["accuracy"] == pytest.approx(
            float(np.mean(base_pred == labels))
        ), f"{name}: effective accuracy must equal base accuracy"
        assert entry["can_change_argmax"] is False


# ---------------------------------------------------------------- K2
def test_tail_permutation_does_not_change_scalar_metrics():
    """Permuting the NON-base classes of the surrogate must be invisible."""
    base, surrogate, labels, base_pred = _adversarial_surrogate(seed=3)
    rng = np.random.default_rng(11)

    permuted = surrogate.copy()
    n, c = surrogate.shape
    for i in range(n):
        others = np.array([k for k in range(c) if k != base_pred[i]])
        permuted[i, others] = surrogate[i, rng.permutation(others)]

    for name in SCALAR_BASE_METHODS:
        a = _make_method_entry(name, surrogate, labels, "f", False, "t", "v", "o", base_probs=base)
        b = _make_method_entry(name, permuted, labels, "f", False, "t", "v", "o", base_probs=base)
        for key in ("accuracy", "top_label_ece", "adaptive_ece", "correctness_auroc",
                    "aurc", "excess_aurc"):
            assert a["metrics"][key] == pytest.approx(b["metrics"][key]), f"{name}/{key}"


# ---------------------------------------------------------------- K3
def test_scalar_metrics_equal_direct_calculation_from_confidence_and_correct():
    base, surrogate, labels, base_pred = _adversarial_surrogate(seed=7)
    confidence = surrogate[np.arange(len(base_pred)), base_pred]
    correct = (base_pred == labels).astype(np.float64)
    expected = evaluate_scalar_confidence(confidence, correct)

    entry = _make_method_entry(
        "gc_dac", surrogate, labels, "f", False, "t", "v", "o", base_probs=base
    )
    for key, value in expected.items():
        assert entry["metrics"][key] == pytest.approx(value) if isinstance(
            value, float
        ) else entry["metrics"][key] == value, key

    # And the extraction helper agrees with the hand calculation.
    scalar = scalar_confidence_from_surrogate(surrogate, base)
    np.testing.assert_array_equal(scalar["effective_pred"], base_pred)
    np.testing.assert_allclose(scalar["confidence"], confidence)


# ---------------------------------------------------------------- K4
def test_scalar_rows_have_null_nll_brier_and_classwise_ece():
    base, surrogate, labels, _ = _adversarial_surrogate(seed=5)
    for name in SCALAR_BASE_METHODS:
        entry = _make_method_entry(
            name, surrogate, labels, "f", False, "t", "v", "o", base_probs=base
        )
        assert entry["metric_bucket"] == "scalar_only"
        for forbidden in ("nll", "brier", "classwise_ece"):
            assert entry["metrics"][forbidden] is None, f"{name}.{forbidden}"
        # The anchor-c_t stratification must not smuggle an NLL back in.
        anchor = np.clip(surrogate.max(axis=1), 0, 1)
        e2 = _make_method_entry(
            name, surrogate, labels, "f", False, "t", "v", "o", base_probs=base,
            anchor_ct_for_stratification=anchor, base_probs_for_anchor_bins=base,
        )
        for b in e2["metrics_by_gc_dac_anchor_ct_bin"].values():
            assert b["nll"] is None
            assert b["argmax_change_rate_vs_base"] in (0.0, None)
            assert b["delta_nll_vs_base"] is None


def test_scalar_method_without_base_probs_fails_loudly():
    base, surrogate, labels, _ = _adversarial_surrogate(seed=9)
    with pytest.raises(ValueError, match="base_probs is required"):
        _make_method_entry("gc_dac", surrogate, labels, "f", False, "t", "v", "o")


# ---------------------------------------------------------------- K5
def test_native_dac_positive_temperature_preserves_argmax():
    """softmax(z / T(x)) with T(x) > 0 cannot reorder classes, for any T."""
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(500, 10)) * 3.0
    base_pred = np.argmax(logits, axis=1)
    for _ in range(50):
        T = rng.uniform(1e-6, 50.0, size=(500, 1))
        scaled = logits / T
        exp = np.exp(scaled - scaled.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)
        np.testing.assert_array_equal(np.argmax(probs, axis=1), base_pred)

    sem = method_semantics("native_dac")
    assert sem["argmax_invariant_by_construction"] is True
    assert sem["can_change_argmax"] is False


def test_runner_registers_native_dac_as_argmax_invariant():
    src = (Path(__file__).resolve().parents[1] / "Experiments" / "run_unified_benchmark.py").read_text()
    block = src[src.index('native_dac_entry = _make_method_entry('):]
    block = block[: block.index('_save_or_skip_method(') + 400]
    assert '"full_vector_density_calibration",\n                False,' in block
    assert '"native_dac",\n                native_dac_entry,\n                native_dac_probs_test,\n                False,' in block


# ---------------------------------------------------------------- K6
def test_registry_entry_and_bucket_are_consistent_everywhere():
    exact, _ = registered_method_names()
    base = np.array([[0.7, 0.2, 0.1], [0.2, 0.5, 0.3], [0.1, 0.3, 0.6], [0.5, 0.4, 0.1]])
    labels = np.array([0, 1, 2, 1])
    for name in exact:
        sem = method_semantics(name)
        entry = _make_method_entry(
            name, base, labels, "fam", sem["can_change_argmax"], "t", "v", "o",
            base_probs=base,
        )
        assert entry["metric_bucket"] == metric_bucket_for_method(name)
        assert entry["can_change_argmax"] == sem["can_change_argmax"]
        assert entry["output_semantics"] == sem["output_semantics"]
        assert entry["effective_prediction_source"] == sem["effective_prediction_source"]
        assert entry["structural_axes"]["can_change_argmax"] == sem["can_change_argmax"]
        assert entry["semantic_schema_version"] == "scalar_semantics_v2"
        # The bucket dictates which metrics may be non-null.
        if entry["metric_bucket"] == "scalar_only":
            assert entry["metrics"]["nll"] is None
        else:
            assert entry["metrics"]["nll"] is not None


def test_call_site_flag_disagreeing_with_registry_raises():
    base = np.array([[0.7, 0.3], [0.4, 0.6]])
    labels = np.array([0, 1])
    with pytest.raises(ValueError, match="canonical registry"):
        _make_method_entry(
            "temperature_scaling", base, labels, "f", True, "t", "v", "o", base_probs=base
        )


def test_argmax_map_consistency_gate_rejects_drift():
    from Experiments.run_unified_benchmark import _assert_argmax_map_matches_registry

    _assert_argmax_map_matches_registry({"temperature_scaling": False, "vector_scaling": True})
    with pytest.raises(RuntimeError, match="can_change_argmax disagrees"):
        _assert_argmax_map_matches_registry({"native_dac": True})
    # Unregistered methods warn but do not fail (external probability imports).
    _assert_argmax_map_matches_registry({"some_external_import": True})


# ---------------------------------------------------------------- K7
def test_glad_pi_arms_are_capacity_matched_except_geometry():
    import inspect

    import torch

    from Calibrators.glad_pi import GLADPICalibrator, _CorrectionNet

    geo = GLADPICalibrator(hidden_dim=16, epochs=2, beta_grid=[0.0, 0.1])
    zero = GLADPICalibrator(hidden_dim=16, epochs=2, beta_grid=[0.0, 0.1], zero_geometry=True)

    # Identical optimization settings; the ONLY constructor difference is the flag.
    for attr in ("hidden_dim", "lr", "weight_decay", "epochs", "patience",
                 "beta_grid", "nll_tolerance", "use_temperature", "margin"):
        assert getattr(geo, attr) == getattr(zero, attr), attr
    assert geo.zero_geometry is False and zero.zero_geometry is True

    # Identical architecture and trainable parameter count.
    torch.manual_seed(0)
    net_a = _CorrectionNet(hidden_dim=16)
    torch.manual_seed(0)
    net_b = _CorrectionNet(hidden_dim=16)
    count = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert count(net_a) == count(net_b)
    assert _CorrectionNet.INPUT_DIM == 5, "both arms feed the same 5-dim per-class input"

    # zero_geometry acts ONLY by zeroing the distance channel.
    src = inspect.getsource(GLADPICalibrator._to_tensors)
    assert "if self.zero_geometry:" in src
    assert "torch.zeros_like(dist_t)" in src
    # ...and nowhere else in the class.
    cls_src = inspect.getsource(GLADPICalibrator)
    behavioural = [
        line.strip() for line in cls_src.splitlines()
        if "self.zero_geometry" in line
        and "self.zero_geometry = " not in line
        and '"zero_geometry":' not in line  # get_params reporting only
    ]
    assert behavioural == ["if self.zero_geometry:"], (
        "zero_geometry must affect behaviour ONLY by zeroing the distance channel; "
        f"other uses would confound the ablation: {behavioural}"
    )


def test_glad_pi_zero_geometry_output_is_invariant_to_the_distance_input():
    """The regular-only arm must not leak geometry through any other path."""
    import numpy as _np

    from Calibrators.glad_pi import GLADPICalibrator

    rng = _np.random.default_rng(0)
    n, c = 80, 4
    logits = rng.normal(size=(n, c)).astype(_np.float64)
    labels = rng.integers(0, c, n)
    dist = rng.random((n, c))

    cal = GLADPICalibrator(hidden_dim=8, epochs=3, beta_grid=[0.0], zero_geometry=True)
    cal.fit(logits[:40], labels[:40], dist[:40], logits[40:], labels[40:], dist[40:])
    a = cal.calibrate(logits, dist)
    b = cal.calibrate(logits, rng.random((n, c)) * 100.0)
    _np.testing.assert_allclose(a, b, atol=1e-12)


# ---------------------------------------------------------------- K8
def test_selective_metrics_on_analytically_checkable_toys():
    # Perfect ranking: all correct ranked above all errors.
    conf = np.array([0.9, 0.8, 0.7, 0.6])
    corr = np.array([1.0, 1.0, 0.0, 0.0])
    assert correctness_auroc(conf, corr) == pytest.approx(1.0)
    # risk(k) = 0, 0, 1/3, 2/4 -> mean = (0+0+1/3+0.5)/4
    assert aurc(conf, corr) == pytest.approx((0 + 0 + 1 / 3 + 0.5) / 4)
    assert coverage_at_risk(conf, corr, 0.0) == pytest.approx(0.5)
    assert risk_at_coverage(conf, corr, 1.0) == pytest.approx(0.5)

    # Worst ranking: all errors first.
    corr_rev = np.array([0.0, 0.0, 1.0, 1.0])
    assert correctness_auroc(conf, corr_rev) == pytest.approx(0.0)
    assert coverage_at_risk(conf, corr_rev, 0.0) == pytest.approx(0.0)

    # All correct: AURC 0, AUROC undefined.
    ones = np.ones(5)
    assert aurc(conf[:5] if conf.size >= 5 else np.linspace(1, 0.1, 5), ones) == pytest.approx(0.0)
    assert correctness_auroc(np.linspace(1, 0.1, 5), ones) is None
    assert coverage_at_risk(np.linspace(1, 0.1, 5), ones, 0.0) == pytest.approx(1.0)

    # All incorrect: AURC 1, AUROC undefined, no coverage achieves risk 0.
    zeros = np.zeros(5)
    assert aurc(np.linspace(1, 0.1, 5), zeros) == pytest.approx(1.0)
    assert correctness_auroc(np.linspace(1, 0.1, 5), zeros) is None
    assert coverage_at_risk(np.linspace(1, 0.1, 5), zeros, 0.0) == pytest.approx(0.0)

    # Constant confidence: AUROC 0.5 exactly, AURC = error rate exactly.
    const = np.full(8, 0.42)
    corr8 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    assert correctness_auroc(const, corr8) == pytest.approx(0.5)
    assert aurc(const, corr8) == pytest.approx(0.25)


def test_selective_metrics_are_invariant_to_tie_ordering():
    rng = np.random.default_rng(4)
    conf = rng.choice([0.2, 0.5, 0.8], size=300)
    corr = (rng.random(300) < 0.7).astype(np.float64)
    ref = (aurc(conf, corr), correctness_auroc(conf, corr), coverage_at_risk(conf, corr, 0.1))
    for _ in range(10):
        p = rng.permutation(300)
        got = (
            aurc(conf[p], corr[p]),
            correctness_auroc(conf[p], corr[p]),
            coverage_at_risk(conf[p], corr[p], 0.1),
        )
        assert got == pytest.approx(ref)


def test_risk_coverage_curve_shape_and_monotone_coverage():
    rng = np.random.default_rng(2)
    conf = rng.random(50)
    corr = (rng.random(50) < 0.6).astype(np.float64)
    curve = risk_coverage_curve(conf, corr)
    assert curve["coverage"].shape == curve["risk"].shape == (50,)
    assert np.all(np.diff(curve["coverage"]) > 0)
    assert curve["coverage"][-1] == pytest.approx(1.0)
    assert curve["risk"][-1] == pytest.approx(1.0 - corr.mean())
    assert np.all((curve["risk"] >= 0.0) & (curve["risk"] <= 1.0))


def test_selective_metric_input_validation():
    with pytest.raises(ValueError, match="Empty"):
        aurc(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="must align"):
        aurc(np.array([0.1, 0.2]), np.array([1.0]))
    with pytest.raises(ValueError, match="boolean/0-1"):
        aurc(np.array([0.1, 0.2]), np.array([0.5, 1.0]))
    with pytest.raises(ValueError, match="NaN or Inf"):
        aurc(np.array([np.nan, 0.2]), np.array([1.0, 0.0]))
    with pytest.raises(ValueError, match="target_risk"):
        coverage_at_risk(np.array([0.1]), np.array([1.0]), 1.5)


def test_confidence_ece_matches_hand_computation():
    conf = np.array([0.05, 0.15, 0.95, 0.95])
    corr = np.array([0.0, 0.0, 1.0, 0.0])
    # 15 equal-width bins: 0.05 -> bin 0, 0.15 -> bin 2, both 0.95 -> bin 14.
    expected = (
        0.25 * abs(0.0 - 0.05) + 0.25 * abs(0.0 - 0.15) + 0.5 * abs(0.5 - 0.95)
    )
    assert confidence_ece(conf, corr) == pytest.approx(expected)
    # Equal-mass with 15 bins on 4 points: each point is its own bin. The two
    # samples at 0.95 form a tie block, so both carry the block's mean
    # correctness 0.5 (tie-averaging is what makes adaptive ECE invariant to
    # row order; see utils/unified_metrics.confidence_adaptive_ece).
    assert confidence_adaptive_ece(conf, corr) == pytest.approx(
        0.25 * (0.05 + 0.15 + 0.45 + 0.45)
    )


def test_adaptive_ece_is_invariant_to_row_order_under_heavy_ties():
    """Regression: equal-mass bins used to cut tied blocks arbitrarily."""
    rng = np.random.default_rng(0)
    conf = rng.choice([0.2, 0.5, 0.9], size=600)   # ~200-sample tie blocks
    corr = (rng.random(600) < 0.7).astype(np.float64)
    ref = confidence_adaptive_ece(conf, corr)
    for _ in range(25):
        p = rng.permutation(600)
        assert confidence_adaptive_ece(conf[p], corr[p]) == pytest.approx(ref)


# ---------------------------------------------------------------- F: fairness
def test_baseline_fairness_checks_pass_for_phase0_1():
    from Experiments.audit_baseline_fairness import PHASE0_1_METHODS, fairness_checks

    assert fairness_checks(PHASE0_1_METHODS) == []


def test_fairness_audit_detects_an_unfair_budget(monkeypatch):
    import Experiments.audit_baseline_fairness as audit
    from utils import method_metadata

    real = method_metadata.method_semantics

    def tampered(name):
        sem = real(name)
        if sem is not None and name == "gc_dac":
            sem = dict(sem)
            sem["information_budget"] = dict(sem["information_budget"])
            sem["information_budget"]["fitting_split"] = "corrupted_test"
        return sem

    monkeypatch.setattr(audit, "method_semantics", tampered)
    problems = audit.fairness_checks(["top_label_isotonic", "gc_dac"])
    assert any("corrupted_test" in p for p in problems)
    assert any("forbids fitting or tuning on the evaluation split" in p for p in problems)


def test_fairness_audit_flags_an_unmatched_glad_pi_ablation(monkeypatch):
    import Experiments.audit_baseline_fairness as audit
    from utils import method_metadata

    real = method_metadata.method_semantics

    def tampered(name):
        sem = real(name)
        if sem is not None and name == "glad_pi":
            sem = dict(sem)
            sem["information_budget"] = dict(sem["information_budget"])
            sem["information_budget"]["hp_search_budget"] = "10x bigger grid"
        return sem

    monkeypatch.setattr(audit, "method_semantics", tampered)
    problems = audit.fairness_checks(list(audit.GLAD_PI_ARMS))
    assert any("not matched on hp_search_budget" in p for p in problems)


# ---------------------------------------------------------------- J: rescoring
def test_rescoring_is_read_only_and_records_no_refit(tmp_path):
    import hashlib
    import json

    from Experiments.rescore_scalar_semantics import rescore_run, write_derived

    rng = np.random.default_rng(0)
    n, c = 120, 4
    base = rng.dirichlet(np.ones(c), n)
    labels = rng.integers(0, c, n)
    base_pred = np.argmax(base, axis=1)
    surrogate = base.copy()
    surrogate[np.arange(n), base_pred] = 1e-4
    surrogate /= surrogate.sum(axis=1, keepdims=True)

    run_dir = tmp_path / "run"
    (run_dir / "per_sample").mkdir(parents=True)
    npz = run_dir / "per_sample" / "per_sample_arrays.npz"
    np.savez(
        npz,
        labels_test=labels.astype(np.int64),
        base_probs_test=base,
        method_index=np.array(["gc_dac", "temperature_scaling"], dtype=np.str_),
        method_probs__gc_dac=surrogate,
        method_probs__temperature_scaling=base,
    )
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in run_dir.rglob("*") if p.is_file()}

    payload = rescore_run(str(run_dir))
    out = write_derived(payload, str(tmp_path / "derived_out"))

    # Source artifacts are byte-identical afterwards.
    after = {p: hashlib.sha256(p.read_bytes()).hexdigest()
             for p in run_dir.rglob("*") if p.is_file()}
    assert before == after

    written = json.loads(Path(out, "summary_metrics_rescored.json").read_text())
    assert written["provenance"]["refit_occurred"] is False
    assert written["provenance"]["source_artifact_sha256"] == before[npz]
    assert written["semantic_schema_version"] == "scalar_semantics_v2"

    gc = written["methods"]["gc_dac"]
    assert gc["metrics"]["nll"] is None
    assert gc["effective_argmax_change_rate"] == 0.0
    assert gc["surrogate_argmax_change_rate"] > 0.0
    assert gc["metrics"]["accuracy"] == pytest.approx(float(np.mean(base_pred == labels)))
    assert written["methods"]["temperature_scaling"]["metrics"]["nll"] is not None


def test_rescoring_refuses_to_clobber_an_existing_derived_dir(tmp_path):
    from Experiments.rescore_scalar_semantics import write_derived

    out = tmp_path / "derived_out"
    out.mkdir()
    with pytest.raises(FileExistsError, match="--force"):
        write_derived({"_per_sample": {}}, str(out))


def test_resume_refuses_to_mix_old_and_new_metric_semantics(tmp_path):
    from Experiments.run_unified_benchmark import _assert_uniform_semantic_schema

    fresh = [{"method_name": "gc_dac", "semantic_schema_version": "scalar_semantics_v2"}]
    _assert_uniform_semantic_schema(fresh, str(tmp_path))

    stale = fresh + [{"method_name": "top_label_isotonic"}]  # pre-v2 checkpoint
    with pytest.raises(RuntimeError, match="older metric semantics"):
        _assert_uniform_semantic_schema(stale, str(tmp_path))
    with pytest.raises(RuntimeError, match="rescore_scalar_semantics"):
        _assert_uniform_semantic_schema(stale, str(tmp_path))


# ------------------------------------------------- Stage 5: matched GLAD arms
def _glad_param_fingerprint(cal, logits, labels, dist):
    """Fingerprint of the trained parameters for a fully-specified fit."""
    import hashlib

    cal.fit(logits[:60], labels[:60], dist[:60], logits[60:], labels[60:], dist[60:])
    h = hashlib.sha256()
    for k, v in sorted(cal._net.state_dict().items()):
        h.update(k.encode()); h.update(np.ascontiguousarray(v.cpu().numpy()).tobytes())
    return h.hexdigest()


def test_seeded_glad_arms_start_from_identical_parameters():
    """The geometry and zero-geometry arms must begin from the same weights."""
    import torch

    from Calibrators.glad_pi import GLADPICalibrator, _CorrectionNet

    def init_state(seed):
        torch.manual_seed(seed)
        net = _CorrectionNet(hidden_dim=8)
        return {k: v.clone() for k, v in net.state_dict().items()}

    a, b = init_state(4242), init_state(4242)
    for k in a:
        assert torch.equal(a[k], b[k]), k

    geo = GLADPICalibrator(hidden_dim=8, epochs=2, beta_grid=[0.0, 0.1], init_seed=4242)
    zero = GLADPICalibrator(hidden_dim=8, epochs=2, beta_grid=[0.0, 0.1],
                            zero_geometry=True, init_seed=4242)
    assert geo.init_seed == zero.init_seed == 4242
    assert geo.get_params()["init_seed"] == 4242
    # Default stays unseeded so already-fitted states keep their semantics.
    assert GLADPICalibrator().init_seed is None


def test_seeded_glad_arms_differ_only_through_the_geometry_input():
    """Same seed + zero distances on BOTH arms => bit-identical fitted models."""
    from Calibrators.glad_pi import GLADPICalibrator

    rng = np.random.default_rng(0)
    n, c = 120, 4
    logits = rng.normal(size=(n, c))
    labels = rng.integers(0, c, n)
    zeros = np.zeros((n, c))
    real = rng.random((n, c))

    # Geometry arm fed all-zero distances == zero-geometry arm fed real ones.
    a = _glad_param_fingerprint(
        GLADPICalibrator(hidden_dim=8, epochs=5, beta_grid=[0.0], init_seed=7),
        logits, labels, zeros)
    b = _glad_param_fingerprint(
        GLADPICalibrator(hidden_dim=8, epochs=5, beta_grid=[0.0], zero_geometry=True, init_seed=7),
        logits, labels, real)
    assert a == b, "with geometry neutralized both ways, the arms must be identical"

    # Real geometry must then actually change the fit (the ablation has bite).
    cal = GLADPICalibrator(hidden_dim=8, epochs=5, beta_grid=[0.0], init_seed=7)
    assert _glad_param_fingerprint(cal, logits, labels, real) != a


def test_seeded_glad_fit_is_deterministic_across_repeats():
    from Calibrators.glad_pi import GLADPICalibrator

    rng = np.random.default_rng(1)
    n, c = 100, 3
    logits = rng.normal(size=(n, c)); labels = rng.integers(0, c, n); dist = rng.random((n, c))
    f = [_glad_param_fingerprint(
            GLADPICalibrator(hidden_dim=8, epochs=5, beta_grid=[0.0, 0.5], init_seed=99),
            logits, labels, dist) for _ in range(2)]
    assert f[0] == f[1]
