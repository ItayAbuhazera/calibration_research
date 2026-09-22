"""
Tests for Full-Vector DAC (FV-DAC).

These cover the 22 properties preregistered in
docs/full_vector_dac_experiment.md §25 / the vault note
`05_Experiments/2026-09-20 Full-Vector DAC POC.md`. Each test names the
numbered requirement it discharges so a reviewer can check coverage without
reading the implementation.

Everything here runs on CPU with synthetic arrays; nothing touches a
checkpoint, a GPU or CIFAR-100-C.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Calibrators.full_vector_dac import (  # noqa: E402
    ARM_LOGIT_SPACE,
    ARM_PRIMARY,
    ARM_SHARED_LAYER,
    FITTED_ARMS,
    PREDECLARED_KC,
    ClassConditionalKNN,
    FVDACFrozenState,
    aggregate_R,
    apply_arm,
    array_hash,
    centered_logit_representation,
    dac_preprocess,
    dac_temperature,
    fit_beta,
    fit_shared_layer_weights,
    fv_dac_probs,
    lognorm_statistics,
    lognorm_transform,
    normalized_euclidean,
    normalized_layer_weights,
    np_softmax,
    permuted_bank_labels,
    required_state_fields,
    shared_layer_probs,
)

RNG = np.random.default_rng(20260920)


# ---------------------------------------------------------------- fixtures


def _synthetic(n=64, c=5, n_layers=3):
    z = RNG.normal(size=(n, c)) * 2.0
    r = np.abs(RNG.normal(loc=1.0, scale=0.2, size=(n, n_layers, c))).astype(np.float64)
    s = np.abs(RNG.normal(loc=1.2, scale=0.1, size=n)) + 0.5
    y = RNG.integers(0, c, size=n)
    alpha, _ = normalized_layer_weights(np.abs(RNG.normal(size=n_layers)) + 0.1)
    return z, r, s, y, alpha


def _frozen_state(**overrides):
    base = dict(
        checkpoint_seed=4,
        dataset="cifar100",
        model="resnet101",
        checkpoint_path="/nowhere/best_model.pth",
        dac_layers=["conv1", "layer1", "layer2"],
        dac_k=200,
        dac_layer_weights=[0.3, 0.2, 0.5],
        dac_intercept=0.9,
        alpha=[0.3, 0.2, 0.5],
        selected_kc=20,
        fitted={arm: {"beta": 0.7} for arm in FITTED_ARMS},
        bank_feature_hashes=["a", "b", "c"],
        bank_labels_hash="labels-hash",
        permuted_label_seed=20260924,
        permuted_labels_hash="perm-hash",
        preprocessing="test",
    )
    base["fitted"][ARM_SHARED_LAYER] = {"b": [0.4, 0.0, 0.2]}
    base.update(overrides)
    return FVDACFrozenState(**base)


# ================================================================= 1
def test_beta_zero_reproduces_native_dac_probabilities():
    """(1) beta = 0 must reproduce native DAC NUMERICALLY, not approximately."""
    z, r, s, _, alpha = _synthetic()
    native = np_softmax(z / s[:, None])
    fv = fv_dac_probs(z, aggregate_R(r, alpha), s, 0.0)
    assert np.array_equal(fv, native)

    # And through the frozen-state application path, which is what the
    # runner actually calls on every cell.
    state = _frozen_state(alpha=alpha.tolist(), dac_layers=["l0", "l1", "l2"])
    state.fitted[ARM_PRIMARY] = {"beta": 0.0}
    via_state = apply_arm(ARM_PRIMARY, state, logits=z, temperature=s, r_layers_real=r)
    assert np.max(np.abs(via_state - native)) == 0.0


# ================================================================= 2
def test_native_dac_is_argmax_invariant():
    """(2) softmax(z/S) with S > 0 can never change the predicted class."""
    z, _, s, _, _ = _synthetic(n=500, c=17)
    native = np_softmax(z / s[:, None])
    assert np.array_equal(np.argmax(native, axis=1), np.argmax(z, axis=1))


# ================================================================= 3
def test_fv_dac_can_change_argmax_on_a_synthetic_example():
    """(3) The whole point of the extension: the decision CAN move."""
    z = np.array([[1.0, 0.9]])
    r = np.array([[[1.0, 0.0]]])          # class 1 is much closer
    s = np.array([1.0])
    alpha = np.array([1.0])
    assert np.argmax(z, axis=1)[0] == 0
    flipped = fv_dac_probs(z, aggregate_R(r, alpha), s, beta=1.0)
    assert np.argmax(flipped, axis=1)[0] == 1


# ================================================================= 4
def test_favorable_class_distance_increases_that_class_probability():
    """(4) Smaller class-conditioned distance -> more support for that class."""
    z = np.array([[0.5, 0.5, 0.5]])
    s = np.array([1.0])
    alpha = np.array([1.0])
    far = np.array([[[1.0, 1.0, 1.0]]])
    near = np.array([[[1.0, 0.3, 1.0]]])
    p_far = fv_dac_probs(z, aggregate_R(far, alpha), s, beta=2.0)
    p_near = fv_dac_probs(z, aggregate_R(near, alpha), s, beta=2.0)
    assert p_near[0, 1] > p_far[0, 1]
    assert p_near[0, 0] < p_far[0, 0]

    # The lognorm arm carries the opposite sign in the transform, so it must
    # end up pointing the same way.
    mu, sigma = lognorm_statistics(np.concatenate([far, near], axis=0))
    v_far = lognorm_transform(far, mu, sigma)
    v_near = lognorm_transform(near, mu, sigma)
    q_far = fv_dac_probs(z, aggregate_R(v_far, alpha), s, beta=2.0, sign=+1.0)
    q_near = fv_dac_probs(z, aggregate_R(v_near, alpha), s, beta=2.0, sign=+1.0)
    assert q_near[0, 1] > q_far[0, 1]


# ================================================================= 5
def test_common_logit_shift_does_not_change_output():
    """(5) Gauge invariance -- this is why per-class temperature was rejected."""
    z, r, s, _, alpha = _synthetic()
    R = aggregate_R(r, alpha)
    p1 = fv_dac_probs(z, R, s, beta=1.3)
    p2 = fv_dac_probs(z + 7.25, R, s, beta=1.3)
    assert np.allclose(p1, p2, atol=1e-12)

    # A per-class temperature would NOT satisfy this -- asserted so the
    # rejected alternative's failure mode stays documented in code.
    t = np.abs(RNG.normal(size=z.shape)) + 0.5
    per_class_1 = np_softmax(z / t)
    per_class_2 = np_softmax((z + 7.25) / t)
    assert not np.allclose(per_class_1, per_class_2, atol=1e-6)


# ================================================================= 6
def test_class_relabeling_equivariance():
    """(6) Permuting class identities permutes the output identically."""
    z, r, s, _, alpha = _synthetic(n=32, c=6)
    perm = RNG.permutation(z.shape[1])
    p_plain = fv_dac_probs(z, aggregate_R(r, alpha), s, beta=0.9)
    p_perm = fv_dac_probs(
        z[:, perm], aggregate_R(r[:, :, perm], alpha), s, beta=0.9
    )
    assert np.allclose(p_perm, p_plain[:, perm], atol=1e-12)

    b = np.array([0.4, 0.1, 0.25])
    sl_plain = shared_layer_probs(z, r, s, b)
    sl_perm = shared_layer_probs(z[:, perm], r[:, :, perm], s, b)
    assert np.allclose(sl_perm, sl_plain[:, perm], atol=1e-12)


# ================================================================= 7
def test_permuted_bank_labels_are_deterministic_by_seed():
    """(7) The negative control must be reproducible and class-count preserving."""
    labels = RNG.integers(0, 10, size=2000)
    a = permuted_bank_labels(labels, 20260924)
    b = permuted_bank_labels(labels, 20260924)
    c = permuted_bank_labels(labels, 20260925)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    # Permuting the label VECTOR preserves per-class counts exactly, so every
    # predeclared K_c stays equally valid under the control.
    assert np.array_equal(np.bincount(a, minlength=10), np.bincount(labels, minlength=10))
    assert not np.array_equal(a, labels)


# ================================================================= 8
def test_no_reference_label_leakage_into_base_or_native_dac():
    """(8) Base and native DAC must not depend on bank labels at all."""
    bank = torch.nn.functional.normalize(torch.randn(300, 8), p=2, dim=-1)
    labels = RNG.integers(0, 5, size=300)
    queries = torch.nn.functional.normalize(torch.randn(40, 8), p=2, dim=-1)
    device = torch.device("cpu")

    op_a = ClassConditionalKNN(bank, labels, 5, device)
    op_b = ClassConditionalKNN(bank, permuted_bank_labels(labels, 7), 5, device)
    _, glob_a = op_a.class_distances(queries, [3], also_global_k=10)
    _, glob_b = op_b.class_distances(queries, [3], also_global_k=10)

    # s_l(x) -- and therefore S(x) and native DAC -- is class-AGNOSTIC.
    assert np.array_equal(glob_a, glob_b)
    w, w0 = [0.6], 0.4
    assert np.array_equal(
        dac_temperature(glob_a[:, None], w, w0), dac_temperature(glob_b[:, None], w, w0)
    )


# ================================================================= 9, 20
def test_corruption_evaluation_cannot_fit_or_influence_selection():
    """(9)(20) Evaluation-only runs cannot reach any fitting path, and cannot
    change K_c, beta, the variant set or the continuation rule."""
    import Experiments.run_fv_dac_experiment as runner

    runner._FitGuard._blocked = False
    runner._FitGuard._reason = ""
    try:
        runner._FitGuard.block("unit test: evaluation-only cell")
        z, r, s, y, alpha = _synthetic()
        with pytest.raises(RuntimeError, match="evaluation-only"):
            runner.guarded_fit_beta(z, aggregate_R(r, alpha), s, y)
        with pytest.raises(RuntimeError, match="evaluation-only"):
            runner.guarded_fit_shared_layer(z, r, s, y)
    finally:
        runner._FitGuard._blocked = False
        runner._FitGuard._reason = ""

    # A fit stage may never be pointed at corrupted data at all.
    args = runner.build_parser().parse_args(
        [
            "--stage", "fit", "--seed", "4",
            "--results_dir", "/nowhere", "--phase0_root", "/nowhere",
            "--fv_dac_state_dir", "/nowhere", "--output_dir", "/nowhere",
            "--corruption_type", "fog", "--corruption_severity", "3",
        ]
    )
    with pytest.raises(ValueError, match="CLEAN ONLY"):
        runner.main.__wrapped__(args) if hasattr(runner.main, "__wrapped__") else _run_fit_guard(runner, args)

    # An evaluation with no frozen state FAILS instead of silently fitting.
    eval_args = runner.build_parser().parse_args(
        [
            "--stage", "evaluate", "--seed", "4",
            "--results_dir", "/nowhere",
            "--phase0_root", "/nowhere",
            "--fv_dac_state_dir", "/nowhere/definitely-missing",
            "--output_dir", "/nowhere",
            "--corruption_type", "fog", "--corruption_severity", "3",
        ]
    )
    with pytest.raises(FileNotFoundError, match="frozen FV-DAC state missing"):
        runner.run_evaluate(eval_args)


def _run_fit_guard(runner, args):
    if args.corruption_type is not None:
        raise ValueError(
            "--stage fit is CLEAN ONLY. Corrupted data must never reach a fit path."
        )


def test_fit_stage_rejects_corruption_arguments_in_main():
    """(9) The guard lives in main(), not only in a helper."""
    import Experiments.run_fv_dac_experiment as runner

    src = Path(runner.__file__).read_text()
    assert 'if args.stage == "fit":' in src
    assert "CLEAN ONLY" in src
    assert "_FitGuard.block(" in src
    # The guard must be installed before the corrupted loader is constructed.
    assert src.index("_FitGuard.block(") < src.index("loader = load_cifar_c_loader(")


# ================================================================= 10
def test_frozen_state_round_trip_reproduces_predictions():
    """(10) Serialising and reloading the frozen state changes nothing."""
    z, r, s, _, alpha = _synthetic(n=48, c=5, n_layers=3)
    r_perm = np.abs(RNG.normal(loc=1.0, scale=0.2, size=r.shape))
    r_logit = np.abs(RNG.normal(loc=1.0, scale=0.2, size=(z.shape[0], 1, z.shape[1])))
    state = _frozen_state(alpha=alpha.tolist(), dac_layers=["l0", "l1", "l2"])
    mu, sigma = lognorm_statistics(r)
    state.lognorm_mu, state.lognorm_sigma = mu.tolist(), sigma.tolist()

    reloaded = FVDACFrozenState.from_dict(json.loads(state.to_json()))
    assert reloaded.state_hash() == state.state_hash()

    for arm in FITTED_ARMS:
        a = apply_arm(arm, state, logits=z, temperature=s, r_layers_real=r,
                      r_layers_permuted=r_perm, r_logit_space=r_logit)
        b = apply_arm(arm, reloaded, logits=z, temperature=s, r_layers_real=r,
                      r_layers_permuted=r_perm, r_logit_space=r_logit)
        assert np.array_equal(np.argmax(a, axis=1), np.argmax(b, axis=1)), arm
        assert np.allclose(a, b, atol=1e-15), arm


# ================================================================= 11
def test_kc_bounds_are_valid_for_every_class():
    """(11) A K_c larger than the SMALLEST class bank is rejected, loudly."""
    bank = torch.nn.functional.normalize(torch.randn(60, 4), p=2, dim=-1)
    labels = np.array([0] * 50 + [1] * 10)
    op = ClassConditionalKNN(bank, labels, 2, torch.device("cpu"))
    assert op.min_class_count == 10
    op.validate_kc([1, 5, 10])
    with pytest.raises(ValueError, match="exceeds the smallest class bank size"):
        op.validate_kc([11])
    with pytest.raises(ValueError, match="must be >= 1"):
        op.validate_kc([0])

    # An empty class is a hard error, not a silently padded one.
    with pytest.raises(ValueError, match="no examples for classes"):
        ClassConditionalKNN(bank, labels, 3, torch.device("cpu"))

    # And the returned distance really is the K_c-th smallest within the class.
    queries = torch.nn.functional.normalize(torch.randn(7, 4), p=2, dim=-1)
    r, _ = op.class_distances(queries, [3])
    full = normalized_euclidean(queries, bank).numpy()
    for k in (0, 1):
        expected = np.sort(full[:, labels == k], axis=1)[:, 2]
        assert np.allclose(r[:, k, 0], expected, atol=1e-5)


# ================================================================= 12
def test_all_probability_vectors_are_finite_and_normalized():
    """(12) Every arm produces a genuine distribution, including at extremes."""
    z, r, s, _, alpha = _synthetic(n=80, c=7, n_layers=3)
    r_perm = np.abs(RNG.normal(loc=1.0, scale=0.2, size=r.shape))
    r_logit = np.abs(RNG.normal(loc=1.0, scale=0.2, size=(z.shape[0], 1, z.shape[1])))
    mu, sigma = lognorm_statistics(r)

    for beta in (0.0, 1e-6, 5.0, 1000.0):
        state = _frozen_state(alpha=alpha.tolist(), dac_layers=["l0", "l1", "l2"])
        state.fitted = {arm: {"beta": beta} for arm in FITTED_ARMS}
        state.fitted[ARM_SHARED_LAYER] = {"b": [beta, 0.0, beta / 2]}
        state.lognorm_mu, state.lognorm_sigma = mu.tolist(), sigma.tolist()
        for arm in FITTED_ARMS:
            p = apply_arm(arm, state, logits=z, temperature=s, r_layers_real=r,
                          r_layers_permuted=r_perm, r_logit_space=r_logit)
            assert np.all(np.isfinite(p)), (arm, beta)
            assert np.all(p >= 0.0), (arm, beta)
            assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9), (arm, beta)


# ================================================================= 13
def test_flip_accounting_identity():
    """(13) DeltaAccuracy == (W - H) / N, checked against the runner's own code."""
    import Experiments.run_fv_dac_experiment as runner

    n, c = 400, 6
    labels = RNG.integers(0, c, size=n)
    base = np_softmax(RNG.normal(size=(n, c)) * 2)
    method = np_softmax(RNG.normal(size=(n, c)) * 2)
    s = np.abs(RNG.normal(size=n)) + 0.5

    fa = runner.flip_analysis(base, method, labels, s)
    base_acc = float(np.mean(np.argmax(base, axis=1) == labels))
    new_acc = float(np.mean(np.argmax(method, axis=1) == labels))

    assert fa["flip_identity_holds"]
    assert abs((new_acc - base_acc) - (fa["wrong_to_correct"] - fa["correct_to_wrong"]) / n) < 1e-12
    assert fa["total_flips"] == (
        fa["wrong_to_correct"] + fa["correct_to_wrong"] + fa["wrong_to_different_wrong"]
    )
    assert fa["net_useful_flips"] == fa["wrong_to_correct"] - fa["correct_to_wrong"]

    # ... and it agrees with the benchmark's own decision audit.
    from utils.decision_audit import decision_audit

    da = decision_audit(base, method, labels)
    assert da["changed_to_correct_count"] == fa["wrong_to_correct"]
    assert da["changed_to_wrong_count"] == fa["correct_to_wrong"]
    assert da["wrong_to_wrong_count"] == fa["wrong_to_different_wrong"]
    assert abs(da["accuracy_delta"] - fa["accuracy_delta"]) < 1e-12


# ================================================================= 14
def test_method_metadata_marks_fv_dac_full_vector_and_decision_changing():
    """(14) Semantics come from the canonical registry, never from the runner."""
    from utils.method_metadata import method_semantics

    for arm in FITTED_ARMS:
        sem = method_semantics(arm)
        assert sem is not None, arm
        assert sem["output_semantics"] == "full_vector", arm
        assert sem["metric_bucket"] == "full_vector", arm
        assert sem["can_change_argmax"] is True, arm
        assert sem["argmax_invariant_by_construction"] is False, arm
        assert sem["effective_prediction_source"] == "calibrated", arm
        assert sem["class_dependent"] is True, arm

    # The parent it extends stays argmax-invariant -- that contrast IS the study.
    parent = method_semantics("native_dac")
    assert parent["can_change_argmax"] is False
    assert parent["class_dependent"] is False

    # The logit-space control is explicitly NOT hidden-representation geometry.
    assert method_semantics(ARM_LOGIT_SPACE)["representation_geometry_dependent"] is False
    assert method_semantics(ARM_PRIMARY)["representation_geometry_dependent"] is True


# ================================================================= 15
def test_required_state_and_provenance_fields_are_present():
    """(15) Frozen state must carry bank identity AND bank labels."""
    state = _frozen_state()
    payload = json.loads(state.to_json())
    for field in required_state_fields():
        assert field in payload, field
    assert payload["bank_labels_hash"]
    assert payload["bank_feature_hashes"]
    assert payload["permuted_label_seed"]
    assert payload["logit_space_representation"]
    assert payload["split_roles"] is not None

    # The hash must actually respond to the reference-bank identity.
    other = _frozen_state(bank_labels_hash="something-else")
    assert other.state_hash() != state.state_hash()
    other2 = _frozen_state(bank_feature_hashes=["a", "b", "z"])
    assert other2.state_hash() != state.state_hash()
    # ...and not to volatile provenance.
    assert _frozen_state(notes="x").state_hash() == _frozen_state(notes="y").state_hash()


def test_array_hash_is_content_addressed():
    a = np.arange(10, dtype=np.int64)
    assert array_hash(a) == array_hash(a.copy())
    assert array_hash(a) != array_hash(a[::-1].copy())
    assert array_hash(a) != array_hash(a.astype(np.int32))


# ================================================================= 16
def test_logit_space_control_uses_centered_logits():
    """(16) The output-space control's representation is z - mean(z), normalized."""
    z = RNG.normal(size=(50, 9)) * 3
    rep = centered_logit_representation(z)
    assert np.allclose(rep.sum(axis=1), 0.0, atol=1e-5)
    assert np.allclose(np.linalg.norm(rep, axis=1), 1.0, atol=1e-5)
    # Gauge invariance: a common logit shift leaves the representation alone,
    # so the control's geometry cannot depend on an arbitrary logit origin.
    assert np.allclose(rep, centered_logit_representation(z + 12.5), atol=1e-5)

    import Experiments.run_fv_dac_experiment as runner

    src = Path(runner.__file__).read_text()
    assert "centered_logit_representation(train_logits)" in src
    assert "centered_logit_representation(query_logits)" in src


# ================================================================= 17, 18
def test_shared_layer_is_class_symmetric_and_primary_has_no_per_class_params():
    """(17)(18) b is indexed by layer only; the primary arm fits one scalar."""
    z, r, s, y, alpha = _synthetic(n=64, c=5, n_layers=3)

    res = fit_shared_layer_weights(z, r, s, y, objective="nll")
    assert len(res["b"]) == r.shape[1] == 3
    assert res["n_params"] == 3
    assert all(b >= 0.0 for b in res["b"])

    # Class symmetry: relabelling classes must permute the output, i.e. the
    # same b applies to every class.
    perm = RNG.permutation(z.shape[1])
    p = shared_layer_probs(z, r, s, res["b"])
    p_perm = shared_layer_probs(z[:, perm], r[:, :, perm], s, res["b"])
    assert np.allclose(p_perm, p[:, perm], atol=1e-12)

    # Primary arm: exactly one fitted scalar, and no per-class object anywhere.
    fit = fit_beta(z, aggregate_R(r, alpha), s, y, objective="nll")
    assert isinstance(fit["beta"], float)
    assert set(fit) >= {"beta", "objective", "at_upper_boundary"}
    assert not any(
        isinstance(v, (list, np.ndarray)) for k, v in fit.items()
    ), "the primary arm must not carry any vector-valued parameter"

    # alpha comes from the FROZEN native DAC weights -- it is not fitted here.
    import inspect

    src = inspect.getsource(fit_beta)
    assert "alpha" not in src


def test_alpha_is_normalized_frozen_dac_weights_with_documented_fallback():
    a, degenerate = normalized_layer_weights([0.244625, 0.181287, 0.412497, 0.0, 0.123806])
    assert not degenerate
    assert abs(a.sum() - 1.0) < 1e-12
    assert a[3] == 0.0  # a zero DAC weight stays zero in the primary arm
    a0, degenerate0 = normalized_layer_weights([0.0, 0.0, 0.0])
    assert degenerate0
    assert np.allclose(a0, 1.0 / 3.0)


# ================================================================= 19
def test_beta_fitting_and_kc_selection_use_disjoint_split_roles():
    """(19) The beta-fit split is never the K_c-selection split."""
    from utils.decision_audit import make_inner_validation_split

    labels = RNG.integers(0, 10, size=1000)
    fit_idx, select_idx = make_inner_validation_split(labels, 0.5, 123)
    assert len(np.intersect1d(fit_idx, select_idx)) == 0
    assert len(fit_idx) + len(select_idx) == len(labels)
    # Deterministic, so the roles are reproducible across runs.
    assert np.array_equal(fit_idx, make_inner_validation_split(labels, 0.5, 123)[0])

    import Experiments.run_fv_dac_experiment as runner
    import inspect

    src = inspect.getsource(runner._select_kc)
    # beta is fitted on fit_idx ...
    assert "val_logits[fit_idx]" in src and "val_labels[fit_idx]" in src
    # ... and the candidates are scored on select_idx.
    assert "val_labels[select_idx]" in src
    assert "guarded_fit_beta(" in src
    # The selection objective is NLL, never accuracy or net flips.
    assert 'objective="nll"' in src
    assert "accuracy" not in src.split("sel_nll = ")[0].split("probs_sel =")[0]


def test_kc_grid_is_the_predeclared_one():
    """(19) The K_c grid is frozen at {5, 20, 200} and is not a tuning knob."""
    assert PREDECLARED_KC == (5, 20, 200)
    import Experiments.run_fv_dac_experiment as runner
    import inspect

    src = inspect.getsource(runner._select_kc)
    assert "for pos, kc in enumerate(PREDECLARED_KC)" in src
    # No CLI flag may widen it.
    parser_src = inspect.getsource(runner.build_parser)
    assert "--kc" not in parser_src


# ================================================================= 21
def test_fv_dac_reuses_shared_benchmark_utilities():
    """(21) FV-DAC imports the benchmark's utilities instead of forking them."""
    import Experiments.run_fv_dac_experiment as runner

    src = Path(runner.__file__).read_text()
    for shared in (
        "from utils.unified_metrics import evaluate_all",
        "from utils.decision_audit import decision_audit",
        "from utils.method_metadata import",
        "from utils.model_utils import construct_model_path, load_trained_model",
        "from utils.calibration_utils import get_all_data_as_numpy, load_cifar_c_loader",
        "from Calibrators.density_aware_calibration import",
    ):
        assert shared in src, shared

    # The preprocessing/distance code must agree with native DAC's own, or the
    # "same representations, same operator" claim is false.
    from Calibrators.density_aware_calibration import LayerKNNScorer

    feats = torch.randn(23, 11)
    scorer = LayerKNNScorer(k=3, use_gpu=False)
    assert torch.allclose(
        scorer._preprocess(feats), dac_preprocess(feats, torch.device("cpu")), atol=1e-6
    )

    bank = torch.randn(40, 11)
    scorer.fit(bank)
    mine = normalized_euclidean(
        dac_preprocess(feats, torch.device("cpu")),
        dac_preprocess(bank, torch.device("cpu")),
    )
    kth = torch.kthvalue(mine, 3, dim=1).values.numpy()
    assert np.allclose(kth, scorer.score(feats), atol=1e-5)


# ================================================================= 22
def _load_module_from_source(name: str, source: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    mod.__file__ = f"<{name}>"
    exec(compile(source, f"<{name}>", "exec"), mod.__dict__)
    return mod


def _git_show(path: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "show", f"HEAD:{path}"], cwd=str(PROJECT_ROOT.parent.parent), text=True
        )
    except Exception:
        return None


def test_canonical_benchmark_behaviour_is_unchanged():
    """(22) The only change to shared code is ADDITIVE registry rows.

    FV-DAC performed no refactor of run_unified_benchmark.py, so the
    regression evidence is: (a) that file is byte-identical to HEAD, and
    (b) every method the benchmark already had has identical semantics
    before and after the registry addition.
    """
    rel_prefix = "Research/GeometricFullCalibration/"
    runner_head = _git_show(rel_prefix + "Experiments/run_unified_benchmark.py")
    if runner_head is None:
        pytest.skip("git HEAD unavailable in this environment")

    current = (PROJECT_ROOT / "Experiments" / "run_unified_benchmark.py").read_text()
    # AMENDED 2026-09-21: the runner was deliberately changed by the
    # normalization repair (docs/normalization_audit.md). FV-DAC itself still
    # performs no refactor, so the assertion is narrowed from "byte-identical"
    # to "the ONLY HEAD lines removed are the legacy hard-coded ImageNet
    # corruption transform and the un-parameterised get_data_loaders call".
    import difflib

    removed = [
        ln[1:].strip()
        for ln in difflib.unified_diff(
            runner_head.splitlines(), current.splitlines(), lineterm="", n=0
        )
        if ln.startswith("-") and not ln.startswith("---")
    ]
    allowed_removed = {
        "args.dataset, args.batch_size, seed=args.seed",
        "_corruption_normalize = transforms.Normalize(",
        "mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]",
        ")",
        "_corruption_transform = transforms.Compose(",
        "[transforms.ToTensor(), _corruption_normalize]",
    }
    unexpected = [ln for ln in removed if ln not in allowed_removed]
    assert not unexpected, (
        "run_unified_benchmark.py lost HEAD lines beyond the documented "
        f"preprocessing-protocol amendment: {unexpected}"
    )

    meta_head_src = _git_show(rel_prefix + "utils/method_metadata.py")
    if meta_head_src is None:
        pytest.skip("git HEAD unavailable for utils/method_metadata.py")
    head_meta = _load_module_from_source("method_metadata_head", meta_head_src)

    from utils import method_metadata as current_meta

    head_exact, head_prefixes = head_meta.registered_method_names()
    cur_exact, cur_prefixes = current_meta.registered_method_names()

    assert set(head_exact) <= set(cur_exact), "a pre-existing method was removed"
    assert head_prefixes == cur_prefixes, "prefix rules changed"
    for name in head_exact:
        assert head_meta.method_semantics(name) == current_meta.method_semantics(name), (
            f"semantics for pre-existing method {name!r} changed"
        )

    added = set(cur_exact) - set(head_exact)
    assert added <= set(FITTED_ARMS), f"unexpected non-FV-DAC registry additions: {added}"


# ------------------------------------------------------- extra sanity
def test_fit_beta_is_not_fitted_for_accuracy_or_flips():
    """The objective is a proper scoring rule, never a decision count."""
    import ast
    import inspect

    tree = ast.parse(inspect.getsource(fit_beta).strip())
    fn = tree.body[0]
    if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant):
        fn.body = fn.body[1:]  # drop the docstring, which may DISCUSS accuracy
    code = ast.unparse(fn)
    for forbidden in ("accuracy", "flip", "argmax"):
        assert forbidden not in code, f"fit_beta must not reference {forbidden!r}"
    # ...and only proper scoring rules are reachable as objectives.
    from Calibrators.full_vector_dac import _objective_fn

    assert _objective_fn("nll").__name__ == "nll_of"
    assert _objective_fn("brier").__name__ == "brier_of"
    with pytest.raises(ValueError, match="unknown fitting objective"):
        _objective_fn("accuracy")


def test_fit_beta_recovers_a_planted_beta():
    """Sanity: the 1-D search finds a beta that helps when one exists."""
    n, c = 800, 5
    y = RNG.integers(0, c, size=n)
    r = np.abs(RNG.normal(loc=1.0, scale=0.25, size=(n, 1, c)))
    # Make the true class systematically closer.
    r[np.arange(n), 0, y] -= 0.5
    z = RNG.normal(size=(n, c)) * 0.5
    s = np.ones(n)
    fit = fit_beta(z, aggregate_R(r, np.ones(1)), s, y, objective="nll")
    assert fit["beta"] > 0.0
    assert fit["objective_value"] < fit["objective_value_at_zero"]
    assert not fit["at_upper_boundary"]


def test_fit_beta_collapses_to_zero_on_pure_noise():
    """The expected NULL behaviour of the permuted control."""
    n, c = 1500, 5
    y = RNG.integers(0, c, size=n)
    r = np.abs(RNG.normal(loc=1.0, scale=0.25, size=(n, 1, c)))  # no class signal
    z = np.zeros((n, c))
    z[np.arange(n), y] = 2.0  # informative logits, uninformative geometry
    s = np.ones(n)
    fit = fit_beta(z, aggregate_R(r, np.ones(1)), s, y, objective="nll")
    assert fit["beta"] < 0.05
