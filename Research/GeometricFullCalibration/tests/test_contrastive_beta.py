import json
from pathlib import Path

import numpy as np

from Experiments.run_unified_benchmark import (
    CONTRASTIVE_BETA_METHOD_NAMES,
    CONTRASTIVE_SELECTION_CRITERION,
    ContrastiveBetaSelector,
    DEFAULT_ENABLED_METHOD_FLAGS,
    _apply_contrastive_correction,
    _contrastive_corrected_logits,
    _contrastive_distance_signal,
    _contrastive_gate_signature,
    _is_method_checkpoint_complete,
    _make_method_entry,
    _save_method_checkpoint,
    _select_contrastive_beta,
)
from Experiments.run_unified_benchmark_sbatch import (
    BenchmarkJob,
    build_python_command,
    parse_args,
)
from utils.unified_metrics import evaluate_all


def _random_contrastive_case(trial_index: int):
    rng = np.random.default_rng(0)
    for _ in range(trial_index + 1):
        logits = rng.normal(size=(30, 3)) * 1.2
        base_probs = _apply_contrastive_correction(
            logits, 1.0, 0.0, 0.0, np.zeros_like(logits)
        )
        labels = rng.integers(3, size=30)
        use_base_label = rng.random(30) < 0.55
        labels[use_base_label] = np.argmax(base_probs, axis=1)[use_base_label]
        signal = rng.normal(size=(30, 3))
    return logits, labels, base_probs, signal


def test_contrastive_distance_signal_uses_base_predicted_class_distance():
    distances = np.array([[3.0, 1.0, 4.0], [2.0, 5.0, 1.0]])
    base_probs = np.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]])

    signal = _contrastive_distance_signal(distances, base_probs)

    np.testing.assert_allclose(signal, [[0.0, 2.0, -1.0], [-1.0, -4.0, 0.0]])


def test_apply_contrastive_correction_supports_vector_and_scalar_parameters():
    logits = np.array([[1.0, 2.0], [-1.0, 0.5]])
    signal = np.array([[0.5, -0.5], [1.0, 0.0]])

    vector_probs = _apply_contrastive_correction(
        logits,
        a=np.array([2.0, 0.5]),
        b=np.array([0.1, -0.2]),
        beta=0.3,
        s=signal,
    )
    scalar_probs = _apply_contrastive_correction(
        logits, a=1.0, b=0.0, beta=0.0, s=signal
    )

    np.testing.assert_allclose(vector_probs.sum(axis=1), 1.0)
    np.testing.assert_allclose(scalar_probs.sum(axis=1), 1.0)
    np.testing.assert_array_equal(np.argmax(scalar_probs, axis=1), np.argmax(logits, axis=1))

    corrected_logits = _contrastive_corrected_logits(
        logits,
        a=np.array([2.0, 0.5]),
        b=np.array([0.1, -0.2]),
        beta=0.3,
        s=signal,
    )
    post_temperature_probs = _apply_contrastive_correction(
        corrected_logits,
        a=1.0 / 2.5,
        b=0.0,
        beta=0.0,
        s=np.zeros_like(corrected_logits),
    )
    np.testing.assert_array_equal(
        np.argmax(post_temperature_probs, axis=1),
        np.argmax(vector_probs, axis=1),
    )


def test_contrastive_beta_selection_records_curve_and_prefers_net_positive_flip():
    logits = np.array([[0.2, 0.0], [2.0, 0.0]])
    labels = np.array([1, 0])
    signal = np.array([[-1.0, 1.0], [-1.0, 1.0]])
    base_probs = _apply_contrastive_correction(
        logits, a=1.0, b=0.0, beta=0.0, s=signal
    )

    result = _select_contrastive_beta(
        logits=logits,
        labels=labels,
        base_probs=base_probs,
        signal=signal,
        a=1.0,
        b=0.0,
        beta_grid=[0.0, 0.2],
        nll_tolerance=1.0,
        brier_tolerance=1.0,
        ece_abs_tolerance=1.0,
    )

    assert result["selected_beta"] == 0.2
    assert result["best_select_net_flips"] == 1
    assert result["selected_fallback"] is False
    assert len(result["sweep_curve"]) == 2
    assert {
        "beta",
        "nll",
        "brier",
        "accuracy",
        "ece",
        "net_flips",
        "flip_to_correct",
        "flip_to_wrong",
        "select_nll",
        "select_brier",
        "select_top_label_ece",
        "select_net_flips",
        "gate_pass",
    }.issubset(result["sweep_curve"][0])
    assert result["selection_criterion"] == CONTRASTIVE_SELECTION_CRITERION
    assert "nll_optimal_beta" in result
    assert "nll_optimal_beta_select_nll" in result


def test_loose_new_gates_reproduce_nll_only_selection():
    logits, labels, base_probs, signal = _random_contrastive_case(76)
    result = _select_contrastive_beta(
        logits,
        labels,
        base_probs,
        signal,
        1.0,
        0.0,
        [0.1, 0.3, 1.0, 3.0],
        nll_tolerance=0.02,
        brier_tolerance=10.0,
        ece_abs_tolerance=10.0,
    )
    legacy_candidates = [
        row
        for row in result["sweep_curve"]
        if row["select_nll"] <= result["nll_budget"]
        and row["select_net_flips"] >= 0
    ]
    legacy_selected = max(
        legacy_candidates,
        key=lambda row: (
            row["select_net_flips"],
            row["select_accuracy"],
            -row["select_nll"],
            -abs(row["beta"]),
        ),
    )

    assert result["selected_beta"] == legacy_selected["beta"]
    assert result["sweep_curve"][0]["beta"] == 0.0


def test_ece_gate_rejects_winning_beta_and_selects_lower_beta():
    logits, labels, base_probs, signal = _random_contrastive_case(76)
    loose = _select_contrastive_beta(
        logits, labels, base_probs, signal, 1.0, 0.0,
        [0.0, 0.1, 0.3, 1.0, 3.0], 10.0, 10.0, 10.0,
    )
    gated = _select_contrastive_beta(
        logits, labels, base_probs, signal, 1.0, 0.0,
        [0.0, 0.1, 0.3, 1.0, 3.0], 10.0, 10.0, 0.0,
    )
    rejected = next(
        row for row in gated["sweep_curve"] if row["beta"] == loose["selected_beta"]
    )

    assert loose["selected_beta"] == 0.3
    assert rejected["ece_gate_pass"] is False
    assert rejected["nll_gate_pass"] is True
    assert rejected["brier_gate_pass"] is True
    assert gated["selected_beta"] == 0.1


def test_brier_gate_rejects_winning_beta_and_selects_lower_beta():
    logits, labels, base_probs, signal = _random_contrastive_case(64)
    loose = _select_contrastive_beta(
        logits, labels, base_probs, signal, 1.0, 0.0,
        [0.0, 0.1, 0.3, 1.0, 3.0], 10.0, 10.0, 10.0,
    )
    gated = _select_contrastive_beta(
        logits, labels, base_probs, signal, 1.0, 0.0,
        [0.0, 0.1, 0.3, 1.0, 3.0], 10.0, 0.0, 10.0,
    )
    rejected = next(
        row for row in gated["sweep_curve"] if row["beta"] == loose["selected_beta"]
    )

    assert loose["selected_beta"] == 0.3
    assert rejected["brier_gate_pass"] is False
    assert rejected["nll_gate_pass"] is True
    assert rejected["ece_gate_pass"] is True
    assert gated["selected_beta"] == 0.1


def test_contrastive_gate_falls_back_to_beta_zero_when_none_pass():
    logits = np.array([[2.0, 0.0], [0.0, 2.0]])
    labels = np.array([0, 1])
    signal = np.zeros_like(logits)
    base_probs = _apply_contrastive_correction(logits, 1.0, 0.0, 0.0, signal)

    result = _select_contrastive_beta(
        logits, labels, base_probs, signal, -1.0, 0.0, [0.0],
    )

    assert result["selected_beta"] == 0.0
    assert result["selected_fallback"] is True
    assert result["sweep_curve"][0]["gate_pass"] is False


def test_post_temperature_gates_final_probs_and_preserves_net_flips():
    logits, labels, base_probs, signal = _random_contrastive_case(3)
    beta_grid = [0.0, 0.3]
    raw = ContrastiveBetaSelector(
        nll_tolerance=10.0,
        brier_tolerance=10.0,
        ece_abs_tolerance=10.0,
    ).select(logits, labels, base_probs, signal, 1.0, 0.0, beta_grid)
    post = ContrastiveBetaSelector(
        nll_tolerance=10.0,
        brier_tolerance=10.0,
        ece_abs_tolerance=10.0,
        post_temperature=True,
    ).select(logits, labels, base_probs, signal, 1.0, 0.0, beta_grid)

    for raw_row, post_row in zip(raw["sweep_curve"], post["sweep_curve"]):
        assert raw_row["beta"] == post_row["beta"]
        assert raw_row["select_net_flips"] == post_row["select_net_flips"]
        assert post_row["post_temperature"] is not None
        corrected = _contrastive_corrected_logits(
            logits, 1.0, 0.0, post_row["beta"], signal
        )
        post_probs = _apply_contrastive_correction(
            corrected,
            1.0 / post_row["post_temperature"],
            0.0,
            0.0,
            np.zeros_like(corrected),
        )
        metrics = evaluate_all(post_probs, labels)
        assert post_row["select_nll"] == metrics["nll"]
        assert post_row["select_brier"] == metrics["brier"]
        assert post_row["select_top_label_ece"] == metrics["top_label_ece"]


def test_contrastive_checkpoint_gate_signature_is_required_and_matched(tmp_path):
    method_name = "contrastive_beta_vs"
    probs = np.array([[0.7, 0.3], [0.2, 0.8]])
    labels = np.array([0, 1])
    signature = _contrastive_gate_signature(
        0.02, 0.05, 0.005, post_temperature=False
    )
    entry = _make_method_entry(
        method_name,
        probs,
        labels,
        "decision_improving_contrastive",
        True,
        "contrastive_beta_grid",
        "inner_val_select",
        CONTRASTIVE_SELECTION_CRITERION,
        {"gate_signature": signature},
    )
    _save_method_checkpoint(
        str(tmp_path), method_name, entry, probs, True, "test"
    )

    assert _is_method_checkpoint_complete(
        str(tmp_path), method_name, 2, 2, expected_gate_signature=signature
    )
    assert not _is_method_checkpoint_complete(
        str(tmp_path),
        method_name,
        2,
        2,
        expected_gate_signature=signature + "-different",
    )

    meta_path = (
        tmp_path / "intermediates" / "method_outputs" / method_name / "meta.json"
    )
    meta = json.loads(meta_path.read_text())
    meta.pop("gate_signature")
    meta_path.write_text(json.dumps(meta))
    assert not _is_method_checkpoint_complete(
        str(tmp_path), method_name, 2, 2, expected_gate_signature=signature
    )

    # Any registered non-contrastive method; _make_method_entry now requires a
    # canonical semantics registry entry, so a made-up name is rejected.
    noncontrast_name = "temperature_scaling"
    noncontrast_entry = _make_method_entry(
        noncontrast_name,
        probs,
        labels,
        "unrelated",
        False,
        "none",
        "none",
        "none",
    )
    _save_method_checkpoint(
        str(tmp_path), noncontrast_name, noncontrast_entry, probs, False, "test"
    )
    assert _is_method_checkpoint_complete(
        str(tmp_path), noncontrast_name, 2, 2
    )


def test_contrastive_flags_default_on_and_launcher_forwards_both_states(tmp_path):
    assert "enable_contrastive_beta_sweep" in DEFAULT_ENABLED_METHOD_FLAGS
    assert CONTRASTIVE_BETA_METHOD_NAMES == (
        "contrastive_beta_vs",
        "contrastive_beta_vs_post_temperature",
        "contrastive_beta_ts",
        "contrastive_beta_ts_post_temperature",
    )
    required = ["--dataset", "cifar10", "--model", "resnet18", "--seeds", "1"]
    assert parse_args(required).enable_contrastive_beta_sweep is True
    assert parse_args([*required, "--no_contrastive_beta_sweep"]).enable_contrastive_beta_sweep is False

    common = dict(
        seed=1,
        benchmark_script=Path("benchmark.py"),
        dataset="cifar10",
        model="resnet18",
        method="baseline_cross_entropy",
        results_dir=tmp_path,
        output_dir=tmp_path / "out",
        batch_size=8,
        target_dimension=16,
        num_layers=2,
        num_coordinates=16,
        device="cpu",
        enable_post_fusion_temperature=False,
        enable_rgcl_tail_hybrids=False,
        rgcl_tail_sources="",
        debug_rgcl_tail_hybrid_smoke=False,
        debug_anchor_probs_diff=False,
        benchmark_extra_args=[],
    )
    enabled_command = build_python_command(BenchmarkJob(**common))
    disabled_command = build_python_command(
        BenchmarkJob(**common, enable_contrastive_beta_sweep=False)
    )

    assert "--enable_contrastive_beta_sweep" in enabled_command
    assert "--disable_contrastive_beta_sweep" in disabled_command
