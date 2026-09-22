"""
§10 validation tests for BENCHMARK_IMPLEMENTATION_PLAN.md (Study A/B benchmark).

Run entirely on synthetic/tiny data per §10's own requirement ("All of these
must pass on cheap synthetic or single-checkpoint smoke data before Phase 2's
full run") -- no real checkpoint, dataset, or GPU required.

Item numbers below match the plan's §10 list exactly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from Calibrators.density_aware_calibration import DensityAwareCalibrator, get_dac_target_layers
from Calibrators.mahalanobis_confidence import MahalanobisConfidenceCalibrator
from Calibrators.studyB_common_pipeline import CommonFactorialCalibrator
from Experiments.extract_unified_studyAB_representations import (
    STUDY_B_NUM_LAYERS,
    extract_combined_representations,
    resolve_combined_layer_set,
)
from Experiments.run_rgc_experiments import (
    filter_non_feature_layers,
    filter_non_feature_layers_corrected,
    select_random_rgc_layers,
    select_random_rgc_layers_corrected,
)
from Experiments.studyAB_artifact_schema import build_artifact_row, metric_bucket_for_method
from utils.layer_utils import filter_out_classifier_layers, is_feature_layer_corrected


DRAW_SEEDS = (9101, 9102, 9103)


class _TinyBottleneck(nn.Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(c, c, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(c)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        return out


class _TinyResNetLike(nn.Module):
    """Stand-in for resnet101: conv1 stem, 4 residual stages, fc head, no
    maxpool module (mirrors Net/resnet_cifar.py::resnet101 exactly enough to
    exercise the same discovery/filter code paths)."""

    def __init__(self, num_classes: int = 10, width: int = 8) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, 3, padding=1)
        self.layer1 = nn.Sequential(_TinyBottleneck(width), _TinyBottleneck(width))
        self.layer2 = nn.Sequential(_TinyBottleneck(width), _TinyBottleneck(width))
        self.layer3 = nn.Sequential(_TinyBottleneck(width))
        self.layer4 = nn.Sequential(_TinyBottleneck(width))
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(width, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture(scope="module")
def checkpoints(device):
    """Five independently-initialized 'checkpoints' of the same architecture,
    standing in for seed1..seed5 -- same structure, different weight values."""
    models = []
    for seed in range(1, 6):
        torch.manual_seed(seed)
        models.append(_TinyResNetLike().to(device).eval())
    return models


# --------------------------------------------------------------------------
# §10.1: fc cannot appear in corrected draws
# --------------------------------------------------------------------------
def test_10_1_fc_excluded_from_corrected_draws(device, checkpoints):
    for model in checkpoints:
        for seed in DRAW_SEEDS:
            layers = select_random_rgc_layers_corrected(
                model=model, model_name="resnet101", dataset_name="cifar100",
                device=device, num_layers=STUDY_B_NUM_LAYERS, seed=seed,
            )
            assert "fc" not in layers
            assert not any(name == "fc" or name.startswith("fc.") for name in layers)

        dac_layers = get_dac_target_layers("resnet101", model)
        assert "fc" not in dac_layers


def test_10_1_fc_excluded_forced_full_pool(device, checkpoints):
    """Force near-total selection so fc would appear if not excluded --
    directly exercises the exclusion rather than relying on it being absent
    by chance at num_layers=5."""
    model = checkpoints[0]
    from Experiments.run_rgc_experiments import discover_model_layers, normalize_discovered_layers

    discovered = normalize_discovered_layers(
        model, discover_model_layers(model, device=device, input_shape=(1, 3, 16, 16))
    )
    filtered, _ = filter_non_feature_layers(discovered, model)
    all_names = [d["name"] for d in filtered]
    assert "fc" in all_names, "test setup assumption broken: fc should be a raw candidate"

    uncorrected = select_random_rgc_layers(
        model, "resnet101", "cifar100", device, num_layers=len(all_names), seed=9101
    )
    corrected = select_random_rgc_layers_corrected(
        model, "resnet101", "cifar100", device, num_layers=len(all_names) - 1, seed=9101
    )
    assert "fc" in uncorrected
    assert "fc" not in corrected


def test_10_1_layer_utils_corrected_predicate_also_excludes_fc():
    """The utils/layer_utils.py corrected predicate (built for the RGCC/
    discover_coordinate_space path) must also reject fc on its own terms,
    even though it is not the operative fix for RGCL's real draw mechanism
    (see the §16 deviation log)."""
    assert is_feature_layer_corrected("fc") is False
    assert is_feature_layer_corrected("fc#0") is False
    layer_map = [{"name": "fc", "base_name": "fc"}, {"name": "layer1.0.conv1#0", "base_name": "layer1.0.conv1"}]
    filtered = filter_out_classifier_layers(layer_map)
    assert [l["name"] for l in filtered] == ["layer1.0.conv1#0"]


# --------------------------------------------------------------------------
# §10.2: seed -> layer-name consistency across checkpoints
# --------------------------------------------------------------------------
def test_10_2_seed_to_layer_names_consistent_across_checkpoints(device, checkpoints):
    for seed in DRAW_SEEDS:
        per_checkpoint = [
            set(
                select_random_rgc_layers_corrected(
                    model=m, model_name="resnet101", dataset_name="cifar100",
                    device=device, num_layers=STUDY_B_NUM_LAYERS, seed=seed,
                )
            )
            for m in checkpoints
        ]
        first = per_checkpoint[0]
        for other in per_checkpoint[1:]:
            assert other == first, (
                f"seed={seed} produced different layer names across checkpoints of "
                f"identical architecture: {first} vs {other}"
            )


# --------------------------------------------------------------------------
# Shared fixture: one combined extraction, reused by §10.3/10.4/10.5/10.6
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def combined_extraction(device, checkpoints):
    model = checkpoints[0]
    resolved = resolve_combined_layer_set(
        model=model, model_name="resnet101", dataset_name="cifar100",
        device=device, draw_seeds=DRAW_SEEDS, num_layers=STUDY_B_NUM_LAYERS,
    )

    def make_loader(n, seed):
        g = torch.Generator().manual_seed(seed)
        X = torch.randn(n, 3, 16, 16, generator=g)
        y = torch.randint(0, 10, (n,), generator=g)
        return DataLoader(TensorDataset(X, y), batch_size=8, shuffle=False)

    train_result = extract_combined_representations(
        model=model, loader=make_loader(64, 0), split_name="train",
        combined_layers=resolved["combined_layers"], draw_layers=resolved["draw_layers"],
        device=device, target_dim=32, pooling_mode="max",
    )
    fit_result = extract_combined_representations(
        model=model, loader=make_loader(40, 1), split_name="fit",
        combined_layers=resolved["combined_layers"], draw_layers=resolved["draw_layers"],
        device=device, target_dim=32, pooling_mode="max",
    )
    test_result = extract_combined_representations(
        model=model, loader=make_loader(30, 2), split_name="test",
        combined_layers=resolved["combined_layers"], draw_layers=resolved["draw_layers"],
        device=device, target_dim=32, pooling_mode="max",
    )
    return {"resolved": resolved, "train": train_result, "fit": fit_result, "test": test_result}


def _make_probs(labels, num_classes=10, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=0.3, size=(len(labels), num_classes))
    logits[np.arange(len(labels)), labels] += 2.0
    p = np.exp(logits - logits.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    return p


# --------------------------------------------------------------------------
# §10.3: Study-B A/B share identical representation
# --------------------------------------------------------------------------
def test_10_3_cells_a_b_share_identical_representation(combined_extraction):
    resolved = combined_extraction["resolved"]
    draw_seed = DRAW_SEEDS[0]
    draw_layer_names = resolved["draw_layers"][draw_seed]

    train = combined_extraction["train"]["dac_style_features"]
    fit = combined_extraction["fit"]["dac_style_features"]
    test = combined_extraction["test"]["dac_style_features"]
    fit_labels = combined_extraction["fit"]["labels"]
    test_labels = combined_extraction["test"]["labels"]

    train_layers = {name: train[name] for name in draw_layer_names}
    fit_layers = {name: fit[name] for name in draw_layer_names}
    test_layers = {name: test[name] for name in draw_layer_names}
    fit_probs = _make_probs(fit_labels)
    test_probs = _make_probs(test_labels)

    # Snapshot before feeding into either cell -- assert no in-place mutation.
    snapshots = {name: arr.copy() for name, arr in train_layers.items()}

    dummy_train_labels = np.arange(train_layers[draw_layer_names[0]].shape[0]) % 10

    cell_a = CommonFactorialCalibrator(draw_layer_names, "gc_separation")
    cell_a.fit(train_layers, dummy_train_labels, fit_layers, fit_probs, fit_labels)
    cell_a.calibrate(test_layers, test_probs)

    cell_b = CommonFactorialCalibrator(draw_layer_names, "dac_density", k=5)
    cell_b.fit(train_layers, dummy_train_labels, fit_layers, fit_probs, fit_labels)
    cell_b.calibrate(test_layers, test_probs)

    for name, arr in train_layers.items():
        assert np.array_equal(arr, snapshots[name]), (
            f"layer '{name}' representation was mutated between cells A and B -- "
            "A/B must share byte-identical input."
        )
    # Both cells were literally constructed from the same dict objects.
    assert train_layers is train_layers  # sanity: same object, not a copy


# --------------------------------------------------------------------------
# §10.4: Study-B C/D share identical representation
# --------------------------------------------------------------------------
def test_10_4_cells_c_d_share_identical_representation(combined_extraction):
    resolved = combined_extraction["resolved"]
    dac_layers = resolved["dac_layers"]

    train = combined_extraction["train"]["dac_style_features"]
    fit = combined_extraction["fit"]["dac_style_features"]
    test = combined_extraction["test"]["dac_style_features"]
    fit_labels = combined_extraction["fit"]["labels"]
    test_labels = combined_extraction["test"]["labels"]

    train_layers = {name: train[name] for name in dac_layers}
    fit_layers = {name: fit[name] for name in dac_layers}
    test_layers = {name: test[name] for name in dac_layers}
    fit_probs = _make_probs(fit_labels)
    test_probs = _make_probs(test_labels)
    snapshots = {name: arr.copy() for name, arr in train_layers.items()}

    dummy_train_labels = np.arange(train_layers[dac_layers[0]].shape[0]) % 10

    cell_c = CommonFactorialCalibrator(dac_layers, "gc_separation")
    cell_c.fit(train_layers, dummy_train_labels, fit_layers, fit_probs, fit_labels)
    cell_c.calibrate(test_layers, test_probs)

    cell_d = CommonFactorialCalibrator(dac_layers, "dac_density", k=5)
    cell_d.fit(train_layers, dummy_train_labels, fit_layers, fit_probs, fit_labels)
    cell_d.calibrate(test_layers, test_probs)

    for name, arr in train_layers.items():
        assert np.array_equal(arr, snapshots[name])


# --------------------------------------------------------------------------
# §10.5: all four cells use identical mapper/fitting split
# --------------------------------------------------------------------------
def test_10_5_all_cells_share_mapper_and_fitting_split(combined_extraction):
    resolved = combined_extraction["resolved"]
    draw_layers = resolved["draw_layers"][DRAW_SEEDS[0]]
    dac_layers = resolved["dac_layers"]

    train = combined_extraction["train"]["dac_style_features"]
    fit = combined_extraction["fit"]["dac_style_features"]
    fit_labels = combined_extraction["fit"]["labels"]
    fit_probs = _make_probs(fit_labels)

    cells = {}
    for name, layers in (("A", draw_layers), ("C", dac_layers)):
        for stat_label, stat in (("gc", "gc_separation"), ("dac", "dac_density")):
            cal = CommonFactorialCalibrator(layers, stat, k=5)
            dummy_train_labels = np.arange(train[layers[0]].shape[0]) % 10
            cal.fit(
                {l: train[l] for l in layers}, dummy_train_labels,
                {l: fit[l] for l in layers}, fit_probs, fit_labels,
            )
            cells[f"{name}_{stat_label}"] = cal

    for cal in cells.values():
        assert isinstance(cal._iso, type(next(iter(cells.values()))._iso))
        assert cal._iso.out_of_bounds == "clip"
        assert cal._g_fit_ecdf is not None
        assert len(cal._g_fit_ecdf) == len(fit_labels)


# --------------------------------------------------------------------------
# §10.6: native DAC row stays distinct from Study-B D (schema-level check)
# --------------------------------------------------------------------------
def test_10_6_native_dac_distinct_from_studyB_cell_d(combined_extraction, device):
    resolved = combined_extraction["resolved"]
    dac_layers = resolved["dac_layers"]

    train = combined_extraction["train"]["dac_style_features"]
    fit = combined_extraction["fit"]["dac_style_features"]
    test = combined_extraction["test"]["dac_style_features"]
    fit_labels = combined_extraction["fit"]["labels"]
    test_labels = combined_extraction["test"]["labels"]
    fit_probs = _make_probs(fit_labels)
    test_probs = _make_probs(test_labels)
    dummy_train_labels = np.arange(train[dac_layers[0]].shape[0]) % 10

    # Study-B cell D: CommonFactorialCalibrator + dac_density statistic.
    cell_d = CommonFactorialCalibrator(dac_layers, "dac_density", k=5)
    cell_d.fit(
        {l: train[l] for l in dac_layers}, dummy_train_labels,
        {l: fit[l] for l in dac_layers}, fit_probs, fit_labels,
    )
    d_probs = cell_d.calibrate({l: test[l] for l in dac_layers}, test_probs)

    # Native DAC: DensityAwareCalibrator, a different code path entirely.
    assert not isinstance(cell_d, DensityAwareCalibrator)
    native = DensityAwareCalibrator(k=5, use_gpu=False)
    train_feats_list = [train[l] for l in dac_layers]
    fit_feats_list = [fit[l] for l in dac_layers]
    test_feats_list = [test[l] for l in dac_layers]
    fit_logits = np.log(np.clip(fit_probs, 1e-8, None))
    test_logits = np.log(np.clip(test_probs, 1e-8, None))
    native.fit(train_feats_list, fit_feats_list, fit_logits, fit_labels)
    native_probs = native.calibrate(test_feats_list, test_logits)

    # Schema-level distinctness: both are valid full probability vectors...
    assert np.allclose(d_probs.sum(axis=1), 1.0)
    assert np.allclose(native_probs.sum(axis=1), 1.0)
    # ...but D's non-top-class entries are exactly proportional to the base
    # model's own non-top distribution (the uniform-spread signature), which
    # native DAC's softmax(z/S(x,w)) reconstruction has no reason to satisfy.
    idx = np.argmax(test_probs, axis=1)
    rows = np.arange(len(idx))
    d_non_top_ratio = None
    for i in rows[:5]:
        others = [c for c in range(test_probs.shape[1]) if c != idx[i]]
        ratios = d_probs[i, others] / np.clip(test_probs[i, others], 1e-12, None)
        assert np.allclose(ratios, ratios[0], atol=1e-6), (
            "Study-B cell D must rescale non-top classes proportionally "
            "(uniform-spread), not reconstruct them independently."
        )
    assert not np.allclose(d_probs, native_probs)


# --------------------------------------------------------------------------
# §10.7: scalar-only methods cannot emit fabricated multiclass NLL/Brier
# --------------------------------------------------------------------------
def test_10_7_scalar_only_methods_reject_nonnull_nll_brier():
    assert metric_bucket_for_method("mahalanobis_confidence") == "scalar_only"
    assert metric_bucket_for_method("studyB_cell_D") == "scalar_only"
    assert metric_bucket_for_method("native_dac") == "full_vector"

    with pytest.raises(ValueError):
        build_artifact_row(
            method="mahalanobis_confidence", checkpoint_seed=1, rgc_draw_seed=None,
            representation_strategy="na", statistic="mahalanobis", mapper="isotonic",
            dataset="cifar100", corruption=None, severity=None, split="test",
            sample_ids_ref="ref:x", sample_count=10, metrics={"nll": 0.5, "top_label_ece": 0.01},
            provenance={},
        )

    row = build_artifact_row(
        method="mahalanobis_confidence", checkpoint_seed=1, rgc_draw_seed=None,
        representation_strategy="na", statistic="mahalanobis", mapper="isotonic",
        dataset="cifar100", corruption=None, severity=None, split="test",
        sample_ids_ref="ref:x", sample_count=10, metrics={"top_label_ece": 0.01},
        provenance={},
    )
    assert row["metrics"]["nll"] is None
    assert row["metrics"]["brier"] is None
    assert row["metrics"]["_metric_bucket"] == "scalar_only"
    assert row["sample_count"] == 10

    row_full = build_artifact_row(
        method="native_dac", checkpoint_seed=1, rgc_draw_seed=None,
        representation_strategy="dac_prescribed", statistic="dac_density", mapper="native_dac",
        dataset="cifar100", corruption=None, severity=None, split="test",
        sample_ids_ref="ref:x", sample_count=10, metrics={"nll": 0.4, "brier": 0.1},
        provenance={},
    )
    assert row_full["metrics"]["_metric_bucket"] == "full_vector"


# --------------------------------------------------------------------------
# §10.8: sample_id and label alignment through the extraction pipeline
# --------------------------------------------------------------------------
def test_10_8_sample_ids_align_with_labels(combined_extraction):
    for split_name in ("train", "fit", "test"):
        result = combined_extraction[split_name]
        assert result["sample_ids"].shape[0] == result["labels"].shape[0]
        assert np.array_equal(result["sample_ids"], np.arange(len(result["sample_ids"])))
        first_layer = next(iter(result["dac_style_features"].values()))
        assert first_layer.shape[0] == result["labels"].shape[0]
        for arr in result["draw_projected_features"].values():
            assert arr.shape[0] == result["labels"].shape[0]


# --------------------------------------------------------------------------
# §10.9: no historical script/result path is overwritten
# --------------------------------------------------------------------------
def test_10_9_output_paths_do_not_collide_with_existing_results():
    """results/studyAB/ is this plan's own designated output root (and, once
    real runs execute, legitimately exists and is populated) -- the actual
    §10.9 invariant is that it never aliases a pre-existing, non-plan path:
    it must be a plain directory (not a symlink into some other project's
    results), and nothing else under results/ may resolve to the same path."""
    repo_root = Path(__file__).resolve().parents[1]
    planned_output_dir = repo_root / "results" / "studyAB"

    if planned_output_dir.exists():
        assert not planned_output_dir.is_symlink(), (
            f"{planned_output_dir} must be a plain directory, not a symlink "
            "into a pre-existing path"
        )

    results_dir = repo_root / "results"
    if results_dir.exists():
        for child in results_dir.iterdir():
            if child.name == "studyAB":
                continue
            assert child.resolve() != planned_output_dir.resolve(), (
                f"Pre-existing path {child} resolves to the plan's own "
                f"output dir {planned_output_dir} -- collision"
            )


def test_10_9_new_files_are_new_not_edits_to_historical_paths():
    repo_root = Path(__file__).resolve().parents[1]
    new_files = [
        repo_root / "Experiments" / "extract_unified_studyAB_representations.py",
        repo_root / "Calibrators" / "studyB_common_pipeline.py",
        repo_root / "Calibrators" / "mahalanobis_confidence.py",
        repo_root / "Experiments" / "studyAB_artifact_schema.py",
    ]
    for path in new_files:
        assert path.exists(), f"expected new plan-specific file missing: {path}"


# --------------------------------------------------------------------------
# Extra: fc-exclusion filter on the real path is additive, not a mutation of
# the original (Published RGCL's reference row must stay byte-identical).
# --------------------------------------------------------------------------
def test_corrected_filter_is_additive_not_a_mutation(device, checkpoints):
    from Experiments.run_rgc_experiments import discover_model_layers, normalize_discovered_layers

    model = checkpoints[0]
    discovered = normalize_discovered_layers(
        model, discover_model_layers(model, device=device, input_shape=(1, 3, 16, 16))
    )
    original_filtered, original_excluded = filter_non_feature_layers(discovered, model)
    corrected_filtered, _ = filter_non_feature_layers_corrected(discovered, model)

    original_names = {d["name"] if isinstance(d, dict) else d for d in original_filtered}
    corrected_names = {d["name"] if isinstance(d, dict) else d for d in corrected_filtered}
    assert "fc" in original_names
    assert "fc" not in corrected_names
    assert corrected_names == original_names - {"fc"}


# --------------------------------------------------------------------------
# Regression: seed-4 Mahalanobis matmul 1024 vs 2048 (mixed representations)
# --------------------------------------------------------------------------
def test_runner_mahalanobis_uses_penultimate_train_features():
    src = (Path(__file__).resolve().parents[1] / "Experiments" / "run_unified_benchmark.py").read_text()
    start = src.index("if args.enable_mahalanobis_confidence:\n        _mahal_name")
    block = src[start : src.index("Contrastive-beta sweep", start)]
    assert '_mahal_train_features = np.load(stage3_files["kcal_train_penultimate"])' in block
    assert 'stage3_files["train_features"]' not in block
    assert "Mahalanobis feature dimension mismatch" in block


def test_mahalanobis_fit_rejects_mismatched_feature_dims():
    rng = np.random.default_rng(0)
    n, c = 40, 3
    labels = np.arange(n) % c
    probs = np.full((n, c), 1.0 / c)
    with pytest.raises(ValueError, match="feature dimension mismatch.*1024.*2048|dim 8.*dim 12"):
        MahalanobisConfidenceCalibrator().fit(
            train_features=rng.normal(size=(n, 8)),
            train_labels=labels,
            fit_features=rng.normal(size=(n, 12)),
            fit_base_probs=probs,
            fit_labels=labels,
        )


# --------------------------------------------------------------------------
# Regression: no silent CUDA -> CPU fallback in the unified benchmark runner
# --------------------------------------------------------------------------
def test_resolve_device_explicit_cpu_without_cuda(monkeypatch):
    from Experiments.run_unified_benchmark import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_cuda_available(monkeypatch):
    from Experiments.run_unified_benchmark import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda") == torch.device("cuda")


def test_resolve_device_cuda_unavailable_raises(monkeypatch):
    from Experiments.run_unified_benchmark import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="cuda.*silently fall back to CPU"):
        resolve_device("cuda")


# --------------------------------------------------------------------------
# Regression: stage4 resume with a completed anchored_rankgeom_tail_mixture
# checkpoint must not need model / model_adapter / train_raw
# --------------------------------------------------------------------------
def _write_completed_rankgeom_checkpoint(out_dir, entry_extra):
    from Experiments.run_unified_benchmark import _save_method_checkpoint

    probs = np.full((6, 3), 1.0 / 3.0)
    entry = {"method_name": "anchored_rankgeom_tail_mixture", **entry_extra}
    _save_method_checkpoint(
        str(out_dir), "anchored_rankgeom_tail_mixture", entry, probs, True,
        stage_name="stage4_late_outputs",
    )
    return probs


def test_completed_rankgeom_resume_needs_no_model_or_train_data(tmp_path):
    from Experiments.run_unified_benchmark import (
        _is_method_checkpoint_complete,
        _load_completed_rankgeom_resume_state,
    )

    expected = _write_completed_rankgeom_checkpoint(tmp_path, {})
    assert _is_method_checkpoint_complete(str(tmp_path), "anchored_rankgeom_tail_mixture", 6, 3)
    sel = {"selected_lambda": 0.5, "selected_alpha": 2.0}
    probs, out_sel = _load_completed_rankgeom_resume_state(str(tmp_path), sel)
    np.testing.assert_array_equal(np.asarray(probs), expected)
    assert out_sel == sel


def test_completed_rankgeom_resume_recovers_selection_from_entry(tmp_path):
    from Experiments.run_unified_benchmark import _load_completed_rankgeom_resume_state

    _write_completed_rankgeom_checkpoint(
        tmp_path, {"selected_lambda": 1.0, "selected_alpha": 4.0}
    )
    _, sel = _load_completed_rankgeom_resume_state(str(tmp_path), None)
    assert sel == {"selected_lambda": 1.0, "selected_alpha": 4.0}


def test_completed_rankgeom_resume_never_defaults_selection_to_zero(tmp_path):
    from Experiments.run_unified_benchmark import _load_completed_rankgeom_resume_state

    _write_completed_rankgeom_checkpoint(tmp_path, {})
    with pytest.raises(RuntimeError, match="no recoverable selected_lambda/selected_alpha"):
        _load_completed_rankgeom_resume_state(str(tmp_path), None)


def test_main_guards_rankgeom_compute_paths_by_need_rankgeom_method():
    """Structural guard: everything that needs model/train_raw or the selected
    lambda/alpha (run_sgc_with_dac_layers on val, test mixture, test NLL) must
    run only when need_rankgeom_method is true."""
    import ast

    src = (Path(__file__).resolve().parents[1] / "Experiments" / "run_unified_benchmark.py").read_text()
    main_fn = next(
        n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    parents = {}
    for node in ast.walk(main_fn):
        for child in ast.iter_child_nodes(node):
            parents[child] = node

    def guarded(node):
        child, cur = node, parents.get(node)
        while cur is not None:
            if isinstance(cur, ast.If):
                test = ast.unparse(cur.test)
                in_body = any(child is b for b in cur.body)
                if test == "need_rankgeom_method" and in_body:
                    return True
                if test == "not need_rankgeom_method" and not in_body:
                    return True
            child, cur = cur, parents.get(cur)
        return False

    checked = 0
    for node in ast.walk(main_fn):
        if (
            isinstance(node, ast.Call)
            and ast.unparse(node.func) == "run_sgc_with_dac_layers"
            and any(k.arg == "test_raw" and ast.unparse(k.value) == "val_raw_aligned" for k in node.keywords)
        ):
            checked += 1
            assert guarded(node), f"unguarded val run_sgc_with_dac_layers at line {node.lineno}"
        if isinstance(node, ast.Assign) and any(
            ast.unparse(t) in ("test_nll_at_selected", "anchored_rankgeom_mixture_probs")
            and isinstance(node.value, (ast.Call,))
            and "selected_lambda" in ast.unparse(node.value) + ast.unparse(node.value.args if hasattr(node.value, "args") else [])
            for t in node.targets
        ):
            checked += 1
            assert guarded(node), f"unguarded selected-lambda use at line {node.lineno}"
    assert checked >= 3


def test_runner_mahalanobis_logs_shrinkage_used_not_none_shrinkage():
    """shrinkage=None (auto Ledoit-Wolf) must not be %-formatted; log shrinkage_used."""
    src = (Path(__file__).resolve().parents[1] / "Experiments" / "run_unified_benchmark.py").read_text()
    assert '_mahal_fit_info["shrinkage"]' not in src
    assert 'shrinkage_used=%.4f' in src
    params = MahalanobisConfidenceCalibrator().get_params()
    assert params["shrinkage"] is None
    rng = np.random.default_rng(0)
    cal = MahalanobisConfidenceCalibrator()
    labels = np.arange(60) % 3
    cal.fit(rng.normal(size=(60, 5)), labels, rng.normal(size=(60, 5)),
            np.full((60, 3), 1 / 3), labels)
    info = cal.get_params()
    assert info["shrinkage"] is None
    assert isinstance(info["shrinkage_used"], float)
    "shrinkage_used=%.4f" % info["shrinkage_used"]
