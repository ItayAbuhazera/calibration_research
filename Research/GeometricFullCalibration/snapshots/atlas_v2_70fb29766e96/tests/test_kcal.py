"""Tests for the full learned-projection KCal implementation."""

from __future__ import annotations

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from types import SimpleNamespace

from Calibrators.kcal import KCalCalibrator, SkipELUProjection
from Experiments.run_unified_benchmark import (
    DEFAULT_ENABLED_METHOD_FLAGS,
    _PenultimateFeatureCapture,
    _apply_disable_all_optional_methods,
    _args_fingerprint,
    _configure_default_method_flags,
    _extract_split_features_to_disk,
    _kcal_factorial_method_name,
)


def _synthetic(seed: int = 7):
    rng = np.random.RandomState(seed)
    train_parts = []
    cal_parts = []
    test_parts = []
    for center in [-3.0, 0.0, 3.0]:
        mean = np.array([center, center * 0.5, -center, 0.25], dtype=np.float32)
        train_parts.append(rng.normal(mean, 0.35, size=(12, 4)).astype(np.float32))
        cal_parts.append(rng.normal(mean, 0.35, size=(6, 4)).astype(np.float32))
        test_parts.append(rng.normal(mean, 0.35, size=(3, 4)).astype(np.float32))
    y_train = np.repeat(np.arange(3), 12)
    y_cal = np.repeat(np.arange(3), 6)
    return (
        np.vstack(train_parts),
        y_train,
        np.vstack(cal_parts),
        y_cal,
        np.vstack(test_parts),
    )


def _small_calibrator(**overrides):
    kwargs = {
        "projection_dim": 3,
        "projection_epochs": 2,
        "query_batch_size": 9,
        "references_per_class": 4,
        "bandwidth_folds": 2,
        "bandwidth_tolerance": 0.75,
        "prediction_batch_size": 8,
        "reference_batch_size": 16,
        "seed": 11,
        "device": "cpu",
    }
    kwargs.update(overrides)
    return KCalCalibrator(**kwargs)


def test_skip_elu_projection_shape_and_gradients():
    projection = SkipELUProjection(7, 3)
    x = torch.randn(10, 7, requires_grad=True)
    output = projection(x)
    assert output.shape == (10, 3)
    output.square().mean().backward()
    assert x.grad is not None
    assert all(parameter.grad is not None for parameter in projection.parameters())


def test_penultimate_capture_uses_final_classifier_input():
    class ToyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Linear(5, 4)
            self.fc = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(torch.relu(self.features(x)))

    model = ToyClassifier()
    capture = _PenultimateFeatureCapture(model)
    inputs = torch.randn(6, 5)
    expected = torch.relu(model.features(inputs))
    model(inputs)
    assert capture.layer_name == "fc"
    assert capture.features is not None
    torch.testing.assert_close(capture.features, expected)
    capture.cleanup()


def test_split_extraction_persists_penultimate_features(tmp_path):
    class ToyClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.features = torch.nn.Linear(5, 4)
            self.fc = torch.nn.Linear(4, 3)

        def forward(self, x):
            return self.fc(torch.relu(self.features(x)))

    class ToyFeatureExtractor:
        def __init__(self, model):
            self.model = model
            self.features = None
            self.layer_features = {}
            self.handle = model.features.register_forward_hook(self._capture)

        def _capture(self, _module, _inputs, output):
            self.features = output

        def cleanup(self):
            self.handle.remove()

    torch.manual_seed(4)
    model = ToyClassifier()
    extractor = ToyFeatureExtractor(model)
    capture = _PenultimateFeatureCapture(model)
    inputs = torch.randn(11, 5)
    labels = torch.arange(11) % 3
    loader = DataLoader(TensorDataset(inputs, labels), batch_size=4, shuffle=False)
    paths = _extract_split_features_to_disk(
        feature_extractor=extractor,
        data_loader=loader,
        device=torch.device("cpu"),
        out_prefix=str(tmp_path / "split"),
        include_raw=True,
        penultimate_capture=capture,
    )
    raw_path, feature_path, logits_path, label_path, penultimate_path = paths
    assert raw_path is not None and penultimate_path is not None
    np.testing.assert_allclose(np.load(raw_path), inputs.numpy())
    np.testing.assert_array_equal(np.load(label_path), labels.numpy())
    assert np.load(feature_path).shape == (11, 4)
    assert np.load(logits_path).shape == (11, 3)
    expected_penultimate = torch.relu(model.features(inputs)).detach().numpy()
    np.testing.assert_allclose(np.load(penultimate_path), expected_penultimate, atol=1e-6)
    extractor.cleanup()
    capture.cleanup()


def test_full_kernel_posterior_matches_manual_formula():
    queries = torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    references = torch.tensor(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 1.0], [1.5, 1.0]], dtype=torch.float32
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    bandwidth = 0.7
    actual = KCalCalibrator._kernel_posterior_torch(
        queries,
        references,
        labels,
        bandwidth=bandwidth,
        num_classes=2,
        reference_batch_size=2,
    ).numpy()

    squared_mean = ((queries[:, None, :] - references[None, :, :]) ** 2).mean(dim=2).numpy()
    kernels = np.exp(-squared_mean / bandwidth)
    expected_scores = np.column_stack(
        [kernels[:, labels.numpy() == class_id].sum(axis=1) for class_id in range(2)]
    )
    expected = expected_scores / expected_scores.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)


def test_training_sampler_keeps_queries_out_of_reference_set():
    labels = np.repeat(np.arange(3), 8)
    calibrator = _small_calibrator()
    class_indices = [np.flatnonzero(labels == class_id) for class_id in range(3)]
    query, reference, weights = calibrator._sample_training_batch(
        labels, class_indices, np.random.RandomState(3)
    )
    assert set(query).isdisjoint(set(reference))
    assert len(reference) == 3 * calibrator.config.references_per_class
    assert len(weights) == len(reference)


def test_bandwidth_search_uses_cv_objective_minimum():
    calibrator = _small_calibrator(bandwidth_lower=0.2, bandwidth_upper=4.0)
    calibrator.num_classes = 2
    projected = np.zeros((8, 2), dtype=np.float32)
    labels = np.repeat([0, 1], 4)

    def quadratic_loss(bandwidth, _projected, _labels, _folds):
        return float((bandwidth - 1.6) ** 2)

    calibrator._bandwidth_cv_loss = quadratic_loss  # type: ignore[method-assign]
    selected = calibrator._select_bandwidth(projected, labels)
    assert abs(selected - 1.6) < calibrator.config.bandwidth_tolerance
    assert calibrator.bandwidth_folds_used == 2


def test_fit_calibrate_is_deterministic_and_normalized():
    X_train, y_train, X_cal, y_cal, X_test = _synthetic()
    first = _small_calibrator().fit(X_train, y_train, X_cal, y_cal)
    second = _small_calibrator().fit(X_train, y_train, X_cal, y_cal)
    first_probs = first.calibrate(X_test)
    second_probs = second.calibrate(X_test)
    assert first_probs.shape == (len(X_test), 3)
    np.testing.assert_allclose(first_probs.sum(axis=1), 1.0, atol=1e-8)
    np.testing.assert_allclose(first_probs, second_probs, rtol=1e-6, atol=1e-7)
    assert np.all(first_probs >= 0.0)
    assert first.get_params()["calibration_reference_count"] == len(X_cal)
    assert "alpha" not in first.get_params()


def test_checkpoint_round_trip(tmp_path):
    X_train, y_train, X_cal, y_cal, X_test = _synthetic()
    fitted = _small_calibrator().fit(X_train, y_train, X_cal, y_cal)
    expected = fitted.calibrate(X_test)
    path = tmp_path / "kcal.pt"
    fitted.save_checkpoint(str(path))
    restored = KCalCalibrator.load_checkpoint(str(path), device="cpu")
    actual = restored.calibrate(X_test)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)
    assert restored.bandwidth == fitted.bandwidth
    assert restored.output_dim == fitted.output_dim


def test_public_transform_matches_internal_projection():
    X_train, y_train, X_cal, y_cal, X_test = _synthetic()
    fitted = _small_calibrator().fit(X_train, y_train, X_cal, y_cal)
    np.testing.assert_allclose(
        fitted.transform(X_test), fitted._project_numpy(X_test), rtol=1e-7, atol=1e-8
    )


def test_factorial_method_name_encodes_all_axes():
    assert _kcal_factorial_method_name(
        "rgcl", "blend_frozen", "topk", "validation", "exp_l2"
    ) == "kcal_factorial_rgcl_blend_frozen_topk_cal_exp_l2"


def test_all_method_families_are_enabled_by_default_and_individually_disableable():
    parser = argparse.ArgumentParser()
    _configure_default_method_flags(parser)
    defaults = parser.parse_args([])
    assert all(getattr(defaults, flag) is True for flag in DEFAULT_ENABLED_METHOD_FLAGS)

    targeted = parser.parse_args(["--disable_kcal_factorial"])
    assert targeted.enable_kcal_factorial is False
    assert targeted.enable_kcal_baseline is True


def test_disable_all_optional_methods_overrides_defaults():
    parser = argparse.ArgumentParser()
    _configure_default_method_flags(parser)
    args = parser.parse_args(["--disable_all_optional_methods"])
    _apply_disable_all_optional_methods(args)
    assert all(getattr(args, flag) is False for flag in DEFAULT_ENABLED_METHOD_FLAGS)


def test_full_kcal_settings_change_resume_fingerprint():
    base = {
        "dataset": "cifar10",
        "model": "resnet18",
        "method": "baseline_cross_entropy",
        "seed": 11,
        "batch_size": 128,
        "target_dimension": 256,
        "num_layers": 6,
        "num_coordinates": 256,
        "enable_post_fusion_temperature": False,
        "enable_rgcl_tail_hybrids": False,
        "rgcl_tail_sources": "base",
        "enable_kcal_baseline": True,
        "kcal_projection": "skip_elu",
        "kcal_projection_dim": 32,
        "kcal_projection_epochs": 50,
        "kcal_references_per_class": 32,
        "kcal_bandwidth_folds": 20,
    }
    first = _args_fingerprint(SimpleNamespace(**base))
    second = _args_fingerprint(
        SimpleNamespace(**{**base, "kcal_projection_epochs": 25})
    )
    assert first != second


def test_factorial_settings_change_resume_fingerprint():
    base = {
        "dataset": "cifar100",
        "model": "resnet18",
        "method": "baseline_cross_entropy",
        "seed": 21,
        "batch_size": 128,
        "target_dimension": 256,
        "num_layers": 6,
        "num_coordinates": 256,
        "enable_post_fusion_temperature": False,
        "enable_rgcl_tail_hybrids": False,
        "rgcl_tail_sources": "base",
        "enable_kcal_baseline": False,
        "enable_kcal_factorial": True,
        "kcal_projection": "skip_elu",
        "kcal_projection_dim": 32,
        "kcal_projection_epochs": 50,
        "kcal_references_per_class": 32,
        "kcal_bandwidth_folds": 20,
        "kcal_factorial_scope": "core",
        "kcal_factorial_fusions": "replace,blend_frozen,blend_joint",
        "kcal_factorial_k_per_class": 5,
        "kcal_factorial_cv_folds": 5,
        "kcal_factorial_strength_grid": "0.1,1,10",
        "kcal_factorial_alpha_grid": "0,0.5,1",
    }
    first = _args_fingerprint(SimpleNamespace(**base))
    second = _args_fingerprint(
        SimpleNamespace(**{**base, "kcal_factorial_scope": "full"})
    )
    assert first != second
