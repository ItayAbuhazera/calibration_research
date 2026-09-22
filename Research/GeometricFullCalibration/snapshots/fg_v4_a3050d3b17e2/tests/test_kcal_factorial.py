"""Tests for controlled KCal/KCal-lite-RGCL factorial components."""

from __future__ import annotations

import numpy as np

from Calibrators.kcal_factorial import (
    PosteriorGrid,
    build_posterior_grid,
    effective_kernel_parameter,
    estimate_distance_normalizer,
    full_posterior_grid,
    select_factorial_outputs,
    topk_posterior_grid,
)


def _features(seed: int = 4):
    rng = np.random.RandomState(seed)
    train = np.vstack(
        [rng.normal(-2.0, 0.2, size=(8, 3)), rng.normal(2.0, 0.2, size=(8, 3))]
    ).astype(np.float32)
    validation = np.vstack(
        [rng.normal(-2.0, 0.2, size=(4, 3)), rng.normal(2.0, 0.2, size=(4, 3))]
    ).astype(np.float32)
    test = np.vstack(
        [rng.normal(-2.0, 0.2, size=(3, 3)), rng.normal(2.0, 0.2, size=(3, 3))]
    ).astype(np.float32)
    return train, np.repeat([0, 1], 8), validation, np.repeat([0, 1], 4), test


def test_full_grid_matches_manual_rbf_formula():
    queries = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    references = np.array(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 1.0], [1.5, 1.0]], dtype=np.float32
    )
    labels = np.array([0, 0, 1, 1])
    strengths = np.array([0.5, 2.0])
    actual = full_posterior_grid(
        queries,
        references,
        labels,
        strengths=strengths,
        kernel="rbf_sq",
        distance_normalizer=1.0,
        num_classes=2,
        device="cpu",
        query_batch_size=1,
        reference_batch_size=2,
    )
    squared_mean = ((queries[:, None] - references[None]) ** 2).mean(axis=2)
    for index, strength in enumerate(strengths):
        kernels = np.exp(-strength * squared_mean)
        scores = np.column_stack(
            [kernels[:, labels == class_id].sum(axis=1) for class_id in range(2)]
        )
        expected = scores / scores.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(actual[index], expected, rtol=1e-6, atol=1e-7)


def test_topk_all_neighbors_matches_full_for_balanced_classes():
    queries = np.array([[0.1, 0.0], [1.1, 1.0]], dtype=np.float32)
    references = np.array(
        [[0.0, 0.0], [0.5, 0.0], [1.0, 1.0], [1.5, 1.0]], dtype=np.float32
    )
    labels = np.array([0, 0, 1, 1])
    distances = np.empty((2, 2, 2), dtype=np.float32)
    for class_id in range(2):
        class_distances = np.linalg.norm(
            queries[:, None] - references[None, labels == class_id], axis=2
        )
        distances[:, class_id] = np.sort(class_distances, axis=1)
    strengths = [0.3, 1.0, 3.0]
    normalizer = 0.8
    topk = topk_posterior_grid(
        distances,
        strengths=strengths,
        kernel="exp_l2",
        distance_normalizer=normalizer,
        feature_dimension=2,
    )
    full = full_posterior_grid(
        queries,
        references,
        labels,
        strengths=strengths,
        kernel="exp_l2",
        distance_normalizer=normalizer,
        num_classes=2,
        device="cpu",
    )
    np.testing.assert_allclose(topk, full, rtol=1e-6, atol=1e-7)


def test_validation_bank_uses_crossfit_and_normalizes():
    train, y_train, validation, y_validation, test = _features()
    result = build_posterior_grid(
        train_features=train,
        train_labels=y_train,
        validation_features=validation,
        validation_labels=y_validation,
        test_features=test,
        estimator="topk",
        reference_bank="validation",
        kernel="rbf_sq",
        strengths=[0.3, 1.0],
        k_per_class=2,
        cv_folds=2,
        seed=9,
        device="cpu",
    )
    assert result.val_probs.shape == (2, len(validation), 2)
    assert result.test_probs.shape == (2, len(test), 2)
    assert result.folds_used == 2
    np.testing.assert_allclose(result.val_probs.sum(axis=2), 1.0, atol=1e-6)
    np.testing.assert_allclose(result.test_probs.sum(axis=2), 1.0, atol=1e-6)


def test_frozen_blend_reuses_replacement_strength_and_alpha_zero_is_exact():
    val_geometry = np.array(
        [
            [[0.8, 0.2], [0.2, 0.8]],
            [[0.7, 0.3], [0.3, 0.7]],
        ],
        dtype=np.float32,
    )
    test_geometry = val_geometry.copy()
    grid = PosteriorGrid(
        strengths=np.array([0.5, 2.0]),
        val_probs=val_geometry,
        test_probs=test_geometry,
        distance_normalizer=1.0,
        folds_used=2,
        estimator="full",
        reference_bank="validation",
        kernel="rbf_sq",
        k_per_class=None,
        reference_count=4,
    )
    labels = np.array([0, 1])
    base = np.full((2, 2), 0.5)
    outputs = select_factorial_outputs(
        grid,
        validation_labels=labels,
        validation_base_probs=base,
        test_base_probs=base,
        alpha_grid=[0.0],
    )
    np.testing.assert_allclose(
        outputs["replace"].test_probs, outputs["blend_frozen"].test_probs
    )
    assert (
        outputs["replace"].selected_strength
        == outputs["blend_frozen"].selected_strength
    )
    assert outputs["blend_frozen"].selected_alpha == 0.0


def test_distance_normalizer_and_effective_parameters_are_positive():
    train, _, _, _, _ = _features()
    rbf_scale = estimate_distance_normalizer(train, kernel="rbf_sq", seed=1)
    l2_scale = estimate_distance_normalizer(train, kernel="exp_l2", seed=1)
    assert rbf_scale > 0.0
    assert l2_scale > 0.0
    assert effective_kernel_parameter(
        kernel="rbf_sq", selected_strength=2.0, distance_normalizer=rbf_scale
    )["effective_bandwidth"] > 0.0
    assert effective_kernel_parameter(
        kernel="exp_l2", selected_strength=2.0, distance_normalizer=l2_scale
    )["effective_gamma"] > 0.0
