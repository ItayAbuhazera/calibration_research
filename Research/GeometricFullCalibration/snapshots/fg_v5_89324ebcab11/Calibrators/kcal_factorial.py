"""Controlled factorial variants spanning full KCal and KCal-lite-RGCL.

The benchmark endpoints differ in representation, fusion rule, density
estimator, reference bank, and kernel.  This module evaluates those axes with
one shared implementation.  It accepts *precomputed* embeddings so callers can
reuse a fitted KCal projection and the RGCL multi-layer representation.

All kernel strengths are dimensionless.  Distances are divided by a robust
median reference-bank distance before applying ``exp(-strength * distance)``.
That keeps one strength grid meaningful across learned-Pi and RGCL spaces while
preserving the usual bandwidth/gamma interpretation in the returned metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from utils.stability_space import StabilitySpace


VALID_ESTIMATORS = {"full", "topk"}
VALID_REFERENCE_BANKS = {"validation", "train"}
VALID_KERNELS = {"rbf_sq", "exp_l2"}
VALID_FUSIONS = {"replace", "blend_frozen", "blend_joint"}


@dataclass
class PosteriorGrid:
    """Validation/test geometry posteriors for a shared strength grid."""

    strengths: np.ndarray
    val_probs: np.ndarray
    test_probs: np.ndarray
    distance_normalizer: float
    folds_used: int
    estimator: str
    reference_bank: str
    kernel: str
    k_per_class: int | None
    reference_count: int


@dataclass
class SelectedFactorialOutput:
    """One fusion variant selected only with validation labels."""

    fusion: str
    test_probs: np.ndarray
    selected_strength: float
    selected_strength_index: int
    selected_alpha: float
    validation_nll: float
    validation_nll_grid: np.ndarray
    geometry_strength_source: str


def _as_2d_float32(name: str, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a non-empty [N, D] array")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains non-finite values")
    return values


def _as_labels(name: str, labels: np.ndarray, n_rows: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != n_rows:
        raise ValueError(f"{name} must align with its feature array")
    if np.any(labels < 0):
        raise ValueError(f"{name} must contain non-negative class indices")
    return labels


def _validate_strengths(strengths: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(strengths), dtype=np.float64)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError("strengths must contain at least one value")
    if np.any(~np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError("strengths must be finite and strictly positive")
    return result


def _validate_alpha_grid(alpha_grid: Iterable[float]) -> np.ndarray:
    result = np.asarray(list(alpha_grid), dtype=np.float64)
    if result.ndim != 1 or len(result) == 0:
        raise ValueError("alpha_grid must contain at least one value")
    if np.any(~np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError("alpha_grid values must lie in [0, 1]")
    return result


def _multiclass_nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    return float(
        -np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], eps, 1.0)))
    )


def estimate_distance_normalizer(
    references: np.ndarray,
    *,
    kernel: str,
    seed: int,
    max_pairs: int = 4096,
) -> float:
    """Estimate a robust, label-free reference-bank distance scale."""
    if kernel not in VALID_KERNELS:
        raise ValueError(f"kernel must be one of {sorted(VALID_KERNELS)}")
    references = _as_2d_float32("references", references)
    if len(references) < 2:
        return 1.0

    rng = np.random.RandomState(seed)
    n_pairs = min(int(max_pairs), max(len(references), 2))
    left = rng.randint(0, len(references), size=n_pairs)
    right = rng.randint(0, len(references), size=n_pairs)
    equal = left == right
    right[equal] = (right[equal] + 1) % len(references)
    diff = references[left].astype(np.float64) - references[right].astype(np.float64)
    squared = np.sum(diff * diff, axis=1)
    if kernel == "rbf_sq":
        raw = squared / float(references.shape[1])
    else:
        raw = np.sqrt(np.maximum(squared, 0.0))
    positive = raw[np.isfinite(raw) & (raw > 0.0)]
    if len(positive) == 0:
        return 1.0
    return float(max(np.median(positive), 1e-12))


def _normalized_distance_torch(
    queries: torch.Tensor,
    references: torch.Tensor,
    *,
    kernel: str,
    normalizer: float,
) -> torch.Tensor:
    query_sq = torch.sum(queries * queries, dim=1, keepdim=True)
    reference_sq = torch.sum(references * references, dim=1).unsqueeze(0)
    squared = torch.clamp(query_sq + reference_sq - 2.0 * queries @ references.T, min=0.0)
    if kernel == "rbf_sq":
        raw = squared / float(queries.shape[1])
    else:
        raw = torch.sqrt(squared)
    return raw / float(normalizer)


def full_posterior_grid(
    queries: np.ndarray,
    references: np.ndarray,
    reference_labels: np.ndarray,
    *,
    strengths: Iterable[float],
    kernel: str,
    distance_normalizer: float,
    num_classes: int,
    device: str | torch.device = "cpu",
    query_batch_size: int = 256,
    reference_batch_size: int = 2048,
) -> np.ndarray:
    """Compute full-reference KDE posteriors for every kernel strength."""
    if kernel not in VALID_KERNELS:
        raise ValueError(f"kernel must be one of {sorted(VALID_KERNELS)}")
    queries = _as_2d_float32("queries", queries)
    references = _as_2d_float32("references", references)
    if queries.shape[1] != references.shape[1]:
        raise ValueError("queries and references must have the same dimension")
    reference_labels = _as_labels("reference_labels", reference_labels, len(references))
    strengths = _validate_strengths(strengths)
    if distance_normalizer <= 0.0 or not np.isfinite(distance_normalizer):
        raise ValueError("distance_normalizer must be finite and positive")
    if query_batch_size < 1 or reference_batch_size < 1:
        raise ValueError("batch sizes must be positive")

    target_device = torch.device(device)
    outputs = np.empty(
        (len(strengths), len(queries), int(num_classes)), dtype=np.float32
    )
    strengths_t = torch.as_tensor(strengths, dtype=torch.float32, device=target_device)

    with torch.no_grad():
        for query_start in range(0, len(queries), query_batch_size):
            query_stop = min(query_start + query_batch_size, len(queries))
            query_t = torch.as_tensor(
                queries[query_start:query_stop], dtype=torch.float32, device=target_device
            )
            scores = torch.zeros(
                (len(strengths), len(query_t), num_classes),
                dtype=torch.float32,
                device=target_device,
            )
            for reference_start in range(0, len(references), reference_batch_size):
                reference_stop = min(
                    reference_start + reference_batch_size, len(references)
                )
                reference_t = torch.as_tensor(
                    references[reference_start:reference_stop],
                    dtype=torch.float32,
                    device=target_device,
                )
                label_t = torch.as_tensor(
                    reference_labels[reference_start:reference_stop],
                    dtype=torch.long,
                    device=target_device,
                )
                normalized_distance = _normalized_distance_torch(
                    query_t,
                    reference_t,
                    kernel=kernel,
                    normalizer=distance_normalizer,
                )
                one_hot = F.one_hot(label_t, num_classes=num_classes).to(torch.float32)
                for strength_index, strength in enumerate(strengths_t):
                    kernels = torch.exp(-strength * normalized_distance)
                    scores[strength_index] += kernels @ one_hot

            denominators = scores.sum(dim=2, keepdim=True)
            uniform = torch.full_like(scores, 1.0 / float(num_classes))
            probs = torch.where(
                denominators > 1e-12,
                scores / denominators.clamp_min(1e-12),
                uniform,
            )
            outputs[:, query_start:query_stop, :] = probs.cpu().numpy()
    return outputs


def topk_posterior_grid(
    knn_distances: np.ndarray,
    *,
    strengths: Iterable[float],
    kernel: str,
    distance_normalizer: float,
    feature_dimension: int,
) -> np.ndarray:
    """Compute classwise top-k KDE posteriors for every kernel strength."""
    if kernel not in VALID_KERNELS:
        raise ValueError(f"kernel must be one of {sorted(VALID_KERNELS)}")
    distances = np.asarray(knn_distances, dtype=np.float64)
    if distances.ndim != 3 or distances.shape[2] < 1:
        raise ValueError("knn_distances must have shape [N, C, k]")
    if np.any(~np.isfinite(distances)) or np.any(distances < 0.0):
        raise ValueError("knn_distances must be finite and non-negative")
    strengths = _validate_strengths(strengths)
    if distance_normalizer <= 0.0 or not np.isfinite(distance_normalizer):
        raise ValueError("distance_normalizer must be finite and positive")
    if feature_dimension < 1:
        raise ValueError("feature_dimension must be positive")

    if kernel == "rbf_sq":
        normalized = (distances * distances / float(feature_dimension)) / float(
            distance_normalizer
        )
    else:
        normalized = distances / float(distance_normalizer)

    outputs = np.empty(
        (len(strengths), distances.shape[0], distances.shape[1]), dtype=np.float64
    )
    for index, strength in enumerate(strengths):
        affinities = np.exp(-float(strength) * normalized).sum(axis=2)
        row_sums = affinities.sum(axis=1, keepdims=True)
        outputs[index] = np.divide(
            affinities,
            np.clip(row_sums, 1e-300, None),
            out=np.full_like(affinities, 1.0 / float(distances.shape[1])),
            where=row_sums > 1e-300,
        )
    return outputs.astype(np.float32)


def _posterior_grid_for_queries(
    queries: np.ndarray,
    references: np.ndarray,
    reference_labels: np.ndarray,
    *,
    estimator: str,
    kernel: str,
    strengths: np.ndarray,
    distance_normalizer: float,
    num_classes: int,
    k_per_class: int,
    device: str | torch.device,
    query_batch_size: int,
    reference_batch_size: int,
) -> np.ndarray:
    if estimator == "full":
        return full_posterior_grid(
            queries,
            references,
            reference_labels,
            strengths=strengths,
            kernel=kernel,
            distance_normalizer=distance_normalizer,
            num_classes=num_classes,
            device=device,
            query_batch_size=query_batch_size,
            reference_batch_size=reference_batch_size,
        )

    stability_space = StabilitySpace(
        references,
        reference_labels,
        library="fast_separation",
        metric="l2",
        num_labels=num_classes,
    )
    distances = stability_space.calc_per_class_knn_distances(queries, k_per_class)
    return topk_posterior_grid(
        distances,
        strengths=strengths,
        kernel=kernel,
        distance_normalizer=distance_normalizer,
        feature_dimension=references.shape[1],
    )


def build_posterior_grid(
    *,
    train_features: np.ndarray,
    train_labels: np.ndarray,
    validation_features: np.ndarray,
    validation_labels: np.ndarray,
    test_features: np.ndarray,
    estimator: str,
    reference_bank: str,
    kernel: str,
    strengths: Iterable[float],
    k_per_class: int,
    cv_folds: int,
    seed: int,
    device: str | torch.device = "cpu",
    query_batch_size: int = 256,
    reference_batch_size: int = 2048,
) -> PosteriorGrid:
    """Build aligned validation/test posterior grids for one geometry cell.

    A training reference bank predicts validation examples directly.  A
    validation reference bank uses stratified cross-fitting for validation
    predictions and all validation examples for test prediction, avoiding the
    zero-distance self-neighbor leak.
    """
    if estimator not in VALID_ESTIMATORS:
        raise ValueError(f"estimator must be one of {sorted(VALID_ESTIMATORS)}")
    if reference_bank not in VALID_REFERENCE_BANKS:
        raise ValueError(
            f"reference_bank must be one of {sorted(VALID_REFERENCE_BANKS)}"
        )
    if kernel not in VALID_KERNELS:
        raise ValueError(f"kernel must be one of {sorted(VALID_KERNELS)}")
    if k_per_class < 1:
        raise ValueError("k_per_class must be positive")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least two")

    train_features = _as_2d_float32("train_features", train_features)
    validation_features = _as_2d_float32(
        "validation_features", validation_features
    )
    test_features = _as_2d_float32("test_features", test_features)
    if not (
        train_features.shape[1]
        == validation_features.shape[1]
        == test_features.shape[1]
    ):
        raise ValueError("all feature arrays must have the same dimension")
    train_labels = _as_labels("train_labels", train_labels, len(train_features))
    validation_labels = _as_labels(
        "validation_labels", validation_labels, len(validation_features)
    )
    strengths = _validate_strengths(strengths)
    classes = np.unique(np.concatenate([train_labels, validation_labels]))
    expected_classes = np.arange(int(classes.max()) + 1)
    if not np.array_equal(classes, expected_classes):
        raise ValueError("class indices must be contiguous and start at zero")
    num_classes = len(expected_classes)

    bank_features = train_features if reference_bank == "train" else validation_features
    bank_labels = train_labels if reference_bank == "train" else validation_labels
    class_counts = np.bincount(bank_labels, minlength=num_classes)
    if estimator == "topk" and int(class_counts.min()) < int(k_per_class):
        raise ValueError(
            f"k_per_class={k_per_class} exceeds the smallest {reference_bank} "
            f"bank class count {int(class_counts.min())}"
        )
    distance_normalizer = estimate_distance_normalizer(
        bank_features, kernel=kernel, seed=seed
    )

    if reference_bank == "train":
        val_probs = _posterior_grid_for_queries(
            validation_features,
            train_features,
            train_labels,
            estimator=estimator,
            kernel=kernel,
            strengths=strengths,
            distance_normalizer=distance_normalizer,
            num_classes=num_classes,
            k_per_class=k_per_class,
            device=device,
            query_batch_size=query_batch_size,
            reference_batch_size=reference_batch_size,
        )
        folds_used = 0
    else:
        validation_counts = np.bincount(validation_labels, minlength=num_classes)
        folds_used = min(int(cv_folds), int(validation_counts.min()))
        if folds_used < 2:
            raise ValueError(
                "validation-bank cross-fitting requires two examples per class"
            )
        splitter = StratifiedKFold(
            n_splits=folds_used, shuffle=True, random_state=seed
        )
        val_probs = np.empty(
            (len(strengths), len(validation_features), num_classes), dtype=np.float32
        )
        for reference_indices, query_indices in splitter.split(
            validation_features, validation_labels
        ):
            fold_counts = np.bincount(
                validation_labels[reference_indices], minlength=num_classes
            )
            if estimator == "topk" and int(fold_counts.min()) < int(k_per_class):
                raise ValueError(
                    f"k_per_class={k_per_class} exceeds the smallest cross-fit "
                    f"reference class count {int(fold_counts.min())}"
                )
            val_probs[:, query_indices, :] = _posterior_grid_for_queries(
                validation_features[query_indices],
                validation_features[reference_indices],
                validation_labels[reference_indices],
                estimator=estimator,
                kernel=kernel,
                strengths=strengths,
                distance_normalizer=distance_normalizer,
                num_classes=num_classes,
                k_per_class=k_per_class,
                device=device,
                query_batch_size=query_batch_size,
                reference_batch_size=reference_batch_size,
            )

    test_probs = _posterior_grid_for_queries(
        test_features,
        bank_features,
        bank_labels,
        estimator=estimator,
        kernel=kernel,
        strengths=strengths,
        distance_normalizer=distance_normalizer,
        num_classes=num_classes,
        k_per_class=k_per_class,
        device=device,
        query_batch_size=query_batch_size,
        reference_batch_size=reference_batch_size,
    )
    return PosteriorGrid(
        strengths=strengths,
        val_probs=val_probs,
        test_probs=test_probs,
        distance_normalizer=float(distance_normalizer),
        folds_used=int(folds_used),
        estimator=estimator,
        reference_bank=reference_bank,
        kernel=kernel,
        k_per_class=int(k_per_class) if estimator == "topk" else None,
        reference_count=int(len(bank_features)),
    )


def select_factorial_outputs(
    posterior_grid: PosteriorGrid,
    *,
    validation_labels: np.ndarray,
    validation_base_probs: np.ndarray,
    test_base_probs: np.ndarray,
    alpha_grid: Iterable[float],
) -> Dict[str, SelectedFactorialOutput]:
    """Select replacement, frozen blend, and joint blend on validation NLL."""
    labels = _as_labels(
        "validation_labels", validation_labels, posterior_grid.val_probs.shape[1]
    )
    validation_base_probs = np.asarray(validation_base_probs, dtype=np.float64)
    test_base_probs = np.asarray(test_base_probs, dtype=np.float64)
    if validation_base_probs.shape != posterior_grid.val_probs.shape[1:]:
        raise ValueError("validation_base_probs shape does not match posterior grid")
    if test_base_probs.shape != posterior_grid.test_probs.shape[1:]:
        raise ValueError("test_base_probs shape does not match posterior grid")
    alpha_grid = _validate_alpha_grid(alpha_grid)

    replacement_nll = np.asarray(
        [_multiclass_nll(probs, labels) for probs in posterior_grid.val_probs],
        dtype=np.float64,
    )
    replacement_index = int(np.argmin(replacement_nll))
    replacement_test = posterior_grid.test_probs[replacement_index].astype(np.float64)

    frozen_nll = np.empty(len(alpha_grid), dtype=np.float64)
    frozen_val_geometry = posterior_grid.val_probs[replacement_index]
    for alpha_index, alpha in enumerate(alpha_grid):
        blended = (
            float(alpha) * validation_base_probs
            + (1.0 - float(alpha)) * frozen_val_geometry
        )
        frozen_nll[alpha_index] = _multiclass_nll(blended, labels)
    frozen_alpha_index = int(np.argmin(frozen_nll))
    frozen_alpha = float(alpha_grid[frozen_alpha_index])
    frozen_test = (
        frozen_alpha * test_base_probs
        + (1.0 - frozen_alpha) * posterior_grid.test_probs[replacement_index]
    )

    joint_nll = np.empty(
        (len(posterior_grid.strengths), len(alpha_grid)), dtype=np.float64
    )
    for strength_index, geometry_probs in enumerate(posterior_grid.val_probs):
        for alpha_index, alpha in enumerate(alpha_grid):
            blended = (
                float(alpha) * validation_base_probs
                + (1.0 - float(alpha)) * geometry_probs
            )
            joint_nll[strength_index, alpha_index] = _multiclass_nll(
                blended, labels
            )
    joint_flat_index = int(np.argmin(joint_nll))
    joint_strength_index, joint_alpha_index = np.unravel_index(
        joint_flat_index, joint_nll.shape
    )
    joint_alpha = float(alpha_grid[joint_alpha_index])
    joint_test = (
        joint_alpha * test_base_probs
        + (1.0 - joint_alpha) * posterior_grid.test_probs[joint_strength_index]
    )

    return {
        "replace": SelectedFactorialOutput(
            fusion="replace",
            test_probs=replacement_test,
            selected_strength=float(posterior_grid.strengths[replacement_index]),
            selected_strength_index=replacement_index,
            selected_alpha=0.0,
            validation_nll=float(replacement_nll[replacement_index]),
            validation_nll_grid=replacement_nll,
            geometry_strength_source="replacement_validation_nll",
        ),
        "blend_frozen": SelectedFactorialOutput(
            fusion="blend_frozen",
            test_probs=frozen_test,
            selected_strength=float(posterior_grid.strengths[replacement_index]),
            selected_strength_index=replacement_index,
            selected_alpha=frozen_alpha,
            validation_nll=float(frozen_nll[frozen_alpha_index]),
            validation_nll_grid=frozen_nll,
            geometry_strength_source="frozen_from_replacement",
        ),
        "blend_joint": SelectedFactorialOutput(
            fusion="blend_joint",
            test_probs=joint_test,
            selected_strength=float(
                posterior_grid.strengths[joint_strength_index]
            ),
            selected_strength_index=int(joint_strength_index),
            selected_alpha=joint_alpha,
            validation_nll=float(joint_nll[joint_strength_index, joint_alpha_index]),
            validation_nll_grid=joint_nll,
            geometry_strength_source="joint_alpha_strength_validation_nll",
        ),
    }


def effective_kernel_parameter(
    *, kernel: str, selected_strength: float, distance_normalizer: float
) -> dict[str, float]:
    """Return the raw-space RBF bandwidth or exponential-L2 gamma."""
    if kernel == "rbf_sq":
        return {
            "effective_bandwidth": float(distance_normalizer / selected_strength)
        }
    if kernel == "exp_l2":
        return {"effective_gamma": float(selected_strength / distance_normalizer)}
    raise ValueError(f"kernel must be one of {sorted(VALID_KERNELS)}")
