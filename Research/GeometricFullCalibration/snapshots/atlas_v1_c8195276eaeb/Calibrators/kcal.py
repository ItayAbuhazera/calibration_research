"""Full KCal kernel-based calibration.

This module implements the core method from Lin, Trivedi, and Sun (ICLR 2023):
a learned low-dimensional projection followed by an RBF kernel-density classifier
whose bandwidth is selected by cross-validated log loss.  Unlike KCal-lite, full
KCal uses every calibration reference point and does not blend its posterior with
the base model probabilities.
"""

from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold


class BNLinearProjection(nn.Module):
    """Batch-normalized linear projection used by the KCal ablation."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.fc = nn.Linear(input_dim, output_dim, bias=False)
        with torch.no_grad():
            self.fc.weight.div_(max(output_dim / 2.0, 1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.bn(x))


class SkipELUProjection(nn.Module):
    """KCal's nonlinear projection: BN, linear map, and an ELU residual block."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        self.input_map = nn.Linear(input_dim, output_dim)
        self.residual = nn.Linear(output_dim, output_dim, bias=False)
        self.activation = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.input_map(self.bn(x))
        return hidden + self.residual(self.activation(hidden))


@dataclass(frozen=True)
class KCalConfig:
    projection_type: str = "skip_elu"
    projection_dim: Optional[int] = None
    projection_epochs: int = 50
    query_batch_size: int = 64
    references_per_class: int = 32
    learning_rate: float = 4e-4
    weight_decay: float = 1e-4
    bandwidth_folds: int = 20
    bandwidth_lower: float = 0.1
    bandwidth_upper: float = 10.0
    bandwidth_tolerance: float = 1e-3
    prediction_batch_size: int = 256
    reference_batch_size: int = 4096
    seed: int = 42


class KCalCalibrator:
    """Learned-projection, full-reference KDE calibrator.

    ``fit`` uses the training embeddings only to learn the projection.  It then
    selects the kernel bandwidth by cross-validation on the calibration split and
    stores that complete split as the final KDE reference set.  Test labels are
    never accepted by this API.
    """

    VALID_PROJECTIONS = {"skip_elu", "bn_linear"}

    def __init__(
        self,
        *,
        projection_type: str = "skip_elu",
        projection_dim: Optional[int] = None,
        projection_epochs: int = 50,
        query_batch_size: int = 64,
        references_per_class: int = 32,
        learning_rate: float = 4e-4,
        weight_decay: float = 1e-4,
        bandwidth_folds: int = 20,
        bandwidth_lower: float = 0.1,
        bandwidth_upper: float = 10.0,
        bandwidth_tolerance: float = 1e-3,
        prediction_batch_size: int = 256,
        reference_batch_size: int = 4096,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        projection_type = projection_type.lower()
        if projection_type not in self.VALID_PROJECTIONS:
            raise ValueError(
                f"projection_type must be one of {sorted(self.VALID_PROJECTIONS)}"
            )
        if projection_dim is not None and projection_dim < 1:
            raise ValueError("projection_dim must be positive when provided")
        if projection_epochs < 1:
            raise ValueError("projection_epochs must be positive")
        if query_batch_size < 1 or references_per_class < 1:
            raise ValueError("query_batch_size and references_per_class must be positive")
        if bandwidth_folds < 2:
            raise ValueError("bandwidth_folds must be at least 2")
        if not 0.0 < bandwidth_lower < bandwidth_upper:
            raise ValueError("bandwidth bounds must satisfy 0 < lower < upper")
        if prediction_batch_size < 1 or reference_batch_size < 1:
            raise ValueError("prediction and reference batch sizes must be positive")

        self.config = KCalConfig(
            projection_type=projection_type,
            projection_dim=projection_dim,
            projection_epochs=int(projection_epochs),
            query_batch_size=int(query_batch_size),
            references_per_class=int(references_per_class),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            bandwidth_folds=int(bandwidth_folds),
            bandwidth_lower=float(bandwidth_lower),
            bandwidth_upper=float(bandwidth_upper),
            bandwidth_tolerance=float(bandwidth_tolerance),
            prediction_batch_size=int(prediction_batch_size),
            reference_batch_size=int(reference_batch_size),
            seed=int(seed),
        )
        self.device = torch.device(device)
        self.projection: Optional[nn.Module] = None
        self.input_dim: Optional[int] = None
        self.output_dim: Optional[int] = None
        self.num_classes: Optional[int] = None
        self.bandwidth: Optional[float] = None
        self.bandwidth_folds_used: Optional[int] = None
        self.bandwidth_search_history: list[Dict[str, float]] = []
        self.projection_loss_history: list[float] = []
        self.calibration_projected: Optional[np.ndarray] = None
        self.calibration_labels: Optional[np.ndarray] = None
        self.is_fitted = False

    @staticmethod
    def _validate_embeddings(name: str, values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
            raise ValueError(f"{name} must be a non-empty [N, D] array")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")
        return values

    @staticmethod
    def _validate_labels(name: str, labels: np.ndarray, n_rows: int) -> np.ndarray:
        labels = np.asarray(labels, dtype=np.int64)
        if labels.ndim != 1 or len(labels) != n_rows:
            raise ValueError(f"{name} must be a [N] array aligned to embeddings")
        if np.any(labels < 0):
            raise ValueError(f"{name} must contain non-negative class indices")
        return labels

    def _build_projection(self, input_dim: int) -> nn.Module:
        # Default: match projection dim to num_classes so the KDE space has one
        # dimension per class.  This is more principled than the paper's fixed d=32
        # which was tuned for ViT-base (768-dim) on CIFAR-100, not for smaller models.
        output_dim = self.config.projection_dim or min(input_dim, self.num_classes or 32)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        if self.config.projection_type == "skip_elu":
            module = SkipELUProjection(input_dim, output_dim)
        else:
            module = BNLinearProjection(input_dim, output_dim)
        return module.to(self.device)

    @staticmethod
    def _squared_mean_distances(
        queries: torch.Tensor, references: torch.Tensor
    ) -> torch.Tensor:
        query_norm = torch.sum(queries * queries, dim=1, keepdim=True)
        ref_norm = torch.sum(references * references, dim=1).unsqueeze(0)
        squared = query_norm + ref_norm - 2.0 * (queries @ references.T)
        return torch.clamp(squared, min=0.0) / float(queries.shape[1])

    @classmethod
    def _kernel_posterior_torch(
        cls,
        queries: torch.Tensor,
        references: torch.Tensor,
        reference_labels: torch.Tensor,
        *,
        bandwidth: float,
        num_classes: int,
        reference_weights: Optional[torch.Tensor] = None,
        reference_batch_size: int = 4096,
        eps: float = 1e-12,
    ) -> torch.Tensor:
        if bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive")
        scores = torch.zeros(
            (queries.shape[0], num_classes),
            dtype=queries.dtype,
            device=queries.device,
        )
        for start in range(0, len(references), reference_batch_size):
            stop = min(start + reference_batch_size, len(references))
            ref_chunk = references[start:stop]
            label_chunk = reference_labels[start:stop]
            distances = cls._squared_mean_distances(queries, ref_chunk)
            kernels = torch.exp(-distances / float(bandwidth))
            one_hot = F.one_hot(label_chunk, num_classes=num_classes).to(queries.dtype)
            if reference_weights is not None:
                one_hot = one_hot * reference_weights[start:stop].unsqueeze(1)
            scores = scores + kernels @ one_hot
        row_sums = scores.sum(dim=1, keepdim=True)
        uniform = torch.full_like(scores, 1.0 / float(num_classes))
        return torch.where(row_sums > eps, scores / row_sums.clamp_min(eps), uniform)

    def _sample_training_batch(
        self,
        labels: np.ndarray,
        class_indices: list[np.ndarray],
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_queries = max(1, len(labels) - len(class_indices))
        query_count = min(self.config.query_batch_size, max_queries)
        for _ in range(20):
            query_indices = rng.choice(len(labels), size=query_count, replace=False)
            query_set = set(int(i) for i in query_indices)
            reference_parts = []
            reference_weights = []
            valid = True
            for indices in class_indices:
                available = np.asarray(
                    [int(i) for i in indices if int(i) not in query_set],
                    dtype=np.int64,
                )
                if len(available) == 0:
                    valid = False
                    break
                sample_count = min(self.config.references_per_class, len(available))
                selected = rng.choice(available, size=sample_count, replace=False)
                reference_parts.append(np.asarray(selected, dtype=np.int64))
                reference_weights.append(
                    np.full(sample_count, len(indices) / float(sample_count), dtype=np.float32)
                )
            if valid:
                return (
                    np.asarray(query_indices, dtype=np.int64),
                    np.concatenate(reference_parts),
                    np.concatenate(reference_weights),
                )
        raise RuntimeError("Unable to sample disjoint KCal query/reference batches")

    def _fit_projection(self, embeddings: np.ndarray, labels: np.ndarray) -> None:
        if self.projection is None or self.num_classes is None:
            raise RuntimeError("Projection and class count must be initialized")
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        rng = np.random.RandomState(self.config.seed)
        class_indices = [np.flatnonzero(labels == c) for c in range(self.num_classes)]
        if any(len(indices) < 2 for indices in class_indices):
            raise ValueError("Projection training requires at least two samples per class")

        optimizer = torch.optim.SGD(
            self.projection.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        steps_per_epoch = max(1, math.ceil(len(embeddings) / self.config.query_batch_size))
        self.projection.train()
        for _epoch in range(self.config.projection_epochs):
            epoch_losses = []
            for _step in range(steps_per_epoch):
                query_idx, reference_idx, reference_weights = self._sample_training_batch(
                    labels, class_indices, rng
                )
                combined_idx = np.concatenate([query_idx, reference_idx])
                combined = torch.as_tensor(
                    embeddings[combined_idx], dtype=torch.float32, device=self.device
                )
                projected = self.projection(combined)
                query_projected = projected[: len(query_idx)]
                reference_projected = projected[len(query_idx) :]
                reference_y = torch.as_tensor(
                    labels[reference_idx], dtype=torch.long, device=self.device
                )
                weights = torch.as_tensor(
                    reference_weights, dtype=torch.float32, device=self.device
                )
                posterior = self._kernel_posterior_torch(
                    query_projected,
                    reference_projected,
                    reference_y,
                    bandwidth=1.0,
                    num_classes=self.num_classes,
                    reference_weights=weights,
                    reference_batch_size=self.config.reference_batch_size,
                )
                query_y = torch.as_tensor(
                    labels[query_idx], dtype=torch.long, device=self.device
                )
                loss = F.nll_loss(torch.log(posterior.clamp_min(1e-12)), query_y)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu()))
            self.projection_loss_history.append(float(np.mean(epoch_losses)))
        self.projection.eval()

    def _project_numpy(self, embeddings: np.ndarray) -> np.ndarray:
        if self.projection is None:
            raise RuntimeError("Projection has not been initialized")
        outputs = []
        self.projection.eval()
        with torch.no_grad():
            for start in range(0, len(embeddings), self.config.prediction_batch_size):
                stop = min(start + self.config.prediction_batch_size, len(embeddings))
                batch = torch.as_tensor(
                    embeddings[start:stop], dtype=torch.float32, device=self.device
                )
                outputs.append(self.projection(batch).cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def _predict_projected(
        self,
        queries: np.ndarray,
        references: np.ndarray,
        reference_labels: np.ndarray,
        bandwidth: float,
    ) -> np.ndarray:
        if self.num_classes is None:
            raise RuntimeError("Class count has not been initialized")
        references_t = torch.as_tensor(references, dtype=torch.float32, device=self.device)
        labels_t = torch.as_tensor(reference_labels, dtype=torch.long, device=self.device)
        outputs = []
        with torch.no_grad():
            for start in range(0, len(queries), self.config.prediction_batch_size):
                stop = min(start + self.config.prediction_batch_size, len(queries))
                query_t = torch.as_tensor(
                    queries[start:stop], dtype=torch.float32, device=self.device
                )
                probs = self._kernel_posterior_torch(
                    query_t,
                    references_t,
                    labels_t,
                    bandwidth=bandwidth,
                    num_classes=self.num_classes,
                    reference_batch_size=self.config.reference_batch_size,
                )
                outputs.append(probs.cpu().numpy().astype(np.float64))
        return np.concatenate(outputs, axis=0)

    @staticmethod
    def _nll(probs: np.ndarray, labels: np.ndarray) -> float:
        return float(
            -np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)))
        )

    def _bandwidth_cv_loss(
        self,
        bandwidth: float,
        projected: np.ndarray,
        labels: np.ndarray,
        folds: list[tuple[np.ndarray, np.ndarray]],
    ) -> float:
        fold_losses = []
        fold_sizes = []
        for reference_idx, query_idx in folds:
            probs = self._predict_projected(
                projected[query_idx],
                projected[reference_idx],
                labels[reference_idx],
                bandwidth,
            )
            fold_losses.append(self._nll(probs, labels[query_idx]))
            fold_sizes.append(len(query_idx))
        return float(np.average(fold_losses, weights=fold_sizes))

    def _select_bandwidth(self, projected: np.ndarray, labels: np.ndarray) -> float:
        class_counts = np.bincount(labels, minlength=self.num_classes)
        folds_used = min(self.config.bandwidth_folds, int(class_counts.min()))
        if folds_used < 2:
            raise ValueError("Bandwidth selection requires at least two calibration samples per class")
        splitter = StratifiedKFold(
            n_splits=folds_used, shuffle=True, random_state=self.config.seed
        )
        folds = list(splitter.split(projected, labels))
        self.bandwidth_folds_used = int(folds_used)
        self.bandwidth_search_history = []
        cache: Dict[float, float] = {}

        def evaluate(value: float) -> float:
            key = float(np.round(value, 10))
            if key not in cache:
                cache[key] = self._bandwidth_cv_loss(key, projected, labels, folds)
                self.bandwidth_search_history.append(
                    {"bandwidth": key, "nll": float(cache[key])}
                )
            return cache[key]

        lower = self.config.bandwidth_lower
        upper = self.config.bandwidth_upper
        golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
        left = upper - (upper - lower) / golden_ratio
        right = lower + (upper - lower) / golden_ratio
        left_loss = evaluate(left)
        right_loss = evaluate(right)
        while upper - lower > self.config.bandwidth_tolerance:
            if left_loss < right_loss:
                upper = right
                right = left
                right_loss = left_loss
                left = upper - (upper - lower) / golden_ratio
                left_loss = evaluate(left)
            else:
                lower = left
                left = right
                left_loss = right_loss
                right = lower + (upper - lower) / golden_ratio
                right_loss = evaluate(right)
        candidates = [lower, left, right, upper, (lower + upper) / 2.0]
        return float(min(candidates, key=evaluate))

    def fit(
        self,
        X_train_embed: np.ndarray,
        y_train: np.ndarray,
        X_cal_embed: np.ndarray,
        y_cal: np.ndarray,
    ) -> "KCalCalibrator":
        train = self._validate_embeddings("X_train_embed", X_train_embed)
        cal = self._validate_embeddings("X_cal_embed", X_cal_embed)
        if train.shape[1] != cal.shape[1]:
            raise ValueError("Training and calibration embeddings must have the same dimension")
        train_y = self._validate_labels("y_train", y_train, len(train))
        cal_y = self._validate_labels("y_cal", y_cal, len(cal))
        classes = np.unique(np.concatenate([train_y, cal_y]))
        expected = np.arange(int(classes.max()) + 1)
        if not np.array_equal(classes, expected):
            raise ValueError("KCal requires contiguous class indices starting at zero")
        self.num_classes = int(len(classes))
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        self.projection = self._build_projection(train.shape[1])
        self._fit_projection(train, train_y)
        projected_cal = self._project_numpy(cal)
        self.bandwidth = self._select_bandwidth(projected_cal, cal_y)
        self.calibration_projected = projected_cal
        self.calibration_labels = cal_y.copy()
        self.is_fitted = True
        return self

    def calibrate(self, X_test_embed: np.ndarray) -> np.ndarray:
        if (
            not self.is_fitted
            or self.bandwidth is None
            or self.calibration_projected is None
            or self.calibration_labels is None
        ):
            raise ValueError("Call fit() before calibrate()")
        projected_test = self.transform(X_test_embed)
        return self._predict_projected(
            projected_test,
            self.calibration_projected,
            self.calibration_labels,
            self.bandwidth,
        )

    def transform(self, X_embed: np.ndarray) -> np.ndarray:
        """Project embeddings into the fitted KCal metric space.

        This public transform is also used by controlled ablations that keep
        KCal's learned representation while changing the reference bank,
        neighborhood estimator, kernel, or fusion rule.
        """
        if not self.is_fitted or self.projection is None:
            raise ValueError("Call fit() before transform()")
        values = self._validate_embeddings("X_embed", X_embed)
        if values.shape[1] != self.input_dim:
            raise ValueError("Embedding dimension does not match fitted projection")
        return self._project_numpy(values)

    def save_checkpoint(self, path: str) -> None:
        if not self.is_fitted or self.projection is None:
            raise ValueError("Cannot save an unfitted KCal calibrator")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "config": asdict(self.config),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_classes": self.num_classes,
            "bandwidth": self.bandwidth,
            "bandwidth_folds_used": self.bandwidth_folds_used,
            "bandwidth_search_history": self.bandwidth_search_history,
            "projection_loss_history": self.projection_loss_history,
            "projection_state_dict": {
                key: value.detach().cpu() for key, value in self.projection.state_dict().items()
            },
            "calibration_projected": self.calibration_projected,
            "calibration_labels": self.calibration_labels,
        }
        tmp = f"{path}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, path)

    @classmethod
    def load_checkpoint(
        cls, path: str, *, device: str | torch.device = "cpu"
    ) -> "KCalCalibrator":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch versions before the weights_only argument.
            payload = torch.load(path, map_location="cpu")
        calibrator = cls(**payload["config"], device=device)
        calibrator.num_classes = int(payload["num_classes"])
        calibrator.projection = calibrator._build_projection(int(payload["input_dim"]))
        calibrator.projection.load_state_dict(payload["projection_state_dict"])
        calibrator.projection.eval()
        calibrator.output_dim = int(payload["output_dim"])
        calibrator.bandwidth = float(payload["bandwidth"])
        calibrator.bandwidth_folds_used = int(payload["bandwidth_folds_used"])
        calibrator.bandwidth_search_history = list(payload["bandwidth_search_history"])
        calibrator.projection_loss_history = list(payload["projection_loss_history"])
        calibrator.calibration_projected = np.asarray(
            payload["calibration_projected"], dtype=np.float32
        )
        calibrator.calibration_labels = np.asarray(
            payload["calibration_labels"], dtype=np.int64
        )
        calibrator.is_fitted = True
        return calibrator

    def get_params(self) -> Dict[str, Any]:
        return {
            **asdict(self.config),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "num_classes": self.num_classes,
            "bandwidth": self.bandwidth,
            "bandwidth_folds_used": self.bandwidth_folds_used,
            "bandwidth_search_history": self.bandwidth_search_history,
            "projection_loss_history": self.projection_loss_history,
            "calibration_reference_count": (
                int(len(self.calibration_labels))
                if self.calibration_labels is not None
                else None
            ),
            "is_fitted": self.is_fitted,
        }
