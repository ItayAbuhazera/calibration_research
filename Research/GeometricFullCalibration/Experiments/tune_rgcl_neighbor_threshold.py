"""Tune an RGCL nearest-neighbor correction threshold on the calibration split.

This is a targeted companion to ``run_unified_benchmark.py``.  It reconstructs
only the stored RGCL representation for the training/reference and validation
splits, chooses a trust/separation threshold on validation accuracy, and then
applies the frozen threshold to the existing per-sample test artifact.

The calibration split is used only as a query split.  Neighbors always come
from the training split, avoiding self-neighbor leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.preprocessing import normalize

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.run_rgc_experiments import apply_spp
from Experiments.run_unified_benchmark import _select_rgcl_layers
from utils.model_utils import get_data_loaders, load_trained_model, set_seed
from utils.stability_space import StabilitySpace


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _load_run_configuration(run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary_metrics.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    metadata = summary["metadata"]

    # Prefer explicit per-method RGCL layer metadata (present when
    # --fusion_feature_source rgcl was used).  Fall back to the standard
    # frontier format where the top-label rgcl method stores no layer info
    # (layers are recomputed deterministically in main()).
    rgcl_method = next(
        (
            method
            for method in summary["methods"]
            if method.get("feature_source") == "rgcl" and method.get("selected_layers")
        ),
        None,
    )

    if rgcl_method is not None:
        selected_layers = list(rgcl_method["selected_layers"])
        target_dimension = int(rgcl_method["target_dimension"])
        pooling_mode = str(rgcl_method["pooling_mode"])
    else:
        # Frontier format: layer selection is deterministic; recomputed in main().
        selected_layers = None
        target_dimension = int(metadata.get("target_dimension", 256))
        pooling_mode = "max"

    num_layers = int(metadata.get("num_layers", 6))

    return {
        "dataset": metadata["dataset"],
        "model": metadata["model"],
        "method": metadata["method"],
        "seed": int(metadata["seed"]),
        "checkpoint_path": str(metadata["checkpoint_path"]),
        "split_sizes": metadata["split_sizes"],
        "selected_layers": selected_layers,
        "num_layers": num_layers,
        "target_dimension": target_dimension,
        "pooling_mode": pooling_mode,
    }


class StreamingRGCLExtractor:
    """Reproduce RGCL SPP + seeded projection without materializing raw features."""

    def __init__(
        self,
        model: torch.nn.Module,
        layer_names: list[str],
        device: torch.device,
        target_dimension: int,
        seed: int,
        pooling_mode: str,
    ) -> None:
        self.model = model
        self.layer_names = layer_names
        self.device = device
        self.target_dimension = int(target_dimension)
        self.pooling_mode = pooling_mode
        self.activations: dict[str, torch.Tensor] = {}
        self.handles = []
        self.projection_matrix: np.ndarray | None = None
        self.rng = np.random.RandomState(seed)

        modules = dict(model.named_modules())
        missing = [name for name in layer_names if name not in modules]
        if missing:
            raise KeyError(f"RGCL layers not found in model: {missing}")
        for name in layer_names:
            self.handles.append(modules[name].register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            self.activations[name] = output.detach()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def _project_batch(self) -> np.ndarray:
        pooled = [
            apply_spp(self.activations[name], pooling_mode=self.pooling_mode)
            .cpu()
            .numpy()
            for name in self.layer_names
        ]
        raw = np.concatenate(pooled, axis=1)

        if raw.shape[1] > self.target_dimension:
            if self.projection_matrix is None:
                # This exactly matches the legacy helper's calibration formula:
                # Z = X R, R_jk ~ N(0, 1 / target_dimension), followed by L2 norm.
                self.projection_matrix = self.rng.randn(
                    raw.shape[1], self.target_dimension
                ) / math.sqrt(self.target_dimension)
            projected = raw @ self.projection_matrix
        else:
            projected = raw
        return normalize(projected, norm="l2", axis=1)

    def extract(self, loader, *, collect_probs: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        feature_chunks: list[np.ndarray] = []
        label_chunks: list[np.ndarray] = []
        probability_chunks: list[np.ndarray] = []
        self.model.eval()

        with torch.no_grad():
            for batch_index, (inputs, labels) in enumerate(loader):
                self.activations.clear()
                inputs = inputs.to(self.device).float()
                logits = self.model(inputs)
                feature_chunks.append(self._project_batch())
                label_chunks.append(labels.numpy())
                if collect_probs:
                    probability_chunks.append(torch.softmax(logits, dim=1).cpu().numpy())
                if batch_index % 50 == 0:
                    print(f"  processed batch {batch_index + 1}/{len(loader)}", flush=True)

        features = np.concatenate(feature_chunks, axis=0)
        labels = np.concatenate(label_chunks, axis=0).astype(np.int64, copy=False)
        probabilities = (
            np.concatenate(probability_chunks, axis=0) if collect_probs else None
        )
        return features, labels, probabilities


def _geometry_arrays(
    distance_matrix: np.ndarray, base_pred: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = np.arange(len(base_pred))
    predicted_distance = distance_matrix[rows, base_pred]
    alternatives = distance_matrix.copy()
    alternatives[rows, base_pred] = np.inf
    nearest_other_pred = np.argmin(alternatives, axis=1)
    nearest_other_distance = alternatives[rows, nearest_other_pred]

    # Trust is a relative geometric margin; trust < 1 iff separation < 0.
    trust = nearest_other_distance / (predicted_distance + 1e-8)
    # Fast separation is half the absolute distance gap around the decision boundary.
    separation = (nearest_other_distance - predicted_distance) / 2.0
    return nearest_other_pred, trust, separation


def _select_lower_tail_threshold(
    score: np.ndarray,
    upper_boundary: float,
    base_pred: np.ndarray,
    alternative_pred: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    """Choose score < threshold for maximum accuracy, breaking ties conservatively."""
    eligible_indices = np.flatnonzero(score < upper_boundary)
    order = eligible_indices[np.argsort(score[eligible_indices], kind="stable")]
    ordered_scores = score[order]
    ordered_gain = (
        (alternative_pred[order] == labels[order]).astype(np.int64)
        - (base_pred[order] == labels[order]).astype(np.int64)
    )
    cumulative_gain = np.r_[0, np.cumsum(ordered_gain)]
    best_gain = int(np.max(cumulative_gain))
    # Conservative tie-break: use the fewest prediction changes.
    switch_count = int(np.flatnonzero(cumulative_gain == best_gain)[0])

    if switch_count == 0:
        threshold = float(ordered_scores[0]) if len(ordered_scores) else upper_boundary
    elif switch_count == len(ordered_scores):
        threshold = float(upper_boundary)
    else:
        threshold = float(
            (ordered_scores[switch_count - 1] + ordered_scores[switch_count]) / 2.0
        )

    switched = score < threshold
    final_pred = np.where(switched, alternative_pred, base_pred)
    fixes = switched & (base_pred != labels) & (alternative_pred == labels)
    breaks = switched & (base_pred == labels) & (alternative_pred != labels)
    wrong_to_wrong = switched & (base_pred != labels) & (alternative_pred != labels)
    return {
        "threshold": threshold,
        "upper_boundary": float(upper_boundary),
        "eligible_count_at_default_boundary": int(len(eligible_indices)),
        "prediction_change_count": int(np.sum(switched)),
        "flip_to_correct_count": int(np.sum(fixes)),
        "flip_to_wrong_count": int(np.sum(breaks)),
        "wrong_to_wrong_count": int(np.sum(wrong_to_wrong)),
        "net_corrections": int(np.sum(fixes) - np.sum(breaks)),
        "base_accuracy": float(np.mean(base_pred == labels)),
        "thresholded_accuracy": float(np.mean(final_pred == labels)),
        "tie_policy": "fewest_prediction_changes",
    }


def _apply_threshold(
    score: np.ndarray,
    threshold: float,
    base_pred: np.ndarray,
    alternative_pred: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    switched = score < threshold
    final_pred = np.where(switched, alternative_pred, base_pred)
    fixes = switched & (base_pred != labels) & (alternative_pred == labels)
    breaks = switched & (base_pred == labels) & (alternative_pred != labels)
    wrong_to_wrong = switched & (base_pred != labels) & (alternative_pred != labels)
    return {
        "threshold": float(threshold),
        "prediction_change_count": int(np.sum(switched)),
        "flip_to_correct_count": int(np.sum(fixes)),
        "flip_to_wrong_count": int(np.sum(breaks)),
        "wrong_to_wrong_count": int(np.sum(wrong_to_wrong)),
        "net_corrections": int(np.sum(fixes) - np.sum(breaks)),
        "base_accuracy": float(np.mean(base_pred == labels)),
        "thresholded_accuracy": float(np.mean(final_pred == labels)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    config = _load_run_configuration(run_dir)
    set_seed(config["seed"])
    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    print(f"device={device}")
    print(f"selected_layers={config['selected_layers']}")

    train_loader, val_loader, _test_loader, num_classes = get_data_loaders(
        config["dataset"], args.batch_size, seed=config["seed"]
    )
    checkpoint_path = Path(config["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = PROJECT_ROOT / checkpoint_path
    model = load_trained_model(
        str(checkpoint_path),
        config["model"],
        num_classes,
        device,
        dataset=config["dataset"],
    )

    selected_layers = config["selected_layers"]
    if selected_layers is None:
        print("selected_layers not stored in summary; recomputing deterministically...", flush=True)
        selected_layers = _select_rgcl_layers(
            model=model,
            model_name=config["model"],
            dataset_name=config["dataset"],
            device=device,
            num_layers=config["num_layers"],
            seed=config["seed"],
        )
    print(f"selected_layers={selected_layers}", flush=True)

    start_time = time.perf_counter()
    extractor = StreamingRGCLExtractor(
        model=model,
        layer_names=selected_layers,
        device=device,
        target_dimension=config["target_dimension"],
        seed=config["seed"],
        pooling_mode=config["pooling_mode"],
    )
    try:
        print("Extracting training/reference RGCL features...", flush=True)
        train_features, train_labels, _ = extractor.extract(
            train_loader, collect_probs=False
        )
        print("Extracting calibration RGCL features and predictions...", flush=True)
        val_features, val_labels, val_probs = extractor.extract(
            val_loader, collect_probs=True
        )
        print("Extracting test RGCL features and predictions...", flush=True)
        test_features, test_labels, test_probs = extractor.extract(
            _test_loader, collect_probs=True
        )
    finally:
        extractor.close()

    if len(train_features) != int(config["split_sizes"]["train"]):
        raise ValueError("Reconstructed training split size does not match run metadata")
    if len(val_features) != int(config["split_sizes"]["validation"]):
        raise ValueError("Reconstructed validation split size does not match run metadata")
    assert val_probs is not None
    assert test_probs is not None

    print("Computing per-class 1-NN distances (val + test)...", flush=True)
    stability_space = StabilitySpace(
        train_features, train_labels, library="fast_separation", metric="l2"
    )
    val_distance_matrix = stability_space.calc_per_class_1nn_distances(val_features)
    test_distance_matrix = stability_space.calc_per_class_1nn_distances(test_features)

    val_base_pred = np.argmax(val_probs, axis=1)
    val_alternative, val_trust, val_separation = _geometry_arrays(
        val_distance_matrix, val_base_pred
    )

    validation_trust = _select_lower_tail_threshold(
        val_trust, 1.0, val_base_pred, val_alternative, val_labels
    )
    validation_separation = _select_lower_tail_threshold(
        val_separation, 0.0, val_base_pred, val_alternative, val_labels
    )

    # Load stored test labels for consistency check; compute base_pred from fresh probs.
    per_sample_dir = run_dir / "per_sample"
    test_npz_path = per_sample_dir / "per_sample_arrays.npz"
    with np.load(test_npz_path) as test_arrays:
        stored_test_labels = test_arrays["labels_test"]
    if not np.array_equal(test_labels, stored_test_labels):
        raise ValueError("Freshly extracted test labels do not match stored per-sample labels")
    test_base_pred = np.argmax(test_probs, axis=1)
    test_alternative, test_trust, test_separation = _geometry_arrays(
        test_distance_matrix, test_base_pred
    )
    test_trust_result = _apply_threshold(
        test_trust,
        validation_trust["threshold"],
        test_base_pred,
        test_alternative,
        test_labels,
    )
    test_separation_result = _apply_threshold(
        test_separation,
        validation_separation["threshold"],
        test_base_pred,
        test_alternative,
        test_labels,
    )

    elapsed = time.perf_counter() - start_time
    result = {
        "analysis_schema_version": 1,
        "metadata": {
            **config,
            "selected_layers": selected_layers,
            "device": str(device),
            "reference_split": "train",
            "threshold_selection_split": "validation",
            "evaluation_split": "test",
            "threshold_objective": "maximize_accuracy",
            "distance_metric": "l2",
            "feature_extraction_time_and_analysis_s": elapsed,
        },
        "formulas": {
            "trust": "d_nearest_other / (d_predicted_class + 1e-8)",
            "separation": "(d_nearest_other - d_predicted_class) / 2",
            "decision": "switch_to_nearest_other_class_if_score_below_threshold",
        },
        "validation_selection": {
            "trust": validation_trust,
            "separation": validation_separation,
        },
        "test_evaluation_with_frozen_validation_threshold": {
            "trust": test_trust_result,
            "separation": test_separation_result,
        },
    }

    arrays_path = per_sample_dir / "validation_neighbor_threshold_arrays.npz"
    np.savez_compressed(
        arrays_path,
        labels_validation=val_labels,
        base_probs_validation=val_probs,
        base_pred_validation=val_base_pred,
        nearest_other_pred_validation=val_alternative,
        distance_matrix__rgcl_validation=val_distance_matrix,
        trust_score__rgcl_validation=val_trust,
        separation__rgcl_validation=val_separation,
        labels_test=test_labels,
        base_probs_test=test_probs,
        base_pred_test=test_base_pred,
        nearest_other_pred_test=test_alternative,
        distance_matrix__rgcl_test=test_distance_matrix,
        trust_score__rgcl_test=test_trust,
        separation__rgcl_test=test_separation,
    )
    result_path = per_sample_dir / "validation_neighbor_threshold_summary.json"
    result_path.write_text(
        json.dumps(_json_ready(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(_json_ready(result), indent=2, sort_keys=True))
    print(f"saved_arrays={arrays_path}")
    print(f"saved_summary={result_path}")


if __name__ == "__main__":
    main()
