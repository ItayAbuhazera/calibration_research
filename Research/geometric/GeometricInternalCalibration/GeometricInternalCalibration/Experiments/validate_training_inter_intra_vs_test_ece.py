#!/usr/bin/env python
"""
Standalone validator: correlate training-time geometric metrics with test-set ECE per layer.

Usage:
    python Experiments/validate_training_inter_intra_vs_test_ece.py \
        --method augmix \
        --dataset cifar10 \
        --model resnet18 \
        --seed 12 \
        --results-base-dir aaai_full_experiments/results \
        --ground-truth aaai_full_experiments/results/layer_selection_analysis_baselines_4/augmix/cifar10/resnet18/seed12/per_layer_ground_truth.json
"""

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader
from scipy.stats import pearsonr, spearmanr

gc.set_threshold(700, 10, 10)  # more aggressive GC
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Calibrators.metrics import (
    ClassBalancedTailCVaRCalculator,
    ClassSeparationQualityCalculator,
    ClassSeparationUniformityCalculator,
    GeometryErrorConcordanceCalculator,
    ImpostorGapCVaRCalculator,
    LabelCKACalculator,
    LocalIntrinsicDimensionalityCalculator,
    MarginSkewKurtosisCalculator,
    MarginTailCVaRCalculator,
    MultiScaleClassHomogeneityCalculator,
)
from Calibrators.nc_metrics import (
    NC1CollapseMetricCalculator,
    NC4SeparabilityCalculator,
)
from Experiments.run_post_hoc_calibration import (
    construct_model_path,
    get_data_loaders,
    load_trained_model,
)
from Experiments.layer_selection import extract_features_directly, NumpyEncoder


def _compute_class_separations(features: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute mean inter/intra-class distance ratio across classes.

    This mirrors the helper the user supplied: for each class, compute the mean
    inter-class distance to all other-class samples and divide by the mean
    intra-class distance. The final metric is the average ratio across classes.
    """
    unique_labels = np.unique(labels)
    separation_scores: List[float] = []

    for cls in unique_labels:
        cls_mask = labels == cls
        cls_feats = features[cls_mask]
        other_feats = features[~cls_mask]

        if len(cls_feats) < 2 or len(other_feats) == 0:
            continue

        # Intra-class: upper-triangular pairwise distances
        intra = np.linalg.norm(
            cls_feats[:, None, :] - cls_feats[None, :, :],
            axis=-1,
        )
        intra = intra[np.triu_indices(len(cls_feats), k=1)]
        intra_mean = float(intra.mean()) if intra.size > 0 else 0.0

        # Inter-class: all pairs between class and others
        inter = np.linalg.norm(
            cls_feats[:, None, :] - other_feats[None, :, :],
            axis=-1,
        )
        inter_mean = float(inter.mean()) if inter.size > 0 else 0.0

        if intra_mean > 0:
            separation_scores.append(inter_mean / intra_mean)

    return float(np.mean(separation_scores)) if separation_scores else 0.0


def _centroid_predictions(features: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    """Nearest-centroid predictions used for geometry_error_concordance."""
    X = features.cpu().numpy()
    y = labels.cpu().numpy()
    classes = np.unique(y)
    centroids = {c: X[y == c].mean(axis=0) for c in classes}
    preds = []
    for x in X:
        best = None
        best_d = float("inf")
        for c, mu in centroids.items():
            d = np.linalg.norm(x - mu)
            if d < best_d:
                best_d = d
                best = c
        preds.append(best if best is not None else classes[0])
    return np.asarray(preds)


def _apply_additional_compression_ratio(features: torch.Tensor, compression_ratio: float) -> torch.Tensor:
    """
    Apply random projection to compress by a given ratio (JL-style).
    Keeps at least 32 dims and normalizes projection by sqrt(target_dim).
    """
    current_dim = features.shape[1]
    target_dim = max(32, int(current_dim / compression_ratio))
    if target_dim >= current_dim or compression_ratio <= 1.0:
        return features
    print(f"[compress] Applying {compression_ratio:.1f}x compression: {current_dim} -> {target_dim} dims")
    projection = torch.randn(current_dim, target_dim, device=features.device)
    projection = projection / torch.sqrt(torch.tensor(target_dim, dtype=torch.float32, device=features.device))
    return features @ projection


def compute_metrics(
    features_by_layer: Dict[int, torch.Tensor],
    labels: torch.Tensor,
    max_samples: int = 10000,
    high_dim_threshold: int = 500,
    extra_compression_ratio: float = 4.0,
) -> Dict[int, Dict[str, float]]:
    """
    Compute all requested geometric metrics per layer.
    Returns: {layer_idx: {metric_name: value}}
    """
    calculators = {
        "inter_intra_ratio": None,  # handled separately
        "nc1_collapse": NC1CollapseMetricCalculator(),
        "nc4_separability": NC4SeparabilityCalculator(),
        "margin_tail_cvar": MarginTailCVaRCalculator(alpha=0.10),
        "class_balanced_margin_cvar": ClassBalancedTailCVaRCalculator(),
        "impostor_gap_cvar": ImpostorGapCVaRCalculator(),
        "margin_skewkurt": MarginSkewKurtosisCalculator(),
        "avg_class_separation_ratio": ClassSeparationQualityCalculator(),
        "class_separation_uniformity": ClassSeparationUniformityCalculator(),
        "multiscale_separation": MultiScaleClassHomogeneityCalculator(k_list=[5, 15, 30]),
        "local_intrinsic_dimensionality": LocalIntrinsicDimensionalityCalculator(k=20),
        "label_cka": LabelCKACalculator(),
        "geometry_error_concordance": GeometryErrorConcordanceCalculator(proxy="centroid_margin"),
    }

    results: Dict[int, Dict[str, float]] = {}
    for layer_idx, feats in features_by_layer.items():
        # Ensure CPU
        if feats.is_cuda:
            feats = feats.cpu()

        n_samples = len(feats)
        original_dims = feats.shape[1] if feats.ndim > 1 else 1

        # STEP 1: Apply additional compression if dimensionality too high
        if extra_compression_ratio is not None and original_dims > high_dim_threshold:
            feats = _apply_additional_compression_ratio(feats, extra_compression_ratio)
            print(f"[info] Layer {layer_idx}: {original_dims}D -> {feats.shape[1]}D ({original_dims/feats.shape[1]:.1f}x compression)")

        # STEP 2: Subsample to max_samples (consistent across layers)
        if max_samples is not None and n_samples > max_samples:
            print(f"[info] Subsampling from {n_samples} to {max_samples} samples (layer {layer_idx})")
            idx = torch.randperm(n_samples)[:max_samples]
            feats = feats[idx]
            labels_subset = labels[idx]
        else:
            labels_subset = labels

        layer_metrics: Dict[str, float] = {}

        # Centroid predictions with guard
        try:
            preds = _centroid_predictions(feats, labels_subset)
        except Exception as e:
            print(f"[error] Centroid predictions failed for layer {layer_idx}: {e}")
            preds = np.zeros(len(feats), dtype=np.int64)

        np_feats = feats.numpy()
        np_labels = labels_subset.numpy()

        # inter/intra ratio
        try:
            layer_metrics["inter_intra_ratio"] = _compute_class_separations(np_feats, np_labels)
        except Exception as e:
            print(f"[warn] inter_intra_ratio failed on layer {layer_idx}: {e}")
            layer_metrics["inter_intra_ratio"] = 0.0

        for name, calc in calculators.items():
            if calc is None:
                continue
            try:
                layer_metrics[name] = float(
                    calc.compute(
                        features=feats,
                        labels=labels_subset,
                        predictions=preds,
                        confidences=None,
                    )
                )
            except Exception as e:
                print(f"[warn] metric {name} failed on layer {layer_idx}: {e}")
                layer_metrics[name] = 0.0

        results[layer_idx] = layer_metrics

        # Cleanup
        del feats, preds, np_feats, np_labels, labels_subset
        gc.collect()

    return results


def plot_scatter(
    x: List[float],
    y: List[float],
    labels: List[int],
    out_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, c="steelblue", alpha=0.75)
    for lx, ly, lbl in zip(x, y, labels):
        plt.text(lx, ly, str(lbl), fontsize=8, alpha=0.8)
    plt.xlabel("Training inter/intra ratio (higher = better separation)")
    plt.ylabel("Test ECE (lower is better)")
    plt.title("Training separation vs. Test ECE per layer")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Correlate training-time separation with test ECE."
    )
    parser.add_argument("--method", type=str, required=True, help="Training method, e.g., augmix")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name, e.g., cifar10")
    parser.add_argument("--model", type=str, required=True, help="Model name, e.g., resnet18")
    parser.add_argument("--seed", type=int, required=True, help="Training seed")
    parser.add_argument(
        "--results-base-dir",
        type=Path,
        default=Path("aaai_full_experiments/results"),
        help="Base directory containing experiment results.",
    )
    parser.add_argument(
        "--exp-name",
        type=str,
        default="layer_selection_analysis_baselines_4",
        help="Experiment name folder containing best_model.pth.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="Optional explicit path to best_model.pth; overrides auto construction.",
    )
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="Path to per_layer_ground_truth.json containing test ECE.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run feature extraction.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write plots/tables. Defaults to ground-truth directory.",
    )
    parser.add_argument(
        "--no-compression",
        action="store_true",
        help="Disable compression for testing (may OOM on some layers)",
    )
    parser.add_argument(
        "--test-layers",
        type=str,
        default=None,
        help="Comma-separated list of specific layers to test (e.g., '1,20,35,50')",
    )
    args = parser.parse_args()

    output_dir = args.output_dir or args.ground_truth.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print(f"[info] Using device: {device}")

    # Load metadata
    with open(args.ground_truth, "r") as f:
        gt = json.load(f)
    candidate_layers = [int(x) for x in gt.get("candidate_layers", [])]
    test_ece_map = {int(k): float(v) for k, v in gt.get("test", {}).get("ece", {}).items()}
    print(f"[info] Loaded {len(candidate_layers)} candidate layers from ground truth.")

    # Data + model via project utilities
    train_loader, val_loader, _test_loader, num_classes = get_data_loaders(
        dataset=args.dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    # Combine train + val to approximate full training set if val_loader exists
    if val_loader is not None:
        full_train_ds = ConcatDataset([train_loader.dataset, val_loader.dataset])
    else:
        full_train_ds = train_loader.dataset
    dataloader = DataLoader(
        full_train_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    if args.checkpoint_path is not None:
        checkpoint_path = Path(args.checkpoint_path)
    else:
        checkpoint_path = Path(
            construct_model_path(
                base_dir=str(args.results_base_dir),
                method=args.method,
                dataset=args.dataset,
                model=args.model,
                seed=args.seed,
            )
        )
    print(f"[info] Loading model from {checkpoint_path}")
    model = load_trained_model(str(checkpoint_path), args.model, num_classes, device, dataset=args.dataset)

    # Feature extraction and metric computation (per-layer to avoid OOM)
    metrics_per_layer: Dict[int, Dict[str, float]] = {}
    labels = None
    if args.no_compression:
        partial_path = output_dir / "metrics_partial_uncompressed.json"
        csv_path = output_dir / "all_metrics_with_test_ece_uncompressed.csv"
        json_path = output_dir / "correlation_summary_uncompressed.json"
        fig_path = output_dir / "all_metrics_vs_test_ece_uncompressed.pdf"
    else:
        partial_path = output_dir / "metrics_partial.json"
        csv_path = output_dir / "all_metrics_with_test_ece.csv"
        json_path = output_dir / "correlation_summary.json"
        fig_path = output_dir / "all_metrics_vs_test_ece.pdf"

    # LOAD EXISTING PARTIAL RESULTS IF THEY EXIST
    if partial_path.exists():
        try:
            with open(partial_path, "r") as f:
                metrics_per_layer = json.load(f)
            metrics_per_layer = {int(k): v for k, v in metrics_per_layer.items()}
            completed_layers = set(metrics_per_layer.keys())
            print(f"[info] Loaded {len(completed_layers)} already-computed layers from {partial_path}")
            print(f"[info] Completed layers: {sorted(completed_layers)}")
        except Exception as e:
            print(f"[warn] Failed to load partial results: {e}")
            metrics_per_layer = {}
            completed_layers = set()
    else:
        completed_layers = set()
        print(f"[info] No partial results found, starting from scratch")

    # Determine which layers to process
    if args.test_layers:
        test_layers_list = [int(x.strip()) for x in args.test_layers.split(",") if x.strip()]
        remaining_layers = [l for l in test_layers_list if l in candidate_layers]
        print(f"[info] Testing specific layers: {remaining_layers}")
    else:
        # FILTER candidate_layers to only include layers NOT yet computed
        remaining_layers = [l for l in candidate_layers if l not in completed_layers]
        print(f"[info] {len(remaining_layers)} layers remaining to compute: {remaining_layers}")
        if len(remaining_layers) == 0:
            print(f"[info] All layers already computed! Proceeding to analysis...")

    # Compression settings
    if args.no_compression:
        compression_ratio = None
        metric_max_samples = 5000  # safer cap without compression
        print("[warning] COMPRESSION DISABLED - may OOM on high-dim layers!")
    else:
        compression_ratio = 16.0 if args.dataset.lower() == "cifar10" else (32.0 if args.dataset.lower() == "cifar100" else None)
        metric_max_samples = 10000

    print(f"[info] Compression ratio: {compression_ratio}")
    print(f"[info] Max samples for metrics: {metric_max_samples}")

    for idx, layer_idx in enumerate(remaining_layers, start=1):
        print(f"\n[info] Computing metrics for layer {layer_idx} ({idx}/{len(remaining_layers)})")
        try:
            feats, labs = extract_features_directly(
                model, dataloader, layer_idx, device,
                compression_ratio=compression_ratio,
                fixed_feature_dim=None if args.no_compression else 512  # disable when no-compression
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[error] OOM on layer {layer_idx} - skipping")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            else:
                raise
        if labels is None:
            labels = labs
        # compute metrics for this layer only (with subsampling/compression to avoid OOM)
        layer_metrics = compute_metrics(
            {layer_idx: feats},
            labels,
            max_samples=metric_max_samples,
            high_dim_threshold=500,
            extra_compression_ratio=None if args.no_compression else 4.0,  # respect --no-compression
        )[layer_idx]
        metrics_per_layer[layer_idx] = layer_metrics

        # Persist progress to survive crashes/kill
        try:
            temp_path = partial_path.with_suffix('.tmp')
            with open(temp_path, "w") as f:
                json.dump(metrics_per_layer, f, indent=2, cls=NumpyEncoder)
            temp_path.replace(partial_path)  # atomic rename
            print(f"[checkpoint] Saved metrics for layer {layer_idx}")
        except Exception as e:
            print(f"[warn] failed to write partial metrics: {e}")

        # free memory
        del feats
        if labs is not labels:
            del labs

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            print(f"[memory] GPU: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved")

    if labels is None:
        raise ValueError("No labels extracted; check dataloader and candidate layers.")

    # Align layers present in both metric and test ECE
    common_layers = sorted(set(metrics_per_layer.keys()) & set(test_ece_map.keys()))
    metric_names = sorted({m for layer in metrics_per_layer.values() for m in layer})

    # Correlations per metric
    corr_summary = []
    for mname in metric_names:
        xs = [metrics_per_layer[l][mname] for l in common_layers if mname in metrics_per_layer[l]]
        ys = [test_ece_map[l] for l in common_layers if mname in metrics_per_layer[l]]
        if len(xs) > 1:
            pr, pp = pearsonr(xs, ys)
            sr_res = spearmanr(xs, ys)
            sr, sp = sr_res.correlation, sr_res.pvalue
        else:
            pr = pp = sr = sp = np.nan
        corr_summary.append(
            {
                "metric": mname,
                "pearson_r": pr,
                "pearson_p": pp,
                "spearman_r": sr,
                "spearman_p": sp,
                "abs_spearman": abs(sr) if np.isfinite(sr) else 0.0,
            }
        )

    corr_summary.sort(key=lambda d: d["abs_spearman"], reverse=True)

    # Print top-5 metrics by |Spearman|
    print("\n=== Top-5 metrics by |Spearman| correlation with test ECE ===")
    for row in corr_summary[:5]:
        print(
            f"{row['metric']:<30}  "
            f"pearson={row['pearson_r']:.4f} (p={row['pearson_p']:.4g})  "
            f"spearman={row['spearman_r']:.4f} (p={row['spearman_p']:.4g})"
        )

    # Scatter plots grid
    n_metrics = len(metric_names)
    cols = 4
    rows = math.ceil(n_metrics / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")

    for idx, mname in enumerate(metric_names):
        r = idx // cols
        c = idx % cols
        ax = axes[r, c]
        ax.axis("on")
        xs = [metrics_per_layer[l][mname] for l in common_layers if mname in metrics_per_layer[l]]
        ys = [test_ece_map[l] for l in common_layers if mname in metrics_per_layer[l]]
        ax.scatter(xs, ys, alpha=0.75)
        for lx, ly, lbl in zip(xs, ys, common_layers):
            ax.text(lx, ly, str(lbl), fontsize=7, alpha=0.7)
        # pick corr values
        match = next((row for row in corr_summary if row["metric"] == mname), None)
        highlight = match and abs(match["spearman_r"]) > 0.7
        title = f"{mname} (ρ={match['spearman_r']:.2f})" if match else mname
        ax.set_title(title, color="darkred" if highlight else "black", fontsize=10)
        ax.set_xlabel(mname)
        ax.set_ylabel("Test ECE")
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_path = output_dir / "all_metrics_vs_test_ece.pdf"
    plt.savefig(fig_path)
    plt.close(fig)
    print(f"[info] Scatter grid saved to {fig_path}")

    # CSV with all metrics + test ECE
    csv_path = output_dir / "all_metrics_with_test_ece.csv"
    header = ["layer"] + metric_names + ["test_ece"]
    with open(csv_path, "w") as f:
        f.write(",".join(header) + "\n")
        for layer in common_layers:
            row = [layer] + [metrics_per_layer[layer].get(m, np.nan) for m in metric_names] + [
                test_ece_map[layer]
            ]
            f.write(",".join(str(x) for x in row) + "\n")
    print(f"[info] Metrics table saved to {csv_path}")

    # JSON correlation summary
    json_path = output_dir / "correlation_summary.json"
    with open(json_path, "w") as f:
        json.dump(corr_summary, f, indent=2, cls=NumpyEncoder)
    print(f"[info] Correlation summary saved to {json_path}")

    # Compare bests by inter/intra ratio
    inter_rows = []
    for l in common_layers:
        inter_rows.append((l, metrics_per_layer[l]["inter_intra_ratio"], test_ece_map[l]))
    inter_rows.sort(key=lambda r: r[1], reverse=True)

    print("\n=== Layers ranked by training inter/intra ratio (desc) ===")
    print(f"{'Layer':>8} | {'Train inter/intra':>18} | {'Test ECE':>8}")
    print("-" * 44)
    for l, sep, ece in inter_rows:
        print(f"{l:>8} | {sep:>18.6f} | {ece:>8.6f}")

    best_training = inter_rows[0] if inter_rows else None
    best_test = min(inter_rows, key=lambda r: r[2]) if inter_rows else None
    if best_training and best_test:
        print("\n=== Best layers (using inter/intra) ===")
        print(
            f"Highest training separation : layer {best_training[0]} "
            f"(ratio={best_training[1]:.4f}, test ECE={best_training[2]:.4f})"
        )
        print(
            f"Lowest test ECE             : layer {best_test[0]} "
            f"(ratio={best_test[1]:.4f}, test ECE={best_test[2]:.4f})"
        )

    print(f"\n[info] Completed computation for {len(metrics_per_layer)}/{len(candidate_layers)} total layers")


if __name__ == "__main__":
    main()

