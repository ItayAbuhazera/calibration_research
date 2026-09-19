#!/usr/bin/env python3
"""
Metrics Top-K Documentation Tool

Computes all layer selection metrics on val-A, then evaluates their Top-K picks
using holdout ECE on val-B (build StabilitySpace on TRAIN only; fit isotonic on VAL-A).

Outputs comprehensive JSON documenting:
- Metric descriptions
- Inferred optimization direction (minimize/maximize)
- Top-1 and Top-K layer picks per metric
- Holdout ECE for each layer
- Gap from best possible ECE
"""

from __future__ import annotations
import os
import json
import logging
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from Calibrators.metrics import AdHocLayerSelector, LayerMetrics
from Calibrators.geometric_calibrator_new import GeometricCalibrator
from Calibrators.calibration_utils import compute_ece

from utils.logging_config import get_logger
logger = get_logger(__name__)


# ============================================================================
# HELPERS
# ============================================================================

def _format7(x):
    """Format to 7 significant digits."""
    try:
        return float(f"{float(x):.7g}")
    except Exception:
        return x


def split_validation(val_x: np.ndarray, val_y: np.ndarray, split: float = 0.5, seed: int = 42):
    """Split validation set into A (metric computation) and B (holdout evaluation)."""
    rng = np.random.RandomState(seed)
    idx = np.arange(len(val_x))
    rng.shuffle(idx)
    cut = int(round(split * len(idx)))
    A, B = idx[:cut], idx[cut:]
    return (val_x[A], val_y[A]), (val_x[B], val_y[B])


def _infer_direction(metric_vals: np.ndarray, holdout_ece: np.ndarray) -> str:
    """
    Infer whether metric should be minimized or maximized based on correlation
    with holdout ECE (lower ECE is better).
    """
    ok = np.isfinite(metric_vals) & np.isfinite(holdout_ece)
    if ok.sum() < 3:
        # Fallback: compare extreme values
        i_min = int(np.nanargmin(metric_vals))
        i_max = int(np.nanargmax(metric_vals))
        return "minimize" if holdout_ece[i_min] <= holdout_ece[i_max] else "maximize"
    
    # Rank correlation: positive correlation means minimize metric → lower ECE
    rx = np.argsort(np.argsort(metric_vals[ok])).astype(float)
    ry = np.argsort(np.argsort(holdout_ece[ok])).astype(float)  # higher rank ⇒ worse ECE
    r = np.corrcoef(rx, ry)[0, 1]
    return "minimize" if (np.isfinite(r) and r > 0) else "maximize"


# Metric descriptions
METRIC_DOCS = {
    # uncertainty–geometry
    "confidence_distance_correlation": "Correlation: farther from class centroid ↔ lower confidence.",
    "boundary_proximity_correlation": "k-NN boundary proximity ↔ uncertainty alignment.",
    "uncertainty_geometry_alignment": "Aggregate alignment of geometry with uncertainty/correctness.",
    # geometry / separability
    "avg_class_separation_ratio": "Inter / intra class distance ratio.",
    "class_separation_uniformity": "Uniformity of separation across classes.",
    "local_intrinsic_dimensionality": "1 / (1 + LID) from k-NN spacing.",
    "multiscale_separation": "k-NN class purity averaged over multiple k's.",
    # calibration-ish
    "reliability_curve_quality": "Monotonicity + area-quality of a proxy reliability curve.",
    "prototype_softmax_ece": "Negative ECE from distance–softmax (higher=better).",
    "kfold_ece_utility": "Cross-validated ECE utility (1-ECE).",
    "logistic_calibratability": "How well correctness fits a 1D sigmoid of margin.",
    "spearman_stability_accuracy": "Correlation between stability score and accuracy.",
    "ece_score": "Negative ECE from geometric calibrator (higher=better).",
    # label association
    "label_cka": "Linear CKA of features vs labels.",
    "label_cka_hsic": "HSIC/CKA hybrid for label–feature association.",
    # margin & risk tails
    "margin_tail_cvar": "Tail safety of margins (higher safer).",
    "class_balanced_margin_cvar": "Per-class z-scored margin tail safety.",
    "impostor_gap_cvar": "Tail of (nearest impostor − nearest same-class) gaps.",
    "margin_skewkurt_safety": "Penalty for left-skew/heavy tails in margins.",
    # confidence shape / abstention
    "decisiveness_tail_mass": "1 − mass near 0.5 confidence (using calibrated confs if present).",
    "confidence_entropy": "1 − entropy of top confidence.",
    "confident_error_rate_tau": "Error rate among high-confidence samples (τ).",
    "aurc_proxy": "Area under risk–coverage curve (lower better).",
    # temp
    "temperature_estimate_strength": "Strength of a 1-param temperature fit (slope + NLL gain).",
    # (optional if present)
    "geometry_error_concordance": "ROC-AUC/Spearman: geometric proxy vs correctness.",
}


@dataclass
class MetricTopKEntry:
    """Results for a single metric's Top-K layer selection."""
    base_metric: str
    direction: str
    top1_layer: int
    topk_layers: List[int]
    chosen_metric_value: float
    holdout_ece_top1: float
    holdout_ece_gap_from_best: float


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def _collect_metrics_valA(
    model, model_adapter, candidate_layers, valA_loader, device: str
) -> List[LayerMetrics]:
    """
    Use AdHocLayerSelector to compute ALL metrics on val-A.
    Set skip_zero_weight_metrics=False to run all calculators.
    """
    logger.info("Computing all metrics on val-A using AdHocLayerSelector...")
    sel = AdHocLayerSelector(skip_zero_weight_metrics=False)
    
    # Call select_best_layer to get all_metrics for candidate layers
    _, all_metrics = sel.select_best_layer(
        model, valA_loader,
        candidate_layers=candidate_layers,
        device=torch.device(device),
        model_adapter=model_adapter
    )
    
    logger.info(f"Computed metrics for {len(all_metrics)} layers")
    return all_metrics


def _holdout_ece_per_layer(
    model,                     # raw torch model
    model_adapter,
    candidate_layers: List[int],
    train_raw, train_labels,
    valA_raw, valA_labels,
    valB_raw, valB_labels,
    device: str,
    batch_size: int
) -> Dict[int, float]:
    """
    For each layer:
      • build StabilitySpace on TRAIN only (features)
      • fit isotonic on VAL-A (features) while the calibrator uses original images for predictions
      • evaluate ECE on VAL-B (features) with original images for predictions
    """
    logger.info(f"Computing holdout ECE on val-B for {len(candidate_layers)} layers...")

    def _make_loader(x_np, y_np):
        return DataLoader(
            TensorDataset(torch.from_numpy(x_np), torch.from_numpy(y_np)),
            batch_size=batch_size, shuffle=False
        )

    dev = torch.device(device)
    sel = AdHocLayerSelector(skip_zero_weight_metrics=False)

    train_loader = _make_loader(train_raw, train_labels)
    valA_loader  = _make_loader(valA_raw,  valA_labels)
    valB_loader  = _make_loader(valB_raw,  valB_labels)

    def _extract(layer_idx: int, loader):
        feats_t, ys_t = sel._extract_features(model, loader, layer_idx, dev)
        X = feats_t.numpy() if isinstance(feats_t, torch.Tensor) else feats_t
        y = ys_t.numpy()    if isinstance(ys_t,   torch.Tensor) else ys_t
        # enforce float32 for embeddings, int64 for labels
        if X.dtype != np.float32:
            X = X.astype(np.float32, copy=False)
        if y.dtype != np.int64:
            y = y.astype(np.int64, copy=False)
        return X, y

    ece_map: Dict[int, float] = {}

    for L in candidate_layers:
        try:
            # TRAIN → StabilitySpace
            Xtr_embed, ytr = _extract(L, train_loader)
            # VAL-A  → isotonic fit
            Xva_embed, yva = _extract(L, valA_loader)
            # VAL-B  → holdout eval
            Xvb_embed, yvb = _extract(L, valB_loader)

            # Calibrator in FEATURE mode (auto_select_layer=False)
            gc = GeometricCalibrator(
                model=model_adapter,
                X_train_embed=Xtr_embed,   # ✅ TRAIN only
                y_train=ytr,
                auto_select_layer=False,
                library="fast_separation",
                device=device
            )

            # Fit isotonic on VAL-A (pass ORIGINAL IMAGES so predict_proba works)
            gc.fit(
                X_val_embed=Xva_embed,
                y_val=yva,
                X_val_original=valA_raw,   # ✅ images (N,C,H,W)
                fit_batch_size=batch_size
            )

            # Predict on VAL-B (pass ORIGINAL IMAGES so predict_proba works)
            probsB = gc.calibrate_batched(
                X_test_embed=Xvb_embed,
                X_test_original=valB_raw,  # ✅ images (N,C,H,W)
                batch_size=batch_size
            )

            ece = compute_ece(probsB, yvb, n_bins=15)
            ece_map[int(L)] = _format7(ece)
            logger.debug(f"  Layer {L}: holdout ECE = {ece:.7g}")

        except Exception as e:
            logger.warning(f"Holdout ECE failed at layer {L}: {e}")
            ece_map[int(L)] = float("inf")

    return ece_map


def run_metrics_doc_topk(
    *,
    model,
    model_adapter,
    candidate_layers: List[int],
    train_raw, train_labels,
    val_raw, val_labels,
    device: str = "cuda",
    batch_size: int = 128,
    topk: int = 3,
    val_split: float = 0.5,
    seed: int = 42,
    out_dir: str,
    try_both: bool = False
) -> Dict:
    """
    Main entry point: document all metrics with Top-K picks and holdout ECE.
    
    Args:
        model: Raw PyTorch model
        model_adapter: PyTorchModelAdapter wrapper
        candidate_layers: Layer indices to evaluate
        train_raw: Training images
        train_labels: Training labels
        val_raw: Validation images
        val_labels: Validation labels
        device: 'cuda' or 'cpu'
        batch_size: Batch size for inference
        topk: How many top layers to report per metric
        val_split: Fraction of val to use for metric computation (rest for holdout)
        seed: Random seed for val split
        out_dir: Output directory
    
    Returns:
        Dictionary with all results
    """
    logger.info("="*80)
    logger.info("METRICS TOP-K DOCUMENTATION")
    logger.info("="*80)
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Split validation into A (metrics) and B (holdout)
    logger.info(f"Splitting validation: {val_split*100:.0f}% for metrics, {(1-val_split)*100:.0f}% for holdout")
    (valA_x, valA_y), (valB_x, valB_y) = split_validation(
        val_raw, val_labels, split=val_split, seed=seed
    )
    logger.info(f"  Val-A: {len(valA_x)} samples (metric computation)")
    logger.info(f"  Val-B: {len(valB_x)} samples (holdout ECE evaluation)")
    
    # Sanity checks: ensure we have 4D images
    for name, x in {"train": train_raw, "valA": valA_x, "valB": valB_x}.items():
        assert x.ndim == 4, f"{name} must be images (N,C,H,W), got shape {x.shape}"
    logger.info("✅ Data shape validation passed: all inputs are 4D images")
    
    # Dataloader for val-A
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(valA_x), torch.from_numpy(valA_y)),
        batch_size=batch_size, shuffle=False
    )
    
    # 1) Compute all metrics on val-A
    all_metrics: List[LayerMetrics] = _collect_metrics_valA(
        model, model_adapter, candidate_layers, va_loader, device
    )
    
    # Extract metric field names
    metric_fields = [
        k for k in all_metrics[0].__dataclass_fields__.keys()
        if k not in {"layer_idx", "composite_score", "computation_times"}
    ]
    logger.info(f"Found {len(metric_fields)} metrics to analyze")
    
    # 2) Compute holdout ECE on val-B per layer
    ece_map = _holdout_ece_per_layer(
        model,                                  # <--- add this
        model_adapter,
        [m.layer_idx for m in all_metrics],
        train_raw, train_labels,
        valA_x, valA_y, valB_x, valB_y,
        device, batch_size
    )
    
    Ls = [m.layer_idx for m in all_metrics]
    ece_vec = np.array([ece_map[int(L)] for L in Ls], dtype=float)
    best_ece = float(np.nanmin(ece_vec)) if np.isfinite(ece_vec).any() else float("inf")
    logger.info(f"Best holdout ECE across all layers: {best_ece:.7g}")
    
    # 3) Analyze Top-K picks per metric
    logger.info("\nAnalyzing Top-K picks per metric...")
    metrics_topk: Dict[str, MetricTopKEntry] = {}
    all_metric_values: Dict[str, Dict[str, float]] = {}
    
    for name in metric_fields:
        vals = np.array([getattr(m, name, np.nan) for m in all_metrics], dtype=float)
        all_metric_values[name] = {str(int(L)): _format7(v) for L, v in zip(Ls, vals)}
        
        # existing inferred direction
        inferred_dir = _infer_direction(vals, ece_vec)
        
        def build_entry(dir_mode: str) -> MetricTopKEntry:
            order = np.argsort(vals) if dir_mode == "minimize" else np.argsort(-vals)
            top_idx = int(order[0])
            topL = int(Ls[top_idx])
            topK = [int(Ls[i]) for i in order[:max(1, topk)]]
            pick_val = _format7(vals[top_idx])
            pick_ece = _format7(ece_vec[top_idx])
            gap = _format7(pick_ece - best_ece)
            return MetricTopKEntry(
                base_metric=name,
                direction=dir_mode,
                top1_layer=topL,
                topk_layers=topK,
                chosen_metric_value=pick_val,
                holdout_ece_top1=pick_ece,
                holdout_ece_gap_from_best=gap
            )
        
        # Always keep the inferred version under the original key for backwards compatibility
        inferred_entry = build_entry(inferred_dir)
        metrics_topk[name] = inferred_entry
        logger.info(f"  {name:40s}: {inferred_dir:8s} → layer {inferred_entry.top1_layer:3d}, "
                    f"holdout ECE={inferred_entry.holdout_ece_top1:.7g} "
                    f"(gap={inferred_entry.holdout_ece_gap_from_best:+.7g})")
        
        # Optionally also write explicit minimize/maximize variants
        if try_both:
            min_entry = build_entry("minimize")
            max_entry = build_entry("maximize")
            metrics_topk[f"{name}__min"] = min_entry
            metrics_topk[f"{name}__max"] = max_entry
            logger.info(f"    ├─ as minimize → layer {min_entry.top1_layer:3d}, "
                        f"ECE={min_entry.holdout_ece_top1:.7g} (gap={min_entry.holdout_ece_gap_from_best:+.7g})")
            logger.info(f"    └─ as maximize → layer {max_entry.top1_layer:3d}, "
                        f"ECE={max_entry.holdout_ece_top1:.7g} (gap={max_entry.holdout_ece_gap_from_best:+.7g})")
    
    # 4) Build output document
    doc = {
        "val_split": {
            "p": val_split,
            "seed": seed,
            "sizes": {
                "A": int(len(valA_x)),
                "B": int(len(valB_x))
            }
        },
        "candidate_layers": list(map(int, Ls)),
        "metric_docs": METRIC_DOCS,
        "metrics_topk": {k: asdict(v) for k, v in metrics_topk.items()},
        "holdout_ece_per_layer": {str(int(L)): _format7(e) for L, e in zip(Ls, ece_vec)},
        "best_holdout_ece": _format7(best_ece),
        "all_metric_values": all_metric_values,
        "includes_both_directions": bool(try_both)
    }
    
    # Save
    out_json = os.path.join(out_dir, "metric_docs_and_topk.json")
    with open(out_json, "w") as f:
        json.dump(doc, f, indent=2)
    
    logger.info(f"\n✅ Metrics documentation saved: {out_json}")
    logger.info("="*80)
    
    return doc


