"""
Compression Experiment Worker Script
Runs a single compression experiment with SPP+JL compression on a specified layer.

This script is called by run_compression_sweep.py via sbatch --wrap.
"""

import argparse
import json
import logging
import gc
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import from your actual codebase
from utils.compression_utils import FixedSizeSPP_JL, create_compression_module
from Experiments.run_post_hoc_calibration import load_trained_model, get_data_loaders, PyTorchModelAdapter
from Calibrators.geometric_calibrator_new import GeometricCalibrator
from Calibrators.calibration_utils import compute_ece
from Experiments.layer_selection import compress_with_spp_jl

from utils.logging_config import get_logger
logger = get_logger(__name__)

def extract_features_directly(model, dataloader, layer_idx, device, fixed_feature_dim=512, 
                              compression_method='fixed_spp_jl', pyramid_levels=[4, 2, 1], mid_channels=None):
    """
    Extract features from a specific layer EXACTLY MATCHING AdHocLayerSelector._extract_features
    
    Args:
        fixed_feature_dim: If None, returns flattened features (no compression).
                           If int, applies SPP+JL to reach the target dim.
    """
    logger.info(f"🔍 Starting feature extraction for layer {layer_idx}")
    
    total_expected_samples = len(dataloader.dataset)
    logger.info(f"  Expected total samples: {total_expected_samples}")
    
    features_list = []
    labels_list = []
    
    # SPP+JL projectors for different channel sizes (matches AdHocLayerSelector)
    spp_projectors = {}
    
    activation = {}
    def hook(module, input, output):
        activation['output'] = output
    
    # 🔧 SAME enumeration as GeometricCalibrator/AdHocLayerSelector
    layers = list(model.modules())
    if layer_idx >= len(layers):
        raise ValueError(f"Layer index {layer_idx} out of bounds")
    
    target_layer = layers[layer_idx]
    logger.info(f"  Hooking layer {layer_idx}: {target_layer.__class__.__name__}")
    handle = target_layer.register_forward_hook(hook)
    # HuggingFace one-time sanity toggle
    try:
        if hasattr(model, "config"):
            model.config.output_attentions = False
            model.config.output_hidden_states = False
            model.config.return_dict = True
    except Exception:
        pass
    
    model.eval()
    use_compression = (fixed_feature_dim is not None)
    
    try:
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Extracting L{layer_idx}", leave=False):
                if len(batch) == 2:
                    data, target = batch
                else:
                    data, target = batch[0], torch.zeros(len(batch[0]), dtype=torch.long)
                
                data = data.to(device).float()
                activation.clear()
                
                # Forward pass
                _ = model(data)
                
                # Get activation
                from utils.tensor_utils import coerce_to_tensor
                raw = activation.get('output', None)
                feat = coerce_to_tensor(raw)
                if feat is None or not torch.is_tensor(feat):
                    logger.warning(f"[L{layer_idx} {target_layer.__class__.__name__}] output type {type(raw)}; skipping batch")
                    continue
                
                # =========================================================
                # ## EXACT SAME LOGIC AS AdHocLayerSelector ##
                # =========================================================
                if feat.ndim == 3:
                    import math
                    B, T, C = feat.shape
                    TT = T - 1
                    has_square_grid = (TT > 0 and int(math.isqrt(TT)) ** 2 == TT)
                    tokens = feat[:, 1:, :] if has_square_grid else feat
                    Tuse = TT if has_square_grid else T
                    s = int(math.isqrt(Tuse))
                    if s * s == Tuse:
                        fmap = tokens.transpose(1, 2).reshape(B, C, s, s)
                        if use_compression:
                            num_channels = fmap.size(1)
                            if num_channels not in spp_projectors:
                                logger.info(f"Creating new {compression_method} projector for C={num_channels} -> {fixed_feature_dim}")
                                spp_projectors[num_channels] = create_compression_module(
                                    method=compression_method,
                                    in_channels=num_channels,
                                    final_output_dim=fixed_feature_dim,
                                    pyramid_levels=pyramid_levels,
                                    mid_channels=mid_channels
                                ).to(device)
                            projector = spp_projectors[num_channels]
                            feat = projector(fmap)
                        else:
                            # Flatten spatial tokens without compression
                            feat = tokens.reshape(B, -1)
                    else:
                        if use_compression:
                            # Fallback: mean-pool tokens if not square grid
                            feat = tokens.mean(dim=1)
                        else:
                            # Flatten token sequence without compression
                            feat = tokens.reshape(B, -1)
                elif feat.ndim > 3:
                    if use_compression:
                        num_channels = feat.size(1)
                        if num_channels not in spp_projectors:
                            logger.info(f"Creating new {compression_method} projector for C={num_channels} -> {fixed_feature_dim}")
                            spp_projectors[num_channels] = create_compression_module(
                                method=compression_method,
                                in_channels=num_channels,
                                final_output_dim=fixed_feature_dim,
                                pyramid_levels=pyramid_levels,
                                mid_channels=mid_channels
                            ).to(device)
                        projector = spp_projectors[num_channels]
                        feat = projector(feat)
                    else:
                        # Flatten convolutional feature map without compression
                        feat = feat.reshape(feat.size(0), -1)
                elif feat.ndim == 2:
                    pass
                else:
                    logger.warning(f"Layer {layer_idx}: unexpected feature ndim {feat.ndim}; skipping batch")
                    continue
                # =========================================================
                
                features_list.append(feat.cpu())
                labels_list.append(target.cpu())
                        
    finally:
        handle.remove()
    
    if not features_list:
        raise ValueError("No features were extracted")
    
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    logger.info(f"✅ Final extracted shape for layer {layer_idx}: {features.shape}")
    
    return features, labels


def _load_uncompressed_baseline_ece(
    output_root: Path,
    training_loss: str,
    dataset: str,
    model: str,
    seed: int,
    layer_type: str,
    baseline_dim: int
):
    """
    LEGACY: Load ECE from fixed-dimension baseline file.
    This helper is retained for backward compatibility with historical results.
    
    Tries multiple filename patterns for backward compatibility:
    1. NEW: {layer_type}_ratio1_dim{baseline_dim}_results.json (ratio-based naming)
    2. LEGACY: {layer_type}_dim{baseline_dim}_results.json (fixed-size naming)
    
    Args:
        output_root: Base output directory
        training_loss: Training loss function name
        dataset: Dataset name
        model: Model architecture name
        seed: Random seed
        layer_type: Layer type (data_layer, best_layer, worst_layer)
        baseline_dim: Dimension of uncompressed baseline
    
    Returns:
        Baseline ECE as float, or None if not found
    """
    base_dir = (
        Path(output_root)
        / training_loss
        / dataset
        / model
        / f"seed{seed}"
    )
    
    if not base_dir.exists():
        logger.warning(f"Baseline directory not found: {base_dir}")
        return None
    
    # Try NEW naming convention first (ratio-based)
    ratio1_pattern = f"{layer_type}_ratio1_dim{baseline_dim}_results.json"
    ratio1_path = base_dir / ratio1_pattern
    
    if ratio1_path.exists():
        try:
            with open(ratio1_path, "r") as f:
                data = json.load(f)
            ece = data.get("ece", None)
            if ece is None:
                logger.warning(f"No 'ece' field in baseline file {ratio1_path}")
                return None
            logger.debug(f"Loaded baseline ECE from {ratio1_path}")
            return float(ece)
        except Exception as e:
            logger.warning(f"Failed to load baseline ECE from {ratio1_path}: {e}")
    
    # Try LEGACY naming convention (fixed-size)
    legacy_pattern = f"{layer_type}_dim{baseline_dim}_results.json"
    legacy_path = base_dir / legacy_pattern
    
    if legacy_path.exists():
        try:
            with open(legacy_path, "r") as f:
                data = json.load(f)
            ece = data.get("ece", None)
            if ece is None:
                logger.warning(f"No 'ece' field in baseline file {legacy_path}")
                return None
            logger.debug(f"Loaded baseline ECE from legacy file {legacy_path}")
            return float(ece)
        except Exception as e:
            logger.warning(f"Failed to load baseline ECE from {legacy_path}: {e}")
    
    # Try searching with wildcards (more flexible but slower)
    wildcard_pattern = f"{layer_type}_ratio1_dim*_results.json"
    candidates = list(base_dir.glob(wildcard_pattern))
    
    for candidate_path in candidates:
        try:
            with open(candidate_path, "r") as f:
                data = json.load(f)
            
            # Check if this is actually a baseline (compression_ratio close to 1.0)
            comp_ratio = data.get("requested_compression_ratio") or data.get("compression_ratio")
            if comp_ratio is None:
                continue
            
            try:
                if abs(float(comp_ratio) - 1.0) <= 0.05:  # Within 5% of 1.0
                    ece = data.get("ece")
                    if ece is not None:
                        logger.info(f"Found baseline ECE via wildcard search: {candidate_path}")
                        return float(ece)
            except (TypeError, ValueError):
                continue
        except Exception:
            continue
    
    logger.warning(
        f"Uncompressed baseline not found. Tried:\n"
        f"  1. {ratio1_path}\n"
        f"  2. {legacy_path}\n"
        f"  3. Wildcard search in {base_dir}"
    )
    return None


def _load_lowest_ratio_baseline_ece(
    output_root: Path,
    training_loss: str,
    dataset: str,
    model: str,
    seed: int,
    layer_type: str,
    target_ratio: Optional[float] = None,
    original_dim: Optional[int] = None,
    exclude_ratios: Optional[List[float]] = None
) -> Optional[float]:
    """
    Load baseline ECE using flexible ratio-based search strategy.
    
    If target_ratio is provided, attempt to load that specific ratio. Otherwise
    select the available result with the lowest compression ratio (closest to 1×).
    """
    base_dir = (
        Path(output_root)
        / training_loss
        / dataset
        / model
        / f"seed{seed}"
    )
    
    if not base_dir.exists():
        logger.debug(f"Baseline directory not found: {base_dir}")
        return None
    
    exclude_ratios = exclude_ratios or []
    
    pattern = f"{layer_type}_ratio*_dim*_results.json"
    candidates = list(base_dir.glob(pattern))
    if not candidates:
        logger.debug(f"No baseline candidates found matching {pattern} in {base_dir}")
        return None
    
    baseline_options: List[Dict] = []
    for candidate_path in candidates:
        try:
            with open(candidate_path) as f:
                data = json.load(f)
            comp_ratio_raw = data.get('requested_compression_ratio') or data.get('compression_ratio')
            if comp_ratio_raw is None:
                continue
            comp_ratio = float(comp_ratio_raw)
            if any(abs(comp_ratio - excl) < 0.01 for excl in exclude_ratios):
                continue
            if original_dim is not None:
                file_orig_dim = data.get('original_dim')
                if file_orig_dim is not None:
                    try:
                        if abs(int(file_orig_dim) - int(original_dim)) > 10:
                            continue
                    except Exception:
                        continue
            ece = data.get('ece')
            if ece is None:
                continue
            baseline_options.append(
                {
                    'path': candidate_path,
                    'ratio': comp_ratio,
                    'ece': float(ece),
                    'target_dim': data.get('target_dim'),
                }
            )
        except Exception as exc:
            logger.debug(f"Failed to parse candidate {candidate_path}: {exc}")
            continue
    
    if not baseline_options:
        return None
    
    if target_ratio is not None:
        tolerance = 0.05
        matching = [
            opt for opt in baseline_options
            if abs(opt['ratio'] - target_ratio) <= tolerance
        ]
        if not matching:
            logger.debug(f"No baseline results found with ratio ≈ {target_ratio}")
            return None
        selected = min(matching, key=lambda x: x['ratio'])
    else:
        selected = min(baseline_options, key=lambda x: x['ratio'])
    
    logger.info(
        f"Selected baseline: {selected['path'].name} "
        f"(ratio={selected['ratio']:.2f}×, target_dim={selected['target_dim']}, "
        f"ECE={selected['ece']:.6f})"
    )
    return selected['ece']

def load_layer_selection_results(results_path: Path, training_loss: str,
                                 dataset: str, model: str, seed: int) -> Dict:
    """
    Load the per_layer_ground_truth.json file and find best/worst layers by ECE.
    Returns layer identifiers (module index or name) and their ECEs.
    """
    json_path = (
        results_path / training_loss / dataset / model /
        f"seed{seed}" / "per_layer_ground_truth.json"
    )
    if not json_path.exists():
        raise FileNotFoundError(f"Layer selection results not found at {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Parse JSON structure (adapt based on actual format)
    ece_values = {}

    # Try structure 1: {"test": {"ece": {"layer_name": value}}}
    if isinstance(data, dict) and 'test' in data:
        ece_data = data.get('test', {}).get('ece', {})
        if ece_data:
            ece_values = {str(k): float(v) for k, v in ece_data.items()}

    # Try structure 2: {"layer_name": {"ece": value, ...}}
    if not ece_values:
        for layer_id, metrics in data.items():
            if isinstance(metrics, dict) and 'ece' in metrics:
                ece_values[str(layer_id)] = float(metrics['ece'])

    if not ece_values:
        raise ValueError(f"Could not find ECE values in {json_path}")

    # Find best (lowest ECE) and worst (highest ECE)
    sorted_layers = sorted(ece_values.items(), key=lambda x: x[1])
    best_layer_id = sorted_layers[0][0]
    best_ece = sorted_layers[0][1]
    worst_layer_id = sorted_layers[-1][0]
    worst_ece = sorted_layers[-1][1]

    logger.info(f"Found {len(ece_values)} layers in results")
    logger.info(f"Best layer: {best_layer_id} (ECE: {best_ece:.6f})")
    logger.info(f"Worst layer: {worst_layer_id} (ECE: {worst_ece:.6f})")

    return {
        'best_layer': best_layer_id,  # Could be "53" or "layer4.1"
        'worst_layer': worst_layer_id,
        'best_ece': best_ece,
        'worst_ece': worst_ece
    }

def load_physical_baseline_ece(results_path: Path, training_loss: str,
                               dataset: str, model: str, seed: int):
    """
    Load baseline ECE for physical (data) layer from baseline_results.json.
    Prefers 'geometric_physical_space' if available; falls back to other methods if needed.
    """
    json_path = (
        results_path / training_loss / dataset / model /
        f"seed{seed}" / "baseline_results.json"
    )
    if not json_path.exists():
        logger.warning(f"Baseline results not found at {json_path}")
        return None
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict) and 'geometric_physical_space' in data:
            e = data['geometric_physical_space'].get('ece', None)
            if e is not None:
                return float(e)
        # Fallbacks if geometric physical baseline is missing
        for key in ['isotonic_toplabel', 'temperature_scaling', 'uncalibrated']:
            if key in data and 'ece' in data[key]:
                logger.warning(f"Using fallback baseline '{key}' ECE for data_layer comparison")
                return float(data[key]['ece'])
        logger.warning(f"No usable ECE found in {json_path}")
        return None
    except Exception as e:
        logger.warning(f"Failed to load baseline ECE from {json_path}: {e}")
        return None

def _resolve_layer_index(model: nn.Module, layer_identifier) -> int:
    """
    Resolve a layer identifier (int index or module name) to a modules() index.
    """
    if isinstance(layer_identifier, int) or (isinstance(layer_identifier, str) and layer_identifier.isdigit()):
        return int(layer_identifier)
    # resolve by name
    named = dict(model.named_modules())
    if layer_identifier in named:
        target_module = named[layer_identifier]
        for idx, m in enumerate(model.modules()):
            if m is target_module:
                return idx
    raise ValueError(f"Could not resolve layer identifier to index: {layer_identifier}")

def _compress_images_with_spp(
    model: nn.Module,
    dataloader: DataLoader,
    target_dim: int,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Compress raw images using SPP+JL and also collect logits.
    Returns (features, labels, logits, original_dim).
    """
    model.eval()
    all_features = []
    all_labels = []
    all_logits = []
    img_compressor = None
    original_dim = None
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting physical_space", leave=False):
            images = images.to(device)
            labels = labels.to(device)
            if img_compressor is None:
                _, C, H, W = images.shape
                original_dim = C * H * W
                img_compressor = FixedSizeSPP_JL(
                    in_channels=C,
                    final_output_dim=target_dim,
                    pyramid_levels=[4, 2, 1],
                    seed=42
                ).to(device)
                logger.info(f"Created image SPP+JL compressor: {C} channels → {target_dim} dims")
            compressed = img_compressor(images)
            logits = model(images)
            all_features.append(compressed.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_logits.append(logits.cpu().numpy())
    features = np.vstack(all_features)
    labels_arr = np.concatenate(all_labels)
    logits_arr = np.vstack(all_logits)
    return features, labels_arr, logits_arr, original_dim

def _estimate_original_dim(model: nn.Module, layer_idx: int, example_images: torch.Tensor, device: torch.device) -> int:
    """
    Run a single forward pass with a hook to estimate original feature dimension of a layer.
    """
    layers = list(model.modules())
    if layer_idx >= len(layers):
        raise ValueError(f"Layer index {layer_idx} out of bounds for model with {len(layers)} modules")
    activation = {}
    def hook(module, input, output):
        activation['out'] = output
    handle = layers[layer_idx].register_forward_hook(hook)
    try:
        model.eval()
        with torch.no_grad():
            _ = model(example_images.to(device))
        feat = activation.get('out', None)
        if feat is None:
            return 0
        shape = feat.shape
        if len(shape) == 4:
            _, C, H, W = shape
            return int(C * H * W)
        elif len(shape) == 3:
            _, T, D = shape
            return int(T * D)
        elif len(shape) == 2:
            return int(shape[1])
        return int(np.prod(shape[1:]))
    finally:
        handle.remove()

MIN_TARGET_DIM_DEFAULT = 16
SEMANTIC_REFERENCE_DIM = 8192  # Reference baseline for semantic layers (avoids OOM)


def _format_ratio_label(ratio: Optional[float]) -> str:
    if ratio is None:
        return "ratioNA"
    if math.isclose(ratio, round(ratio), rel_tol=1e-9, abs_tol=1e-9):
        return f"ratio{int(round(ratio))}"
    ratio_str = f"{ratio:.3f}".rstrip("0").rstrip(".")
    return f"ratio{ratio_str.replace('.', 'p')}"


def _compute_target_dim_from_ratio(
    original_dim: Optional[int],
    compression_ratio: Optional[float],
    min_target_dim: int,
) -> Optional[int]:
    if original_dim is None or original_dim <= 0:
        return None
    if compression_ratio is None or compression_ratio <= 0:
        return None
    if compression_ratio < 1.0:
        return None
    target = int(original_dim / compression_ratio)
    if target < min_target_dim:
        return None
    return target if target > 0 else None


def run_compression_experiment(args):
    """
    Main experiment function.
    
    Supports two compression modes:
    1. Ratio-based (NEW): Compress by a factor (2×, 4×, etc.)
       - Each layer compressed by the same relative amount
       - Specified via --compression_ratio parameter
    2. Fixed-size (LEGACY): Compress to a specific dimension (128, 256, etc.)
       - All layers compressed to the same target dimension
       - Specified via --target_dim parameter
    
    Ratio-based mode is preferred for new experiments.
    """
    
    logger.info("="*80)
    logger.info("COMPRESSION EXPERIMENT")
    logger.info("="*80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Training Loss: {args.training_loss}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Layer Type: {args.layer_type}")

    requested_ratio = getattr(args, "compression_ratio", None)
    min_target_dim = getattr(args, "min_target_dim", MIN_TARGET_DIM_DEFAULT)
    ratio_mode = requested_ratio is not None
    use_compression = getattr(args, "use_compression", False)
    if ratio_mode:
        use_compression = requested_ratio > 1.0

    if ratio_mode:
        logger.info(f"Requested Compression Ratio: {requested_ratio}")
    else:
        logger.info(f"Target Dimension: {args.target_dim}")
    logger.info(f"Minimum Target Dimension: {min_target_dim}")
    logger.info(f"Compression Enabled: {use_compression}")
    logger.info("="*80)

    def _emit_skip_result(
        *,
        skip_reason: str,
        layer_name: Optional[str] = None,
        target_dim_entry: Optional[int] = None,
        original_dim_entry: Optional[int] = None,
        compression_ratio_entry: Optional[float] = None,
        extra_fields: Optional[Dict] = None,
    ):
        output_dir = (
            Path(args.output_dir)
            / args.training_loss
            / args.dataset
            / args.model
            / f"seed{args.seed}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        if compression_ratio_entry is None:
            if ratio_mode and requested_ratio is not None:
                compression_ratio_entry = float(requested_ratio)
            elif not use_compression:
                compression_ratio_entry = 1.0
        ratio_label = _format_ratio_label(compression_ratio_entry)
        dim_for_name = target_dim_entry if target_dim_entry is not None else 0
        output_file = output_dir / f"{args.layer_type}_{ratio_label}_dim{dim_for_name}_results.json"
        result = {
            'model': args.model,
            'dataset': args.dataset,
            'training_loss': args.training_loss,
            'seed': args.seed,
            'layer_type': args.layer_type,
            'layer_name': layer_name,
            'target_dim': target_dim_entry,
            'original_dim': original_dim_entry,
            'compression_ratio': compression_ratio_entry,
            'requested_compression_ratio': float(requested_ratio) if ratio_mode else None,
            'ece': None,
            'accuracy': None,
            'baseline_ece': None,
            'relative_ece_change': None,
            'skipped': True,
            'skip_reason': skip_reason,
        }
        if ratio_mode:
            result['min_target_dim'] = min_target_dim
        if extra_fields:
            result.update(extra_fields)
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        logger.info(f"✓ Skipped results saved to: {output_file}")
        return result
    
    # Setup device
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # --- 1. Load Model ---
    logger.info("Loading model...")
    dynamic_folder_name = f"{args.training_loss}_{args.dataset}_{args.model}_seed{args.seed}"
    model_path = (
        Path(args.checkpoint_base_dir) 
        / args.training_loss 
        / args.dataset 
        / args.model 
        / f"seed{args.seed}"
        / dynamic_folder_name
        / "best_model.pth"
    )
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    # Determine num_classes using shared data loader helper
    _, _, _, num_classes = get_data_loaders(args.dataset, args.batch_size, seed=args.seed)

    # Load trained model
    model = load_trained_model(str(model_path), args.model, num_classes, device, dataset=args.dataset)
    model.eval()
    logger.info(f"Loaded model from {model_path}")
    # Wrap with adapter for calibrator API
    model_adapter = PyTorchModelAdapter(model, device, dataset_name=args.dataset)
    
    # --- 2. Load Data ---
    logger.info("Loading data...")
    train_loader, val_loader, test_loader, _ = get_data_loaders(
        args.dataset,
        args.batch_size,
        seed=args.seed
    )
    # Quick validation accuracy gate
    logger.info("Running quick validation accuracy check...")
    try:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        val_acc = float(correct / max(1, total))
        logger.info(f"Validation accuracy: {val_acc:.4f}")
        acc_threshold = 0.60
        if val_acc < acc_threshold:
            logger.warning(f"Skipping experiment due to low validation accuracy ({val_acc:.4f} < {acc_threshold:.2f})")
            target_entry = None
            if use_compression and not ratio_mode and args.target_dim is not None:
                target_entry = int(args.target_dim)
            elif not use_compression:
                target_entry = 0
            skipped_results = _emit_skip_result(
                skip_reason='low_validation_accuracy',
                target_dim_entry=target_entry,
                compression_ratio_entry=float(requested_ratio) if ratio_mode else (1.0 if not use_compression else None),
                extra_fields={
                    'validation_accuracy': float(val_acc),
                    'accuracy_threshold': float(acc_threshold),
                },
            )
            return skipped_results
    except Exception as gate_e:
        logger.warning(f"Validation accuracy gate failed: {gate_e}")
    
    # --- 3. Determine Which Layer to Use ---
    if args.layer_type == 'data_layer':
        layer_name = 'physical_space'
        logger.info("Using data_layer (physical_space / input images)")
    else:
        # Load layer selection results to get best/worst layer
        logger.info("Loading layer selection results...")
        layer_info = load_layer_selection_results(
            Path(args.layer_selection_results_path),
            args.training_loss,
            args.dataset,
            args.model,
            args.seed
        )
        
        if args.layer_type == 'best_layer':
            layer_name = layer_info['best_layer']
        else:  # worst_layer
            layer_name = layer_info['worst_layer']
        
        logger.info(f"Using {args.layer_type}: {layer_name}")
    
    # --- 4. Prepare Raw Arrays (exact order) ---
    logger.info("\n" + "-"*80)
    logger.info("FEATURE EXTRACTION AND COMPRESSION")
    logger.info("-"*80)
    
    def get_all_data_as_numpy(loader: DataLoader):
        xs, ys = [], []
        for batch in loader:
            x, y = batch[:2] if isinstance(batch, (list, tuple)) else (batch, None)
            xs.append(x.cpu().numpy())
            if y is not None:
                ys.append(y.cpu().numpy())
        X = np.concatenate(xs, axis=0)
        Y = np.concatenate(ys, axis=0) if ys else None
        return X, Y
    
    # Freeze splits as numpy arrays to guarantee alignment across all steps
    train_raw, train_labels = get_all_data_as_numpy(train_loader)
    val_raw,   val_labels   = get_all_data_as_numpy(val_loader)
    test_raw,  test_labels  = get_all_data_as_numpy(test_loader)
    
    # Free dataloaders early (we will rebuild deterministic loaders from numpy)
    del train_loader, val_loader, test_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # --- 4a. Extract/Compress Features using frozen arrays ---
    t_feat_start = time.perf_counter()
    target_dim = None

    # Helper for batch compression with arbitrary method
    def compress_batch_numpy(X_np, method, dim, pyr_levels, mid_ch, device):
        if X_np.ndim != 4:
             # Fallback for non-image data if any? Assuming images here.
             return X_np
        N, C, H, W = X_np.shape
        
        projector = create_compression_module(
            method=method,
            in_channels=C,
            final_output_dim=dim,
            pyramid_levels=pyr_levels,
            mid_channels=mid_ch,
            seed=42
        ).to(device).eval()
        
        bs = min(512, args.batch_size)
        # Use torch DataLoader for efficient batching
        from torch.utils.data import TensorDataset, DataLoader
        dl = DataLoader(TensorDataset(torch.from_numpy(X_np)), batch_size=bs, shuffle=False)
        
        outs = []
        with torch.no_grad():
            for (xb,) in dl:
                xb = xb.to(device).float()
                z = projector(xb)
                outs.append(z.cpu())
        return torch.cat(outs, dim=0).numpy()

    if layer_name == 'physical_space':
        # Match proven code path: SPP+JL on raw images
        _, C, H, W = train_raw.shape
        original_dim = int(C * H * W)
        logger.info(f"Original (physical) dim: {original_dim}")
        if not use_compression:
            target_dim = original_dim
            logger.info(f"BASELINE MODE: Using uncompressed flatten ({original_dim} dims)")
            train_features = train_raw.reshape(train_raw.shape[0], -1)
            val_features   = val_raw.reshape(val_raw.shape[0], -1)
            test_features  = test_raw.reshape(test_raw.shape[0], -1)
        else:
            if ratio_mode:
                # Ratio-based compression: derive target_dim from original_dim and requested ratio
                target_dim = _compute_target_dim_from_ratio(original_dim, requested_ratio, min_target_dim)
                if target_dim is None:
                    logger.warning(
                        f"Requested compression ratio {requested_ratio} yields target_dim < {min_target_dim} "
                        f"for original_dim {original_dim}; skipping."
                    )
                    tentative_target = int(original_dim / requested_ratio) if requested_ratio else None
                    skipped_results = _emit_skip_result(
                        skip_reason='target_dim_below_min',
                        layer_name=layer_name,
                        original_dim_entry=original_dim,
                        target_dim_entry=None,
                        compression_ratio_entry=float(requested_ratio),
                        extra_fields={
                            'requested_target_dim': tentative_target,
                        },
                    )
                    return skipped_results
                achieved_ratio = original_dim / target_dim
                logger.info(
                    f"COMPRESSION MODE (ratio {requested_ratio}× requested, {achieved_ratio:.2f}× achieved): "
                    f"{original_dim} → {target_dim} dims"
                )
            else:
                target_dim = int(args.target_dim)
                logger.info(f"COMPRESSION MODE: Compressing {original_dim} → {target_dim} dims")
            
            # Use the helper to compress
            train_features = compress_batch_numpy(
                train_raw, args.compression_method, target_dim, 
                args.pyramid_levels, args.mid_channels, device
            )
            val_features = compress_batch_numpy(
                val_raw, args.compression_method, target_dim, 
                args.pyramid_levels, args.mid_channels, device
            )
            test_features = compress_batch_numpy(
                test_raw, args.compression_method, target_dim, 
                args.pyramid_levels, args.mid_channels, device
            )
    else:
        # Build deterministic loaders from the same numpy arrays
        from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
        tr_dl = TorchDataLoader(
            TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
            batch_size=args.batch_size, shuffle=False
        )
        va_dl = TorchDataLoader(
            TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
            batch_size=args.batch_size, shuffle=False
        )
        te_dl = TorchDataLoader(
            TensorDataset(torch.from_numpy(test_raw), torch.from_numpy(test_labels)),
            batch_size=args.batch_size, shuffle=False
        )
        
        layer_idx = _resolve_layer_index(model, layer_name)
        if not use_compression:
            logger.info(f"BASELINE MODE: Extracting without compression")
            tr_feat_t, _ = extract_features_directly(
                model, tr_dl, layer_idx, device, fixed_feature_dim=None
            )
            va_feat_t, _ = extract_features_directly(
                model, va_dl, layer_idx, device, fixed_feature_dim=None
            )
            te_feat_t, _ = extract_features_directly(
                model, te_dl, layer_idx, device, fixed_feature_dim=None
            )
            original_dim = int(tr_feat_t.shape[1])
            target_dim = original_dim
            logger.info(f"Original (flattened) dim: {original_dim}")
        else:
            # Estimate original dimension from a small train batch
            with torch.no_grad():
                example = torch.from_numpy(train_raw[:min(8, len(train_raw))])
                original_dim_estimate = _estimate_original_dim(model, layer_idx, example, device)
            if original_dim_estimate is None or original_dim_estimate <= 0:
                logger.error("Unable to estimate original dimensionality; skipping compression run.")
                skipped_results = _emit_skip_result(
                    skip_reason='original_dim_unavailable',
                    layer_name=layer_name,
                    original_dim_entry=None,
                    compression_ratio_entry=float(requested_ratio) if ratio_mode else None,
                )
                return skipped_results
            original_dim = int(original_dim_estimate)
            if ratio_mode:
                # Ratio-based compression: derive target_dim from original_dim and requested ratio
                target_dim = _compute_target_dim_from_ratio(original_dim, requested_ratio, min_target_dim)
                if target_dim is None:
                    logger.warning(
                        f"Requested compression ratio {requested_ratio} yields target_dim < {min_target_dim} "
                        f"for layer {layer_name} (original_dim {original_dim}); skipping."
                    )
                    tentative_target = int(original_dim / requested_ratio) if requested_ratio else None
                    skipped_results = _emit_skip_result(
                        skip_reason='target_dim_below_min',
                        layer_name=layer_name,
                        original_dim_entry=original_dim,
                        target_dim_entry=None,
                        compression_ratio_entry=float(requested_ratio),
                        extra_fields={
                            'requested_target_dim': tentative_target,
                        },
                    )
                    return skipped_results
                achieved_ratio = original_dim / target_dim
                logger.info(
                    f"COMPRESSION MODE (ratio {requested_ratio}× requested, {achieved_ratio:.2f}× achieved): "
                    f"{original_dim} → {target_dim}"
                )
            else:
                target_dim = int(args.target_dim)
                logger.info(f"COMPRESSION MODE: Compressing {original_dim} → {target_dim}")
            tr_feat_t, _ = extract_features_directly(
                model, tr_dl, layer_idx, device, fixed_feature_dim=target_dim,
                compression_method=args.compression_method,
                pyramid_levels=args.pyramid_levels,
                mid_channels=args.mid_channels
            )
            va_feat_t, _ = extract_features_directly(
                model, va_dl, layer_idx, device, fixed_feature_dim=target_dim,
                compression_method=args.compression_method,
                pyramid_levels=args.pyramid_levels,
                mid_channels=args.mid_channels
            )
            te_feat_t, _ = extract_features_directly(
                model, te_dl, layer_idx, device, fixed_feature_dim=target_dim,
                compression_method=args.compression_method,
                pyramid_levels=args.pyramid_levels,
                mid_channels=args.mid_channels
            )
        train_features = tr_feat_t.numpy()
        val_features   = va_feat_t.numpy()
        test_features  = te_feat_t.numpy()
    
    if ratio_mode:
        logger.info("\n" + "="*60)
        logger.info("COMPRESSION MODE: RATIO-BASED")
        logger.info(f"  Requested ratio: {requested_ratio}×")
        logger.info(f"  Original dim: {original_dim}")
        logger.info(f"  Target dim: {target_dim}")
        actual_ratio = (original_dim / target_dim) if target_dim else None
        logger.info(
            f"  Actual ratio achieved: {actual_ratio:.2f}×" if actual_ratio else "  Actual ratio achieved: N/A"
        )
        logger.info(f"  Min allowed dim: {min_target_dim}")
        logger.info("="*60 + "\n")
    else:
        logger.info("\n" + "="*60)
        logger.info("COMPRESSION MODE: FIXED-SIZE (LEGACY)")
        logger.info(f"  Target dim: {target_dim}")
        logger.info("="*60 + "\n")
    
    # Memory cleanup after heavy extraction
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    feature_extract_time_s = time.perf_counter() - t_feat_start
    logger.info(f"Feature extraction/compression time: {feature_extract_time_s:.3f}s")
    
    effective_dim = int(train_features.shape[1])
    compression_ratio = (original_dim / effective_dim) if effective_dim > 0 else 1.0
    logger.info(f"Compression ratio: {compression_ratio:.2f}x ({original_dim} → {effective_dim})")
    
    # Feature statistics logging
    def log_feature_stats(features: np.ndarray, name: str):
        logger.info(f"{name} feature statistics:")
        logger.info(f"  Shape: {features.shape}")
        logger.info(f"  Mean: {features.mean():.6f}")
        logger.info(f"  Std: {features.std():.6f}")
        logger.info(f"  Min: {features.min():.6f}")
        logger.info(f"  Max: {features.max():.6f}")
        logger.info(f"  NaNs: {np.isnan(features).sum()}")
        logger.info(f"  Infs: {np.isinf(features).sum()}")
    
    log_feature_stats(train_features, "Train")
    log_feature_stats(val_features, "Val")
    log_feature_stats(test_features, "Test")
    
    # --- Alignment diagnostics (sanity checks before calibration) ---
    try:
        # Sanity: recompute features directly from raw arrays and compare to cached
        if layer_name == 'physical_space':
            if use_compression:
                z_from_raw = compress_with_spp_jl(
                    test_raw,
                    final_output_dim=target_dim,
                    pyramid_levels=[4, 2, 1],
                    seed=42,
                    batch_size=min(512, args.batch_size),
                    device=(device.type if isinstance(device, torch.device) else str(device))
                )
            else:
                z_from_raw = test_raw.reshape(test_raw.shape[0], -1)
            diffs = np.linalg.norm(z_from_raw - test_features, axis=1)
            logger.info(f"Sanity (features, physical): mean|Z(raw)-Z(cached)|={diffs.mean():.6f}, max={diffs.max():.6f}")
        else:
            from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
            layer_idx = _resolve_layer_index(model, layer_name)
            te_dl = TorchDataLoader(
                TensorDataset(torch.from_numpy(test_raw), torch.from_numpy(test_labels)),
                batch_size=args.batch_size, shuffle=False
            )
            fixed_dim_chk = target_dim if use_compression else None
            te_feat_t_chk, _ = extract_features_directly(
                model, te_dl, layer_idx, device, fixed_feature_dim=fixed_dim_chk
            )
            diffs = np.linalg.norm(te_feat_t_chk.numpy() - test_features, axis=1)
            logger.info(f"Sanity (features, semantic): mean|Z(raw)-Z(cached)|={diffs.mean():.6f}, max={diffs.max():.6f}")
    except Exception as diag_e:
        logger.warning(f"Alignment diagnostics failed: {diag_e}")

    # Determine baseline ECE for relative comparison
    # Strategy: prefer the lowest compression ratio available as reference.
    # Physical layers (data_layer) can leverage true uncompressed features (ratio=1.0).
    # Semantic layers typically cannot due to memory limits; instead we rely on the
    # lowest stored ratio (e.g., 2.0×) to serve as the comparison baseline.
    if layer_name == 'physical_space':
        baseline_ratio_target = 1.0
        baseline_dim = int(original_dim)
        if not use_compression:
            baseline_ece = None
            logger.info("This is the UNCOMPRESSED physical baseline; establishing baseline ECE.")
        else:
            baseline_ece = _load_lowest_ratio_baseline_ece(
                Path(args.output_dir),
                args.training_loss,
                args.dataset,
                args.model,
                args.seed,
                args.layer_type,
                target_ratio=baseline_ratio_target,
                original_dim=original_dim
            )
            if baseline_ece is None:
                logger.warning(
                    f"Physical baseline ECE (ratio={baseline_ratio_target}) not found; relative change will be skipped.\n"
                    f"  Run baseline first: --compression_ratio 1.0 --layer_type data_layer"
                )
            else:
                logger.info(f"✓ Loaded physical baseline ECE (ratio={baseline_ratio_target}): {baseline_ece:.6f}")
    else:
        baseline_ratio_candidates = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        current_ratio = compression_ratio if use_compression else 1.0
        is_potential_baseline = any(abs(current_ratio - cand) <= 0.05 for cand in baseline_ratio_candidates)
        if is_potential_baseline:
            baseline_ece = _load_lowest_ratio_baseline_ece(
                Path(args.output_dir),
                args.training_loss,
                args.dataset,
                args.model,
                args.seed,
                args.layer_type,
                target_ratio=None,
                original_dim=original_dim,
                exclude_ratios=[current_ratio]
            )
            if baseline_ece is None:
                baseline_ece = None
                logger.info(f"This is the semantic REFERENCE baseline (ratio={current_ratio:.2f}×); establishing baseline ECE.")
            else:
                logger.info(f"✓ Loaded semantic baseline ECE from lower ratio: {baseline_ece:.6f}")
        else:
            baseline_ece = _load_lowest_ratio_baseline_ece(
                Path(args.output_dir),
                args.training_loss,
                args.dataset,
                args.model,
                args.seed,
                args.layer_type,
                target_ratio=None,
                original_dim=original_dim
            )
            if baseline_ece is None:
                logger.warning(
                    "Semantic reference baseline not found; relative change will be skipped.\n"
                    f"  Need a low-ratio reference (≈2×-4×) for {args.layer_type}"
                )
            else:
                logger.info(f"✓ Loaded semantic baseline ECE from lowest available ratio: {baseline_ece:.6f}")

    # --- 5. Train Geometric Calibrator ---
    logger.info("\n" + "-"*80)
    logger.info("CALIBRATOR TRAINING")
    logger.info("-"*80)
    
    calibrator = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
        device=(device.type if isinstance(device, torch.device) else str(device))
    )
    
    logger.info("Fitting calibrator on validation data...")
    t_fit_start = time.perf_counter()
    calibrator.fit(
        X_val_embed=val_features,
        y_val=val_labels,
        X_val_original=val_raw,
        fit_batch_size=max(64, args.batch_size)
    )
    fit_time_s = time.perf_counter() - t_fit_start
    logger.info(f"Calibrator fit time: {fit_time_s:.3f}s")
    
    # --- 6. Evaluate on Test Set ---
    logger.info("\n" + "-"*80)
    logger.info("EVALUATION")
    logger.info("-"*80)
    
    logger.info("Calibrating test predictions...")
    t_cal_start = time.perf_counter()
    calibrated_probs = calibrator.calibrate_batched(
        X_test_embed=test_features,
        X_test_original=test_raw,
        batch_size=max(64, args.batch_size)
    )
    calibrate_time_s = time.perf_counter() - t_cal_start
    num_test = int(test_raw.shape[0])
    calibrate_time_per_image_s = (calibrate_time_s / num_test) if num_test > 0 else float('inf')
    calibrate_fps = (num_test / calibrate_time_s) if calibrate_time_s > 0 else 0.0
    logger.info(f"Calibration (inference) time: {calibrate_time_s:.3f}s "
                f"({calibrate_time_per_image_s:.4f} s/img, {calibrate_fps:.1f} FPS)")
    
    # Compute metrics
    ece = compute_ece(calibrated_probs, test_labels)
    accuracy = float(np.mean(np.argmax(calibrated_probs, axis=1) == test_labels))
    
    # Uncalibrated baseline for sanity check
    # Use model adapter to get probabilities on raw images (works for both physical and semantic paths)
    uncalibrated_probs = model_adapter.predict_proba(test_raw, batch_size=max(64, args.batch_size))
    uncal_ece = compute_ece(uncalibrated_probs, test_labels)
    uncal_acc = float(np.mean(np.argmax(uncalibrated_probs, axis=1) == test_labels))
    
    logger.info(f"Uncalibrated: ECE={uncal_ece:.6f}, Acc={uncal_acc:.6f}")
    logger.info(f"Calibrated:   ECE={ece:.6f}, Acc={accuracy:.6f}")
    
    # Sanity check: accuracy should not decrease significantly
    if abs(accuracy - uncal_acc) > 0.02:
        logger.warning(f"⚠️  Large accuracy change! Uncal: {uncal_acc:.4f} → Cal: {accuracy:.4f}")
        logger.warning("This may indicate a problem with the calibrator or features.")
    
    # Calculate relative ECE change if baseline available
    if baseline_ece is not None:
        relative_change = (ece - baseline_ece) / baseline_ece * 100
        logger.info(f"Baseline ECE: {baseline_ece:.6f}")
        logger.info(f"Relative change: {relative_change:+.2f}%")
        # Check for extreme degradation
        if relative_change > 100:
            logger.warning(f"⚠️  SEVERE ECE DEGRADATION: +{relative_change:.1f}%")
            logger.warning("Possible causes:")
            logger.warning("  1. Compression too aggressive for this layer")
            logger.warning("  2. Feature extraction error")
            logger.warning("  3. Data split mismatch")
    else:
        relative_change = None
        logger.info(f"ECE: {ece:.6f} (no baseline for comparison)")
    
    # --- 7. Save Results ---
    pyramid_str = 'x'.join(map(str, args.pyramid_levels))
    compression_id = f"{args.compression_method}_pyramid_{pyramid_str}"
    if args.mid_channels is not None:
        compression_id += f"_mid{args.mid_channels}"

    results = {
        'model': args.model,
        'dataset': args.dataset,
        'training_loss': args.training_loss,
        'seed': args.seed,
        'layer_type': args.layer_type,
        'layer_name': layer_name,
        'target_dim': int(target_dim) if target_dim is not None else None,
        'original_dim': int(original_dim),
        'compression_ratio': float(compression_ratio),
        'requested_compression_ratio': float(requested_ratio) if ratio_mode else None,
        'ece': float(ece),
        'accuracy': float(accuracy),
        'baseline_ece': float(baseline_ece) if baseline_ece is not None else None,
        'relative_ece_change': float(relative_change) if relative_change is not None else None,
        # Compression details
        'compression_method': args.compression_method,
        'pyramid_levels': args.pyramid_levels,
        'mid_channels': args.mid_channels,
        'compression_id': compression_id,
        # Timing metrics
        'feature_extract_time_s': float(feature_extract_time_s),
        'fit_time_s': float(fit_time_s),
        'calibrate_time_s': float(calibrate_time_s),
        'calibrate_time_per_image_s': float(calibrate_time_per_image_s),
        'calibrate_fps': float(calibrate_fps),
    }
    if ratio_mode:
        results['min_target_dim'] = int(min_target_dim)
    
    # Create output directory
    output_dir = (
        Path(args.output_dir)
        / args.training_loss
        / args.dataset
        / args.model
        / f"seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Use effective feature dimensionality for filename to make baselines unambiguous
    # Note: use requested target_dim for stable naming; actual achieved dim is recorded in JSON payload
    output_dim_for_filename = int(target_dim) if target_dim is not None else int(train_features.shape[1])
    ratio_for_filename = results['requested_compression_ratio'] if ratio_mode else results['compression_ratio']
    ratio_label = _format_ratio_label(ratio_for_filename)
    
    # Include compression ID in filename if not default
    # Default is fixed_spp_jl with 4-2-1. If that's the case, maybe keep old name for compat?
    # Or just always include it? The prompt implies new filenames.
    # Let's append compression_id if it's not the standard one, or just append it always to be safe.
    # But for backward compatibility (loading baselines), maybe checking if it is default is good.
    # Default: fixed_spp_jl, [4,2,1], mid=None.
    is_default = (args.compression_method == 'fixed_spp_jl' and args.pyramid_levels == [4,2,1] and args.mid_channels is None)
    
    if is_default:
         # Stick to old naming for default to avoid breaking everything?
         # Or just switch to new naming. The user says "Update compression_experiment.py sweep logic... results[compression_id] = ..."
         # I will append compression_id to filename.
         output_file = output_dir / f"{args.layer_type}_{ratio_label}_dim{output_dim_for_filename}_results.json"
    else:
         output_file = output_dir / f"{args.layer_type}_{compression_id}_{ratio_label}_dim{output_dim_for_filename}_results.json"

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info(f"✓ Results saved to: {output_file}")
    logger.info("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compression experiment worker")
    
    # Experiment config (passed by runner)
    parser.add_argument('--model', type=str, required=True,
                        help='Model architecture')
    parser.add_argument('--dataset', type=str, required=True,
                        help='Dataset name')
    parser.add_argument('--training_loss', type=str, required=True,
                        help='Training loss function')
    parser.add_argument('--seed', type=int, required=True,
                        help='Random seed')
    parser.add_argument('--layer_type', type=str, required=True,
                        choices=['data_layer', 'best_layer', 'worst_layer'],
                        help='Which layer to use')
    parser.add_argument('--target_dim', type=int, default=None,
                        help='Legacy: fixed target dimension for SPP+JL compression')
    parser.add_argument('--compression_ratio', type=float, default=None,
                        help='Compression ratio (original_dim / target_dim). Overrides target_dim when provided.')
    
    # Paths (passed by runner)
    parser.add_argument('--layer_selection_results_path', type=Path, required=True,
                        help='Path to layer selection results')
    parser.add_argument('--checkpoint_base_dir', type=Path, required=True,
                        help='Base directory for model checkpoints')
    parser.add_argument('--output_dir', type=Path, required=True,
                        help='Output directory for results')
    parser.add_argument('--data_dir', type=Path, required=True,
                        help='Data directory')
    
    # Runtime config
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for feature extraction')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--use_compression', action='store_true', default=False,
                        help='Apply SPP+JL compression. If False, use flatten only (for baselines).')
    parser.add_argument('--min_target_dim', type=int, default=MIN_TARGET_DIM_DEFAULT,
                        help='Minimum allowed target dimension when using compression ratios.')
    
    # Advanced compression config
    parser.add_argument('--compression_method', type=str, default='fixed_spp_jl',
                        help='Compression method: fixed_spp_jl, channel_first_spp_jl, multiscale_spp_jl')
    parser.add_argument('--pyramid_levels', type=int, nargs='+', default=[4, 2, 1],
                        help='Pyramid levels for SPP (e.g. 4 2 1)')
    parser.add_argument('--mid_channels', type=int, default=None,
                        help='Intermediate channels for channel_first_spp_jl')
    
    args = parser.parse_args()
    
    if args.compression_ratio is None and args.target_dim is None:
        parser.error("Either --compression_ratio or --target_dim must be provided.")
    if args.compression_ratio is not None and args.compression_ratio <= 0:
        parser.error("--compression_ratio must be positive.")
    if args.target_dim is not None and args.target_dim <= 0:
        parser.error("--target_dim must be positive.")
    if args.min_target_dim <= 0:
        parser.error("--min_target_dim must be a positive integer.")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    try:
        run_compression_experiment(args)
        return 0
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())