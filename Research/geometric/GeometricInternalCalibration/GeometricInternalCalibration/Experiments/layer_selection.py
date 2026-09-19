import torch
import numpy as np
import os
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import json
from typing import Dict, Any, List, Optional
from typing import Tuple
import pandas as pd

ECE_BIN_COUNT = 15  # unified bin count for scalar ECE calculations

# Assume the following are in your project structure
from Calibrators.geometric_calibrator_new import GeometricCalibrator, AdHocLayerSelector, LayerMetrics
from torch.utils.data import TensorDataset, DataLoader
from Experiments.run_post_hoc_calibration import PyTorchModelAdapter, get_data_loaders, load_trained_model
from Calibrators.isotonic_regression import TopLabelIsotonicCalibrator
from Calibrators.temperature_scaling import TemperatureScaling
from Calibrators.platt_scaling import PlattScaling
from Calibrators.dirichlet_calibration import DirichletCalibration
from Calibrators.beta_calibration import BetaCalibration
from Calibrators.geometric_calibrator import GeometricCalibrator as GeometricCalibrator_old
from utils.compression_utils import FixedSizeSPP_JL
from Calibrators.geometric_quality_predictors import GeometricQualityAnalyzer

def unit_weights_for_all_metrics() -> Dict[str, float]:
    """
    Uniform weights over the supported 'new' metrics.
    Keep this list aligned with AdHocLayerSelector defaults and the special calculators.
    """
    names = [
        "confidence_distance_correlation",
        "uncertainty_geometry_alignment",
        "boundary_proximity_correlation",
        "ece_score",
        "kfold_ece_utility",
        "spearman_stability_accuracy",
        "calibration_decisiveness",
        "decisiveness_tail_mass",
        "confidence_entropy",
        "class_balanced_margin_cvar",
        "impostor_gap_cvar",
        "margin_skewkurt_safety",
        "temperature_estimate_strength",
        "confident_error_rate_tau",
        "aurc_proxy",
        "reliability_curve_quality",
        "label_cka",
        "margin_tail_cvar",
        "multiscale_separation",
        "logistic_calibratability",
        "geometry_error_concordance",
        "label_cka_hsic",
        # include only if your selector exposes them; otherwise remove:
        # "validation_brier",
    ]
    return {n: 1.0 for n in names}

def run_full_metric_scan(model_adapter,
                         train_raw, train_labels,
                         val_raw,   val_labels,    # Use ALL validation for metrics
                         test_raw=None, test_labels=None,  # Test set available for holdout evaluation (optional)
                         device=None, batch_size=None):
    """
    Compute metrics on FULL validation set (e.g., 5k samples for CIFAR-10/100).
    Test set is available for holdout evaluation but not used in this function.
    
    This approach gives more data for metric computation compared to splitting validation.
    """
    # =====================================================================
    # === PRE-CALCULATE PROBABILITIES ON FULL VALIDATION SET ===
    # =====================================================================
    logger.info("Computing metrics on FULL validation set...")
    logger.info(f"  Validation samples: {len(val_raw)} (using all for metrics)")
    if test_raw is not None:
        logger.info(f"  Test samples: {len(test_raw)} (available for holdout evaluation)")
    
    try:
        # This gives us the full (n_samples, n_classes) probability matrix for validation
        val_probs = model_adapter.predict_proba(val_raw, batch_size=min(256, batch_size))
        logger.info(f"Successfully calculated validation probabilities, shape: {val_probs.shape}")
    except Exception as e:
        logger.error(f"Failed to pre-calculate model probabilities: {e}")
        # If this fails, metrics that depend on it will gracefully return 0.0
        val_probs = None
    # =====================================================================
    cal = GeometricCalibrator(
        model=model_adapter,
        X_train_original=train_raw,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=True,          # doesn't affect metric computation
        device=device,
        score_weights={},                # empty dict instead of None to avoid AttributeError
        compute_all_metrics=True,        # force compute of all metrics
    )
    # ✅ Fit on FULL validation set (all samples used for metrics)
    cal.fit(X_val_original=val_raw, y_val=val_labels, fit_batch_size=batch_size, model_probs=val_probs)
    return cal.candidate_layers, cal.layer_metrics

def _serialize_layer_metrics(layer_metrics: List[LayerMetrics]) -> List[Dict[str, Any]]:
    """Convert LayerMetrics objects to plain dicts with all metric fields, limiting numbers to 7 digits."""
    results: List[Dict[str, Any]] = []
    for lm in layer_metrics:
        d = {k: v for k, v in vars(lm).items() if not k.startswith('_')}
        # Ensure floats are JSON-friendly and limited to 7 digits
        for k, v in list(d.items()):
            try:
                if isinstance(v, (np.floating, float)):
                    d[k] = _format_number_7digits(float(v))
                elif isinstance(v, (np.integer, int)):
                    d[k] = int(v)
            except Exception:
                pass
        results.append(d)
    return results

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from utils.logging_config import get_logger
logger = get_logger(__name__)

def _format_number_7digits(num):
    """Format a number to have at most 7 significant digits."""
    if not isinstance(num, (int, float, np.integer, np.floating)) or not np.isfinite(num):
        return num
    
    # Convert to float for processing
    val = float(num)
    
    # Handle special cases
    if val == 0:
        return 0
    
    # For very large or very small numbers, use scientific notation
    abs_val = abs(val)
    if abs_val >= 1e6 or abs_val < 1e-6:
        return float(f"{val:.6e}")
    
    # For normal range numbers, limit to 7 total digits
    # Count digits before decimal point
    if abs_val >= 1:
        digits_before_decimal = len(str(int(abs_val)))
        digits_after_decimal = max(0, 7 - digits_before_decimal)
        return round(val, digits_after_decimal)
    else:
        # For numbers < 1, use up to 6 decimal places
        return round(val, 6)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NumPy types and limits numbers to 7 digits max."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return _format_number_7digits(float(obj))
        elif isinstance(obj, np.ndarray):
            return [_format_number_7digits(x) if isinstance(x, (int, float, np.integer, np.floating)) else x for x in obj.tolist()]
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'item'):  # NumPy scalars
            item = obj.item()
            return _format_number_7digits(item) if isinstance(item, (int, float)) else item
        elif isinstance(obj, (int, float)):
            return _format_number_7digits(obj)
        return super().default(obj)


# Recursively format all numeric values to at most 7 significant digits
def format_numbers_7digits(obj):
    if isinstance(obj, dict):
        return {k: format_numbers_7digits(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [format_numbers_7digits(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(format_numbers_7digits(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return [
            _format_number_7digits(float(x)) if isinstance(x, (int, float, np.integer, np.floating)) else x
            for x in obj.tolist()
        ]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return _format_number_7digits(float(obj))
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


# --- Helper Functions ---
def get_all_data_as_numpy(loader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Iterates through a DataLoader and returns the entire dataset as NumPy arrays."""
    all_images = []
    all_labels = []
    for images, labels in tqdm(loader, desc="Extracting NumPy data from loader"):
        all_images.append(images.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
    return np.concatenate(all_images, axis=0), np.concatenate(all_labels, axis=0)


class UncertaintyMetrics:
    """A helper class to compute common uncertainty and calibration metrics."""
    
    @staticmethod
    def calculate_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BIN_COUNT) -> float:
        """Calculate Expected Calibration Error (ECE)."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        
        ece = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        for i in range(n_bins):
            # Make bins half-open on the upper side to include confidences exactly equal to 0
            in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i+1])
            # Include rightmost edge for the last bin
            if i == n_bins - 1:
                in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
            prop_in_bin = np.mean(in_bin)
            
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    @staticmethod
    def calculate_adaptive_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BIN_COUNT) -> float:
        """Calculate Adaptive ECE with equal-frequency confidence bins."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)
        n_samples = len(confidences)
        if n_samples == 0:
            return 0.0

        sorted_confidences = np.sort(confidences)
        bin_boundaries = np.interp(
            np.linspace(0, n_samples, n_bins + 1),
            np.arange(n_samples),
            sorted_confidences,
        )

        ece = 0.0
        for i in range(n_bins):
            if i == n_bins - 1:
                in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            else:
                in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])

            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    @staticmethod
    def calculate_calibration_mce(probs: np.ndarray, labels: np.ndarray, n_bins: int = ECE_BIN_COUNT) -> float:
        """Calculate Maximum Calibration Error (worst non-empty bin gap)."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        max_error = 0.0
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        for i in range(n_bins):
            if i == n_bins - 1:
                in_bin = (confidences >= bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            else:
                in_bin = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])

            if np.any(in_bin):
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                max_error = max(max_error, abs(accuracy_in_bin - avg_confidence_in_bin))
        return max_error

    @staticmethod
    def calculate_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Brier Score."""
        num_classes = probs.shape[1]
        one_hot_labels = np.eye(num_classes)[labels]
        return np.mean(np.sum((probs - one_hot_labels)**2, axis=1))

    @staticmethod
    def calculate_top_label_brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
        """Calculate top-label Brier Score from max confidence and correctness."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        correct = (predictions == labels).astype(float)
        return np.mean((confidences - correct) ** 2)

    @staticmethod
    def calculate_accuracy(probs: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Accuracy."""
        predictions = np.argmax(probs, axis=1)
        return np.mean(predictions == labels)


# ===== CIFAR-C support =====
CIFAR_C_CORRUPTIONS = [
    "brightness","contrast","defocus_blur","elastic_transform","fog","frost",
    "gaussian_noise","glass_blur","impulse_noise","jpeg_compression",
    "motion_blur","pixelate","shot_noise","snow","zoom_blur"
]
# If you want strictly the original 15, slice with [:15] where used.

def _cifar_c_root(dataset_name: str, base_dir: Optional[str] = None) -> str:
    # Defaults to data/cifar10-c or data/cifar100-c unless overridden
    if base_dir is not None:
        return base_dir
    name = dataset_name.replace('-c','')
    return f"data/{'cifar10-c' if name=='cifar10' else 'cifar100-c'}"

class _CIFAR_C_Dataset(torch.utils.data.Dataset):
    """Wraps CIFAR-C .npy arrays and applies a torchvision-style transform"""
    def __init__(self, images_npy: np.ndarray, labels_npy: np.ndarray, transform=None):
        assert images_npy.ndim == 4 and images_npy.shape[-1] == 3  # [N,H,W,3], uint8
        self.x = images_npy
        self.y = labels_npy.astype(np.int64)
        self.transform = transform

    def __len__(self): return self.x.shape[0]

    def __getitem__(self, i):
        img = self.x[i]
        lbl = int(self.y[i])
        # PIL conversion keeps your existing test transforms happy
        from PIL import Image
        pil = Image.fromarray(img)
        if self.transform is not None:
            pil = self.transform(pil)
        else:
            import torchvision.transforms as T
            pil = T.ToTensor()(pil)
        return pil, lbl

def load_cifar_c_loader(dataset_name: str,
                        corruption: str,
                        severity: int,
                        test_transform,
                        batch_size: int,
                        cifar_c_dir: Optional[str] = None,
                        num_workers: int = 4) -> torch.utils.data.DataLoader:
    """
    Returns a DataLoader for a single (corruption, severity).
    CIFAR-C stores all severities in one .npy; we slice 10k chunks.
    """
    root = _cifar_c_root(dataset_name, cifar_c_dir)
    imgs = np.load(os.path.join(root, f"{corruption}.npy"))  # [50000,32,32,3], uint8
    labels = np.load(os.path.join(root, "labels.npy"))       # [50000]
    assert 1 <= severity <= 5
    start, end = (severity-1)*10000, severity*10000
    ds = _CIFAR_C_Dataset(imgs[start:end], labels[start:end], transform=test_transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      pin_memory=True, num_workers=num_workers)

def compute_mce_for_method(
    method_name: str,
    predict_fn,                 # callable(test_raw_np)->probs_np
    labels_getter,              # callable()->np.ndarray of labels (length=10000) (unused here)
    dataset_name: str,
    test_transform,
    batch_size: int,
    cifar_c_dir: Optional[str] = None
) -> float:
    """
    Loops all corruptions × severities, computes ECE per split with predict_fn, and returns mean ECE (mCE).
    predict_fn should fully map raw tensors (N,C,H,W) -> calibrated probabilities (N, K).
    """
    eces = []
    corruptions = CIFAR_C_CORRUPTIONS[:15]
    for corr in corruptions:
        for sev in range(1, 6):
            dl = load_cifar_c_loader(dataset_name, corr, sev, test_transform, batch_size, cifar_c_dir)
            Xc, Yc = get_all_data_as_numpy(dl)
            probs = predict_fn(Xc)
            eces.append(UncertaintyMetrics.calculate_ece(probs, Yc, n_bins=ECE_BIN_COUNT))
    mce = float(np.mean(eces))
    logger.info(f"[{method_name}] mCE over {len(corruptions)*5} splits = {mce:.5f}")
    return mce

def compress_with_spp_jl(
    X_np: np.ndarray,
    final_output_dim: int = 1024,
    pyramid_levels = [4, 2, 1],
    seed: int = 42,
    batch_size: int = 512,
    device: str = "cuda",
) -> np.ndarray:
    """
    Compress images [N, C, H, W] -> [N, final_output_dim] using SPP + fixed JL projection.
    """
    assert X_np.ndim == 4, f"Expected [N,C,H,W], got {X_np.shape}"
    N, C, H, W = X_np.shape

    projector = FixedSizeSPP_JL(
        in_channels=C,
        final_output_dim=final_output_dim,
        pyramid_levels=pyramid_levels,
        seed=seed,
    ).to(device).eval()

    ds = TensorDataset(torch.from_numpy(X_np))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=True)

    outs = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device).float()
            z  = projector(xb)
            outs.append(z.cpu())
    Z = torch.cat(outs, dim=0).numpy()
    return Z

def _compute_reliability_bins(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bucket_idx = np.digitize(conf, bins, right=True) - 1
    xs, ys, ws = [], [], []
    for b in range(n_bins):
        m = bucket_idx == b
        if np.any(m):
            xs.append(conf[m].mean())
            ys.append(correct[m].mean())
            ws.append(m.mean())
        else:
            xs.append((bins[b] + bins[b+1]) * 0.5)
            ys.append(np.nan)
            ws.append(0.0)
    return np.array(xs), np.array(ys), np.array(ws)

def plot_reliability_curves_baselines(curves: dict, out_path: str, n_bins: int = 15):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    xs = np.linspace(0,1,101)
    plt.plot(xs, xs, linestyle='--', linewidth=1.5, label='Perfect')
    for name, (probs, labels) in curves.items():
        x, y, _ = _compute_reliability_bins(probs, labels, n_bins=n_bins)
        plt.plot(x, y, marker='o', linewidth=2, label=name)
    plt.xlim(0,1); plt.ylim(0,1)
    plt.xlabel('Confidence'); plt.ylabel('Empirical Accuracy')
    plt.title('Reliability Curves (Baselines)')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()

class TULIPWrapper:
    """
    Adapts the efficient Hook-based TULIP implementation to the 
    numpy/index-based API expected by the experiment script.
    """
    def __init__(self, model_adapter, candidate_layers_indices: List[int], device: str, num_classes: int):
        self.model_adapter = model_adapter
        self.device = device
        self.num_classes = num_classes
        
        # 1. Convert Layer Indices (int) -> Layer Names (str) for Hooks
        self.layer_names = self._map_indices_to_names(model_adapter.model, candidate_layers_indices)
        if not self.layer_names:
            raise ValueError(f"Could not map indices {candidate_layers_indices} to named modules.")
            
        # 2. Initialize the efficient TULIP Calibrator
        from Calibrators.tulip import TULIPCalibrator 
        self.tulip = TULIPCalibrator(
            model=model_adapter.model,
            candidate_layers=self.layer_names,
            num_classes=num_classes,
            device=device
        )

    def _map_indices_to_names(self, model, indices):
        """Maps list(model.modules())[i] to named_modules() names."""
        # Create a lookup: object id -> name
        module_to_name = {id(m): name for name, m in model.named_modules()}
        
        # Get the ordered list of modules that corresponds to 'indices'
        all_modules = list(model.modules())
        
        target_names = []
        for idx in indices:
            if idx >= len(all_modules):
                continue
            mod = all_modules[idx]
            # Find name associated with this module instance
            name = module_to_name.get(id(mod))
            if name:
                target_names.append(name)
        return target_names

    def fit(self, train_raw, train_labels, val_raw, val_labels, batch_size=128):
        # Convert NumPy -> DataLoaders
        train_ds = TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels))
        val_ds = TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels))
        
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        
        # Train
        self.tulip.fit(train_loader, val_loader)

    def calibrate(self, test_raw, batch_size=128):
        """
        Returns (N, K) probabilities suitable for ECE calculation.
        Since TULIP outputs uncertainty u in [0,1], we convert this to a probability distribution.
        """
        # Convert NumPy -> DataLoader
        # Dummy labels for test set
        test_ds = TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long))
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        
        # 1. Get TULIP Uncertainty scores (N,)
        # u is in [0, 1] (0 = certain, 1 = uncertain)
        uncertainties = self.tulip.predict_uncertainty(test_loader)
        
        # 2. Get Original Model Predictions to determine *which* class is predicted
        orig_probs = self.model_adapter.predict_proba(test_raw, batch_size=batch_size)
        preds = np.argmax(orig_probs, axis=1)
        
        # 3. Construct calibrated probability matrix
        # Logic: P(predicted_class) = 1 - uncertainty
        # The remaining uncertainty is distributed evenly among other classes
        N, K = orig_probs.shape
        confidences = 1.0 - uncertainties
        confidences = np.clip(confidences, 0.0, 1.0) # Safety clip
        
        # Initialize with uniform mass based on uncertainty
        # if conf=0.8, u=0.2. Remaining 0.2 spread over K-1 classes.
        residual_prob = uncertainties[:, None] / (K - 1 + 1e-10)
        calibrated_probs = np.ones((N, K)) * residual_prob
        
        # Set the predicted class probability to the calculated confidence
        calibrated_probs[np.arange(N), preds] = confidences
        
        # Renormalize rows to sum to 1.0 (handling numerical float issues)
        calibrated_probs /= calibrated_probs.sum(axis=1, keepdims=True)
        
        return calibrated_probs

    @property
    def combination_weights(self):
        return self.tulip.combination_weights

    def fit_from_features(self, train_features_list, train_labels, val_features_list, val_labels, batch_size=128):
        """
        Fit TULIP using pre-extracted and pre-compressed features.
        
        Args:
            train_features_list: List of arrays, one per layer, shape (N, compressed_dim)
            train_labels: Training labels (N,)
            val_features_list: List of arrays, one per layer, shape (M, compressed_dim)
            val_labels: Validation labels (M,)
            batch_size: Not used, kept for API compatibility
        """
        # Convert lists to dict format expected by TULIPCalibrator
        if len(train_features_list) != len(self.layer_names):
            raise ValueError(f"Expected {len(self.layer_names)} feature arrays, got {len(train_features_list)}")
        if len(val_features_list) != len(self.layer_names):
            raise ValueError(f"Expected {len(self.layer_names)} validation feature arrays, got {len(val_features_list)}")
        
        train_features_dict = {name: feats for name, feats in zip(self.layer_names, train_features_list)}
        val_features_dict = {name: feats for name, feats in zip(self.layer_names, val_features_list)}
        
        # Convert labels to numpy if needed
        if isinstance(train_labels, torch.Tensor):
            train_labels = train_labels.numpy()
        if isinstance(val_labels, torch.Tensor):
            val_labels = val_labels.numpy()
        
        # Train using pre-extracted features
        self.tulip.fit_from_features(
            train_features_dict=train_features_dict,
            train_labels=train_labels,
            val_features_dict=val_features_dict,
            val_labels=val_labels
        )

    def calibrate_from_features(self, test_features_list, batch_size=128):
        """
        Calibrate using pre-extracted compressed features.
        
        Args:
            test_features_list: List of arrays, one per layer, shape (N, compressed_dim)
            batch_size: Not used, kept for API compatibility
        
        Returns:
            Calibrated probabilities (N, K) suitable for ECE calculation.
        """
        if not self.tulip.is_fitted:
            raise RuntimeError("Call fit_from_features() first")
        
        # Convert list to dict format
        if len(test_features_list) != len(self.layer_names):
            raise ValueError(f"Expected {len(self.layer_names)} feature arrays, got {len(test_features_list)}")
        
        test_features_dict = {name: feats for name, feats in zip(self.layer_names, test_features_list)}
        
        # Get TULIP Uncertainty scores (N,)
        uncertainties = self.tulip.predict_uncertainty_from_features(test_features_dict)
        
        # Get Original Model Predictions to determine *which* class is predicted
        # We need to run the model once to get predictions
        # For efficiency, we'll use the first test sample's features to infer the input shape
        # Actually, we can't get predictions without the raw input, so we'll need to pass test_raw
        # Or we can use the IC predictions from the deepest layer as a proxy
        # Let's use the deepest layer's predictions as the base prediction
        if not test_features_list:
            raise ValueError("test_features_list is empty")
        
        # Use the last (deepest) layer's IC predictions as the base prediction
        deepest_layer_name = self.layer_names[-1]
        deepest_ic = self.tulip.internal_classifiers[deepest_layer_name]
        deepest_ic.eval()
        
        dev = torch.device(self.device)
        deepest_feats = torch.from_numpy(test_features_dict[deepest_layer_name]).float().to(dev)
        with torch.no_grad():
            deepest_logits = deepest_ic(deepest_feats)
            preds = deepest_logits.argmax(dim=1).cpu().numpy()
        
        # Construct calibrated probability matrix
        # Logic: P(predicted_class) = 1 - uncertainty
        # The remaining uncertainty is distributed evenly among other classes
        N = len(uncertainties)
        K = self.num_classes
        confidences = 1.0 - uncertainties
        confidences = np.clip(confidences, 0.0, 1.0)  # Safety clip
        
        # Initialize with uniform mass based on uncertainty
        residual_prob = uncertainties[:, None] / (K - 1 + 1e-10)
        calibrated_probs = np.ones((N, K)) * residual_prob
        
        # Set the predicted class probability to the calculated confidence
        calibrated_probs[np.arange(N), preds] = confidences
        
        # Renormalize rows to sum to 1.0 (handling numerical float issues)
        calibrated_probs /= calibrated_probs.sum(axis=1, keepdims=True)
        
        return calibrated_probs


def run_standard_baselines(model_adapter,
                           val_raw,   val_labels,
                           test_raw,  test_labels,
                           batch_size: int,
                           output_dir: str,
                           train_raw: Optional[np.ndarray] = None,
                           train_labels: Optional[np.ndarray] = None,
                           candidate_layers_all: Optional[List[int]] = None,
                           seed: int = 42,
                           device: str = "cuda",
                           include_dac: bool = True):
    """
    Runs Uncalibrated, Temperature Scaling (logits), and Isotonic Top-Label,
    returning metrics and saving artifacts:
      - baseline_results.json
      - reliability_curves_baselines.png
    """
    import os, json, time
    os.makedirs(output_dir, exist_ok=True)
    dataset_name = getattr(model_adapter, 'dataset_name', 'cifar10')
    dataset_name = 'cifar10' if dataset_name is None else str(dataset_name).lower()

    probs_val  = model_adapter.predict_proba(val_raw,  batch_size=min(256, batch_size))
    probs_test = model_adapter.predict_proba(test_raw, batch_size=min(256, batch_size))
    # device passed as arg or fallback
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    def _metrics(probs):
        return dict(
            ece=float(UncertaintyMetrics.calculate_ece(probs, test_labels, n_bins=ECE_BIN_COUNT)),
            adaptive_ece=float(UncertaintyMetrics.calculate_adaptive_ece(probs, test_labels, n_bins=ECE_BIN_COUNT)),
            calibration_mce=float(UncertaintyMetrics.calculate_calibration_mce(probs, test_labels, n_bins=ECE_BIN_COUNT)),
            brier=float(UncertaintyMetrics.calculate_brier_score(probs, test_labels)),
            top_label_brier=float(UncertaintyMetrics.calculate_top_label_brier_score(probs, test_labels)),
            acc=float(UncertaintyMetrics.calculate_accuracy(probs, test_labels)),
        )

    results = {}
    # Holders for plotting
    probs_ts = None
    probs_iso = None
    probs_platt = None
    probs_dirichlet = None
    probs_beta = None
    probs_geo = None
    probs_geo_raw = None

    # 1) Uncalibrated (direct safe normalization)
    uncal_start = time.perf_counter()
    probs_uncal = np.clip(probs_test, 1e-8, 1-1e-8)
    probs_uncal /= probs_uncal.sum(axis=1, keepdims=True)
    uncal_elapsed = time.perf_counter() - uncal_start
    results["uncalibrated"] = _metrics(probs_uncal)
    results["uncalibrated"]["timing_seconds"] = uncal_elapsed

    # 2) Temperature Scaling (needs logits)
    logger.info("Calculating Temperature Scaling (post hoc) ...")
    ts_start = time.perf_counter()
    try:
        logits_val  = logits_val  if 'logits_val'  in locals() else model_adapter.predict_logits(val_raw,  batch_size=min(256, batch_size))
        logits_test = logits_test if 'logits_test' in locals() else model_adapter.predict_logits(test_raw, batch_size=min(256, batch_size))
        # Diagnostics and probability-like check on validation logits
        try:
            import numpy as _np
            lv = _np.asarray(logits_val)
            logger.info(f"TS logits_val shape: {lv.shape} | range: [{lv.min():.4f}, {lv.max():.4f}] | mean: {lv.mean():.4f} | std: {lv.std():.4f}")
            if lv.min() >= 0.0 and lv.max() <= 1.0:
                row_sums = lv.sum(axis=1)
                if _np.allclose(row_sums, _np.ones_like(row_sums), atol=1e-3, rtol=1e-3):
                    logger.warning("TS: logits_val look like probabilities. Converting to log-probabilities (fallback).")
                    eps = 1e-8
                    logits_val = _np.log(_np.clip(lv, eps, 1.0))
        except Exception:
            pass
        ts = TemperatureScaling()
        ts.fit(logits_val, val_labels)
        # Ensure test logits are consistent; convert if prob-like
        try:
            lt = _np.asarray(logits_test)
            if lt.min() >= 0.0 and lt.max() <= 1.0:
                row_sums = lt.sum(axis=1)
                if _np.allclose(row_sums, _np.ones_like(row_sums), atol=1e-3, rtol=1e-3):
                    logger.warning("TS: logits_test look like probabilities. Converting to log-probabilities (fallback).")
                    eps = 1e-8
                    logits_test = _np.log(_np.clip(lt, eps, 1.0))
        except Exception:
            pass
        probs_ts = ts.calibrate(logits_test)
        ts_elapsed = time.perf_counter() - ts_start
        results["temperature_scaling"] = _metrics(probs_ts)
        results["temperature_scaling"]["timing_seconds"] = ts_elapsed
        try:
            results["temperature_scaling"]["T"] = float(ts.temperature.detach().cpu().item())
        except Exception:
            pass
    except Exception as e:
        ts_elapsed = time.perf_counter() - ts_start
        logger.warning(f"Temperature scaling failed: {e}")
        results["temperature_scaling"] = {"ece": None, "brier": None, "acc": None, "error": str(e), "timing_seconds": ts_elapsed}
        probs_ts = None

    # 3) Isotonic (Top-Label)
    logger.info("Calculating Isotonic (Top-Label) Calibration (post hoc) ...")
    iso_start = time.perf_counter()
    try:
        # Reuse logits already fetched for temperature scaling if available; otherwise fetch now
        if 'logits_val' not in locals() or 'logits_test' not in locals() or logits_val is None or logits_test is None:
            logits_val  = logits_val  if 'logits_val'  in locals() else model_adapter.predict_logits(val_raw,  batch_size=min(256, batch_size))
            logits_test = logits_test if 'logits_test' in locals() else model_adapter.predict_logits(test_raw, batch_size=min(256, batch_size))
        iso = TopLabelIsotonicCalibrator()
        iso.fit(logits_val, val_labels)
        probs_iso = iso.calibrate(logits_test)
        iso_elapsed = time.perf_counter() - iso_start
        results["isotonic_toplabel"] = _metrics(probs_iso)
        results["isotonic_toplabel"]["timing_seconds"] = iso_elapsed
    except Exception as e:
        iso_elapsed = time.perf_counter() - iso_start
        logger.warning(f"Isotonic regression failed: {e}")
        results["isotonic_toplabel"] = {"ece": None, "brier": None, "acc": None, "error": str(e), "timing_seconds": iso_elapsed}
        probs_iso = None

    # 4) Platt Scaling (multiclass one-vs-rest over probabilities from logits)
    logger.info("Calculating Platt Scaling (post hoc) ...")
    platt_start = time.perf_counter()
    try:
        logits_val  = logits_val  if 'logits_val'  in locals() else model_adapter.predict_logits(val_raw,  batch_size=min(256, batch_size))
        logits_test = logits_test if 'logits_test' in locals() else model_adapter.predict_logits(test_raw, batch_size=min(256, batch_size))
        platt = PlattScaling()
        platt.fit(logits_val, val_labels)
        probs_platt = platt.calibrate(logits_test)
        platt_elapsed = time.perf_counter() - platt_start
        results["platt_scaling"] = _metrics(probs_platt)
        results["platt_scaling"]["timing_seconds"] = platt_elapsed
    except Exception as e:
        platt_elapsed = time.perf_counter() - platt_start
        logger.warning(f"Platt scaling failed: {e}")
        results["platt_scaling"] = {"ece": None, "brier": None, "acc": None, "error": str(e), "timing_seconds": platt_elapsed}
        probs_platt = None

    # # 5) Dirichlet Calibration (GPU-optimized)
    # try:
    #     logits_val  = logits_val  if 'logits_val'  in locals() else model_adapter.predict_logits(val_raw,  batch_size=min(256, batch_size))
    #     logits_test = logits_test if 'logits_test' in locals() else model_adapter.predict_logits(test_raw, batch_size=min(256, batch_size))
    #     dirichlet = DirichletCalibration()
    #     dirichlet.fit(logits_val, val_labels)
    #     probs_dirichlet = dirichlet.calibrate(logits_test)
    #     results["dirichlet_calibration"] = _metrics(probs_dirichlet)
    # except Exception as e:
    #     logger.warning(f"Dirichlet calibration failed: {e}")
    #     probs_dirichlet = None

    # 6) Beta Calibration (one-vs-rest)
    logger.info("Calculating Beta Calibration (post hoc) ...")
    beta_start = time.perf_counter()
    try:
        if 'logits_val' not in locals() or logits_val is None:
            logits_val = model_adapter.predict_logits(val_raw, batch_size=min(256, batch_size))
        if 'logits_test' not in locals() or logits_test is None:
            logits_test = model_adapter.predict_logits(test_raw, batch_size=min(256, batch_size))
        beta = BetaCalibration()
        beta.fit(logits_val, val_labels)
        probs_beta = beta.calibrate(logits_test)
        beta_elapsed = time.perf_counter() - beta_start
        results["beta_calibration"] = _metrics(probs_beta)
        results["beta_calibration"]["timing_seconds"] = beta_elapsed
    except Exception as e:
        beta_elapsed = time.perf_counter() - beta_start
        logger.warning(f"Beta calibration failed: {e}")
        results["beta_calibration"] = {"ece": None, "brier": None, "acc": None, "error": str(e), "timing_seconds": beta_elapsed}
        probs_beta = None

    # 7) Density-Aware Calibration (DAC)
    if not include_dac:
        logger.info("Skipping Density-Aware Calibration (DAC) baseline ...")
        results["density_aware_calibration"] = {
            "skipped": True,
            "reason": "DAC baseline disabled for this run",
        }
        probs_dac = None
    else:
        logger.info("Calculating Density-Aware Calibration (DAC) ...")
    dac_start = time.perf_counter()
    try:
        if not include_dac:
            raise StopIteration
        from Calibrators.density_aware_calibration import (
            DensityAwareCalibrator,
            get_dac_target_layers,
            extract_dac_features,
            get_dac_k_value
        )
        
        raw_model = model_adapter.model
        dev = torch.device(device)
        
        # Get model name from model class
        model_name = raw_model.__class__.__name__.lower()
        if 'resnet' in model_name:
            # Extract ResNet variant (e.g., 'resnet18', 'resnet50')
            model_name = model_name.replace('resnet', 'resnet')
            # Try to infer variant from model structure
            if hasattr(raw_model, 'layer1'):
                # Count blocks to determine variant
                num_blocks = [len(raw_model.layer1), len(raw_model.layer2), 
                             len(raw_model.layer3), len(raw_model.layer4)]
                if num_blocks == [2, 2, 2, 2]:
                    model_name = 'resnet18'
                elif num_blocks == [3, 4, 6, 3]:
                    model_name = 'resnet50'
                elif num_blocks == [3, 4, 23, 3]:
                    model_name = 'resnet101'  # ✅ ADD THIS
                elif num_blocks == [3, 8, 36, 3]:
                    model_name = 'resnet152'  # ✅ ADD THIS
                else:
                    model_name = 'resnet50'  # Better fallback for deep models

        elif 'densenet' in model_name:
            # Extract DenseNet variant
            if '121' in model_name or hasattr(raw_model, 'features') and len(raw_model.features) > 100:
                model_name = 'densenet121'
            else:
                model_name = 'densenet121'  # default fallback
        
        # Get dataset name from model_adapter
        dataset_name = getattr(model_adapter, 'dataset_name', 'cifar10')
        
        # Get intermediate layer names (NOT including logits)
        try:
            layer_names = get_dac_target_layers(model_name, raw_model)
            logger.info(f"   DAC target layers for {model_name}: {layer_names}")
        except Exception as e:
            dac_elapsed = time.perf_counter() - dac_start
            logger.warning(f"Could not get DAC target layers: {e}. Skipping DAC.")
            probs_dac = None
            results["density_aware_calibration"] = {
                "ece": None,
                "brier": None,
                "top_label_brier": None,
                "acc": None,
                "error": f"Layer extraction failed: {e}",
                "timing_seconds": dac_elapsed,
            }
        else:
            # Create data loaders
            tr_loader = DataLoader(
                TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                batch_size=batch_size, shuffle=False
            )
            va_loader = DataLoader(
                TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                batch_size=batch_size, shuffle=False
            )
            te_loader = DataLoader(
                TensorDataset(torch.from_numpy(test_raw), torch.from_numpy(test_labels)),
                batch_size=batch_size, shuffle=False
            )
            
            # Extract intermediate features and logits separately
            logger.info("   Extracting DAC features from intermediate layers...")
            dac_extract_start = time.perf_counter()
            train_feats_list, _, train_labels_dac = extract_dac_features(raw_model, tr_loader, layer_names, dev)
            val_feats_list, logits_val_dac, val_labels_dac = extract_dac_features(raw_model, va_loader, layer_names, dev)
            test_feats_list, logits_test_dac, _ = extract_dac_features(raw_model, te_loader, layer_names, dev)
            dac_extract_elapsed = time.perf_counter() - dac_extract_start
            
            # Convert to numpy arrays
            train_feats_list_np = [feat.numpy() for feat in train_feats_list]
            val_feats_list_np = [feat.numpy() for feat in val_feats_list]
            test_feats_list_np = [feat.numpy() for feat in test_feats_list]
            
            # Get k value based on dataset
            k_value = get_dac_k_value(dataset_name)
            logger.info(f"   Using k={k_value} for {dataset_name}")
            
            # Fit DAC
            dac_fit_start = time.perf_counter()
            dac = DensityAwareCalibrator(k=k_value, use_gpu=(device == 'cuda'))
            dac.fit(
                train_features_list=train_feats_list_np,
                val_features_list=val_feats_list_np,
                val_logits=logits_val_dac.numpy(),
                val_labels=val_labels_dac.numpy()
            )
            dac_fit_elapsed = time.perf_counter() - dac_fit_start
            
            # Calibrate test set
            dac_calibrate_start = time.perf_counter()
            probs_dac = dac.calibrate(
                test_features_list=test_feats_list_np,
                test_logits=logits_test_dac.numpy()
            )
            dac_calibrate_elapsed = time.perf_counter() - dac_calibrate_start
            
            dac_elapsed = time.perf_counter() - dac_start
            results["density_aware_calibration"] = _metrics(probs_dac)
            results["density_aware_calibration"]["timing_seconds"] = dac_elapsed
            results["density_aware_calibration"]["extraction_time_s"] = dac_extract_elapsed
            results["density_aware_calibration"]["fit_time_s"] = dac_fit_elapsed
            results["density_aware_calibration"]["calibrate_time_s"] = dac_calibrate_elapsed
            # Store DAC's selected layers for use in ablation experiments
            results["density_aware_calibration"]["selected_layers"] = layer_names
            logger.info(f"DAC calibration completed - extract: {dac_extract_elapsed:.2f}s, fit: {dac_fit_elapsed:.2f}s, calibrate: {dac_calibrate_elapsed:.2f}s, total: {dac_elapsed:.2f}s")

    except Exception as e:
        if not include_dac and isinstance(e, StopIteration):
            pass
        else:
            dac_elapsed = time.perf_counter() - dac_start
            logger.warning(f"Density-Aware Calibration failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            results["density_aware_calibration"] = {"ece": None, "brier": None, "top_label_brier": None, "acc": None, "error": str(e), "timing_seconds": dac_elapsed}
            probs_dac = None

    # # 8) Probabilistic Skip Connections (PSC)
    # logger.info("🧪 Calculating Probabilistic Skip Connections (PSC) ...")
    # try:
    #     from Calibrators.probabilistic_skip_connections import ProbabilisticSkipConnection
        
    #     # PSC uses candidate layers already computed in full metric scan
    #     if 'candidate_layers_all' not in locals() or not candidate_layers_all:
    #         logger.warning("candidate_layers_all not available, using default layer selection")
    #         candidate_layers_psc = list(range(10, len(list(model_adapter.model.modules())), 5))
    #     else:
    #         candidate_layers_psc = candidate_layers_all
        
    #     logger.info(f"   PSC will analyze {len(candidate_layers_psc)} candidate layers using NC metrics")
        
    #     # Create and fit PSC
    #     psc = ProbabilisticSkipConnection(
    #         model_adapter=model_adapter,
    #         candidate_layers=candidate_layers_psc,
    #         device=device,
    #         epsilon=0.2  # NC1 threshold from PSC paper
    #     )
        
    #     # Fit PSC (automatically selects best layer + fits calibrator)
    #     psc.fit(
    #         train_raw=train_raw,
    #         train_labels=train_labels,
    #         val_raw=val_raw,
    #         val_labels=val_labels,
    #         batch_size=batch_size
    #     )
        
    #     # Calibrate test set
    #     probs_psc = psc.calibrate(test_raw=test_raw, batch_size=batch_size)
        
    #     # Get feature density for OOD score
    #     ood_scores_psc = psc.get_feature_density(test_raw=test_raw, batch_size=batch_size)
        
    #     results["probabilistic_skip_connections"] = dict(
    #         ece=float(UncertaintyMetrics.calculate_ece(probs_psc, test_labels, n_bins=ECE_BIN_COUNT)),
    #         brier=float(UncertaintyMetrics.calculate_brier_score(probs_psc, test_labels)),
    #         acc=float(UncertaintyMetrics.calculate_accuracy(probs_psc, test_labels)),
    #         selected_layer=int(psc.selected_layer),
    #         mean_ood_score=float(ood_scores_psc.mean())
    #     )
        
    #     logger.info(f"✅ PSC completed (selected layer {psc.selected_layer})")
        
    # except Exception as e:
    #     logger.warning(f"Probabilistic Skip Connections failed: {e}")
    #     import traceback
    #     logger.warning(traceback.format_exc())
    #     results["probabilistic_skip_connections"] = {
    #         "ece": None, "brier": None, "acc": None, "error": str(e)
    #     }
    #     probs_psc = None

    # # 9) TULIP - Transitional Uncertainty with Layered Intermediate Predictions
    # logger.info("🌷 Calculating TULIP ...")
    # try:
    #     # NOTE: We use the local TULIPWrapper to bridge the new implementation 
    #     # with this script's expected API.
        
    #     # TULIP uses candidate layers from full metric scan
    #     if 'candidate_layers_all' not in locals() or not candidate_layers_all:
    #         logger.warning("candidate_layers_all not available, using default layer selection")
    #         # Fallback logic to pick indices roughly evenly spaced
    #         total_mods = len(list(model_adapter.model.modules()))
    #         candidate_layers_tulip = list(range(10, total_mods, 5))
    #     else:
    #         # Paper uses uniform spacing. We subsample the available candidates to ~5.
    #         total_layers = len(candidate_layers_all)
    #         if total_layers <= 5:
    #             candidate_layers_tulip = candidate_layers_all
    #         else:
    #             spacing = total_layers // 5
    #             candidate_layers_tulip = candidate_layers_all[::spacing]
    #             if len(candidate_layers_tulip) > 10:
    #                  candidate_layers_tulip = candidate_layers_tulip[::2]
        
    #     logger.info(f"   TULIP will use {len(candidate_layers_tulip)} internal classifiers")
    #     logger.info(f"   at layer INDICES: {candidate_layers_tulip}")
        
    #     # --- MODIFIED: Use Wrapper ---
    #     tulip = TULIPWrapper(
    #         model_adapter=model_adapter,
    #         candidate_layers_indices=candidate_layers_tulip,
    #         device=device,
    #         num_classes=len(np.unique(train_labels))
    #     )
        
    #     # Fit TULIP (Wrapper converts numpy -> loaders internally)
    #     tulip.fit(
    #         train_raw=train_raw,
    #         train_labels=train_labels,
    #         val_raw=val_raw,
    #         val_labels=val_labels,
    #         batch_size=batch_size
    #     )
        
    #     # Calibrate test set (Wrapper returns (N,K) matrix based on uncertainty)
    #     probs_tulip = tulip.calibrate(test_raw, batch_size=batch_size)
        
    #     results["tulip"] = dict(
    #         ece=float(UncertaintyMetrics.calculate_ece(probs_tulip, test_labels, n_bins=ECE_BIN_COUNT)),
    #         brier=float(UncertaintyMetrics.calculate_brier_score(probs_tulip, test_labels)),
    #         acc=float(UncertaintyMetrics.calculate_accuracy(probs_tulip, test_labels)),
    #         num_internal_classifiers=len(candidate_layers_tulip),
    #         combination_weights=tulip.combination_weights.tolist() if tulip.combination_weights is not None else []
    #     )
        
    #     logger.info(f"✅ TULIP completed ({len(candidate_layers_tulip)} ICs)")
        
    # except Exception as e:
    #     logger.warning(f"TULIP failed: {e}")
    #     import traceback
    #     logger.warning(traceback.format_exc())
    #     results["tulip"] = {
    #         "ece": None, "brier": None, "acc": None, "error": str(e)
    #     }
    #     probs_tulip = None

    # 10) Geometric Calibrator (Physical Space)
    logger.info("Geometric Calibrator (Physical Space with SPP+JL compression) ...")
    geo_total_start = time.perf_counter()
    try:
        FINAL_DIM = 512  # Standard dimension (Tiny ImageNet is subsampled to 45k, so same as CIFAR)
        PYR = [4, 2, 1]

        # --- INIT TIMING (compression) ---
        geo_init_start = time.perf_counter()
        Xtr_c = compress_with_spp_jl(
            train_raw, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
            seed=seed, batch_size=min(512, batch_size), device=device
        )
        Xva_c = compress_with_spp_jl(
            val_raw,   final_output_dim=FINAL_DIM, pyramid_levels=PYR,
            seed=seed, batch_size=min(512, batch_size), device=device
        )
        Xte_c = compress_with_spp_jl(
            test_raw,  final_output_dim=FINAL_DIM, pyramid_levels=PYR,
            seed=seed, batch_size=min(512, batch_size), device=device
        )
        logger.info(f"Compressed training data shape: {Xtr_c.shape}")
        logger.info(f"Compressed validation data shape: {Xva_c.shape}")
        logger.info(f"Compressed test data shape: {Xte_c.shape}")
        
        geo = GeometricCalibrator(
            model=model_adapter,
            X_train_embed=Xtr_c,
            y_train=train_labels,
            auto_select_layer=False,
            library="fast_separation",
            device=device,
        )
        geo_init_elapsed = time.perf_counter() - geo_init_start

        # --- FIT TIMING ---
        geo_fit_start = time.perf_counter()
        logger.info(f"Fitting GeometricCalibrator with batch size {batch_size}")
        geo.fit(X_val_embed=Xva_c, X_val_original=val_raw, y_val=val_labels, fit_batch_size=batch_size)
        logger.info(f"Fitted GeometricCalibrator with batch size {batch_size}")
        geo_fit_elapsed = time.perf_counter() - geo_fit_start

        # --- CALIBRATE TIMING (online inference) ---
        geo_calibrate_start = time.perf_counter()
        probs_geo = geo.calibrate_batched(X_test_embed=Xte_c, X_test_original=test_raw, batch_size=batch_size)
        geo_calibrate_elapsed = time.perf_counter() - geo_calibrate_start
        
        logger.info(f"Calibrated test set with batch size {batch_size}")
        logger.info(f"Calibrated test set shape: {probs_geo.shape}")
        
        geo_total_elapsed = time.perf_counter() - geo_total_start
        
        # Calculate extraction time (init + fit) and calibration time (online)
        geo_extraction_time = geo_init_elapsed + geo_fit_elapsed
        
        results["geometric_physical_space"] = _metrics(probs_geo)
        results["geometric_physical_space"].update(
            timing_init_seconds=geo_init_elapsed,
            timing_fit_seconds=geo_fit_elapsed,
            timing_calibrate_seconds=geo_calibrate_elapsed,
            timing_total_seconds=geo_total_elapsed,
            extraction_time_s=geo_extraction_time,
            calibrate_time_s=geo_calibrate_elapsed,
        )
        logger.info(f"Geometric calibrator completed - init: {geo_init_elapsed:.2f}s, fit: {geo_fit_elapsed:.2f}s, calibrate: {geo_calibrate_elapsed:.2f}s, total: {geo_total_elapsed:.2f}s")

        # DIAGNOSTIC: Compare geometric vs isotonic outputs (ENHANCED)
        if 'probs_iso' in locals() and probs_iso is not None:
            logger.info("="*60)
            logger.info("COMPARING GEOMETRIC VS ISOTONIC OUTPUTS")
            logger.info("="*60)

            # Check if outputs are identical
            max_diff = np.max(np.abs(probs_geo - probs_iso))
            are_identical = np.allclose(probs_geo, probs_iso, atol=1e-8)

            logger.info(f"Max absolute difference: {max_diff:.10f}")
            logger.info(f"Are they identical? {are_identical}")

            # Compare first 3 samples
            for i in range(min(3, len(probs_geo))):
                logger.info(f"\nSample {i}:")
                logger.info(f"  Geometric: {probs_geo[i]}")
                logger.info(f"  Isotonic:  {probs_iso[i]}")
                logger.info(f"  Difference: {probs_geo[i] - probs_iso[i]}")

            # Check if they're using same predicted classes
            geo_preds = np.argmax(probs_geo, axis=1)
            iso_preds = np.argmax(probs_iso, axis=1)
            same_predictions = np.mean(geo_preds == iso_preds)
            logger.info(f"\nFraction with same predicted class: {same_predictions:.4f}")

            logger.info("="*60)

            # Check if all samples are uniform distributions
            geometric_entropy = -np.sum(probs_geo * np.log(probs_geo + 1e-10), axis=1).mean()
            isotonic_entropy = -np.sum(probs_iso * np.log(probs_iso + 1e-10), axis=1).mean()
            logger.info(f"   Geometric avg entropy: {geometric_entropy:.6f} (max={np.log(probs_geo.shape[1]):.6f})")
            logger.info(f"   Isotonic avg entropy: {isotonic_entropy:.6f} (max={np.log(probs_iso.shape[1]):.6f})")
        else:
            logger.warning("Cannot compare: probs_iso not available")
    except Exception as e:
        geo_total_elapsed = time.perf_counter() - geo_total_start
        logger.warning(f"Geometric calibrator (SPP+JL) failed: {e}")
        results["geometric_physical_space"] = {
            "ece": None,
            "adaptive_ece": None,
            "calibration_mce": None,
            "brier": None,
            "top_label_brier": None,
            "acc": None,
            "error": str(e),
            "timing_init_seconds": None,
            "timing_fit_seconds": None,
            "timing_calibrate_seconds": None,
            "timing_total_seconds": geo_total_elapsed,
        }
        probs_geo = None

    # 11) Geometric Calibrator (Raw Images)
    logger.info("Geometric Calibrator (Raw Images without compression) ...")
    if dataset_name in ['tiny_imagenet', 'imagenet', 'cifar100']:
        logger.warning("Geometric Calibrator (Raw Images without compression) is not supported for tiny_imagenet and imagenet")
        results["geometric_raw_images"] = {
            "ece": None,
            "adaptive_ece": None,
            "calibration_mce": None,
            "brier": None,
            "top_label_brier": None,
            "acc": None,
            "error": f"raw_geometric_not_supported_for_{dataset_name}",
            "timing_init_seconds": None,
            "timing_fit_seconds": None,
            "timing_calibrate_seconds": None,
            "timing_total_seconds": None,
        }
    else:
        geo_raw_total_start = time.perf_counter()
        try:
            # --- INIT TIMING (flattening) ---
            geo_raw_init_start = time.perf_counter()
            # Flatten images to 2D: [N, -1]
            Xtr_raw = train_raw.reshape(train_raw.shape[0], -1)
            Xva_raw = val_raw.reshape(val_raw.shape[0], -1)
            Xte_raw = test_raw.reshape(test_raw.shape[0], -1)
            logger.info(f"Flattened training data shape: {Xtr_raw.shape}")
            logger.info(f"Flattened validation data shape: {Xva_raw.shape}")
            logger.info(f"Flattened test data shape: {Xte_raw.shape}")
            
            geo_raw = GeometricCalibrator(
                model=model_adapter,
                X_train_embed=Xtr_raw,
                y_train=train_labels,
                auto_select_layer=False,
                library="fast_separation",
                device=device,
            )
            geo_raw_init_elapsed = time.perf_counter() - geo_raw_init_start

            # --- FIT TIMING ---
            geo_raw_fit_start = time.perf_counter()
            logger.info(f"Fitting GeometricCalibrator (raw images) with batch size {batch_size}")
            geo_raw.fit(X_val_embed=Xva_raw, X_val_original=val_raw, y_val=val_labels, fit_batch_size=batch_size)
            logger.info(f"Fitted GeometricCalibrator (raw images) with batch size {batch_size}")
            geo_raw_fit_elapsed = time.perf_counter() - geo_raw_fit_start

            # --- CALIBRATE TIMING (online inference) ---
            geo_raw_calibrate_start = time.perf_counter()
            probs_geo_raw = geo_raw.calibrate_batched(X_test_embed=Xte_raw, X_test_original=test_raw, batch_size=batch_size)
            geo_raw_calibrate_elapsed = time.perf_counter() - geo_raw_calibrate_start
            
            logger.info(f"Calibrated test set (raw images) with batch size {batch_size}")
            logger.info(f"Calibrated test set shape: {probs_geo_raw.shape}")
            
            geo_raw_total_elapsed = time.perf_counter() - geo_raw_total_start
            
            # Calculate extraction time (init + fit) and calibration time (online)
            geo_raw_extraction_time = geo_raw_init_elapsed + geo_raw_fit_elapsed
            
            results["geometric_raw_images"] = _metrics(probs_geo_raw)
            results["geometric_raw_images"].update(
                timing_init_seconds=geo_raw_init_elapsed,
                timing_fit_seconds=geo_raw_fit_elapsed,
                timing_calibrate_seconds=geo_raw_calibrate_elapsed,
                timing_total_seconds=geo_raw_total_elapsed,
                extraction_time_s=geo_raw_extraction_time,
                calibrate_time_s=geo_raw_calibrate_elapsed,
            )
            logger.info(f"Geometric calibrator (raw images) completed - init: {geo_raw_init_elapsed:.2f}s, fit: {geo_raw_fit_elapsed:.2f}s, calibrate: {geo_raw_calibrate_elapsed:.2f}s, total: {geo_raw_total_elapsed:.2f}s")
        except Exception as e:
            geo_raw_total_elapsed = time.perf_counter() - geo_raw_total_start
            logger.warning(f"Geometric calibrator (raw images) failed: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            results["geometric_raw_images"] = {
                "ece": None,
                "adaptive_ece": None,
                "calibration_mce": None,
                "brier": None,
                "top_label_brier": None,
                "acc": None,
                "error": str(e),
                "timing_init_seconds": None,
                "timing_fit_seconds": None,
                "timing_calibrate_seconds": None,
                "timing_total_seconds": geo_raw_total_elapsed,
            }
            probs_geo_raw = None

    # Save JSON
    # out_json = os.path.join(output_dir, "baseline_results.json")
    # with open(out_json, "w") as f:
    #     json.dump(format_numbers_7digits(results), f, indent=2, cls=NumpyEncoder)
    # logger.info(f"💾 Baseline results saved: {out_json}")

    # # Reliability curves figure
    # curves = {"uncalibrated": (probs_uncal, test_labels)}
    # if 'probs_ts' in locals() and probs_ts is not None:
    #     curves["temperature_scaling"] = (probs_ts, test_labels)
    # if 'probs_iso' in locals() and probs_iso is not None:
    #     curves["isotonic_toplabel"]   = (probs_iso, test_labels)
    # if 'probs_platt' in locals() and probs_platt is not None:
    #     curves["platt_scaling"] = (probs_platt, test_labels)
    # if 'probs_dirichlet' in locals() and probs_dirichlet is not None:
    #     curves["dirichlet_calibration"] = (probs_dirichlet, test_labels)
    # if 'probs_beta' in locals() and probs_beta is not None:
    #     curves["beta_calibration"] = (probs_beta, test_labels)
    # if 'probs_dac' in locals() and probs_dac is not None:
    #     curves["density_aware_calibration"] = (probs_dac, test_labels)
    # if 'probs_psc' in locals() and probs_psc is not None:
    #     curves["probabilistic_skip_connections"] = (probs_psc, test_labels)
    # if 'probs_tulip' in locals() and probs_tulip is not None:
    #     curves["tulip"] = (probs_tulip, test_labels)
    # if 'probs_geo' in locals() and probs_geo is not None:
    #     curves["geometric_physical_space"] = (probs_geo, test_labels)
    # rc_path = os.path.join(output_dir, "reliability_curves_baselines.png")
    # plot_reliability_curves_baselines(curves, rc_path, n_bins=ECE_BIN_COUNT)
    # logger.info(f"📈 Reliability curves saved: {rc_path}")

    return results


def _topk_layers_by_min_gap(output_dir: str, k: int = 3,
                            results_json_filename: str = "multi_composite_analysis.json") -> List[int]:
    """
    Pick top-k DISTINCT layers with smallest 'ece_gap_from_optimal' from per_metric_selection.json.
    Fallback to approach predictions (minimal/ece_dominant/pure_ece/learned) if the JSON is missing.
    """
    per_metric_path = os.path.join(output_dir, "per_metric_selection.json")
    layers: List[int] = []
    if os.path.exists(per_metric_path):
        with open(per_metric_path, "r") as f:
            D = json.load(f)
        # leaderboard: list of [metric_key, { chosen_layer, ece_gap_from_optimal, ... }]
        # collect best gap per layer
        best_gap_per_layer: Dict[int, float] = {}
        for key, entry in D.get("leaderboard", []):
            L = entry.get("chosen_layer")
            gap = entry.get("ece_gap_from_optimal")
            if L is None or gap is None or not np.isfinite(gap):
                continue
            L = int(L)
            best_gap_per_layer[L] = min(gap, best_gap_per_layer.get(L, float("inf")))
        # sort by gap and take top-k distinct
        layers = [L for L, _ in sorted(best_gap_per_layer.items(), key=lambda kv: kv[1])][:k]

    if not layers:  # fallback — use selector predictions from the main results json
        results_json_path = os.path.join(output_dir, results_json_filename)
        try:
            with open(results_json_path, "r") as f:
                R = json.load(f)
            candidates = [
                R.get("selector_prediction_minimal"),
                R.get("selector_prediction_ece_dominant"),
                R.get("selector_prediction_pure_ece"),
                R.get("selector_prediction_learned"),
            ]
            layers = [int(x) for x in candidates if x is not None]
            # keep order & distinct, then cut to k
            seen, uniq = set(), []
            for L in layers:
                if L not in seen:
                    uniq.append(L); seen.add(L)
            layers = uniq[:k]
        except Exception:
            layers = []
    return layers


def _calibrated_probs_for_layer(model_adapter,
                                layer_idx: int,
                                train_raw, train_labels,
                                val_raw,   val_labels,
                                test_raw,
                                batch_size: int,
                                device: str) -> np.ndarray:
    """
    Reproduce the empirical pipeline for a SINGLE layer and return calibrated probs on TEST.
    """
    raw_model = model_adapter.model
    dev = torch.device(device)

    train_loader = DataLoader(TensorDataset(torch.from_numpy(train_raw),
                                            torch.from_numpy(train_labels)),
                              batch_size=batch_size, shuffle=False)
    val_loader   = DataLoader(TensorDataset(torch.from_numpy(val_raw),
                                            torch.from_numpy(val_labels)),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(TensorDataset(torch.from_numpy(test_raw),
                                            torch.zeros(len(test_raw), dtype=torch.long)),
                              batch_size=batch_size, shuffle=False)

    # features
    Xtr, _ = extract_features_directly(raw_model, train_loader, layer_idx, dev)
    Xva, _ = extract_features_directly(raw_model, val_loader,   layer_idx, dev)
    Xte, _ = extract_features_directly(raw_model, test_loader,  layer_idx, dev)

    # fit calibrator for this fixed layer
    cal = GeometricCalibrator(
        model=model_adapter,
        X_train_embed=Xtr.numpy(),
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=False,
    )
    cal.fit(X_val_embed=Xva.numpy(), y_val=val_labels, X_val_original=val_raw, fit_batch_size=batch_size)
    probs = cal.calibrate_batched(X_test_embed=Xte.numpy(), X_test_original=test_raw, batch_size=batch_size)
    return probs


def plot_reliability_curves_with_topk_layers(model_adapter,
                                             train_raw, train_labels,
                                             val_raw,   val_labels,
                                             test_raw,  test_labels,
                                             batch_size: int,
                                             device: str,
                                             output_dir: str,
                                             k: int = 3,
                                             out_name: str = "reliability_curves_with_topk.png"):
    """
    Regenerates the baseline curves and overlays the reliability curves of the top-k layers.
    """
    # --- baselines (same as run_standard_baselines) ---
    probs_val  = model_adapter.predict_proba(val_raw,  batch_size=min(256, batch_size))
    probs_test = model_adapter.predict_proba(test_raw, batch_size=min(256, batch_size))

    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    # uncalibrated
    probs_uncal = np.clip(probs_test, 1e-8, 1-1e-8); probs_uncal /= probs_uncal.sum(axis=1, keepdims=True)
    curves["uncalibrated"] = (probs_uncal, test_labels)

    # temperature scaling
    try:
        logits_val  = model_adapter.predict_logits(val_raw,  batch_size=min(256, batch_size))
        logits_test = model_adapter.predict_logits(test_raw, batch_size=min(256, batch_size))
        ts = TemperatureScaling(); ts.fit(logits_val, val_labels)
        probs_ts = ts.calibrate(logits_test)
        curves["temperature_scaling"] = (probs_ts, test_labels)
    except Exception:
        pass

    # isotonic top-label
    try:
        # Use real logits for the calibrator (it does softmax internally)
        logits_val  = model_adapter.predict_logits(val_raw,  batch_size=min(256, batch_size))
        logits_test = model_adapter.predict_logits(test_raw, batch_size=min(256, batch_size))
        iso = TopLabelIsotonicCalibrator(); iso.fit(logits_val, val_labels)
        probs_iso = iso.calibrate(logits_test)
        curves["isotonic_toplabel"] = (probs_iso, test_labels)
    except Exception:
        pass

    # --- top-k layers (by min gap) ---
    top_layers = _topk_layers_by_min_gap(output_dir, k=k)
    gaps: Dict[int, float] = {}
    # Try to read gaps so we can print them next to layer id
    pm_path = os.path.join(output_dir, "per_metric_selection.json")
    if os.path.exists(pm_path):
        with open(pm_path, "r") as f:
            D = json.load(f)
        best_gap_per_layer: Dict[int, float] = {}
        for _, entry in D.get("leaderboard", []):
            L = entry.get("chosen_layer")
            g = entry.get("ece_gap_from_optimal")
            if L is None or g is None or not np.isfinite(g):
                continue
            L = int(L)
            best_gap_per_layer[L] = min(g, best_gap_per_layer.get(L, float("inf")))
        gaps.update(best_gap_per_layer)

    for L in top_layers:
        try:
            probs_L = _calibrated_probs_for_layer(
                model_adapter, L,
                train_raw, train_labels,
                val_raw,   val_labels,
                test_raw,
                batch_size, device
            )
            gap_str = f" (gap {gaps[L]:.4f})" if L in gaps and np.isfinite(gaps[L]) else ""
            curves[f"layer_{L}{gap_str}"] = (probs_L, test_labels)
        except Exception as e:
            logger.warning(f"Top-k layer {L} failed for reliability plot: {e}")

    # --- draw ---
    out_path = os.path.join(output_dir, out_name)
    plot_reliability_curves_baselines(curves, out_path, n_bins=ECE_BIN_COUNT)
    logger.info(f"📈 Reliability curves with top-{k} layers saved: {out_path}")

def load_linear_weights_csv(path: str):
    """Return (coef_dict, intercept) from a weights CSV with a 'feature' and 'coef' column."""
    W = pd.read_csv(path)
    if 'feature' not in W.columns or 'coef' not in W.columns:
        raise ValueError(f"{path} must have columns ['feature','coef', ...]")
    coef = dict(zip(W['feature'].astype(str), W['coef'].astype(float)))
    intercept = float(W['intercept'].iloc[0]) if 'intercept' in W.columns else 0.0
    return coef, intercept

def score_layers_with_linear_model(layer_metrics: List[LayerMetrics], coef: Dict[str, float], intercept: float = 0.0):
    """
    Compute predicted gap per layer from learned linear weights.
    Returns:
      scores       : dict[layer_idx] = (-pred_gap)   # higher is better
      predicted_gap: dict[layer_idx] = pred_gap      # lower is better
    """
    scores: Dict[int, float] = {}
    predicted_gap: Dict[int, float] = {}
    for lm in layer_metrics:
        s = intercept
        for name, w in coef.items():
            v = getattr(lm, name, np.nan)
            if np.isfinite(v):
                s += w * float(v)
        gap = float(s)
        predicted_gap[int(lm.layer_idx)] = gap
        scores[int(lm.layer_idx)] = -gap
    return scores, predicted_gap

def pick_best_by_scores(candidate_layers: List[int], scores_dict: Dict[int, float]) -> Tuple[int, float]:
    arr = np.array([scores_dict.get(int(L), -np.inf) for L in candidate_layers], dtype=float)
    best_idx = int(np.nanargmax(arr))
    return int(candidate_layers[best_idx]), float(arr[best_idx])

class MultiCompositeScoreAnalyzer:
    """Analyze and visualize results comparing multiple composite score approaches"""
    
    def __init__(self, results: Dict[str, Any]):
        self.results = results
        self.layers_data = results.get('results_per_layer', [])
        
    def analyze_approach_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance of each composite score approach"""
        approaches = ['minimal', 'ece_dominant', 'pure_ece', 'learned']
        performance_analysis = {}
        
        # Get empirical best layer for reference (prefer new keys)
        empirical_best = self.results.get('empirical_best_layer')
        if empirical_best is None:
            empirical_best = self.results.get('empirical_best_layer_test_ece')
        if empirical_best is None:
            return {}
        
        # Find empirical best layer data
        empirical_best_data = next((layer for layer in self.layers_data 
                                  if layer['layer_idx'] == empirical_best), None)
        if not empirical_best_data:
            return {}
        
        empirical_best_ece = empirical_best_data.get('empirical_test_ece', float('inf'))
        
        for approach in approaches:
            predicted_layer = self.results.get(f'selector_prediction_{approach}')
            if predicted_layer is None:
                continue
                
            # Find predicted layer data
            predicted_data = next((layer for layer in self.layers_data 
                                 if layer['layer_idx'] == predicted_layer), None)
            if not predicted_data:
                continue
            
            predicted_ece = predicted_data.get('empirical_test_ece', float('inf'))
            predicted_accuracy = predicted_data.get('empirical_test_accuracy', 0.0)
            composite_score = predicted_data.get(f'selector_composite_score_{approach}', 0.0)
            
            # Calculate performance metrics
            ece_gap = predicted_ece - empirical_best_ece
            correct_prediction = (predicted_layer == empirical_best)
            
            performance_analysis[approach] = {
                'predicted_layer': predicted_layer,
                'predicted_ece': predicted_ece,
                'predicted_accuracy': predicted_accuracy,
                'composite_score': composite_score,
                'ece_gap_from_optimal': ece_gap,
                'correct_prediction': correct_prediction,
                'relative_performance': 1.0 - (ece_gap / empirical_best_ece) if empirical_best_ece > 0 else 0.0
            }
        
        return performance_analysis
    
    def create_multi_approach_comparison_plot(self, output_path: str):
        """Create comprehensive comparison plot for all approaches"""
        if not self.layers_data:
            logger.warning("No layer data available for plotting")
            return
        
        # Extract data
        layers = [layer['layer_idx'] for layer in self.layers_data]
        test_eces = [layer.get('empirical_test_ece', float('inf')) for layer in self.layers_data]
        
        # Extract composite scores for each approach
        minimal_scores = [layer.get('selector_composite_score_minimal', 0) for layer in self.layers_data]
        ece_dominant_scores = [layer.get('selector_composite_score_ece_dominant', 0) for layer in self.layers_data]
        pure_ece_scores = [layer.get('selector_composite_score_pure_ece', 0) for layer in self.layers_data]
        
        # Create comprehensive plot
        fig = plt.figure(figsize=(20, 20))
        
        # Main comparison plot (top row, spans all 3 columns)
        ax1 = plt.subplot(4, 3, (1, 3))  # Top row, spans all 3 columns
        
        # Plot composite scores
        width = 0.2
        x = np.arange(len(layers))
        
        learned_scores_plot = [layer.get('selector_composite_score_learned', 0) for layer in self.layers_data]
        if any(minimal_scores) or any(ece_dominant_scores) or any(pure_ece_scores) or any(learned_scores_plot):
            bars1 = ax1.bar(x - 1.5*width, minimal_scores, width, label='Minimal (3 metrics)', alpha=0.8, color='blue')
            bars2 = ax1.bar(x - 0.5*width, ece_dominant_scores, width, label='ECE-Dominant (2 metrics)', alpha=0.8, color='green')
            bars3 = ax1.bar(x + 0.5*width, pure_ece_scores, width, label='Pure ECE (1 metric)', alpha=0.8, color='orange')
            bars4 = ax1.bar(x + 1.5*width, learned_scores_plot, width, label='Learned (linear CSV)', alpha=0.8, color='purple')
        else:
            bars1 = bars2 = bars3 = bars4 = []
            ax1.set_title('Test ECE across layers', fontsize=14, fontweight='bold')
        
        ax1.set_xlabel('Layer Index', fontsize=12)
        ax1.set_ylabel('Composite Score (Higher is Better)', fontsize=12)
        ax1.set_title('Multi-Approach Composite Score Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layers)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # ECE line plot overlay
        ax2 = ax1.twinx()
        ax2.plot(x, test_eces, color='red', marker='o', linestyle='-', 
                linewidth=2, markersize=6, label="Test ECE (Ground Truth)")
        ax2.set_ylabel('Empirical Test ECE (Lower is Better)', color='red', fontsize=12)
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Add vertical lines for predictions
        approaches = ['minimal', 'ece_dominant', 'pure_ece', 'learned']
        colors = ['blue', 'green', 'orange', 'purple']
        
        for approach, color in zip(approaches, colors):
            predicted_layer = self.results.get(f'selector_prediction_{approach}')
            if predicted_layer is not None and predicted_layer in layers:
                layer_pos = layers.index(predicted_layer)
                ax1.axvline(x=layer_pos, color=color, linestyle='--', linewidth=2, alpha=0.7)
        
        # Best empirical layer (prefer new keys)
        empirical_best = self.results.get('empirical_best_layer')
        if empirical_best is None:
            empirical_best = self.results.get('empirical_best_layer_test_ece')
        if empirical_best is not None and empirical_best in layers:
            layer_pos = layers.index(empirical_best)
            ax2.axvline(x=layer_pos, color='red', linestyle='--', linewidth=3, alpha=0.8,
                       label=f'Optimal Layer {empirical_best}')
        
        # Performance comparison (middle row)
        performance = self.analyze_approach_performance()
        
        if performance:
            # ECE Performance comparison
            ax3 = plt.subplot(4, 3, 4)
            approach_names = list(performance.keys())
            approach_eces = [performance[app]['predicted_ece'] for app in approach_names]
            approach_colors = ['blue', 'green', 'orange'][:len(approach_names)]
            
            bars = ax3.bar(approach_names, approach_eces, color=approach_colors, alpha=0.7)
            
            # Add optimal ECE line
            if empirical_best:
                empirical_best_data = next((layer for layer in self.layers_data 
                                          if layer['layer_idx'] == empirical_best), None)
                if empirical_best_data:
                    optimal_ece = empirical_best_data.get('empirical_test_ece', 0)
                    ax3.axhline(y=optimal_ece, color='red', linestyle='--', linewidth=2, 
                               label=f'Optimal ECE: {optimal_ece:.4f}')
            
            ax3.set_title('ECE Performance by Approach')
            ax3.set_ylabel('Test ECE')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Add ECE values on bars
            for bar, ece in zip(bars, approach_eces):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{ece:.4f}', ha='center', va='bottom', fontsize=10)
            
            # Accuracy comparison
            ax4 = plt.subplot(4, 3, 5)
            approach_accs = [performance[app]['predicted_accuracy'] for app in approach_names]
            
            bars = ax4.bar(approach_names, approach_accs, color=approach_colors, alpha=0.7)
            ax4.set_title('Accuracy by Approach')
            ax4.set_ylabel('Test Accuracy')
            ax4.grid(True, alpha=0.3)
            
            # Add accuracy values on bars
            for bar, acc in zip(bars, approach_accs):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
            
            # ECE Gap from Optimal
            ax5 = plt.subplot(4, 3, 6)
            approach_gaps = [performance[app]['ece_gap_from_optimal'] for app in approach_names]
            gap_colors = ['green' if gap <= 0 else 'red' for gap in approach_gaps]
            
            bars = ax5.bar(approach_names, approach_gaps, color=gap_colors, alpha=0.7)
            ax5.set_title('ECE Gap from Optimal (Lower is Better)')
            ax5.set_ylabel('ECE Gap')
            ax5.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax5.grid(True, alpha=0.3)
            
            # Add gap values on bars
            for bar, gap in zip(bars, approach_gaps):
                height = bar.get_height()
                y_pos = height + (height*0.1 if height >= 0 else height*0.1)
                ax5.text(bar.get_x() + bar.get_width()/2., y_pos,
                        f'{gap:+.4f}', ha='center', va='bottom' if height >= 0 else 'top', fontsize=10)
        
        # Prediction success summary (bottom row)
        # --- New: Baselines (row 3, col 1) ---
        ax_bas = plt.subplot(4, 3, 7)
        baseline = self.results.get("baselines", {})
        if baseline:
            names = []
            eces  = []
            for k in ["uncalibrated", "temperature_scaling", "isotonic_toplabel"]:
                if k in baseline and baseline[k].get("ece") is not None:
                    names.append(k.replace("_"," "))
                    eces.append(baseline[k]["ece"])
            if names:
                ax_bas.bar(names, eces, alpha=0.8)
                empirical_best = self.results.get('empirical_best_layer')
                if empirical_best is None:
                    empirical_best = self.results.get('empirical_best_layer_test_ece')
                if empirical_best is not None:
                    best_layer = next((layer for layer in self.layers_data if layer['layer_idx']==empirical_best), None)
                    if best_layer:
                        oracle_ece = best_layer.get('empirical_test_ece', None)
                        if oracle_ece is not None:
                            ax_bas.axhline(oracle_ece, linestyle='--', color='red', linewidth=2, label=f'Oracle ECE={oracle_ece:.4f}')
                            ax_bas.legend()
                ax_bas.set_title('Classical baselines (ECE on test)')
                ax_bas.set_ylabel('ECE ↓')
                ax_bas.grid(True, alpha=0.3)
        else:
            ax_bas.axis('off')
            ax_bas.set_title('Classical baselines (ECE on test)')

        ax6 = plt.subplot(4, 3, (10, 12))  # Bottom row, spans all columns
        
        # Create summary table
        summary_data = []
        if performance:
            for approach in approach_names:
                perf = performance[approach]
                summary_data.append([
                    approach.replace('_', ' ').title(),
                    f"Layer {perf['predicted_layer']}",
                    f"{perf['predicted_ece']:.4f}",
                    f"{perf['predicted_accuracy']:.3f}",
                    f"{perf['ece_gap_from_optimal']:+.4f}",
                    "✅" if perf['correct_prediction'] else "❌"
                ])
        
        if summary_data:
            columns = ['Approach', 'Predicted Layer', 'Test ECE', 'Test Accuracy', 'ECE Gap', 'Correct?']
            
            # Create table
            table = ax6.table(cellText=summary_data, colLabels=columns, 
                            cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            
            # Style the table
            for i in range(len(columns)):
                table[(0, i)].set_facecolor('#4CAF50')
                table[(0, i)].set_text_props(weight='bold', color='white')
            
            # Color code the rows
            colors = ['lightblue', 'lightgreen', 'lightyellow']
            for i, color in enumerate(colors[:len(summary_data)]):
                for j in range(len(columns)):
                    table[(i+1, j)].set_facecolor(color)
            
            ax6.set_title('Performance Summary Comparison', fontsize=12, fontweight='bold', pad=20)
            ax6.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✅ Multi-approach comparison plot saved to {output_path}")
    
    def generate_multi_approach_report(self) -> str:
        """Generate detailed report comparing all approaches"""
        report = []
        report.append("="*80)
        report.append("MULTI-COMPOSITE SCORE APPROACH COMPARISON REPORT")
        report.append("="*80)
        
        # Basic info
        config = self.results.get('config', {})
        report.append(f"\n📊 Experiment Configuration:")
        report.append(f"   Model: {config.get('model_name', 'Unknown')}")
        report.append(f"   Dataset: {config.get('dataset_name', 'Unknown')}")
        report.append(f"   Training: {config.get('training_method', 'Unknown')}")
        report.append(f"   Seed: {config.get('seed', 'Unknown')}")
        
        # Approach definitions
        report.append(f"\n🎯 TESTED APPROACHES:")
        report.append(f"   1. Minimal (3 metrics): spearman_stability_accuracy(2.0) + ece_score(2.0) + boundary_proximity_correlation(1.0)")
        report.append(f"   2. ECE-Dominant (3 metrics): ece_score(4.0) + kfold_ece_utility(2.0) + spearman_stability_accuracy(1.0)")
        report.append(f"   3. Pure ECE (1 metric): ece_score(1.0) only")
        
        # Performance analysis
        performance = self.analyze_approach_performance()
        
        if performance:
            report.append(f"\n📈 PERFORMANCE COMPARISON:")
            
            # Sort by ECE performance
            sorted_approaches = sorted(performance.items(), key=lambda x: x[1]['predicted_ece'])
            
            report.append(f"\n🏆 RANKING BY TEST ECE (Lower is Better):")
            for rank, (approach, perf) in enumerate(sorted_approaches, 1):
                status = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank}."
                approach_name = approach.replace('_', ' ').title()
                report.append(f"   {status} {approach_name}: ECE={perf['predicted_ece']:.4f}, Layer={perf['predicted_layer']}")
            
            # Gap analysis
            report.append(f"\n📊 GAP FROM OPTIMAL ANALYSIS:")
            best_gap = min(perf['ece_gap_from_optimal'] for perf in performance.values())
            
            for approach, perf in performance.items():
                approach_name = approach.replace('_', ' ').title()
                gap = perf['ece_gap_from_optimal']
                status = "✅ Perfect" if gap == 0 else "🟢 Excellent" if gap <= 0.001 else "🟡 Good" if gap <= 0.01 else "🔴 Poor"
                report.append(f"   {approach_name}: {gap:+.4f} ECE gap ({status})")
            
            # Correct predictions
            correct_predictions = sum(1 for perf in performance.values() if perf['correct_prediction'])
            total_approaches = len(performance)
            
            report.append(f"\n✅ CORRECT PREDICTIONS: {correct_predictions}/{total_approaches}")
            for approach, perf in performance.items():
                approach_name = approach.replace('_', ' ').title()
                status = "✅ Correct" if perf['correct_prediction'] else "❌ Incorrect"
                report.append(f"   {approach_name}: {status}")
        
        # Computational efficiency
        report.append(f"\n⚡ COMPUTATIONAL EFFICIENCY:")
        report.append(f"   Pure ECE: Fastest (1 metric computed)")
        report.append(f"   ECE-Dominant: Fast (2 metrics computed)")
        report.append(f"   Minimal: Moderate (3 metrics computed)")
        report.append(f"   Original Complex: Slowest (15+ metrics computed)")
        
        # Recommendations
        report.append(f"\n💡 RECOMMENDATIONS:")
        
        if performance:
            best_approach = min(performance.items(), key=lambda x: x[1]['predicted_ece'])
            best_name = best_approach[0].replace('_', ' ').title()
            best_ece = best_approach[1]['predicted_ece']
            
            report.append(f"   🏆 WINNER: {best_name} (ECE: {best_ece:.4f})")
            
            # Check if simple approaches work well
            pure_ece_perf = performance.get('pure_ece')
            if pure_ece_perf and pure_ece_perf['ece_gap_from_optimal'] <= 0.01:
                report.append(f"   🚀 Pure ECE performs well - consider using for simplicity")
            
            # Check if added complexity helps
            gaps = [perf['ece_gap_from_optimal'] for perf in performance.values()]
            if max(gaps) - min(gaps) < 0.005:  # Very small difference
                report.append(f"   ⚡ All approaches perform similarly - choose fastest (Pure ECE)")
            else:
                report.append(f"   🎯 Metric combination matters - use winning approach")
        
        # --- Classical baselines section ---
        baseline = self.results.get("baselines", {})
        if baseline:
            report.append("\n" + "="*80)
            report.append("CLASSICAL BASELINES")
            report.append("="*80)
            empirical_best = self.results.get('empirical_best_layer_test_ece')
            oracle_ece = None
            if empirical_best is not None:
                best_layer = next((layer for layer in self.layers_data if layer['layer_idx']==empirical_best), None)
                if best_layer:
                    oracle_ece = best_layer.get('empirical_test_ece', None)

            header = f"{'Method':<24}{'ECE':>10}{'Brier':>12}{'Acc':>10}{'ΔECE vs oracle':>16}"
            report.append(header)
            report.append("-"*len(header))

            def _row(name, key):
                r = baseline.get(key, {})
                if not r or r.get("ece") is None:
                    report.append(f"{name:<24}{'—':>10}{'—':>12}{'—':>10}{'(failed)':>16}")
                else:
                    de = (r['ece'] - oracle_ece) if oracle_ece is not None else np.nan
                    de_str = f"{de:>16.5f}" if (isinstance(de, (float, int)) and np.isfinite(de)) else f"{float('nan'):>16}"
                    report.append(f"{name:<24}{r['ece']:>10.5f}{r['brier']:>12.5f}{r['acc']:>10.4f}{de_str}")

            _row("Uncalibrated", "uncalibrated")
            _row("Temperature scaling", "temperature_scaling")
            _row("Isotonic (top-label)", "isotonic_toplabel")

        return "\n".join(report)




def create_optimized_score_weights(approach_name: str) -> Optional[Dict[str, float]]:
    """Create score weights for each approach, setting unused metrics to 0"""
    
    # Define the three approaches with CORRECT KEY NAMES (no "selector_" prefix)
    weight_configs = {
        'enhanced': None,  # Use selector defaults (all enhanced metrics & weights)
        'minimal': {
            "spearman_stability_accuracy": 2.0,
            "ece_score": 2.0,  # Note: using ece_score not validation_ece  
            "boundary_proximity_correlation": 1.0,
        },
        'ece_dominant': {
            "ece_score": 4.0,
            "kfold_ece_utility": 2.0,   # ✅ stabilizer
            "spearman_stability_accuracy": 1.0,
        },
        'pure_ece': {
            "ece_score": 1.0,  # Note: using ece_score not validation_ece
        }
    }
    
    # Get base config for this approach
    base_weights = weight_configs.get(approach_name, {})
    if base_weights is None:
        # Signal to use AdHocLayerSelector defaults
        return None
    
    # Create full weights dict with all metrics set to 0 except the ones we want
    # Use CORRECT metric names that match AdHocLayerSelector expectations
    all_possible_metrics = [
        "confidence_distance_correlation",
        "uncertainty_geometry_alignment",
        "boundary_proximity_correlation",
        "ece_score",
        "kfold_ece_utility",
        "spearman_stability_accuracy",
        "calibration_decisiveness",
        # include only if computed:
        # "validation_brier",
    ]
    
    # Set all to 0
    full_weights = {metric: 0.0 for metric in all_possible_metrics}
    
    # Override with our specific weights
    full_weights.update(base_weights)
    
    return full_weights


def get_required_metrics_for_approach(approach_name: str) -> List[str]:
    """Get list of metrics required for each approach - FIXED KEY NAMES"""
    metric_requirements = {
        'enhanced': ['(selector defaults)'],
        'minimal': [
            'spearman_stability_accuracy',
            'ece_score',  # ✅ Fixed: was 'validation_ece'
            'boundary_proximity_correlation'
        ],
        'ece_dominant': [
            'ece_score',  # ✅ Fixed: was 'validation_ece'
            'kfold_ece_utility',
            'spearman_stability_accuracy'
        ],
        'pure_ece': [
            'ece_score'  # ✅ Fixed: was 'validation_ece'
        ]
    }
    return metric_requirements.get(approach_name, [])

def run_smart_multi_approach_layer_selection(model_adapter, train_raw, train_labels, val_raw, val_labels, device, batch_size):
    """
    SMART: Run layer selection ONCE with minimal approach, then reuse metrics for other approaches
    """
    
    logger.info("🧠 SMART Multi-Approach Layer Selection")
    logger.info("   Strategy: Compute minimal metrics once, reuse for other approaches")
    
    # Step 1: Run layer selection ONCE with enhanced approach (compute ALL metrics)
    logger.info("\n🔍 Running layer selection with enhanced approach (computing all metrics)...")
    logger.info("   Active metrics: selector defaults (enhanced) — compute_all_metrics=True")
    
    # Create and fit calibrator with defaults and full metric computation
    calibrator = GeometricCalibrator(
        model=model_adapter,
        X_train_original=train_raw,
        y_train=train_labels,
        library="fast_separation",
        auto_select_layer=True,
        device=device,
        score_weights=None,              # use selector defaults (enhanced)
        compute_all_metrics=True         # force compute of all metrics
    )
    
    calibrator.fit(
        X_val_original=val_raw, 
        y_val=val_labels,
        fit_batch_size=batch_size
    )
    
    if not hasattr(calibrator, 'layer_metrics') or calibrator.layer_metrics is None:
        logger.error("❌ Layer selection failed - no metrics computed")
        return None
    
    logger.info(f"   ✅ Enhanced approach run complete; feature_layer = {calibrator.feature_layer}")
    
    # Step 2: Reuse the computed metrics to calculate other approaches
    logger.info("\n🔄 Reusing computed metrics for other approaches...")
    
    approaches = {
        'minimal': create_optimized_score_weights('minimal'),
        'ece_dominant': create_optimized_score_weights('ece_dominant'), 
        'pure_ece': create_optimized_score_weights('pure_ece')
    }
    
    approach_results = {}
    
    for approach_name, approach_weights in approaches.items():
        logger.info(f"\n   📊 Calculating {approach_name} composite scores from cached metrics...")
        
        # Recalculate composite scores using this approach's weights
        updated_metrics = []
        composite_scores = []
        
        for metrics in calibrator.layer_metrics:
            # Calculate composite score for this approach
            score = 0.0
            total_weight = 0.0
            used_metrics = []
            
            for metric_name, weight in approach_weights.items():
                if weight != 0:
                    value = getattr(metrics, metric_name, 0.0)
                    if not np.isnan(value) and not np.isinf(value):
                        # Values are already produced by the selector; use as-is
                        
                        score += weight * value
                        total_weight += abs(weight)
                        used_metrics.append((metric_name, value, weight))
            
                final_score = score / total_weight if total_weight > 0 else -np.inf
            composite_scores.append(final_score)
            
            # Create updated metrics with new composite score
            metrics_copy = LayerMetrics(layer_idx=metrics.layer_idx)
            
            # Copy all existing metric values
            for attr_name in dir(metrics):
                if not attr_name.startswith('_') and attr_name not in ['composite_score', 'computation_times']:
                    try:
                        setattr(metrics_copy, attr_name, getattr(metrics, attr_name))
                    except:
                        pass
            
            metrics_copy.composite_score = final_score
            updated_metrics.append(metrics_copy)
            
            # Log score breakdown for debugging
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"   Layer {metrics.layer_idx} {approach_name} score: {final_score:.4f}")
                for name, val, weight in used_metrics:
                    logger.debug(f"     {name}: {val:.4f} × {weight:.2f} = {val*weight:.4f}")
        
        # Find best layer for this approach
        best_idx = np.argmax(composite_scores)
        best_layer = calibrator.candidate_layers[best_idx]
        best_score = composite_scores[best_idx]
        
        approach_results[approach_name] = {
            'predicted_layer': best_layer,
            'layer_metrics': updated_metrics,
            'candidate_layers': calibrator.candidate_layers
        }
        
        # Log which metrics were used
        active_approach_metrics = [k for k, v in approach_weights.items() if v != 0.0]
        logger.info(f"     Used metrics: {active_approach_metrics}")
        logger.info(f"     ✅ {approach_name} selected layer: {best_layer} (score: {best_score:.4f})")
    
    # Step 3: Log efficiency gains
    total_layers = len(calibrator.candidate_layers)
    total_unique_metrics = 3  # ece_score, spearman_stability_accuracy, boundary_proximity_correlation
    
    logger.info(f"\n⚡ EFFICIENCY GAINS:")
    logger.info(f"   ❌ Naive approach: {total_layers * total_unique_metrics * 3} metric computations")
    logger.info(f"   ✅ Smart approach: {total_layers * total_unique_metrics * 1} metric computations") 
    logger.info(f"   🚀 Speedup: 3.0x faster (no redundant computation)!")
    logger.info(f"   💡 Reused metrics: ece_score, spearman_stability_accuracy")
    
    return approach_results



def run_multi_approach_layer_selection(model_adapter, train_raw, train_labels, val_raw, val_labels, device, batch_size):
    """Run layer selection with all three approaches and return results"""
    
    approaches = ['enhanced', 'minimal', 'ece_dominant', 'pure_ece']
    approach_results = {}
    
    for approach in approaches:
        logger.info(f"\n🔍 Running layer selection with {approach} approach...")
        
        # Get optimized weights for this approach
        score_weights = create_optimized_score_weights(approach)
        
        # Log which metrics are being used
        if score_weights is None:
            logger.info("   Active metrics: using selector default metrics and weights (enhanced)")
        else:
            active_metrics = [k for k, v in score_weights.items() if v != 0.0]
            logger.info(f"   Active metrics: {len(active_metrics)}")
            for metric in active_metrics:
                weight = score_weights[metric]
                clean_name = metric.replace('selector_', '').replace('_', ' ').title()
                logger.info(f"     - {clean_name}: {weight}")
        
        # Create calibrator with optimized weights
        calibrator = GeometricCalibrator(
            model=model_adapter,
            X_train_original=train_raw,
            y_train=train_labels,
            library="fast_separation",
            auto_select_layer=True,
            score_weights=score_weights  # None => defaults; dict => custom
        )
        
        # Fit calibrator
        calibrator.fit(
            X_val_original=val_raw, 
            y_val=val_labels,
            fit_batch_size=batch_size
        )
        
        if hasattr(calibrator, 'feature_layer') and calibrator.feature_layer is not None:
            approach_results[approach] = {
                'predicted_layer': calibrator.feature_layer,
                'layer_metrics': calibrator.layer_metrics,
                'candidate_layers': calibrator.candidate_layers
            }
            logger.info(f"   ✅ {approach} selected layer: {calibrator.feature_layer}")
        else:
            logger.error(f"   ❌ {approach} layer selection failed")
            approach_results[approach] = None
    
    return approach_results


def extract_features_directly(model, dataloader, layer_idx, device, fixed_feature_dim=512,
                             compression_ratio: float = None):
    """
    Extract features from a specific layer EXACTLY MATCHING AdHocLayerSelector._extract_features
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
    total_modules = len(layers)
    print(f"\n=== Debugging Layer {layer_idx} ===")
    print(f"Model type detected: {type(model)}")
    print(f"Model name: {model.__class__.__name__}")
    print(f"Total modules in model: {total_modules}")
    if 0 <= layer_idx < total_modules:
        debug_layer = layers[layer_idx]
        print(f"Layer {layer_idx} type: {type(debug_layer)}")
        print(f"Layer {layer_idx} name: {debug_layer.__class__.__name__}")
        if hasattr(debug_layer, "out_channels"):
            print(f"  Channels: {debug_layer.out_channels}")
    else:
        print(f"Layer {layer_idx} is out of bounds for total modules ({total_modules}).")
    print("\nFirst 10 layers:")
    for i, layer in enumerate(layers[:10]):
        print(f"  {i}: {layer.__class__.__name__}")
    print("\nLast 10 layers:")
    for i, layer in enumerate(layers[-10:], start=max(0, total_modules - 10)):
        print(f"  {i}: {layer.__class__.__name__}")
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
    
    feature_stats_logged = False
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
                if not feature_stats_logged:
                    try:
                        feat_np = feat.detach().cpu().float().numpy()
                        print(f"Layer {layer_idx} feature stats:")
                        print(f"  Shape: {feat_np.shape}")
                        print(f"  Mean: {feat_np.mean():.6f}")
                        print(f"  Std: {feat_np.std():.6f}")
                        print(f"  Min: {feat_np.min():.6f}")
                        print(f"  Max: {feat_np.max():.6f}")
                        first_sample = feat_np[0].reshape(-1) if feat_np.ndim > 1 else np.array([feat_np[0]])
                        print(f"  First sample L2 norm: {float(np.linalg.norm(first_sample)):.6f}")
                    except Exception as e:
                        logger.warning(f"Failed to log feature stats for layer {layer_idx}: {e}")
                    finally:
                        feature_stats_logged = True
                
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
                        num_channels = fmap.size(1)
                        if num_channels not in spp_projectors:
                            target_dim = None if compression_ratio is not None else fixed_feature_dim
                            log_target = (f"ratio={compression_ratio}" if compression_ratio is not None
                                          else f"{fixed_feature_dim}")
                            logger.info(f"Creating new SPP+JL projector for C={num_channels} -> {log_target}")
                            from utils.compression_utils import FixedSizeSPP_JL
                            spp_projectors[num_channels] = FixedSizeSPP_JL(
                                in_channels=num_channels,
                                final_output_dim=target_dim,
                                compression_ratio=compression_ratio
                            ).to(device)
                        projector = spp_projectors[num_channels]
                        feat = projector(fmap)
                    else:
                        feat = tokens.mean(dim=1)
                elif feat.ndim > 3:
                    num_channels = feat.size(1)
                    if compression_ratio is None and fixed_feature_dim is None:
                        # No compression: flatten spatial dimensions (global average if 4D)
                        feat = feat.mean(dim=[2, 3]) if feat.ndim == 4 else feat.flatten(1)
                    else:
                        if num_channels not in spp_projectors:
                            target_dim = None if compression_ratio is not None else fixed_feature_dim
                            log_target = (f"ratio={compression_ratio}" if compression_ratio is not None
                                          else f"{fixed_feature_dim}")
                            logger.info(f"Creating new SPP+JL projector for C={num_channels} -> {log_target}")
                            from utils.compression_utils import FixedSizeSPP_JL
                            spp_projectors[num_channels] = FixedSizeSPP_JL(
                                in_channels=num_channels,
                                final_output_dim=target_dim,
                                compression_ratio=compression_ratio
                            ).to(device)
                        projector = spp_projectors[num_channels]
                        feat = projector(feat)
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
        activation.clear()
    
    if not features_list:
        raise ValueError("No features were extracted")
    
    features = torch.cat(features_list, dim=0)
    labels = torch.cat(labels_list, dim=0)
    
    logger.info(f"✅ Final extracted shape for layer {layer_idx}: {features.shape}")
    
    feat_np = features.numpy()
    print(f"\n=== FEATURE FINGERPRINT Layer {layer_idx} ===")
    print(f"  Shape: {feat_np.shape}")
    print(f"  Mean: {feat_np.mean():.8f}")
    print(f"  Std: {feat_np.std():.8f}")
    print(f"  L2 norm: {np.linalg.norm(feat_np):.8f}")
    flat_vals = feat_np.reshape(-1)
    preview = flat_vals[:5] if flat_vals.size >= 5 else flat_vals
    print(f"  First 5 values: {preview}")
    print("=" * 50)
    
    return features, labels


def run_empirical_ground_truth_evaluation(model_adapter, candidate_layers, train_raw, train_labels, 
                                         val_raw, val_labels, test_raw, test_labels, device, batch_size):
    """
    Run empirical evaluation WITHOUT AdHocLayerSelector - just direct feature extraction
    """
    
    logger.info("🎯 Running empirical ground truth evaluation (NO layer selection metrics)")
    logger.info(f"   Evaluating {len(candidate_layers)} layers with direct feature extraction")
    
    raw_pytorch_model = model_adapter.model
    device_obj = torch.device(device)
    
    # Create dataloaders
    train_feat_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)), 
        batch_size=batch_size, shuffle=False
    )
    val_feat_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)), 
        batch_size=batch_size, shuffle=False
    )
    test_feat_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_raw), torch.from_numpy(test_labels)), 
        batch_size=batch_size, shuffle=False
    )
    
    evaluation_results = []
    first_layer_probs = None
    first_layer_idx = candidate_layers[0] if candidate_layers else None
    
    for layer_idx in tqdm(candidate_layers, desc="Empirically Evaluating Layers"):
        logger.info(f"🔍 Evaluating layer {layer_idx} empirically...")
        
        try:
            # Extract features DIRECTLY (no AdHocLayerSelector!)
            logger.debug(f"   Extracting features from layer {layer_idx}...")
            
            train_features, _ = extract_features_directly(
                raw_pytorch_model, train_feat_loader, layer_idx, device_obj
            )
            val_features, _ = extract_features_directly(
                raw_pytorch_model, val_feat_loader, layer_idx, device_obj
            )
            test_features, _ = extract_features_directly(
                raw_pytorch_model, test_feat_loader, layer_idx, device_obj
            )
            
            logger.debug(f"   ✅ Features extracted: train={train_features.shape}, val={val_features.shape}, test={test_features.shape}")
            
            # Fit calibrator WITHOUT layer selection (auto_select_layer=False)
            calibrator = GeometricCalibrator(
                model=model_adapter,
                X_train_embed=train_features.numpy(),
                y_train=train_labels,
                library="fast_separation",
                auto_select_layer=False,  # ← NO layer selection!
                device=device,
            )
            
            # Fit the calibrator
            calibrator.fit(
                X_val_embed=val_features.numpy(), 
                y_val=val_labels, 
                X_val_original=val_raw, 
                fit_batch_size=batch_size
            )
            
            # Evaluate on test set
            if calibrator.is_fitted:
                calibrated_probs = calibrator.calibrate_batched(
                    X_test_embed=test_features.numpy(), 
                    X_test_original=test_raw, 
                    batch_size=batch_size
                )
                
                # Calculate metrics
                test_ece = UncertaintyMetrics.calculate_ece(calibrated_probs, test_labels, n_bins=ECE_BIN_COUNT)
                test_brier = UncertaintyMetrics.calculate_brier_score(calibrated_probs, test_labels)
                test_acc = UncertaintyMetrics.calculate_accuracy(calibrated_probs, test_labels)
                
                logger.info(f"   ✅ Layer {layer_idx}: ECE={test_ece:.4f}, Acc={test_acc:.3f}")
                
                print(f"\n=== CALIBRATION RESULT Layer {layer_idx} ===")
                print(f"  Test ECE: {test_ece:.10f}")
                print(f"  Test Acc: {test_acc:.6f}")
                print(f"  Calibrated probs shape: {calibrated_probs.shape}")
                print(f"  Calibrated probs mean: {calibrated_probs.mean():.8f}")
                if first_layer_probs is not None:
                    identical = np.allclose(calibrated_probs, first_layer_probs)
                    print(f"  Are all probs identical to first layer? {identical}")
                else:
                    print("  Are all probs identical to first layer? N/A")
                print("=" * 50)
                
                if first_layer_probs is None and layer_idx == first_layer_idx:
                    first_layer_probs = calibrated_probs.copy()
                
            else:
                logger.warning(f"   ❌ Layer {layer_idx}: Calibrator failed to fit")
                test_ece, test_brier, test_acc = float('inf'), float('inf'), 0.0
            
        except Exception as e:
            logger.error(f"   ❌ Layer {layer_idx}: Failed with error: {e}")
            test_ece, test_brier, test_acc = float('inf'), float('inf'), 0.0
        
        evaluation_results.append({
            "layer_idx": layer_idx, 
            "test_ece": test_ece, 
            "test_brier": test_brier, 
            "test_accuracy": test_acc
        })
    
    logger.info(f"✅ Empirical evaluation complete for {len(candidate_layers)} layers")

    return evaluation_results


def screen_layers_with_meta_predictor(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    candidate_layers: List[int],
    device: torch.device,
    model_adapter: PyTorchModelAdapter,
    val_raw: np.ndarray
) -> Dict[int, Dict]:
    """
    Screen candidate layers using geometric quality meta-predictor.

    This analyzes whether geometric separation will work well for calibration
    on each layer WITHOUT running full calibration experiments.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        val_loader: Validation data loader
        candidate_layers: List of layer indices to screen
        device: Torch device
        model_adapter: Model adapter for predictions
        val_raw: Raw validation data for predictions

    Returns:
        Dictionary mapping layer_idx -> prediction results
    """
    from Experiments.analyze_geometric_quality import extract_layer_features

    logger.info("\n" + "="*80)
    logger.info("META-PREDICTOR SCREENING: Analyzing layer quality")
    logger.info("="*80)
    logger.info(f"Screening {len(candidate_layers)} candidate layers...")

    screening_results = {}

    # Get model predictions on validation set
    logger.info("Computing validation set predictions...")
    val_probs = model_adapter.predict_proba(val_raw, batch_size=256)
    val_predictions = np.argmax(val_probs, axis=1)

    for layer_idx in tqdm(candidate_layers, desc="Screening layers"):
        try:
            logger.info(f"\n🔍 Screening Layer {layer_idx}...")

            # Extract features
            train_features, train_labels = extract_layer_features(
                model, train_loader, layer_idx, device
            )
            val_features, val_labels = extract_layer_features(
                model, val_loader, layer_idx, device
            )

            # Create analyzer
            analyzer = GeometricQualityAnalyzer(
                features=val_features,
                labels=val_labels,
                train_features=train_features,
                train_labels=train_labels,
                predictions=val_predictions
            )

            # Get prediction
            prediction = analyzer.predict_calibration_success()

            screening_results[layer_idx] = prediction

            # Log summary
            will_work = "✅ WILL WORK" if prediction['will_work'] else "❌ UNLIKELY"
            confidence = prediction['confidence'].upper()
            n_cond = prediction['n_conditions_satisfied']

            logger.info(f"  Layer {layer_idx}: {will_work} ({confidence} confidence, {n_cond}/4 conditions)")
            logger.info(f"  Reason: {prediction['reason']}")

        except Exception as e:
            logger.error(f"Failed to screen layer {layer_idx}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            continue

    # Summary
    logger.info("\n" + "="*80)
    logger.info("META-PREDICTOR SCREENING SUMMARY")
    logger.info("="*80)

    will_work_layers = [idx for idx, pred in screening_results.items() if pred['will_work']]
    unlikely_layers = [idx for idx, pred in screening_results.items() if not pred['will_work']]

    logger.info(f"✅ Layers predicted to work well ({len(will_work_layers)}): {will_work_layers}")
    logger.info(f"❌ Layers unlikely to work ({len(unlikely_layers)}): {unlikely_layers}")

    if will_work_layers:
        # Find best layer
        best_layer = max(will_work_layers,
                        key=lambda idx: screening_results[idx]['n_conditions_satisfied'])
        best_pred = screening_results[best_layer]
        logger.info(f"\n🏆 Recommended layer: {best_layer} ({best_pred['n_conditions_satisfied']}/4 conditions)")
        logger.info(f"   {best_pred['reason']}")
    else:
        logger.warning("⚠️  No layers predicted to work well!")
        logger.warning("   Consider:")
        logger.warning("   - Using a different calibration method")
        logger.warning("   - Trying layers from a different part of the network")
        logger.warning("   - Checking if geometric separation is appropriate for this model/dataset")

    logger.info("="*80 + "\n")

    return screening_results


def run_multi_composite_experiment(args):
    """
    Executes the multi-composite score comparison experiment.
    """
    # Setup Environment & Paths
    logger.info(f"🚀 Setting up multi-composite experiment for seed {args.seed} on device {args.device}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Dataset normalization: use clean dataset for model loading and data, but keep provided
    # name (possibly with -c) for output directories and logging separation
    dataset_provided = args.dataset_name
    if 'cifar10' in dataset_provided:
        dataset_clean = 'cifar10'
    elif 'cifar100' in dataset_provided:
        dataset_clean = 'cifar100'
    else:
        dataset_clean = dataset_provided

    # If user passed a -c dataset, ALWAYS evaluate corruptions (mCE)
    try:
        if dataset_provided.endswith('-c'):
            setattr(args, 'eval_corruptions', True)
            logger.info("Detected CIFAR-C dataset; mCE evaluation will run and be reported in baselines.*.mce")
    except Exception:
        pass

    results_base_dir = args.results_base_dir
    output_base_dir = args.output_base_dir
    acc_suffix = None
    if getattr(args, "target_acc", None) is not None:
        acc_suffix = f"acc{int(args.target_acc)}"
        baseline_segment = os.path.normpath(os.path.join(acc_suffix, f"baseline_{acc_suffix}"))
        normalized_results_base = os.path.normpath(results_base_dir)
        if not normalized_results_base.endswith(baseline_segment):
            results_base_dir = os.path.join(results_base_dir, acc_suffix, f"baseline_{acc_suffix}")
        output_base_dir = os.path.join(output_base_dir, acc_suffix)
        logger.info(f"🎯 Using accuracy-specific paths: acc_suffix={acc_suffix}")
    
    # Construct paths
    exp_name = f"{args.training_method}_{dataset_clean}_{args.model_name}_seed{args.seed}"
    model_path = os.path.join(
        results_base_dir, args.training_method, dataset_clean,
        args.model_name, f"seed{args.seed}", exp_name, "best_model.pth"
    )
    
    output_dir = os.path.join(
        output_base_dir, args.training_method, dataset_provided,
        args.model_name, f"seed{args.seed}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    output_json_path = os.path.join(output_dir, "multi_composite_analysis.json")
    output_plot_path = os.path.join(output_dir, "multi_composite_comparison.png")
    output_report_path = os.path.join(output_dir, "multi_composite_report.txt")

    logger.info(f"📁 Model Path: {model_path}")
    logger.info(f"📁 Output Dir: {output_dir}")

    if not os.path.exists(model_path):
        logger.error(f"❌ Model not found at: {model_path}")
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Load Data
    logger.info(f"📊 Loading {dataset_clean} dataset...")
    num_classes = 10 if dataset_clean == 'cifar10' else 100
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        dataset_clean, args.batch_size, seed=args.seed
    )

    train_raw, train_labels = get_all_data_as_numpy(train_loader)
    val_raw, val_labels = get_all_data_as_numpy(val_loader)
    test_raw, test_labels = get_all_data_as_numpy(test_loader)

    # Load Model
    logger.info(f"🤖 Loading pretrained model: {args.model_name}")
    model = load_trained_model(model_path, args.model_name, num_classes, 
                              torch.device(args.device), dataset=dataset_clean)
    model_adapter = PyTorchModelAdapter(model, torch.device(args.device))
    raw_pytorch_model = model_adapter.model

    # === Early Accuracy Check (Skip if model is too poor) ===
    logger.info("🔍 Performing early accuracy check on test set...")
    try:
        test_probs_quick = model_adapter.predict_proba(test_raw, batch_size=min(256, args.batch_size))
        test_acc_quick = UncertaintyMetrics.calculate_accuracy(test_probs_quick, test_labels)
        logger.info(f"📊 Model test accuracy: {test_acc_quick:.2%}")
        
        ACCURACY_THRESHOLD = 0.65
        if test_acc_quick < ACCURACY_THRESHOLD:
            logger.warning(f"⚠️  Model accuracy ({test_acc_quick:.2%}) is below threshold ({ACCURACY_THRESHOLD:.0%})")
            logger.warning(f"⏩ SKIPPING EXPERIMENT - Model performance too low to warrant full analysis")
            logger.warning(f"💾 Saving minimal results file and exiting...")
            
            # Save a minimal results file indicating the skip
            skip_results = {
                "config": vars(args),
                "skipped": True,
                "skip_reason": f"Model accuracy ({test_acc_quick:.4f}) below threshold ({ACCURACY_THRESHOLD})",
                "test_accuracy": float(test_acc_quick),
                "accuracy_threshold": ACCURACY_THRESHOLD,
            }
            output_json_path = os.path.join(output_dir, "multi_composite_analysis.json")
            with open(output_json_path, 'w') as f:
                json.dump(skip_results, f, indent=4, cls=NumpyEncoder)
            logger.info(f"💾 Skip notice saved to {output_json_path}")
            logger.info("✅ Early exit complete.")
            return  # Exit the experiment early
        else:
            logger.info(f"✅ Model accuracy check passed ({test_acc_quick:.2%} >= {ACCURACY_THRESHOLD:.0%})")
    except Exception as e:
        logger.warning(f"⚠️  Could not perform early accuracy check: {e}")
        logger.info("Continuing with experiment anyway...")

    # (removed early CIFAR-C return; everything runs once and CIFAR-C is handled later)

    # Initialize empty approach_results for now (since the multi-approach section is disabled)
    approach_results = {}

    # === Baselines (run once per (model,dataset,seed)) ===
    # MOVED: Baselines now run AFTER candidate_layers_all calculation so DAC can use layer features
    baseline_results = None

    # === PART 2.5: All-metric scan + single-metric selection leaderboard (auto-direction) ===
    logger.info("\n" + "="*80)
    logger.info("PART 2.5: All-metric scan + single-metric selection leaderboard (auto-direction)")
    logger.info("="*80)

    # Cache read for full metric scan
    full_metrics_path = os.path.join(output_dir, "all_metrics_per_layer.json")
    if os.path.exists(full_metrics_path):
        try:
            with open(full_metrics_path, "r") as f:
                cached = json.load(f)
            candidate_layers_all = cached.get("candidate_layers", [])
            layer_metrics_all = []
            for d in cached.get("metrics", []):
                try:
                    lm = LayerMetrics(layer_idx=int(d.get("layer_idx")))
                    for k, v in d.items():
                        if k != "layer_idx":
                            setattr(lm, k, v)
                    layer_metrics_all.append(lm)
                except Exception:
                    continue
            logger.info("🟢 Loaded cached per-layer metrics; skipping scan.")
        except Exception:
            candidate_layers_all, layer_metrics_all = run_full_metric_scan(
                model_adapter, train_raw, train_labels, 
                val_raw, val_labels,      # FULL validation for metrics
                test_raw, test_labels,     # Test set available for holdout evaluation
                args.device, args.batch_size
            )
    else:
        candidate_layers_all, layer_metrics_all = run_full_metric_scan(
            model_adapter, train_raw, train_labels, 
            val_raw, val_labels,      # FULL validation for metrics
            test_raw, test_labels,     # Test set available for holdout evaluation
            args.device, args.batch_size
        )
    
    logger.info(f"📊 Found {len(candidate_layers_all)} candidate layers: {candidate_layers_all}")

    # === Baselines (run once per (model,dataset,seed)) ===
    # Run here because DAC needs candidate_layers_all to pick a feature layer
    if getattr(args, 'eval_baselines', True) and not getattr(args, 'mce_only', False):
        if baseline_results is None:
            baseline_results = run_standard_baselines(
                model_adapter=model_adapter,
                train_raw=train_raw,   train_labels=train_labels,
                val_raw=val_raw,   val_labels=val_labels,
                test_raw=test_raw, test_labels=test_labels,
                batch_size=args.batch_size,
                output_dir=output_dir,
                candidate_layers_all=candidate_layers_all,
                device=args.device,
                seed=args.seed,
            )

    # === META-PREDICTOR SCREENING (optional) ===
    screening_results = None
    if getattr(args, 'screen_layers', False):
        logger.info("\n🔍 Meta-predictor screening enabled - analyzing layer quality...")
        try:
            screening_results = screen_layers_with_meta_predictor(
                model=raw_pytorch_model,
                train_loader=train_loader,
                val_loader=val_loader,
                candidate_layers=candidate_layers_all,
                device=torch.device(args.device),
                model_adapter=model_adapter,
                val_raw=val_raw
            )

            # Save screening results
            screening_output_path = os.path.join(output_dir, "meta_predictor_screening.json")
            with open(screening_output_path, 'w') as f:
                json.dump(screening_results, f, indent=4, cls=NumpyEncoder)
            logger.info(f"💾 Screening results saved to: {screening_output_path}")

            # Optionally filter candidate layers based on screening
            # (Commented out by default - screening is informational only)
            # will_work_layers = [idx for idx, pred in screening_results.items() if pred['will_work']]
            # if will_work_layers:
            #     logger.info(f"🔬 Filtering to {len(will_work_layers)} layers predicted to work well")
            #     candidate_layers_all = will_work_layers
            #     # Would need to filter layer_metrics_all accordingly

        except Exception as e:
            logger.error(f"❌ Meta-predictor screening failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.info("Continuing with full candidate layer set...")

    # ===========================
    # CIFAR-C / mCE-ONLY FAST PATH
    # ===========================
    if getattr(args, 'mce_only', False):
        logger.info("mCE-only mode detected — running CIFAR-C evaluation now and skipping clean ECE parts.")

        # Use the same test transform as the clean test loader
        try:
            test_transform = getattr(test_loader.dataset, 'transform', None)
        except Exception:
            test_transform = None

        def _predict_uncal(Xc):
            P = model_adapter.predict_proba(Xc, batch_size=min(256, args.batch_size))
            P = np.clip(P, 1e-8, 1-1e-8); P /= P.sum(axis=1, keepdims=True)
            return P

        # 1) Uncalibrated
        mce_uncal = compute_mce_for_method(
            "uncalibrated", _predict_uncal,
            labels_getter=lambda: None,
            dataset_name=dataset_clean,
            test_transform=test_transform,
            batch_size=args.batch_size,
            cifar_c_dir=getattr(args, 'cifar_c_dir', None)
        )

        # 2) Temperature Scaling (fit once on clean val)
        ts_obj = None
        try:
            logits_val = model_adapter.predict_logits(val_raw, batch_size=min(256, args.batch_size))
            ts_obj = TemperatureScaling(); ts_obj.fit(logits_val, val_labels)
        except Exception as e:
            logger.warning(f"TS fit (for CIFAR-C) failed, skipping: {e}")

        def _predict_ts(Xc):
            if ts_obj is None:
                return _predict_uncal(Xc)
            logits_c = model_adapter.predict_logits(Xc, batch_size=min(256, args.batch_size))
            return ts_obj.calibrate(logits_c)

        mce_ts = compute_mce_for_method(
            "temperature_scaling", _predict_ts,
            labels_getter=lambda: None,
            dataset_name=dataset_clean,
            test_transform=test_transform,
            batch_size=args.batch_size,
            cifar_c_dir=getattr(args, 'cifar_c_dir', None)
        )

        # 3) Other post-hoc calibrators (fit once on clean val)
        iso_obj = platt_obj = beta_obj = None
        try:
            logits_val = model_adapter.predict_logits(val_raw, batch_size=min(256, args.batch_size))
            try:
                iso_obj = TopLabelIsotonicCalibrator(); iso_obj.fit(logits_val, val_labels)
            except Exception as e:
                logger.warning(f"Isotonic fit (for CIFAR-C) failed: {e}")
            try:
                platt_obj = PlattScaling(); platt_obj.fit(logits_val, val_labels)
            except Exception as e:
                logger.warning(f"Platt fit (for CIFAR-C) failed: {e}")
            try:
                beta_obj = BetaCalibration(); beta_obj.fit(logits_val, val_labels)
            except Exception as e:
                logger.warning(f"Beta fit (for CIFAR-C) failed: {e}")
        except Exception as e:
            logger.warning(f"Could not compute val logits for CIFAR-C baselines: {e}")

        def _predict_iso(Xc):
            if iso_obj is None:
                return _predict_uncal(Xc)
            logits_c = model_adapter.predict_logits(Xc, batch_size=min(256, args.batch_size))
            return iso_obj.calibrate(logits_c)

        def _predict_platt(Xc):
            if platt_obj is None:
                return _predict_uncal(Xc)
            logits_c = model_adapter.predict_logits(Xc, batch_size=min(256, args.batch_size))
            return platt_obj.calibrate(logits_c)

        def _predict_beta(Xc):
            if beta_obj is None:
                return _predict_uncal(Xc)
            logits_c = model_adapter.predict_logits(Xc, batch_size=min(256, args.batch_size))
            return beta_obj.calibrate(logits_c)

        mce_iso = compute_mce_for_method(
            "isotonic_toplabel", _predict_iso,
            labels_getter=lambda: None,
            dataset_name=dataset_clean,
            test_transform=test_transform,
            batch_size=args.batch_size,
            cifar_c_dir=getattr(args, 'cifar_c_dir', None)
        )
        mce_platt = compute_mce_for_method(
            "platt_scaling", _predict_platt,
            labels_getter=lambda: None,
            dataset_name=dataset_clean,
            test_transform=test_transform,
            batch_size=args.batch_size,
            cifar_c_dir=getattr(args, 'cifar_c_dir', None)
        )
        mce_beta = compute_mce_for_method(
            "beta_calibration", _predict_beta,
            labels_getter=lambda: None,
            dataset_name=dataset_clean,
            test_transform=test_transform,
            batch_size=args.batch_size,
            cifar_c_dir=getattr(args, 'cifar_c_dir', None)
        )

        # 4) Geometric Physical Space (SPP+JL on raw images)
        logger.info("🧪 Calculating Geometric Physical Space mCE (SPP+JL compression)...")
        geo_physical_obj = None
        mce_geo_physical = float('nan')
        try:
            FINAL_DIM = 1024
            PYR = [4, 2, 1]
            device = args.device

            Xtr_c = compress_with_spp_jl(
                train_raw, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
                seed=42, batch_size=min(512, args.batch_size), device=device
            )
            Xva_c = compress_with_spp_jl(
                val_raw, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
                seed=42, batch_size=min(512, args.batch_size), device=device
            )

            geo_physical_obj = GeometricCalibrator(
                model=model_adapter,
                X_train_embed=Xtr_c,
                y_train=train_labels,
                auto_select_layer=False,
                library="fast_separation",
            )
            geo_physical_obj.fit(X_val_embed=Xva_c, X_val_original=val_raw, 
                                y_val=val_labels, fit_batch_size=args.batch_size)
            
            def _predict_geo_physical(Xc):
                Xte_c = compress_with_spp_jl(
                    Xc, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
                    seed=42, batch_size=min(512, args.batch_size), device=device
                )
                return geo_physical_obj.calibrate_batched(X_test_embed=Xte_c, 
                                                         X_test_original=Xc, 
                                                         batch_size=args.batch_size)
            
            mce_geo_physical = compute_mce_for_method(
                "geometric_physical_space", _predict_geo_physical,
                labels_getter=lambda: None,
                dataset_name=dataset_clean,
                test_transform=test_transform,
                batch_size=args.batch_size,
                cifar_c_dir=getattr(args, 'cifar_c_dir', None)
            )
            logger.info(f"✅ Geometric Physical Space mCE: {mce_geo_physical:.5f}")
        except Exception as e:
            logger.warning(f"Geometric Physical Space mCE failed: {e}")

        # 5) Geometric calibrator — per-layer mCE (semantic space)
        raw_model = model_adapter.model
        dev = torch.device(args.device)
        tr_loader = DataLoader(TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                               batch_size=args.batch_size, shuffle=False)
        va_loader = DataLoader(TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                               batch_size=args.batch_size, shuffle=False)

        geo_mce_per_layer = {}
        best_layer_mce = None
        best_layer = None
        for L in candidate_layers_all:
            try:
                Xtr_f, _ = extract_features_directly(raw_model, tr_loader, L, dev)
                Xva_f, _ = extract_features_directly(raw_model, va_loader, L, dev)
                geo_L = GeometricCalibrator(
                    model=model_adapter,
                    X_train_embed=Xtr_f.numpy(),
                    y_train=train_labels,
                    library="fast_separation",
                    auto_select_layer=False,
                    device=args.device,
                )
                geo_L.fit(X_val_embed=Xva_f.numpy(), y_val=val_labels,
                          X_val_original=val_raw, fit_batch_size=args.batch_size)

                def _predict_geo_L(Xc):
                    te_loader = DataLoader(TensorDataset(torch.from_numpy(Xc),
                                                         torch.zeros(len(Xc), dtype=torch.long)),
                                           batch_size=args.batch_size, shuffle=False)
                    Xte_f, _ = extract_features_directly(raw_model, te_loader, L, dev)
                    return geo_L.calibrate_batched(X_test_embed=Xte_f.numpy(),
                                                   X_test_original=Xc,
                                                   batch_size=args.batch_size)

                mce_L = compute_mce_for_method(
                    f"geometric_semantic_L{L}", _predict_geo_L,
                    labels_getter=lambda: None,
                    dataset_name=dataset_clean,
                    test_transform=test_transform,
                    batch_size=args.batch_size,
                    cifar_c_dir=getattr(args, 'cifar_c_dir', None)
                )
                geo_mce_per_layer[int(L)] = float(mce_L)
                if best_layer_mce is None or mce_L < best_layer_mce:
                    best_layer_mce, best_layer = float(mce_L), int(L)
            except Exception as e:
                logger.warning(f"Per-layer geometric mCE failed at L{L}: {e}")

        mce_geo = best_layer_mce if best_layer_mce is not None else float('nan')

        final_results = {
            "config": vars(args),
            "candidate_layers": list(map(int, candidate_layers_all)),
            "baselines": {
                "uncalibrated": {"mce": mce_uncal},
                "temperature_scaling": {"mce": mce_ts},
                "isotonic_toplabel": {"mce": mce_iso},
                "platt_scaling": {"mce": mce_platt},
                "beta_calibration": {"mce": mce_beta},
                "geometric_physical_space": {"mce": mce_geo_physical},
                "geometric_semantic": {
                    "mce": mce_geo,
                    "best_layer": best_layer,
                    "mce_per_layer": {str(k): v for k, v in geo_mce_per_layer.items()},
                },
            },
        }

        with open(output_json_path, 'w') as f:
            json.dump(final_results, f, indent=4, sort_keys=True, cls=NumpyEncoder)
        logger.info(f"💾 (mCE) multi-composite results saved to {output_json_path}")

        try:
            mce_dump_path = os.path.join(output_dir, "cifar_c_mce_summary.json")
            mce_summary_data = {
                "uncalibrated": final_results["baselines"]["uncalibrated"]["mce"],
                "temperature_scaling": final_results["baselines"]["temperature_scaling"]["mce"],
                "isotonic_toplabel": final_results["baselines"]["isotonic_toplabel"]["mce"],
                "platt_scaling": final_results["baselines"]["platt_scaling"]["mce"],
                "beta_calibration": final_results["baselines"]["beta_calibration"]["mce"],
                "geometric_physical_space": final_results["baselines"]["geometric_physical_space"]["mce"],
                "geometric_semantic": final_results["baselines"]["geometric_semantic"],
            }
            with open(mce_dump_path, "w") as f:
                json.dump(format_numbers_7digits(mce_summary_data), f, indent=2, cls=NumpyEncoder)
            logger.info(f"💾 CIFAR-C mCE summary saved: {mce_dump_path}")
        except Exception as e:
            logger.warning(f"Failed to save cifar_c_mce_summary.json: {e}")

        # Save per-layer mCE ground truth
        gt_mce_path = os.path.join(output_dir, "per_layer_ground_truth_mce.json")
        gt_mce_data = {
            "candidate_layers": list(map(int, candidate_layers_all)),
            "test": {"mce": {str(k): float(v) for k, v in geo_mce_per_layer.items()}},
        }
        with open(gt_mce_path, "w") as f:
            json.dump(format_numbers_7digits(gt_mce_data), f, indent=2, cls=NumpyEncoder)
        logger.info(f"💾 Ground-truth per-layer (mCE) saved: {gt_mce_path}")

        logger.info("✅ mCE-only flow finished.")
        return

    # PART 2: Empirical Evaluation (NO AdHocLayerSelector!)
    if not getattr(args, 'mce_only', False):
        logger.info("\n" + "="*80)
        logger.info("PART 2: Empirical Ground Truth Evaluation (Direct Feature Extraction)")
        logger.info("="*80)

        # Cache read for empirical ground truth
        gt_path = os.path.join(output_dir, "per_layer_ground_truth.json")
        if os.path.exists(gt_path):
            try:
                with open(gt_path, "r") as f:
                    gt = json.load(f)
                candidate_layers_all = gt.get("candidate_layers", candidate_layers_all)
                evaluation_results = [
                    {
                        "layer_idx": int(k),
                        "test_ece": float(v),
                        "test_brier": float(gt.get("test", {}).get("brier", {}).get(k, np.nan)),
                        "test_accuracy": float(gt.get("test", {}).get("acc", {}).get(k, np.nan)),
                    }
                    for k, v in gt.get("test", {}).get("ece", {}).items()
                ]
                logger.info("🟢 Loaded cached empirical ground truth; skipping re-evaluation.")
            except Exception:
                evaluation_results = run_empirical_ground_truth_evaluation(
                    model_adapter, candidate_layers_all, train_raw, train_labels,
                    val_raw, val_labels, test_raw, test_labels, args.device, args.batch_size
                )
        else:
            # Use the new empirical evaluation function
            evaluation_results = run_empirical_ground_truth_evaluation(
                model_adapter, candidate_layers_all, train_raw, train_labels, 
                val_raw, val_labels, test_raw, test_labels, args.device, args.batch_size
            )

        # Save compact per-layer ground truth (no selector filtering)
        gt_path = os.path.join(output_dir, "per_layer_ground_truth.json")
        try:
            test_ece_map   = {int(r["layer_idx"]): float(r["test_ece"])   for r in evaluation_results}
            test_acc_map   = {int(r["layer_idx"]): float(r["test_accuracy"]) for r in evaluation_results}
            test_brier_map = {int(r["layer_idx"]): float(r["test_brier"]) for r in evaluation_results}

            # (Optional) compute val ECE with the same routine if you want parity
            val_ece_map = {}  # fill similarly if you compute it

            with open(gt_path, "w") as f:
                json.dump({
                    "candidate_layers": list(map(int, candidate_layers_all)),
                    "val":  {"ece": {str(k): v for k, v in val_ece_map.items()}},
                    "test": {
                        "ece":   {str(k): v for k, v in test_ece_map.items()},
                        "acc":   {str(k): v for k, v in test_acc_map.items()},
                        "brier": {str(k): v for k, v in test_brier_map.items()},
                    },
                }, f, indent=2, cls=NumpyEncoder)
            logger.info(f"💾 Ground-truth per-layer saved: {gt_path}")
        except Exception as e:
            logger.warning(f"Failed to save ground-truth JSON: {e}")
    else:
        evaluation_results = []

    # --- mCE ground truth (corrupted) ---
    try:
        if getattr(args, "eval_corruptions", False):
            geo_mce_per_layer_local = locals().get("geo_mce_per_layer", None)
            if not geo_mce_per_layer_local:
                mce_summary = os.path.join(output_dir, "cifar_c_mce_summary.json")
                if os.path.exists(mce_summary):
                    with open(mce_summary, "r") as _f:
                        _D = json.load(_f)
                    geo_mce_per_layer_local = _D.get("geometric_semantic", {}).get("mce_per_layer", None)

            if geo_mce_per_layer_local:
                mce_map = {int(k): float(v) for k, v in geo_mce_per_layer_local.items()}
                gt_mce_path = os.path.join(output_dir, "per_layer_ground_truth_mce.json")
                gt_mce_data = {
                    "candidate_layers": list(map(int, candidate_layers_all)),
                    "test": {"mce": {str(k): v for k, v in mce_map.items()}},
                }
                with open(gt_mce_path, "w") as f:
                    json.dump(format_numbers_7digits(gt_mce_data), f, indent=2, cls=NumpyEncoder)
                logger.info(f"💾 Ground-truth per-layer (mCE) saved: {gt_mce_path}")
            else:
                logger.info("ℹ️ Skipping per-layer mCE ground-truth save (no per-layer mCE available yet).")
    except Exception as e:
        logger.warning(f"Failed to save per_layer_ground_truth_mce.json: {e}")

    # ===== Learned linear scorer (ridge/lasso/enet) =====
    learned_best_layer = None
    learned_scores: Dict[int, float] = {}
    learned_pred_gap: Dict[int, float] = {}
    learned_json_path = None
    if args.weights_csv and os.path.exists(args.weights_csv):
        try:
            coef, intercept = load_linear_weights_csv(args.weights_csv)
            if layer_metrics_all:
                learned_scores, learned_pred_gap = score_layers_with_linear_model(
                    layer_metrics_all, coef, intercept
                )

                # choose by lowest predicted gap (i.e., highest -gap score)
                learned_best_layer, _ = pick_best_by_scores(
                    candidate_layers_all, learned_scores
                )
            else:
                logger.warning("No per-layer metrics available for learned scorer; skipping.")

            # persist both score (−gap) and raw predicted gap
            learned_out = {
                "predicted_gap": {str(k): float(v) for k, v in learned_pred_gap.items()},
                "score":         {str(k): float(v) for k, v in learned_scores.items()},
                "chosen_layer":  int(learned_best_layer) if learned_best_layer is not None else None,
            }
            weights_tag = os.path.splitext(os.path.basename(args.weights_csv))[0]
            learned_json_path = os.path.join(output_dir, f"{weights_tag}_layer_scores.json")
            with open(learned_json_path, "w") as f:
                json.dump(learned_out, f, indent=2, cls=NumpyEncoder)
            logger.info(f"💾 Learned scorer saved: {learned_json_path}")
        except Exception as e:
            logger.warning(f"Failed to compute learned linear scores: {e}")

    # Save full metric dump for reproducibility
    full_metrics_path = os.path.join(output_dir, "all_metrics_per_layer.json")
    try:
        with open(full_metrics_path, 'w') as f:
            json.dump({
                "candidate_layers": list(map(int, candidate_layers_all)),
                "metrics": _serialize_layer_metrics(layer_metrics_all),
            }, f, indent=2, cls=NumpyEncoder)
        logger.info(f"💾 Full per-layer metrics saved: {full_metrics_path}")
    except Exception as e:
        logger.warning(f"Failed to save full metrics JSON: {e}")

    # ---- helpers ----
    def _metric_field_names(example: LayerMetrics) -> List[str]:
        skip = {"layer_idx","composite_score","computation_times"}
        return [k for k in vars(example).keys() if k not in skip]

    def _rank_like(x: np.ndarray) -> np.ndarray:
        # Simple rank transform (ties get arbitrary but stable order)
        return np.argsort(np.argsort(x))

    def _infer_direction(metric_vals: np.ndarray, test_eces: np.ndarray) -> int:
        """
        Infer whether to maximize (+1) or minimize (-1) this metric to lower ECE.
        Uses Spearman-like correlation of ranks; falls back to argmin/argmax comparison.
        """
        ok = np.isfinite(metric_vals) & np.isfinite(test_eces)
        if ok.sum() < 3:
            try:
                i_max = int(np.nanargmax(metric_vals))
                i_min = int(np.nanargmin(metric_vals))
                return +1 if test_eces[i_max] < test_eces[i_min] else -1
            except Exception:
                return -1
        rx = _rank_like(metric_vals[ok]).astype(float)
        ry = _rank_like(test_eces[ok]).astype(float)
        if rx.size < 3:
            return -1
        r = np.corrcoef(rx, ry)[0, 1]
        if not np.isfinite(r):
            i_max = int(np.nanargmax(metric_vals))
            i_min = int(np.nanargmin(metric_vals))
            return +1 if test_eces[i_max] < test_eces[i_min] else -1
        return +1 if r < 0 else -1

    # Build arrays per layer to line up results
    layer_to_ece = {r["layer_idx"]: r["test_ece"] for r in evaluation_results}
    ordered_test_ece = np.array([layer_to_ece.get(int(L), np.inf) for L in candidate_layers_all], dtype=float)

    # empirical best (ground truth)
    empirical_best_idx = int(np.nanargmin(ordered_test_ece))
    empirical_best_layer = int(candidate_layers_all[empirical_best_idx])
    empirical_best_ece = float(ordered_test_ece[empirical_best_idx])

    # Collect per-metric winners (both directions)
    names = _metric_field_names(layer_metrics_all[0]) if layer_metrics_all else []
    # --- BEGIN PATCH A: per-metric winners in BOTH directions ---
    from typing import Any, Optional
    per_metric_winners: Dict[str, Any] = {}

    for mname in names:
        vals = np.array([getattr(lm, mname, np.nan) for lm in layer_metrics_all], dtype=float)
        # Skip degenerate metrics
        if np.all(~np.isfinite(vals)) or np.nanstd(vals) == 0:
            continue

        def _safe_argmin(x: np.ndarray) -> Optional[int]:
            ok = np.isfinite(x)
            if ok.any():
                return int(np.argmin(np.where(ok, x, np.inf)))
            return None

        def _safe_argmax(x: np.ndarray) -> Optional[int]:
            ok = np.isfinite(x)
            if ok.any():
                return int(np.argmax(np.where(ok, x, -np.inf)))
            return None

        picks = {
            "minimize": _safe_argmin(vals),
            "maximize": _safe_argmax(vals),
        }

        for direction_name, pick_idx in picks.items():
            if pick_idx is None:
                continue
            chosen_layer = int(candidate_layers_all[pick_idx])
            chosen_ece = float(ordered_test_ece[pick_idx])
            gap = chosen_ece - empirical_best_ece

            metric_key = f"{mname}_{direction_name}"
            per_metric_winners[metric_key] = {
                "base_metric": mname,
                "direction": direction_name,  # "minimize" | "maximize"
                "chosen_layer": chosen_layer,
                "chosen_metric_value": _format_number_7digits(float(vals[pick_idx])),
                "test_ece": _format_number_7digits(chosen_ece),
                "ece_gap_from_optimal": _format_number_7digits(gap),
                "hit_optimal": (chosen_layer == empirical_best_layer),
            }

    # Leaderboard ranks both metric+direction pairs by realized ECE gap (lower is better)
    leaderboard = sorted(
        per_metric_winners.items(),
        key=lambda kv: (
            not (isinstance(kv[1].get("ece_gap_from_optimal"), (int, float)) and np.isfinite(float(kv[1]["ece_gap_from_optimal"]))),
            float(kv[1]["ece_gap_from_optimal"]) if isinstance(kv[1].get("ece_gap_from_optimal"), (int, float)) else float("inf")
        )
    )

    per_metric_results = {
        "empirical_best_layer": empirical_best_layer,
        "empirical_best_ece": _format_number_7digits(empirical_best_ece),
        "winners": per_metric_winners,   # includes _minimize/_maximize keys
        "leaderboard": leaderboard,
    }
    # --- END PATCH A ---

    per_metric_json = os.path.join(output_dir, "per_metric_selection.json")
    with open(per_metric_json, 'w') as f:
        json.dump(per_metric_results, f, indent=2, cls=NumpyEncoder)
    logger.info(f"💾 Per-metric results saved: {per_metric_json}")

    # quick leaderboard plot (same filename as before)
    try:
        top_k = min(25, len(leaderboard))
        labels = [k for k, _ in leaderboard[:top_k]]
        gaps = [v["ece_gap_from_optimal"] for _, v in leaderboard[:top_k]]
        hits = [v["hit_optimal"] for _, v in leaderboard[:top_k]]
        colors = ['tab:green' if h else 'tab:red' for h in hits]
        x = np.arange(len(labels))
        plt.figure(figsize=(12, 6))
        plt.bar(x, gaps, color=colors, alpha=0.9)
        plt.axhline(0, color='black', linewidth=2)
        plt.xticks(x, labels, rotation=60, ha='right')
        plt.ylabel('ECE gap from empirical best (↓ better)')
        plt.title('Single-metric layer pickers — leaderboard (auto-direction)')
        plt.tight_layout()
        per_metric_png = os.path.join(output_dir, "per_metric_leaderboard.png")
        plt.savefig(per_metric_png, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"📈 Per-metric leaderboard saved: {per_metric_png}")
    except Exception as e:
        logger.warning(f"Per-metric leaderboard plotting failed: {e}")

    # --- mCE version of per-metric selection & leaderboard ---
    try:
        if getattr(args, "eval_corruptions", False):
            geo_mce_per_layer_local = locals().get("geo_mce_per_layer", None)
            if not geo_mce_per_layer_local:
                mce_summary = os.path.join(output_dir, "cifar_c_mce_summary.json")
                if os.path.exists(mce_summary):
                    with open(mce_summary, "r") as _f:
                        _D = json.load(_f)
                    geo_mce_per_layer_local = _D.get("geometric_semantic", {}).get("mce_per_layer", None)

            if geo_mce_per_layer_local:
                layer_to_mce = {int(k): float(v) for k, v in geo_mce_per_layer_local.items()}
                ordered_test_mce = np.array([layer_to_mce.get(int(L), np.inf) for L in candidate_layers_all], dtype=float)
                emp_best_mce_idx = int(np.nanargmin(ordered_test_mce))
                emp_best_mce_layer = int(candidate_layers_all[emp_best_mce_idx])
                emp_best_mce = float(ordered_test_mce[emp_best_mce_idx])

                names_m = _metric_field_names(layer_metrics_all[0]) if layer_metrics_all else []
                per_metric_winners_mce = {}

                for mname in names_m:
                    vals = np.array([getattr(lm, mname, np.nan) for lm in layer_metrics_all], dtype=float)
                    if np.all(~np.isfinite(vals)) or np.nanstd(vals) == 0:
                        continue

                    def _safe_argmin(x: np.ndarray):
                        ok = np.isfinite(x)
                        return int(np.argmin(np.where(ok, x, np.inf))) if ok.any() else None

                    def _safe_argmax(x: np.ndarray):
                        ok = np.isfinite(x)
                        return int(np.argmax(np.where(ok, x, -np.inf))) if ok.any() else None

                    picks = {"minimize": _safe_argmin(vals), "maximize": _safe_argmax(vals)}

                    for direction_name, pick_idx in picks.items():
                        if pick_idx is None:
                            continue
                        chosen_layer = int(candidate_layers_all[pick_idx])
                        chosen_mce = float(ordered_test_mce[pick_idx])
                        gap = chosen_mce - emp_best_mce
                        metric_key = f"{mname}_{direction_name}"
                        per_metric_winners_mce[metric_key] = {
                            "base_metric": mname,
                            "direction": direction_name,
                            "chosen_layer": chosen_layer,
                            "chosen_metric_value": _format_number_7digits(float(vals[pick_idx])),
                            "test_mce": _format_number_7digits(chosen_mce),
                            "mce_gap_from_optimal": _format_number_7digits(gap),
                            "hit_optimal": (chosen_layer == emp_best_mce_layer),
                        }

                leaderboard_mce = sorted(
                    per_metric_winners_mce.items(),
                    key=lambda kv: (
                        not (isinstance(kv[1].get("mce_gap_from_optimal"), (int, float)) and np.isfinite(float(kv[1]["mce_gap_from_optimal"]))),
                        float(kv[1]["mce_gap_from_optimal"]) if isinstance(kv[1].get("mce_gap_from_optimal"), (int, float)) else float("inf")
                    )
                )

                per_metric_results_mce = {
                    "empirical_best_layer": emp_best_mce_layer,
                    "empirical_best_mce": _format_number_7digits(emp_best_mce),
                    "winners": per_metric_winners_mce,
                    "leaderboard": leaderboard_mce,
                }

                per_metric_json_mce = os.path.join(output_dir, "per_metric_selection_mce.json")
                with open(per_metric_json_mce, "w") as f:
                    json.dump(per_metric_results_mce, f, indent=2, cls=NumpyEncoder)
                logger.info(f"💾 Per-metric (mCE) results saved: {per_metric_json_mce}")

                # leaderboard plot (mCE)
                try:
                    top_k = min(25, len(leaderboard_mce))
                    labels = [k for k, _ in leaderboard_mce[:top_k]]
                    gaps  = [v["mCE_gap_from_optimal"] if "mCE_gap_from_optimal" in v else v["mce_gap_from_optimal"] for _, v in leaderboard_mce[:top_k]]
                    hits  = [v["hit_optimal"] for _, v in leaderboard_mce[:top_k]]
                    colors = ['tab:green' if h else 'tab:red' for h in hits]
                    x = np.arange(len(labels))
                    plt.figure(figsize=(12, 6))
                    plt.bar(x, gaps, color=colors, alpha=0.9)
                    plt.axhline(0, color='black', linewidth=2)
                    plt.xticks(x, labels, rotation=60, ha='right')
                    plt.ylabel('mCE gap from empirical best (↓ better)')
                    plt.title('Single-metric layer pickers — leaderboard (mCE, auto-direction)')
                    plt.tight_layout()
                    per_metric_png_mce = os.path.join(output_dir, "per_metric_leaderboard_mce.png")
                    plt.savefig(per_metric_png_mce, dpi=150, bbox_inches='tight')
                    plt.close()
                    logger.info(f"📈 Per-metric leaderboard (mCE) saved: {per_metric_png_mce}")
                except Exception as e:
                    logger.warning(f"Per-metric mCE leaderboard plotting failed: {e}")
            else:
                logger.info("ℹ️ Skipping per-metric mCE artifacts (no per-layer mCE available).")
    except Exception as e:
        logger.warning(f"Failed to create/save mCE per-metric artifacts: {e}")

    # PART 3: Analysis and Results
    logger.info("\n" + "="*80)
    logger.info("PART 3: Multi-Approach Analysis and Comparison")
    logger.info("="*80)

    if not evaluation_results:
        logger.error("❌ No layers were successfully evaluated.")
        return
        
    # Find best layer empirically
    empirical_best_layer_info = min(evaluation_results, key=lambda x: x['test_ece'])
    empirical_best_layer = empirical_best_layer_info['layer_idx']
    
    # Prepare comprehensive results (add new clearer keys; keep legacy for back-compat)
    final_results = {
        "config": vars(args),
        "empirical_best_layer": int(empirical_best_layer),
        "empirical_best_ece": float(empirical_best_layer_info['test_ece']),
        "empirical_best_layer_test_ece": int(empirical_best_layer),
        "results_per_layer": []
    }
    # Attach baselines if available
    if 'baseline_results' in locals() and baseline_results is not None:
        final_results["baselines"] = baseline_results
    
    # Add predictions for each approach
    for approach, result in approach_results.items():
        if result is not None:
            final_results[f'selector_prediction_{approach}'] = result['predicted_layer']
        else:
            final_results[f'selector_prediction_{approach}'] = None

    # Add learned predictor layer (if available)
    final_results['selector_prediction_learned'] = learned_best_layer
    # Also stash learned scores to be accessible by analyzer if needed
    final_results['learned_scores'] = learned_scores

    # Build enhanced layer results with composite scores for each approach
    for result in evaluation_results:
        layer_idx = result['layer_idx']
        
        layer_result = {
            "layer_idx": layer_idx,
            "empirical_test_ece": _format_number_7digits(result['test_ece']),
            "empirical_test_brier": _format_number_7digits(result['test_brier']),
            "empirical_test_accuracy": _format_number_7digits(result['test_accuracy'])
        }
        
        # Add composite scores for each approach
        for approach, approach_result in approach_results.items():
            if approach_result is not None:
                # Find layer metrics for this layer in this approach
                layer_metrics = next((m for m in approach_result['layer_metrics'] if m.layer_idx == layer_idx), None)
                if layer_metrics:
                    layer_result[f'selector_composite_score_{approach}'] = _format_number_7digits(layer_metrics.composite_score)
                else:
                    layer_result[f'selector_composite_score_{approach}'] = 0.0
            else:
                layer_result[f'selector_composite_score_{approach}'] = 0.0

        # Add learned composite score (−predicted_gap) if available
        learned_score_val = learned_scores.get(layer_idx) if learned_scores else None
        layer_result['selector_composite_score_learned'] = _format_number_7digits(learned_score_val if learned_score_val is not None else 0.0)
        
        final_results["results_per_layer"].append(layer_result)

    # Save JSON results
    with open(output_json_path, 'w') as f:
        json.dump(final_results, f, indent=4, sort_keys=True, cls=NumpyEncoder)
    logger.info(f"💾 Multi-composite results saved to {output_json_path}")

    # mCE-tagged companion file for corrupted runs
    try:
        if getattr(args, "eval_corruptions", False):
            output_json_path_mce = os.path.join(output_dir, "multi_composite_analysis_mce.json")
            with open(output_json_path_mce, "w") as f:
                json.dump(final_results, f, indent=4, sort_keys=True, cls=NumpyEncoder)
            logger.info(f"💾 (mCE) multi-composite results saved to {output_json_path_mce}")
    except Exception as e:
        logger.warning(f"Failed to save multi_composite_analysis_mce.json: {e}")

    # Analysis and Visualization
    analyzer = MultiCompositeScoreAnalyzer(final_results)
    
    # Create comparison plot
    analyzer.create_multi_approach_comparison_plot(output_plot_path)
    
    # Generate detailed report
    detailed_report = analyzer.generate_multi_approach_report()
    with open(output_report_path, 'w') as f:
        f.write(detailed_report)
    logger.info(f"📄 Multi-approach report saved to {output_report_path}")
    
    # Print summary to console
    performance = analyzer.analyze_approach_performance()
    
    print("\n" + "="*80)
    print("MULTI-COMPOSITE SCORE EXPERIMENT SUMMARY")
    print("="*80)
    
    if performance:
        # Sort by ECE performance
        sorted_approaches = sorted(performance.items(), key=lambda x: x[1]['predicted_ece'])
        
        print(f"🏆 PERFORMANCE RANKING (by Test ECE):")
        for rank, (approach, perf) in enumerate(sorted_approaches, 1):
            medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
            approach_name = approach.replace('_', ' ').title()
            gap = perf['ece_gap_from_optimal']
            gap_status = "Perfect!" if gap == 0 else f"{gap:+.4f} gap"
            print(f"   {medal} {approach_name}: Layer {perf['predicted_layer']}, ECE {perf['predicted_ece']:.4f} ({gap_status})")
        
        print(f"\n✅ Optimal Layer (Ground Truth): {empirical_best_layer} (ECE: {empirical_best_layer_info['test_ece']:.4f})")
        
        # Best approach
        best_approach, best_perf = sorted_approaches[0]
        best_name = best_approach.replace('_', ' ').title()
        print(f"\n🎯 WINNER: {best_name}")
        print(f"   - Uses {len(get_required_metrics_for_approach(best_approach))} metrics")
        print(f"   - Achieves {best_perf['predicted_ece']:.4f} ECE")
        print(f"   - {best_perf['ece_gap_from_optimal']:+.4f} gap from optimal")
    
    print(f"\n📊 Files saved:")
    print(f"   📄 JSON: {output_json_path}")
    print(f"   📈 Plot: {output_plot_path}")
    print(f"   📋 Report: {output_report_path}")
    print("="*80)

    # Also make a reliability plot that overlays the top-3 layer selections
    try:
        plot_reliability_curves_with_topk_layers(
            model_adapter=model_adapter,
            train_raw=train_raw, train_labels=train_labels,
            val_raw=val_raw,     val_labels=val_labels,
            test_raw=test_raw,   test_labels=test_labels,
            batch_size=args.batch_size,
            device=args.device,
            output_dir=output_dir,
            k=3,
            out_name="reliability_curves_baselines_plus_top3.png"
        )
    except Exception as e:
        logger.warning(f"Failed to generate reliability_curves_baselines_plus_top3.png: {e}")

    # ----- CIFAR-C / mCE evaluation (optional) -----
    if getattr(args, 'eval_corruptions', False):
        logger.info("🌪 Evaluating on CIFAR-C to compute mCE…")

        # Use the same test transform as the clean test loader
        try:
            test_transform = getattr(test_loader.dataset, 'transform', None)
        except Exception:
            test_transform = None

        # 1) Uncalibrated
        def _predict_uncal(Xc):
            P = model_adapter.predict_proba(Xc, batch_size=min(256, args.batch_size))
            P = np.clip(P, 1e-8, 1-1e-8); P /= P.sum(axis=1, keepdims=True)
            return P

        mce_uncal = compute_mce_for_method(
            "uncalibrated", _predict_uncal,
            labels_getter=lambda: None,
            dataset_name=dataset_clean,
            test_transform=test_transform,
            batch_size=args.batch_size,
            cifar_c_dir=args.cifar_c_dir
        )

        # 2) Temperature scaling (fit on clean val)
        ts_obj = None
        try:
            logits_val = model_adapter.predict_logits(val_raw, batch_size=min(256, args.batch_size))
            ts_obj = TemperatureScaling(); ts_obj.fit(logits_val, val_labels)
        except Exception as e:
            logger.warning(f"TS fit (for CIFAR-C) failed, skipping: {e}")

        def _predict_ts(Xc):
            if ts_obj is None:
                return _predict_uncal(Xc)
            logits_c = model_adapter.predict_logits(Xc, batch_size=min(256, args.batch_size))
            return ts_obj.calibrate(logits_c)

        mce_ts = compute_mce_for_method(
            "temperature_scaling", _predict_ts,
            labels_getter=lambda: None,
            dataset_name=dataset_clean,
            test_transform=test_transform,
            batch_size=args.batch_size,
            cifar_c_dir=args.cifar_c_dir
        )

        # ---- Fit once on CLEAN val; reuse for all CIFAR-C splits ----
        iso_obj = None
        platt_obj = None
        beta_obj = None
        try:
            logits_val = model_adapter.predict_logits(val_raw, batch_size=min(256, args.batch_size))

            # 2a) Isotonic (Top-Label)
            try:
                iso_obj = TopLabelIsotonicCalibrator()
                iso_obj.fit(logits_val, val_labels)
            except Exception as e:
                logger.warning(f"Isotonic fit (for CIFAR-C) failed, skipping: {e}")

            # 2b) Platt Scaling (one-vs-rest on logits)
            try:
                platt_obj = PlattScaling()
                platt_obj.fit(logits_val, val_labels)
            except Exception as e:
                logger.warning(f"Platt fit (for CIFAR-C) failed, skipping: {e}")

            # 2c) Beta Calibration (one-vs-rest on logits)
            try:
                beta_obj = BetaCalibration()
                beta_obj.fit(logits_val, val_labels)
            except Exception as e:
                logger.warning(f"Beta fit (for CIFAR-C) failed, skipping: {e}")

        except Exception as e:
            logger.warning(f"Logits for val could not be computed for CIFAR-C baselines; only Uncal/TS will run. {e}")

        # ---- Prediction wrappers for a single CIFAR-C split ----
        def _predict_iso(Xc):
            if iso_obj is None:
                return _predict_uncal(Xc)
            logits_c = model_adapter.predict_logits(Xc, batch_size=min(256, args.batch_size))
            return iso_obj.calibrate(logits_c)

        def _predict_platt(Xc):
            if platt_obj is None:
                return _predict_uncal(Xc)
            logits_c = model_adapter.predict_logits(Xc, batch_size=min(256, args.batch_size))
            return platt_obj.calibrate(logits_c)

        def _predict_beta(Xc):
            if beta_obj is None:
                return _predict_uncal(Xc)
            logits_c = model_adapter.predict_logits(Xc, batch_size=min(256, args.batch_size))
            return beta_obj.calibrate(logits_c)

        mce_iso = compute_mce_for_method(
            "isotonic_toplabel", _predict_iso,
            labels_getter=lambda: None,
            dataset_name=dataset_clean,
            test_transform=test_transform,
            batch_size=args.batch_size,
            cifar_c_dir=args.cifar_c_dir
        )

        mce_platt = compute_mce_for_method(
            "platt_scaling", _predict_platt,
            labels_getter=lambda: None,
            dataset_name=dataset_clean,
            test_transform=test_transform,
            batch_size=args.batch_size,
            cifar_c_dir=args.cifar_c_dir
        )

        mce_beta = compute_mce_for_method(
            "beta_calibration", _predict_beta,
            labels_getter=lambda: None,
            dataset_name=dataset_clean,
            test_transform=test_transform,
            batch_size=args.batch_size,
            cifar_c_dir=args.cifar_c_dir
        )

        # 3) Geometric Physical Space (SPP+JL on raw images)
        logger.info("🧪 Calculating Geometric Physical Space mCE (SPP+JL compression)...")
        geo_physical_for_c = None
        mce_geo_physical = float('nan')
        try:
            FINAL_DIM = 1024
            PYR = [4, 2, 1]

            Xtr_c = compress_with_spp_jl(
                train_raw, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
                seed=42, batch_size=min(512, args.batch_size), device=args.device
            )
            Xva_c = compress_with_spp_jl(
                val_raw, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
                seed=42, batch_size=min(512, args.batch_size), device=args.device
            )

            geo_physical_for_c = GeometricCalibrator(
                model=model_adapter,
                X_train_embed=Xtr_c,
                y_train=train_labels,
                auto_select_layer=False,
                library="fast_separation",
            )
            geo_physical_for_c.fit(X_val_embed=Xva_c, X_val_original=val_raw, 
                                   y_val=val_labels, fit_batch_size=args.batch_size)
            
            def _predict_geo_physical(Xc):
                Xte_c = compress_with_spp_jl(
                    Xc, final_output_dim=FINAL_DIM, pyramid_levels=PYR,
                    seed=42, batch_size=min(512, args.batch_size), device=args.device
                )
                return geo_physical_for_c.calibrate_batched(X_test_embed=Xte_c, 
                                                            X_test_original=Xc, 
                                                            batch_size=args.batch_size)
            
            mce_geo_physical = compute_mce_for_method(
                "geometric_physical_space", _predict_geo_physical,
                labels_getter=lambda: None,
                dataset_name=dataset_clean,
                test_transform=test_transform,
                batch_size=args.batch_size,
                cifar_c_dir=args.cifar_c_dir
            )
            logger.info(f"✅ Geometric Physical Space mCE: {mce_geo_physical:.5f}")
        except Exception as e:
            logger.warning(f"Geometric Physical Space mCE failed: {e}")

        # 4) Our geometric semantic calibrator at the selected layer (use empirical best layer)
        geo_for_c = None
        chosen_layer = None
        try:
            if 'empirical_best_layer' in locals() or 'empirical_best_layer' in globals():
                chosen_layer = empirical_best_layer
            if chosen_layer is not None:
                raw_model = model_adapter.model
                dev = torch.device(args.device)
                tr_loader = DataLoader(TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                                       batch_size=args.batch_size, shuffle=False)
                va_loader = DataLoader(TensorDataset(torch.from_numpy(val_raw),   torch.from_numpy(val_labels)),
                                       batch_size=args.batch_size, shuffle=False)
                Xtr_f, _ = extract_features_directly(raw_model, tr_loader, chosen_layer, dev)
                Xva_f, _ = extract_features_directly(raw_model, va_loader, chosen_layer, dev)

                geo_for_c = GeometricCalibrator(
                    model=model_adapter,
                    X_train_embed=Xtr_f.numpy(),
                    y_train=train_labels,
                    library="fast_separation",
                    auto_select_layer=False,
                    device=args.device,
                )
                geo_for_c.fit(X_val_embed=Xva_f.numpy(), y_val=val_labels,
                              X_val_original=val_raw, fit_batch_size=args.batch_size)
        except Exception as e:
            logger.warning(f"Geometric calibrator setup for CIFAR-C failed: {e}")

        def _predict_geo(Xc):
            if geo_for_c is None:
                return _predict_uncal(Xc)
            raw_model = model_adapter.model
            dev = torch.device(args.device)
            te_loader = DataLoader(TensorDataset(torch.from_numpy(Xc),
                                                 torch.zeros(len(Xc), dtype=torch.long)),
                                   batch_size=args.batch_size, shuffle=False)
            Xte_f, _ = extract_features_directly(raw_model, te_loader, geo_for_c.feature_layer if hasattr(geo_for_c, 'feature_layer') and geo_for_c.feature_layer is not None else chosen_layer, dev)
            return geo_for_c.calibrate_batched(X_test_embed=Xte_f.numpy(),
                                               X_test_original=Xc,
                                               batch_size=args.batch_size)

        mce_geo = compute_mce_for_method(
            "geometric_semantic", _predict_geo,
            labels_getter=lambda: None,
            dataset_name=dataset_clean,
            test_transform=test_transform,
            batch_size=args.batch_size,
            cifar_c_dir=args.cifar_c_dir
        )

        # Stash into final_results (JSON)
        if "baselines" not in final_results: final_results["baselines"] = {}
        for key in ["uncalibrated","temperature_scaling","isotonic_toplabel","platt_scaling","beta_calibration","geometric_physical_space","geometric_semantic"]:
            final_results["baselines"].setdefault(key, {})

        final_results["baselines"]["uncalibrated"]["mce"] = mce_uncal
        final_results["baselines"]["temperature_scaling"]["mce"] = mce_ts
        final_results["baselines"]["isotonic_toplabel"]["mce"] = mce_iso
        final_results["baselines"]["platt_scaling"]["mce"] = mce_platt
        final_results["baselines"]["beta_calibration"]["mce"] = mce_beta
        final_results["baselines"]["geometric_physical_space"]["mce"] = mce_geo_physical
        final_results["baselines"]["geometric_semantic"]["mce"] = mce_geo

        logger.info(
            "mCE summary — Uncal: %.4f | TS: %.4f | Iso: %.4f | Platt: %.4f | Beta: %.4f | GeoPhy: %.4f | GeoSem: %.4f",
            mce_uncal, mce_ts, mce_iso, mce_platt, mce_beta, mce_geo_physical, mce_geo
        )

        # --- Per-layer mCE for geometric semantic calibrator ---
        try:
            geo_mce_per_layer = {}
            raw_model = model_adapter.model
            dev = torch.device(args.device)

            # Ensure candidate_layers_all is available; try loading from clean dump if missing
            if 'candidate_layers_all' not in locals() or not candidate_layers_all:
                clean_out_dir = os.path.join(
                    output_base_dir, args.training_method, dataset_clean,
                    args.model_name, f"seed{args.seed}"
                )
                clean_metrics_path = os.path.join(clean_out_dir, "all_metrics_per_layer.json")
                with open(clean_metrics_path, "r") as f:
                    D = json.load(f)
                candidate_layers_all = D.get("candidate_layers", [])

            tr_loader = DataLoader(TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
                                   batch_size=args.batch_size, shuffle=False)
            va_loader = DataLoader(TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
                                   batch_size=args.batch_size, shuffle=False)

            try:
                test_transform = getattr(test_loader.dataset, 'transform', None)
            except Exception:
                test_transform = None

            for L in candidate_layers_all:
                Xtr_f, _ = extract_features_directly(raw_model, tr_loader, L, dev)
                Xva_f, _ = extract_features_directly(raw_model, va_loader, L, dev)

                geo_L = GeometricCalibrator(
                    model=model_adapter,
                    X_train_embed=Xtr_f.numpy(),
                    y_train=train_labels,
                    library="fast_separation",
                    auto_select_layer=False,
                    device=args.device,
                )
                geo_L.fit(X_val_embed=Xva_f.numpy(), y_val=val_labels,
                          X_val_original=val_raw, fit_batch_size=args.batch_size)

                def _predict_geo_L(Xc):
                    te_loader = DataLoader(TensorDataset(torch.from_numpy(Xc),
                                                         torch.zeros(len(Xc), dtype=torch.long)),
                                           batch_size=args.batch_size, shuffle=False)
                    Xte_f, _ = extract_features_directly(raw_model, te_loader, L, dev)
                    return geo_L.calibrate_batched(X_test_embed=Xte_f.numpy(),
                                                   X_test_original=Xc,
                                                   batch_size=args.batch_size)

                mce_L = compute_mce_for_method(
                    f"geometric_semantic_L{L}", _predict_geo_L,
                    labels_getter=lambda: None,
                    dataset_name=dataset_clean,
                    test_transform=test_transform,
                    batch_size=args.batch_size,
                    cifar_c_dir=args.cifar_c_dir
                )
                geo_mce_per_layer[int(L)] = float(mce_L)

            final_results.setdefault("baselines", {}).setdefault("geometric_semantic", {})
            final_results["baselines"]["geometric_semantic"]["mce_per_layer"] = {str(k): v for k, v in geo_mce_per_layer.items()}
            if geo_mce_per_layer:
                best_L = min(geo_mce_per_layer, key=geo_mce_per_layer.get)
                final_results["baselines"]["geometric_semantic"]["mce"] = geo_mce_per_layer[best_L]
                final_results["baselines"]["geometric_semantic"]["best_layer"] = int(best_L)
        except Exception as e:
            logger.warning(f"Per-layer geometric semantic mCE failed: {e}")

        # Persist CIFAR-C updates (mCEs + per-layer)
        try:
            with open(output_json_path, 'w') as f:
                json.dump(final_results, f, indent=4, sort_keys=True, cls=NumpyEncoder)
            logger.info(f"💾 Updated multi-composite results (with mCE) saved to {output_json_path}")
        except Exception as e:
            logger.warning(f"Failed to save updated results with mCE: {e}")

        # Optional: write a compact summary file for quick inspection
        try:
            mce_dump_path = os.path.join(output_dir, "cifar_c_mce_summary.json")
            mce_summary_data = {
                "uncalibrated": final_results["baselines"]["uncalibrated"].get("mce"),
                "temperature_scaling": final_results["baselines"]["temperature_scaling"].get("mce"),
                "isotonic_toplabel": final_results["baselines"]["isotonic_toplabel"].get("mce"),
                "platt_scaling": final_results["baselines"]["platt_scaling"].get("mce"),
                "beta_calibration": final_results["baselines"]["beta_calibration"].get("mce"),
                "geometric_physical_space": final_results["baselines"]["geometric_physical_space"].get("mce"),
                "geometric_semantic": final_results["baselines"]["geometric_semantic"],
            }
            with open(mce_dump_path, "w") as f:
                json.dump(format_numbers_7digits(mce_summary_data), f, indent=2, cls=NumpyEncoder)
            logger.info(f"💾 CIFAR-C mCE summary saved: {mce_dump_path}")
        except Exception as e:
            logger.warning(f"Failed to save cifar_c_mce_summary.json: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run multi-composite score layer selection comparison.")
    
    # Experiment Configuration
    parser.add_argument('--model-name', type=str, required=True, 
                       help="Model architecture (e.g., resnet18, densenet121)")
    parser.add_argument('--dataset-name', type=str, required=True, 
                       help="Dataset name (e.g., cifar10, cifar100)")
    parser.add_argument('--training-method', type=str, required=True, 
                       help="Training method (e.g., baseline_cross_entropy, augmix)")
    parser.add_argument('--seed', type=int, required=True, 
                       help="Random seed for training and evaluation")
    
    # Paths
    parser.add_argument('--results-base-dir', type=str, required=True, 
                       help="Base directory with pre-trained models")
    parser.add_argument('--output-base-dir', type=str, required=True, 
                       help="Output directory for experiment results")
    parser.add_argument(
        '--target-acc',
        type=float,
        default=None,
        help="Optional target accuracy (percent) to pick accuracy-specific subdirectories (e.g., 85.0 -> acc85).",
    )

    # Technical Settings
    parser.add_argument('--batch-size', type=int, default=256, 
                       help="Batch size for data processing")
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                       help="Device for computations")
    parser.add_argument('--weights-csv', type=str, default=None,
                       help="Path to learned weights CSV for scoring layers")
    parser.add_argument("--eval-baselines", dest="eval_baselines", action="store_true", default=True,
                        help="Evaluate classical baselines (uncalibrated, TS, isotonic top-label). Default: True")
    parser.add_argument("--no-eval-baselines", dest="eval_baselines", action="store_false")
    # CIFAR-C / mCE evaluation flags
    parser.add_argument('--eval-corruptions', action='store_true', default=False,
                       help="If set, also evaluate CIFAR-*-C and compute mCE for selected methods.")
    parser.add_argument('--cifar-c-dir', type=str, default=None,
                       help="Optional override for CIFAR-C root (default: data/cifar10-c or data/cifar100-c).")
    parser.add_argument('--mce-only', action='store_true', default=False,
                       help="Skip clean ECE evaluation; compute mCE-only artifacts (incl. per-layer mCE).")

    # Meta-predictor screening
    parser.add_argument('--screen-layers', action='store_true', default=False,
                       help="Run geometric quality screening before layer selection to identify which layers will work well for calibration")

    args = parser.parse_args()
    # Auto-enable mCE-only for CIFAR-*C unless user overrides
    try:
        if isinstance(args.dataset_name, str) and args.dataset_name.endswith('-c'):
            args.mce_only = True
    except Exception:
        pass
    
    try:
        run_multi_composite_experiment(args)
        logger.info("🎉 Multi-composite experiment completed successfully!")
    except Exception as e:
        logger.error(f"💥 Experiment failed: {e}")
        raise
