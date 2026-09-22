# Experiments/run_post_hoc_calibration.py
"""
Phase 2: Post-Hoc Calibration Experiments
Loads trained models from Phase 1 and applies post-hoc calibration methods
"""

import argparse
import sys
import os
import json
import time
import logging
import copy
import re
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tabulate import tabulate
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import model architectures
from Net.resnet import resnet18, resnet50
from Net.densenet import densenet121
from Net.dinov2 import DINOv2Classifier

# Import data loaders
from data.cifar10 import (
    get_train_valid_loader as cifar10_loaders,
    get_test_loader as cifar10_test,
)
from data.cifar100 import (
    get_train_valid_loader as cifar100_loaders,
    get_test_loader as cifar100_test,
)
from data.svhn import (
    get_train_valid_loader as svhn_loaders,
    get_test_loader as svhn_test,
)
from data.pacs import (
    get_train_valid_loader as pacs_loaders,
    get_test_loader as pacs_test,
)
from data.tiny_imagenet import (
    get_data_loader as tiny_imagenet_get_data_loader,
    TinyImageNet,
)

# Import corruption data loaders for enhanced evaluation
from data.cifar10_c import get_cifar10c_loader, CIFAR10C
from data.cifar100_c import get_cifar100c_loader, CIFAR100C


# Import calibrators
from Calibrators.temperature_scaling import TemperatureScaling
from Calibrators.isotonic_regression import IsotonicRegressionCalibrator
from Calibrators.geometric_calibrator import (
    GeometricCalibrator,
    FullVectorDistanceFusionCalibrator,
)

# Import metrics
from Metrics.metrics import expected_calibration_error

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


class NumpyEncoder(json.JSONEncoder):
    """
     FIXED: Custom JSON encoder that handles NumPy types for calibration results serialization.

    Converts NumPy numeric types to native Python types that can be JSON serialized:
    - np.integer → int
    - np.floating → float
    - np.ndarray → list
    - np.bool_ → bool
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, "item"):  # NumPy scalars
            return obj.item()
        return super().default(obj)


def multiclass_nll(probs: np.ndarray, labels: np.ndarray, eps: float = 1e-12) -> float:
    idx = np.arange(len(labels))
    return float(-np.mean(np.log(np.clip(probs[idx, labels], eps, 1.0))))


def multiclass_brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int) -> float:
    one_hot = np.eye(num_classes, dtype=np.float64)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def classwise_ece(probs: np.ndarray, labels: np.ndarray, num_bins: int = 10) -> float:
    n_samples, num_classes = probs.shape
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    class_eces = []
    for c in range(num_classes):
        class_conf = probs[:, c]
        class_true = (labels == c).astype(np.float64)
        ece_c = 0.0
        for i in range(num_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            if i == num_bins - 1:
                in_bin = (class_conf >= lo) & (class_conf <= hi)
            else:
                in_bin = (class_conf >= lo) & (class_conf < hi)
            if not np.any(in_bin):
                continue
            prop = np.mean(in_bin)
            acc = np.mean(class_true[in_bin])
            conf = np.mean(class_conf[in_bin])
            ece_c += prop * abs(acc - conf)
        class_eces.append(ece_c)
    return float(np.mean(class_eces)) if class_eces else 0.0


class PyTorchModelAdapter:
    """
    Enhanced adapter that handles both model predictions and feature extraction.
    """

    def __init__(
        self, pytorch_model, device, dataset_name="cifar10", feature_extractor=None
    ):
        self.model = pytorch_model
        self.device = device
        self.model.to(self.device)
        self.model.eval()

        # Store dataset info for proper preprocessing
        self.dataset_name = dataset_name

        # Store feature extractor for consistent feature extraction
        self.feature_extractor = feature_extractor

        # Detect if model requires 224x224 input (ViT/DINOv2 models)
        self.requires_224_input = self._detect_224_input_requirement()

        logger.info(f"Enhanced PyTorch adapter initialized on {self.device}")
        if feature_extractor:
            logger.info(
                f" Feature extractor integrated for {feature_extractor.model_name}"
            )
        if self.requires_224_input:
            logger.info(
                f"  Model requires 224x224 input - will resize images automatically"
            )

    def _detect_224_input_requirement(self) -> bool:
        """Detect if the model requires 224x224 input (ViT/DINOv2 models)."""
        # Check if model has backbone with patch_embed (timm-style ViT)
        if hasattr(self.model, "backbone"):
            backbone = self.model.backbone
            if hasattr(backbone, "patch_embed"):
                if hasattr(backbone.patch_embed, "img_size"):
                    img_size = backbone.patch_embed.img_size
                    if isinstance(img_size, (tuple, list)) and len(img_size) >= 1:
                        return img_size[0] == 224
                    elif isinstance(img_size, int):
                        return img_size == 224
                # If patch_embed exists, likely a ViT model that needs 224x224
                return True

        # Check if model has patch_embed directly (some ViT architectures)
        if hasattr(self.model, "patch_embed"):
            if hasattr(self.model.patch_embed, "img_size"):
                img_size = self.model.patch_embed.img_size
                if isinstance(img_size, (tuple, list)) and len(img_size) >= 1:
                    return img_size[0] == 224
                elif isinstance(img_size, int):
                    return img_size == 224
            return True

        # Check model type/name for DINOv2/ViT indicators
        model_str = str(type(self.model)).lower()
        if (
            "dinov2" in model_str
            or "vision_transformer" in model_str
            or "vit" in model_str
        ):
            return True

        return False

    def __call__(self, x):
        """Make the adapter callable like a model"""
        return self.predict_proba(x)

    @torch.no_grad()
    def predict_proba(self, X, batch_size: int = 256):
        """Get probability predictions in memory-safe batches."""
        self.model.eval()

        # Normalize input to torch.float32 tensor on CPU first
        if isinstance(X, np.ndarray):
            X_t = torch.from_numpy(X)
        else:
            X_t = X
        if X_t.dtype != torch.float32:
            X_t = X_t.to(torch.float32)

        # Quick sanity check to catch accidental logits
        if X_t.ndim == 2 and X_t.shape[1] in [10, 100]:
            raise ValueError(
                f"predict_proba received logits-shaped tensor: {tuple(X_t.shape)}"
            )

        # Lightweight dataset/loader
        ds = torch.utils.data.TensorDataset(X_t)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

        probs_chunks = []
        use_amp = self.device == "cuda"
        for (xb,) in dl:
            # Channels-last to channels-first if needed
            if xb.ndim == 4 and xb.shape[1] not in (1, 3) and xb.shape[3] in (1, 3):
                xb = xb.permute(0, 3, 1, 2)

            xb = xb.to(self.device, non_blocking=True)

            # Resize to 224x224 if model requires it (ViT/DINOv2 models)
            if self.requires_224_input and xb.ndim == 4:
                _, _, h, w = xb.shape
                if h != 224 or w != 224:
                    xb = F.interpolate(
                        xb, size=(224, 224), mode="bilinear", align_corners=False
                    )
            with torch.amp.autocast("cuda", enabled=use_amp):
                if hasattr(self.model, "base_model"):
                    logits = self.model.base_model(xb)
                else:
                    logits = self.model(xb)

            #  ENHANCED DIAGNOSTICS for first batch
            if len(probs_chunks) == 0:
                try:
                    min_v = float(logits.min().item())
                    max_v = float(logits.max().item())
                    mean_v = float(logits.mean().item())
                    std_v = float(logits.std().item())
                    logger.info(f" First batch logits stats:")
                    logger.info(f"   Shape: {logits.shape}")
                    logger.info(f"   Range: [{min_v:.4f}, {max_v:.4f}]")
                    logger.info(f"   Mean: {mean_v:.4f}, Std: {std_v:.4f}")
                    logger.info(f"   Model eval mode: {not self.model.training}")

                    # Check for uniform logits (potential bug indicator)
                    logits_np = logits[0].detach().to("cpu").numpy()
                    logger.info(f"   First sample logits: {logits_np}")

                    # Check if all logits in first sample are nearly identical (uniform distribution indicator)
                    if std_v < 1e-4:
                        logger.error(
                            " CRITICAL: Logits have near-zero variance! This will produce uniform probabilities."
                        )
                        logger.error(
                            f"   This suggests the model is not making predictions (weights might not be loaded)."
                        )

                    # Check if logits are all zeros or very small
                    if abs(mean_v) < 1e-6 and std_v < 1e-6:
                        logger.error(
                            " CRITICAL: Logits are all near-zero! Model appears uninitialized."
                        )

                except Exception as log_err:
                    logger.warning(f"Failed to log logits diagnostics: {log_err}")

            pb = F.softmax(logits, dim=1).float().cpu()
            probs_chunks.append(pb)

        return torch.cat(probs_chunks, dim=0).numpy()

    @torch.no_grad()
    def predict_logits(self, X, batch_size: int = 256):
        """Return raw model logits (pre-softmax) in batches. Mirrors predict_proba."""
        self.model.eval()

        # Normalize input to torch.float32 tensor on CPU first
        if isinstance(X, np.ndarray):
            X_t = torch.from_numpy(X)
        else:
            X_t = X
        if X_t.dtype != torch.float32:
            X_t = X_t.to(torch.float32)

        # Lightweight dataset/loader
        ds = torch.utils.data.TensorDataset(X_t)
        dl = torch.utils.data.DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True
        )

        logits_chunks = []
        use_amp = self.device == "cuda"
        for (xb,) in dl:
            # Channels-last to channels-first if needed
            if xb.ndim == 4 and xb.shape[1] not in (1, 3) and xb.shape[3] in (1, 3):
                xb = xb.permute(0, 3, 1, 2)

            xb = xb.to(self.device, non_blocking=True)

            # Resize to 224x224 if model requires it (ViT/DINOv2 models)
            if self.requires_224_input and xb.ndim == 4:
                _, _, h, w = xb.shape
                if h != 224 or w != 224:
                    xb = F.interpolate(
                        xb, size=(224, 224), mode="bilinear", align_corners=False
                    )

            with torch.amp.autocast("cuda", enabled=use_amp):
                if hasattr(self.model, "base_model"):
                    out = self.model.base_model(xb)
                else:
                    out = self.model(xb)
            # HF-style compatibility
            lb = out.logits if hasattr(out, "logits") else out
            logits_chunks.append(lb.float().cpu())

        # Concatenate and run sanity checks/logging
        logits_tensor = torch.cat(logits_chunks, dim=0)
        try:
            min_v = float(logits_tensor.min())
            max_v = float(logits_tensor.max())
            mean_v = float(logits_tensor.mean())
            std_v = float(logits_tensor.std())
            logger.info(
                f"Logits shape: {tuple(logits_tensor.shape)} | range: [{min_v:.4f}, {max_v:.4f}] | mean: {mean_v:.4f} | std: {std_v:.4f}"
            )
        except Exception:
            pass

        # Detect probability-like outputs and convert to log-probabilities as a safe fallback
        try:
            looks_prob = bool((logits_tensor >= 0).all().item())
            if looks_prob:
                row_sums = logits_tensor.sum(dim=1)
                if torch.allclose(
                    row_sums, torch.ones_like(row_sums), atol=1e-3, rtol=1e-3
                ):
                    logger.warning(
                        "Values look like probabilities, not logits. Converting to log-probabilities for downstream calibration."
                    )
                    logits_tensor = torch.log(torch.clamp(logits_tensor, 1e-8, 1.0))
        except Exception:
            pass

        return logits_tensor.numpy()

    def predict(self, X):
        """Get class predictions."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def extract_features(self, X):
        """
        Extract features using the integrated feature extractor.

        Args:
            X: Input data (numpy array or torch tensor)

        Returns:
            np.ndarray: Extracted features
        """
        if self.feature_extractor is None:
            logger.warning(
                "No feature extractor found - using model logits as features"
            )
            # Fallback: use logits as features
            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X_tensor = torch.FloatTensor(X).to(self.device)
                else:
                    X_tensor = X.to(self.device)

                logits = self.model(X_tensor)
                return logits.cpu().numpy()

        self.model.eval()

        with torch.no_grad():
            # Handle different input types
            if isinstance(X, np.ndarray):
                X_tensor = torch.FloatTensor(X).to(self.device)
            else:
                X_tensor = X.to(self.device)

            # Ensure correct shape
            if len(X_tensor.shape) == 4 and X_tensor.shape[1] not in [1, 3]:
                if X_tensor.shape[3] in [1, 3]:
                    X_tensor = X_tensor.permute(0, 3, 1, 2)

            # Resize to 224x224 if model requires it (ViT/DINOv2 models)
            if self.requires_224_input and X_tensor.ndim == 4:
                _, _, h, w = X_tensor.shape
                if h != 224 or w != 224:
                    X_tensor = F.interpolate(
                        X_tensor, size=(224, 224), mode="bilinear", align_corners=False
                    )

            # Forward pass to trigger feature extraction hook
            _ = self.model(X_tensor)

            # Get features from the hook
            if self.feature_extractor.features is not None:
                features = self.feature_extractor.features
                # Flatten spatial dimensions but keep feature dimension
                if len(features.shape) == 4:  # [B, C, H, W]
                    features = features.mean(dim=[2, 3])  # Global average pooling

                return features.cpu().numpy()
            else:
                logger.error("Feature extraction failed - no features captured")
                return None


class FeatureExtractor:
    """Extract features from specific model layers"""

    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name.lower()
        self.features = None
        self.hook = None
        self.selected_layer_name: str = "unknown"
        self._register_hook()

    def _register_hook(self):
        """Register hook with research-backed optimal extraction points"""

        #  Research-based optimal extraction points
        extraction_config = {
            # ResNet family - based on Kornblith et al. 2019 transfer learning study
            "resnet18": {
                "layer": "layer4",
                "expected_dim": 512,
                "reason": "Final conv layer, optimal for smaller ResNets",
            },
            "resnet50": {
                "layer": "layer3",
                "expected_dim": 1024,
                "reason": "Avoids over-fitting of layer4, better transferability",
            },
            "resnet101": {
                "layer": "layer3",
                "expected_dim": 1024,
                "reason": "Consistent with ResNet50, avoids depth over-specialization",
            },
            # DenseNet family - based on feature reuse analysis
            "densenet121": {
                "layer": "denseblock3",  # Before final transition
                "expected_dim": "variable",
                "reason": "Balance between discrimination and generalization",
            },
            "densenet169": {
                "layer": "denseblock3",
                "expected_dim": "variable",
                "reason": "Consistent with DenseNet121 strategy",
            },
        }

        if self.model_name in extraction_config:
            config = extraction_config[self.model_name]
            target_layer = self._get_target_layer(config["layer"])
            self.selected_layer_name = config["layer"]
            logger.info(f" Research-optimized extraction: {config['layer']}")
            logger.info(f"   Expected dim: {config['expected_dim']}")
            logger.info(f"   Rationale: {config['reason']}")
        elif "wide" in self.model_name and "resnet" in self.model_name:
            config = {
                "layer": "block3",
                "expected_dim": 640,
                "reason": "Final conv block for Wide-ResNet",
            }
            target_layer = self._get_target_layer(config["layer"])
            self.selected_layer_name = config["layer"]
            logger.info(f" Wide-ResNet extraction: {config['layer']}")
            logger.info(f"   Expected dim: {config['expected_dim']}")
            logger.info(f"   Rationale: {config['reason']}")
        else:
            # Fallback to current logic with warning
            target_layer = self._get_fallback_layer()
            logger.warning(f" Using fallback extraction for {self.model_name}")

        # Register the hook
        if target_layer is not None:
            self.hook = target_layer.register_forward_hook(self._feature_hook)
            logger.info(f" Feature extraction hook registered successfully")

            #  Add feature dimension validation
            self._validate_feature_dimensions(target_layer)
        else:
            raise ValueError("Failed to identify target layer for feature extraction")

    def _get_target_layer(self, layer_name: str):
        """Get the target layer based on architecture"""

        if "resnet" in self.model_name:
            if layer_name == "layer3":
                return self.model.layer3
            elif layer_name == "layer4":
                return self.model.layer4
            else:
                raise ValueError(f"Unknown ResNet layer: {layer_name}")

        elif "wide" in self.model_name and "resnet" in self.model_name:
            if hasattr(self.model, "block3"):
                return self.model.block3
            elif hasattr(self.model, "layer3"):
                return self.model.layer3
            else:
                raise ValueError(
                    "Wide-ResNet target layer not found (expected block3/layer3)"
                )

        elif "densenet" in self.model_name:
            if layer_name == "denseblock3":
                # Try to find denseblock3 in features module
                if hasattr(self.model, "features"):
                    # Standard torchvision DenseNet
                    for name, module in self.model.features.named_children():
                        if "denseblock3" in name:
                            return module
                    # Fallback: use transition2 (after denseblock3)
                    for name, module in self.model.features.named_children():
                        if "transition2" in name:
                            logger.info(" Using transition2 as denseblock3 fallback")
                            return module

                # Custom DenseNet implementation
                elif hasattr(self.model, "dense3"):
                    return self.model.dense3

                # Final fallback
                logger.warning(" Could not find denseblock3, using features module")
                return self.model.features

        return None

    def _get_fallback_layer(self):
        """Fallback layer selection with current logic"""
        modules = list(self.model.named_modules())

        if "resnet" in self.model_name:
            return self.model.layer4

        elif "densenet" in self.model_name:
            if hasattr(self.model, "features"):
                return self.model.features
            elif hasattr(self.model, "dense4"):
                return self.model.dense4
            elif hasattr(self.model, "bn"):
                return self.model.bn

        return None

    def _validate_feature_dimensions(self, target_layer):
        """Validate that extracted features have reasonable dimensions"""

        # Expected dimension ranges for different architectures
        dimension_ranges = {
            "resnet18": (400, 600),  # ~512
            "resnet50": (800, 1200),  # ~1024 for layer3
            "densenet121": (500, 1500),  # Variable due to growth rate
        }

        if self.model_name in dimension_ranges:
            min_dim, max_dim = dimension_ranges[self.model_name]
            logger.info(f" Expected feature dimensions: {min_dim}-{max_dim}")

            # This validation will run during first forward pass
            def validation_hook(module, input, output):
                if len(output.shape) == 4:
                    feat_dim = output.shape[1]  # Channel dimension
                else:
                    feat_dim = output.shape[-1]

                if feat_dim < min_dim or feat_dim > max_dim:
                    logger.warning(
                        f" Feature dimension {feat_dim} outside expected range {min_dim}-{max_dim}"
                    )
                else:
                    logger.info(
                        f" Feature dimension {feat_dim} within expected range"
                    )

                # Remove this validation hook after first use
                validation_hook.remove()

            validation_hook = target_layer.register_forward_hook(validation_hook)

    def _feature_hook(self, module, input, output):
        """Hook function to capture features"""
        self.features = output.detach()

    def extract_features(self, data_loader, device):
        """Extract features for entire dataset"""
        self.model.eval()
        all_features = []
        all_logits = []
        all_labels = []

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(data_loader):
                data, targets = data.to(device), targets.to(device)

                # Forward pass (this triggers the hook)
                logits = self.model(data)

                # Get features (captured by hook)
                if self.features is not None:
                    # Flatten spatial dimensions but keep feature dimension
                    if len(self.features.shape) == 4:  # [B, C, H, W]
                        features = self.features.mean(
                            dim=[2, 3]
                        )  # Global average pooling
                    else:
                        features = self.features

                    all_features.append(features.cpu().numpy())
                    all_logits.append(logits.cpu().numpy())
                    all_labels.append(targets.cpu().numpy())

                if batch_idx % 50 == 0:
                    logger.info(f"   Processed batch {batch_idx}/{len(data_loader)}")

        features = np.concatenate(all_features, axis=0)
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)

        logger.info(
            f" Extracted features: {features.shape}, logits: {logits.shape}, labels: {labels.shape}"
        )
        return features, logits, labels

    def cleanup(self):
        """Remove the hook"""
        if self.hook:
            self.hook.remove()


def set_seed(seed=42):
    """Set all random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_trained_model(
    model_path: str, model_name: str, num_classes: int, device, dataset: str = "cifar10"
):
    """Load a trained model from checkpoint, handling model wrappers and architecture mismatches."""

    #  FIXED: Use the correct model architecture based on the dataset
    if dataset == "pacs":
        # For PACS, we need ImageNet-style models
        import torchvision.models as models

        if model_name == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "resnet50":
            model = models.resnet50(weights=None)
            model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        elif model_name == "densenet121":
            model = models.densenet121(weights=None)
            model.classifier = torch.nn.Linear(
                model.classifier.in_features, num_classes
            )
        else:
            raise ValueError(f"Unknown model for PACS: {model_name}")
    elif dataset == "tiny_imagenet":
        # For Tiny ImageNet, use Tiny ImageNet-specific ResNet models
        if model_name == "resnet18":
            from Net.resnet_tiny_imagenet import resnet18 as tiny_resnet18

            model = tiny_resnet18(num_classes=num_classes, temp=1.0)
        elif model_name == "resnet34":
            from Net.resnet_tiny_imagenet import resnet34 as tiny_resnet34

            model = tiny_resnet34(num_classes=num_classes, temp=1.0)
        elif model_name == "resnet50":
            from Net.resnet_tiny_imagenet import resnet50 as tiny_resnet50

            model = tiny_resnet50(num_classes=num_classes, temp=1.0)
        elif model_name == "resnet101":
            from Net.resnet_tiny_imagenet import resnet101 as tiny_resnet101

            model = tiny_resnet101(num_classes=num_classes, temp=1.0)
        elif model_name == "resnet152":
            from Net.resnet_tiny_imagenet import resnet152 as tiny_resnet152

            model = tiny_resnet152(num_classes=num_classes, temp=1.0)
        else:
            raise ValueError(f"Unknown model for Tiny ImageNet: {model_name}")
    else:
        # Original logic for CIFAR/SVHN
        if model_name == "resnet18":
            from Net.resnet import resnet18

            model = resnet18(num_classes=num_classes)
        elif model_name == "resnet50":
            from Net.resnet import resnet50

            model = resnet50(num_classes=num_classes)
        elif model_name == "resnet101":
            from Net.resnet_cifar import resnet101

            model = resnet101(num_classes=num_classes)
        elif model_name == "resnet152":
            from Net.resnet_cifar import resnet152

            model = resnet152(num_classes=num_classes)
        elif model_name == "densenet121":
            from Net.densenet import densenet121

            model = densenet121(num_classes=num_classes)
        elif any(
            token in model_name.lower()
            for token in ("wide-resnet", "wide_resnet", "wideresnet")
        ):
            from Net.wide_resnet import Wide_ResNet_Cifar, BasicBlock

            lower_name = model_name.lower().replace("_", "-")
            base_name = lower_name.replace("wide-resnet", "").replace("wideresnet", "")
            base_name = base_name.lstrip("-")
            parts = [part for part in base_name.split("-") if part]

            depth = int(parts[0]) if parts else 28
            widen_factor = int(parts[1]) if len(parts) > 1 else 10

            if (depth - 4) % 6 == 0:
                n = (depth - 4) // 6
            elif (depth - 2) % 6 == 0:
                n = (depth - 2) // 6
            else:
                raise ValueError(
                    f"Unsupported Wide-ResNet depth: {depth}. Expected 6n+4 or 6n+2."
                )

            layers = [int(n), int(n), int(n)]

            #  FIX: Extract temperature from checkpoint if available to match training config
            temp_param = 1.0  # Default temperature
            if os.path.exists(model_path):
                checkpoint_state = torch.load(model_path, map_location="cpu")
                if isinstance(checkpoint_state, dict):
                    # Try to extract temperature from model state
                    actual_state = checkpoint_state.get("model_state", checkpoint_state)
                    # Check if 'temp' parameter exists in the checkpoint
                    if "temp" in actual_state:
                        temp_param = float(actual_state["temp"])
                        logger.info(
                            f" Extracted temperature from checkpoint: {temp_param}"
                        )

            model = Wide_ResNet_Cifar(
                BasicBlock,
                layers,
                widen_factor,
                num_classes=num_classes,
                temp=temp_param,
            )
            logger.info(
                f" Wide-ResNet instantiated: depth={depth}, width={widen_factor}, temp={temp_param}"
            )
            logger.info(f"   Model type: {type(model)}")
            logger.info(f"   Has base_model attr: {hasattr(model, 'base_model')}")
            logger.info(f"   Model has fc layer: {hasattr(model, 'fc')}")
            if hasattr(model, "fc"):
                logger.info(
                    f"   FC layer shape: in={model.fc.in_features}, out={model.fc.out_features}"
                )
        elif "dinov2" in model_name.lower():
            # Support DINOv2 variants (small/base/large/giant, optional _scratch)
            lower_name = model_name.lower()
            if "giant" in lower_name:
                size = "giant"
            elif "large" in lower_name:
                size = "large"
            elif "base" in lower_name:
                size = "base"
            elif "small" in lower_name:
                size = "small"
            else:
                raise ValueError(f"Unknown DINOv2 size in model name: {model_name}")

            # Check if checkpoint is timm-style by peeking at keys
            checkpoint_state = torch.load(model_path, map_location="cpu")
            if isinstance(checkpoint_state, dict) and "model_state" in checkpoint_state:
                checkpoint_keys = list(checkpoint_state["model_state"].keys())
            else:
                checkpoint_keys = list(checkpoint_state.keys())

            is_timm_style = any(
                k.startswith("backbone.blocks.") or k == "backbone.cls_token"
                for k in checkpoint_keys
            )

            if is_timm_style:
                logger.info(
                    " Detected timm-style DINOv2 checkpoint, using timm model"
                )
                import timm

                # Map size to timm model name
                timm_model_map = {
                    "small": "vit_small_patch14_dinov2.lvd142m",
                    "base": "vit_base_patch14_dinov2.lvd142m",
                    "large": "vit_large_patch14_dinov2.lvd142m",
                    "giant": "vit_giant_patch14_dinov2.lvd142m",
                }
                timm_name = timm_model_map[size]

                # Create timm model with classifier head
                backbone = timm.create_model(
                    timm_name, pretrained=False, num_classes=0, img_size=224
                )

                # Get hidden dimension
                hidden_dim = backbone.embed_dim

                # Build wrapper matching checkpoint structure
                class TimmDINOv2Classifier(nn.Module):
                    def __init__(self, backbone, num_classes, hidden_dim):
                        super().__init__()
                        self.backbone = backbone
                        self.head = nn.Linear(hidden_dim, num_classes)

                    def forward(self, x):
                        # DINOv2 expects 224x224 input - resize if needed (CIFAR images are 32x32)
                        if x.ndim == 4 and x.size(-1) != 224:
                            x = F.interpolate(
                                x, size=(224, 224), mode="bilinear", align_corners=False
                            )
                        features = self.backbone(x)
                        return self.head(features)

                model = TimmDINOv2Classifier(backbone, num_classes, hidden_dim)
                logger.info(
                    f" Created timm DINOv2 model: {timm_name} with {num_classes} classes"
                )
                # Store flag to indicate timm model (for Strategy 6)
                model._is_timm_model = True
            else:
                # Use HuggingFace for HuggingFace-style checkpoints
                logger.info(
                    " Detected HuggingFace-style DINOv2 checkpoint, using HuggingFace model"
                )
                use_pretrained = "scratch" not in lower_name
                freeze_backbone = True if use_pretrained else False

                model = DINOv2Classifier(
                    num_classes=num_classes,
                    model_size=size,
                    freeze_backbone=freeze_backbone,
                    use_pretrained=use_pretrained,
                )
                # Store flag to indicate HuggingFace model (for Strategy 6)
                model._is_timm_model = False
        else:
            raise ValueError(f"Unknown model: {model_name}")

    # Load weights
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    state_dict = torch.load(model_path, map_location=device)

    # Handle common wrapping formats (e.g., {"model_state": {...}, "epoch": ...})
    if isinstance(state_dict, dict) and "model_state" in state_dict:
        logger.info(
            " Detected wrapped checkpoint with 'model_state'; extracting weights"
        )
        state_dict = state_dict["model_state"]

    model_keys = list(model.state_dict().keys())
    checkpoint_keys = list(state_dict.keys())

    logger.info(
        f" Checkpoint has {len(checkpoint_keys)} keys, model expects {len(model_keys)} keys"
    )
    logger.info(f"   Sample checkpoint keys: {checkpoint_keys[:3]}")
    logger.info(f"   Sample model keys: {model_keys[:3]}")

    original_state = copy.deepcopy(model.state_dict())

    def reset_model():
        model.load_state_dict(original_state, strict=True)

    def try_load_with_remapping(candidate_state, strategy_name):
        """Attempt loading with a specific key remapping strategy."""
        try:
            missing, unexpected = model.load_state_dict(candidate_state, strict=False)
            loaded = len(candidate_state) - len(unexpected)
            logger.info(
                f"Strategy '{strategy_name}': loaded {loaded}/{len(candidate_state)} keys"
            )
            logger.info(f"   Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if missing:
                logger.info(f"   Sample missing keys: {missing[:5]}")
            if unexpected:
                logger.info(f"   Sample unexpected keys: {unexpected[:5]}")
            return loaded, missing, unexpected
        except Exception as e:
            logger.error(f"Strategy '{strategy_name}' failed: {e}")
            return 0, [], []
        finally:
            reset_model()

    def remap_keys(src_dict, transform_fn):
        return src_dict.__class__((transform_fn(k), v) for k, v in src_dict.items())

    best_state_dict = state_dict
    best_strategy = "direct"
    best_loaded, best_missing, best_unexpected = try_load_with_remapping(
        best_state_dict, best_strategy
    )

    # Strategy 2: Remove 'module.' prefix
    if any(k.startswith("module.") for k in state_dict.keys()):
        cleaned = remap_keys(
            state_dict,
            lambda k: k.replace("module.", "", 1) if k.startswith("module.") else k,
        )
        loaded, missing, unexpected = try_load_with_remapping(cleaned, "remove_module")
        if loaded > best_loaded:
            best_loaded, best_missing, best_unexpected = loaded, missing, unexpected
            best_state_dict = cleaned
            best_strategy = "remove_module"

    # Strategy 3: Remove 'base_model.' prefix
    if any(k.startswith("base_model.") for k in state_dict.keys()):
        cleaned = remap_keys(
            state_dict,
            lambda k: (
                k.replace("base_model.", "", 1) if k.startswith("base_model.") else k
            ),
        )
        loaded, missing, unexpected = try_load_with_remapping(
            cleaned, "remove_base_model"
        )
        if loaded > best_loaded:
            best_loaded, best_missing, best_unexpected = loaded, missing, unexpected
            best_state_dict = cleaned
            best_strategy = "remove_base_model"

    # Strategy 4: downsample -> shortcut (ResNet only, applied after best prefix handling)
    if "resnet" in model_name.lower() and any(
        "downsample" in k for k in best_state_dict.keys()
    ):
        cleaned = remap_keys(
            best_state_dict, lambda k: k.replace("downsample", "shortcut")
        )
        loaded, missing, unexpected = try_load_with_remapping(
            cleaned, "downsample_to_shortcut"
        )
        if loaded > best_loaded:
            best_loaded, best_missing, best_unexpected = loaded, missing, unexpected
            best_state_dict = cleaned
            best_strategy = "downsample_to_shortcut"

    # Strategy 5: Handle Wide-ResNet specific key mappings (shortcut -> downsample)
    if "wide" in model_name.lower() and "resnet" in model_name.lower():
        if any("shortcut" in k for k in best_state_dict.keys()):
            cleaned = remap_keys(
                best_state_dict, lambda k: k.replace("shortcut", "downsample")
            )
            loaded, missing, unexpected = try_load_with_remapping(
                cleaned, "shortcut_to_downsample"
            )
            if loaded > best_loaded:
                best_loaded, best_missing, best_unexpected = loaded, missing, unexpected
                best_state_dict = cleaned
                best_strategy = "shortcut_to_downsample"
            # Reset for next strategy
            reset_model()

    # Strategy 6: DINOv2 handling (timm direct load or timm -> HuggingFace remapping)
    if "dinov2" in model_name.lower():
        # Check if model is timm or HuggingFace
        is_timm_model = getattr(model, "_is_timm_model", False)

        if is_timm_model:
            # Strategy 6a: Direct timm loading (for timm models)
            # Check if checkpoint keys match timm model structure directly
            if any(
                k.startswith("backbone.blocks.") or k == "backbone.cls_token"
                for k in best_state_dict.keys()
            ):
                # Try direct loading first (keys should match)
                loaded, missing, unexpected = try_load_with_remapping(
                    best_state_dict, "dinov2_timm_direct"
                )
                if loaded > best_loaded:
                    best_loaded, best_missing, best_unexpected = (
                        loaded,
                        missing,
                        unexpected,
                    )
                    best_strategy = "dinov2_timm_direct"
                reset_model()
        else:
            # Strategy 6b: timm -> HuggingFace key remapping (for HuggingFace models)
            def remap_dinov2_keys(key):
                # Embeddings
                if key == "backbone.cls_token":
                    return "backbone.embeddings.cls_token"
                if key == "backbone.pos_embed":
                    return "backbone.embeddings.position_embeddings"
                if key.startswith("backbone.patch_embed.proj."):
                    suffix = key.split("backbone.patch_embed.proj.")[-1]
                    return f"backbone.embeddings.patch_embeddings.projection.{suffix}"

                # Final norm
                if key.startswith("backbone.norm."):
                    suffix = key.split("backbone.norm.")[-1]
                    return f"backbone.layernorm.{suffix}"

                # Classifier head
                if key.startswith("head."):
                    suffix = key.split("head.")[-1]
                    return f"classifier.{suffix}"

                # Transformer blocks
                block_match = re.match(r"backbone\.blocks\.(\d+)\.(.+)", key)
                if block_match:
                    block_num = block_match.group(1)
                    rest = block_match.group(2)

                    # Layer scale
                    if rest == "ls1.gamma":
                        return (
                            f"backbone.encoder.layer.{block_num}.layer_scale1.lambda1"
                        )
                    if rest == "ls2.gamma":
                        return (
                            f"backbone.encoder.layer.{block_num}.layer_scale2.lambda1"
                        )

                    # Attention
                    if rest == "attn.proj.weight":
                        return f"backbone.encoder.layer.{block_num}.attention.output.dense.weight"
                    if rest == "attn.proj.bias":
                        return f"backbone.encoder.layer.{block_num}.attention.output.dense.bias"

                    # QKV needs special handling - skip for now, will be handled separately
                    if rest.startswith("attn.qkv"):
                        return None  # Signal to handle separately

                    # MLP
                    if rest.startswith("mlp.fc1."):
                        suffix = rest.split("mlp.fc1.")[-1]
                        return f"backbone.encoder.layer.{block_num}.mlp.fc1.{suffix}"
                    if rest.startswith("mlp.fc2."):
                        suffix = rest.split("mlp.fc2.")[-1]
                        return f"backbone.encoder.layer.{block_num}.mlp.fc2.{suffix}"

                    # Norms
                    if rest.startswith("norm1."):
                        suffix = rest.split("norm1.")[-1]
                        return f"backbone.encoder.layer.{block_num}.norm1.{suffix}"
                    if rest.startswith("norm2."):
                        suffix = rest.split("norm2.")[-1]
                        return f"backbone.encoder.layer.{block_num}.norm2.{suffix}"

                return key  # Return unchanged if no match

            # Check if this looks like a timm-style checkpoint
            if any(
                k.startswith("backbone.blocks.") or k == "backbone.cls_token"
                for k in best_state_dict.keys()
            ):
                remapped = {}
                qkv_weights = {}  # Collect QKV weights for splitting

                for k, v in best_state_dict.items():
                    new_key = remap_dinov2_keys(k)
                    if new_key is None:
                        # Handle QKV splitting
                        block_match = re.match(
                            r"backbone\.blocks\.(\d+)\.attn\.qkv\.(.+)", k
                        )
                        if block_match:
                            block_num = block_match.group(1)
                            suffix = block_match.group(2)  # weight or bias
                            qkv_weights[(block_num, suffix)] = v
                    elif new_key != k:
                        remapped[new_key] = v
                    else:
                        remapped[k] = v

                # Split QKV into separate Q, K, V
                for (block_num, suffix), qkv in qkv_weights.items():
                    hidden_size = qkv.shape[0] // 3
                    q, k, v = qkv.chunk(3, dim=0)
                    remapped[
                        f"backbone.encoder.layer.{block_num}.attention.attention.query.{suffix}"
                    ] = q
                    remapped[
                        f"backbone.encoder.layer.{block_num}.attention.attention.key.{suffix}"
                    ] = k
                    remapped[
                        f"backbone.encoder.layer.{block_num}.attention.attention.value.{suffix}"
                    ] = v

                loaded, missing, unexpected = try_load_with_remapping(
                    remapped, "dinov2_timm_to_hf"
                )
                if loaded > best_loaded:
                    best_loaded, best_missing, best_unexpected = (
                        loaded,
                        missing,
                        unexpected,
                    )
                    best_state_dict = remapped
                    best_strategy = "dinov2_timm_to_hf"
                reset_model()

    logger.info(f" Best strategy: '{best_strategy}' with {best_loaded} keys loaded")
    final_missing, final_unexpected = model.load_state_dict(
        best_state_dict, strict=False
    )

    matched_keys = len(model_keys) - len(final_missing)
    if matched_keys < max(1, int(len(model_keys) * 0.5)):
        raise RuntimeError(
            f" Failed to load model weights properly. Only {matched_keys}/{len(model_keys)} model keys matched."
        )

    if final_missing:
        logger.info(
            f"   Remaining missing keys after best strategy: {final_missing[:5]}"
        )
    if final_unexpected:
        logger.info(
            f"   Remaining unexpected keys after best strategy: {final_unexpected[:5]}"
        )

    #  ENHANCED: Verify critical weights were actually loaded (especially for Wide-ResNet)
    if "wide" in model_name.lower() and "resnet" in model_name.lower():
        logger.info(" Verifying Wide-ResNet weights loaded properly...")

        # Check FC layer weights
        if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
            fc_weight = model.fc.weight.data
            fc_bias = model.fc.bias.data if model.fc.bias is not None else None

            fc_weight_mean = float(fc_weight.mean().item())
            fc_weight_std = float(fc_weight.std().item())
            fc_weight_abs_max = float(fc_weight.abs().max().item())

            logger.info(
                f"   FC weight stats: mean={fc_weight_mean:.6f}, std={fc_weight_std:.6f}, abs_max={fc_weight_abs_max:.6f}"
            )

            if fc_bias is not None:
                fc_bias_mean = float(fc_bias.mean().item())
                fc_bias_std = float(fc_bias.std().item())
                logger.info(
                    f"   FC bias stats: mean={fc_bias_mean:.6f}, std={fc_bias_std:.6f}"
                )

            # Check if weights look uninitialized (all zeros or very small values)
            if fc_weight_abs_max < 1e-6:
                logger.error(
                    " FC layer weights appear to be zeros! Model may not be properly loaded."
                )
            elif fc_weight_std < 1e-6:
                logger.warning(
                    " FC layer weights have zero variance - this is suspicious!"
                )
            else:
                logger.info("    FC layer weights look reasonable")

        # Check a sample conv layer weight
        if hasattr(model, "conv1"):
            conv1_weight = model.conv1.weight.data
            conv1_mean = float(conv1_weight.mean().item())
            conv1_std = float(conv1_weight.std().item())
            logger.info(
                f"   Conv1 weight stats: mean={conv1_mean:.6f}, std={conv1_std:.6f}"
            )

    model = model.to(device)
    logger.info(
        f" Model loaded successfully with {matched_keys}/{len(model_keys)} model keys"
    )
    return model


def get_data_loaders(
    dataset: str,
    batch_size: int = 128,
    seed: int = 42,
    target_domain: str = "sketch",
    corruption_type: str = None,
    corruption_severity: int = None,
    image_size: int = None,
    preprocessing_protocol: str = None,
):
    """
    Get data loaders with same split as Phase 1.

    preprocessing_protocol: None keeps this LEGACY paper-reproduction path
    bit-for-bit (CIFAR-100 clean test uses ImageNet statistics although the
    checkpoints were trained with CIFAR statistics -- see
    utils/preprocessing_protocol.py, docs/normalization_audit.md). Opt in to
    the corrected behaviour with --preprocessing_protocol corrected_v2_train_norm.

    For corruption experiments:
    - Train/Val loaders remain CLEAN (for calibration fitting)
    - Test loader uses corrupted data (for evaluation)

    Args:
        image_size: If provided, resize images to this size (for models like DINOv2 that require larger inputs)
    """

    # Helper function to wrap transforms with resize if needed
    def wrap_transform_with_resize(base_transform, image_size):
        """Wrap a transform with resize if image_size is specified and different from default"""
        if image_size is None or image_size == 32:
            return base_transform

        # Extract the transform list from Compose
        if isinstance(base_transform, transforms.Compose):
            transform_list = list(base_transform.transforms)
        else:
            transform_list = [base_transform] if base_transform is not None else []

        # Insert resize at the beginning (before ToTensor)
        # Find ToTensor index
        totensor_idx = -1
        for i, t in enumerate(transform_list):
            if isinstance(t, transforms.ToTensor):
                totensor_idx = i
                break

        if totensor_idx >= 0:
            transform_list.insert(
                totensor_idx, transforms.Resize((image_size, image_size))
            )
        else:
            # If no ToTensor found, prepend resize
            transform_list.insert(0, transforms.Resize((image_size, image_size)))

        return transforms.Compose(transform_list) if transform_list else None

    if dataset == "cifar10":
        # Get base transforms from the loader function
        # We'll need to modify the dataset transforms after creation
        train_loader, val_loader = cifar10_loaders(
            batch_size=batch_size, augment=False, random_seed=seed, valid_size=0.1
        )

        # Apply resize to train/val datasets if needed
        if image_size is not None and image_size != 32:
            # Modify dataset transforms
            for dataset in [train_loader.dataset, val_loader.dataset]:
                if hasattr(dataset, "transform") and dataset.transform is not None:
                    dataset.transform = wrap_transform_with_resize(
                        dataset.transform, image_size
                    )

        if corruption_type is not None and corruption_severity is not None:
            from data.cifar10_c import get_cifar10c_loader

            # Create custom transform with resize for corruption loader
            if image_size is not None and image_size != 32:
                normalize = transforms.Normalize(
                    mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]
                )
                corruption_transform = transforms.Compose(
                    [
                        transforms.Resize((image_size, image_size)),
                        transforms.ToTensor(),
                        normalize,
                    ]
                )
            else:
                corruption_transform = None

            test_loader = get_cifar10c_loader(
                root="./data/cifar10-c",
                corruption_type=corruption_type,
                severity=corruption_severity,
                batch_size=batch_size,
            )
            # Apply resize to corruption dataset if needed
            if (
                image_size is not None
                and image_size != 32
                and hasattr(test_loader.dataset, "transform")
            ):
                test_loader.dataset.transform = wrap_transform_with_resize(
                    test_loader.dataset.transform or transforms.ToTensor(), image_size
                )
        else:
            test_loader = cifar10_test(batch_size=batch_size, shuffle=False)
            # Apply resize to test dataset if needed
            if (
                image_size is not None
                and image_size != 32
                and hasattr(test_loader.dataset, "transform")
            ):
                test_loader.dataset.transform = wrap_transform_with_resize(
                    test_loader.dataset.transform, image_size
                )

        num_classes = 10

    elif dataset == "cifar100":
        train_loader, val_loader = cifar100_loaders(
            batch_size=batch_size, augment=False, random_seed=seed, valid_size=0.1
        )

        # Apply resize to train/val datasets if needed
        if image_size is not None and image_size != 32:
            for dataset in [train_loader.dataset, val_loader.dataset]:
                if hasattr(dataset, "transform") and dataset.transform is not None:
                    dataset.transform = wrap_transform_with_resize(
                        dataset.transform, image_size
                    )

        if corruption_type is not None and corruption_severity is not None:
            from data.cifar100_c import get_cifar100c_loader

            test_loader = get_cifar100c_loader(
                root="./data/cifar100-c",
                corruption_type=corruption_type,
                severity=corruption_severity,
                batch_size=batch_size,
            )
            # Apply resize to corruption dataset if needed
            if (
                image_size is not None
                and image_size != 32
                and hasattr(test_loader.dataset, "transform")
            ):
                test_loader.dataset.transform = wrap_transform_with_resize(
                    test_loader.dataset.transform or transforms.ToTensor(), image_size
                )
        else:
            test_loader = cifar100_test(
                batch_size=batch_size,
                shuffle=False,
                preprocessing_protocol=preprocessing_protocol,
            )
            # Apply resize to test dataset if needed
            if (
                image_size is not None
                and image_size != 32
                and hasattr(test_loader.dataset, "transform")
            ):
                test_loader.dataset.transform = wrap_transform_with_resize(
                    test_loader.dataset.transform, image_size
                )

        num_classes = 100

    elif dataset == "svhn":
        if corruption_type is not None:
            raise ValueError("SVHN does not have a corruption benchmark")
        train_loader, val_loader = svhn_loaders(
            batch_size=batch_size, augment=False, random_seed=seed, valid_size=0.1
        )
        test_loader = svhn_test(batch_size=batch_size, shuffle=False)
        num_classes = 10

    elif dataset == "tiny_imagenet":
        if corruption_type is not None:
            raise ValueError("Tiny ImageNet does not have a corruption benchmark")

        from data.tiny_imagenet import get_train_valid_loader as tiny_imagenet_loaders
        from data.tiny_imagenet import get_test_loader as tiny_imagenet_test

        train_loader, val_loader = tiny_imagenet_loaders(
            batch_size=batch_size, augment=False, random_seed=seed, valid_size=0.1
        )
        test_loader = tiny_imagenet_test(batch_size=batch_size, shuffle=False)
        num_classes = 200

    elif dataset == "pacs":
        if corruption_type is not None:
            raise ValueError("PACS does not have a corruption benchmark")
        from data.pacs_domain_adaptation import get_domain_adaptation_loaders

        train_loader, val_loader, test_loader = get_domain_adaptation_loaders(
            target_domain=target_domain,
            batch_size=batch_size,
            random_seed=seed,
            valid_size=0.1,  # This 10% will be used for calibration fitting
        )
        num_classes = 7
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return train_loader, val_loader, test_loader, num_classes


def construct_model_path(
    base_dir: str,
    method: str,
    dataset: str,
    model: str,
    seed: int,
    target_domain: str = "sketch",
) -> str:
    """Construct the path to a trained model"""
    import glob

    # Determine experiment type based on method
    if method.startswith("baseline_"):
        exp_type = "baseline"
    elif method in ["augmix", "augmix_constellation"]:
        exp_type = "baseline"  # Based on your directory structure
    else:
        exp_type = "geometric"

    # Construct path
    exp_name = f"{method}_{dataset}_{model}_seed{seed}"
    # Add target domain for PACS
    if dataset == "pacs" and target_domain:
        exp_name += f"_target_{target_domain}"

    # Special handling for DINOv2 models
    is_dinov2 = "dino" in model.lower()
    is_baseline_method = method.startswith("baseline_") or method in [
        "augmix",
        "augmix_constellation",
    ]

    if is_dinov2 and not is_baseline_method:
        # DINOv2 models are saved in calibration_comparison directory with wildcard suffix
        model_name_pattern = f"{method}_{dataset}_{model}_seed{seed}*"
        calibration_pattern = os.path.join(
            base_dir, "calibration_comparison", model_name_pattern, "best_model.pth"
        )
        matching_files = glob.glob(calibration_pattern)

        if matching_files:
            # Use the first matching file
            model_path = matching_files[0]
            logger.info(f" Found DINOv2 model in calibration_comparison: {model_path}")
            return model_path
        else:
            logger.warning(
                f" DINOv2 model not found with pattern: {calibration_pattern}, trying standard path"
            )

    # Standard path structure (for non-DINOv2 or baseline DINOv2 models)
    base_path = os.path.join(base_dir, exp_type, method, dataset, model, f"seed{seed}")
    flat_checkpoint = os.path.join(base_path, "best_model.pth")
    nested_checkpoint = os.path.join(base_path, exp_name, "best_model.pth")

    if os.path.exists(flat_checkpoint):
        logger.info(f" Using flat checkpoint path: {flat_checkpoint}")
        return flat_checkpoint

    logger.info(
        f" Flat checkpoint missing; falling back to nested path: {nested_checkpoint}"
    )
    return nested_checkpoint


def extract_raw_data_and_features(feature_extractor, data_loader, device):
    """
     FIXED: Extract both raw data AND features with robust error handling.
    """
    logger.info(f" Extracting raw data and features...")

    # Initialize GPU memory management
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import os

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
            "max_split_size_mb:128,expandable_segments:False"
        )

    # Storage for different data types
    all_raw_data = []  # Original images for model predictions
    all_features = []  # Features for geometric calculations
    all_logits = []  # Model outputs
    all_labels = []  # Ground truth labels

    feature_extractor.model.eval()

    # Track failed batches for debugging
    failed_batches = 0
    successful_batches = 0

    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(data_loader):
            try:
                data, targets = data.to(device), targets.to(device)

                # Validate input data
                if data.numel() == 0 or targets.numel() == 0:
                    logger.warning(f" Batch {batch_idx}: Empty input data, skipping")
                    failed_batches += 1
                    continue

                # Check for NaN or infinite values in input
                if torch.isnan(data).any() or torch.isinf(data).any():
                    logger.warning(
                        f" Batch {batch_idx}: Invalid input data (NaN/Inf), skipping"
                    )
                    failed_batches += 1
                    continue

                # Clear previous features to ensure clean state
                if hasattr(feature_extractor, "features"):
                    feature_extractor.features = None
                if hasattr(feature_extractor, "layer_features"):
                    feature_extractor.layer_features.clear()

                # Forward pass to get logits and trigger feature extraction
                logits = feature_extractor.model(data)

                # Validate logits output
                if logits is None or logits.numel() == 0:
                    logger.warning(
                        f" Batch {batch_idx}: Empty logits output, skipping"
                    )
                    failed_batches += 1
                    continue

                # Check for NaN or infinite values in logits
                if torch.isnan(logits).any() or torch.isinf(logits).any():
                    logger.warning(
                        f" Batch {batch_idx}: Invalid logits (NaN/Inf), skipping"
                    )
                    failed_batches += 1
                    continue

                # Get features (captured by hook in feature_extractor)
                features = None
                if feature_extractor.features is not None:
                    features = feature_extractor.features
                    # Flatten spatial dimensions but keep feature dimension
                    if len(features.shape) == 4:  # [B, C, H, W]
                        features = features.mean(dim=[2, 3])  # Global average pooling
                elif (
                    hasattr(feature_extractor, "layer_features")
                    and feature_extractor.layer_features
                ):
                    # Try to get features from layer_features dict
                    if (
                        feature_extractor.forced_layer
                        and feature_extractor.forced_layer
                        in feature_extractor.layer_features
                    ):
                        features = feature_extractor.layer_features[
                            feature_extractor.forced_layer
                        ]
                    elif len(feature_extractor.layer_features) == 1:
                        # If only one layer, use it
                        features = list(feature_extractor.layer_features.values())[0]

                # Validate extracted features
                if features is None:
                    logger.warning(
                        f" Batch {batch_idx}: No features extracted, skipping"
                    )
                    failed_batches += 1
                    continue

                if features.numel() == 0:
                    logger.warning(
                        f" Batch {batch_idx}: Empty features extracted, skipping"
                    )
                    failed_batches += 1
                    continue

                # Check for NaN or infinite values in features
                if torch.isnan(features).any() or torch.isinf(features).any():
                    logger.warning(
                        f" Batch {batch_idx}: Invalid features (NaN/Inf), skipping"
                    )
                    failed_batches += 1
                    continue

                # Successfully extracted - add to lists
                all_raw_data.append(data.cpu())
                all_features.append(features.cpu())
                all_logits.append(logits.cpu())
                all_labels.append(targets.cpu())

                successful_batches += 1

                #  DIAGNOSTIC: Log feature extraction verification for first batch only
                if batch_idx == 0:
                    logger.info("=" * 60)
                    logger.info(" FEATURE EXTRACTION VERIFICATION (First Batch)")
                    logger.info("=" * 60)
                    logger.info(f"Model type: {type(feature_extractor.model)}")
                    logger.info(f"Raw data shape: {data.shape}")
                    logger.info(
                        f"Raw data stats: mean={data.mean().item():.6f}, std={data.std().item():.6f}"
                    )
                    logger.info(f"Features shape: {features.shape}")
                    logger.info(
                        f"Features stats: mean={features.mean().item():.6f}, std={features.std().item():.6f}"
                    )
                    logger.info(f"Logits shape: {logits.shape}")
                    logger.info(
                        f"Logits stats: mean={logits.mean().item():.6f}, std={logits.std().item():.6f}"
                    )

                    # Check if features look like logits (this would be a bug)
                    if features.shape[1] == 10 or features.shape[1] == 100:
                        logger.warning(
                            " WARNING: Features have 10/100 dims - might be logits instead of features!"
                        )

                    # Check if features are diverse (not uniform)
                    feature_variance = torch.var(features, dim=0).mean().item()
                    if feature_variance < 0.001:
                        logger.warning(
                            " WARNING: Features have very low variance - extraction might have failed!"
                        )

                    logger.info("=" * 60)

                if batch_idx % 50 == 0:
                    logger.info(f"   Processed batch {batch_idx}/{len(data_loader)}")

                # Clear GPU cache periodically
                if batch_idx % 10 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

            except Exception as e:
                logger.warning(f" Batch {batch_idx} failed with error: {e}")
                failed_batches += 1
                continue

    #  CRITICAL FIX: Check if we have any successful extractions
    if not all_features or len(all_features) == 0:
        error_msg = f"No features were extracted successfully. Failed batches: {failed_batches}, Successful: {successful_batches}"
        logger.error(f" {error_msg}")
        raise RuntimeError(error_msg)

    logger.info(
        f" Extraction summary: {successful_batches} successful, {failed_batches} failed batches"
    )

    #  ENHANCED: Validate lists before concatenation
    def validate_tensor_list(tensor_list, name):
        """Validate tensor list before concatenation."""
        if not tensor_list:
            raise RuntimeError(f"Empty {name} list - no tensors to concatenate")

        # Check for None or invalid tensors
        valid_tensors = []
        for i, tensor in enumerate(tensor_list):
            if tensor is None:
                logger.warning(f" None tensor in {name} at index {i}, skipping")
                continue
            if tensor.numel() == 0:
                logger.warning(f" Empty tensor in {name} at index {i}, skipping")
                continue
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                logger.warning(f" Invalid tensor in {name} at index {i}, skipping")
                continue
            valid_tensors.append(tensor)

        if not valid_tensors:
            raise RuntimeError(f"No valid tensors in {name} list after filtering")

        return valid_tensors

    # Validate and filter tensor lists
    try:
        all_raw_data = validate_tensor_list(all_raw_data, "raw_data")
        all_features = validate_tensor_list(all_features, "features")
        all_logits = validate_tensor_list(all_logits, "logits")
        all_labels = validate_tensor_list(all_labels, "labels")

        # Ensure all lists have the same length
        min_length = min(
            len(all_raw_data), len(all_features), len(all_logits), len(all_labels)
        )
        if min_length < len(all_raw_data):
            logger.warning(
                f" Truncating tensor lists to consistent length: {min_length}"
            )
            all_raw_data = all_raw_data[:min_length]
            all_features = all_features[:min_length]
            all_logits = all_logits[:min_length]
            all_labels = all_labels[:min_length]

    except RuntimeError as e:
        logger.error(f" Tensor validation failed: {e}")
        raise

    # Concatenate all batches with error handling
    try:
        logger.info(" Concatenating extracted tensors...")
        raw_data = torch.cat(all_raw_data, dim=0).numpy()
        features = torch.cat(all_features, dim=0).numpy()
        logits = torch.cat(all_logits, dim=0).numpy()
        labels = torch.cat(all_labels, dim=0).numpy()

        logger.info(f" Extracted:")
        logger.info(f"   Raw data: {raw_data.shape} (for model predictions)")
        logger.info(f"   Features: {features.shape} (for geometric calculations)")
        logger.info(f"   Logits: {logits.shape}")
        logger.info(f"   Labels: {labels.shape}")

        # Final validation
        if raw_data.shape[0] == 0 or features.shape[0] == 0:
            raise RuntimeError("Final concatenated arrays are empty")

        return raw_data, features, logits, labels

    except Exception as e:
        logger.error(f" Error concatenating results: {e}")
        # Print debug info
        logger.error(
            f"Tensor list lengths: raw_data={len(all_raw_data)}, features={len(all_features)}, logits={len(all_logits)}, labels={len(all_labels)}"
        )
        if all_features:
            logger.error(
                f"Feature shapes: {[f.shape for f in all_features[:5]]}"
            )  # First 5 shapes
        raise


def evaluate_on_corrupted_test(
    calibrators: dict,
    dataset_name: str,
    model_adapter,
    feature_extractor,
    device,
    cifar10c_dir: str = None,
    cifar100c_dir: str = None,
):
    """
     ENHANCED: Evaluate all calibrators on corrupted test data with robust error handling.
    """
    logger.info(f" Evaluating corruption robustness on {dataset_name.upper()}-C...")

    # Define corruption types
    corruption_types = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]

    # Select appropriate data directory and loader function
    if dataset_name == "cifar10":
        if cifar10c_dir is None:
            logger.warning(
                " CIFAR-10-C directory not provided, skipping corruption evaluation"
            )
            return {}
        corruption_data_dir = cifar10c_dir
        loader_fn = get_cifar10c_loader
    elif dataset_name == "cifar100":
        if cifar100c_dir is None:
            logger.warning(
                " CIFAR-100-C directory not provided, skipping corruption evaluation"
            )
            return {}
        corruption_data_dir = cifar100c_dir
        loader_fn = get_cifar100c_loader
    else:
        logger.warning(f" Corruption evaluation not supported for {dataset_name}")
        return {}

    # Check if corruption data directory exists
    if not os.path.exists(corruption_data_dir):
        logger.warning(f" Corruption data directory not found: {corruption_data_dir}")
        return {}

    all_corruption_results = {}

    for cal_name, calibrator in calibrators.items():
        logger.info(f"    Evaluating {cal_name} on corruptions...")
        corruption_accuracies = []
        corruption_eces = []

        # Track detailed results for debugging
        corruption_details = {}

        # Evaluate on all 15 corruptions × 5 severities = 75 combinations
        for corruption in corruption_types:
            corruption_details[corruption] = {}

            for severity in range(1, 6):
                try:
                    # Load corrupted test data
                    loader = loader_fn(
                        root=corruption_data_dir,
                        corruption_type=corruption,
                        severity=severity,
                        batch_size=128,
                        num_workers=0,
                    )

                    #  ENHANCED: Extract features with robust error handling
                    try:
                        test_raw, test_features, test_logits, test_labels = (
                            extract_raw_data_and_features(
                                feature_extractor, loader, device
                            )
                        )

                        # Validate extracted data
                        if test_raw.shape[0] == 0 or test_features.shape[0] == 0:
                            raise RuntimeError("Empty data extracted")

                    except Exception as extract_error:
                        logger.error(
                            f"       Failed {corruption} severity {severity}: {extract_error}"
                        )

                        # Store error information
                        corruption_details[corruption][f"severity_{severity}"] = {
                            "accuracy": 0.0,
                            "ece": 1.0,
                            "num_samples": 0,
                            "error": str(extract_error),
                        }

                        corruption_accuracies.append(0.0)
                        corruption_eces.append(1.0)
                        continue

                    # Apply calibrator with enhanced error handling
                    try:
                        if cal_name == "Uncalibrated":
                            cal_probs = F.softmax(
                                torch.tensor(test_logits), dim=1
                            ).numpy()
                        elif cal_name.startswith("Geometric"):
                            # For geometric methods, decide input type
                            if "Semantic" in cal_name:
                                cal_probs = calibrator.calibrate(
                                    test_features, X_test_original=test_raw
                                )
                            else:
                                cal_probs = calibrator.calibrate(
                                    test_raw, X_test_original=test_raw
                                )
                        else:
                            # Standard calibrators use logits
                            cal_probs = calibrator.calibrate(test_logits)

                        # Validate calibration output
                        if cal_probs is None or cal_probs.shape[0] == 0:
                            raise RuntimeError("Empty calibration output")

                        if np.isnan(cal_probs).any() or np.isinf(cal_probs).any():
                            raise RuntimeError("Invalid calibration output (NaN/Inf)")

                        # Calculate accuracy and ECE
                        cal_preds = np.argmax(cal_probs, axis=1)
                        cal_confs = np.max(cal_probs, axis=1)

                        accuracy = 100.0 * np.mean(cal_preds == test_labels)
                        ece = expected_calibration_error(
                            cal_confs, cal_preds, test_labels
                        )

                        # Store detailed results
                        corruption_details[corruption][f"severity_{severity}"] = {
                            "accuracy": float(accuracy),
                            "ece": float(ece),
                            "num_samples": len(test_labels),
                        }

                        corruption_accuracies.append(accuracy)
                        corruption_eces.append(ece)

                        logger.info(
                            f"       {corruption} (severity {severity}): Accuracy={accuracy:.2f}%, ECE={ece:.4f}"
                        )

                    except Exception as calib_error:
                        logger.error(
                            f"       Calibration failed for {corruption} severity {severity}: {calib_error}"
                        )

                        # Store error information
                        corruption_details[corruption][f"severity_{severity}"] = {
                            "accuracy": 0.0,
                            "ece": 1.0,
                            "num_samples": (
                                len(test_labels) if "test_labels" in locals() else 0
                            ),
                            "error": str(calib_error),
                        }

                        corruption_accuracies.append(0.0)
                        corruption_eces.append(1.0)

                except Exception as e:
                    logger.error(
                        f"       Failed {corruption} severity {severity}: {e}"
                    )

                    # Store error information
                    corruption_details[corruption][f"severity_{severity}"] = {
                        "accuracy": 0.0,
                        "ece": 1.0,
                        "num_samples": 0,
                        "error": str(e),
                    }

                    corruption_accuracies.append(0.0)
                    corruption_eces.append(1.0)

        # Calculate mean corruption performance
        mean_corruption_acc = (
            np.mean(corruption_accuracies) if corruption_accuracies else 0.0
        )
        mean_corruption_ece = np.mean(corruption_eces) if corruption_eces else 1.0

        # Count successful vs failed evaluations
        successful_evaluations = len([acc for acc in corruption_accuracies if acc > 0])
        failed_evaluations = len([acc for acc in corruption_accuracies if acc == 0])

        all_corruption_results[cal_name] = {
            "mean_corruption_accuracy": mean_corruption_acc,
            "mean_corruption_ece": mean_corruption_ece,
            "individual_accuracies": corruption_accuracies,
            "individual_eces": corruption_eces,
            "num_evaluations": len(corruption_accuracies),
            "successful_evaluations": successful_evaluations,
            "failed_evaluations": failed_evaluations,
            "corruption_details": corruption_details,
        }

        logger.info(f"      Mean corruption accuracy: {mean_corruption_acc:.2f}%")
        logger.info(f"      Mean corruption ECE: {mean_corruption_ece:.4f}")
        logger.info(
            f"      Evaluations: {successful_evaluations} successful, {failed_evaluations} failed"
        )

    return all_corruption_results


def calculate_mce_for_all_calibrators(corruption_results: dict):
    """
    Calculate mean Corruption Error (mCE) using uncalibrated as reference

    Args:
        corruption_results: Dictionary with corruption results for each calibrator

    Returns:
        Dictionary with mCE values for each calibrator
    """
    if "Uncalibrated" not in corruption_results:
        logger.warning(" Cannot calculate mCE without uncalibrated baseline")
        return {cal_name: None for cal_name in corruption_results.keys()}

    uncalibrated_acc = corruption_results["Uncalibrated"]["mean_corruption_accuracy"]
    uncalibrated_error = 100.0 - uncalibrated_acc

    mce_results = {}
    for cal_name, results in corruption_results.items():
        cal_acc = results["mean_corruption_accuracy"]
        cal_error = 100.0 - cal_acc

        # mCE = calibrated_error / uncalibrated_error
        if uncalibrated_error > 0:
            mce = cal_error / uncalibrated_error
        else:
            mce = float("inf") if cal_error > 0 else 1.0

        mce_results[cal_name] = mce

    return mce_results


def combine_clean_and_corruption_results(
    clean_results: dict, corruption_results: dict, mce_results: dict
):
    """
    Combine clean test results with corruption robustness results

    Args:
        clean_results: Results on clean test data
        corruption_results: Results on corrupted test data
        mce_results: Mean corruption error values

    Returns:
        Combined results dictionary
    """
    enhanced_results = {}

    for cal_name in clean_results.keys():
        enhanced_results[cal_name] = {
            # Existing clean metrics
            "accuracy": clean_results[cal_name]["accuracy"],
            "ece": clean_results[cal_name]["ece"],
            "num_samples": clean_results[cal_name]["num_samples"],
            "calibration_time": clean_results[cal_name].get("calibration_time", 0.0),
            # NEW: Corruption metrics
            "corruption_accuracy": corruption_results.get(cal_name, {}).get(
                "mean_corruption_accuracy", 0.0
            ),
            "corruption_ece": corruption_results.get(cal_name, {}).get(
                "mean_corruption_ece", 1.0
            ),
            "mce": mce_results.get(cal_name, None),
            "corruption_error_rate": 100.0
            - corruption_results.get(cal_name, {}).get("mean_corruption_accuracy", 0.0),
        }

    return enhanced_results


def evaluate_calibration_methods_with_corruption(
    model,
    feature_extractor,
    train_loader,
    val_loader,
    test_loader,
    device,
    num_classes,
    method_name,
    dataset_name,
    cifar10c_dir=None,
    cifar100c_dir=None,
):
    """Enhanced evaluation with corruption robustness and proper data handling"""

    logger.info(
        f" Starting ENHANCED calibration + corruption evaluation for {method_name}..."
    )

    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info(" Cleared GPU cache")

    # 1. EXISTING: Extract data for all splits
    train_raw, train_features, train_logits, train_labels = (
        extract_raw_data_and_features(feature_extractor, train_loader, device)
    )
    val_raw, val_features, val_logits, val_labels = extract_raw_data_and_features(
        feature_extractor, val_loader, device
    )
    test_raw, test_features, test_logits, test_labels = extract_raw_data_and_features(
        feature_extractor, test_loader, device
    )

    # Create model adapter with proper data handling
    model_adapter = PyTorchModelAdapter(model, device, dataset_name, feature_extractor)

    # 2. EXISTING: Initialize and fit calibrators
    calibrators = {
        "Uncalibrated": None,
        "Temperature": TemperatureScaling(),
        "Isotonic": IsotonicRegressionCalibrator(),
        "Geometric_Semantic_Fast_Separation": GeometricCalibrator(
            model=model_adapter,
            X_train_embed=train_features,
            y_train=train_labels,
            X_train_original=train_raw,
            metric="l2",
            library="fast_separation",
            use_binning=True,
            n_bins=200,
        ),
        "Geometric_Physical_Fast_Separation": GeometricCalibrator(
            model=model_adapter,
            X_train_embed=train_raw,
            y_train=train_labels,
            X_train_original=train_raw,
            metric="l2",
            library="fast_separation",
            use_binning=True,
            n_bins=200,
        ),
        "Geometric_Semantic_Fast_Separation_AugMix": GeometricCalibrator(
            model=model_adapter,
            X_train_embed=train_features,
            y_train=train_labels,
            X_train_original=train_raw,
            metric="l2",
            library="fast_separation",
            use_binning=True,
            n_bins=200,
            use_augmix=True,
            augmix_versions=3,
            augmix_severity=3,
            augmix_width=3,
        ),
        "Geometric_Physical_Fast_Separation_AugMix": GeometricCalibrator(
            model=model_adapter,
            X_train_embed=train_raw,
            y_train=train_labels,
            X_train_original=train_raw,
            metric="l2",
            library="fast_separation",
            use_binning=True,
            n_bins=200,
            use_augmix=True,
            augmix_versions=3,
            augmix_severity=3,
            augmix_width=3,
        ),
    }

    # Add FAISS-based calibrators only if not CIFAR-100 (memory constraints)
    if dataset_name != "cifar100":
        calibrators.update(
            {
                "Geometric_Semantic_FAISS": GeometricCalibrator(
                    model=model_adapter,
                    X_train_embed=train_features,
                    y_train=train_labels,
                    X_train_original=train_raw,
                    metric="l2",
                    library="faiss",
                    use_binning=True,
                    n_bins=200,
                ),
                "Geometric_Physical_FAISS": GeometricCalibrator(
                    model=model_adapter,
                    X_train_embed=train_raw,
                    y_train=train_labels,
                    X_train_original=train_raw,
                    metric="l2",
                    library="faiss",
                    use_binning=True,
                    n_bins=200,
                ),
                "Geometric_Semantic_FAISS_AugMix": GeometricCalibrator(
                    model=model_adapter,
                    X_train_embed=train_features,
                    y_train=train_labels,
                    X_train_original=train_raw,
                    metric="l2",
                    library="faiss",
                    use_binning=True,
                    n_bins=200,
                    use_augmix=True,
                    augmix_versions=3,
                    augmix_severity=3,
                    augmix_width=3,
                ),
                "Geometric_Physical_FAISS_AugMix": GeometricCalibrator(
                    model=model_adapter,
                    X_train_embed=train_raw,
                    y_train=train_labels,
                    X_train_original=train_raw,
                    metric="l2",
                    library="faiss",
                    use_binning=True,
                    n_bins=200,
                    use_augmix=True,
                    augmix_versions=3,
                    augmix_severity=3,
                    augmix_width=3,
                ),
            }
        )

    # 3. EXISTING: Evaluate on clean test data
    clean_results = {}

    for cal_name, calibrator in calibrators.items():
        logger.info(f" Evaluating {cal_name} on CLEAN test data...")

        try:
            if cal_name == "Uncalibrated":
                start_time = time.time()
                test_probs = F.softmax(torch.tensor(test_logits), dim=1).numpy()
                calibration_time = time.time() - start_time

            elif cal_name.startswith("Geometric"):
                # Fit calibrator
                if "Semantic" in cal_name:
                    calibrator.fit(val_features, val_labels, X_val_original=val_raw)
                else:
                    calibrator.fit(val_raw, val_labels, X_val_original=val_raw)

                # Time only the calibration step
                start_time = time.time()
                if "Semantic" in cal_name:
                    test_probs = calibrator.calibrate(
                        test_features, X_test_original=test_raw
                    )
                else:
                    test_probs = calibrator.calibrate(
                        test_raw, X_test_original=test_raw
                    )
                calibration_time = time.time() - start_time

            else:
                # Standard calibrators
                calibrator.fit(val_logits, val_labels)
                start_time = time.time()
                test_probs = calibrator.calibrate(test_logits)
                calibration_time = time.time() - start_time

            # Evaluate performance
            test_preds = np.argmax(test_probs, axis=1)
            test_confs = np.max(test_probs, axis=1)

            accuracy = 100.0 * np.mean(test_preds == test_labels)
            ece = expected_calibration_error(test_confs, test_preds, test_labels)

            clean_results[cal_name] = {
                "accuracy": accuracy,
                "ece": ece,
                "num_samples": len(test_labels),
                "calibration_time": calibration_time,
            }

            logger.info(
                f"   {cal_name}: Accuracy={accuracy:.2f}%, ECE={ece:.4f}, Time={calibration_time:.2f}s"
            )

        except Exception as e:
            logger.error(f"    {cal_name} failed: {e}")
            clean_results[cal_name] = {
                "accuracy": 0.0,
                "ece": 1.0,
                "num_samples": len(test_labels),
                "error": str(e),
                "calibration_time": 0.0,
            }

    # 4. NEW: Evaluate on corrupted test data
    logger.info(f" Starting corruption robustness evaluation...")
    corruption_results = evaluate_on_corrupted_test(
        calibrators,
        dataset_name,
        model_adapter,
        feature_extractor,
        device,
        cifar10c_dir=cifar10c_dir,
        cifar100c_dir=cifar100c_dir,
    )

    # 5. NEW: Calculate mCE (mean Corruption Error)
    mce_results = calculate_mce_for_all_calibrators(corruption_results)

    # 6. NEW: Combine clean and corruption results
    enhanced_results = combine_clean_and_corruption_results(
        clean_results, corruption_results, mce_results
    )

    return enhanced_results


def run_calibration_experiment_fixed(
    model_adapter,
    train_features,
    train_raw,
    train_labels,
    val_features,
    val_raw,
    val_labels,
    test_features,
    test_raw,
    test_labels,
    test_logits,
    dataset_name,
):
    """
    Fixed calibration experiment that properly handles data types
    """

    results = {}

    # Define calibration methods with proper configuration
    calibration_configs = {
        "Uncalibrated": {"type": "baseline"},
        "Temperature": {"type": "baseline"},
        "Isotonic": {"type": "baseline"},
        "Geometric_Semantic_Fast_Separation": {
            "type": "geometric",
            "use_semantic": True,
            "use_faiss": False,
            "use_augmix": False,
        },
        "Geometric_Physical_Fast_Separation": {
            "type": "geometric",
            "use_semantic": False,
            "use_faiss": False,
            "use_augmix": False,
        },
        "Geometric_Semantic_FAISS": {
            "type": "geometric",
            "use_semantic": True,
            "use_faiss": True,
            "use_augmix": False,
        },
        "Geometric_Physical_FAISS": {
            "type": "geometric",
            "use_semantic": False,
            "use_faiss": True,
            "use_augmix": False,
        },
        "Geometric_Semantic_Fast_Separation_AugMix": {
            "type": "geometric",
            "use_semantic": True,
            "use_faiss": False,
            "use_augmix": True,
        },
        "Geometric_Physical_Fast_Separation_AugMix": {
            "type": "geometric",
            "use_semantic": False,
            "use_faiss": False,
            "use_augmix": True,
        },
        "Geometric_Semantic_FAISS_AugMix": {
            "type": "geometric",
            "use_semantic": True,
            "use_faiss": True,
            "use_augmix": True,
        },
        "Geometric_Physical_FAISS_AugMix": {
            "type": "geometric",
            "use_semantic": False,
            "use_faiss": True,
            "use_augmix": True,
        },
    }

    for cal_name, config in calibration_configs.items():
        logger.info(f" Evaluating {cal_name} calibration...")

        try:
            start_time = time.time()

            if config["type"] == "baseline":
                if cal_name == "Uncalibrated":
                    test_probs = F.softmax(torch.tensor(test_logits), dim=1).numpy()

                elif cal_name == "Temperature":
                    calibrator = TemperatureScaling()
                    calibrator.fit(val_logits, val_labels)
                    test_probs = calibrator.calibrate(test_logits)

                elif cal_name == "Isotonic":
                    calibrator = IsotonicRegressionCalibrator()
                    calibrator.fit(val_logits, val_labels)
                    test_probs = calibrator.calibrate(test_logits)

            elif config["type"] == "geometric":
                #  KEY FIX: Proper data selection for geometric calibration
                use_semantic = config["use_semantic"]
                use_faiss = config["use_faiss"]
                use_augmix = config.get("use_augmix", False)

                # Select embedding data based on semantic vs physical
                if use_semantic:
                    # SEMANTIC: Use features for geometric calculations
                    train_embed_data = train_features
                    val_embed_data = val_features
                    test_embed_data = test_features
                    logger.info(" Using features for geometric calculations")
                else:
                    # PHYSICAL: Use raw images for geometric calculations
                    train_embed_data = train_raw
                    val_embed_data = val_raw
                    test_embed_data = test_raw
                    logger.info(" Using raw images for geometric calculations")

                # Force fast_separation for CIFAR-100 to avoid memory issues
                if dataset_name == "cifar100" and use_faiss:
                    logger.warning(" Forcing fast_separation for CIFAR-100")
                    use_faiss = False

                # Initialize calibrator
                calibrator = GeometricCalibrator(
                    model=model_adapter,
                    X_train_embed=train_embed_data,  #  Features OR raw for geometric calc
                    y_train=train_labels,
                    X_train_original=train_raw,  #  ALWAYS raw images for model predictions
                    metric="l2",
                    library="faiss" if use_faiss else "fast_separation",
                    use_binning=True,
                    n_bins=200,
                    use_augmix=use_augmix,
                    augmix_versions=3 if use_augmix else 0,
                    augmix_severity=3 if use_augmix else 3,
                    augmix_width=3 if use_augmix else 3,
                )

                #  CRITICAL FIX: Always pass X_val_original parameter
                logger.info(f" Fitting calibrator...")
                logger.info(f"   Embedding data shape: {val_embed_data.shape}")
                logger.info(f"   Original data shape: {val_raw.shape}")

                calibrator.fit(
                    val_embed_data,  # Features OR raw for geometric calculations
                    val_labels,
                    X_val_original=val_raw,  #  ALWAYS raw images for model predictions
                )

                #  CRITICAL FIX: Always pass X_test_original parameter
                logger.info(f" Calibrating test data...")
                test_probs = calibrator.calibrate(
                    test_embed_data,  # Features OR raw for geometric calculations
                    X_test_original=test_raw,  #  ALWAYS raw images for model predictions
                )

            calibration_time = time.time() - start_time

            # Evaluate performance
            test_preds = np.argmax(test_probs, axis=1)
            test_confs = np.max(test_probs, axis=1)

            accuracy = 100.0 * np.mean(test_preds == test_labels)
            ece = expected_calibration_error(test_confs, test_preds, test_labels)

            results[cal_name] = {
                "accuracy": accuracy,
                "ece": ece,
                "calibration_time": calibration_time,
                "num_samples": len(test_labels),
            }

            logger.info(f"    {cal_name}: Accuracy={accuracy:.2f}%, ECE={ece:.4f}")

        except Exception as e:
            logger.error(f"    {cal_name} failed: {e}")
            results[cal_name] = {
                "accuracy": 0.0,
                "ece": 1.0,
                "error": str(e),
                "calibration_time": 0.0,
                "num_samples": len(test_labels),
            }

        finally:
            # Clear GPU cache after each method
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return results


def create_enhanced_results_table(
    results_dict: dict, dataset: str, model_name: str
) -> str:
    """Create a comprehensive table with both clean and corruption results"""

    table_title = f"\n ENHANCED CALIBRATION RESULTS"
    table_subtitle = f"{'='*60}"
    experiment_info = f"Method: {list(results_dict.keys())[0]} | Dataset: {dataset.upper()} | Model: {model_name.upper()}"

    # Prepare data for the comprehensive table
    table_data = []

    for method, results in results_dict.items():
        for cal_name, cal_results in results.items():
            if "error" in cal_results:
                # Handle error cases
                row = [
                    cal_name,
                    "ERROR",  # Clean Acc
                    "ERROR",  # Clean ECE
                    "ERROR",  # Corruption Acc
                    "ERROR",  # mCE
                    f"{cal_results.get('calibration_time', 0.0):.2f}s",
                ]
            else:
                # Format regular results
                clean_acc = f"{cal_results['accuracy']:.2f}%"
                clean_ece = f"{cal_results['ece']:.4f}"
                corruption_acc = f"{cal_results.get('corruption_accuracy', 0.0):.2f}%"

                # Handle mCE formatting
                mce = cal_results.get("mce", None)
                if mce is None:
                    mce_str = "N/A"
                elif mce == float("inf"):
                    mce_str = "∞"
                else:
                    mce_str = f"{mce:.3f}"

                calibration_time = f"{cal_results.get('calibration_time', 0.0):.2f}s"

                row = [
                    cal_name,
                    clean_acc,
                    clean_ece,
                    corruption_acc,
                    mce_str,
                    calibration_time,
                ]

            table_data.append(row)

    # Create the comprehensive table
    headers = ["Method", "Clean Acc", "Clean ECE", "Corruption Acc", "mCE", "Time"]
    comprehensive_table = tabulate(
        table_data, headers=headers, tablefmt="grid", floatfmt=".4f"
    )

    # Find best performers
    valid_results = {
        cal_name: cal_results
        for method, results in results_dict.items()
        for cal_name, cal_results in results.items()
        if "error" not in cal_results
    }

    if valid_results:
        # Best calibration (lowest ECE)
        best_calibration = min(valid_results.items(), key=lambda x: x[1]["ece"])
        best_cal_name, best_cal_results = best_calibration

        # Best robustness (lowest mCE, excluding None values)
        valid_mce_results = {
            name: results
            for name, results in valid_results.items()
            if results.get("mce") is not None and results.get("mce") != float("inf")
        }

        if valid_mce_results:
            best_robustness = min(valid_mce_results.items(), key=lambda x: x[1]["mce"])
            best_rob_name, best_rob_results = best_robustness

            best_performers = f"\n Best Calibration (Clean): {best_cal_name} (ECE={best_cal_results['ece']:.4f})"
            best_performers += f"\n Best Robustness (Corruption): {best_rob_name} (mCE={best_rob_results['mce']:.3f})"
        else:
            best_performers = f"\n Best Calibration (Clean): {best_cal_name} (ECE={best_cal_results['ece']:.4f})"
            best_performers += (
                f"\n Corruption results not available for robustness comparison"
            )
    else:
        best_performers = "\n No valid results available for comparison"

    return f"{table_title}\n{table_subtitle}\n{experiment_info}\n\n Comprehensive Results:\n{comprehensive_table}\n{best_performers}\n"


def create_results_table(results_dict: dict, dataset: str, model_name: str) -> str:
    """Create a formatted table of results including timing information (legacy function)"""

    # Prepare table data
    table_data = []
    headers = [
        "Method",
        "Uncalibrated",
        "Temperature",
        "Isotonic",
        "Geometric_Semantic_FAISS",
        "Geometric_Physical_FAISS",
        "Geometric_Semantic_Fast_Separation",
        "Geometric_Physical_Fast_Separation",
        # NEW: AugMix-enhanced calibrators
        "Geometric_Semantic_FAISS_AugMix",
        "Geometric_Physical_FAISS_AugMix",
        "Geometric_Semantic_Fast_Separation_AugMix",
        "Geometric_Physical_Fast_Separation_AugMix",
    ]

    # Create two tables: one for ECE and one for timing
    ece_table_data = []
    timing_table_data = []

    for method, results in results_dict.items():
        ece_row = [method]
        timing_row = [method]

        # Add standard calibrators
        for cal_method in ["Uncalibrated", "Temperature", "Isotonic"]:
            if cal_method in results:
                if "error" in results[cal_method]:
                    ece_row.append("ERROR")
                    timing_row.append("ERROR")
                else:
                    ece_row.append(f"{results[cal_method]['ece']:.4f}")
                    # Handle missing calibration_time gracefully
                    if "calibration_time" in results[cal_method]:
                        timing_row.append(
                            f"{results[cal_method]['calibration_time']:.2f}s"
                        )
                    else:
                        timing_row.append("N/A")
            else:
                ece_row.append("N/A")
                timing_row.append("N/A")

        # Add geometric calibrators (both standard and AugMix-enhanced)
        for cal_method in [
            "Geometric_Semantic_FAISS",
            "Geometric_Physical_FAISS",
            "Geometric_Semantic_Fast_Separation",
            "Geometric_Physical_Fast_Separation",
            # NEW: AugMix-enhanced calibrators
            "Geometric_Semantic_FAISS_AugMix",
            "Geometric_Physical_FAISS_AugMix",
            "Geometric_Semantic_Fast_Separation_AugMix",
            "Geometric_Physical_Fast_Separation_AugMix",
        ]:
            if cal_method in results:
                if "error" in results[cal_method]:
                    ece_row.append("ERROR")
                    timing_row.append("ERROR")
                else:
                    ece_row.append(f"{results[cal_method]['ece']:.4f}")
                    # Handle missing calibration_time gracefully
                    if "calibration_time" in results[cal_method]:
                        timing_row.append(
                            f"{results[cal_method]['calibration_time']:.2f}s"
                        )
                    else:
                        timing_row.append("N/A")
            else:
                ece_row.append("N/A")
                timing_row.append("N/A")

        ece_table_data.append(ece_row)
        timing_table_data.append(timing_row)

    # Create tables
    table_title = f"\n Calibration Results - {dataset.upper()} + {model_name.upper()}"
    table_subtitle = f"{'='*60}"

    ece_table = tabulate(
        ece_table_data, headers=headers, tablefmt="grid", floatfmt=".4f"
    )
    timing_table = tabulate(
        timing_table_data, headers=headers, tablefmt="grid", floatfmt=".2f"
    )

    return f"{table_title}\n{table_subtitle}\n\nExpected Calibration Error (ECE):\n{ece_table}\n\nCalibration Time (seconds):\n{timing_table}\n"


def run_full_vector_distance_fusion_experiment(
    model_adapter,
    train_features,
    train_labels,
    val_features,
    val_raw,
    val_logits,
    val_labels,
    test_features,
    test_logits,
    test_labels,
    output_dir,
    method_name,
    dataset_name,
    model_name,
    seed,
    stab_metric="l2",
    whitening_components=128,
    whitening_eps=1e-6,
):
    """Run minimal full-vector geometric fusion experiment and persist artifacts."""
    logger.info(" Running FullVectorDistanceFusion experiment...")
    if dataset_name != "cifar10":
        raise ValueError("Full-vector experiment path is scoped to CIFAR-10 only.")

    os.makedirs(output_dir, exist_ok=True)
    num_classes = int(np.max(train_labels) + 1)

    val_base_probs = F.softmax(torch.tensor(val_logits), dim=1).numpy()
    test_base_probs = F.softmax(torch.tensor(test_logits), dim=1).numpy()

    fusion = FullVectorDistanceFusionCalibrator(
        model=model_adapter,
        X_train_embed=train_features,
        y_train=train_labels,
        metric=stab_metric,
        library="fast_separation",
        whitening_components=whitening_components,
        whitening_eps=whitening_eps,
    )

    # Validation gate: predicted-class distance consistency for random validation batch.
    rng = np.random.default_rng(seed)
    batch_size = min(128, len(val_labels))
    batch_idx = rng.choice(len(val_labels), size=batch_size, replace=False)
    val_batch_embed = val_features[batch_idx]
    val_batch_probs = val_base_probs[batch_idx]
    batch_pred = np.argmax(val_batch_probs, axis=1)
    dist_matrix_batch = fusion.stab_space.calc_per_class_1nn_distances(val_batch_embed)
    vector_pred_dist = dist_matrix_batch[np.arange(batch_size), batch_pred]
    scalar_same_dist = fusion.stab_space.calc_predicted_class_same_distances(
        val_batch_embed, val_batch_probs
    )
    consistency_max_abs_diff = float(
        np.max(np.abs(vector_pred_dist - scalar_same_dist))
    )

    # Fit beta on calibration split with fixed grid by NLL.
    fusion.fit(
        X_val_embed=val_features,
        y_val=val_labels,
        model_probs=val_base_probs,
    )

    fused_test_probs, fusion_details = fusion.calibrate(
        X_test_embed=test_features,
        model_probs=test_base_probs,
        return_details=True,
    )
    distance_matrix = fusion_details["distance_matrix"]

    # Optional baseline: temperature scaling.
    temp = TemperatureScaling()
    temp.fit(val_logits, val_labels)
    temp_test_probs = temp.calibrate(test_logits)

    def _compute_metrics(name, probs):
        preds = np.argmax(probs, axis=1)
        confs = np.max(probs, axis=1)
        return {
            "method": name,
            "accuracy": float(100.0 * np.mean(preds == test_labels)),
            "nll": multiclass_nll(probs, test_labels),
            "brier": multiclass_brier_score(probs, test_labels, num_classes),
            "top_label_ece": float(expected_calibration_error(confs, preds, test_labels)),
            "classwise_ece": classwise_ece(probs, test_labels),
        }

    base_metrics = _compute_metrics("base_model", test_base_probs)
    temp_metrics = _compute_metrics("temperature", temp_test_probs)
    fused_metrics = _compute_metrics("full_vector_distance_fusion", fused_test_probs)

    base_preds = np.argmax(test_base_probs, axis=1)
    fused_preds = np.argmax(fused_test_probs, axis=1)
    changed = fused_preds != base_preds
    changed_to_correct = int(np.sum(changed & (fused_preds == test_labels) & (base_preds != test_labels)))
    changed_to_wrong = int(np.sum(changed & (fused_preds != test_labels) & (base_preds == test_labels)))
    argmax_change_rate = float(np.mean(changed))
    net_flips = int(changed_to_correct - changed_to_wrong)

    fused_metrics.update(
        {
            "argmax_change_rate": argmax_change_rate,
            "changed_to_correct": changed_to_correct,
            "changed_to_wrong": changed_to_wrong,
            "net_flips": net_flips,
        }
    )

    nll_improved = fused_metrics["nll"] < base_metrics["nll"]
    brier_improved = fused_metrics["brier"] < base_metrics["brier"]
    acc_delta = fused_metrics["accuracy"] - base_metrics["accuracy"]
    calibration_improved = (
        fused_metrics["top_label_ece"] < base_metrics["top_label_ece"]
        or fused_metrics["classwise_ece"] < base_metrics["classwise_ece"]
    )
    beta_near_zero = abs(float(fusion.best_beta)) <= 1e-6

    if nll_improved and brier_improved and net_flips > 0:
        go_no_go = "continue"
    elif calibration_improved and abs(acc_delta) <= 0.1:
        go_no_go = "continue_cautiously"
    elif beta_near_zero or net_flips < 0:
        go_no_go = "stop_or_pivot"
    else:
        go_no_go = "continue_cautiously"

    aggregate = {
        "experiment": "full_vector_distance_fusion",
        "dataset": dataset_name,
        "model": model_name,
        "method": method_name,
        "seed": seed,
        "stability_space": fusion.stab_space.get_params(),
        "beta_grid": fusion.DEFAULT_BETA_GRID.tolist(),
        "selected_beta": float(fusion.best_beta),
        "beta_selection": fusion.beta_selection,
        "validation_gates": {
            "beta0_max_abs_diff": fusion.beta0_max_abs_diff,
            "predicted_class_distance_consistency_max_abs_diff": consistency_max_abs_diff,
        },
        "metrics": {
            "base_model": base_metrics,
            "temperature": temp_metrics,
            "full_vector_distance_fusion": fused_metrics,
        },
        "go_no_go": go_no_go,
        "go_no_go_context": {
            "nll_improved": nll_improved,
            "brier_improved": brier_improved,
            "net_flips": net_flips,
            "acc_delta": acc_delta,
            "beta_near_zero": beta_near_zero,
        },
    }

    table_rows = [
        ["base_model", f"{base_metrics['accuracy']:.2f}", f"{base_metrics['nll']:.4f}", f"{base_metrics['brier']:.4f}", f"{base_metrics['top_label_ece']:.4f}", f"{base_metrics['classwise_ece']:.4f}", "0.0000", "0", "0", "0"],
        ["temperature", f"{temp_metrics['accuracy']:.2f}", f"{temp_metrics['nll']:.4f}", f"{temp_metrics['brier']:.4f}", f"{temp_metrics['top_label_ece']:.4f}", f"{temp_metrics['classwise_ece']:.4f}", "0.0000", "0", "0", "0"],
        ["full_vector_distance_fusion", f"{fused_metrics['accuracy']:.2f}", f"{fused_metrics['nll']:.4f}", f"{fused_metrics['brier']:.4f}", f"{fused_metrics['top_label_ece']:.4f}", f"{fused_metrics['classwise_ece']:.4f}", f"{fused_metrics['argmax_change_rate']:.4f}", str(fused_metrics['changed_to_correct']), str(fused_metrics['changed_to_wrong']), str(fused_metrics['net_flips'])],
    ]
    print(
        tabulate(
            table_rows,
            headers=[
                "Method",
                "Acc",
                "NLL",
                "Brier",
                "TopECE",
                "ClassECE",
                "ArgmaxChangeRate",
                "ChangedToCorrect",
                "ChangedToWrong",
                "NetFlips",
            ],
            tablefmt="grid",
        )
    )

    metrics_json_path = os.path.join(
        output_dir,
        f"full_vector_fusion_metrics_{method_name}_{dataset_name}_{model_name}_seed{seed}.json",
    )
    with open(metrics_json_path, "w") as f:
        json.dump(aggregate, f, indent=2, cls=NumpyEncoder)

    sample_idx = np.arange(len(test_labels), dtype=np.int64)
    per_sample_npz_path = os.path.join(
        output_dir,
        f"full_vector_fusion_per_sample_{method_name}_{dataset_name}_{model_name}_seed{seed}.npz",
    )
    np.savez_compressed(
        per_sample_npz_path,
        sample_idx=sample_idx,
        y_true=test_labels.astype(np.int64),
        base_pred=base_preds.astype(np.int64),
        fused_pred=fused_preds.astype(np.int64),
        argmax_changed=changed.astype(np.bool_),
        base_probs=test_base_probs.astype(np.float32),
        fused_probs=fused_test_probs.astype(np.float32),
        distance_vector=distance_matrix.astype(np.float32),
    )

    # Optional parquet output for tabular workflow compatibility.
    per_sample_parquet_path = os.path.join(
        output_dir,
        f"full_vector_fusion_per_sample_{method_name}_{dataset_name}_{model_name}_seed{seed}.parquet",
    )
    try:
        import pandas as pd

        df = pd.DataFrame(
            {
                "sample_idx": sample_idx,
                "y_true": test_labels.astype(np.int64),
                "base_pred": base_preds.astype(np.int64),
                "fused_pred": fused_preds.astype(np.int64),
                "argmax_changed": changed.astype(np.bool_),
                "base_probs": [row for row in test_base_probs.astype(np.float32)],
                "fused_probs": [row for row in fused_test_probs.astype(np.float32)],
                "distance_vector": [row for row in distance_matrix.astype(np.float32)],
            }
        )
        df.to_parquet(per_sample_parquet_path, index=False)
    except Exception as parquet_err:
        logger.warning(f"Could not save parquet artifact: {parquet_err}")

    logger.info(f" Full-vector metrics saved to: {metrics_json_path}")
    logger.info(f" Full-vector per-sample NPZ saved to: {per_sample_npz_path}")
    return aggregate


def check_model_accuracy(model, test_loader, device):
    """Check model accuracy on test set and return accuracy percentage"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, targets in test_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100.0 * correct / total
    return accuracy


def check_results_completeness(output_file: str) -> bool:
    """
    Check if post-hoc calibration results are complete and valid.

    Args:
        output_file: Path to the results JSON file

    Returns:
        bool: True if all expected calibrators are present and valid
    """
    # Expected calibrators from create_results_table function
    expected_calibrators = [
        "Uncalibrated",
        "Temperature",
        "Isotonic",
        "Geometric_Semantic_FAISS",
        "Geometric_Physical_FAISS",
        "Geometric_Semantic_Fast_Separation",
        "Geometric_Physical_Fast_Separation",
        # NEW: AugMix-enhanced calibrators
        "Geometric_Semantic_FAISS_AugMix",
        "Geometric_Physical_FAISS_AugMix",
        "Geometric_Semantic_Fast_Separation_AugMix",
        "Geometric_Physical_Fast_Separation_AugMix",
    ]

    # Required fields for each calibrator result
    required_fields = ["accuracy", "ece", "num_samples"]

    # Check if file exists
    if not os.path.exists(output_file):
        logger.info(f" Results file does not exist: {os.path.basename(output_file)}")
        return False

    try:
        # Load existing results
        with open(output_file, "r") as f:
            data = json.load(f)

        results = data.get("results", {})
        logger.info(f" Found existing results with {len(results)} calibrators")

        # Check if all expected calibrators are present
        missing_calibrators = []
        error_calibrators = []
        invalid_calibrators = []

        for calibrator in expected_calibrators:
            if calibrator not in results:
                missing_calibrators.append(calibrator)
                continue

            calibrator_result = results[calibrator]

            # Check for error field
            if "error" in calibrator_result:
                error_calibrators.append(calibrator)
                continue

            # Check for required fields
            missing_fields = [
                field for field in required_fields if field not in calibrator_result
            ]
            if missing_fields:
                invalid_calibrators.append(f"{calibrator} (missing: {missing_fields})")
                continue

        # Report status
        if missing_calibrators:
            logger.info(f" Missing calibrators: {missing_calibrators}")
            return False

        if error_calibrators:
            logger.info(f" Calibrators with errors: {error_calibrators}")
            return False

        if invalid_calibrators:
            logger.info(f" Invalid calibrators: {invalid_calibrators}")
            return False

        logger.info(f" All {len(expected_calibrators)} calibrators present and valid")
        return True

    except Exception as e:
        logger.error(f" Error reading results file {output_file}: {e}")
        return False


def main():
    """Main function for post-hoc calibration experiments"""
    parser = argparse.ArgumentParser(description="Run post-hoc calibration experiments")

    # Model specification
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="Training method (e.g., baseline_focal, augmix, etc.)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["cifar10", "cifar100", "svhn"],
        help="Dataset name",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet18", "resnet50", "densenet121", "wide-resnet28-10"],
        help="Model architecture",
    )
    parser.add_argument(
        "--preprocessing_protocol",
        choices=["legacy_v1_mixed_norm", "corrected_v2_train_norm"],
        default="legacy_v1_mixed_norm",
        help=(
            "Legacy paper-reproduction path: default keeps the historical CIFAR-100 "
            "clean-test normalization (ImageNet statistics; a verified train/test "
            "mismatch). Opt in to corrected_v2_train_norm for corrected runs; output "
            "files are NOT versioned by this script, so use a fresh --output_dir."
        ),
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="Random seed used in training"
    )

    # Paths
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/results",
        help="Base directory containing Phase 1 results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/phase2_calibration",
        help="Output directory for Phase 2 results",
    )

    # Experimental parameters
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for data loading"
    )

    # NEW: Corruption dataset directories for enhanced evaluation
    parser.add_argument(
        "--cifar10c_dir",
        type=str,
        default=None,
        help="Path to CIFAR-10-C dataset directory",
    )
    parser.add_argument(
        "--cifar100c_dir",
        type=str,
        default=None,
        help="Path to CIFAR-100-C dataset directory",
    )
    parser.add_argument(
        "--enable_corruption_eval",
        action="store_true",
        help="Enable corruption robustness evaluation",
    )
    parser.add_argument(
        "--run_full_vector_fusion",
        action="store_true",
        help="Run minimal full-vector distance fusion experiment path (CIFAR-10 only).",
    )
    parser.add_argument(
        "--stab_metric",
        choices=["l2", "cosine", "whitened_cosine"],
        default="l2",
        help="Geometry metric for the full-vector fusion path (default: l2).",
    )
    parser.add_argument(
        "--whitening_components",
        type=int,
        default=128,
        help="Maximum PCA components for --stab_metric whitened_cosine (default: 128).",
    )
    parser.add_argument(
        "--whitening_eps",
        type=float,
        default=1e-6,
        help="Variance floor for --stab_metric whitened_cosine (default: 1e-6).",
    )

    args = parser.parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f" Using device: {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Construct paths
    model_path = construct_model_path(
        args.results_dir, args.method, args.dataset, args.model, args.seed
    )
    output_file = os.path.join(
        args.output_dir,
        f"calibration_{args.method}_{args.dataset}_{args.model}_seed{args.seed}.json",
    )

    # Get data loaders
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(
        args.dataset, args.batch_size, args.seed,
        preprocessing_protocol=(
            None if args.preprocessing_protocol == "legacy_v1_mixed_norm" else args.preprocessing_protocol
        ),
    )

    # Load trained model
    model = load_trained_model(
        model_path, args.model, num_classes, device, args.dataset
    )

    # Check model accuracy and exit if below 50%
    accuracy = check_model_accuracy(model, test_loader, device)
    logger.info(f" Model accuracy on test set: {accuracy:.2f}%")

    if accuracy < 50.0:
        logger.info(" Model accuracy is below 50%. Exiting peacefully.")
        return

    # Define expected calibrators based on dataset
    expected_calibrators = [
        "Uncalibrated",
        "Temperature",
        "Isotonic",
        "Geometric_Semantic_Fast_Separation",
        "Geometric_Physical_Fast_Separation",
        "Geometric_Semantic_Fast_Separation_AugMix",
        "Geometric_Physical_Fast_Separation_AugMix",
    ]

    # Add FAISS-based calibrators only if not CIFAR-100
    if args.dataset != "cifar100":
        expected_calibrators.extend(
            [
                "Geometric_Semantic_FAISS",
                "Geometric_Physical_FAISS",
                "Geometric_Semantic_FAISS_AugMix",
                "Geometric_Physical_FAISS_AugMix",
            ]
        )

    # Load existing results if available
    existing_results = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r") as f:
                data = json.load(f)
                existing_results = data.get("results", {})
                logger.info(
                    f" Loaded existing results with {len(existing_results)} calibrators"
                )
        except Exception as e:
            logger.error(f" Error reading existing results: {e}")

    # Find missing calibrators
    missing_calibrators = []
    for cal_name in expected_calibrators:
        if cal_name not in existing_results or "error" in existing_results[cal_name]:
            missing_calibrators.append(cal_name)

    if not missing_calibrators and not args.enable_corruption_eval:
        logger.info(
            " All expected calibrators are present in the results. No need to run additional calculations."
        )
        return

    logger.info(
        f" Found {len(missing_calibrators)} missing calibrators: {missing_calibrators}"
    )

    # Setup feature extraction
    feature_extractor = FeatureExtractor(model, args.model)

    try:
        if args.run_full_vector_fusion:
            logger.info(" FULL-VECTOR FUSION MODE enabled")
            train_raw, train_features, train_logits, train_labels = (
                extract_raw_data_and_features(feature_extractor, train_loader, device)
            )
            val_raw, val_features, val_logits, val_labels = extract_raw_data_and_features(
                feature_extractor, val_loader, device
            )
            test_raw, test_features, test_logits, test_labels = (
                extract_raw_data_and_features(feature_extractor, test_loader, device)
            )
            model_adapter = PyTorchModelAdapter(
                model, device, args.dataset, feature_extractor
            )
            run_full_vector_distance_fusion_experiment(
                model_adapter=model_adapter,
                train_features=train_features,
                train_labels=train_labels,
                val_features=val_features,
                val_raw=val_raw,
                val_logits=val_logits,
                val_labels=val_labels,
                test_features=test_features,
                test_logits=test_logits,
                test_labels=test_labels,
                output_dir=args.output_dir,
                method_name=args.method,
                dataset_name=args.dataset,
                model_name=args.model,
                seed=args.seed,
                stab_metric=args.stab_metric,
                whitening_components=args.whitening_components,
                whitening_eps=args.whitening_eps,
            )
            return

        # Check if we should use enhanced evaluation with corruption assessment
        if args.enable_corruption_eval:
            logger.info(
                " ENHANCED EVALUATION MODE: Including corruption robustness assessment"
            )

            # Use enhanced evaluation function
            enhanced_results = evaluate_calibration_methods_with_corruption(
                model,
                feature_extractor,
                train_loader,
                val_loader,
                test_loader,
                device,
                num_classes,
                args.method,
                args.dataset,
                cifar10c_dir=args.cifar10c_dir,
                cifar100c_dir=args.cifar100c_dir,
            )

            # Create enhanced results table and display
            enhanced_results_dict = {args.method: enhanced_results}
            enhanced_table_str = create_enhanced_results_table(
                enhanced_results_dict, args.dataset, args.model
            )
            print(enhanced_table_str)

            # Save enhanced results
            with open(output_file, "w") as f:
                json.dump(
                    {
                        "experiment_config": vars(args),
                        "results": enhanced_results,
                        "enhanced_evaluation": True,
                        "corruption_evaluation_enabled": True,
                        "timestamp": time.time(),
                    },
                    f,
                    indent=2,
                    cls=NumpyEncoder,
                )

            logger.info(f" Enhanced results saved to: {output_file}")
            return

        # EXISTING: Legacy evaluation for missing calibrators only
        # Extract data for calibration
        train_raw, train_features, train_logits, train_labels = (
            extract_raw_data_and_features(feature_extractor, train_loader, device)
        )
        val_raw, val_features, val_logits, val_labels = extract_raw_data_and_features(
            feature_extractor, val_loader, device
        )
        test_raw, test_features, test_logits, test_labels = (
            extract_raw_data_and_features(feature_extractor, test_loader, device)
        )

        # Create model adapter
        model_adapter = PyTorchModelAdapter(
            model, device, args.dataset, feature_extractor
        )

        # Initialize results with existing data
        results = existing_results.copy()

        #  FIXED: Use the corrected calibration experiment function
        logger.info(
            " Using fixed calibration experiment with proper data type handling"
        )

        # Run the fixed calibration experiment
        fixed_results = run_calibration_experiment_fixed(
            model_adapter=model_adapter,
            train_features=train_features,
            train_raw=train_raw,
            train_labels=train_labels,
            val_features=val_features,
            val_raw=val_raw,
            val_labels=val_labels,
            test_features=test_features,
            test_raw=test_raw,
            test_labels=test_labels,
            test_logits=test_logits,
            dataset_name=args.dataset,
        )

        # Update results with fixed results (only for missing calibrators)
        for cal_name in missing_calibrators:
            if cal_name in fixed_results:
                results[cal_name] = fixed_results[cal_name]
                logger.info(f" Updated {cal_name} with fixed implementation")

        # Create and display results table
        results_dict = {args.method: results}
        table_str = create_results_table(results_dict, args.dataset, args.model)
        print(table_str)

        # Save updated results
        with open(output_file, "w") as f:
            json.dump(
                {
                    "experiment_config": vars(args),
                    "results": results,
                    "timestamp": time.time(),
                },
                f,
                indent=2,
                cls=NumpyEncoder,
            )

        logger.info(f" Results saved to: {output_file}")

    finally:
        # Cleanup
        feature_extractor.cleanup()


if __name__ == "__main__":
    main()
