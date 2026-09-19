"""
Model utilities for loading models, extracting features, and handling data.

This module consolidates model-related utilities including:
- PyTorchModelAdapter: Wrapper for PyTorch models with predict_proba interface
- FeatureExtractor: Extract features from specific model layers
- load_trained_model: Load trained model checkpoints
- get_data_loaders: Get train/val/test data loaders
- construct_model_path: Construct path to trained model checkpoints
"""

import os
import sys
import copy
import re
import glob
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logging_config import get_logger
logger = get_logger(__name__)


# =============================================================================
# SEED UTILITY
# =============================================================================

def set_seed(seed: int = 42):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =============================================================================
# PYTORCH MODEL ADAPTER
# =============================================================================

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
                f"Feature extractor integrated for {feature_extractor.model_name}"
            )
        if self.requires_224_input:
            logger.info(
                f" Model requires 224x224 input - will resize images automatically"
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

            # ENHANCED DIAGNOSTICS for first batch
            if len(probs_chunks) == 0:
                try:
                    min_v = float(logits.min().item())
                    max_v = float(logits.max().item())
                    mean_v = float(logits.mean().item())
                    std_v = float(logits.std().item())
                    logger.info(f"First batch logits stats:")
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
                            "CRITICAL: Logits have near-zero variance! This will produce uniform probabilities."
                        )
                        logger.error(
                            f"   This suggests the model is not making predictions (weights might not be loaded)."
                        )

                    # Check if logits are all zeros or very small
                    if abs(mean_v) < 1e-6 and std_v < 1e-6:
                        logger.error(
                            "CRITICAL: Logits are all near-zero! Model appears uninitialized."
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


# =============================================================================
# FEATURE EXTRACTOR
# =============================================================================

class FeatureExtractor:
    """Extract features from specific model layers"""

    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name.lower()
        self.features = None
        self.hook = None
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
            logger.info(f"Research-optimized extraction: {config['layer']}")
            logger.info(f"   Expected dim: {config['expected_dim']}")
            logger.info(f"   Rationale: {config['reason']}")
        elif "wide" in self.model_name and "resnet" in self.model_name:
            config = {
                "layer": "block3",
                "expected_dim": 640,
                "reason": "Final conv block for Wide-ResNet",
            }
            target_layer = self._get_target_layer(config["layer"])
            logger.info(f"Wide-ResNet extraction: {config['layer']}")
            logger.info(f"   Expected dim: {config['expected_dim']}")
            logger.info(f"   Rationale: {config['reason']}")
        else:
            # Fallback to current logic with warning
            target_layer = self._get_fallback_layer()
            logger.warning(f"Using fallback extraction for {self.model_name}")

        # Register the hook
        if target_layer is not None:
            self.hook = target_layer.register_forward_hook(self._feature_hook)
            logger.info(f"Feature extraction hook registered successfully")

            # Add feature dimension validation
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
                            logger.info("Using transition2 as denseblock3 fallback")
                            return module

                # Custom DenseNet implementation
                elif hasattr(self.model, "dense3"):
                    return self.model.dense3

                # Final fallback
                logger.warning("Could not find denseblock3, using features module")
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
            logger.info(f"Expected feature dimensions: {min_dim}-{max_dim}")

            # This validation will run during first forward pass
            def validation_hook(module, input, output):
                if len(output.shape) == 4:
                    feat_dim = output.shape[1]  # Channel dimension
                else:
                    feat_dim = output.shape[-1]

                if feat_dim < min_dim or feat_dim > max_dim:
                    logger.warning(
                        f"Feature dimension {feat_dim} outside expected range {min_dim}-{max_dim}"
                    )
                else:
                    logger.info(
                        f"Feature dimension {feat_dim} within expected range"
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
            f"Extracted features: {features.shape}, logits: {logits.shape}, labels: {labels.shape}"
        )
        return features, logits, labels

    def cleanup(self):
        """Remove the hook"""
        if self.hook:
            self.hook.remove()


# =============================================================================
# MODEL LOADING
# =============================================================================

def load_trained_model(
    model_path: str, model_name: str, num_classes: int, device, dataset: str = "cifar10"
):
    """Load a trained model from checkpoint, handling model wrappers and architecture mismatches."""

    # FIXED: Use the correct model architecture based on the dataset
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

            # FIX: Extract temperature from checkpoint if available to match training config
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
                            f"Extracted temperature from checkpoint: {temp_param}"
                        )

            model = Wide_ResNet_Cifar(
                BasicBlock,
                layers,
                widen_factor,
                num_classes=num_classes,
                temp=temp_param,
            )
            logger.info(
                f"Wide-ResNet instantiated: depth={depth}, width={widen_factor}, temp={temp_param}"
            )
            logger.info(f"   Model type: {type(model)}")
            logger.info(f"   Has base_model attr: {hasattr(model, 'base_model')}")
            logger.info(f"   Model has fc layer: {hasattr(model, 'fc')}")
            if hasattr(model, "fc"):
                logger.info(
                    f"   FC layer shape: in={model.fc.in_features}, out={model.fc.out_features}"
                )
        elif "dinov2" in model_name.lower():
            from Net.dinov2 import DINOv2Classifier
            
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
                    "Detected timm-style DINOv2 checkpoint, using timm model"
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
                    f"Created timm DINOv2 model: {timm_name} with {num_classes} classes"
                )
                # Store flag to indicate timm model (for Strategy 6)
                model._is_timm_model = True
            else:
                # Use HuggingFace for HuggingFace-style checkpoints
                logger.info(
                    "Detected HuggingFace-style DINOv2 checkpoint, using HuggingFace model"
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
            "Detected wrapped checkpoint with 'model_state'; extracting weights"
        )
        state_dict = state_dict["model_state"]

    model_keys = list(model.state_dict().keys())
    checkpoint_keys = list(state_dict.keys())

    logger.info(
        f"Checkpoint has {len(checkpoint_keys)} keys, model expects {len(model_keys)} keys"
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

    logger.info(f"Best strategy: '{best_strategy}' with {best_loaded} keys loaded")
    final_missing, final_unexpected = model.load_state_dict(
        best_state_dict, strict=False
    )

    matched_keys = len(model_keys) - len(final_missing)
    if matched_keys < max(1, int(len(model_keys) * 0.5)):
        raise RuntimeError(
            f"Failed to load model weights properly. Only {matched_keys}/{len(model_keys)} model keys matched."
        )

    if final_missing:
        logger.info(
            f"   Remaining missing keys after best strategy: {final_missing[:5]}"
        )
    if final_unexpected:
        logger.info(
            f"   Remaining unexpected keys after best strategy: {final_unexpected[:5]}"
        )

    # ENHANCED: Verify critical weights were actually loaded (especially for Wide-ResNet)
    if "wide" in model_name.lower() and "resnet" in model_name.lower():
        logger.info("Verifying Wide-ResNet weights loaded properly...")

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
                    "FC layer weights appear to be zeros! Model may not be properly loaded."
                )
            elif fc_weight_std < 1e-6:
                logger.warning(
                    "FC layer weights have zero variance - this is suspicious!"
                )
            else:
                logger.info("   FC layer weights look reasonable")

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
        f"Model loaded successfully with {matched_keys}/{len(model_keys)} model keys"
    )
    return model


# =============================================================================
# DATA LOADERS
# =============================================================================

def get_data_loaders(
    dataset: str,
    batch_size: int = 128,
    seed: int = 42,
    target_domain: str = "sketch",
    corruption_type: str = None,
    corruption_severity: int = None,
    image_size: int = None,
):
    """
    Get data loaders with same split as Phase 1.

    For corruption experiments:
    - Train/Val loaders remain CLEAN (for calibration fitting)
    - Test loader uses corrupted data (for evaluation)

    Args:
        image_size: If provided, resize images to this size (for models like DINOv2 that require larger inputs)
    """
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
            for ds in [train_loader.dataset, val_loader.dataset]:
                if hasattr(ds, "transform") and ds.transform is not None:
                    ds.transform = wrap_transform_with_resize(
                        ds.transform, image_size
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
            for ds in [train_loader.dataset, val_loader.dataset]:
                if hasattr(ds, "transform") and ds.transform is not None:
                    ds.transform = wrap_transform_with_resize(
                        ds.transform, image_size
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
            test_loader = cifar100_test(batch_size=batch_size, shuffle=False)
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


# =============================================================================
# MODEL PATH CONSTRUCTION
# =============================================================================

def construct_model_path(
    base_dir: str,
    method: str,
    dataset: str,
    model: str,
    seed: int,
    target_domain: str = "sketch",
) -> str:
    """Construct the path to a trained model"""

    # First, check for the path structure used by train_model.py
    # Structure: base_dir/method/baseline_cross_entropy/dataset/model/seed{seed}/best_model.pth
    training_script_path = os.path.join(
        base_dir, method, "baseline_cross_entropy", dataset, model, f"seed{seed}", "best_model.pth"
    )
    if os.path.exists(training_script_path):
        logger.info(f"Found model at training script path: {training_script_path}")
        return training_script_path

    # Determine experiment type based on method
    if method.startswith("baseline_"):
        exp_type = "baseline"
    elif method in ["augmix", "augmix_constellation", "baseline"]:
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
            logger.info(f"Found DINOv2 model in calibration_comparison: {model_path}")
            return model_path
        else:
            logger.warning(
                f"DINOv2 model not found with pattern: {calibration_pattern}, trying standard path"
            )

    # Standard path structure (for non-DINOv2 or baseline DINOv2 models)
    base_path = os.path.join(base_dir, exp_type, method, dataset, model, f"seed{seed}")
    flat_checkpoint = os.path.join(base_path, "best_model.pth")
    nested_checkpoint = os.path.join(base_path, exp_name, "best_model.pth")

    if os.path.exists(flat_checkpoint):
        logger.info(f"Using flat checkpoint path: {flat_checkpoint}")
        return flat_checkpoint

    logger.info(
        f"Flat checkpoint missing; falling back to nested path: {nested_checkpoint}"
    )
    return nested_checkpoint
