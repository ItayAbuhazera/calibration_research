"""
Geometric Model Wrapper  – *fixed-layer version*
------------------------------------------------
 • grabs features from one pre-chosen layer (no collection / selection)
 • feeds them to the in-training geometric calibrator
 • everything else (loss-mixing, constellation, etc.) is unchanged
"""

from typing import Dict, Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from utils.fixed_feature_extractor import FixedLayerExtractor
from .in_training_geometric_calibrator import InTrainingGeometricCalibrator

from utils.logging_config import get_logger
logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
#  Helper: pick a sensible default layer for popular backbones                #
# --------------------------------------------------------------------------- #
def _choose_default_layer(model: nn.Module) -> nn.Module:
    """
    Return the nn.Module we want to hook for feature extraction.
    Extend this function if you add more architectures.
    """
    import logging
    from utils.logging_config import get_logger
logger = get_logger(__name__)
    
    logger.info(f"🔍 Analyzing model architecture: {type(model).__name__}")
    
    # Debug: Print model structure
    logger.info("📋 Model modules:")
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Leaf modules only
            logger.info(f"   {name}: {type(module).__name__}")
    
    # ResNet-like detection
    if hasattr(model, "layer3"):
        logger.info("✅ Detected ResNet-like architecture")
        return model.layer3[-1]
    
    # DenseNet detection - improved logic
    if hasattr(model, "features"):
        logger.info("🔍 Found 'features' module, checking for DenseNet...")
        
        # Check for DenseNet structure
        features = model.features
        
        # Try different DenseNet naming patterns
        densenet_candidates = []
        
        # Check for denseblock4 (most common)
        if hasattr(features, "denseblock4"):
            logger.info("   Found denseblock4")
            densenet_candidates.append(("denseblock4", features.denseblock4))
        
        # Check for denseblock3 (fallback)
        if hasattr(features, "denseblock3"):
            logger.info("   Found denseblock3")
            densenet_candidates.append(("denseblock3", features.denseblock3))
        
        # Check for norm5 (final batch norm in DenseNet)
        if hasattr(features, "norm5"):
            logger.info("   Found norm5 (DenseNet final norm)")
            densenet_candidates.append(("norm5", features.norm5))
        
        # Try to find the best candidate
        for name, module in densenet_candidates:
            try:
                # For denseblock modules, try to get the last sub-module
                if "denseblock" in name:
                    if hasattr(module, "__len__") and len(module) > 0:
                        # Sequential-like module
                        layer = module[-1]
                        logger.info(f"✅ Selected DenseNet layer: {name}[-1] = {type(layer).__name__}")
                        return layer
                    elif hasattr(module, "_modules") and len(module._modules) > 0:
                        # Get the last named module
                        last_key = list(module._modules.keys())[-1]
                        layer = module._modules[last_key]
                        logger.info(f"✅ Selected DenseNet layer: {name}.{last_key} = {type(layer).__name__}")
                        return layer
                else:
                    # For norm5 or other single modules
                    logger.info(f"✅ Selected DenseNet layer: {name} = {type(module).__name__}")
                    return module
            except Exception as e:
                logger.warning(f"   Failed to use {name}: {e}")
                continue
        
        # If we reach here, try a more general approach for features
        logger.info("🔍 Trying general feature extraction...")
        try:
            # Get all named children of features
            feature_modules = list(features.named_children())
            if feature_modules:
                # Take the last significant module (not relu, dropout, etc.)
                for name, module in reversed(feature_modules):
                    if any(keyword in name.lower() for keyword in ['block', 'conv', 'norm', 'bn']):
                        logger.info(f"✅ Selected general feature layer: features.{name} = {type(module).__name__}")
                        return module
                
                # If no good candidates, just take the last module
                last_name, last_module = feature_modules[-1]
                logger.info(f"✅ Selected last feature layer: features.{last_name} = {type(last_module).__name__}")
                return last_module
        except Exception as e:
            logger.warning(f"   General feature extraction failed: {e}")
    
    # VGG-like detection
    if hasattr(model, "classifier") and hasattr(model, "features"):
        logger.info("🔍 Detected VGG-like architecture")
        try:
            # For VGG, use the last convolutional layer in features
            features_list = list(model.features.children())
            for module in reversed(features_list):
                if isinstance(module, nn.Conv2d):
                    logger.info(f"✅ Selected VGG layer: Conv2d in features")
                    return module
        except Exception as e:
            logger.warning(f"   VGG layer selection failed: {e}")
    
    # EfficientNet detection
    if "efficientnet" in type(model).__name__.lower():
        logger.info("🔍 Detected EfficientNet architecture")
        try:
            if hasattr(model, "features"):
                # EfficientNet from torchvision
                features_list = list(model.features.children())
                if features_list:
                    last_block = features_list[-2]  # Usually the last MBConv block
                    logger.info(f"✅ Selected EfficientNet layer: features[-2]")
                    return last_block
            elif hasattr(model, "_blocks"):
                # EfficientNet from other implementations
                if model._blocks:
                    last_block = model._blocks[-1]
                    logger.info(f"✅ Selected EfficientNet layer: _blocks[-1]")
                    return last_block
        except Exception as e:
            logger.warning(f"   EfficientNet layer selection failed: {e}")
    
    # Last resort: try to find any reasonable layer
    logger.info("🔍 Trying last resort layer selection...")
    try:
        all_modules = list(model.named_modules())
        
        # Look for common layer types in reverse order
        for name, module in reversed(all_modules):
            # Skip the top-level model and final classifier
            if name == "" or "classifier" in name.lower() or "fc" in name.lower():
                continue
            
            # Look for convolutional, batch norm, or block-like modules
            if isinstance(module, (nn.Conv2d, nn.BatchNorm2d)) or "block" in name.lower():
                # Make sure it's not too early in the network
                if len(name.split('.')) >= 2:  # At least 2 levels deep
                    logger.info(f"✅ Selected fallback layer: {name} = {type(module).__name__}")
                    return module
    except Exception as e:
        logger.warning(f"   Last resort layer selection failed: {e}")
    
    # If all else fails, provide detailed error with suggestions
    logger.error("❌ Could not automatically select a layer")
    logger.error("Available top-level modules:")
    for name, module in model.named_children():
        logger.error(f"   {name}: {type(module).__name__}")
    
    raise ValueError(
        f"Automatic layer choice not implemented for this backbone: {type(model).__name__}. "
        f"Available modules: {[name for name, _ in model.named_children()]}. "
        f"Please pass a custom layer via `fixed_layer` kwarg. "
        f"For DenseNet, try: model.features.norm5 or model.features.denseblock4"
    )
# --------------------------------------------------------------------------- #
#  Geometric-calibrated model                                                 #
# --------------------------------------------------------------------------- #
class GeometricCalibratedModel(nn.Module):
    """
    Wrapper that adds:
      • Fixed-layer feature extractor
      • In-training geometric calibrator
      • (optional) constellation loss
    """

    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        fixed_layer: Optional[nn.Module] = None,
        use_constellation_loss: bool = True,
        # -- calibrator hyper-params (pass straight through) --
        warmup_epochs: int = 10,
        accuracy_threshold: float = 0.5,
        calibration_update_frequency: int = 10,
        stability_data_ratio: float = 0.75,  # 75% for stability, 25% for isotonic
        stability_metric: str = "l2",
        min_samples_for_fitting: int = 200,
        temperature_init: float = 1.0,
        # -- constellation hyper-params --
        constellation_alpha: float = 1.0,
        constellation_beta: float = 1.0,
        constellation_gamma: float = 1.0,
    ):
        super().__init__()

        self.base_model = base_model
        self.num_classes = num_classes

        # ------------------------------------------------------------------ #
        # 1.  Fixed-layer feature extractor                                  #
        # ------------------------------------------------------------------ #
        if fixed_layer is None:
            fixed_layer = _choose_default_layer(base_model)
            logger.info(f"Using default tapped layer: {fixed_layer}")

        self.feature_extractor = FixedLayerExtractor(fixed_layer)
        self.layer_selected = True          # always true in the fixed version

        # ------------------------------------------------------------------ #
        # 2.  Geometric calibrator                                           #
        # ------------------------------------------------------------------ #
        self.calibrator = InTrainingGeometricCalibrator(
            num_classes=num_classes,
            warmup_epochs=warmup_epochs,
            accuracy_threshold=accuracy_threshold,
            calibration_update_frequency=calibration_update_frequency,
            stability_data_ratio=stability_data_ratio,
            stability_metric=stability_metric,
            min_samples_for_fitting=min_samples_for_fitting,
            temperature_init=temperature_init,
        )

        # ------------------------------------------------------------------ #
        # 3.  Constellation loss settings                                    #
        # ------------------------------------------------------------------ #
        self.use_constellation_loss = use_constellation_loss
        self.constellation_alpha = constellation_alpha
        self.constellation_beta = constellation_beta
        self.constellation_gamma = constellation_gamma

        # misc tracking
        self.current_epoch = 0
        self.forward_call_count = 0
        self.compute_loss_call_count = 0

        logger.info("✅ GeometricCalibratedModel (fixed-layer) initialised.")

    # ---------------------------------------------------------------------- #
    #  Training-loop helpers                                                 #
    # ---------------------------------------------------------------------- #
    def update_epoch(self, epoch: int, validation_accuracy: float = None):
        """Notify calibrator of epoch change."""
        self.current_epoch = epoch
        self.calibrator.update_epoch(epoch, validation_accuracy)

    # ---------------------------------------------------------------------- #
    #  Forward                                                               #
    # ---------------------------------------------------------------------- #
    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        *,
        return_components: bool = False,
        apply_calibration: Optional[bool] = None,
    ) -> Union[torch.Tensor, Dict[str, Any]]:
        """
        Forward with optional calibration.
        """
        self.forward_call_count += 1
        # Log every 50 forward calls to avoid spam
        if self.forward_call_count % 500 == 1:
            logger.info(f"🔍 GeometricModel forward #{self.forward_call_count}: "
                    f"training={self.training}, targets={'provided' if targets is not None else 'None'}")

        logits = self.base_model(x)

        # ---- grab features from the tapped layer ------------------------- #
        features = self.feature_extractor.get_current_features()
        if features is not None and len(features.shape) > 2:
            features = features.flatten(1)
        if features is not None:
            features = F.normalize(features, dim=1)

        # ---- feed calibrator --------------------------------------------- #
        if apply_calibration is None:
            apply_calibration = not self.training

        if apply_calibration and features is not None:
            probs = self.calibrator(
                logits=logits,
                features=features,
                targets=targets,
                apply_calibration=True,
            )
        else:
            probs = F.softmax(logits, dim=1)

        # ---- collect data (only during training) ------------------------- #
        if self.training and targets is not None and features is not None:
            self.calibrator.collect_training_data(features, targets, logits)

        if return_components:
            return {
                "logits": logits,
                "probabilities": probs,
                "features": features,
                "calibration_stats": self.calibrator.get_calibration_stats(),
            }
        return probs

    # ---------------------------------------------------------------------- #
    #  Loss                                                                  #
    # ---------------------------------------------------------------------- #
    def compute_constellation_loss(
        self, features: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        if not self.use_constellation_loss or features is None:
            return torch.zeros(
                1, device=targets.device, dtype=torch.float32, requires_grad=False
            )

        from Losses.loss import _constellation_loss_impl  # local import

        loss = _constellation_loss_impl(
            features,
            labels=targets,
            alpha=self.constellation_alpha,
            beta=self.constellation_beta,
            gamma=self.constellation_gamma,
        )
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)

    def compute_total_loss(
        self,
        x: torch.Tensor,
        targets: torch.Tensor,
        *,
        cross_entropy_weight: float = 0.5,
        constellation_weight: float = 0.3,
        triplet_weight: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        self.compute_loss_call_count += 1

        logits = self.base_model(x)
        ce_loss = F.cross_entropy(logits, targets)

        features = self.feature_extractor.get_current_features()
        if features is not None and len(features.shape) > 2:
            features = features.flatten(1)
        if features is not None:
            features = F.normalize(features, dim=1)

        const_loss = self.compute_constellation_loss(features, targets)

        total = (
            cross_entropy_weight * ce_loss
            + constellation_weight * const_loss
            + triplet_weight * torch.tensor(0.0, device=x.device)
        )

        return total, {
            "cross_entropy": ce_loss,
            "constellation": const_loss,
            "total": total,
        }

    # ---------------------------------------------------------------------- #
    #  House-keeping                                                         #
    # ---------------------------------------------------------------------- #
    def cleanup(self):
        self.feature_extractor.cleanup()
        logger.info(
            f"🧹 cleanup: forward_calls={self.forward_call_count}, "
            f"loss_calls={self.compute_loss_call_count}"
        )


# --------------------------------------------------------------------------- #
#  Factory                                                                    #
# --------------------------------------------------------------------------- #
def create_geometric_calibrated_model(
    base_model: nn.Module, num_classes: int, **kwargs
) -> GeometricCalibratedModel:
    return GeometricCalibratedModel(
        base_model=base_model,
        num_classes=num_classes,
        **kwargs,
    )
