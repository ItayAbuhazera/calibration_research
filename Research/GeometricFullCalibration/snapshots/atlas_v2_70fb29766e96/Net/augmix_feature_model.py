import torch
import torch.nn as nn
import torch.nn.functional as F

class AugMixFeatureModel(nn.Module):
    """Wrapper to extract normalized features during AugMix training."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.features = None
        self._register_feature_hook()

    def _register_feature_hook(self):
        """Register forward hook to capture features before the final layer."""
        if hasattr(self.base_model, "fc"):
            for name, module in self.base_model.named_modules():
                if name == "avgpool" or (name.startswith("layer") and "avgpool" not in name):
                    module.register_forward_hook(self._feature_hook)
                    break
        elif hasattr(self.base_model, "classifier"):
            for name, module in self.base_model.named_modules():
                if "features" in name and isinstance(module, (nn.BatchNorm2d, nn.ReLU)):
                    module.register_forward_hook(self._feature_hook)
                    break

    def _feature_hook(self, module, input, output):
        if isinstance(output, torch.Tensor):
            if output.dim() > 2:
                feats = torch.flatten(output, 1)
            else:
                feats = output
            self.features = F.normalize(feats, dim=1)

    def forward(self, x, return_features: bool = False):
        self.features = None
        logits = self.base_model(x)
        if return_features:
            return logits, self.features
        return logits
