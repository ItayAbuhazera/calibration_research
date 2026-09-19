# utils/fixed_feature_extractor.py  (new helper)
import torch
import torch.nn.functional as F

class FixedLayerExtractor:
    """
    Grab features from a single, user-chosen layer of the backbone.
    No layer selection, no collection epoch.
    """
    def __init__(self, layer):
        self.layer = layer            # nn.Module to tap
        self._activation = None
        # register a forward hook once
        self.hook = layer.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, inp, out):
        # detach to avoid keeping the graph
        self._activation = out.detach()

    def get_current_features(self):
        if self._activation is None:
            return None
        feat = self._activation
        self._activation = None       # clear for next batch
        # flatten & L2-normalise
        feat = feat.flatten(1)
        feat = F.normalize(feat, dim=1)
        return feat

    def cleanup(self):
        self.hook.remove()
