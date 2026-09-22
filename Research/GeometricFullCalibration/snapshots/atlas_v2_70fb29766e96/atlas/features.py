"""Backbone access: post-activation hooks at every site, and pooled-feature helpers.

``SiteTap`` registers forward hooks on the requested sites of the CIFAR ResNet-101
(``Net/resnet_cifar.py``). Block outputs are hooked directly (they are post-ReLU). The stem is
``relu(bn1(conv1(x)))`` in ``forward``; the functional ReLU has no module, so ``bn1`` is hooked and
the ReLU is applied to its output. Feature maps are handed to a callback per batch; nothing is stored here.
"""
from __future__ import annotations

from typing import Callable, Dict, Sequence

import torch
import torch.nn.functional as F


class SiteTap:
    def __init__(self, model: torch.nn.Module, sites: Sequence[str]):
        mods = dict(model.named_modules())
        self.maps: Dict[str, torch.Tensor] = {}
        self.handles = []
        for s in sites:
            if s == "stem":
                self.handles.append(mods["bn1"].register_forward_hook(self._mk(s, relu=True)))
            else:
                self.handles.append(mods[s].register_forward_hook(self._mk(s, relu=False)))

    def _mk(self, name: str, relu: bool):
        def hook(_m, _i, out):
            self.maps[name] = F.relu(out) if relu else out
        return hook

    def close(self):
        for h in self.handles:
            h.remove()


@torch.no_grad()
def forward_maps(model, tap: SiteTap, x: torch.Tensor):
    """Returns (logits float32 [b,C], {site: [b,C,H,W]})."""
    tap.maps.clear()
    z = model(x).float()
    return z, dict(tap.maps)


def batches(arr, bs: int, device):
    import numpy as np
    for s in range(0, arr.shape[0], bs):
        yield s, torch.from_numpy(np.ascontiguousarray(arr[s:s + bs])).to(device).float()
