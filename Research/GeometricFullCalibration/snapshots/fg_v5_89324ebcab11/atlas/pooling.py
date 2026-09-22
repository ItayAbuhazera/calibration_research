"""Pooling families for the representation atlas (pre-L2 pooled vector a_{l,P}).

  gap   : adaptive average pool to 1x1, flattened                              -> C
  grid2 : adaptive average pool to 2x2, flattened (channel-major)              -> 4C
  spp   : concat over valid levels q in {1,2,4} of (1/q) * flatten(avg_pool(q x q)),
          then divided by sqrt(#valid levels)                                  -> C * sum q^2

The 1/q weight makes each level's coordinates count 1/q rather than 1 (a q x q level has q^2 cells
per channel, so an unweighted concatenation would weight resolutions by their cell count). It does
NOT equalize empirical signal energy; per-level squared norms are recorded (``spp_level_sq_norms``).
Nothing here is learned. The repository's ``FixedSizeSPP_JL`` / ``ChannelFirstSPP_JL`` project or
max-pool and are NOT numerically equivalent to this definition, so they are not used.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from . import spec

EPS = 1e-12


def pool(x: torch.Tensor, kind: str) -> torch.Tensor:
    """x: [N, C, H, W] post-activation -> [N, d] pre-L2 pooled vector a."""
    n, _, h, w = x.shape
    if kind == "gap":
        return x.mean(dim=(2, 3))
    if kind == "grid2":
        return F.adaptive_avg_pool2d(x, (2, 2)).reshape(n, -1)
    if kind == "spp":
        lv = spec.valid_levels(min(h, w))
        parts = [F.adaptive_avg_pool2d(x, (q, q)).reshape(n, -1) / q for q in lv]
        return torch.cat(parts, dim=1) / (len(lv) ** 0.5)
    raise ValueError(kind)


def spp_level_sq_norms(x: torch.Tensor) -> torch.Tensor:
    """[N, n_levels] squared norm contributed by each (weighted) SPP level, before the 1/sqrt(L) factor."""
    n, _, h, w = x.shape
    lv = spec.valid_levels(min(h, w))
    return torch.stack([(F.adaptive_avg_pool2d(x, (q, q)).reshape(n, -1) / q).pow(2).sum(1) for q in lv], dim=1)


def unit(a: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """(v, ||a||) with v = a / max(||a||, eps). Zero vectors map to zero (reported by callers)."""
    nrm = a.norm(dim=1)
    return a / nrm.clamp_min(EPS).unsqueeze(1), nrm
