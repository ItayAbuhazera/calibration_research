"""Frozen constants, site enumeration and small hashing helpers for the atlas program.

Shapes (ResNet-101, CIFAR variant, 32x32 input): stem 64x32x32; layer1 3 blocks 256x32x32;
layer2 4 blocks 512x16x16; layer3 23 blocks 1024x8x8; layer4 3 blocks 2048x4x4.
Every site is a POST-activation tensor: a Bottleneck block output (already ReLU'd) or the stem
``relu(bn1(conv1(x)))``. The final pooled representation ``avg_pool2d(layer4.2, 4)`` equals the
GAP of ``layer4.2`` (verified numerically in the tests), so it is a marked control, not a 35th site.
"""
from __future__ import annotations

import hashlib
import json
from typing import Dict, List, Tuple

import numpy as np

K_NN = 50            # global nearest neighbours; NO sweep
ALPHA = 1.0          # additive smoothing mass of p_geo
NUM_CLASSES = 100
SPP_LEVELS = (1, 2, 4)
POOLS = ("gap", "grid2", "spp")
METRICS = ("unit_l2", "raw_l2", "mahalanobis")
TOPK_CAND = 64       # candidates kept before the deterministic (distance, index) sort
MAH_RANK = 256
MAH_SHRINK = 0.1
MAH_NITER = 6
MAH_FLOOR = 1e-12
DEV_SEEDS = (2, 4)
DEV_CORRUPTIONS = ("gaussian_noise", "defocus_blur", "fog", "jpeg_compression")
DEV_SEVERITIES = (1, 3, 5)
CELLS = tuple(f"{c}_s{s}" for c in DEV_CORRUPTIONS for s in DEV_SEVERITIES)
CONDITIONS = ("clean",) + CELLS
SUBSET_PER_CLASS = 20      # 2 000 test images, class-stratified
SUBSET_SEED = 20260926
ROLE_SPLIT_SEED = 20260927
INNER_SEED = 123           # inherited fit/select split of the old study
KMEANS_K = 100
KMEANS_SEED = 20260928
KMEANS_NINIT = 3
KMEANS_ITERS = 50
MEM_BUDGET_BYTES = 13 * 1024 ** 3   # GPU memory reserved for one group's banks


def block_sites() -> List[Tuple[str, int, int]]:
    """(module name, channels, spatial size) for the stem and every residual block, depth order."""
    out = [("stem", 64, 32)]
    for stage, (n, c, s) in enumerate(((3, 256, 32), (4, 512, 16), (23, 1024, 8), (3, 2048, 4)), start=1):
        out += [(f"layer{stage}.{b}", c, s) for b in range(n)]
    return out


SITES = block_sites()
SITE_NAMES = tuple(s[0] for s in SITES)
SITE_CHANNELS = {s[0]: s[1] for s in SITES}
SITE_SIZE = {s[0]: s[2] for s in SITES}


def valid_levels(size: int) -> Tuple[int, ...]:
    """SPP levels retained for a feature map of side ``size`` (a level q needs q <= size)."""
    return tuple(q for q in SPP_LEVELS if q <= size)


def pool_dim(site: str, pool: str) -> int:
    c, s = SITE_CHANNELS[site], SITE_SIZE[site]
    if pool == "gap":
        return c
    if pool == "grid2":
        return 4 * c
    if pool == "spp":
        return c * sum(q * q for q in valid_levels(s))
    raise ValueError(pool)


def bank_bytes(site: str, n_bank: int = 45000) -> int:
    return sum(pool_dim(site, p) for p in POOLS) * n_bank * 4


def plan_groups(budget: int = MEM_BUDGET_BYTES, n_bank: int = 45000) -> List[List[str]]:
    """Deterministic depth-ordered greedy grouping of sites so one group's three pooled banks fit the budget."""
    groups, cur, used = [], [], 0
    for name in SITE_NAMES:
        b = bank_bytes(name, n_bank)
        if b > budget:
            raise ValueError(f"single site {name} exceeds the memory budget")
        if used + b > budget and cur:
            groups.append(cur); cur, used = [], 0
        cur.append(name); used += b
    if cur:
        groups.append(cur)
    return groups


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=float).encode()).hexdigest()


def sha_arr(*arrs: np.ndarray) -> str:
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
