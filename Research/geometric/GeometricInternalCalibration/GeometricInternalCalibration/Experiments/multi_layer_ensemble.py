#!/usr/bin/env python3
"""
Multi-Layer Geometric Calibration Ensemble - Phase 1 Starter

Ensembles calibrated confidences from multiple semantic layers.
Learns optimal layer weights via proper scoring rules (Brier/NLL).

Theory: Within-model deep ensemble over representation depths.
"""

import os
import sys
import json
import logging
import argparse
import gc
import tempfile
import fcntl
import errno
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pathlib import Path

# Import shared utilities
from Experiments.layer_selection import (
    extract_features_directly,
    NumpyEncoder
)
from Calibrators.calibration_utils import (
    compute_ece,
    compute_error_detection_auroc,
    compute_aurc,
)
# Import calibrator
from Calibrators.geometric_calibrator_new import GeometricCalibrator

# Same utilities you used before
from Experiments.run_post_hoc_calibration import (
    PyTorchModelAdapter,
    get_data_loaders,
    load_trained_model,
)

# Import from layer_selection (for now - will refactor later)
try:
    from Experiments.layer_selection import (
        compute_mce_for_method,
        run_empirical_ground_truth_evaluation,
        run_standard_baselines,
    )
except ImportError:
    compute_mce_for_method = None
    run_empirical_ground_truth_evaluation = None
    run_standard_baselines = None
    print("Warning: layer_selection helpers not found, some features disabled")

# Import metrics documentation tool
try:
    from Experiments.metrics_topk_doc import run_metrics_doc_topk
except ImportError:
    run_metrics_doc_topk = None
    print("Warning: metrics_topk_doc not found, --doc-metrics will be disabled")

from utils.logging_config import get_logger
from utils.layer_utils import filter_feature_layers
logger = get_logger(__name__)


# ============================================================================
# DATA LOADING HELPERS
# ============================================================================

def get_all_data_as_numpy(loader: torch.utils.data.DataLoader):
    xs, ys = [], []
    for batch in loader:
        if isinstance(batch, (list, tuple)):
            x, y = batch[:2]
        else:
            x, y = batch, None
        xs.append(x.cpu().numpy())
        if y is not None:
            ys.append(y.cpu().numpy())
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0) if ys else None
    return X, Y


# ============================================================================
# DYNAMIC LAYER DISCOVERY + SAMPLING + VERIFICATION
# ============================================================================

def _safe_to_tensor(x):
    try:
        from utils.tensor_utils import coerce_to_tensor
        return coerce_to_tensor(x)
    except Exception:
        if isinstance(x, (tuple, list)):
            x = x[0]
        return x if torch.is_tensor(x) else torch.as_tensor(x)


def discover_model_layers(model, *, device='cuda', input_shape=(1, 3, 32, 32),
                          keep_kinds=('Conv', 'BatchNorm', 'ReLU', 'SiLU', 'GELU', 'Bottleneck', 'BasicBlock', 'Linear'),
                          always_include_last=True) -> list:
    """
    Returns a list of dicts describing extractable layers in forward order:
      {'idx': int, 'name': str, 'type': str, 'shape': tuple|None, 'spatial': int|None}
    We consider a layer 'valid' if it yields a tensor output on a dummy forward hook.
    
    Uses single-pass forward with all hooks registered at once for efficiency.
    """
    model = model.to(device).eval()
    modules = list(model.modules())
    dummy = torch.randn(*input_shape, device=device)

    # Register all hooks at once
    acts = [None] * len(modules)
    hooks = []
    
    def make_hook(i):
        def _h(mod, inp, out):
            t = _safe_to_tensor(out)
            acts[i] = t.detach() if torch.is_tensor(t) else None
        return _h

    for i, m in enumerate(modules):
        hooks.append(m.register_forward_hook(make_hook(i)))

    # Single forward pass
    with torch.no_grad():
        try:
            _ = model(dummy)
        except Exception as e:
            logger.warning(f"Forward pass failed during discovery: {e}")

    # Remove all hooks
    for h in hooks:
        h.remove()

    # Build discovered list
    discovered = []
    for idx, (m, t) in enumerate(zip(modules, acts)):
        mtype = m.__class__.__name__
        
        # Keep "semantic" modules by type hint
        keep = any(k in mtype for k in keep_kinds)
        if keep and t is not None:
            tshape = tuple(t.shape)
            spatial = int(t.shape[2]) if t.dim() >= 3 else None
            discovered.append({
                'idx': idx, 
                'name': mtype, 
                'type': mtype,
                'shape': tshape, 
                'spatial': spatial
            })

    # Filter out activation and non-feature layers before returning
    # Note: At this point, layers have 'name' set to module type, not full path
    # The filtering will work on module types, and we'll get proper names later via normalize_discovered_layers
    filtered = filter_feature_layers(discovered, model)
    
    return filtered


def sample_candidate_layers(
    discovered: list,
    *,
    sample_every: int = 5,
    block_names=('layer', 'dense', 'transition', 'block')
) -> list:
    """
    Two sampling modes:
      - sample_every > 0: keep every Nth discovered layer (by order)
      - sample_every == 0: per-block sampling (first occurrence per block token)

    Always preserves order and deduplicates indices; **forces inclusion of the
    deepest spatial layer** (penultimate conv-like layer) instead of the very last
    module (which may be a Linear classifier).
    """
    if not discovered:
        return []

    # deepest layer that still has a spatial map
    spatial_idxs = [d['idx'] for d in discovered if d.get('spatial') is not None]
    if len(spatial_idxs) > 0:
        must_keep = spatial_idxs[-1]
    else:
        # fallback: second-to-last discovered if available, otherwise last
        must_keep = discovered[-2]['idx'] if len(discovered) >= 2 else discovered[-1]['idx']

    if sample_every > 0:
        cand = [d['idx'] for i, d in enumerate(discovered) if (i % sample_every == 0)]
    else:
        cand, seen_blocks = [], set()
        for d in discovered:
            nm = (d['name'] or '') + ' ' + (d['type'] or '')
            token = next((b for b in block_names if b.lower() in nm.lower()), None)
            if token and token not in seen_blocks:
                cand.append(d['idx'])
                seen_blocks.add(token)

    # ensure we include the deepest spatial layer (penultimate conv-like)
    if must_keep not in cand:
        cand.append(must_keep)

    cand = sorted(set(cand))
    return cand


def verify_layer_depth_order(model, candidate_layers: list, device='cuda', input_shape=(1, 3, 32, 32)) -> bool:
    """
    Heuristic check: for CNNs, deeper layers tend to have non-increasing spatial dims.
    Logs spatial sizes; returns True if non-increasing for all defined spatial entries.
    
    Uses single-pass forward with hooks on candidate layers only.
    """
    model = model.to(device).eval()
    modules = list(model.modules())
    dummy = torch.randn(*input_shape, device=device)

    # Register hooks only on candidate layers
    acts = {i: None for i in candidate_layers}
    hooks = []
    
    for i in candidate_layers:
        if i >= len(modules):
            continue
            
        def make_hook(idx):
            def _h(mod, inp, out):
                t = _safe_to_tensor(out)
                acts[idx] = t.detach() if torch.is_tensor(t) else None
            return _h
        
        hooks.append(modules[i].register_forward_hook(make_hook(i)))

    # Single forward pass
    with torch.no_grad():
        try:
            _ = model(dummy)
        except Exception as e:
            logger.warning(f"Forward pass failed during verification: {e}")
    
    # Remove hooks
    for h in hooks:
        h.remove()

    # Extract spatial dimensions
    spatial = []
    for i in candidate_layers:
        t = acts.get(i)
        s = int(t.shape[2]) if (t is not None and t.dim() >= 3) else None
        spatial.append(s)
        logger.info(f"candidate idx={i}: spatial={s}, shape={tuple(t.shape) if t is not None else None}")

    # Check non-increasing property
    ok, prev = True, None
    for s in spatial:
        if s is None:
            continue
        if prev is not None and s > prev:
            ok = False
            break
        prev = s

    if ok:
        logger.info("✅ candidate layers appear depth-ordered by spatial size.")
    else:
        logger.warning(f"⚠️ depth order may be incorrect; spatial sequence = {spatial}")
    return ok

# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def format_floats_7sig(obj):
    """Recursively format all floats to 7 significant digits."""
    if isinstance(obj, (float, np.floating)):
        return float(f"{float(obj):.7g}")  # 7 significant digits, stays numeric
    if isinstance(obj, dict):
        return {k: format_floats_7sig(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [format_floats_7sig(v) for v in obj]
    return obj


def _ees(weights: Dict[int, float]) -> float:
    """Compute Effective Ensemble Size: exp(-∑ w_i log w_i)."""
    w = np.array(list(weights.values()), dtype=float)
    w = np.clip(w, 1e-12, 1)
    w /= w.sum()
    H = -np.sum(w * np.log(w))
    return float(np.exp(H))


# ============================================================================
# CANONICAL DEPTH + HASHING + DEDUP CONFIG
# ============================================================================
import hashlib

def _md5_str(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:10]

def _atomic_save(path: str, arr: np.ndarray):
    """Atomically save array to avoid half-written files if job crashes."""
    # 1) Ensure target dir exists
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

    # 2) Create a unique temp file *in the same directory*
    fd, tmp = tempfile.mkstemp(
        dir=d,
        prefix=os.path.basename(path) + ".",
        suffix=".tmp",
    )
    try:
        # 3) Write the file completely, then flush & fsync
        with os.fdopen(fd, "wb") as f:
            np.save(f, arr, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())

        # 4) Atomic replace
        os.replace(tmp, path)
    finally:
        # If something went wrong before replace, clean up the temp file
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass

def _oracle_from_cached(layer_probs: Dict[int, Dict[str, np.ndarray]], test_labels: np.ndarray, n_bins: int = 15):
    """Compute oracle single-layer results from cached probabilities (no refits)."""
    out = []
    C = int(next(iter(layer_probs.values()))['test'].shape[1])
    oh = np.eye(C, dtype=np.float32)[test_labels]
    
    for L, d in layer_probs.items():
        P = d['test']
        ece = float(compute_ece(P, test_labels, n_bins=n_bins))
        acc = float(np.mean(np.argmax(P, axis=1) == test_labels))
        brier = float(np.mean(np.sum((P - oh)**2, axis=1)))
        out.append({
            "layer_idx": int(L),
            "test_ece": ece,
            "test_accuracy": acc,
            "test_brier": brier,
            "test_error_auroc": float(compute_error_detection_auroc(P, test_labels)),
            "test_aurc": float(compute_aurc(P, test_labels))
        })
    return out

def model_signature(model: torch.nn.Module) -> str:
    # Cheap, shape-only signature (robust across tiny numeric changes)
    meta = [(k, tuple(v.shape), str(v.dtype)) for k, v in model.state_dict().items()]
    return _md5_str(json.dumps(meta, sort_keys=True))

def arrays_signature(*arrays: np.ndarray, labels_only: bool = False) -> str:
    """
    Build a lightweight signature of the arrays we care about.
    If labels_only=True, hash only label vectors to avoid huge memory copies.
    """
    h = hashlib.md5()
    for a in arrays:
        if a is None:
            h.update(b"NONE")
            continue
        if labels_only and a.ndim == 1:
            h.update(a.astype(np.int64, copy=False).tobytes())
        else:
            h.update(str(a.shape).encode())
            h.update(str(a.dtype).encode())
    return h.hexdigest()[:10]

def build_depth_map_from_model(
    model_adapter,
    candidate_layers: List[int],
    sample_np: np.ndarray,     # use val_raw[:1]
    device: str = "cuda"
) -> Dict[int, int]:
    """
    One forward pass with hooks on candidate layers to capture true execution order.
    Returns {layer_idx -> depth_rank} where 0 = shallowest.
    """
    model = model_adapter.model.to(device).eval()
    modules = list(model.modules())

    seen: List[int] = []
    hooks = []

    def make_hook(L):
        def _h(_m, _i, _o):
            seen.append(L)
        return _h

    for L in candidate_layers:
        if 0 <= L < len(modules):
            hooks.append(modules[L].register_forward_hook(make_hook(L)))

    with torch.no_grad():
        x = torch.from_numpy(sample_np).to(device)
        _ = model(x)

    for h in hooks:
        h.remove()

    # Keep first occurrence per L in forward order; fallback to given order
    order = []
    seen_set = set()
    for L in seen:
        if L not in seen_set:
            order.append(L); seen_set.add(L)
    for L in candidate_layers:
        if L not in seen_set:
            order.append(L); seen_set.add(L)

    return {L: i for i, L in enumerate(order)}

def canonical_layers(layers: List[int], depth_map: Dict[int, int]) -> Tuple[int, ...]:
    """Sort by depth rank, then by id as tie-breaker."""
    return tuple(sorted(layers, key=lambda L: (depth_map.get(L, 10**9), L)))

def depth_weights_CANONICAL(
    selected_layers: List[int],
    depth_map: Dict[int, int],
    mode: str = "deep",
    alpha: float = 1.5
) -> Dict[int, float]:
    if not selected_layers:
        return {}
    ranks = np.array([depth_map.get(L, 0) for L in selected_layers], dtype=float)
    if mode == "shallow":
        scores = np.exp(-alpha * ranks)
    else:
        scores = np.exp(+alpha * ranks)
    w = scores / scores.sum()
    return {int(L): float(w[i]) for i, L in enumerate(selected_layers)}

@dataclass(frozen=True)
class EnsembleConfig:
    layers: Tuple[int, ...]   # canonical (depth-sorted)
    K: int
    weighting: str
    model_sig: str
    data_sig: str
    hp_sig: str

    def __hash__(self):
        return hash((self.layers, self.K, self.weighting, self.model_sig, self.data_sig, self.hp_sig))

class EnsembleConfigCache:
    """
    Request de-duplication + results cache for metric-picked ensembles.
    """
    def __init__(self, depth_map: Dict[int, int], model_sig: str, data_sig: str, hyperparams: Dict):
        self.depth_map = depth_map
        self.model_sig = model_sig
        self.data_sig = data_sig
        self.hp_sig = _md5_str(json.dumps(hyperparams, sort_keys=True))
        self.cache: Dict[EnsembleConfig, Dict] = {}
        self.requesters: Dict[EnsembleConfig, List[Dict]] = {}

    def make_config(self, layers: List[int], K: int, wm: str) -> EnsembleConfig:
        canon = canonical_layers(layers[:K], self.depth_map)
        return EnsembleConfig(
            layers=canon, K=K, weighting=wm,
            model_sig=self.model_sig, data_sig=self.data_sig, hp_sig=self.hp_sig
        )

    def register(self, cfg: EnsembleConfig, metric_name: str, original_layers: List[int]):
        self.requesters.setdefault(cfg, []).append({"metric": metric_name, "original": list(original_layers[:cfg.K])})

    def unique(self) -> List[EnsembleConfig]:
        return list(self.requesters.keys())

    def put(self, cfg: EnsembleConfig, result: Dict):
        self.cache[cfg] = result

    def get(self, cfg: EnsembleConfig) -> Optional[Dict]:
        return self.cache.get(cfg)

    def stats(self) -> Dict[str, float]:
        total = sum(len(v) for v in self.requesters.values())
        uniq = len(self.requesters)
        ratio = (total / max(1, uniq))
        return {"total_requests": total, "unique_configs": uniq, "dedup_ratio": ratio, "speedup_est": f"{ratio:.1f}x"}


# ============================================================================
# FEATURE CACHE
# ============================================================================

class FeatureCache:
    """Cache extracted features to avoid recomputation."""
    
    def __init__(self, root: str):
        os.makedirs(root, exist_ok=True)
        self.root = root
        self.lock_dir = os.path.join(root, ".locks")
        os.makedirs(self.lock_dir, exist_ok=True)
    
    def _path(self, split: str, layer: int) -> str:
        return os.path.join(self.root, f"{split}.layer{layer}.npy")
    
    def _lock_path(self, split: str, layer: int) -> str:
        return os.path.join(self.lock_dir, f"{split}.layer{layer}.lock")
    
    def get(self, split: str, layer: int) -> Optional[np.ndarray]:
        path = self._path(split, layer)
        lock_path = self._lock_path(split, layer)
        
        if not os.path.exists(path):
            return None
        
        # Acquire shared lock for reading
        lock_file = open(lock_path, 'w')
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)  # Shared lock
            
            # Double-check file still exists and has content
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return None
            
            logger.debug(f"✓ Cache hit: {split}, layer {layer}")
            arr = np.load(path, mmap_mode='r')
            # Make a copy to avoid mmap issues after releasing lock
            result = np.array(arr, dtype=np.float32)
            return result
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)  # Release lock
            lock_file.close()
    
    def put(self, split: str, layer: int, arr: np.ndarray):
        path = self._path(split, layer)
        lock_path = self._lock_path(split, layer)
        
        # Force float32 for consistent dtype and smaller files
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        
        # Acquire exclusive lock for writing
        lock_file = open(lock_path, 'w')
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)  # Exclusive lock
            
            # Check if another process already wrote this while we waited
            if os.path.exists(path) and os.path.getsize(path) > 0:
                logger.debug(f"✓ Cache already written by another process: {split}, layer {layer}")
                return
            
            _atomic_save(path, arr)
            logger.debug(f"✓ Cached: {split}, layer {layer}")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)  # Release lock
            lock_file.close()


# ============================================================================
# PROBABILITIES CACHE
# ============================================================================

class ProbsCache:
    """Cache per-layer calibrated probabilities to avoid recomputation."""
    
    def __init__(self, root: str):
        self.root = os.path.join(root, "probs_cache")
        os.makedirs(self.root, exist_ok=True)
        self.lock_dir = os.path.join(self.root, ".locks")
        os.makedirs(self.lock_dir, exist_ok=True)
    
    def _path(self, split: str, layer: int) -> str:
        return os.path.join(self.root, f"{split}.layer{int(layer)}.npy")
    
    def _lock_path(self, split: str, layer: int) -> str:
        return os.path.join(self.lock_dir, f"{split}.layer{int(layer)}.lock")
    
    def get(self, split: str, layer: int) -> Optional[np.ndarray]:
        """Load cached probabilities for a layer and split."""
        path = self._path(split, layer)
        lock_path = self._lock_path(split, layer)
        
        if not os.path.exists(path):
            return None
        
        lock_file = open(lock_path, 'w')
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_SH)
            
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return None
            
            logger.debug(f"✓ Probs cache hit: {split}, layer {layer}")
            arr = np.load(path, mmap_mode='r')
            result = np.array(arr, dtype=np.float32)
            return result
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
    
    def put(self, split: str, layer: int, arr: np.ndarray):
        """Cache probabilities for a layer and split."""
        path = self._path(split, layer)
        lock_path = self._lock_path(split, layer)
        
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        
        lock_file = open(lock_path, 'w')
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            
            if os.path.exists(path) and os.path.getsize(path) > 0:
                logger.debug(f"✓ Probs cache already written: {split}, layer {layer}")
                return
            
            _atomic_save(path, arr)
            logger.debug(f"✓ Cached probs: {split}, layer {layer}")
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            lock_file.close()
    
    def has_all_layers(self, split: str, layers: List[int]) -> bool:
        """Check if all layers are cached for a split."""
        return all(os.path.exists(self._path(split, L)) for L in layers)


class InMemoryFeatureCache:
    """In-memory feature cache used when disk caching is disabled."""

    def __init__(self):
        self.store: Dict[Tuple[str, int], np.ndarray] = {}

    def get(self, split: str, layer: int) -> Optional[np.ndarray]:
        key = (split, int(layer))
        arr = self.store.get(key)
        if arr is None:
            return None
        return np.array(arr, dtype=np.float32, copy=True)

    def put(self, split: str, layer: int, arr: np.ndarray):
        key = (split, int(layer))
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        self.store[key] = np.array(arr, dtype=np.float32, copy=True)


class InMemoryProbsCache:
    """In-memory probabilities cache used when disk caching is disabled."""

    def __init__(self):
        self.store: Dict[Tuple[str, int], np.ndarray] = {}

    def get(self, split: str, layer: int) -> Optional[np.ndarray]:
        key = (split, int(layer))
        arr = self.store.get(key)
        if arr is None:
            return None
        return np.array(arr, dtype=np.float32, copy=True)

    def put(self, split: str, layer: int, arr: np.ndarray):
        key = (split, int(layer))
        if arr.dtype != np.float32:
            arr = arr.astype(np.float32, copy=False)
        self.store[key] = np.array(arr, dtype=np.float32, copy=True)

    def has_all_layers(self, split: str, layers: List[int]) -> bool:
        return all((split, int(L)) in self.store for L in layers)


def make_feature_cache(
    cache_root: Optional[Path],
    disable_cache: bool,
    subdir: Optional[str] = None,
):
    """Factory to build the appropriate feature cache."""
    if disable_cache:
        return InMemoryFeatureCache()
    if cache_root is None:
        raise ValueError("cache_root must be provided when caching is enabled")
    target = Path(cache_root)
    if subdir is not None:
        target = target / subdir
    return FeatureCache(str(target))


def make_probs_cache(cache_root: Optional[Path], disable_cache: bool):
    """Factory to build the appropriate probabilities cache."""
    if disable_cache:
        return InMemoryProbsCache()
    if cache_root is None:
        raise ValueError("cache_root must be provided when caching is enabled")
    return ProbsCache(str(cache_root))


def extract_with_cache(raw_model, loader, layer, device, fc: FeatureCache, split: str):
    """Extract features with caching."""
    cached = fc.get(split, layer)
    if cached is not None:
        # Copy memmap to avoid in-place modification issues
        return np.array(cached, dtype=np.float32)
    
    logger.info(f"Extracting features: {split}, layer {layer}")
    # Use fixed-size SPP+JL compression to control feature dimensionality (matches earlier 512-dim setup)
    feats, _ = extract_features_directly(
        raw_model,
        loader,
        layer,
        device,
        fixed_feature_dim=512,   # compress to 512 dims before calibration
    )
    arr = feats.detach().cpu().numpy()
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    fc.put(split, layer, arr)
    return arr


def precompute_layer_probabilities(
    model_adapter,
    candidate_layers: List[int],
    train_raw: np.ndarray, train_labels: np.ndarray,
    val_raw: np.ndarray, val_labels: np.ndarray,
    test_raw: np.ndarray, test_labels: np.ndarray,
    feature_cache: FeatureCache,
    probs_cache: ProbsCache,
    device: str = 'cuda',
    batch_size: int = 128,
    learned_loss: str = 'brier'
) -> Dict[int, Dict[str, np.ndarray]]:
    """
    Precompute calibrated probabilities for all candidate layers.
    Returns dict mapping layer_idx -> {'val': probs, 'test': probs}
    """
    logger.info("🚀 Precomputing calibrated probabilities for all candidate layers...")
    
    raw_model = model_adapter.model
    dev = torch.device(device)
    
    # Dataloaders
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
        batch_size=batch_size, shuffle=False
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(test_raw), torch.zeros(len(test_raw), dtype=torch.long)),
        batch_size=batch_size, shuffle=False
    )
    
    layer_probs = {}
    
    for layer_idx in tqdm(candidate_layers, desc="Precomputing layer probabilities"):
        logger.info(f"Processing layer {layer_idx}...")
        
        # Check if already cached
        val_probs = probs_cache.get('val', layer_idx)
        test_probs = probs_cache.get('test', layer_idx)
        
        if val_probs is not None and test_probs is not None:
            logger.info(f"  ✓ Layer {layer_idx} probabilities already cached")
            layer_probs[layer_idx] = {
                'val': np.array(val_probs, dtype=np.float32),
                'test': np.array(test_probs, dtype=np.float32)
            }
            continue
        
        # Extract features
        X_train_feat = extract_with_cache(
            raw_model, train_loader, layer_idx, dev, feature_cache, 'train'
        )
        X_val_feat = extract_with_cache(
            raw_model, val_loader, layer_idx, dev, feature_cache, 'val'
        )
        X_test_feat = extract_with_cache(
            raw_model, test_loader, layer_idx, dev, feature_cache, 'test'
        )
        
        # Fit calibrator
        logger.info(f"  Fitting calibrator for layer {layer_idx}...")
        calib = GeometricCalibrator(
            model=model_adapter,
            X_train_embed=X_train_feat,
            y_train=train_labels,
            library="fast_separation",
            auto_select_layer=False,
            device=device
        )
        
        calib.fit(
            X_val_embed=X_val_feat,
            y_val=val_labels,
            X_val_original=val_raw,
            fit_batch_size=batch_size
        )
        
        # Compute probabilities
        logger.info(f"  Computing calibrated probabilities for layer {layer_idx}...")
        val_probs = calib.calibrate_batched(
            X_test_embed=X_val_feat,
            X_test_original=val_raw,
            batch_size=batch_size
        )
        test_probs = calib.calibrate_batched(
            X_test_embed=X_test_feat,
            X_test_original=test_raw,
            batch_size=batch_size
        )
        
        # Cache probabilities
        probs_cache.put('val', layer_idx, val_probs)
        probs_cache.put('test', layer_idx, test_probs)
        
        layer_probs[layer_idx] = {
            'val': val_probs,
            'test': test_probs
        }
        
        # Free memory
        del X_train_feat, X_val_feat, X_test_feat, calib
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"  ✓ Layer {layer_idx} completed")
    
    logger.info(f"✅ Precomputed probabilities for {len(layer_probs)} layers")
    return layer_probs


# ============================================================================
# WEIGHT LEARNING
# ============================================================================

def learn_layer_weights(val_targets: np.ndarray,
                        layer_val_probs: Dict[int, np.ndarray],
                        loss: str = "brier",
                        max_iter: int = 500,
                        tol: float = 1e-6,
                        *,
                        lr: float = 0.1,
                        use_lbfgs: bool = False,
                        seed: int = 0) -> Dict[int, float]:
    """
    PyTorch-only learned weights on VALIDATION ONLY.
    - Parameterize w = softmax(theta) (simplex guaranteed).
    - Optimize Brier (or NLL) on the validation set.
    - No test leakage.

    Returns: {layer_idx -> weight} with sum=1, weights>=0.
    """
    layers = sorted(layer_val_probs.keys())
    K = len(layers)
    if K == 0:
        return {}
    P = np.stack([layer_val_probs[L] for L in layers], axis=0)  # (K,N,C)
    N, C = P.shape[1], P.shape[2]
    if len(val_targets) != N:
        raise ValueError(
            f"Weight learning split mismatch: probs N={N} but labels N={len(val_targets)}. "
            "Did you pass Val-A labels with full-Val probs (or vice versa)?"
        )
    eps = 1e-12
    P = np.clip(P, eps, 1.0)
    P /= P.sum(axis=2, keepdims=True)

    y = val_targets.astype(int)
    Y = np.eye(C, dtype=np.float32)[y]  # (N,C) - use float32 for consistency

    device = torch.device('cpu')
    Pt = torch.from_numpy(P.astype(np.float32, copy=False)).to(device)   # (K,N,C)
    Yt = torch.from_numpy(Y).to(device)   # (N,C)

    # theta -> w = softmax(theta)
    torch.manual_seed(seed)
    theta = torch.zeros(K, dtype=torch.float32, device=device, requires_grad=True)
    theta.data.add_(0.01 * torch.randn_like(theta))  # tiny asymmetry

    if use_lbfgs:
        opt = torch.optim.LBFGS([theta], lr=1.0, max_iter=max_iter, tolerance_grad=1e-7, tolerance_change=tol)
        def closure():
            opt.zero_grad()
            w = torch.softmax(theta, dim=0)
            pmix = torch.einsum('k,knc->nc', w, Pt)
            pmix = torch.clamp(pmix, 1e-12, 1.0)
            pmix = pmix / pmix.sum(dim=1, keepdim=True)
            if loss == "nll":
                obj = torch.nn.functional.nll_loss(torch.log(pmix), torch.argmax(Yt, dim=1))
            else:
                D = pmix - Yt
                obj = torch.mean(torch.sum(D * D, dim=1))  # Brier
            obj.backward()
            return obj
        opt.step(closure)
    else:
        opt = torch.optim.Adam([theta], lr=lr)
        best_val, best_theta = None, theta.detach().clone()
        for _ in range(max_iter):
            opt.zero_grad()
            w = torch.softmax(theta, dim=0)
            pmix = torch.einsum('k,knc->nc', w, Pt)
            pmix = torch.clamp(pmix, 1e-12, 1.0)
            pmix = pmix / pmix.sum(dim=1, keepdim=True)
            if loss == "nll":
                obj = torch.nn.functional.nll_loss(torch.log(pmix), torch.argmax(Yt, dim=1))
            else:
                D = pmix - Yt
                obj = torch.mean(torch.sum(D * D, dim=1))  # Brier
            val = float(obj.item())
            obj.backward()
            opt.step()
            if best_val is None or val < best_val - 1e-7:
                best_val, best_theta = val, theta.detach().clone()
        theta = best_theta

    with torch.no_grad():
        w = torch.softmax(theta, dim=0).cpu().numpy()
    
    weights = {int(L): float(w[i]) for i, L in enumerate(layers)}
    logger.info(f"Learned weights ({loss}, {'LBFGS' if use_lbfgs else 'Adam'}): {weights}")
    return weights


# ============================================================================
# LAYER SELECTION
# ============================================================================

def get_candidate_layers(model, model_name: str, sample_every: int = 5, device='cuda', input_shape=(1, 3, 32, 32)) -> List[int]:
    """
    Discover and sample candidate layers from the concrete model instance.
    Uses dummy forward hooks; returns ordered indices.
    """
    logger.info("Discovering model layers for candidate selection...")
    discovered = discover_model_layers(model, device=device, input_shape=input_shape)
    logger.info(f"Discovered {len(discovered)} valid layers")
    
    # Log sample of discovered layers for debugging
    if discovered:
        logger.info("Sample of discovered layers:")
        for d in discovered[:3]:
            logger.info(f"  idx={d['idx']}, type={d['type']}, spatial={d['spatial']}, shape={d['shape']}")
        if len(discovered) > 6:
            logger.info("  ...")
            for d in discovered[-3:]:
                logger.info(f"  idx={d['idx']}, type={d['type']}, spatial={d['spatial']}, shape={d['shape']}")

    candidates = sample_candidate_layers(discovered, sample_every=sample_every)
    logger.info(f"Sample_every={sample_every} -> {len(candidates)} candidate layers: {candidates}")

    _ = verify_layer_depth_order(model, candidates, device=device, input_shape=input_shape)
    return candidates


def depth_weights(selected_layers: List[int], mode: str = "deep", alpha: float = 1.5) -> Dict[int, float]:
    """
    Monotonic schedules over depth. Uses selection order (assumes already depth-ordered).
    mode: "deep" favors deeper; "shallow" favors shallower.
    """
    if not selected_layers:
        return {}

    # Use selection order directly (already depth-ordered from discovery)
    ordered = list(selected_layers)
    if mode == "shallow":
        ordered = list(reversed(ordered))

    ranks = np.arange(len(ordered))
    w = np.exp(alpha * ranks)
    w = w / w.sum()
    return {L: float(w[i]) for i, L in enumerate(ordered)}


def pick_topk_layers_simple(candidate_layers: List[int], k: int) -> List[int]:
    """
    Layer selection logic:
    - If k == 0: use ALL sampled candidates (K=0 means "all layers, vary weights only")
    - If k >= len(candidate_layers): use ALL candidates
    - Otherwise: use last K layers (deeper = better hypothesis)
    
    For more sophisticated selection, integrate with layer_selection.py metrics.
    """
    if k == 0 or k >= len(candidate_layers):
        return candidate_layers
    
    # Hypothesis: deeper layers have better semantic features
    return candidate_layers[-k:]


# ============================================================================
# MULTI-LAYER ENSEMBLE
# ============================================================================

@dataclass
class EnsembleResult:
    """Results from ensemble evaluation."""
    k: int
    selected_layers: List[int]
    weights: Dict[int, float]
    weighting_method: str
    test_ece: float
    test_accuracy: float
    test_brier: float
    mce: Optional[float] = None


class MultiLayerEnsemble:
    """Multi-layer geometric calibration ensemble."""
    
    def __init__(self, model_adapter, device: str = 'cuda'):
        self.model_adapter = model_adapter
        self.device = device
        self.calibrators: Dict[int, GeometricCalibrator] = {}
        self.weights: Dict[int, float] = {}
        self.selected_layers: List[int] = []
        # allow passing depth alpha via attribute if set by caller
        self.depth_alpha: float = 1.5
        self.depth_map: Optional[Dict[int,int]] = None   # ← NEW
        # For cached mode
        self.use_cached_probs: bool = False
        self.layer_probs: Optional[Dict[int, Dict[str, np.ndarray]]] = None
    
    def fit_with_cached_probs(self, selected_layers: List[int],
                             val_labels: np.ndarray,
                             layer_probs: Dict[int, Dict[str, np.ndarray]],
                             weighting_method: str = 'learned',
                             learned_loss: str = 'brier',
                             initial_weights: Optional[List[float]] = None):
        """Fit ensemble using precomputed layer probabilities (fast path).

        Args:
            selected_layers: List of layer indices to ensemble
            val_labels: Validation labels for learning weights
            layer_probs: Precomputed probabilities for each layer
            weighting_method: Method for computing weights ('learned', 'uniform', 'deep', 'shallow', 'nc4_rank', 'nc1_rank', 'pareto_rank')
            learned_loss: Loss function for learned weights ('brier' or 'nll')
            initial_weights: Optional precomputed weights (for NC-based methods). If provided and weighting_method is
                           one of the NC-based methods, these weights are used directly.
        """
        import time
        t0 = time.time()

        self.selected_layers = selected_layers
        self.use_cached_probs = True
        self.layer_probs = layer_probs

        logger.info(f"⚡ Fast ensemble fitting with cached probabilities...")

        # Learn or set weights
        if len(selected_layers) == 1:
            self.weights = {selected_layers[0]: 1.0}
        elif initial_weights is not None and weighting_method in ['nc4_rank', 'nc1_rank', 'pareto_rank']:
            # Use provided NC-based initial weights
            if len(initial_weights) != len(selected_layers):
                raise ValueError(f"initial_weights length ({len(initial_weights)}) must match selected_layers length ({len(selected_layers)})")
            self.weights = {L: float(w) for L, w in zip(selected_layers, initial_weights)}
            logger.info(f"Using provided initial weights for {weighting_method}: {self.weights}")
        elif weighting_method == 'learned':
            # Use cached validation probabilities for weight learning
            layer_val_probs = {L: layer_probs[L]['val'] for L in selected_layers}
            self.weights = learn_layer_weights(
                val_labels, layer_val_probs, loss=learned_loss,
                max_iter=getattr(self, 'learned_max_iter', 200),  # Reduced from 500
                lr=getattr(self, 'learned_lr', 0.1),
                use_lbfgs=getattr(self, 'learned_use_lbfgs', False),
                seed=int(getattr(self, 'seed', 0)) if hasattr(self, 'seed') else 0
            )
        elif weighting_method == 'uniform':
            self.weights = {L: 1.0 / len(selected_layers) for L in selected_layers}
        elif weighting_method == 'deep':
            alpha = getattr(self, 'depth_alpha', 1.5)
            if self.depth_map is not None:
                self.weights = depth_weights_CANONICAL(selected_layers, self.depth_map, mode="deep", alpha=alpha)
            else:
                self.weights = depth_weights(selected_layers, mode="deep", alpha=alpha)
        elif weighting_method == 'shallow':
            alpha = getattr(self, 'depth_alpha', 1.5)
            if self.depth_map is not None:
                self.weights = depth_weights_CANONICAL(selected_layers, self.depth_map, mode="shallow", alpha=alpha)
            else:
                self.weights = depth_weights(selected_layers, mode="shallow", alpha=alpha)
        else:
            raise ValueError(f"Unknown weighting method: {weighting_method}")

        # normalize (safety)
        s = sum(self.weights.values())
        if s <= 0:
            self.weights = {L: 1.0 / len(selected_layers) for L in selected_layers}
        else:
            self.weights = {k: v / s for k, v in self.weights.items()}
        
        logger.info(f"⚡ Fast fitting completed in {time.time()-t0:.2f}s")
        logger.info(f"Final weights ({weighting_method}): {self.weights}")
        
        return self.weights

    def fit(self, selected_layers: List[int],
            train_raw: np.ndarray, train_labels: np.ndarray,
            val_raw: np.ndarray, val_labels: np.ndarray,
            feature_cache: FeatureCache,
            batch_size: int = 128,
            weighting_method: str = 'learned',
            learned_loss: str = 'brier',
            initial_weights: Optional[List[float]] = None):
        """Fit calibrators and learn weights (legacy method).

        Args:
            selected_layers: List of layer indices to ensemble
            train_raw: Training data
            train_labels: Training labels
            val_raw: Validation data
            val_labels: Validation labels
            feature_cache: Cache for features
            batch_size: Batch size for processing
            weighting_method: Method for computing weights
            learned_loss: Loss function for learned weights
            initial_weights: Optional precomputed weights (for NC-based methods)
        """
        import time
        t0 = time.time()
        
        self.selected_layers = selected_layers
        raw_model = self.model_adapter.model
        dev = torch.device(self.device)
        
        # Dataloaders
        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(train_raw), torch.from_numpy(train_labels)),
            batch_size=batch_size, shuffle=False
        )
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_raw), torch.from_numpy(val_labels)),
            batch_size=batch_size, shuffle=False
        )
        
        # Only compute val_probs if we need them for learned weights
        need_val_probs = (weighting_method == 'learned')
        layer_val_probs = {} if need_val_probs else None
        
        for layer_idx in tqdm(selected_layers, desc="Fitting calibrators"):
            logger.info(f"Layer {layer_idx}: extracting features...")
            
            X_train_feat = extract_with_cache(
                raw_model, train_loader, layer_idx, dev, feature_cache, 'train'
            )
            X_val_feat = extract_with_cache(
                raw_model, val_loader, layer_idx, dev, feature_cache, 'val'
            )
            
            logger.info(f"Layer {layer_idx}: fitting calibrator...")
            calib = GeometricCalibrator(
                model=self.model_adapter,
                X_train_embed=X_train_feat,
                y_train=train_labels,
                library="fast_separation",
                auto_select_layer=False,
                device=self.device
            )
            
            calib.fit(
                X_val_embed=X_val_feat,
                y_val=val_labels,
                X_val_original=val_raw,
                fit_batch_size=batch_size
            )
            
            self.calibrators[layer_idx] = calib
            
            # Only compute validation probs if needed for weight learning
            if need_val_probs:
                val_probs = calib.calibrate_batched(
                    X_test_embed=X_val_feat,
                    X_test_original=val_raw,
                    batch_size=batch_size
                )
                layer_val_probs[layer_idx] = val_probs
            
            logger.info(f"  ✓ Layer {layer_idx} fitted")
            
            # Free memory after processing this layer
            del X_train_feat, X_val_feat
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Learn or set weights
        if len(selected_layers) == 1:
            self.weights = {selected_layers[0]: 1.0}
        elif initial_weights is not None and weighting_method in ['nc4_rank', 'nc1_rank', 'pareto_rank']:
            # Use provided NC-based initial weights
            if len(initial_weights) != len(selected_layers):
                raise ValueError(f"initial_weights length ({len(initial_weights)}) must match selected_layers length ({len(selected_layers)})")
            self.weights = {L: float(w) for L, w in zip(selected_layers, initial_weights)}
            logger.info(f"Using provided initial weights for {weighting_method}: {self.weights}")
        elif weighting_method == 'learned':
            self.weights = learn_layer_weights(
                val_labels, layer_val_probs, loss=learned_loss,
                max_iter=getattr(self, 'learned_max_iter', 500),
                lr=getattr(self, 'learned_lr', 0.1),
                use_lbfgs=getattr(self, 'learned_use_lbfgs', False),
                seed=int(getattr(self, 'seed', 0)) if hasattr(self, 'seed') else 0
            )
            # Free val_probs after learning
            del layer_val_probs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        elif weighting_method == 'uniform':
            self.weights = {L: 1.0 / len(selected_layers) for L in selected_layers}
        elif weighting_method == 'deep':
            alpha = getattr(self, 'depth_alpha', 1.5)
            if self.depth_map is not None:
                self.weights = depth_weights_CANONICAL(selected_layers, self.depth_map, mode="deep", alpha=alpha)
            else:
                self.weights = depth_weights(selected_layers, mode="deep", alpha=alpha)
        elif weighting_method == 'shallow':
            alpha = getattr(self, 'depth_alpha', 1.5)
            if self.depth_map is not None:
                self.weights = depth_weights_CANONICAL(selected_layers, self.depth_map, mode="shallow", alpha=alpha)
            else:
                self.weights = depth_weights(selected_layers, mode="shallow", alpha=alpha)
        else:
            raise ValueError(f"Unknown weighting method: {weighting_method}")

        # normalize (safety)
        s = sum(self.weights.values())
        if s <= 0:
            self.weights = {L: 1.0 / len(selected_layers) for L in selected_layers}
        else:
            self.weights = {k: v / s for k, v in self.weights.items()}
        
        logger.info(f"Legacy fitting completed in {time.time()-t0:.2f}s")
        logger.info(f"Final weights ({weighting_method}): {self.weights}")
        
        return self.weights
    
    def predict(self, X_split: np.ndarray, feature_cache: FeatureCache,
                batch_size: int = 128, *, split_name: str = "test") -> np.ndarray:
        """Predict using weighted ensemble (streaming to reduce memory)."""
        import time
        t0 = time.time()
        
        # Ensure no gradient accumulation
        with torch.no_grad():
            if self.use_cached_probs and self.layer_probs is not None:
                # Fast path: use cached probabilities
                logger.info("⚡ Fast prediction with cached probabilities...")
                ensemble_probs = None
                
                for layer_idx in self.selected_layers:
                    if split_name not in self.layer_probs[layer_idx]:
                        raise KeyError(f"Cached probs missing split '{split_name}' for layer {layer_idx}")
                    probs = self.layer_probs[layer_idx][split_name]
                    
                    # Check for NaNs
                    if not np.all(np.isfinite(probs)):
                        logger.warning(f"Layer {layer_idx}: non-finite probs, using uniform")
                        n_classes = probs.shape[1]
                        probs = np.full_like(probs, 1.0 / n_classes)
                    
                    # Accumulate weighted probabilities directly
                    if ensemble_probs is None:
                        ensemble_probs = self.weights[layer_idx] * probs
                    else:
                        ensemble_probs += self.weights[layer_idx] * probs
                
                # Renormalize
                eps = 1e-12
                ensemble_probs = np.clip(ensemble_probs, eps, 1.0)
                ensemble_probs /= ensemble_probs.sum(axis=1, keepdims=True)
                
                logger.info(f"⚡ Fast prediction completed in {time.time()-t0:.2f}s")
                return ensemble_probs
            
            # Legacy path: refit calibrators
            logger.info("🐌 Legacy prediction (refitting calibrators)...")
            raw_model = self.model_adapter.model
            dev = torch.device(self.device)
            
            test_loader = DataLoader(
                TensorDataset(torch.from_numpy(X_split), torch.zeros(len(X_split), dtype=torch.long)),
                batch_size=batch_size, shuffle=False
            )
            
            # Stream weighted probs instead of storing all layer_probs
            ensemble_probs = None
            
            for layer_idx in self.selected_layers:
                X_test_feat = extract_with_cache(
                    raw_model, test_loader, layer_idx, dev, feature_cache, split_name
                )
                
                probs = self.calibrators[layer_idx].calibrate_batched(
                    X_test_embed=X_test_feat,
                    X_test_original=X_split,
                    batch_size=batch_size
                )
                
                # Check for NaNs
                if not np.all(np.isfinite(probs)):
                    logger.warning(f"Layer {layer_idx}: non-finite probs, using uniform")
                    n_classes = probs.shape[1]
                    probs = np.full_like(probs, 1.0 / n_classes)
                
                # Accumulate weighted probabilities directly
                if ensemble_probs is None:
                    ensemble_probs = self.weights[layer_idx] * probs
                else:
                    ensemble_probs += self.weights[layer_idx] * probs
                
                # Free memory after processing this layer
                del X_test_feat, probs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Renormalize
            eps = 1e-12
            ensemble_probs = np.clip(ensemble_probs, eps, 1.0)
            ensemble_probs /= ensemble_probs.sum(axis=1, keepdims=True)
            
            logger.info(f"🐌 Legacy prediction completed in {time.time()-t0:.2f}s")
            return ensemble_probs


# ============================================================================
# DETERMINISTIC SETUP
# ============================================================================

def set_all_seeds(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.no_grad()
def quick_test_accuracy(model, X_test: np.ndarray, y_test: np.ndarray, device: str, batch_size: int = 256) -> float:
    """Quick accuracy check on test set to determine if model is worth calibrating."""
    mdl = model.to(device).eval() if hasattr(model, 'to') else model
    dl = DataLoader(TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test)),
                    batch_size=batch_size, shuffle=False)
    correct, total = 0, 0
    for xb, yb in dl:
        xb = xb.to(device, non_blocking=True)
        logits = mdl(xb)            # assumes get_data_loaders() already applied normalization/transforms
        pred = logits.argmax(dim=1).cpu()
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(1, total)


# ============================================================================
# METRIC PICK EVALUATION ON THE TEST SET (+ DIAGNOSTICS)
# ============================================================================

def _safe_unique_wms(default_wms: List[str], cli_wms: Optional[List[str]]):
    if cli_wms is None or len(cli_wms) == 0:
        return default_wms
    # keep order but dedupe
    seen, out = set(), []
    for w in cli_wms:
        if w not in seen:
            out.append(w)
            seen.add(w)
    return out


def evaluate_metric_picks_on_test(
    *,
    metrics_doc: dict,
    model_adapter,
    train_raw: np.ndarray, train_labels: np.ndarray,
    val_raw:   np.ndarray, val_labels:   np.ndarray,
    test_raw:  np.ndarray, test_labels:  np.ndarray,
    feature_cache: FeatureCache,
    layer_probs: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    device: str = "cuda",
    batch_size: int = 128,
    Ks: List[int] = (1, 3, 5),
    weighting_methods: Optional[List[str]] = None,
    depth_alpha: float = 1.5,
    learned_loss: str = "brier",
    learned_lr: float = 0.1,
    learned_max_iter: int = 500,
    learned_use_lbfgs: bool = False,
    seed: int = 0,
    n_bins_ece: int = 15,
) -> Dict[str, Any]:
    """
    De-duplicating, canonical-depth evaluator:
      • Canonicalize layer-sets by true depth
      • Build each (layers,K,wm) UNIQUE config ONCE
      • Share results across all metrics requesting the same config
    """
    default_wms = ["learned", "uniform", "deep", "shallow"]
    weighting_methods = _safe_unique_wms(default_wms, weighting_methods)

    # --- Depth map from the actual model (one tiny forward on val sample)
    cand = metrics_doc.get("candidate_layers", [])
    if not cand:
        # Fallback: try keys from holdout map
        hmap = metrics_doc.get("holdout_ece_per_layer", {})
        cand = sorted(map(int, hmap.keys()))
    sample = val_raw[:1] if len(val_raw) > 0 else train_raw[:1]
    depth_map = build_depth_map_from_model(model_adapter, list(map(int, cand)), sample, device)

    # --- Stable signatures for cache keys
    mdl_sig  = model_signature(model_adapter.model)
    data_sig = arrays_signature(train_labels, val_labels, test_labels, labels_only=True)
    hp = dict(depth_alpha=float(depth_alpha), learned_loss=str(learned_loss),
              learned_lr=float(learned_lr), learned_max_iter=int(learned_max_iter),
              learned_use_lbfgs=bool(learned_use_lbfgs), n_bins=int(n_bins_ece),
              batch_size=int(batch_size))
    cache = EnsembleConfigCache(depth_map, mdl_sig, data_sig, hp)

    mtopk: Dict[str, any] = metrics_doc.get("metrics_topk", {})
    logger.info("🔍 Scanning metric selections for dedup...")

    # STEP 1: register all requests (build unique configs)
    for metric_name, entry in mtopk.items():
        if not isinstance(entry, dict):
            continue
        rank = [int(x) for x in entry.get("topk_layers", [])]
        if not rank:
            continue
        for K in Ks:
            if K > len(rank): 
                continue
            chosen = rank[:K]
            for wm in weighting_methods:
                cfg = cache.make_config(chosen, K, wm)
                cache.register(cfg, metric_name, chosen)

    stats = cache.stats()
    logger.info(f"✅ Dedup complete | total={stats['total_requests']} unique={stats['unique_configs']} speedup≈{stats['speedup_est']}")

    # STEP 2: evaluate each unique config ONCE
    uniques = cache.unique()
    logger.info(f"🚀 Evaluating {len(uniques)} unique ensembles...")

    for i, cfg in enumerate(uniques, 1):
        layers_list = list(cfg.layers)
        logger.info(f"  [{i}/{len(uniques)}] layers={layers_list} K={cfg.K} wm={cfg.weighting}")
        try:
            ens = MultiLayerEnsemble(model_adapter, device=device)
            ens.depth_map = depth_map                  # ← order-invariant deep/shallow
            ens.depth_alpha = depth_alpha
            ens.learned_lr = learned_lr
            ens.learned_max_iter = learned_max_iter
            ens.learned_use_lbfgs = learned_use_lbfgs
            ens.seed = seed

            if layer_probs is not None:
                # Use fast cached probability path
                ens.fit_with_cached_probs(
                    layers_list, val_labels, layer_probs,
                    weighting_method=cfg.weighting, learned_loss=learned_loss
                )
            else:
                # Fallback to legacy path
                ens.fit(
                    layers_list,
                    train_raw, train_labels,
                    val_raw,   val_labels,
                    feature_cache,
                    batch_size=batch_size,
                    weighting_method=cfg.weighting,
                    learned_loss=learned_loss,
                )

            probs = ens.predict(test_raw, feature_cache, batch_size=batch_size, split_name="test")
            ece   = float(compute_ece(probs, test_labels, n_bins=n_bins_ece))
            acc   = float(np.mean(np.argmax(probs, axis=1) == test_labels))
            oh    = np.eye(probs.shape[1])[test_labels]
            brier = float(np.mean(np.sum((probs - oh) ** 2, axis=1)))
            result = {
                "selected_layers": list(map(int, layers_list)),
                "weights": {int(k): float(v) for k, v in ens.weights.items()},
                "test_ece": ece,
                "test_accuracy": acc,
                "test_brier": brier,
                "test_error_auroc": float(compute_error_detection_auroc(probs, test_labels)),
                "test_aurc": float(compute_aurc(probs, test_labels)),
            }
            cache.put(cfg, result)
        except Exception as e:
            logger.warning(f"  ❌ Failed: {e}")
            cache.put(cfg, {"error": str(e), "selected_layers": layers_list})
        finally:
            # Free memory after processing this config
            del ens
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # STEP 3: map cached results back to all metrics
    by_metric: Dict[str, any] = {}
    for metric_name, entry in mtopk.items():
        if not isinstance(entry, dict):
            continue
        rank = [int(x) for x in entry.get("topk_layers", [])]
        if not rank:
            continue
        metric_bucket = {}
        for K in Ks:
            if K > len(rank):
                continue
            chosen = rank[:K]
            per_wm = {}
            for wm in weighting_methods:
                cfg = cache.make_config(chosen, K, wm)
                per_wm[wm] = cache.get(cfg) or {"error": "not found"}
            metric_bucket[f"K{K}"] = per_wm
        if metric_bucket:
            by_metric[metric_name] = metric_bucket

    # Small leaderboard (best ECE per K,wm)
    best_per_k: Dict[str, any] = {}
    for K in Ks:
        kk = f"K{K}"
        best_per_k[kk] = {}
        for wm in weighting_methods:
            best = None
            for m, mres in by_metric.items():
                r = mres.get(kk, {}).get(wm, {})
                if "test_ece" in r:
                    e = r["test_ece"]
                    if best is None or e < best[1]:
                        best = (m, e)
            if best is not None:
                best_per_k[kk][wm] = {"best_metric": best[0], "best_test_ece": float(best[1])}

    payload = {"by_metric": by_metric, "best_per_k": best_per_k, "dedup": cache.stats()}
    return format_floats_7sig(payload)


def metric_signal_strength_diagnostics(metrics_doc: dict) -> Dict[str, Any]:
    """
    Lightweight diagnostics to understand selector quality.
    Produces for each metric:
      - Spearman rho between metric values and holdout ECE (val-B). (Higher |rho| = stronger signal)
      - Inferred direction ('minimize' if rho>0 else 'maximize', mirroring your logic)
      - Gap of the metric-chosen Top-1 from the best holdout (already in doc, copied here)
    Also returns a Jaccard Top-K overlap matrix (K=5 by default) and a depth bias
    indicator (mean rank of Top-K within the candidate layer order, normalized 0..1).
    """
    all_vals = metrics_doc.get("all_metric_values", {})
    holdout_map = metrics_doc.get("holdout_ece_per_layer", {})
    candidates = [int(i) for i in metrics_doc.get("candidate_layers", [])]
    if not candidates:
        candidates = sorted(map(int, holdout_map.keys()))

    # align helper
    def _align_vecs(metric_dict: Dict[str, float]):
        xs, ys = [], []
        for L in candidates:
            k = str(int(L))
            if k in metric_dict and k in holdout_map:
                xs.append(float(metric_dict[k]))
                ys.append(float(holdout_map[k]))
        return np.array(xs, dtype=float), np.array(ys, dtype=float)

    def _spearman(x, y) -> float:
        # ranks then Pearson
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        if rx.std() == 0 or ry.std() == 0:
            return float("nan")
        return float(np.corrcoef(rx, ry)[0, 1])

    # per-metric stats
    per_metric = {}
    for mname, mdict in all_vals.items():
        x, y = _align_vecs(mdict)
        rho = _spearman(x, y) if (len(x) >= 3 and len(y) >= 3) else float("nan")
        direction = "minimize" if (np.isfinite(rho) and rho > 0) else "maximize"
        gap = None
        if mname in metrics_doc.get("metrics_topk", {}):
            gap = float(metrics_doc["metrics_topk"][mname].get("holdout_ece_gap_from_best", float("nan")))
        per_metric[mname] = {
            "spearman_rho_vs_holdout_ece": float(rho) if np.isfinite(rho) else None,
            "inferred_direction": direction,
            "gap_from_best_holdout_top1": gap,
        }

    # Jaccard overlap of Top-K (K=5)
    K = 5
    tops = {}
    for mname, entry in metrics_doc.get("metrics_topk", {}).items():
        tops[mname] = set(map(int, entry.get("topk_layers", [])[:K]))

    metrics = list(tops.keys())
    jaccard = {}
    for i, a in enumerate(metrics):
        row = {}
        for j, b in enumerate(metrics):
            if i > j:
                continue
            A, B = tops[a], tops[b]
            inter = len(A & B)
            union = max(1, len(A | B))
            row[b] = float(inter / union)
        jaccard[a] = row

    # depth bias: mean normalized rank in candidate list
    idx_map = {L: i for i, L in enumerate(sorted(candidates))}
    depth_bias = {}
    for mname, S in tops.items():
        if not S:
            continue
        ranks = [idx_map.get(int(L), 0) for L in S]
        mean_rank = float(np.mean(ranks)) if ranks else float("nan")
        depth_bias[mname] = {
            "mean_rank_index": mean_rank,
            "normalized_0_shallow_1_deep": float(mean_rank / max(1, len(candidates) - 1)),
        }

    out = {
        "per_metric": format_floats_7sig(per_metric),
        "jaccard_topK5_upper_triangle": format_floats_7sig(jaccard),
        "depth_bias_topK5": format_floats_7sig(depth_bias),
    }
    return out


def save_metric_pick_test_eval_and_diagnostics(
    *,
    output_dir: str,
    metrics_doc: dict,
    model_adapter,
    train_raw, train_labels,
    val_raw,   val_labels,
    test_raw,  test_labels,
    feature_cache: FeatureCache,
    layer_probs: Optional[Dict[int, Dict[str, np.ndarray]]] = None,
    device: str,
    batch_size: int,
    Ks: List[int],
    weighting_methods: Optional[List[str]],
    depth_alpha: float,
    learned_loss: str,
    learned_lr: float,
    learned_max_iter: int,
    learned_use_lbfgs: bool,
    seed: int,
    n_bins_ece: int = 15,
) -> Dict[str, Any]:
    """
    Convenience wrapper: runs test evaluation + diagnostics and writes JSON files.
    """
    os.makedirs(os.path.join(output_dir, "metric_docs"), exist_ok=True)
    eval_payload = evaluate_metric_picks_on_test(
        metrics_doc=metrics_doc,
        model_adapter=model_adapter,
        train_raw=train_raw, train_labels=train_labels,
        val_raw=val_raw,     val_labels=val_labels,
        test_raw=test_raw,   test_labels=test_labels,
        feature_cache=feature_cache,
        layer_probs=layer_probs,
        device=device,
        batch_size=batch_size,
        Ks=Ks,
        weighting_methods=weighting_methods,
        depth_alpha=depth_alpha,
        learned_loss=learned_loss,
        learned_lr=learned_lr,
        learned_max_iter=learned_max_iter,
        learned_use_lbfgs=learned_use_lbfgs,
        seed=seed,
        n_bins_ece=n_bins_ece,
    )

    diag_payload = metric_signal_strength_diagnostics(metrics_doc)

    # save
    test_json = os.path.join(output_dir, "metric_docs", "metric_picks_test_eval.json")
    diag_json = os.path.join(output_dir, "metric_docs", "metric_selector_diagnostics.json")
    with open(test_json, "w") as f:
        json.dump(format_floats_7sig(eval_payload), f, indent=2, cls=NumpyEncoder)
    with open(diag_json, "w") as f:
        json.dump(format_floats_7sig(diag_payload), f, indent=2, cls=NumpyEncoder)

    return {"test_eval": eval_payload, "diagnostics": diag_payload}


# ============================================================================
# MODEL & DATA LOADING
# ============================================================================

def load_model_and_data_like_before(
    dataset_name: str,
    model_name: str,
    training_method: str,
    results_base_dir: str,
    seed: int,
    device: str,
    batch_size: int,
    corruption_dataset_path: Optional[str] = None,
):
    """
    Mirrors the previous script's loading:
      • Builds model checkpoint path like:
        {results_base_dir}/{training_method}/{clean_dataset}/{model_name}/seed{seed}/{exp_name}/best_model.pth
      • Uses get_data_loaders(...) for clean dataset for train/val
      • For test: uses corruption dataset if dataset_name ends with 'c' and corruption_dataset_path is provided
      • Wraps model with PyTorchModelAdapter
      • Returns numpy arrays for train/val/test and num_classes
    """

    # Keep the 'clean dataset' rule you used before (cifar10-c → cifar10, cifar100c → cifar100, etc.)
    # IMPORTANT: check cifar100 BEFORE cifar10 to avoid the substring bug
    if dataset_name.startswith('cifar100'):
        dataset_clean = 'cifar100'
    elif dataset_name.startswith('cifar10'):
        dataset_clean = 'cifar10'
    else:
        dataset_clean = dataset_name

    exp_name = f"{training_method}_{dataset_clean}_{model_name}_seed{seed}"
    model_path = os.path.join(
        results_base_dir, training_method, dataset_clean,
        model_name, f"seed{seed}", exp_name, "best_model.pth"
    )

    logger.info(f"📁 Expecting model checkpoint at: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    # Data - train and val always use clean dataset
    logger.info(f"📊 Loading {dataset_clean} (train/val/test) with batch_size={batch_size}")
    train_loader, val_loader, test_loader, n_classes = get_data_loaders(dataset_clean, batch_size, seed=seed)

    train_raw, train_labels = get_all_data_as_numpy(train_loader)
    val_raw,   val_labels   = get_all_data_as_numpy(val_loader)

    # For test data: check if we should use corruption dataset
    is_corruption_dataset = dataset_name.endswith('c') and corruption_dataset_path is not None
    if is_corruption_dataset:
        logger.info(f"📊 Using corruption dataset from: {corruption_dataset_path}")
        # For corruption datasets, we'll return None for test data here
        # and handle it separately in the evaluation
        test_raw, test_labels = None, None
    else:
        test_raw,  test_labels  = get_all_data_as_numpy(test_loader)

    # Model
    logger.info(f"🤖 Loading pretrained {model_name} ({n_classes} classes)")
    model = load_trained_model(
        model_path, model_name, n_classes,
        torch.device(device), dataset=dataset_clean
    )
    model_adapter = PyTorchModelAdapter(model, torch.device(device))

    return (model_adapter,
            train_raw, train_labels,
            val_raw,   val_labels,
            test_raw,  test_labels,
            n_classes,
            dataset_clean)


def evaluate_ensemble_on_corruptions(
    ensemble,
    corruption_dataset_path: str,
    dataset_name: str,
    feature_cache: FeatureCache,
    batch_size: int = 128,
    n_bins: int = 15
) -> Dict[str, Any]:
    logger.info(f"[DEBUG] evaluate_ensemble_on_corruptions called with path: {corruption_dataset_path}")
    logger.info(f"[DEBUG] Dataset name: {dataset_name}")

    """
    Evaluate ensemble on all corruption types and severities.
    Calculate mCE (mean Corruption Error across all corruptions and severities).

    Returns:
        Dictionary with results for each corruption type, severity, and overall mCE
    """
    if dataset_name.startswith('cifar10'):
        from Data.cifar10_c import CIFAR10C, get_cifar10c_loader
        corruption_types = CIFAR10C.CORRUPTION_TYPES
    elif dataset_name.startswith('cifar100'):
        from Data.cifar100_c import CIFAR100C, get_cifar100c_loader
        corruption_types = CIFAR100C.CORRUPTION_TYPES
    else:
        raise ValueError(f"Unsupported corruption dataset: {dataset_name}")

    logger.info(f"[DEBUG] Found {len(corruption_types)} corruption types: {corruption_types}")

    logger.info(f"[DEBUG] Corruption path exists: {os.path.exists(corruption_dataset_path)}")
    if os.path.exists(corruption_dataset_path):
        try:
            logger.info(f"[DEBUG] Contents: {os.listdir(corruption_dataset_path)}")
        except Exception as e:
            logger.warning(f"[DEBUG] Failed to list contents of {corruption_dataset_path}: {e}")

    results = {}
    all_eces = []
    all_accuracies = []
    all_briers = []

    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating on {dataset_name.upper()} corruptions")
    logger.info(f"{'='*80}")

    import gc
    
    for corruption in corruption_types:
        logger.info(f"\n📊 Evaluating {corruption}...")
        corruption_results = {}

        for severity in range(1, 6):
            logger.info(f"   Severity {severity}...")

            # Load corruption data
            if dataset_name.startswith('cifar10'):
                loader = get_cifar10c_loader(
                    root=corruption_dataset_path,
                    corruption_type=corruption,
                    severity=severity,
                    batch_size=batch_size
                )
            else:  # cifar100
                from Data.cifar100_c import get_cifar100c_loader
                loader = get_cifar100c_loader(
                    root=corruption_dataset_path,
                    corruption_type=corruption,
                    severity=severity,
                    batch_size=batch_size
                )

            # Get data as numpy
            test_x, test_y = get_all_data_as_numpy(loader)

            # Predict with ensemble (legacy path: uses calibrators + feature cache)
            split_tag = f"{dataset_name}_{corruption}_s{severity}"
            ensemble_probs = ensemble.predict(
                test_x,
                feature_cache,
                batch_size=batch_size,
                split_name=split_tag
            )

            # Calculate metrics
            preds = np.argmax(ensemble_probs, axis=1)
            accuracy = (preds == test_y).mean()

            # ECE
            ece = compute_ece(ensemble_probs, test_y, n_bins=n_bins)

            # Brier score
            y_onehot = np.eye(ensemble_probs.shape[1])[test_y]
            brier = np.mean((ensemble_probs - y_onehot) ** 2)

            corruption_results[f'severity_{severity}'] = {
                'ece': float(ece),
                'accuracy': float(accuracy),
                'brier': float(brier),
                'error_auroc': float(compute_error_detection_auroc(ensemble_probs, test_y)),
                'aurc': float(compute_aurc(ensemble_probs, test_y)),
                'num_samples': int(len(test_y))
            }

            all_eces.append(ece)
            all_accuracies.append(accuracy)
            all_briers.append(brier)

            logger.info(f"      Accuracy: {accuracy*100:.2f}%, ECE: {ece:.4f}, Brier: {brier:.4f}")

            # CRITICAL: Cleanup after each severity
            del test_x, test_y, ensemble_probs, loader, preds, y_onehot
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

        # Additional cleanup after each corruption
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info(f"   Completed {corruption}, memory cleaned")

        # Mean across severities for this corruption
        mean_ece = np.mean([corruption_results[f'severity_{s}']['ece'] for s in range(1, 6)])
        mean_acc = np.mean([corruption_results[f'severity_{s}']['accuracy'] for s in range(1, 6)])
        mean_brier = np.mean([corruption_results[f'severity_{s}']['brier'] for s in range(1, 6)])

        corruption_results['mean'] = {
            'ece': float(mean_ece),
            'accuracy': float(mean_acc),
            'brier': float(mean_brier)
        }

        results[corruption] = corruption_results
        logger.info(f"   Mean: Acc={mean_acc*100:.2f}%, ECE={mean_ece:.4f}, Brier={mean_brier:.4f}")

    # Calculate mCE (mean Corruption Error) - mean ECE across ALL corruptions and severities
    mCE = float(np.mean(all_eces))
    mean_accuracy = float(np.mean(all_accuracies))
    mean_brier = float(np.mean(all_briers))

    results['mCE'] = {
        'ece': mCE,
        'accuracy': mean_accuracy,
        'brier': mean_brier,
        'num_corruptions': len(corruption_types),
        'num_severities': 5,
        'total_evaluations': len(all_eces)
    }

    logger.info(f"\n{'='*80}")
    logger.info(f"📈 Overall mCE Results:")
    logger.info(f"   mCE (mean ECE): {mCE:.4f}")
    logger.info(f"   Mean Accuracy: {mean_accuracy*100:.2f}%")
    logger.info(f"   Mean Brier: {mean_brier:.4f}")
    logger.info(f"{'='*80}\n")

    return results


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment(args):
    """
    Run multi-layer ensemble experiment.
    
    K semantics:
    - K=0: Use ALL eligible layers (after sampling/filtering), vary only weights per scheme
    - K>0: Pick top-K layers + apply weighting scheme
    """
    import time
    total_start = time.time()
    
    logger.info(f"Running script: {__file__}")
    
    # Normalize selection strategy naming from launcher
    if getattr(args, 'selection_metric_name', None) is None and getattr(args, 'selection_method', None):
        args.selection_metric_name = args.selection_method
    
    # Determine the weighting methods to use (prioritize explicit list from launcher)
    if getattr(args, 'weighting_methods', None):
        weighting_methods = list(args.weighting_methods)
        logger.info(f"Using weighting methods from --weighting-methods: {weighting_methods}")
    elif getattr(args, 'weighting_strategy', None):
        weighting_methods = [s.strip() for s in str(args.weighting_strategy).split(',') if s.strip()]
        logger.warning(f"Using weighting methods derived from --weighting-strategy: {weighting_methods}. Prefer --weighting-methods.")
    else:
        weighting_methods = [args.ensemble_weights]
        logger.info(f"Using default weighting method: {weighting_methods}")

    # Ensure it's a list, dedupe while preserving order
    if not isinstance(weighting_methods, list):
        weighting_methods = [weighting_methods]
    _seen_wm = set()
    weighting_methods = [wm for wm in weighting_methods if not (wm in _seen_wm or _seen_wm.add(wm))]
    logger.info(f"Final weighting methods to run: {weighting_methods}")

    # Parse initial weights if provided
    initial_weights = None
    if getattr(args, 'initial_weights', None):
        try:
            initial_weights = [float(w.strip()) for w in args.initial_weights.split(',')]
            logger.info(f"Parsed initial weights: {initial_weights}")
        except Exception as e:
            logger.error(f"Failed to parse initial_weights '{args.initial_weights}': {e}")
            raise ValueError(f"Invalid initial_weights format: {args.initial_weights}") from e

    # Handle multiple alphas
    if args.depth_alphas is None:
        args.depth_alphas = [args.depth_alpha]
    logger.info(f"Depth alphas: {args.depth_alphas}")
    
    set_all_seeds(args.seed)
    logger.info(f"Seed set to {args.seed}")
    
    # Use the exact output directory passed by the launcher or fallback
    if getattr(args, 'output_dir', None):
        base_dir = args.output_dir
    elif getattr(args, 'output_base_dir', None):
        logger.warning("Using legacy --output-base-dir. --output-dir is preferred.")
        run_folder_name = getattr(args, 'selection_metric_name', 'multi_layer_ensemble')
        if getattr(args, 'selection_metric_name', None):
            run_folder_name = os.path.join("hybrid_ensembles", args.selection_metric_name)
        base_dir = os.path.join(
            args.output_base_dir, args.training_method, args.dataset,
            args.model_name, f"seed{args.seed}", run_folder_name
        )
    else:
        raise ValueError("Must provide either --output-dir (preferred) or --output-base-dir")

    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    # folders reused across runs
    metric_docs_dir = base_dir / "metric_docs"
    baselines_dir   = base_dir / "baselines"
    metric_docs_dir.mkdir(exist_ok=True)
    baselines_dir.mkdir(exist_ok=True)

    disable_cache = getattr(args, 'disable_cache', False)
    if disable_cache:
        cache_dir = None
        logger.info("Cache: disabled (in-memory only; expect slower runs).")
    else:
        cache_dir = Path(args.feat_cache) if args.feat_cache else (base_dir / "feat_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Cache: {cache_dir}")

    feature_cache = make_feature_cache(cache_dir, disable_cache)

    # No per-weighting subdirectories; all outputs go under base_dir
    
    logger.info(f"Base output: {base_dir}")
    
    # --- Load model & data exactly like the previous script ---
    logger.info("Loading data and model...")
    t0 = time.time()
    (model_adapter,
     train_raw, train_labels,
     val_raw,   val_labels,
     test_raw,  test_labels,
     num_classes,
     dataset_clean) = load_model_and_data_like_before(
        dataset_name=args.dataset,
        model_name=args.model_name,
        training_method=args.training_method,
        results_base_dir=args.results_base_dir,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        corruption_dataset_path=getattr(args, 'corruption_dataset_path', None),
    )
    logger.info(f"⏱️  Data/model loading took {time.time()-t0:.2f}s")

    # ============================================================================
    # Compression Ratio Auto-Selection
    # ============================================================================
    compression_ratio = getattr(args, 'compression_ratio', None)
    if compression_ratio is None:
        # Auto-select based on dataset
        if 'cifar100' in args.dataset.lower():
            compression_ratio = 32.0
            logger.info(f"🔧 Auto-selected compression ratio: {compression_ratio}x (CIFAR-100 dataset)")
        else:
            compression_ratio = 16.0
            logger.info(f"🔧 Auto-selected compression ratio: {compression_ratio}x (default for {args.dataset})")
    elif compression_ratio <= 0:
        # Use fixed feature dimension instead (backward compatibility)
        compression_ratio = None
        logger.info("🔧 Using fixed feature dimension: 512 (compression ratio disabled)")
    else:
        logger.info(f"🔧 Using explicit compression ratio: {compression_ratio}x")

    # Check if we're using corruption dataset
    is_corruption_dataset = args.dataset.endswith('c') and getattr(args, 'corruption_dataset_path', None) is not None
    logger.info(f"[DEBUG] Corruption dataset mode: {is_corruption_dataset}")

    # --- Pre-flight accuracy check ---
    if is_corruption_dataset:
        logger.info("Skipping pre-flight accuracy check for corruption dataset (will evaluate on corruptions)")
        raw_test_acc = 1.0  # Dummy value to pass the check
    else:
        raw_test_acc = quick_test_accuracy(model_adapter.model, test_raw, test_labels, args.device, max(128, args.batch_size))
        logger.info(f"Raw model test accuracy: {raw_test_acc:.4f}")

    if raw_test_acc < args.min_raw_acc and not is_corruption_dataset:
        # make a tiny JSON so the runner can detect this was intentionally skipped
        results_json = os.path.join(base_dir, "ensemble_results.json")
        payload = {
            "skipped_low_acc": True,
            "raw_test_acc": float(raw_test_acc),
            "threshold": float(args.min_raw_acc),
            "reason": "Raw model accuracy below threshold; skipping ensemble calibration."
        }
        with open(results_json, "w") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"⏭️  Skipped: raw acc {raw_test_acc:.4f} < {args.min_raw_acc:.2f}. Wrote {results_json}")
        return payload
    
    # ============================================================================
    # EARLY HYBRID MODE DETECTION + FAST PATH
    # ============================================================================
    hybrid_mode = getattr(args, 'layer_indices', None) is not None
    if hybrid_mode:
        logger.info("\n" + "="*80)
        logger.info("🚀 HYBRID MODE DETECTED - FAST PATH ACTIVATED")
        logger.info("="*80)
        provided_layers = [int(L.strip()) for L in str(args.layer_indices).split(',') if L.strip()]
        logger.info(f"Using {len(provided_layers)} provided layers: {provided_layers}")
        if not provided_layers:
            raise ValueError("--layer_indices provided but empty after parsing")
        if any(L < 0 for L in provided_layers):
            raise ValueError(f"Invalid negative layer index in: {provided_layers}")
        candidate_layers = provided_layers
        skip_expensive_analysis = True
        logger.info("⚡ Skipping layer discovery (using provided indices)")
        logger.info("⚡ Will skip metrics documentation, baselines, and oracle computation")
    else:
        logger.info("\n" + "="*80)
        logger.info("📊 BATCH MODE - FULL ANALYSIS")
        logger.info("="*80)
        skip_expensive_analysis = False
        # If you want discovery to use the true input size, keep CLI overrides
        input_shape = (1, 3, args.input_h, args.input_w)
        logger.info("Discovering model layers for candidate selection...")
        t0 = time.time()
        candidate_layers = get_candidate_layers(
            model_adapter.model,
            args.model_name,
            sample_every=args.sample_every,
            device=args.device,
            input_shape=input_shape,
        )
        logger.info(f"⏱️  Layer discovery took {time.time()-t0:.2f}s")
        logger.info(f"Candidate layers: {candidate_layers}")
    
    # Canonical depth map (one tiny forward on a sample)
    depth_map = build_depth_map_from_model(
        model_adapter=model_adapter,
        candidate_layers=list(map(int, candidate_layers)),
        sample_np=val_raw[:1] if len(val_raw) > 0 else train_raw[:1],
        device=args.device
    )
    
    # ============================================================================
    # PRECOMPUTE LAYER PROBABILITIES (MAJOR OPTIMIZATION) - CLEAN DATA ONLY
    # ============================================================================
    if is_corruption_dataset:
        logger.info("Corruption dataset detected – skipping layer probability precomputation (will use legacy calibrators).")
        layer_probs = None
    else:
        probs_cache = make_probs_cache(cache_dir, disable_cache)
        
        # Check if we already have all probabilities cached
        t0 = time.time()
        if probs_cache.has_all_layers('val', candidate_layers) and probs_cache.has_all_layers('test', candidate_layers):
            logger.info("✅ All layer probabilities already cached - skipping precomputation")
            layer_probs = {}
            for L in candidate_layers:
                layer_probs[L] = {
                    'val': np.array(probs_cache.get('val', L), dtype=np.float32),
                    'test': np.array(probs_cache.get('test', L), dtype=np.float32)
                }
            logger.info(f"⏱️  Loading cached probabilities took {time.time()-t0:.2f}s")
        else:
            logger.info("🚀 Precomputing layer probabilities (this is the expensive part)...")
            layer_probs = precompute_layer_probabilities(
                model_adapter=model_adapter,
                candidate_layers=candidate_layers,
                train_raw=train_raw, train_labels=train_labels,
                val_raw=val_raw, val_labels=val_labels,
                test_raw=test_raw, test_labels=test_labels,
                feature_cache=feature_cache,
                probs_cache=probs_cache,
                device=args.device,
                batch_size=args.batch_size,
                learned_loss=args.learned_loss
            )
            logger.info(f"⏱️  Probability precomputation took {time.time()-t0:.2f}s")
    
    # --- Oracle / per-layer ground-truth (from cached probabilities)
    per_layer_json = os.path.join(metric_docs_dir, "per_layer_ground_truth.json")
    oracle_json    = os.path.join(metric_docs_dir, "oracle_best_layer.json")

    if skip_expensive_analysis:
        logger.info("⏭️  Skipping oracle computation (hybrid mode)")
        per_layer_results = None
    else:
        need_oracle = (not os.path.isfile(per_layer_json)) or (os.path.getsize(per_layer_json) == 0)
        if need_oracle:
            logger.info("\n" + "="*80)
            logger.info("COMPUTING ORACLE FROM CACHED PROBABILITIES")
            logger.info("="*80)
            try:
                per_layer_results = _oracle_from_cached(layer_probs, test_labels, n_bins=args.bins)
                with open(per_layer_json, "w") as f:
                    json.dump(format_floats_7sig(per_layer_results), f, indent=2, cls=NumpyEncoder)

                # extract best-by-ECE on TEST
                valid = [r for r in per_layer_results if np.isfinite(r.get("test_ece", np.inf))]
                if valid:
                    best = min(valid, key=lambda x: x["test_ece"])
                    oracle_payload = {
                        "layer_idx": int(best["layer_idx"]),
                        "test_ece": float(best["test_ece"]),
                        "test_accuracy": float(best.get("test_accuracy", float("nan"))),
                        "test_brier": float(best.get("test_brier", float("nan")))
                    }
                    with open(oracle_json, "w") as f:
                        json.dump(format_floats_7sig(oracle_payload), f, indent=2, cls=NumpyEncoder)
                    logger.info(f"✅ Oracle files written: {per_layer_json} ; {oracle_json}")
                    
                    # Also save a tiny CSV for quick diffs
                    import csv
                    csv_path = os.path.join(metric_docs_dir, "per_layer_ece.csv")
                    with open(csv_path, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["layer_idx", "test_ece", "test_accuracy", "test_brier"])
                        for r in per_layer_results:
                            if np.isfinite(r.get("test_ece", np.inf)):
                                w.writerow([
                                    int(r["layer_idx"]),
                                    float(r["test_ece"]),
                                    float(r.get("test_accuracy", float("nan"))),
                                    float(r.get("test_brier", float("nan")))
                                ])
                    logger.info(f"📄 Per-layer CSV: {csv_path}")
                else:
                    logger.warning("No valid per-layer entries to define an oracle.")
            except Exception as e:
                logger.warning(f"Oracle-from-cache failed: {e}")
                per_layer_results = None
        else:
            per_layer_results = None
    
    # ============================================================================
    # METRICS DOCUMENTATION (optional)
    # ============================================================================
    metrics_doc = None

    if skip_expensive_analysis:
        logger.info("⏭️  Skipping metrics documentation (hybrid mode)")
        if args.metrics_doc_dir:
            md_path = os.path.join(args.metrics_doc_dir, "metric_docs_and_topk.json")
            if os.path.isfile(md_path):
                logger.info(f"📂 Loading pre-computed metric docs from: {md_path}")
                with open(md_path, "r") as f:
                    metrics_doc = json.load(f)
    else:
        # First check if we should load pre-computed metrics
        if args.metrics_doc_dir:
            md_path = os.path.join(args.metrics_doc_dir, "metric_docs_and_topk.json")
            if os.path.isfile(md_path):
                logger.info(f"📂 Loading pre-computed metric docs from: {md_path}")
                with open(md_path, "r") as f:
                    metrics_doc = json.load(f)
            else:
                logger.warning(f"--metrics-doc-dir specified but file not found: {md_path}")
        
        # If not loaded and docs requested, compute them
        if metrics_doc is None and args.doc_metrics and run_metrics_doc_topk is not None:
            logger.info("\n" + "="*80)
            logger.info("DOCUMENTING LAYER SELECTION METRICS")
            logger.info("="*80)
            metrics_out = metric_docs_dir
            try:
                metrics_doc = run_metrics_doc_topk(
                    model=model_adapter.model,
                    model_adapter=model_adapter,
                    candidate_layers=candidate_layers,
                    train_raw=train_raw, train_labels=train_labels,
                    val_raw=val_raw,   val_labels=val_labels,
                    device=args.device, batch_size=args.batch_size,
                    topk=args.metrics_topk, val_split=args.val_split,
                    seed=args.seed, out_dir=metrics_out,
                    try_both=args.doc_metrics_both
                )
            except Exception as e:
                logger.warning(f"Metrics documentation failed: {e}")
                import traceback
                logger.debug(traceback.format_exc())
    
    # ============================================================================
    # NEW: Evaluate metric-selected Top-K ensembles on the TEST set
    # ============================================================================
    if skip_expensive_analysis:
        logger.info("⏭️  Skipping metric picks test evaluation (hybrid mode)")
        metric_picks_test_best = None
    elif args.eval_metric_picks_on_test:
        # if we didn't just compute metrics_doc, try to load it
        if metrics_doc is None:
            md_path = os.path.join(metric_docs_dir, "metric_docs_and_topk.json")
            if os.path.isfile(md_path):
                logger.info(f"Loading metrics_doc from {md_path}")
                with open(md_path, "r") as f:
                    metrics_doc = json.load(f)
            else:
                logger.warning("Requested --eval-metric-picks-on-test but metrics_doc is missing; skipping.")
                metrics_doc = None

        if metrics_doc is not None:
            logger.info("\n" + "="*80)
            logger.info("EVALUATING METRIC PICKS ON TEST SET")
            logger.info("="*80)
            
            Ks_for_metric_eval = args.metric_pick_ks if (args.metric_pick_ks and len(args.metric_pick_ks) > 0) else (args.ensemble_test_ks or [1, 3, 5])
            test_eval_bundle = save_metric_pick_test_eval_and_diagnostics(
                output_dir=base_dir,
                metrics_doc=metrics_doc,
                model_adapter=model_adapter,
                train_raw=train_raw, train_labels=train_labels,
                val_raw=val_raw,     val_labels=val_labels,
                test_raw=test_raw,   test_labels=test_labels,
                feature_cache=feature_cache,
                layer_probs=layer_probs,
                device=args.device,
                batch_size=args.batch_size,
                Ks=Ks_for_metric_eval,
                weighting_methods=args.metric_pick_weighting_methods,
                depth_alpha=args.depth_alpha,
                learned_loss=args.learned_loss,
                learned_lr=args.learned_lr,
                learned_max_iter=args.learned_max_iter,
                learned_use_lbfgs=args.learned_use_lbfgs,
                seed=args.seed,
                n_bins_ece=args.bins,
            )
            logger.info("✅ Metric-pick TEST evaluation written to metric_docs/metric_picks_test_eval.json")
            logger.info("✅ Metric selector diagnostics written to metric_docs/metric_selector_diagnostics.json")
            
            # Log summary of best results
            best = test_eval_bundle['test_eval'].get('best_per_k', {})
            if best:
                logger.info("\n📊 Best Metric Picks on TEST (per K and weighting):")
                for kkey in sorted(best.keys()):
                    logger.info(f"  {kkey}:")
                    for wm, info in sorted(best[kkey].items()):
                        m = info.get('best_metric', 'N/A')
                        e = info.get('best_test_ece', float('inf'))
                        logger.info(f"    {wm:10s}: {m:40s} ECE={e:.7g}")
            
            # Store for later inclusion in results
            metric_picks_test_best = test_eval_bundle['test_eval'].get('best_per_k', {})
        else:
            metric_picks_test_best = None
    else:
        metric_picks_test_best = None
    
    # ============================================================================
    # METRIC-BASED ENSEMBLE EVALUATION (val-B + test, all weighting methods)
    # ============================================================================
    topk_ensemble_results = None
    if skip_expensive_analysis:
        logger.info("⏭️  Skipping metric-based ensemble evaluation (hybrid mode)")
    elif args.eval_topk_ensembles and metrics_doc is not None:
        logger.info("\n" + "="*80)
        logger.info("EVALUATING METRIC-BASED TOP-K ENSEMBLES (val-B + test)")
        logger.info("="*80)

        from Experiments.metrics_topk_doc import split_validation
        (valA_x, valA_y), (valB_x, valB_y) = split_validation(
            val_raw, val_labels, split=args.val_split, seed=args.seed
        )
        logger.info(f"Selecting layers from metrics (Val-A); fitting calibrators+weights on Val-B ({len(valB_x)}); evaluating on TEST ({len(test_raw)}).")

        mtopk = metrics_doc.get("metrics_topk", {})
        test_ks = args.ensemble_test_ks
        weightings = args.eval_topk_weighting_methods
        alpha = args.depth_alpha

        topk_ensemble_results = {}
        logger.info(f"Testing K values: {test_ks}  | weightings: {weightings}")
        logger.info(f"Evaluating {len(mtopk)} metrics...")

        # Setup dedup cache for valB+test ensembles
        hp2 = dict(depth_alpha=float(alpha), n_bins=int(args.bins), batch_size=int(args.batch_size))
        cache2 = EnsembleConfigCache(depth_map, model_signature(model_adapter.model),
                                     arrays_signature(valA_y, valB_y, test_labels, labels_only=True), hp2)

        for metric_name, entry in tqdm(mtopk.items(), desc="Evaluating metric ensembles"):
            if not isinstance(entry, dict):
                continue

            rank = entry.get("topk_layers", [])
            if not rank:
                continue

            metric_results = {}
            for K in test_ks:
                if K > len(rank):
                    continue
                chosen = rank[:K]
                metric_results[f"K{K}"] = {}

                for wm in weightings:
                    ens = None  # Initialize for cleanup
                    try:
                        cfg = cache2.make_config(chosen, K, wm)
                        out = cache2.get(cfg)
                        if out is None:
                            ens = MultiLayerEnsemble(model_adapter, device=args.device)
                            ens.depth_map = depth_map
                            ens.seed = args.seed
                            ens.depth_alpha = alpha

                            # IMPORTANT: fit on Val-B to avoid leakage from Val-A selection
                            # Use a separate cache root so features don't collide with other runs
                            valB_cache = make_feature_cache(cache_dir, disable_cache, subdir="valB_ctx")
                            ens.fit(
                                chosen,
                                train_raw, train_labels,
                                valB_x,   valB_y,
                                valB_cache,
                                args.batch_size,
                                weighting_method=wm,
                                learned_loss="brier"
                            )

                            # Evaluate on val-B
                            probsB = ens.predict(valB_x, valB_cache, args.batch_size, split_name="valB")
                            eceB = compute_ece(probsB, valB_y, n_bins=args.bins)
                            accB = float(np.mean(np.argmax(probsB, axis=1) == valB_y))
                            yB_oh = np.eye(probsB.shape[1])[valB_y]
                            brierB = float(np.mean(np.sum((probsB - yB_oh)**2, axis=1)))

                            # Evaluate on TEST
                            probsT = ens.predict(test_raw, valB_cache, args.batch_size, split_name="test")
                            eceT = compute_ece(probsT, test_labels, n_bins=args.bins)
                            accT = float(np.mean(np.argmax(probsT, axis=1) == test_labels))
                            yT_oh = np.eye(probsT.shape[1])[test_labels]
                            brierT = float(np.mean(np.sum((probsT - yT_oh)**2, axis=1)))

                            out = {
                                "chosen_layers": chosen,
                                "chosen_layers_canonical": list(canonical_layers(chosen, depth_map)),
                                "weights": format_floats_7sig(ens.weights),
                                "valB": {"ece": float(eceB), "accuracy": float(accB), "brier": float(brierB), "error_auroc": float(compute_error_detection_auroc(probsB, valB_y)), "aurc": float(compute_aurc(probsB, valB_y))},
                                "test": {"ece": float(eceT), "accuracy": float(accT), "brier": float(brierT), "error_auroc": float(compute_error_detection_auroc(probsT, test_labels)), "aurc": float(compute_aurc(probsT, test_labels))},
                            }
                            cache2.put(cfg, out)

                            if args.save_probs_metric_eval:
                                base = os.path.join(output_dir, "metric_docs", f"{metric_name}_K{K}_{wm}")
                                np.save(base + "_valB_probs.npy", probsB)
                                np.save(base + "_test_probs.npy", probsT)

                        metric_results[f"K{K}"][wm] = out

                    except Exception as e:
                        logger.warning(f"  {metric_name} K={K} {wm} failed: {e}")
                        metric_results[f"K{K}"][wm] = {"error": str(e)}
                    finally:
                        # Free memory after processing this config
                        if ens is not None:
                            del ens
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

            if metric_results:
                topk_ensemble_results[metric_name] = metric_results

        # Leaderboards: best metric per K and weighting, on val-B and on test
        leaderboards = {"valB": {}, "test": {}}
        for split_name in ["valB", "test"]:
            for K in test_ks:
                kkey = f"K{K}"
                leaderboards[split_name][kkey] = {}
                for wm in weightings:
                    winners = []
                    for metric_name, kdict in topk_ensemble_results.items():
                        if kkey in kdict and wm in kdict[kkey] and split_name in kdict[kkey][wm] \
                           and "ece" in kdict[kkey][wm][split_name]:
                            winners.append((metric_name, kdict[kkey][wm][split_name]["ece"]))
                    if winners:
                        best_metric, best_ece = min(winners, key=lambda x: x[1])
                        leaderboards[split_name][kkey][wm] = {
                            "best_metric": best_metric,
                            "best_ece": float(best_ece)
                        }

        topk_ensemble_results["_leaderboards"] = leaderboards
        topk_ensemble_results["_dedup_stats"] = cache2.stats()

        topk_ens_json = os.path.join(metric_docs_dir, "topk_ensemble_eval.json")
        with open(topk_ens_json, "w") as f:
            json.dump(format_floats_7sig(topk_ensemble_results), f, indent=2, cls=NumpyEncoder)
        
        stats2 = cache2.stats()
        logger.info(f"✅ Metric-based ensemble (valB+test) saved: {topk_ens_json}")
        logger.info(f"   Dedup: {stats2['unique_configs']}/{stats2['total_requests']} unique configs (speedup≈{stats2['speedup_est']})")
        
        # Optional: flatten to CSV for quick plotting
        import csv
        flat_csv = os.path.join(metric_docs_dir, "topk_ensemble_eval_flat.csv")
        with open(flat_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric","K","weighting","valB_ece","test_ece","layers","weights"])
            for metric, kdict in topk_ensemble_results.items():
                if metric.startswith("_"): continue
                for kkey, mdict in kdict.items():
                    for wm, res in mdict.items():
                        if "valB" in res and "test" in res:
                            w.writerow([
                                metric, kkey, wm,
                                res["valB"]["ece"], res["test"]["ece"],
                                "|".join(map(str, res["chosen_layers"])),
                                json.dumps(res["weights"])
                            ])
        logger.info(f"📄 Flattened CSV: {flat_csv}")
    
    # ============================================================================
    # BASELINE EVALUATION (optional)
    # ============================================================================
    baseline_results = None
    if skip_expensive_analysis:
        logger.info("⏭️  Skipping baseline evaluation (hybrid mode)")
    elif args.eval_baselines and run_standard_baselines is not None:
        logger.info("\n" + "="*80)
        logger.info("RUNNING BASELINE CALIBRATION METHODS")
        logger.info("="*80)
        base_out = baselines_dir
        base_json = os.path.join(base_out, "baseline_results.json")

        try:
            if os.path.isfile(base_json) and os.path.getsize(base_json) > 0:
                logger.info(f"📄 Loading existing baseline results: {base_json}")
                with open(base_json, "r") as f:
                    baseline_results = json.load(f)
            else:
                logger.info("🚀 Computing baselines...")
                baseline_results = run_standard_baselines(
                    model_adapter=model_adapter,
                    train_raw=train_raw, train_labels=train_labels,
                    val_raw=val_raw,     val_labels=val_labels,
                    test_raw=test_raw,   test_labels=test_labels,
                    batch_size=args.batch_size,
                    output_dir=base_out,
                    seed=args.seed,
                )
                # Save with 7-digit formatting
                with open(base_json, "w") as f:
                    json.dump(format_floats_7sig(baseline_results), f, indent=2, cls=NumpyEncoder)

            # Log tidy summary
            def _num7(x):
                return f"{x:.7g}" if isinstance(x, (int, float, np.floating)) else "N/A"
            logger.info("✅ Baseline evaluation complete")
            for method, metrics in baseline_results.items():
                if isinstance(metrics, dict) and 'ece' in metrics:
                    ece_str = _num7(metrics.get('ece'))
                    acc_str = _num7(metrics.get('acc'))
                    logger.info(f"  {method:25s}: ECE={ece_str}, Acc={acc_str}")

        except Exception as e:
            logger.warning(f"Baseline evaluation failed: {e}")
            baseline_results = {"error": str(e)}
    
    # Per-layer evaluation is now done earlier for oracle computation
    # (Probability precomputation moved up earlier in the function)
    
    # ============================================================================
    # MULTI-LAYER ENSEMBLE EVALUATION
    # ============================================================================
    results = {}
    # SINGLE HYBRID MODE: if layer indices are provided, run exactly once with those layers
    if getattr(args, 'layer_indices', None) is not None:
        logger.info(f"Running in SINGLE HYBRID mode for {getattr(args, 'selection_metric_name', 'custom_layers')}")
        k_values = [-1]  # dummy loop value
        provided_layers = [int(L.strip()) for L in str(args.layer_indices).split(',') if L.strip() != '']
    else:
        logger.info("Running in BATCH (topk) mode")
        k_values = args.topk if isinstance(args.topk, list) else [args.topk]
        provided_layers = None
    # Use resolved weighting_methods from earlier
    
    for k in k_values:
        if provided_layers is not None:
            selected_layers = provided_layers
            k = len(selected_layers)
            logger.info(f"Using provided {k} layers: {selected_layers}")
        else:
            logger.info(f"\n{'='*80}")
            if k == 0:
                logger.info(f"EVALUATING ALL-LAYER ENSEMBLE (K=0, using all {len(candidate_layers)} sampled layers)")
            elif k >= len(candidate_layers):
                logger.info(f"EVALUATING ALL-LAYER ENSEMBLE (K={k} >= {len(candidate_layers)}, using all sampled layers)")
            else:
                logger.info(f"EVALUATING TOP-{k} ENSEMBLE")
            logger.info(f"{'='*80}")
            
            # Select layers
            selected_layers = pick_topk_layers_simple(candidate_layers, k)
            logger.info(f"Selected {len(selected_layers)} layers: {selected_layers}")
        
        # Evaluate with different weighting methods
        for wm in weighting_methods:
            logger.info(f"\n--- Weighting method: {wm} ---")
            
            # Determine which alphas to run for this weighting method
            alphas_for_wm = args.depth_alphas if wm in ('deep', 'shallow') else [args.depth_alphas[0]]
            
            # Track if we've already computed learned/uniform (they're alpha-agnostic)
            already_done_this_wm = False
            
            for alpha in alphas_for_wm:
                # Skip recomputing learned/uniform for subsequent alphas
                if wm not in ('deep', 'shallow') and already_done_this_wm:
                    logger.info(f"  ⏭️  Skipping {wm} for alpha {alpha} (already computed)")
                    continue
                
                logger.info(f"  🔧 Alpha: {alpha}")
                
                # Fit ensemble with this weighting method and alpha
                ensemble = MultiLayerEnsemble(model_adapter, device=args.device)
                ensemble.depth_map = depth_map
                ensemble.depth_alpha = alpha  # Use current alpha
                ensemble.learned_lr = args.learned_lr
                ensemble.learned_max_iter = args.learned_max_iter
                ensemble.learned_use_lbfgs = args.learned_use_lbfgs
                ensemble.seed = args.seed
                
                if is_corruption_dataset:
                    # Corruption mode: use legacy path (fits calibrators, no cached probs)
                    weights = ensemble.fit(
                        selected_layers,
                        train_raw, train_labels,
                        val_raw,   val_labels,
                        feature_cache,
                        batch_size=args.batch_size,
                        weighting_method=wm,
                        learned_loss=args.learned_loss,
                        initial_weights=initial_weights
                    )
                else:
                    # Clean data: fast path using cached probabilities
                    weights = ensemble.fit_with_cached_probs(
                        selected_layers, val_labels, layer_probs, wm, args.learned_loss,
                        initial_weights=initial_weights
                    )
                
                # Log selected layers with their weights in a compact format
                weight_strs = [f"{weights[L]:.3f}" for L in selected_layers]
                logger.info(f"  Layer weights: {dict(zip(selected_layers, weight_strs))}")

                # Evaluate (using cached probabilities or corruption datasets)
                logger.info(f"[DEBUG] Dataset name: {args.dataset}")
                logger.info(f"[DEBUG] Corruption dataset path: {getattr(args, 'corruption_dataset_path', None)}")
                logger.info(f"[DEBUG] is_corruption_dataset flag: {is_corruption_dataset}")
                logger.info(f"[DEBUG] test_raw is None: {test_raw is None}")
                logger.info(f"[DEBUG] test_labels is None: {test_labels is None}")

                if is_corruption_dataset:
                    # Evaluate on all corruption types and severities
                    logger.info("  Evaluating on corruption datasets...")
                    corruption_results = evaluate_ensemble_on_corruptions(
                        ensemble=ensemble,
                        corruption_dataset_path=args.corruption_dataset_path,
                        dataset_name=args.dataset,
                        feature_cache=feature_cache,
                        batch_size=32,  # REDUCED from args.batch_size to prevent OOM
                        n_bins=args.bins
                    )

                    # Use mCE as the primary metric
                    test_ece = corruption_results['mCE']['ece']
                    test_acc = corruption_results['mCE']['accuracy']
                    test_brier = corruption_results['mCE']['brier']

                    # Store full corruption results for later
                    corruption_details = corruption_results
                else:
                    logger.info("  Evaluating on test set...")
                    test_probs = ensemble.predict(test_raw, feature_cache, args.batch_size)

                    # Metrics
                    test_ece = compute_ece(test_probs, test_labels, n_bins=args.bins)
                    test_acc = np.mean(np.argmax(test_probs, axis=1) == test_labels)

                    # Brier score
                    n_classes = test_probs.shape[1]
                    y_onehot = np.eye(n_classes)[test_labels]
                    test_brier = np.mean(np.sum((test_probs - y_onehot)**2, axis=1))

                    corruption_details = None

                # Effective Ensemble Size
                ees = _ees(weights)

                logger.info(f"  Results:")
                if is_corruption_dataset:
                    logger.info(f"    mCE (mean ECE): {test_ece:.7g}")
                    logger.info(f"    Mean Accuracy:  {test_acc:.7g}")
                    logger.info(f"    Mean Brier:     {test_brier:.7g}")
                else:
                    logger.info(f"    ECE:      {test_ece:.7g}")
                    logger.info(f"    Accuracy: {test_acc:.7g}")
                    logger.info(f"    Brier:    {test_brier:.7g}")
                logger.info(f"    EES:      {ees:.3f}")
                
                # Build result key and output directory
                if getattr(args, 'selection_metric_name', None):
                    # Hybrid single-run mode: use provided name
                    if wm in ['deep', 'shallow']:
                        key = f"{args.selection_metric_name}_{wm}_a{alpha}"
                    else:
                        key = f"{args.selection_metric_name}_{wm}"
                    wm_dir = base_dir
                else:
                    if k == 0:
                        k_prefix = 'K0'
                    elif k >= len(candidate_layers):
                        k_prefix = 'all'
                    else:
                        k_prefix = f'top{k}'
                    
                    if wm in ['deep', 'shallow']:
                        key = f'{k_prefix}_{wm}_a{alpha}'
                    else:
                        key = f'{k_prefix}_{wm}'
                    
                    # Save all outputs directly under base_dir (aggregate results)
                    wm_dir = base_dir
                
                # Store result
                result = EnsembleResult(
                    k=k,
                    selected_layers=selected_layers,
                    weights=weights,
                    weighting_method=wm,
                    test_ece=test_ece,
                    test_accuracy=test_acc,
                    test_brier=test_brier,
                    mce=test_ece if is_corruption_dataset else None
                )
                
                result_dict = asdict(result)
                result_dict["selected_layers_canonical"] = list(canonical_layers(selected_layers, depth_map))
                result_dict["ees"] = ees
                result_dict["depth_alpha"] = alpha  # Track which alpha was used

                # Add new ranking-based metrics
                if is_corruption_dataset and corruption_details is not None:
                    # Extract from corruption_results for corruption datasets
                    result_dict["test_error_auroc"] = corruption_details['mCE'].get('error_auroc', float("nan"))
                    result_dict["test_aurc"] = corruption_details['mCE'].get('aurc', float("nan"))
                else:
                    # Compute directly for non-corruption datasets
                    result_dict["test_error_auroc"] = float(compute_error_detection_auroc(test_probs, test_labels))
                    result_dict["test_aurc"] = float(compute_aurc(test_probs, test_labels))

                # Add corruption details if available
                if is_corruption_dataset and corruption_details is not None:
                    result_dict["corruption_results"] = corruption_details
                    result_dict["is_corruption_dataset"] = True

                # Defensive assert to catch malformed results early
                assert all(k in result_dict for k in ("test_ece","test_accuracy","test_brier")), \
                    f"Malformed ensemble result for {key}: keys={list(result_dict.keys())}"

                results[key] = result_dict

                # Save test probabilities only for clean datasets
                if not is_corruption_dataset:
                    probs_path = os.path.join(wm_dir, f"{key}_test_probs.npy")
                    np.save(probs_path, test_probs)
                    logger.info(f"    Saved probs: {probs_path}")

                # Save a compact ensemble_results.json under the same per-K/alpha dir
                results_json = os.path.join(wm_dir, "ensemble_results.json")

                # store only this run's entry (or append if file exists)
                onedict_entry = {
                    "k": k,
                    "selected_layers": selected_layers,
                    "selected_layers_canonical": list(canonical_layers(selected_layers, depth_map)),
                    "weights": weights,
                    "weighting_method": wm,
                    "depth_alpha": alpha,
                    "test_ece": float(test_ece),
                    "test_accuracy": float(test_acc),
                    "test_brier": float(test_brier),
                    "ees": float(ees),
                    "mce": float(test_ece) if is_corruption_dataset else None,
                }

                # Add corruption results if available
                if is_corruption_dataset and corruption_details is not None:
                    onedict_entry["corruption_results"] = corruption_details
                    onedict_entry["is_corruption_dataset"] = True

                onedict = { key: onedict_entry }
                if os.path.isfile(results_json) and os.path.getsize(results_json) > 0:
                    try:
                        with open(results_json, "r") as f:
                            existing = json.load(f)
                        existing.update(onedict)
                        with open(results_json, "w") as f:
                            json.dump(format_floats_7sig(existing), f, indent=2, cls=NumpyEncoder)
                    except Exception:
                        with open(results_json, "w") as f:
                            json.dump(format_floats_7sig(onedict), f, indent=2, cls=NumpyEncoder)
                else:
                    with open(results_json, "w") as f:
                        json.dump(format_floats_7sig(onedict), f, indent=2, cls=NumpyEncoder)
                logger.info(f"    Saved per-run results: {results_json}")
                
                # Mark this weighting method as done for alpha-agnostic methods
                already_done_this_wm = True

        # --- MODIFICATION 5: Add a break for single-run mode ---
        if provided_layers is not None:
            logger.info("Single hybrid run complete.")
            break

    # Add baseline results if available
    if baseline_results is not None:
        results['baselines'] = baseline_results
    
    # Add metric picks test evaluation if available
    if metric_picks_test_best is not None:
        results['metric_picks_test_best_per_k'] = metric_picks_test_best
    
    # Add per-layer ground truth results if available
    if per_layer_results is not None:
        results['per_layer_ground_truth'] = per_layer_results
        # Also add oracle best layer summary for easy reference
        valid_layers = [r for r in per_layer_results if r['test_ece'] != float('inf')]
        if valid_layers:
            best_by_ece = min(valid_layers, key=lambda x: x['test_ece'])
            results['oracle_best_layer'] = {
                'layer_idx': best_by_ece['layer_idx'],
                'test_ece': best_by_ece['test_ece'],
                'test_accuracy': best_by_ece['test_accuracy'],
                'test_brier': best_by_ece['test_brier']
            }
    
    # Save results with 7-digit formatting (at base level for summary)
    results_json = os.path.join(base_dir, "ensemble_results.json")
    rounded_results = format_floats_7sig(results)
    with open(results_json, 'w') as f:
        json.dump(rounded_results, f, indent=2, cls=NumpyEncoder)
    
    logger.info(f"\n✅ Results saved: {results_json}")
    
    # Print timing summary
    total_time = time.time() - total_start
    logger.info(f"\n⏱️  TOTAL EXPERIMENT TIME: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("="*80)
    if baseline_results:
        logger.info("\n📊 Baselines:")
        for method, metrics in baseline_results.items():
            if isinstance(metrics, dict) and 'ece' in metrics:
                ece = metrics.get('ece')
                acc = metrics.get('acc')
                if ece is not None and acc is not None:
                    logger.info(f"  {method:25s}: ECE={ece:.4f}, Acc={acc:.4f}")
    
    if 'oracle_best_layer' in results:
        oracle = results['oracle_best_layer']
        logger.info("\n🔮 Oracle Best Single Layer:")
        logger.info(f"  Layer {oracle['layer_idx']:3d} (best by ECE): "
                   f"ECE={oracle['test_ece']:.4f}, "
                   f"Acc={oracle['test_accuracy']:.4f}, "
                   f"Brier={oracle['test_brier']:.4f}")
    
    if topk_ensemble_results and "_leaderboards" in topk_ensemble_results:
        lb = topk_ensemble_results["_leaderboards"]

        logger.info("\n🏆 Best Metric-Based Ensembles (on val-B):")
        if "valB" in lb:
            for kkey in sorted(lb["valB"].keys()):
                winners = lb["valB"][kkey]
                logger.info(f"  {kkey}:")
                if winners:
                    for wm, info in sorted(winners.items()):
                        logger.info(f"    {wm:10s}: {info['best_metric']:40s} ECE={info['best_ece']:.7g}")

        logger.info("\n🎯 Best Metric-Based Ensembles (on TEST):")
        if "test" in lb:
            for kkey in sorted(lb["test"].keys()):
                winners = lb["test"][kkey]
                logger.info(f"  {kkey}:")
                if winners:
                    for wm, info in sorted(winners.items()):
                        logger.info(f"    {wm:10s}: {info['best_metric']:40s} ECE={info['best_ece']:.7g}")
    
    logger.info("\n🎯 Multi-Layer Ensembles:")

    def _is_ensemble_entry(v: Any) -> bool:
        return isinstance(v, dict) and {'test_ece', 'test_accuracy', 'test_brier'}.issubset(v.keys())

    # Collect only valid ensemble entries (stable order)
    ensemble_items = sorted(
        [(k, v) for k, v in results.items() if _is_ensemble_entry(v)],
        key=lambda kv: kv[0]
    )

    for key, result in ensemble_items:
        # by construction these exist
        e = float(result['test_ece'])
        a = float(result['test_accuracy'])
        b = float(result['test_brier'])
        logger.info(f"  {key:30s}: ECE={e:.4f}, Acc={a:.4f}, Brier={b:.4f}")

    if not ensemble_items:
        # Help debug if nothing printed
        logger.debug(f"No ensemble entries to print. results keys: {list(results.keys())}")
        # Also show any suspicious top-level scalars that might have caused issues
        weird = {k: type(v).__name__ for k, v in results.items() if not isinstance(v, dict)}
        if weird:
            logger.debug(f"Non-dict top-level entries: {weird}")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Layer Ensemble")
    
    # Data & model
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--model-name', type=str, default='resnet50')
    parser.add_argument('--training-method', type=str, default='standard')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--input-h', type=int, default=32,
                       help='Input height for model discovery')
    parser.add_argument('--input-w', type=int, default=32,
                       help='Input width for model discovery')
    
    # Ensemble config
    parser.add_argument('--topk', type=int, nargs='+', default=[1, 3],
                       help='K values to test')
    parser.add_argument('--sample-every', type=int, default=5,
                       help='Sample every Nth discovered layer (0 = per-block sampling)')
    parser.add_argument('--ensemble-weights', type=str, default='uniform',
                       choices=['uniform', 'deep', 'shallow', 'learned'],
                       help='[DEPRECATED] Use --weighting-methods instead')
    parser.add_argument('--weighting-methods', type=str, nargs='+', 
                       default=None,
                       help='List of weighting methods to try (e.g., uniform learned deep shallow)')
    parser.add_argument('--depth-alpha', type=float, default=1.5,
                       help='Temperature for depth-biased weights (legacy)')
    parser.add_argument('--depth-alphas', type=float, nargs='+', default=None,
                       help='Run deep/shallow for these alphas in one pass')
    parser.add_argument('--learned-loss', type=str, default='brier',
                       choices=['brier', 'nll'])
    parser.add_argument('--learned-lr', type=float, default=0.1,
                       help='Learning rate for Adam optimizer (ignored if using L-BFGS)')
    parser.add_argument('--learned-max-iter', type=int, default=500,
                       help='Max optimization iterations')
    parser.add_argument('--learned-use-lbfgs', action='store_true', default=False,
                       help='Use L-BFGS optimizer instead of Adam')
    
    # Additional analysis
    parser.add_argument('--scan-single-layers', action='store_true', default=False,
                       help='Evaluate each candidate layer independently')
    parser.add_argument('--eval-baselines', action='store_true', default=False,
                       help='Run standard baseline calibration methods')
    parser.add_argument('--doc-metrics', action='store_true', default=False,
                       help='Document selector metrics on val(A) and report Top-K picks per metric')
    parser.add_argument('--doc-metrics-both', action='store_true', default=False,
                       help='For each metric, document Top-K for both minimize and maximize.')
    parser.add_argument('--metrics-topk', type=int, default=5,
                       help='Number of top layers to report per metric in documentation')
    parser.add_argument('--val-split', type=float, default=0.5,
                       help='Fraction of validation set to use for metric computation (rest for holdout)')
    parser.add_argument('--eval-topk-ensembles', action='store_true', default=False,
                       help='Evaluate ensembles built from each metric\'s Top-K layers on val-B')
    parser.add_argument('--ensemble-test-ks', type=int, nargs='+', default=[1, 3, 5],
                       help='K values to test when evaluating metric-based ensembles')
    parser.add_argument('--eval-topk-weighting-methods', type=str, nargs='+',
                       default=['learned','shallow','deep','uniform'],
                       help='Weighting methods to try in metric-based ensemble eval')
    parser.add_argument('--save-probs-metric-eval', action='store_true', default=False,
                       help='If set, also saves valB/test probs for metric-based eval')
    parser.add_argument('--eval-metric-picks-on-test', action='store_true', default=False,
                       help='Evaluate each metric\'s Top-K picks on the TEST set under multiple weighting schemes')
    parser.add_argument('--metric-pick-weighting-methods', type=str, nargs='+',
                       default=None,
                       help='Weighting methods to use for metric-pick TEST eval (default: learned uniform deep shallow)')
    parser.add_argument('--metric-pick-ks', type=int, nargs='+', default=None,
                       help='K values to use for metric-pick TEST eval (default: args.ensemble_test_ks or [1,3,5])')
    
    # Evaluation
    parser.add_argument('--bins', type=int, default=15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--min-raw-acc', type=float, default=0.60,
                       help='Skip ensembles if raw model top-1 test accuracy is below this threshold')
    
    # Paths
    parser.add_argument('--output-dir', type=Path, required=True,
                       help='Dedicated output directory for this specific run (passed by launcher)')
    # Keep output-base-dir for non-launcher (manual) runs
    parser.add_argument('--output-base-dir', type=str, default=None)
    parser.add_argument('--feat-cache', type=str, default=None)
    parser.add_argument('--disable-cache', action='store_true', default=False,
                       help='Disable disk-backed feature/probability caching (recompute every time)')
    parser.add_argument('--results-base-dir', type=str, required=True,
                       help='Base directory with pre-trained models (same tree as before)')
    parser.add_argument('--metrics-doc-dir', type=str, default=None,
                       help='Directory containing pre-computed metric_docs_and_topk.json (for reuse across K jobs)')
    parser.add_argument('--corruption-dataset-path', type=str, default=None,
                       help='Optional path to corruption benchmark dataset (e.g., data/cifar10-c or data/cifar100-c).')
    
    # Optional single weighting strategy override (used by hybrid launcher)
    parser.add_argument('--weighting-strategy', type=str, default=None,
                       help='Override weighting methods with a single strategy (uniform, learned, deep, shallow)')
    
    
    # --- ADD THESE TWO ARGUMENTS ---
    parser.add_argument('--layer_indices', type=str, default=None,
                       help='Manually override layer selection with a comma-sep list')
    parser.add_argument('--selection_metric_name', type=str, default=None,
                       help='Override the K-value run and use this name for the output folder/key')
    parser.add_argument('--selection-method', type=str, default=None,
                       help='Alias for selection strategy name (from launcher)')
    parser.add_argument('--initial-weights', type=str, default=None,
                       help='Comma-separated initial weights for layers (e.g., "0.2,0.3,0.5"). '
                            'If provided with nc4_rank/nc1_rank/pareto_rank weighting methods, '
                            'these weights are used directly instead of learning from scratch.')

    # Compression ratio for feature extraction
    parser.add_argument('--compression-ratio', type=float, default=None,
                       help='Compression ratio for FixedSizeSPP_JL feature compression. '
                            'If not specified, auto-selects: 32x for CIFAR-100, 16x for others. '
                            'Falls back to fixed_feature_dim=512 when set to 0 or negative.')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    args = parse_args()
    run_experiment(args)