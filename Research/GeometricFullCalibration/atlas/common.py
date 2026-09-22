"""Shared helpers: model loading, atomic writes, completion markers, protocol guards."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict

import numpy as np
import torch

from . import data, spec


def setup_torch():
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        if os.environ.get("ATLAS_ALLOW_CPU_SMOKE") == "1":   # engineering smoke tests on the cpu partition ONLY
            return torch.device("cpu")
        raise SystemExit("CUDA unavailable (no CPU fallback for GPU stages)")
    return torch.device("cuda")


def load_model(seed: int, device):
    from utils.model_utils import construct_model_path, load_trained_model
    ckpt = construct_model_path(data.CKPT_ROOT, "baseline_cross_entropy", "cifar100", "resnet101", seed)
    return load_trained_model(ckpt, "resnet101", 100, device, dataset="cifar100").eval(), ckpt


def check_protocol(seed: int):
    from utils import preprocessing_protocol as pp
    for d in (f"{data.BENCH}/evaluation/checkpoint_seed{seed}/clean/intermediates", f"{data.BENCH}/fitted_method/checkpoint_seed{seed}"):
        if pp.read_protocol(d) != pp.PROTOCOL_CORRECTED:
            raise SystemExit(f"{d} is not corrected-protocol")


def atomic_npz(path: str, **arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def atomic_json(path: str, obj: Any):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1, default=float)
    os.replace(tmp, path)


def file_sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def mark_done(marker: str, payload: Dict):
    atomic_json(marker, {**payload, "completed_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "code_snapshot": os.environ.get("ATLAS_SNAPSHOT", "live_tree")})


def is_done(marker: str) -> bool:
    return os.path.exists(marker)
