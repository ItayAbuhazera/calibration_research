"""
Authoritative deterministic input-preprocessing specification.

WHY THIS MODULE EXISTS
----------------------
An audit (2026-09-21, see docs/normalization_audit.md) found that the
unified benchmark fed CIFAR-100 checkpoints inputs normalized with two
different sets of constants:

    checkpoint training / train split / validation split : CIFAR statistics
    clean test split / CIFAR-100-C                       : ImageNet statistics

The checkpoints were trained through ``data.cifar100.get_train_valid_loader``
(CIFAR statistics; see ``Experiments/train_model.py::build_dataloaders``), so
the CIFAR statistics are the ones the trained weights expect. Evaluation must
apply the SAME deterministic normalization. Training-time augmentation
(random crop / flip) is intentionally not part of this specification: it is a
training-only stochastic transform and is never applied at evaluation.

PROTOCOL IDENTIFIERS
--------------------
``legacy_v1_mixed_norm``
    The historical behaviour, preserved bit-for-bit so that legacy results
    remain reproducible and can be labelled as such. CIFAR-100 clean test and
    CIFAR-10/100-C use ImageNet statistics; train/val use CIFAR statistics.
``corrected_v2_train_norm``
    Every evaluation split uses the normalization the checkpoint was trained
    with (CIFAR statistics for CIFAR-10/100, ImageNet statistics for Tiny
    ImageNet, whose loaders already trained and evaluated consistently).

The choice of constants is made from CHECKPOINT PROVENANCE ONLY. It is not,
and must never be, selected by looking at test accuracy.

Cached logits, features, reference banks and fitted calibrator state are only
compatible with a run whose protocol identifier matches. Artifacts written
before this module existed carry no stamp and are treated as
``legacy_v1_mixed_norm``.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, Optional, Tuple

PROTOCOL_LEGACY = "legacy_v1_mixed_norm"
PROTOCOL_CORRECTED = "corrected_v2_train_norm"
PROTOCOLS = (PROTOCOL_LEGACY, PROTOCOL_CORRECTED)

#: Name of the stamp file written next to fitted state / cached intermediates.
STAMP_FILENAME = "preprocessing_stamp.json"

CIFAR_STATS: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
    (0.4914, 0.4822, 0.4465),
    (0.2023, 0.1994, 0.2010),
)
IMAGENET_STATS: Tuple[Tuple[float, ...], Tuple[float, ...]] = (
    (0.485, 0.456, 0.406),
    (0.229, 0.224, 0.225),
)

#: dataset -> statistics its checkpoints were TRAINED with. Verified against
#: data/cifar10.py, data/cifar100.py and data/tiny_imagenet.py train loaders.
#: Datasets not listed here have not been audited; requesting the corrected
#: protocol for them is an error rather than a guess.
TRAIN_NORMALIZATION: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    "cifar10": CIFAR_STATS,
    "cifar100": CIFAR_STATS,
    "tiny_imagenet": IMAGENET_STATS,
}

#: dataset -> statistics the LEGACY evaluation path applied to the clean TEST
#: split, and to CIFAR-*-C through run_unified_benchmark.py.
LEGACY_TEST_NORMALIZATION: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    "cifar10": CIFAR_STATS,          # data/cifar10.py::get_test_loader
    "cifar100": IMAGENET_STATS,      # data/cifar100.py::get_test_loader  <-- the mismatch
    "tiny_imagenet": IMAGENET_STATS,
}
LEGACY_CORRUPTION_NORMALIZATION: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...]]] = {
    # run_unified_benchmark.py hard-coded ImageNet statistics for every
    # CIFAR-*-C cell, including CIFAR-10-C whose train split is CIFAR-normalized.
    "cifar10": IMAGENET_STATS,
    "cifar100": IMAGENET_STATS,
}


def _canon(dataset: str) -> str:
    name = str(dataset).lower().replace("-c", "").replace("-", "").replace("_", "")
    if name in ("tinyimagenet",):
        return "tiny_imagenet"
    return name


def check_protocol(protocol: str) -> str:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown preprocessing protocol {protocol!r}; expected one of {PROTOCOLS}")
    return protocol


def train_normalization(dataset: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    name = _canon(dataset)
    if name not in TRAIN_NORMALIZATION:
        raise ValueError(
            f"dataset {dataset!r} has no audited training normalization; refusing to "
            "guess one. Audit its training loader and add it to TRAIN_NORMALIZATION."
        )
    return TRAIN_NORMALIZATION[name]


def eval_normalization(
    dataset: str, protocol: str, *, corruption: bool = False
) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    """(mean, std) applied to an EVALUATION split under ``protocol``."""
    check_protocol(protocol)
    name = _canon(dataset)
    if protocol == PROTOCOL_CORRECTED:
        return train_normalization(name)
    table = LEGACY_CORRUPTION_NORMALIZATION if corruption else LEGACY_TEST_NORMALIZATION
    if name not in table:
        raise ValueError(f"no legacy evaluation normalization recorded for {dataset!r}")
    return table[name]


def eval_transform(
    dataset: str,
    protocol: str,
    *,
    corruption: bool = False,
    image_size: Optional[int] = None,
):
    """Deterministic evaluation transform: [Resize] -> ToTensor -> Normalize.

    ToTensor maps uint8 HWC in [0, 255] to float CHW in [0, 1] (channel order
    is RGB throughout: torchvision CIFAR datasets and the CIFAR-C .npy files
    are both RGB). Normalize is applied exactly once.
    """
    from torchvision import transforms

    mean, std = eval_normalization(dataset, protocol, corruption=corruption)
    steps = []
    if image_size is not None and int(image_size) != 32:
        steps.append(transforms.Resize((int(image_size), int(image_size))))
    steps.extend([transforms.ToTensor(), transforms.Normalize(mean=list(mean), std=list(std))])
    return transforms.Compose(steps)


def preprocessing_stamp(dataset: str, protocol: str) -> Dict[str, Any]:
    """Serializable description of the preprocessing a run used."""
    check_protocol(protocol)
    name = _canon(dataset)
    stamp: Dict[str, Any] = {
        "protocol": protocol,
        "dataset": name,
        "train_normalization": [list(v) for v in train_normalization(name)] if name in TRAIN_NORMALIZATION else None,
        "clean_test_normalization": [list(v) for v in eval_normalization(name, protocol)],
    }
    if name in LEGACY_CORRUPTION_NORMALIZATION:
        stamp["corruption_normalization"] = [
            list(v) for v in eval_normalization(name, protocol, corruption=True)
        ]
    stamp["stamp_hash"] = hashlib.sha256(
        json.dumps(stamp, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return stamp


def write_stamp(directory: str, dataset: str, protocol: str) -> str:
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, STAMP_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preprocessing_stamp(dataset, protocol), f, indent=2)
    return path


def read_protocol(directory: str) -> str:
    """Protocol recorded in ``directory``; an unstamped directory is legacy."""
    path = os.path.join(directory, STAMP_FILENAME)
    if not os.path.exists(path):
        return PROTOCOL_LEGACY
    with open(path, "r", encoding="utf-8") as f:
        return str(json.load(f).get("protocol", PROTOCOL_LEGACY))


class IncompatiblePreprocessingError(RuntimeError):
    """A cached artifact was produced under a different preprocessing protocol."""


def require_compatible(directory: str, dataset: str, protocol: str, what: str) -> None:
    """Reject reuse of ``directory`` (cache / fitted state / bank) under a
    different protocol. Unstamped directories are legacy by definition.

    An EMPTY or non-existent directory is compatible (nothing to reuse yet).
    """
    check_protocol(protocol)
    if not os.path.isdir(directory) or not os.listdir(directory):
        return
    found = read_protocol(directory)
    if found != protocol:
        raise IncompatiblePreprocessingError(
            f"{what} at {directory!r} was produced under preprocessing protocol "
            f"{found!r} but this run requests {protocol!r}. Cached logits, features, "
            "reference banks and fitted calibrators are not interchangeable across "
            "protocols; use a fresh directory (refit) instead of reusing it."
        )
    stamp_path = os.path.join(directory, STAMP_FILENAME)
    if os.path.exists(stamp_path):
        with open(stamp_path, "r", encoding="utf-8") as f:
            recorded = json.load(f)
        expected = preprocessing_stamp(dataset, protocol)
        if recorded.get("stamp_hash") != expected["stamp_hash"]:
            raise IncompatiblePreprocessingError(
                f"{what} at {directory!r} carries stamp {recorded.get('stamp_hash')} "
                f"but the current definition of {protocol!r} hashes to "
                f"{expected['stamp_hash']}; the constants changed since it was written."
            )
