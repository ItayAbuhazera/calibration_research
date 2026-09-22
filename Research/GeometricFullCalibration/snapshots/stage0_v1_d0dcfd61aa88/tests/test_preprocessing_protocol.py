"""Regression checks for the authoritative preprocessing specification.

Covers the four properties the normalization repair must guarantee:

1. identical clean pixels are preprocessed identically by every deterministic
   loader path (train/val loader, clean test loader, CIFAR-C transform);
2. normalization is applied exactly once;
3. artifacts produced under a different protocol are rejected, not reused;
4. sample identities, labels and split membership are unchanged by the repair.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from utils import preprocessing_protocol as pp  # noqa: E402

CIFAR_MEAN = np.array(pp.CIFAR_STATS[0], dtype=np.float32)
CIFAR_STD = np.array(pp.CIFAR_STATS[1], dtype=np.float32)
IN_MEAN = np.array(pp.IMAGENET_STATS[0], dtype=np.float32)
IN_STD = np.array(pp.IMAGENET_STATS[1], dtype=np.float32)


@pytest.fixture(scope="module")
def pixels() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8), mode="RGB")


@pytest.fixture(scope="module")
def cifar100_root(tmp_path_factory):
    """The loaders read ./data relative to the CWD; run from the project root."""
    if not (PROJECT_ROOT / "data" / "cifar-100-python").exists():
        pytest.skip("CIFAR-100 not available locally")
    old = os.getcwd()
    os.chdir(PROJECT_ROOT)
    yield PROJECT_ROOT
    os.chdir(old)


def _expected(pixels: Image.Image, mean, std) -> np.ndarray:
    x = np.asarray(pixels, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return (x - mean[:, None, None]) / std[:, None, None]


# ------------------------------------------------------------ 1. equivalence
def test_corrected_protocol_matches_training_normalization_on_identical_pixels(pixels):
    train_like = pp.eval_transform("cifar100", pp.PROTOCOL_CORRECTED)
    clean_test = pp.eval_transform("cifar100", pp.PROTOCOL_CORRECTED)
    corruption = pp.eval_transform("cifar100", pp.PROTOCOL_CORRECTED, corruption=True)
    a, b, c = train_like(pixels), clean_test(pixels), corruption(pixels)
    assert torch.equal(a, b) and torch.equal(a, c)
    np.testing.assert_allclose(a.numpy(), _expected(pixels, CIFAR_MEAN, CIFAR_STD), atol=1e-6)


def test_real_loader_paths_agree_under_corrected_protocol(cifar100_root, pixels):
    from data.cifar100 import get_test_loader, get_train_valid_loader

    _, val_loader = get_train_valid_loader(
        batch_size=8, augment=False, random_seed=4, valid_size=0.1
    )
    test_loader = get_test_loader(
        batch_size=8, shuffle=False, preprocessing_protocol=pp.PROTOCOL_CORRECTED
    )
    corruption = pp.eval_transform("cifar100", pp.PROTOCOL_CORRECTED, corruption=True)
    v = val_loader.dataset.transform(pixels)
    t = test_loader.dataset.transform(pixels)
    assert torch.equal(v, t), "clean-test and validation loaders normalize identical pixels differently"
    assert torch.equal(v, corruption(pixels))


def test_legacy_protocol_is_preserved_bit_for_bit(cifar100_root, pixels):
    """The legacy path must keep reproducing the historical (mismatched) inputs."""
    from data.cifar100 import get_test_loader

    default_loader = get_test_loader(batch_size=8, shuffle=False)  # historical call signature
    legacy_loader = get_test_loader(
        batch_size=8, shuffle=False, preprocessing_protocol=pp.PROTOCOL_LEGACY
    )
    expected = _expected(pixels, IN_MEAN, IN_STD)
    np.testing.assert_allclose(default_loader.dataset.transform(pixels).numpy(), expected, atol=1e-6)
    assert torch.equal(
        default_loader.dataset.transform(pixels), legacy_loader.dataset.transform(pixels)
    )
    legacy_c = pp.eval_transform("cifar100", pp.PROTOCOL_LEGACY, corruption=True)
    np.testing.assert_allclose(legacy_c(pixels).numpy(), expected, atol=1e-6)


def test_legacy_and_corrected_differ_only_for_cifar_test_paths():
    assert pp.eval_normalization("cifar100", pp.PROTOCOL_LEGACY) != pp.eval_normalization(
        "cifar100", pp.PROTOCOL_CORRECTED
    )
    # CIFAR-10 clean test was already consistent; CIFAR-10-C was not.
    assert pp.eval_normalization("cifar10", pp.PROTOCOL_LEGACY) == pp.eval_normalization(
        "cifar10", pp.PROTOCOL_CORRECTED
    )
    assert pp.eval_normalization(
        "cifar10", pp.PROTOCOL_LEGACY, corruption=True
    ) != pp.eval_normalization("cifar10", pp.PROTOCOL_CORRECTED, corruption=True)
    # Tiny ImageNet loaders were consistent throughout.
    assert pp.eval_normalization("tiny_imagenet", pp.PROTOCOL_LEGACY) == pp.eval_normalization(
        "tiny_imagenet", pp.PROTOCOL_CORRECTED
    )


def test_unaudited_dataset_is_refused_not_guessed():
    with pytest.raises(ValueError, match="no audited training normalization"):
        pp.eval_transform("pacs", pp.PROTOCOL_CORRECTED)


# -------------------------------------------------------- 2. exactly once
@pytest.mark.parametrize("protocol", pp.PROTOCOLS)
def test_normalization_applied_exactly_once(pixels, protocol):
    tf = pp.eval_transform("cifar100", protocol)
    norms = [t for t in tf.transforms if isinstance(t, transforms.Normalize)]
    assert len(norms) == 1
    mean, std = pp.eval_normalization("cifar100", protocol)
    const = Image.new("RGB", (32, 32), (128, 128, 128))
    out = tf(const).numpy()
    expected = ((128 / 255.0) - np.array(mean)) / np.array(std)
    np.testing.assert_allclose(out[:, 0, 0], expected, atol=1e-6)
    # a double application would land elsewhere:
    twice = (expected - np.array(mean)) / np.array(std)
    assert not np.allclose(out[:, 0, 0], twice, atol=1e-3)


def test_cached_arrays_are_already_normalized_not_raw(pixels):
    """Benchmark caches store POST-transform tensors; re-normalizing them would
    be a second normalization. Value range identifies the constants used."""
    tf = pp.eval_transform("cifar100", pp.PROTOCOL_CORRECTED)
    lo = tf(Image.new("RGB", (32, 32), (0, 0, 0))).numpy()[:, 0, 0]
    hi = tf(Image.new("RGB", (32, 32), (255, 255, 255))).numpy()[:, 0, 0]
    np.testing.assert_allclose(lo.min(), -0.4914 / 0.2023, atol=1e-4)  # -2.4291
    np.testing.assert_allclose(hi.max(), (1 - 0.4465) / 0.2010, atol=1e-4)  # 2.7537


# ---------------------------------------------- 3. incompatible-cache rejection
def test_unstamped_directory_is_legacy_and_rejected_by_corrected(tmp_path):
    d = tmp_path / "fitted"
    d.mkdir()
    (d / "native_dac.pkl").write_bytes(b"x")
    assert pp.read_protocol(str(d)) == pp.PROTOCOL_LEGACY
    with pytest.raises(pp.IncompatiblePreprocessingError, match="not interchangeable"):
        pp.require_compatible(str(d), "cifar100", pp.PROTOCOL_CORRECTED, "fitted state")
    pp.require_compatible(str(d), "cifar100", pp.PROTOCOL_LEGACY, "fitted state")  # legacy ok


def test_corrected_directory_rejected_by_legacy_and_stamp_hash_verified(tmp_path):
    d = tmp_path / "cache"
    d.mkdir()
    (d / "logits.npy").write_bytes(b"x")
    pp.write_stamp(str(d), "cifar100", pp.PROTOCOL_CORRECTED)
    pp.require_compatible(str(d), "cifar100", pp.PROTOCOL_CORRECTED, "cache")
    with pytest.raises(pp.IncompatiblePreprocessingError):
        pp.require_compatible(str(d), "cifar100", pp.PROTOCOL_LEGACY, "cache")
    # a stamp whose constants no longer match the protocol definition is rejected
    stamp_path = d / pp.STAMP_FILENAME
    stamp = json.loads(stamp_path.read_text())
    stamp["stamp_hash"] = "0" * 64
    stamp_path.write_text(json.dumps(stamp))
    with pytest.raises(pp.IncompatiblePreprocessingError, match="constants changed"):
        pp.require_compatible(str(d), "cifar100", pp.PROTOCOL_CORRECTED, "cache")


def test_empty_or_missing_directory_is_compatible(tmp_path):
    pp.require_compatible(str(tmp_path / "missing"), "cifar100", pp.PROTOCOL_CORRECTED, "x")
    (tmp_path / "empty").mkdir()
    pp.require_compatible(str(tmp_path / "empty"), "cifar100", pp.PROTOCOL_CORRECTED, "x")


def _runner_guard():
    from Experiments.run_unified_benchmark import _guard_preprocessing_protocol

    return _guard_preprocessing_protocol


def _ns(out, fitted, protocol, dataset="cifar100"):
    return argparse.Namespace(
        dataset=dataset,
        output_dir=str(out),
        fitted_state_dir=str(fitted) if fitted else None,
        reuse_non_metric_from="",
        preprocessing_protocol=protocol,
    )


def test_runner_rejects_legacy_fitted_state_and_cache_under_corrected(tmp_path):
    guard = _runner_guard()
    fitted = tmp_path / "fitted"
    fitted.mkdir()
    (fitted / "native_dac.pkl").write_bytes(b"legacy pickle")
    parser = argparse.ArgumentParser()
    with pytest.raises(SystemExit):
        guard(_ns(tmp_path / "out", fitted, pp.PROTOCOL_CORRECTED), parser)

    out = tmp_path / "out2" / "intermediates"
    out.mkdir(parents=True)
    (out / "progress.json").write_text("{}")
    with pytest.raises(SystemExit):
        guard(_ns(tmp_path / "out2", None, pp.PROTOCOL_CORRECTED), parser)
    # legacy resumption of legacy directories stays possible and leaves them unstamped
    guard(_ns(tmp_path / "out2", None, pp.PROTOCOL_LEGACY), parser)
    assert not (out / pp.STAMP_FILENAME).exists()


def test_runner_stamps_fresh_corrected_directories(tmp_path):
    guard = _runner_guard()
    parser = argparse.ArgumentParser()
    fitted = tmp_path / "fitted"
    guard(_ns(tmp_path / "out", fitted, pp.PROTOCOL_CORRECTED), parser)
    assert pp.read_protocol(str(fitted)) == pp.PROTOCOL_CORRECTED
    assert pp.read_protocol(str(tmp_path / "out" / "intermediates")) == pp.PROTOCOL_CORRECTED


def test_runner_fingerprint_separates_protocols_and_keeps_legacy_stable():
    from Experiments.run_unified_benchmark import _args_fingerprint

    base = dict(
        dataset="cifar100", model="resnet101", method="baseline_cross_entropy", seed=4,
        batch_size=256, target_dimension=1024, num_layers=5, num_coordinates=1,
        enable_post_fusion_temperature=False, enable_rgcl_tail_hybrids=False,
        rgcl_tail_sources="", stab_metric="l2", enable_kcal_baseline=False,
        enable_kcal_factorial=False,
    )
    legacy = argparse.Namespace(**base, preprocessing_protocol=pp.PROTOCOL_LEGACY)
    corrected = argparse.Namespace(**base, preprocessing_protocol=pp.PROTOCOL_CORRECTED)
    no_field = argparse.Namespace(**base)  # a pre-amendment caller
    assert _args_fingerprint(legacy) == _args_fingerprint(no_field)
    assert _args_fingerprint(legacy) != _args_fingerprint(corrected)


# -------------------------- 4. sample identities, labels, split membership
def test_split_membership_and_labels_unchanged_by_protocol(cifar100_root):
    from utils.model_utils import get_data_loaders

    tr_a, va_a, te_a, _ = get_data_loaders(
        "cifar100", 64, seed=4, preprocessing_protocol=pp.PROTOCOL_LEGACY
    )
    tr_b, va_b, te_b, _ = get_data_loaders(
        "cifar100", 64, seed=4, preprocessing_protocol=pp.PROTOCOL_CORRECTED
    )
    assert sorted(tr_a.sampler.indices) == sorted(tr_b.sampler.indices)
    assert sorted(va_a.sampler.indices) == sorted(va_b.sampler.indices)
    assert set(tr_b.sampler.indices).isdisjoint(va_b.sampler.indices)
    # identical membership of the *validation* subset for the same seed
    assert set(va_a.sampler.indices) == set(va_b.sampler.indices)
    assert np.array_equal(np.asarray(te_a.dataset.targets), np.asarray(te_b.dataset.targets))
    assert np.array_equal(te_a.dataset.data, te_b.dataset.data)
    assert len(te_b.dataset) == 10000
    assert not isinstance(te_b.sampler, torch.utils.data.RandomSampler)  # deterministic order
    # only the transform differs
    assert te_a.dataset.transform is not te_b.dataset.transform


def test_train_and_validation_transforms_untouched_by_repair(cifar100_root):
    from data.cifar100 import get_train_valid_loader

    _, val = get_train_valid_loader(batch_size=8, augment=False, random_seed=4, valid_size=0.1)
    norms = [t for t in val.dataset.transform.transforms if isinstance(t, transforms.Normalize)]
    assert [tuple(np.round(n.mean, 4)) for n in norms] == [pp.CIFAR_STATS[0]]
    assert [tuple(np.round(n.std, 4)) for n in norms] == [pp.CIFAR_STATS[1]]
