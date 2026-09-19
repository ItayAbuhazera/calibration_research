#!/usr/bin/env python3
"""Export RGCL recoverability cells without changing the method geometry.

The exporter has two gated phases. ``preflight`` extracts the reference, the
seed-specific validation cell, and clean CIFAR-100 test cell, then builds the
single authorised global L2 kNN instrument. ``corruptions`` reloads those
artifacts and reads each requested CIFAR-100-C cell once.

Both recoverability arms are written from one computation. The per-class arm
uses RGC's per-class 1-NN decision and intentionally exports no probability
vectors. The global arm uses a 200-NN vote with a distance-based tie break and
fixed Laplace smoothing for its class distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Iterable

import joblib
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Calibrators.geometric_calibrator import FullVectorDistanceFusionCalibrator
from Experiments.run_rgc_experiments import (
    extract_and_aggregate_sgc_features,
    select_random_rgc_layers,
)
from utils.calibration_utils import load_cifar_c_loader
from utils.model_utils import get_data_loaders, load_trained_model, set_seed
from utils.stability_space import StabilitySpace


SEEDS = (1, 2, 3, 4, 5)
CORRUPTIONS = ("gaussian_noise", "defocus_blur", "fog", "jpeg_compression")
SEVERITIES = (1, 3, 5)
K_VOTE = 200
K_RADIUS = 200
N_CLASSES = 100
TEMPERATURE = 1.1
BETA = 30.0
TARGET_DIM = 256
NUM_LAYERS = 6
POOLING_MODE = "max"
EXPECTED_CLEAN_ACCURACY = {
    1: 0.7528,
    2: 0.7620,
    3: 0.7638,
    4: 0.7631,
    5: 0.7635,
}
CLEAN_ACCURACY_TOLERANCE = 0.003
CHECKPOINT_SHA256 = {
    1: "ccc2e65a5444ee2c6d363f4371733ce6cdafcf0dcbe62dfbf16807c03f882a6d",
    2: "bf04fa32d26f5731d5e0138eb12a135ab357a42a9dc3de1f3031029eada7dc89",
    3: "b9f22464d67ae48ce29568ab743d8c68479953e6701c7b7d8b641cddac6d06a2",
    4: "5903abd2169517a9782ea05a37fb863d53443a540dbc1da7ea0a86a64ad68828",
    5: "768bbf00977a5ae753ed8dfe3108c6e4e1c4c6e34d45d8f72b7ff2fcac9a6ba6",
}

PERCLASS_EXTRA_MISSING = {"head_probabilities", "knn_distribution"}
GEOMETRY_AGREEMENT_MIN = 0.20
GEOMETRY_PERMUTED_MAX = 0.05
GEOMETRY_COLLAPSE_MIN = 0.15


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class IndexedDataset(Dataset):
    """Attach the original dataset index before delegating to any transform."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        # These IDs are created with the dataset, before any item is transformed
        # or any model stage has produced rows.
        self.original_ids = np.arange(len(dataset), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        sample_id = np.int64(index)
        image, label = self.dataset[index]
        return image, np.int64(label), sample_id


def _indexed_loader(loader: DataLoader) -> DataLoader:
    indexed = IndexedDataset(loader.dataset)
    return DataLoader(
        indexed,
        batch_size=loader.batch_size,
        sampler=loader.sampler,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        drop_last=loader.drop_last,
    )


def _master_ids(loader: DataLoader) -> np.ndarray:
    sampler = loader.sampler
    if hasattr(sampler, "indices"):
        ids = np.asarray(list(sampler.indices), dtype=np.int64)
    else:
        ids = np.asarray(loader.dataset.original_ids, dtype=np.int64)
    ids = np.sort(ids)
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("dataset-created sample IDs are not unique")
    return ids


class StageCollector:
    """Maintain independent ID/payload buffers for the three export stages."""

    def __init__(self, auxiliary_layer: str = "layer3"):
        self.auxiliary_layer = auxiliary_layer
        self._records: dict[str, dict[str, list[np.ndarray]]] = {}

    def __call__(
        self,
        split_name: str,
        batch_data: Any,
        logits: torch.Tensor,
        activations: dict[str, torch.Tensor],
    ) -> None:
        if not isinstance(batch_data, (tuple, list)) or len(batch_data) < 3:
            raise RuntimeError("indexed batches must carry image, label, and sample_id")
        labels = np.asarray(batch_data[1].detach().cpu(), dtype=np.int64)
        ids = np.asarray(batch_data[2].detach().cpu(), dtype=np.int64)
        logits_np = logits.detach().cpu().numpy().astype(np.float32, copy=False)
        layer = activations.get(self.auxiliary_layer)
        if layer is None:
            raise RuntimeError(f"observer did not capture {self.auxiliary_layer}")
        if layer.ndim == 4:
            # This is the exact global-average pooling used by the published
            # FeatureExtractor single-layer arm.
            layer = layer.mean(dim=(2, 3))
        single = layer.detach().cpu().numpy().astype(np.float32, copy=False)

        rec = self._records.setdefault(
            split_name,
            {
                "sample_id_logits": [],
                "logits": [],
                "y_pred_head": [],
                "sample_id_geometry": [],
                "single_layer_features": [],
                "sample_id_labels": [],
                "y_true": [],
            },
        )
        # IDs are appended independently at each stage boundary. They are not
        # reconstructed from row counts at export time.
        rec["sample_id_logits"].append(ids.copy())
        rec["logits"].append(logits_np)
        rec["y_pred_head"].append(logits_np.argmax(axis=1).astype(np.int64))
        rec["sample_id_geometry"].append(ids.copy())
        rec["single_layer_features"].append(single)
        rec["sample_id_labels"].append(ids.copy())
        rec["y_true"].append(labels)

    def finish(self, split_name: str) -> dict[str, np.ndarray]:
        if split_name not in self._records:
            raise RuntimeError(f"no observed batches for split {split_name!r}")
        out = {
            key: np.concatenate(parts, axis=0)
            for key, parts in self._records.pop(split_name).items()
        }
        lengths = {key: value.shape[0] for key, value in out.items()}
        if len(set(lengths.values())) != 1:
            raise RuntimeError(f"stage collector row mismatch: {lengths}")
        return out


def _join_stage(
    master_ids: np.ndarray,
    stage_ids: np.ndarray,
    payload: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    master = np.asarray(master_ids, dtype=np.int64).ravel()
    ids = np.asarray(stage_ids, dtype=np.int64).ravel()
    if len(np.unique(ids)) != len(ids):
        raise RuntimeError("stage IDs contain duplicates")
    if set(ids.tolist()) != set(master.tolist()):
        raise RuntimeError("stage IDs do not contain exactly the dataset-created IDs")
    positions = {int(sample_id): row for row, sample_id in enumerate(ids.tolist())}
    order = np.asarray([positions[int(sample_id)] for sample_id in master], dtype=np.int64)
    joined = {}
    for key, value in payload.items():
        array = np.asarray(value)
        if array.shape[0] != len(ids):
            raise RuntimeError(f"{key}: {array.shape[0]} payload rows for {len(ids)} IDs")
        joined[key] = array[order]
    return ids[order], joined


def _prepare_split(
    master_ids: np.ndarray,
    rgcl_features: np.ndarray,
    records: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    logits_ids, logits = _join_stage(
        master_ids,
        records["sample_id_logits"],
        {
            "logits": records["logits"],
            "y_pred_head": records["y_pred_head"],
        },
    )
    geometry_ids, geometry = _join_stage(
        master_ids,
        records["sample_id_geometry"],
        {
            "rgcl_features": rgcl_features,
            "single_layer_features": records["single_layer_features"],
        },
    )
    labels_ids, labels = _join_stage(
        master_ids,
        records["sample_id_labels"],
        {"y_true": records["y_true"]},
    )
    return {
        "sample_id": np.asarray(master_ids, dtype=np.int64),
        "sample_id_logits": logits_ids,
        "sample_id_geometry": geometry_ids,
        "sample_id_labels": labels_ids,
        **logits,
        **geometry,
        **labels,
    }


def _torch_softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(logits)).to(dtype=torch.float64)
    return torch.softmax(tensor / float(temperature), dim=1).cpu().numpy()


def _global_neighbour_fields(
    index: NearestNeighbors,
    query: np.ndarray,
    train_ids: np.ndarray,
    train_labels: np.ndarray,
) -> dict[str, Any]:
    distances, bank_rows = index.kneighbors(
        query, n_neighbors=K_RADIUS, return_distance=True
    )
    distances = distances.astype(np.float32, copy=False)
    bank_rows = bank_rows.astype(np.int64, copy=False)
    neighbour_labels = np.asarray(train_labels, dtype=np.int64)[bank_rows]
    neighbours = np.asarray(train_ids, dtype=np.int64)[bank_rows]

    n = len(query)
    counts = np.zeros((n, N_CLASSES), dtype=np.int64)
    rows = np.repeat(np.arange(n, dtype=np.int64), K_VOTE)
    np.add.at(counts, (rows, neighbour_labels.reshape(-1)), 1)

    top_count = counts.max(axis=1)
    tied = counts == top_count[:, None]
    tied_fraction = float(np.mean(tied.sum(axis=1) > 1))

    distance_sums = np.zeros((n, N_CLASSES), dtype=np.float64)
    np.add.at(distance_sums, (rows, neighbour_labels.reshape(-1)), distances.reshape(-1))
    mean_distances = np.full((n, N_CLASSES), np.inf, dtype=np.float64)
    np.divide(distance_sums, counts, out=mean_distances, where=counts > 0)
    mean_distances[~tied] = np.inf
    # Vote-count ties are broken by the smallest mean distance among that
    # class's tied neighbours. np.argmin is only an unreachable final tie break
    # when floating-point mean distances are exactly equal.
    y_global = mean_distances.argmin(axis=1).astype(np.int64)
    if not np.all(tied[np.arange(n), y_global]):
        raise RuntimeError("distance tie-break selected a non-modal class")

    top_two = np.partition(counts, kth=N_CLASSES - 2, axis=1)[:, -2:]
    margin = (top_two[:, 1] - top_two[:, 0]).astype(np.float64) / K_VOTE
    distribution = (counts.astype(np.float64) + 1.0) / (K_VOTE + N_CLASSES)
    if not np.allclose(distribution.sum(axis=1), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Laplace-smoothed kNN distribution does not sum to one")

    return {
        "neighbours": neighbours,
        "neighbour_labels": neighbour_labels,
        "knn_counts": counts,
        "knn_distribution": distribution,
        "knn_radius": distances[:, K_RADIUS - 1],
        "neighbour_margin": margin,
        "y_pred_globalknn": y_global,
        "tie_stats": {
            "mean_top1_vote_count": float(np.mean(top_count)),
            "median_top1_vote_count": float(np.median(top_count)),
            "fraction_top1_count_tied_before_distance_break": tied_fraction,
            "k_vote": K_VOTE,
            "tie_break": "smallest mean distance among modal-count classes",
        },
    }


def _freeze_full_vector(
    train_features: np.ndarray,
    train_labels: np.ndarray,
) -> FullVectorDistanceFusionCalibrator:
    fusion = FullVectorDistanceFusionCalibrator(
        model=None,
        X_train_embed=train_features,
        y_train=train_labels,
        metric="l2",
        library="fast_separation",
        beta_grid=np.asarray([BETA], dtype=np.float64),
    )
    # The public class has no fixed-beta constructor. Populate only the fitted
    # state required by its existing calibrate() implementation; do not call
    # fit() and therefore do not select on these data.
    fusion.best_beta = BETA
    fusion.beta_selection = {
        "source": "inherited validation-selected published result",
        "best_beta": BETA,
    }
    fusion._inner.best_lambda = BETA
    fusion._inner.lambda_selection = {
        "source": "inherited validation-selected published result",
        "best_lambda": BETA,
    }
    fusion._inner.is_fitted = True
    fusion.is_fitted = True
    return fusion


def _compute_cell(
    cell_name: str,
    prepared: dict[str, np.ndarray],
    rgcl_space: StabilitySpace,
    full_vector: FullVectorDistanceFusionCalibrator,
    global_index: NearestNeighbors,
    train_ids: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    y_head = prepared["y_pred_head"]
    rgcl_features = prepared["rgcl_features"]
    single_features = prepared["single_layer_features"]

    per_class_distances = rgcl_space.calc_per_class_1nn_distances(rgcl_features)
    y_perclass = per_class_distances.argmin(axis=1).astype(np.int64)
    contrast = np.asarray(rgcl_space.calc_stab(rgcl_features, y_head), dtype=np.float64)
    if not np.isfinite(contrast).all():
        raise RuntimeError(f"{cell_name}: contrast contains non-finite values")

    global_fields = _global_neighbour_fields(
        global_index, rgcl_features, train_ids, train_labels
    )
    untempered_head = _torch_softmax(prepared["logits"], temperature=1.0)
    vector_probs = full_vector.calibrate(
        X_test_embed=single_features,
        model_probs=untempered_head,
    )
    vector_probs = np.asarray(vector_probs, dtype=np.float64)

    common = {
        "sample_id": prepared["sample_id"],
        "sample_id_logits": prepared["sample_id_logits"],
        "sample_id_geometry": prepared["sample_id_geometry"],
        "sample_id_labels": prepared["sample_id_labels"],
        "y_true": prepared["y_true"].astype(np.int64, copy=False),
        "y_pred_head": y_head.astype(np.int64, copy=False),
        "y_pred_vector": vector_probs.argmax(axis=1).astype(np.int64),
        "vector_probabilities": vector_probs,
        "neighbour_labels": global_fields["neighbour_labels"],
        "neighbours": global_fields["neighbours"],
        "knn_counts": global_fields["knn_counts"],
        "contrast": contrast,
        "logits": prepared["logits"].astype(np.float32, copy=False),
        "knn_radius": global_fields["knn_radius"],
        "neighbour_margin": global_fields["neighbour_margin"],
        "temperature": np.full(len(y_head), TEMPERATURE, dtype=np.float64),
    }
    perclass = {**common, "y_pred_knn": y_perclass}
    globalknn = {
        **common,
        "y_pred_knn": global_fields["y_pred_globalknn"],
        "head_probabilities": _torch_softmax(prepared["logits"], TEMPERATURE),
        "knn_distribution": global_fields["knn_distribution"],
    }
    _assert_arm_equivalence(perclass, globalknn, cell_name)
    return perclass, globalknn, global_fields["tie_stats"]


def _assert_arm_equivalence(
    perclass: dict[str, np.ndarray],
    globalknn: dict[str, np.ndarray],
    cell_name: str,
) -> None:
    per_keys, global_keys = set(perclass), set(globalknn)
    if global_keys - per_keys != PERCLASS_EXTRA_MISSING or per_keys - global_keys:
        raise RuntimeError(
            f"{cell_name}: unexpected arm key difference: "
            f"global-only={sorted(global_keys - per_keys)}, "
            f"perclass-only={sorted(per_keys - global_keys)}"
        )
    for key in sorted(per_keys & global_keys):
        if key == "y_pred_knn":
            continue
        if not np.array_equal(np.asarray(perclass[key]), np.asarray(globalknn[key])):
            raise RuntimeError(f"{cell_name}: shared arm field {key!r} differs")


def _atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=path.stem + ".", suffix=".npz", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_arms(
    export_root: Path,
    seed: int,
    cell_name: str,
    perclass: dict[str, np.ndarray],
    globalknn: dict[str, np.ndarray],
) -> None:
    _assert_arm_equivalence(perclass, globalknn, cell_name)
    per_path = export_root / f"seed{seed}" / "perclass" / f"{cell_name}.npz"
    global_path = export_root / f"seed{seed}" / "globalknn" / f"{cell_name}.npz"
    _atomic_npz(per_path, perclass)
    _atomic_npz(global_path, globalknn)
    with np.load(per_path, allow_pickle=False) as per_disk, np.load(
        global_path, allow_pickle=False
    ) as global_disk:
        _assert_arm_equivalence(
            {key: np.asarray(per_disk[key]) for key in per_disk.files},
            {key: np.asarray(global_disk[key]) for key in global_disk.files},
            cell_name + " (reloaded)",
        )
    print(f"ARM_EQUALITY PASS seed={seed} cell={cell_name}", flush=True)


def _derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.arange(n)
    for _ in range(100):
        permutation = rng.permutation(n)
        if not np.any(permutation == base):
            return permutation
    # A non-zero rotation is always a derangement and remains deterministic.
    return np.roll(base, 1 + seed % (n - 1))


def _rejoin_one_stage(
    baseline: dict[str, np.ndarray],
    stage_id_key: str,
    stage_keys: Iterable[str],
    permutation: np.ndarray,
    permute_ids: bool,
) -> dict[str, np.ndarray]:
    out = {key: np.asarray(value).copy() for key, value in baseline.items()}
    ids = np.asarray(baseline[stage_id_key])
    source_ids = ids[permutation] if permute_ids else ids
    source_payload = {key: np.asarray(baseline[key])[permutation] for key in stage_keys}
    joined_ids, joined = _join_stage(baseline["sample_id"], source_ids, source_payload)
    out[stage_id_key] = joined_ids
    out.update(joined)
    return out


def _run_harness(
    harness_script: Path,
    cells: list[Path],
    clean_name: str,
    output: Path,
    expect_success: bool,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(harness_script),
        "--cells",
        *[str(path) for path in cells],
        "--clean-name",
        clean_name,
        "--alpha-metric",
        "nll",
        "--out",
        str(output),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if expect_success and result.returncode != 0:
        raise RuntimeError(
            f"control harness unexpectedly failed ({result.returncode}):\n"
            f"{result.stdout}\n{result.stderr}"
        )
    if not expect_success and result.returncode == 0:
        raise RuntimeError("payload/ID mismatch passed the harness silently")
    return result


def _run_alignment_controls(
    seed: int,
    perclass: dict[str, np.ndarray],
    globalknn: dict[str, np.ndarray],
    harness_script: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    from rgc_shift.recoverability import assert_temperature_preserves_argmax

    n = len(globalknn["sample_id"])
    logits_permutation = _derangement(n, 100_000 + seed)
    bad_logits = _rejoin_one_stage(
        globalknn,
        "sample_id_logits",
        ("logits", "head_probabilities"),
        logits_permutation,
        permute_ids=False,
    )
    caught_direct = False
    try:
        assert_temperature_preserves_argmax(
            bad_logits["logits"], bad_logits["temperature"], bad_logits["y_pred_head"]
        )
    except AssertionError:
        caught_direct = True
    if not caught_direct:
        raise RuntimeError("Control 1 logits mismatch was not caught by argmax invariant")

    control_dir = artifact_dir / "controls"
    control_dir.mkdir(parents=True, exist_ok=True)
    bad_logits_path = control_dir / "logits_mismatch" / "clean.npz"
    _atomic_npz(bad_logits_path, bad_logits)
    logits_harness = _run_harness(
        harness_script,
        [bad_logits_path],
        "clean",
        control_dir / "logits_mismatch.json",
        expect_success=False,
    )
    print(
        "CONTROL1_LOGITS PASS "
        f"seed={seed} harness_returncode={logits_harness.returncode}",
        flush=True,
    )

    geometry_results: dict[str, Any] = {}
    for arm_index, (arm_name, baseline) in enumerate(
        (("perclass", perclass), ("globalknn", globalknn))
    ):
        permutation = _derangement(n, 200_000 + 10 * seed + arm_index)
        geometry_keys = [
            key
            for key in (
                "y_pred_knn",
                "y_pred_vector",
                "vector_probabilities",
                "neighbour_labels",
                "neighbours",
                "knn_counts",
                "contrast",
                "knn_distribution",
                "knn_radius",
                "neighbour_margin",
            )
            if key in baseline
        ]
        bad_geometry = _rejoin_one_stage(
            baseline,
            "sample_id_geometry",
            geometry_keys,
            permutation,
            permute_ids=False,
        )
        intact = float(np.mean(baseline["y_pred_head"] == baseline["y_pred_knn"]))
        permuted = float(
            np.mean(bad_geometry["y_pred_head"] == bad_geometry["y_pred_knn"])
        )
        collapse = intact - permuted
        passed = (
            intact >= GEOMETRY_AGREEMENT_MIN
            and permuted <= GEOMETRY_PERMUTED_MAX
            and collapse >= GEOMETRY_COLLAPSE_MIN
        )
        if not passed:
            raise RuntimeError(
                f"Control 1 geometry mismatch not detected for {arm_name}: "
                f"intact={intact:.6f}, permuted={permuted:.6f}, collapse={collapse:.6f}"
            )
        geometry_results[arm_name] = {
            "intact_head_geometry_agreement": intact,
            "permuted_head_geometry_agreement": permuted,
            "collapse": collapse,
            "passed": True,
        }
        print(
            "CONTROL1_GEOMETRY PASS "
            f"seed={seed} arm={arm_name} intact={intact:.6f} "
            f"permuted={permuted:.6f} collapse={collapse:.6f}",
            flush=True,
        )

    order_results: dict[str, Any] = {}
    stage_specs = {
        "labels": ("sample_id_labels", ("y_true",)),
        "logits": (
            "sample_id_logits",
            ("logits", "y_pred_head", "temperature", "head_probabilities"),
        ),
        "geometry": (
            "sample_id_geometry",
            (
                "y_pred_knn",
                "y_pred_vector",
                "vector_probabilities",
                "neighbour_labels",
                "neighbours",
                "knn_counts",
                "contrast",
                "knn_distribution",
                "knn_radius",
                "neighbour_margin",
            ),
        ),
    }
    for arm_index, (arm_name, baseline) in enumerate(
        (("perclass", perclass), ("globalknn", globalknn))
    ):
        candidate = baseline
        arm_results = {}
        for stage_index, (stage_name, (id_key, possible_keys)) in enumerate(
            stage_specs.items()
        ):
            keys = tuple(key for key in possible_keys if key in baseline)
            permutation = _derangement(
                n, 300_000 + 100 * seed + 10 * arm_index + stage_index
            )
            candidate = _rejoin_one_stage(
                baseline, id_key, keys, permutation, permute_ids=True
            )
            equal = set(candidate) == set(baseline) and all(
                np.array_equal(candidate[key], baseline[key]) for key in baseline
            )
            if not equal:
                raise RuntimeError(
                    f"Control 2 order-only permutation changed {arm_name}/{stage_name}"
                )
            arm_results[stage_name] = True
            print(
                f"CONTROL2_ORDER PASS seed={seed} arm={arm_name} stage={stage_name}",
                flush=True,
            )
        smoke_path = control_dir / f"order_only_{arm_name}" / "clean.npz"
        _atomic_npz(smoke_path, candidate)
        smoke = _run_harness(
            harness_script,
            [smoke_path],
            "clean",
            control_dir / f"order_only_{arm_name}.json",
            expect_success=True,
        )
        arm_results["harness_returncode"] = smoke.returncode
        order_results[arm_name] = arm_results

    results = {
        "seed": seed,
        "control1_logits": {
            "argmax_invariant_caught": True,
            "harness_returncode": logits_harness.returncode,
            "passed": True,
        },
        "control1_geometry": geometry_results,
        "control2_order_only": order_results,
        "thresholds": {
            "intact_min": GEOMETRY_AGREEMENT_MIN,
            "permuted_max": GEOMETRY_PERMUTED_MAX,
            "collapse_min": GEOMETRY_COLLAPSE_MIN,
        },
    }
    (artifact_dir / "alignment_controls.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    return results


def _validate_export(payload: dict[str, np.ndarray], arm: str, cell_name: str) -> None:
    required = {
        "sample_id",
        "y_true",
        "y_pred_head",
        "y_pred_knn",
        "sample_id_logits",
        "sample_id_geometry",
        "sample_id_labels",
        "y_pred_vector",
        "neighbour_labels",
        "neighbours",
        "knn_counts",
        "contrast",
        "logits",
        "knn_radius",
        "neighbour_margin",
        "temperature",
        "vector_probabilities",
    }
    if arm == "globalknn":
        required |= {"head_probabilities", "knn_distribution"}
    missing = required - set(payload)
    if missing:
        raise RuntimeError(f"{cell_name}/{arm}: missing fields {sorted(missing)}")
    if arm == "perclass" and PERCLASS_EXTRA_MISSING & set(payload):
        raise RuntimeError(f"{cell_name}/perclass: probability vectors must be omitted")
    n = len(payload["sample_id"])
    if payload["neighbour_labels"].shape != (n, K_VOTE):
        raise RuntimeError("neighbour_labels shape mismatch")
    if payload["neighbours"].shape != (n, K_VOTE):
        raise RuntimeError("neighbours shape mismatch")
    if payload["knn_counts"].shape != (n, N_CLASSES):
        raise RuntimeError("knn_counts shape mismatch")
    for id_key in ("sample_id_logits", "sample_id_geometry", "sample_id_labels"):
        if not np.array_equal(payload[id_key], payload["sample_id"]):
            raise RuntimeError(f"{cell_name}/{arm}: {id_key} is not joined")


def _make_spaces(
    cache: dict[str, np.ndarray], global_index: NearestNeighbors
) -> tuple[StabilitySpace, FullVectorDistanceFusionCalibrator, NearestNeighbors]:
    rgcl_space = StabilitySpace(
        cache["train_rgcl"],
        cache["train_y"],
        library="fast_separation",
        metric="l2",
    )
    full_vector = _freeze_full_vector(cache["train_single"], cache["train_y"])
    return rgcl_space, full_vector, global_index


def _save_tie_stats(artifact_dir: Path, cell_name: str, stats: dict[str, Any]) -> None:
    stats_dir = artifact_dir / "tie_stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / f"{cell_name}.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    prominent = " UNDER_RESOLVED_CLEAN" if (
        cell_name == "clean"
        and stats["fraction_top1_count_tied_before_distance_break"] > 0.25
    ) else ""
    print(
        f"TIE_STATS cell={cell_name} mean_top1={stats['mean_top1_vote_count']:.6f} "
        f"median_top1={stats['median_top1_vote_count']:.6f} "
        f"tie_fraction={stats['fraction_top1_count_tied_before_distance_break']:.6f}"
        f"{prominent}",
        flush=True,
    )


def _extract(
    model: torch.nn.Module,
    layer_names: list[str],
    device: torch.device,
    seed: int,
    collector: StageCollector,
    train_loader: DataLoader | None,
    val_loader: DataLoader | None,
    test_loader: DataLoader | None,
    known_spp_concat_dim: int | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    return extract_and_aggregate_sgc_features(
        model=model,
        layer_names=layer_names,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        target_dim=TARGET_DIM,
        seed=seed,
        pooling_mode=POOLING_MODE,
        batch_observer=collector,
        observer_layer_names=["layer3"],
        known_spp_concat_dim=known_spp_concat_dim,
    )


def _load_and_verify_model(
    checkpoint: Path, seed: int, device: torch.device
) -> torch.nn.Module:
    actual_hash = _sha256(checkpoint)
    if actual_hash != CHECKPOINT_SHA256[seed]:
        raise RuntimeError(
            f"seed {seed}: checkpoint SHA mismatch: {actual_hash} != {CHECKPOINT_SHA256[seed]}"
        )
    model = load_trained_model(
        str(checkpoint), "resnet101", N_CLASSES, device, dataset="cifar100"
    )
    if float(model.temp) != 1.0:
        raise RuntimeError(f"seed {seed}: expected model.temp=1.0, got {model.temp}")
    model.eval()
    return model


def _cache_path(artifact_dir: Path) -> Path:
    return artifact_dir / "reference_arrays.npz"


def run_preflight(args: argparse.Namespace) -> None:
    seed = args.seed
    set_seed(seed)
    artifact_dir = args.artifact_root / f"seed{seed}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    train_raw, val_raw, clean_raw, num_classes = get_data_loaders(
        "cifar100", args.batch_size, seed=seed
    )
    if num_classes != N_CLASSES:
        raise RuntimeError(f"expected {N_CLASSES} classes, got {num_classes}")
    train_loader = _indexed_loader(train_raw)
    val_loader = _indexed_loader(val_raw)
    clean_loader = _indexed_loader(clean_raw)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _load_and_verify_model(args.checkpoint, seed, device)
    selected_layers = select_random_rgc_layers(
        model, "resnet101", "cifar100", device, NUM_LAYERS, seed
    )
    print(f"SELECTED_LAYERS seed={seed} layers={selected_layers}", flush=True)

    collector = StageCollector()
    train_features, val_features, clean_features, extraction_info = _extract(
        model,
        selected_layers,
        device,
        seed,
        collector,
        train_loader,
        val_loader,
        clean_loader,
    )
    assert train_features is not None and val_features is not None and clean_features is not None
    train_prepared = _prepare_split(
        _master_ids(train_loader), train_features, collector.finish("train")
    )
    val_prepared = _prepare_split(
        _master_ids(val_loader), val_features, collector.finish("val")
    )
    clean_prepared = _prepare_split(
        _master_ids(clean_loader), clean_features, collector.finish("test")
    )

    measured_accuracy = float(
        np.mean(clean_prepared["y_pred_head"] == clean_prepared["y_true"])
    )
    expected = EXPECTED_CLEAN_ACCURACY[seed]
    difference = measured_accuracy - expected
    print(
        f"CLEAN_ACCURACY seed={seed} measured={measured_accuracy:.6f} "
        f"expected={expected:.6f} difference_pp={100.0 * difference:+.4f}",
        flush=True,
    )
    if abs(difference) > CLEAN_ACCURACY_TOLERANCE:
        raise RuntimeError(
            f"seed {seed}: clean accuracy outside +/-0.3pp tolerance"
        )

    np.savez(
        _cache_path(artifact_dir),
        train_rgcl=train_prepared["rgcl_features"].astype(np.float32, copy=False),
        train_single=train_prepared["single_layer_features"].astype(np.float32, copy=False),
        train_y=train_prepared["y_true"].astype(np.int64, copy=False),
        train_ids=train_prepared["sample_id"].astype(np.int64, copy=False),
    )
    global_index = NearestNeighbors(n_neighbors=K_RADIUS, metric="l2")
    global_index.fit(train_prepared["rgcl_features"])
    joblib.dump(global_index, artifact_dir / "global_knn.joblib", compress=0)

    cache = {
        "train_rgcl": train_prepared["rgcl_features"],
        "train_single": train_prepared["single_layer_features"],
        "train_y": train_prepared["y_true"],
        "train_ids": train_prepared["sample_id"],
    }
    rgcl_space, full_vector, global_index = _make_spaces(cache, global_index)
    exported: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
    for cell_name, prepared in (("val", val_prepared), ("clean", clean_prepared)):
        perclass, globalknn, tie_stats = _compute_cell(
            cell_name,
            prepared,
            rgcl_space,
            full_vector,
            global_index,
            cache["train_ids"],
            cache["train_y"],
        )
        _validate_export(perclass, "perclass", cell_name)
        _validate_export(globalknn, "globalknn", cell_name)
        _write_arms(args.export_root, seed, cell_name, perclass, globalknn)
        _save_tie_stats(artifact_dir, cell_name, tie_stats)
        exported[cell_name] = (perclass, globalknn)

    controls = _run_alignment_controls(
        seed,
        exported["clean"][0],
        exported["clean"][1],
        args.harness_script,
        artifact_dir,
    )
    metadata = {
        "seed": seed,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": CHECKPOINT_SHA256[seed],
        "expected_clean_accuracy": expected,
        "measured_clean_accuracy": measured_accuracy,
        "clean_accuracy_tolerance_pp": 0.3,
        "temperature": TEMPERATURE,
        "temperature_source": "validation-selected inherited result",
        "beta": BETA,
        "beta_source": "published full-vector arm, single-layer source, beta=30",
        "selected_layers": selected_layers,
        "extraction_info": extraction_info,
        "metric": "true Euclidean L2",
        "k_vote": K_VOTE,
        "k_radius": K_RADIUS,
        "knn_distribution": "Laplace: (count + 1) / (k_vote + C)",
        "controls": controls,
    }
    (artifact_dir / "preflight_complete.json").write_text(
        json.dumps(metadata, indent=2, default=float), encoding="utf-8"
    )
    print(f"PREFLIGHT PASS seed={seed}", flush=True)


def run_corruptions(args: argparse.Namespace) -> None:
    seed = args.seed
    set_seed(seed)
    artifact_dir = args.artifact_root / f"seed{seed}"
    marker = artifact_dir / "preflight_complete.json"
    if not marker.exists():
        raise RuntimeError(f"seed {seed}: preflight marker is absent")
    metadata = json.loads(marker.read_text(encoding="utf-8"))

    with np.load(_cache_path(artifact_dir), allow_pickle=False) as stored:
        cache = {key: np.asarray(stored[key]) for key in stored.files}
    global_index = joblib.load(artifact_dir / "global_knn.joblib")
    rgcl_space, full_vector, global_index = _make_spaces(cache, global_index)

    _, _, clean_raw, _ = get_data_loaders("cifar100", args.batch_size, seed=seed)
    clean_transform = clean_raw.dataset.transform
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _load_and_verify_model(args.checkpoint, seed, device)
    selected_layers = select_random_rgc_layers(
        model, "resnet101", "cifar100", device, NUM_LAYERS, seed
    )
    if selected_layers != metadata["selected_layers"]:
        raise RuntimeError("selected RGC layers differ from preflight")
    known_dim = int(metadata["extraction_info"]["total_spp_dims"])

    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            cell_name = f"{corruption}_s{severity}"
            raw_loader = load_cifar_c_loader(
                "cifar100",
                corruption,
                severity,
                clean_transform,
                args.batch_size,
                cifar_c_dir=str(args.cifar_c_dir),
                num_workers=args.num_workers,
            )
            loader = _indexed_loader(raw_loader)
            collector = StageCollector()
            _, _, features, extraction_info = _extract(
                model,
                selected_layers,
                device,
                seed,
                collector,
                None,
                None,
                loader,
                known_spp_concat_dim=known_dim,
            )
            if features is None:
                raise RuntimeError(f"{cell_name}: no RGCL features returned")
            if int(extraction_info["total_spp_dims"]) != known_dim:
                raise RuntimeError(f"{cell_name}: SPP dimension drift")
            prepared = _prepare_split(
                _master_ids(loader), features, collector.finish("test")
            )
            perclass, globalknn, tie_stats = _compute_cell(
                cell_name,
                prepared,
                rgcl_space,
                full_vector,
                global_index,
                cache["train_ids"],
                cache["train_y"],
            )
            _validate_export(perclass, "perclass", cell_name)
            _validate_export(globalknn, "globalknn", cell_name)
            _write_arms(args.export_root, seed, cell_name, perclass, globalknn)
            _save_tie_stats(artifact_dir, cell_name, tie_stats)
            del features, prepared, perclass, globalknn
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"CORRUPTIONS PASS seed={seed}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("preflight", "corruptions"))
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cifar-c-dir", required=True, type=Path)
    parser.add_argument("--export-root", type=Path, default=PROJECT_ROOT / "exports")
    parser.add_argument(
        "--artifact-root", type=Path, default=PROJECT_ROOT / "artifacts" / "recoverability"
    )
    parser.add_argument("--harness-script", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "preflight":
        run_preflight(args)
    else:
        run_corruptions(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
