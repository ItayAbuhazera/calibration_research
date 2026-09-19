"""
Unified combined-hook extraction for Study A/B (§8b of BENCHMARK_IMPLEMENTATION_PLAN.md).

For a single checkpoint, one combined forward pass per data split captures:

  (a) the corrected internal-only RGCL draws (D1=9101, D2=9102, D3=9103),
      via Experiments.run_rgc_experiments.select_random_rgc_layers_corrected
      (see the §16 deviation this plan's §8a/§4 required: the plan named
      utils/layer_utils.py's discovery mechanism, but RGCL's real layer draw
      goes through a separate, independently-implemented filter in
      run_rgc_experiments.py -- select_random_rgc_layers_corrected is built
      on the corrected version of *that* function).

  (b) DAC's deterministic layer set, via
      Calibrators.density_aware_calibration.get_dac_target_layers.

Both are hooked in one pass by delegating to the existing, unmodified
Experiments.run_rgc_experiments.extract_and_aggregate_sgc_features with
layer_names=[] (so its own SPP/projection/aggregation path is a no-op) and
observer_layer_names=<the full union> (so every needed layer is hooked and
its raw activation handed to our batch_observer callback, per that
function's own existing observer mechanism -- this repo already supports
"hook extra layers during the same forward pass" for exactly this reason,
so no new hook-registration code was written here).

For each captured layer, the batch_observer callback immediately computes:
  (i)  DAC-style pooling: spatial-average-pool + L2-normalize (reusing
       Calibrators.density_aware_calibration.LayerKNNScorer._preprocess,
       the exact function DAC's own pipeline uses -- not reimplemented).
       Applied to every layer in the combined set (DAC's 5 layers AND the
       union of all three draws' layers), giving the representation-neutral,
       per-layer cache Study B needs for cells A/B/C/D.
  (ii) SPP + per-draw Gaussian projection to target_dim, reusing
       Experiments.run_rgc_experiments.apply_spp and reproducing the exact
       projection formula extract_and_aggregate_sgc_features uses
       (np.random.seed(seed); proj = randn(concat_dim, target_dim) /
       sqrt(target_dim)) -- applied only to each draw's own layer subset, to
       reproduce Study A's native RGCL vector for that draw.

Raw/SPP intermediates are discarded per-batch, immediately after the small
derived vectors are computed (§14's storage-bounding requirement) -- only the
small per-layer pooled arrays and the three (N, target_dim) projected arrays
are accumulated across the split.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from Calibrators.density_aware_calibration import LayerKNNScorer, get_dac_target_layers
from Experiments.run_rgc_experiments import (
    apply_spp,
    extract_and_aggregate_sgc_features,
    select_random_rgc_layers_corrected,
)

DEFAULT_DRAW_SEEDS: Tuple[int, int, int] = (9101, 9102, 9103)
STUDY_B_NUM_LAYERS = 5  # count-matched to DAC, per plan §4


def resolve_combined_layer_set(
    model: torch.nn.Module,
    model_name: str,
    dataset_name: str,
    device: torch.device,
    draw_seeds: Tuple[int, ...] = DEFAULT_DRAW_SEEDS,
    num_layers: int = STUDY_B_NUM_LAYERS,
) -> Dict[str, Any]:
    """
    Resolve DAC's deterministic layers and all corrected draws' layers, plus
    their union (the full hook set for one combined forward pass).

    Returns a dict with keys: "dac_layers", "draw_layers" (Dict[seed, List[str]]),
    "combined_layers" (deduplicated union, deterministic order).
    """
    dac_layers = list(get_dac_target_layers(model_name, model))

    draw_layers: Dict[int, List[str]] = {}
    for seed in draw_seeds:
        draw_layers[seed] = select_random_rgc_layers_corrected(
            model=model,
            model_name=model_name,
            dataset_name=dataset_name,
            device=device,
            num_layers=num_layers,
            seed=seed,
        )
        if "fc" in draw_layers[seed] or any(
            name == "fc" or name.startswith("fc.") for name in draw_layers[seed]
        ):
            raise RuntimeError(
                f"Corrected draw seed={seed} still contains the classifier layer; "
                "this must never happen -- see validation test §10.1."
            )

    combined_layers = list(dict.fromkeys([*dac_layers, *[n for names in draw_layers.values() for n in names]]))

    return {
        "dac_layers": dac_layers,
        "draw_layers": {int(k): v for k, v in draw_layers.items()},
        "combined_layers": combined_layers,
    }


def _pool_and_l2_normalize(raw: torch.Tensor) -> torch.Tensor:
    """DAC's own spatial-average-pool + L2-normalize convention (§4's
    "Common pooling" requirement), reused via a throwaway LayerKNNScorer
    instance rather than reimplemented -- LayerKNNScorer._preprocess is the
    exact function DAC's own pipeline calls on every layer it scores."""
    return LayerKNNScorer(k=1)._preprocess(raw)


class _DrawProjector:
    """Streaming SPP + fixed Gaussian projection for one draw, reproducing
    extract_and_aggregate_sgc_features's exact projection formula
    (np.random.seed(seed); randn(concat_dim, target_dim) / sqrt(target_dim)),
    but applied per-batch against activations captured by the *combined*
    hook set instead of a dedicated forward pass for this draw alone."""

    def __init__(self, layer_names: List[str], seed: int, target_dim: int, pooling_mode: str) -> None:
        self.layer_names = list(layer_names)
        self.seed = seed
        self.target_dim = target_dim
        self.pooling_mode = pooling_mode
        self._proj_matrix: Optional[np.ndarray] = None
        self._concat_dim: Optional[int] = None
        self.batches: List[np.ndarray] = []

    def consume_batch(self, activations: Dict[str, torch.Tensor]) -> None:
        spp_parts = []
        for name in self.layer_names:
            feat = activations.get(name)
            if feat is None:
                raise KeyError(f"Draw layer '{name}' missing from captured activations")
            spp_parts.append(apply_spp(feat, pooling_mode=self.pooling_mode).cpu().numpy().astype(np.float32, copy=False))
        batch_spp = np.concatenate(spp_parts, axis=1).astype(np.float32, copy=False)

        if self._concat_dim is None:
            self._concat_dim = int(batch_spp.shape[1])
            if self._concat_dim > self.target_dim:
                np.random.seed(self.seed)
                self._proj_matrix = np.random.randn(self._concat_dim, self.target_dim) / np.sqrt(self.target_dim)
            else:
                self._proj_matrix = None

        if self._proj_matrix is not None:
            projected = (batch_spp @ self._proj_matrix).astype(np.float32, copy=False)
        else:
            projected = batch_spp
        self.batches.append(projected)

    def finalize(self) -> np.ndarray:
        if not self.batches:
            return np.empty((0, self.target_dim), dtype=np.float32)
        out = np.concatenate(self.batches, axis=0)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-12, None)
        return (out / norms).astype(np.float32, copy=False)


def extract_combined_representations(
    model: torch.nn.Module,
    loader: DataLoader,
    split_name: str,
    combined_layers: List[str],
    draw_layers: Dict[int, List[str]],
    device: torch.device,
    target_dim: int = 256,
    pooling_mode: str = "max",
) -> Dict[str, Any]:
    """
    One combined-hook forward-pass sweep over `loader`. Returns:
      - "dac_style_features": {layer_name: (N, C_l) np.ndarray} for every
        layer in combined_layers (representation-neutral, Study B input).
      - "draw_projected_features": {seed: (N, target_dim) np.ndarray}
        (Study A's native RGCL vector per draw).
      - "labels": (N,) np.ndarray if the loader yields labels, else None.
      - "sample_ids": (N,) np.ndarray, dataset-index-based per plan §7.
    """
    pooled_accum: Dict[str, List[np.ndarray]] = {name: [] for name in combined_layers}
    projectors = {
        seed: _DrawProjector(layer_names, seed, target_dim, pooling_mode)
        for seed, layer_names in draw_layers.items()
    }
    labels_accum: List[np.ndarray] = []
    n_seen = 0
    any_labels = False

    def _batch_observer(_split_name: str, batch_data: Any, _logits: torch.Tensor, activations: Dict[str, torch.Tensor]) -> None:
        nonlocal n_seen, any_labels
        for name in combined_layers:
            raw = activations.get(name)
            if raw is None:
                raise KeyError(f"Combined layer '{name}' missing from captured activations")
            pooled = _pool_and_l2_normalize(raw).cpu().numpy().astype(np.float32, copy=False)
            pooled_accum[name].append(pooled)
            n_seen = max(n_seen, pooled.shape[0])
        for projector in projectors.values():
            projector.consume_batch(activations)
        if isinstance(batch_data, (tuple, list)) and len(batch_data) > 1:
            labels_accum.append(np.asarray(batch_data[1]))
            any_labels = True

    # layer_names must be non-empty: with layer_names=[], every batch's own
    # SPP feature has width 0, extract_with_spp's `batch_features.size > 0`
    # guard then drops every batch, and the function's own final
    # normalize(np.array([])) call crashes on the resulting empty 1D array.
    # Passing one harmless real layer keeps that internal path alive; its
    # output is discarded -- everything this function needs comes from
    # observer_layer_names/batch_observer instead.
    extract_and_aggregate_sgc_features(
        model=model,
        layer_names=[combined_layers[0]],
        train_loader=None,
        val_loader=None,
        test_loader=loader,
        device=device,
        target_dim=target_dim,
        seed=0,  # unused: this function's own projection output is discarded
        pooling_mode=pooling_mode,
        batch_observer=_batch_observer,
        observer_layer_names=combined_layers,
    )

    dac_style_features = {name: np.concatenate(arrs, axis=0) for name, arrs in pooled_accum.items()}
    draw_projected_features = {seed: proj.finalize() for seed, proj in projectors.items()}
    labels = np.concatenate(labels_accum, axis=0) if any_labels else None
    n_total = next(iter(dac_style_features.values())).shape[0]
    sample_ids = np.arange(n_total, dtype=np.int64)

    return {
        "split_name": split_name,
        "dac_style_features": dac_style_features,
        "draw_projected_features": draw_projected_features,
        "labels": labels,
        "sample_ids": sample_ids,
    }


def save_combined_representations(output_dir: str, split_name: str, result: Dict[str, Any]) -> None:
    os.makedirs(output_dir, exist_ok=True)
    layer_dir = os.path.join(output_dir, split_name, "dac_style_layers")
    os.makedirs(layer_dir, exist_ok=True)
    for name, arr in result["dac_style_features"].items():
        safe_name = name.replace("/", "_").replace("#", "_")
        np.save(os.path.join(layer_dir, f"{safe_name}.npy"), arr)

    draw_dir = os.path.join(output_dir, split_name, "draw_projected")
    os.makedirs(draw_dir, exist_ok=True)
    for seed, arr in result["draw_projected_features"].items():
        np.save(os.path.join(draw_dir, f"seed{seed}.npy"), arr)

    if result["labels"] is not None:
        np.save(os.path.join(output_dir, split_name, "labels.npy"), result["labels"])
    np.save(os.path.join(output_dir, split_name, "sample_ids.npy"), result["sample_ids"])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--model", default="resnet101")
    parser.add_argument("--checkpoint", required=True, help="e.g. seed1")
    parser.add_argument(
        "--checkpoint_path",
        default=None,
        help="Explicit checkpoint path; if omitted, resolved from --checkpoint under the "
        "repo's standard baseline_cross_entropy layout.",
    )
    parser.add_argument("--draw_seeds", default=",".join(str(s) for s in DEFAULT_DRAW_SEEDS))
    parser.add_argument("--num_layers", type=int, default=STUDY_B_NUM_LAYERS)
    parser.add_argument("--target_dimension", type=int, default=256)
    parser.add_argument("--pooling_mode", default="max")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_samples", type=int, default=None, help="Smoke-test subset size.")
    parser.add_argument(
        "--split", default="train", choices=("train", "val", "test"),
        help="Which loader to sweep for a smoke run.",
    )
    parser.add_argument("--data_seed", type=int, default=42, help="Seed for get_data_loaders' train/val split.")
    parser.add_argument("--output_dir", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    from utils.model_utils import get_data_loaders, load_trained_model
    from utils.calibration_utils import get_all_data_as_numpy
    from torch.utils.data import TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = args.checkpoint_path
    if checkpoint_path is None:
        checkpoint_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "geometric",
            "GeometricInternalCalibration",
            "GeometricInternalCalibration",
            "aaai_full_experiments",
            "results",
            "baseline",
            "baseline_cross_entropy",
            args.dataset,
            args.model,
            args.checkpoint,
            "best_model.pth",
        )

    num_classes = 100 if "cifar100" in args.dataset.lower() else 10
    model = load_trained_model(checkpoint_path, args.model, num_classes, device, dataset=args.dataset)

    train_loader, val_loader, test_loader, _ = get_data_loaders(
        args.dataset, args.batch_size, seed=args.data_seed
    )
    split_loader = {"train": train_loader, "val": val_loader, "test": test_loader}[args.split]
    split_raw, split_labels = get_all_data_as_numpy(split_loader)
    if args.max_samples is not None:
        split_raw = split_raw[: args.max_samples]
        split_labels = split_labels[: args.max_samples]

    loader = DataLoader(
        TensorDataset(torch.from_numpy(split_raw), torch.from_numpy(split_labels)),
        batch_size=args.batch_size,
        shuffle=False,
    )

    draw_seeds = tuple(int(s) for s in args.draw_seeds.split(",") if s.strip())
    resolved = resolve_combined_layer_set(
        model=model,
        model_name=args.model,
        dataset_name=args.dataset,
        device=device,
        draw_seeds=draw_seeds,
        num_layers=args.num_layers,
    )

    result = extract_combined_representations(
        model=model,
        loader=loader,
        split_name=args.split,
        combined_layers=resolved["combined_layers"],
        draw_layers=resolved["draw_layers"],
        device=device,
        target_dim=args.target_dimension,
        pooling_mode=args.pooling_mode,
    )

    save_combined_representations(args.output_dir, args.split, result)
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "layer_resolution.json"), "w") as f:
        json.dump(
            {
                "dac_layers": resolved["dac_layers"],
                "draw_layers": resolved["draw_layers"],
                "combined_layers": resolved["combined_layers"],
                "checkpoint": args.checkpoint,
                "checkpoint_path": checkpoint_path,
            },
            f,
            indent=2,
        )
    print(f"Wrote combined representations for split={args.split} to {args.output_dir}")


if __name__ == "__main__":
    main()
