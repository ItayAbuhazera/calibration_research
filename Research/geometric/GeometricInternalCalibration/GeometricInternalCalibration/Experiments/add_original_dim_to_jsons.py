# Save as: Experiments/add_original_dim_to_jsons.py

"""
Add original_dim field to existing per_layer_ground_truth.json files.
Runs quick inference to determine layer dimensions.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Experiments.run_post_hoc_calibration import load_trained_model, get_data_loaders

from utils.logging_config import get_logger
logger = get_logger(__name__)


def get_layer_dim_from_model(model, layer_name: str, sample_images: torch.Tensor) -> int:
    """Extract feature dimensionality for a given layer using a single batch."""
    device = next(model.parameters()).device
    model.eval()

    sample_images = sample_images.to(device)

    if layer_name == "data_layer":
        return int(sample_images.flatten(1).shape[1])

    activation = {}

    def hook(module, _input, output):
        activation["output"] = output

    target_layer = None
    for name, module in model.named_modules():
        if name == layer_name:
            target_layer = module
            break

    if target_layer is None:
        raise ValueError(f"Layer {layer_name} not found in model")

    handle = target_layer.register_forward_hook(hook)

    with torch.no_grad():
        _ = model(sample_images)

    handle.remove()

    feat = activation.get("output")
    if feat is None:
        raise ValueError(f"No features extracted for layer {layer_name}")

    feat = feat.detach()
    if feat.ndim == 4:  # (B, C, H, W)
        dim = feat.shape[1] * feat.shape[2] * feat.shape[3]
    elif feat.ndim == 3:  # (B, T, D) e.g., transformer tokens
        dim = feat.shape[1] * feat.shape[2]
    elif feat.ndim == 2:  # (B, D)
        dim = feat.shape[1]
    else:
        raise ValueError(f"Unexpected feature shape: {feat.shape}")

    return int(dim)


def get_layer_name_by_index(model: torch.nn.Module, index: int) -> str:
    """
    Map an integer index (as used in candidate_layers / JSON keys)
    to the corresponding module name from model.named_modules().

    The index is assumed to match the enumeration order used when
    generating per_layer_ground_truth.json.
    """
    for i, (name, _module) in enumerate(model.named_modules()):
        if i == index:
            return name
    raise IndexError(
        f"Layer index {index} is out of range. "
        f"Model has only {i + 1} modules (0-based indexing)."
    )


def process_json_file(json_path, model, sample_images):
    """
    Add `original_dim` for each candidate layer index in this JSON.

    Args:
        json_path: Path to per_layer_ground_truth.json.
        model: Pre-loaded model matching this experiment config.
        sample_images: Sample batch tensor on the correct device.

    Returns:
        True if file modified, False otherwise.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    candidate_layers = data.get("candidate_layers", [])
    if not candidate_layers:
        logger.warning(f"No candidate_layers found in {json_path}")
        return False

    data.setdefault("test", {})
    data["test"].setdefault("original_dim", {})

    modified = False

    for idx in candidate_layers:
        idx = int(idx)

        if str(idx) in data["test"]["original_dim"]:
            continue

        try:
            layer_name = get_layer_name_by_index(model, idx)
        except Exception as e:
            logger.warning(f"[{json_path}] Invalid layer index {idx}: {e}")
            continue

        try:
            dim = get_layer_dim_from_model(model, layer_name, sample_images)
            data["test"]["original_dim"][str(idx)] = dim
            modified = True
            logger.debug(
                f"[{json_path}] idx={idx}, layer='{layer_name}', original_dim={dim}"
            )
        except Exception as e:
            logger.warning(
                f"[{json_path}] Failed to compute dim for idx={idx}, "
                f"layer='{layer_name}': {e}"
            )

    if modified:
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

    return modified


def main():
    parser = argparse.ArgumentParser(description="Add original_dim to layer selection JSONs")
    parser.add_argument('--layer_selection_results_path', type=Path, required=True)
    parser.add_argument('--checkpoint_base_dir', type=Path, required=True)
    parser.add_argument('--data_dir', type=Path, default=Path("./Data"))
    parser.add_argument('--log_level', default="INFO")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=args.log_level)
    
    # Find all per_layer_ground_truth.json files
    json_files = list(args.layer_selection_results_path.rglob("per_layer_ground_truth.json"))
    
    logger.info(f"Found {len(json_files)} JSON files to process")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cache = {}
    sample_cache = {}
    failed_configs = set()

    processed = 0
    modified = 0
    failed = 0
    
    for json_path in tqdm(json_files, desc="Processing JSONs"):
        # Parse path to get training_loss, dataset, model, seed
        parts = json_path.parts
        try:
            seed_idx = [i for i, p in enumerate(parts) if p.startswith("seed")][0]
            seed_dir = parts[seed_idx]
            model_name = parts[seed_idx - 1]
            dataset = parts[seed_idx - 2]
            training_loss = parts[seed_idx - 3]
            seed_value = int(seed_dir.replace("seed", ""))
        except:
            logger.warning(f"Could not parse path: {json_path}")
            failed += 1
            continue
        
        config_key = (training_loss, dataset, model_name, seed_value)

        if config_key in failed_configs:
            logger.debug(f"Skipping {json_path} due to previous failures for config {config_key}")
            continue

        # Find checkpoint
        dynamic_folder_name = f"{training_loss}_{dataset}_{model_name}_{seed_dir}"
        checkpoint_path = (
            args.checkpoint_base_dir / training_loss / dataset / model_name / seed_dir /
            dynamic_folder_name / "best_model.pth"
        )
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            failed += 1
            failed_configs.add(config_key)
            continue

        if config_key not in model_cache:
            try:
                _, _, test_loader, num_classes = get_data_loaders(
                    dataset=dataset,
                    batch_size=32,
                    seed=seed_value,
                )
                sample_batch = next(iter(test_loader))
                if isinstance(sample_batch, (list, tuple)):
                    sample_images = sample_batch[0]
                else:
                    sample_images = sample_batch
                sample_images = sample_images.to(device)
            except Exception as e:
                logger.error(
                    f"Failed to prepare sample batch for config {config_key}: {e}"
                )
                failed += 1
                failed_configs.add(config_key)
                continue

            try:
                model = load_trained_model(
                    str(checkpoint_path),
                    model_name,
                    num_classes,
                    device,
                    dataset=dataset,
                )
            except Exception as e:
                logger.error(
                    f"Failed to load model for config {config_key} from {checkpoint_path}: {e}"
                )
                failed += 1
                failed_configs.add(config_key)
                continue

            model_cache[config_key] = model
            sample_cache[config_key] = sample_images

        try:
            was_modified = process_json_file(
                json_path,
                model_cache[config_key],
                sample_cache[config_key],
            )
            processed += 1
            if was_modified:
                modified += 1
        except Exception as e:
            logger.error(f"Failed to process {json_path}: {e}")
            failed += 1
    
    logger.info(f"\nSummary:")
    logger.info(f"  Processed: {processed}")
    logger.info(f"  Modified:  {modified}")
    logger.info(f"  Failed:    {failed}")


if __name__ == "__main__":
    main()