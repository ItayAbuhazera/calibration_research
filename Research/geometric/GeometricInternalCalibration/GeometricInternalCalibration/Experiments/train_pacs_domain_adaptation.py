#!/usr/bin/env python3
"""
Train ResNet models on PACS with domain adaptation (train on 3 domains, test on 1).

Outputs (per seed / method / target domain):
    aaai_full_experiments/results/
        domain_adaptation/{method}/{target_domain}/{model_name}/seed{seed}/
            best_model.pth
            config.json
            results.json
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import torch
from torch import nn, optim
from torch.cuda import amp
from torch.utils.data import DataLoader
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Calibrators.calibration_utils import compute_ece
from Losses.loss import get_loss_function
from Data.pacs_domain_adaptation import (
    get_domain_adaptation_loaders,
    get_domain_adaptation_augmix_loaders,
)
from Data.augmix_transforms import AugMixTransforms


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
from utils.logging_config import get_logger
logger = get_logger(__name__)


@dataclass
class TrainingConfig:
    method: str = "cross_entropy"
    dataset: str = "pacs"
    model: str = "resnet18"
    target_domain: str = "photo"
    seed: int = 1
    num_classes: int = 7
    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-3
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: int = 0
    accuracy_threshold: float = 0.0
    ece_bins: int = 15
    grad_clip: float = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ResNet models on PACS with domain adaptation."
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "aaai_full_experiments" / "results"),
        help="Root directory for experiment outputs.",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "brier", "augmix"],
        help="Training objective / calibration method.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="pacs",
        help="Dataset name (fixed to PACS, used for bookkeeping only).",
    )
    parser.add_argument(
        "--target-domain",
        type=str,
        required=True,
        choices=["photo", "art_painting", "cartoon", "sketch"],
        help="Held-out target domain used only for testing.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet50"],
        help="Backbone architecture.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--num-classes", type=int, default=7)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force training on CPU even if CUDA is available.",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (AMP) training.",
    )
    parser.add_argument(
        "--augmix-consistency-weight",
        type=float,
        default=12.0,
        help="Weight for AugMix JS-consistency term.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_dataloaders(
    target_domain: str,
    batch_size: int,
    num_workers: int,
    method: str,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train/validation/test loaders for PACS domain adaptation.

    For AugMix, uses the domain-adaptation AugMix loaders for train/val (3 source
    domains only) and the standard domain adaptation loader to obtain the
    target-domain test loader.
    """
    if method == "augmix":
        augmix_transform = AugMixTransforms(dataset_name="pacs")
        train_loader, val_loader = get_domain_adaptation_augmix_loaders(
            target_domain=target_domain,
            batch_size=batch_size,
            random_seed=seed,
            valid_size=0.1,
            augmix_transform=augmix_transform,
        )
        # Get test loader (and discard its train/val) from the standard loader
        _, _, test_loader = get_domain_adaptation_loaders(
            target_domain=target_domain,
            batch_size=batch_size,
            random_seed=seed,
            valid_size=0.1,
        )
    else:
        train_loader, val_loader, test_loader = get_domain_adaptation_loaders(
            target_domain=target_domain,
            batch_size=batch_size,
            random_seed=seed,
            valid_size=0.1,
        )

    # Override num_workers / pin_memory on created loaders.
    # NOTE: This val_loader uses the same split (via random_seed) that will be used
    # for geometric calibration fitting in run_post_hoc_calibration.py.
    def _rebuild(loader: DataLoader, shuffle: bool) -> DataLoader:
        return DataLoader(
            loader.dataset,
            batch_size=batch_size,
            shuffle=shuffle if loader.sampler is None else False,
            sampler=loader.sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

    train_loader = _rebuild(train_loader, shuffle=True)
    val_loader = _rebuild(val_loader, shuffle=False)
    test_loader = _rebuild(test_loader, shuffle=False)
    return train_loader, val_loader, test_loader


def build_model(model_name: str, num_classes: int, device: torch.device) -> nn.Module:
    if model_name == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif model_name == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    in_features = backbone.fc.in_features
    backbone.fc = nn.Linear(in_features, num_classes)
    return backbone.to(device)


def create_optimizer_and_scheduler(
    model: nn.Module, args: argparse.Namespace
) -> Tuple[optim.Optimizer, optim.lr_scheduler._LRScheduler]:
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )

    if args.warmup_epochs > 0:
        cosine_epochs = max(1, args.epochs - args.warmup_epochs)
        cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cosine_epochs)
        warmup = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-4,
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )
    return optimizer, scheduler


def training_paths(args: argparse.Namespace) -> Dict[str, Path]:
    output_root = Path(args.output_root).expanduser().resolve()
    seed_dir = (
        output_root
        / "domain_adaptation"
        / args.method
        / args.target_domain
        / args.model_name
        / f"seed{args.seed}"
    )
    exp_dir = seed_dir
    exp_dir.mkdir(parents=True, exist_ok=True)
    return {
        "seed_dir": seed_dir,
        "exp_dir": exp_dir,
        "config": exp_dir / "config.json",
        "results": exp_dir / "results.json",
        "best_model": exp_dir / "best_model.pth",
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn,
    ece_bins: int,
    is_augmix: bool = False,
) -> Dict[str, float]:
    """
    Standard evaluation on a (images, labels) DataLoader.
    AugMix loaders for validation/test return (image, label), so is_augmix=False.
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    probs_all: List[torch.Tensor] = []
    labels_all: List[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            if is_augmix:
                # For safety, but our val/test loaders for AugMix already yield standard (x, y)
                images, targets = batch
            else:
                images, targets = batch

            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = loss_fn(logits, targets) / targets.size(0)
            total_loss += loss.item() * targets.size(0)
            preds = logits.argmax(dim=1)
            total_correct += preds.eq(targets).sum().item()
            total_samples += targets.size(0)
            probs_all.append(torch.softmax(logits, dim=1).cpu())
            labels_all.append(targets.cpu())

    avg_loss = total_loss / max(1, total_samples)
    acc = 100.0 * total_correct / max(1, total_samples)
    probs_np = torch.cat(probs_all).numpy()
    labels_np = torch.cat(labels_all).numpy()
    ece = compute_ece(probs_np, labels_np, n_bins=ece_bins)
    return {"loss": float(avg_loss), "accuracy": float(acc), "ece": float(ece)}


def js_divergence(p_clean: torch.Tensor, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between three distributions (per-sample)."""
    m = (p_clean + p1 + p2) / 3.0
    log_m = torch.log(m + 1e-12)
    log_p_clean = torch.log(p_clean + 1e-12)
    log_p1 = torch.log(p1 + 1e-12)
    log_p2 = torch.log(p2 + 1e-12)

    kl_clean = torch.sum(p_clean * (log_p_clean - log_m), dim=1)
    kl_1 = torch.sum(p1 * (log_p1 - log_m), dim=1)
    kl_2 = torch.sum(p2 * (log_p2 - log_m), dim=1)
    js = (kl_clean + kl_1 + kl_2) / 3.0
    return js


def run_training(args: argparse.Namespace) -> None:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    set_seed(args.seed)

    train_loader, val_loader, test_loader = build_dataloaders(
        target_domain=args.target_domain,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        method=args.method,
        seed=args.seed,
    )

    model = build_model(args.model_name, args.num_classes, device)
    optimizer, scheduler = create_optimizer_and_scheduler(model, args)
    scaler = amp.GradScaler(enabled=not args.no_amp and device.type == "cuda")

    # Map method name to base loss for training
    if args.method == "cross_entropy":
        base_loss_name = "cross_entropy"
    elif args.method == "brier":
        base_loss_name = "brier_score"
    elif args.method == "augmix":
        base_loss_name = "cross_entropy"
    else:
        raise ValueError(f"Unknown method: {args.method}")

    loss_fn = get_loss_function(base_loss_name)

    paths = training_paths(args)
    config = TrainingConfig(
        method=args.method,
        dataset=args.dataset,
        model=args.model_name,
        target_domain=args.target_domain,
        seed=args.seed,
        num_classes=args.num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        accuracy_threshold=0.0,
        ece_bins=args.ece_bins,
        grad_clip=args.grad_clip,
    )
    with open(paths["config"], "w", encoding="utf-8") as fp:
        json.dump(asdict(config), fp, indent=2)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss_source": [],
        "val_acc_source": [],
        "val_ece_source": [],
        "test_loss_target": [],
        "test_acc_target": [],
        "test_ece_target": [],
    }

    best_source_acc = 0.0
    best_epoch = 0

    start_time = time.time()
    total_train_samples = len(train_loader.dataset)
    logger.info(
        f"Starting training: method={args.method}, model={args.model_name}, "
        f"target_domain={args.target_domain}, seed={args.seed}, "
        f"train_samples={total_train_samples}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        correct = 0
        seen = 0

        for batch in train_loader:
            if args.method == "augmix":
                clean, aug1, aug2, targets = batch
                clean = clean.to(device, non_blocking=True)
                aug1 = aug1.to(device, non_blocking=True)
                aug2 = aug2.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
            else:
                images, targets = batch
                clean = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp.autocast(enabled=scaler.is_enabled()):
                if args.method == "augmix":
                    logits_clean = model(clean)
                    logits_aug1 = model(aug1)
                    logits_aug2 = model(aug2)

                    base_loss = loss_fn(logits_clean, targets) / targets.size(0)

                    p_clean = torch.softmax(logits_clean, dim=1)
                    p1 = torch.softmax(logits_aug1, dim=1)
                    p2 = torch.softmax(logits_aug2, dim=1)
                    js = js_divergence(p_clean, p1, p2)
                    consistency_loss = js.mean()

                    loss = base_loss + args.augmix_consistency_weight * consistency_loss
                    logits_for_metrics = logits_clean
                else:
                    logits = model(clean)
                    loss = loss_fn(logits, targets) / targets.size(0)
                    logits_for_metrics = logits

            # NaN/Inf protection
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            has_nan_grad = False
            for param in model.parameters():
                if param.grad is not None and (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                ):
                    has_nan_grad = True
                    break
            if has_nan_grad:
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                continue

            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item() * targets.size(0)
            preds = logits_for_metrics.argmax(dim=1)
            correct += preds.eq(targets).sum().item()
            seen += targets.size(0)

        scheduler.step()

        train_loss_avg = epoch_loss / max(1, seen)
        train_acc = 100.0 * correct / max(1, seen)

        if epoch % max(1, args.log_every) == 0:
            logger.info(
                f"[Train {epoch:03d}] Loss {train_loss_avg:.4f} | Acc {train_acc:.2f}%"
            )

        # Evaluate on source (val_loader) and target (test_loader)
        val_metrics = evaluate_model(
            model,
            val_loader,
            device,
            loss_fn,
            ece_bins=args.ece_bins,
            is_augmix=False,
        )
        test_metrics = evaluate_model(
            model,
            test_loader,
            device,
            loss_fn,
            ece_bins=args.ece_bins,
            is_augmix=False,
        )

        history["train_loss"].append(float(train_loss_avg))
        history["train_acc"].append(float(train_acc))
        history["val_loss_source"].append(val_metrics["loss"])
        history["val_acc_source"].append(val_metrics["accuracy"])
        history["val_ece_source"].append(val_metrics["ece"])
        history["test_loss_target"].append(test_metrics["loss"])
        history["test_acc_target"].append(test_metrics["accuracy"])
        history["test_ece_target"].append(test_metrics["ece"])

        if epoch % max(1, args.log_every) == 0 or epoch == 1:
            logger.info(
                f"[Epoch {epoch:03d}] "
                f"Train Loss {train_loss_avg:.4f} | Train Acc {train_acc:.2f}% | "
                f"Source Acc {val_metrics['accuracy']:.2f}% | Source ECE {val_metrics['ece']:.4f} | "
                f"Target Acc {test_metrics['accuracy']:.2f}% | Target ECE {test_metrics['ece']:.4f} | "
                f"Best Source {best_source_acc:.2f}% (Epoch {best_epoch})"
            )

        # Selection based on SOURCE (validation) accuracy only.
        # In domain adaptation we do not assume access to target labels during training.
        if val_metrics["accuracy"] > best_source_acc:
            best_source_acc = val_metrics["accuracy"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "source_acc": best_source_acc,
                    "args": vars(args),
                },
                paths["best_model"],
            )

    training_time = time.time() - start_time

    # Reload best checkpoint before final evaluation
    checkpoint = torch.load(paths["best_model"], map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    final_val_metrics = evaluate_model(
        model, val_loader, device, loss_fn, ece_bins=args.ece_bins, is_augmix=False
    )
    final_test_metrics = evaluate_model(
        model, test_loader, device, loss_fn, ece_bins=args.ece_bins, is_augmix=False
    )

    results = {
        "experiment_name": f"{args.method}_{args.dataset}_{args.model_name}_target_{args.target_domain}_seed{args.seed}",
        "config": asdict(config),
        "training_time": training_time,
        "best_source_acc": best_source_acc,
        "best_epoch": best_epoch,
        "training_history": history,
        "final_source_metrics": final_val_metrics,
        "final_target_metrics": final_test_metrics,
    }

    with open(paths["results"], "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)

    logger.info(
        "Finished training. Best source-domain accuracy "
        f"{best_source_acc:.2f}% at epoch {best_epoch}. "
        f"Model saved to {paths['best_model']}, metrics to {paths['results']}."
    )


def main() -> None:
    args = parse_args()
    run_training(args)


if __name__ == "__main__":
    main()


