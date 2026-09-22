import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional

from .base_calibrator import BaseCalibrator
from utils.logging_config import get_logger

logger = get_logger(__name__)


class VectorScaling(BaseCalibrator):
    """
    Multiclass vector scaling:
      q = softmax(a ⊙ z + b)
    where a, b are per-class vectors.
    """

    def __init__(
        self,
        lr: float = 1e-2,
        epochs: int = 500,
        patience: int = 50,
        lambda_scale_center: float = 1e-4,
        lambda_bias: float = 1e-4,
        device: Optional[str] = None,
        eps: float = 1e-12,
        ts_collapse_std_tol: float = 1e-3,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.lambda_scale_center = float(lambda_scale_center)
        self.lambda_bias = float(lambda_bias)
        self.eps = float(eps)
        self.ts_collapse_std_tol = float(ts_collapse_std_tol)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.n_classes: Optional[int] = None
        self.a: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None

        self.best_epoch: Optional[int] = None
        self.epochs_ran: int = 0
        self.final_val_nll: Optional[float] = None
        self.best_val_nll: Optional[float] = None
        self.final_total_loss: Optional[float] = None
        self.best_total_loss: Optional[float] = None

        self.scale_mean: Optional[float] = None
        self.scale_std: Optional[float] = None
        self.scale_min: Optional[float] = None
        self.scale_max: Optional[float] = None
        self.bias_l2_norm: Optional[float] = None
        self.is_near_temperature_scaling: Optional[bool] = None

    def fit(self, logits: np.ndarray, labels: np.ndarray, features: Optional[np.ndarray] = None) -> None:
        del features
        x_np = np.asarray(logits, dtype=np.float32)
        y_np = np.asarray(labels, dtype=np.int64)
        if x_np.ndim != 2:
            raise ValueError("logits must be a 2D array of shape (n_samples, n_classes)")
        if y_np.ndim != 1:
            raise ValueError("labels must be a 1D array")
        if x_np.shape[0] != y_np.shape[0]:
            raise ValueError("logits rows and labels length must match")

        self.n_classes = int(x_np.shape[1])
        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).to(self.device)

        # Exact required initialization.
        a = torch.ones(self.n_classes, dtype=torch.float32, device=self.device, requires_grad=True)
        b = torch.zeros(self.n_classes, dtype=torch.float32, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([a, b], lr=self.lr)

        best_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        best_total_loss = float("inf")
        best_nll = float("inf")
        final_nll = float("inf")
        final_total_loss = float("inf")
        patience_count = 0

        logger.info(" Fitting vector scaling...")
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            transformed_logits = x * a.unsqueeze(0) + b.unsqueeze(0)
            nll = F.cross_entropy(transformed_logits, y)
            reg = self.lambda_scale_center * torch.sum((a - 1.0) ** 2) + self.lambda_bias * torch.sum(
                b * b
            )
            loss = nll + reg

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered while fitting vector scaling")

            loss.backward()
            optimizer.step()

            self.epochs_ran = epoch + 1
            final_nll = float(nll.detach().item())
            final_total_loss = float(loss.detach().item())

            if final_total_loss < best_total_loss:
                best_total_loss = final_total_loss
                best_nll = final_nll
                self.best_epoch = epoch
                best_state = (a.detach().clone(), b.detach().clone())
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    logger.info(
                        " Vector scaling early stopping at epoch %d (best epoch: %d)",
                        epoch,
                        -1 if self.best_epoch is None else self.best_epoch,
                    )
                    break

        if best_state is None:
            raise RuntimeError("Vector scaling failed to find a finite optimization state")

        a_best, b_best = best_state
        self.a = a_best.detach().cpu().numpy()
        self.b = b_best.detach().cpu().numpy()
        self.final_val_nll = final_nll
        self.best_val_nll = best_nll
        self.final_total_loss = final_total_loss
        self.best_total_loss = best_total_loss

        self.scale_mean = float(np.mean(self.a))
        self.scale_std = float(np.std(self.a))
        self.scale_min = float(np.min(self.a))
        self.scale_max = float(np.max(self.a))
        self.bias_l2_norm = float(np.linalg.norm(self.b))
        self.is_near_temperature_scaling = bool(
            self.scale_std <= self.ts_collapse_std_tol and self.bias_l2_norm <= self.ts_collapse_std_tol
        )
        self.is_fitted = True

    def calibrate(self, logits: np.ndarray, features: Optional[np.ndarray] = None) -> np.ndarray:
        del features
        if not self.is_fitted or self.a is None or self.b is None or self.n_classes is None:
            raise RuntimeError("Model must be fitted before calibration")

        x_np = np.asarray(logits, dtype=np.float32)
        if x_np.ndim != 2 or x_np.shape[1] != self.n_classes:
            raise ValueError(
                f"logits must have shape (n_samples, {self.n_classes}) for calibrated inference"
            )
        x = torch.from_numpy(x_np).to(self.device)
        a = torch.from_numpy(self.a.astype(np.float32)).to(self.device)
        b = torch.from_numpy(self.b.astype(np.float32)).to(self.device)

        with torch.no_grad():
            transformed_logits = x * a.unsqueeze(0) + b.unsqueeze(0)
            probs = torch.softmax(transformed_logits, dim=1).cpu().numpy()

        if not np.all(np.isfinite(probs)):
            raise ValueError("vector_scaling: calibrated output contains non-finite values")
        probs = np.clip(probs, 0.0, 1.0)
        row_sums = np.sum(probs, axis=1, keepdims=True)
        probs = probs / np.clip(row_sums, self.eps, None)
        return probs

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_classes": self.n_classes,
            "lr": self.lr,
            "epochs": self.epochs,
            "patience": self.patience,
            "epochs_ran": self.epochs_ran,
            "lambda_scale_center": self.lambda_scale_center,
            "lambda_bias": self.lambda_bias,
            "best_epoch": self.best_epoch,
            "final_val_nll": self.final_val_nll,
            "best_val_nll": self.best_val_nll,
            "final_total_loss": self.final_total_loss,
            "best_total_loss": self.best_total_loss,
            "scale_mean": self.scale_mean,
            "scale_std": self.scale_std,
            "scale_min": self.scale_min,
            "scale_max": self.scale_max,
            "bias_l2_norm": self.bias_l2_norm,
            "ts_collapse_std_tol": self.ts_collapse_std_tol,
            "is_near_temperature_scaling": self.is_near_temperature_scaling,
        }

    def save(self, path: str) -> None:
        torch.save(
            {
                "a": self.a,
                "b": self.b,
                "n_classes": self.n_classes,
                "params": self.get_params(),
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.a = np.asarray(checkpoint["a"], dtype=np.float32)
        self.b = np.asarray(checkpoint["b"], dtype=np.float32)
        self.n_classes = int(checkpoint["n_classes"])

        params = checkpoint.get("params", {})
        self.best_epoch = params.get("best_epoch")
        self.epochs_ran = int(params.get("epochs_ran", 0))
        self.final_val_nll = params.get("final_val_nll")
        self.best_val_nll = params.get("best_val_nll")
        self.final_total_loss = params.get("final_total_loss")
        self.best_total_loss = params.get("best_total_loss")
        self.scale_mean = params.get("scale_mean")
        self.scale_std = params.get("scale_std")
        self.scale_min = params.get("scale_min")
        self.scale_max = params.get("scale_max")
        self.bias_l2_norm = params.get("bias_l2_norm")
        self.is_near_temperature_scaling = params.get("is_near_temperature_scaling")
        self.is_fitted = True
