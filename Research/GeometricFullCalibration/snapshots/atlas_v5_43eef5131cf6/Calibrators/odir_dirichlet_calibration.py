import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple

from .base_calibrator import BaseCalibrator
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ODIRDirichletCalibration(BaseCalibrator):
    """
    Standard ODIR-style multiclass Dirichlet calibration baseline.

    Formulation:
      x = log(clip(p, eps, 1.0))
      z = x @ W^T + b
      q = softmax(z)

    Training objective:
      NLL(q, y)
      + lambda_off * ||offdiag(W)||^2
      + lambda_diag * ||diag(W) - 1||^2
      + lambda_bias * ||b||^2
    """

    def __init__(
        self,
        eps: float = 1e-8,
        lr: float = 1e-2,
        epochs: int = 500,
        patience: int = 50,
        lambda_off: float = 1e-3,
        lambda_diag: float = 1e-3,
        lambda_bias: float = 1e-3,
        optimizer_name: str = "adam",
        device: Optional[str] = None,
        input_prob_tolerance: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.eps = float(eps)
        self.lr = float(lr)
        self.epochs = int(epochs)
        self.patience = int(patience)
        self.lambda_off = float(lambda_off)
        self.lambda_diag = float(lambda_diag)
        self.lambda_bias = float(lambda_bias)
        self.optimizer_name = optimizer_name.lower()
        self.input_prob_tolerance = float(input_prob_tolerance)
        self.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.n_classes: Optional[int] = None
        self.W: Optional[np.ndarray] = None
        self.b: Optional[np.ndarray] = None
        self.fit_input_mode: Optional[str] = None
        self.last_calibrate_input_mode: Optional[str] = None
        self.final_val_nll: Optional[float] = None
        self.best_val_nll: Optional[float] = None
        self.best_epoch: Optional[int] = None
        self.epochs_ran: int = 0

    def _is_probability_matrix(self, x: np.ndarray) -> bool:
        if x.ndim != 2:
            return False
        if not np.all(np.isfinite(x)):
            return False
        if np.any(x < -self.input_prob_tolerance):
            return False
        row_sums = np.sum(x, axis=1)
        return np.allclose(row_sums, 1.0, atol=self.input_prob_tolerance, rtol=0.0)

    def _softmax_numpy(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(shifted)
        denom = np.sum(exp_x, axis=1, keepdims=True)
        return exp_x / np.clip(denom, self.eps, None)

    def _to_probs(self, logits_or_probs: np.ndarray) -> Tuple[np.ndarray, str]:
        x = np.asarray(logits_or_probs, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("Input must be a 2D array of shape (n_samples, n_classes)")

        if self._is_probability_matrix(x):
            probs = np.clip(x, self.eps, 1.0)
            probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), self.eps, None)
            return probs, "probabilities"

        probs = self._softmax_numpy(x)
        probs = np.clip(probs, self.eps, 1.0)
        probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), self.eps, None)
        return probs, "logits"

    def _validate_output(self, probs: np.ndarray, name: str) -> np.ndarray:
        if probs.ndim != 2:
            raise ValueError(f"{name}: output must be 2D NxK probabilities")
        if not np.all(np.isfinite(probs)):
            raise ValueError(f"{name}: output contains non-finite values")
        probs = np.clip(probs, self.eps, 1.0)
        probs = probs / np.clip(np.sum(probs, axis=1, keepdims=True), self.eps, None)
        if not np.all(np.isfinite(probs)):
            raise ValueError(f"{name}: output is non-finite after normalization")
        return probs

    def _build_optimizer(self, params: list[torch.nn.Parameter]) -> torch.optim.Optimizer:
        if self.optimizer_name == "adam":
            return torch.optim.Adam(params, lr=self.lr)
        raise ValueError(f"Unsupported optimizer_name={self.optimizer_name}; supported: adam")

    def fit(
        self, logits_or_probs: np.ndarray, labels: np.ndarray, features: Optional[np.ndarray] = None
    ) -> None:
        del features
        probs, mode = self._to_probs(logits_or_probs)
        self.fit_input_mode = mode
        logger.info(" Fitting ODIR Dirichlet calibration (fit input interpreted as %s)", mode)

        labels = np.asarray(labels)
        if labels.ndim != 1:
            raise ValueError("labels must be a 1D array")
        if probs.shape[0] != labels.shape[0]:
            raise ValueError("Input rows and labels length must match")

        self.n_classes = int(probs.shape[1])
        x_np = np.log(np.clip(probs, self.eps, 1.0)).astype(np.float32)
        y_np = labels.astype(np.int64)

        x = torch.from_numpy(x_np).to(self.device)
        y = torch.from_numpy(y_np).to(self.device)

        # Deterministic baseline initialization: exact identity and zero bias.
        W = torch.eye(self.n_classes, dtype=torch.float32, device=self.device, requires_grad=True)
        b = torch.zeros(self.n_classes, dtype=torch.float32, device=self.device, requires_grad=True)

        optimizer = self._build_optimizer([W, b])

        best_state: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        best_total_loss = float("inf")
        best_nll = float("inf")
        patience_count = 0
        final_nll = float("inf")

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = x @ W.T + b.unsqueeze(0)
            nll = F.cross_entropy(logits, y)

            diag_W = torch.diag(W)
            offdiag_W = W - torch.diag(diag_W)
            reg_off = self.lambda_off * torch.sum(offdiag_W * offdiag_W)
            reg_diag = self.lambda_diag * torch.sum((diag_W - 1.0) ** 2)
            reg_bias = self.lambda_bias * torch.sum(b * b)
            loss = nll + reg_off + reg_diag + reg_bias

            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss encountered while fitting ODIR Dirichlet")

            loss.backward()
            optimizer.step()

            self.epochs_ran = epoch + 1
            final_nll = float(nll.detach().item())
            total_loss = float(loss.detach().item())

            if total_loss < best_total_loss:
                best_total_loss = total_loss
                best_nll = final_nll
                self.best_epoch = epoch
                best_state = (W.detach().clone(), b.detach().clone())
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    logger.info(
                        " ODIR early stopping at epoch %d (best epoch: %d)",
                        epoch,
                        -1 if self.best_epoch is None else self.best_epoch,
                    )
                    break

        if best_state is None:
            raise RuntimeError("ODIR Dirichlet failed to find a finite optimization state")

        W_best, b_best = best_state
        self.W = W_best.detach().cpu().numpy()
        self.b = b_best.detach().cpu().numpy()
        self.final_val_nll = final_nll
        self.best_val_nll = best_nll
        self.is_fitted = True

    def calibrate(
        self, logits_or_probs: np.ndarray, features: Optional[np.ndarray] = None
    ) -> np.ndarray:
        del features
        if not self.is_fitted or self.W is None or self.b is None or self.n_classes is None:
            raise RuntimeError("Model must be fitted before calibration")

        probs, mode = self._to_probs(logits_or_probs)
        self.last_calibrate_input_mode = mode

        x_np = np.log(np.clip(probs, self.eps, 1.0)).astype(np.float32)
        x = torch.from_numpy(x_np).to(self.device)
        W = torch.from_numpy(self.W.astype(np.float32)).to(self.device)
        b = torch.from_numpy(self.b.astype(np.float32)).to(self.device)

        with torch.no_grad():
            logits = x @ W.T + b.unsqueeze(0)
            q = torch.softmax(logits, dim=1).cpu().numpy()
        return self._validate_output(q, "odir_dirichlet")

    def get_params(self) -> Dict[str, Any]:
        return {
            "n_classes": self.n_classes,
            "eps": self.eps,
            "optimizer": self.optimizer_name,
            "lr": self.lr,
            "epochs": self.epochs,
            "patience": self.patience,
            "epochs_ran": self.epochs_ran,
            "lambda_off": self.lambda_off,
            "lambda_diag": self.lambda_diag,
            "lambda_bias": self.lambda_bias,
            "fit_input_mode": self.fit_input_mode,
            "calibrate_input_mode": self.last_calibrate_input_mode,
            "best_epoch": self.best_epoch,
            "final_val_nll": self.final_val_nll,
            "best_val_nll": self.best_val_nll,
        }
