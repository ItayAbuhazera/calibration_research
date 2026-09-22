"""
Parameterized Temperature Scaling (PTS) — sample-adaptive scalar temperature calibrator.

Maps each test sample's logit-derived features to a positive scalar temperature T(x),
then returns softmax(logits / T(x)).

Structural lemma (preserved here as a theorem, not just an assertion):
    For any positive scalar function T: R^K -> R_{>0},
    argmax_k z_k = argmax_k (z_k / T(z)).
    Proof: dividing all logits by a positive constant preserves relative order.

This means PTS is argmax-invariant by construction, regardless of what features
are used to predict T. The assertion in `calibrate()` catches implementation bugs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _extract_logit_features(logits: np.ndarray) -> np.ndarray:
    """Compute 5 scalar features per sample from logits only."""
    probs = F.softmax(torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
    max_prob = probs.max(axis=1, keepdims=True)  # [N, 1]
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1, keepdims=True)  # [N, 1]

    sorted_probs = np.sort(probs, axis=1)[:, ::-1]
    top1_margin = (sorted_probs[:, 0] - sorted_probs[:, 1]).reshape(-1, 1)

    logit_norm = np.linalg.norm(logits, axis=1, keepdims=True)
    top_logit = logits.max(axis=1, keepdims=True)

    return np.concatenate([max_prob, entropy, top1_margin, logit_norm, top_logit], axis=1).astype(
        np.float32
    )


class _TemperatureNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        log_t = self.net(x).squeeze(-1)
        # softplus ensures T > 0; shift by 0.05 prevents extreme shrinkage
        return F.softplus(log_t) + 0.05


class ParameterizedTemperatureScaling:
    """
    Post-hoc parameterized temperature scaling (PTS).

    Predicts a positive per-sample scalar temperature from logit-derived features
    and returns softmax(logits / T(x)). By the scalar-temperature structural lemma,
    this cannot change the argmax prediction.

    Interface:
        fit(logits_val, labels_val)
        calibrate(logits_test) -> np.ndarray [N, C] probabilities
        get_params() -> dict
    """

    def __init__(
        self,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        epochs: int = 50,
        patience: int = 5,
        device: Optional[str] = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._net: Optional[_TemperatureNet] = None
        self._feature_mean: Optional[np.ndarray] = None
        self._feature_std: Optional[np.ndarray] = None
        self._training_history: list = []

    def fit(
        self,
        logits_val: np.ndarray,
        labels_val: np.ndarray,
    ) -> Dict[str, Any]:
        """Fit the temperature network to minimize validation NLL."""
        feats = _extract_logit_features(logits_val)
        self._feature_mean = feats.mean(axis=0)
        self._feature_std = np.clip(feats.std(axis=0), 1e-6, None)
        feats_norm = (feats - self._feature_mean) / self._feature_std

        x = torch.tensor(feats_norm, dtype=torch.float32, device=self._device)
        z = torch.tensor(logits_val, dtype=torch.float32, device=self._device)
        y = torch.tensor(labels_val, dtype=torch.long, device=self._device)

        input_dim = feats_norm.shape[1]
        self._net = _TemperatureNet(input_dim, self.hidden_dim).to(self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        best_loss = float("inf")
        best_state = None
        no_improve = 0
        history = []

        self._net.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            T = self._net(x)  # [N]
            scaled_logits = z / T.unsqueeze(-1)
            loss = F.cross_entropy(scaled_logits, y)
            loss.backward()
            optimizer.step()
            val_loss = float(loss.item())
            history.append(val_loss)

            if val_loss < best_loss - 1e-6:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            if no_improve >= self.patience:
                break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        self._training_history = history

        return {"best_val_nll": best_loss, "epochs_trained": len(history)}

    def calibrate(self, logits_test: np.ndarray) -> np.ndarray:
        """
        Apply PTS calibration to test logits.

        Asserts argmax invariance: T(x) > 0 ensures logit ordering is preserved.
        Raises AssertionError if implementation produces different predictions
        (which would indicate a bug — not a theoretical failure).
        """
        if self._net is None or self._feature_mean is None:
            raise RuntimeError("Call fit() before calibrate()")

        feats = _extract_logit_features(logits_test)
        feats_norm = (feats - self._feature_mean) / self._feature_std

        x = torch.tensor(feats_norm, dtype=torch.float32, device=self._device)
        z = torch.tensor(logits_test, dtype=torch.float32, device=self._device)

        self._net.eval()
        with torch.no_grad():
            T = self._net(x)
            scaled_logits = z / T.unsqueeze(-1)
            probs = F.softmax(scaled_logits, dim=-1)

        probs_np = probs.cpu().numpy().astype(np.float64)

        # Structural lemma check: positive scalar T cannot change argmax
        base_preds = np.argmax(
            F.softmax(torch.tensor(logits_test, dtype=torch.float32), dim=-1).numpy(), axis=1
        )
        pts_preds = np.argmax(probs_np, axis=1)
        if not np.array_equal(base_preds, pts_preds):
            n_diff = int(np.sum(base_preds != pts_preds))
            raise AssertionError(
                f"PTS violated argmax invariance on {n_diff}/{len(base_preds)} samples. "
                "A positive scalar T cannot change argmax — this indicates a numerical bug."
            )

        return probs_np

    def get_params(self) -> Dict[str, Any]:
        if self._net is None:
            return {"fitted": False}
        return {
            "fitted": True,
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "training_history": self._training_history,
            "feature_mean": self._feature_mean.tolist() if self._feature_mean is not None else None,
            "feature_std": self._feature_std.tolist() if self._feature_std is not None else None,
        }
