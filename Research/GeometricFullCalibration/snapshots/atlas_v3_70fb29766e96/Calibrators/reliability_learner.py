"""
Reliability learner: predicts BASE-MODEL correctness from a feature block.

This is a Study-A (reliability / information) object, NOT a decision
corrector. Its target is fixed:

    C_base(x) = 1[ argmax(base_probs(x)) == y ]

and it never changes a prediction. Its output is a scalar confidence in the
base model's existing decision, so it is evaluated exactly like every other
scalar-confidence method in this benchmark (top-label ECE, adaptive ECE,
correctness AUROC, AURC, risk-coverage, coverage at matched risk).

MATCHED-ARM CONTRACT
--------------------
Every arm of the incremental-geometry experiment uses this same class with the
same `seed`, the same fit/select indices, the same optimizer, the same epoch
budget, the same early-stopping rule and the same hyperparameter grid. Arms
differ ONLY in the feature block handed to `fit`. Because the geometry block
is present-but-zeroed in the control arms, the input dimensionality and the
trainable parameter count are identical across the zero / shuffled / logit /
hidden arms, and `initial_parameter_fingerprint()` is identical for all of
them before any data is seen.

Determinism: `torch.manual_seed(seed)` is called immediately before each
network is constructed, and training is full-batch, so two arms with the same
seed start from bit-identical parameters and see identical batch ordering.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _ReliabilityNet(nn.Module):
    """Deliberately small MLP: input -> hidden -> 1 logit."""

    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _auroc(scores: np.ndarray, correct: np.ndarray) -> float:
    """Mid-rank Mann-Whitney AUROC; 0.5 when degenerate."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    n_pos = float(correct.sum())
    n_neg = float(correct.size) - n_pos
    if n_pos == 0.0 or n_neg == 0.0:
        return 0.5
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * ((i + 1) + (j + 1))
        i = j + 1
    return float((ranks[correct == 1.0].sum() - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg))


class ReliabilityLearner:
    """
    fit(features_fit, correct_fit, features_select, correct_select) -> dict
    predict_confidence(features) -> np.ndarray in (0, 1)
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 16,
        lr: float = 1e-2,
        epochs: int = 400,
        patience: int = 40,
        weight_decay_grid: Sequence[float] = (1e-4, 1e-3, 1e-2),
        seed: int = 20260920,
        device: Optional[str] = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.weight_decay_grid = tuple(weight_decay_grid)
        self.seed = int(seed)
        # CPU by default: the model is tiny and CPU keeps runs bit-reproducible.
        self._device = torch.device(device if device is not None else "cpu")
        self._net: Optional[_ReliabilityNet] = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._input_dim: Optional[int] = None
        self._selected_weight_decay: Optional[float] = None
        self._selection_results: Optional[List[Dict[str, Any]]] = None

    # -------------------------------------------------------------- helpers
    def _build_net(self, input_dim: int) -> _ReliabilityNet:
        torch.manual_seed(self.seed)
        return _ReliabilityNet(input_dim, self.hidden_dim).to(self._device)

    def initial_parameter_fingerprint(self, input_dim: int) -> str:
        """sha256 of the freshly-initialized parameters, before any training.

        Two arms with the same seed and input dimension must produce the same
        fingerprint; the regression test for matched initialization uses this.
        """
        net = self._build_net(input_dim)
        h = hashlib.sha256()
        for name, p in sorted(net.state_dict().items()):
            h.update(name.encode("utf-8"))
            h.update(np.ascontiguousarray(p.detach().cpu().numpy()).tobytes())
        return h.hexdigest()

    def parameter_count(self, input_dim: int) -> int:
        net = self._build_net(input_dim)
        return int(sum(p.numel() for p in net.parameters() if p.requires_grad))

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        assert self._mean is not None and self._std is not None
        return (np.asarray(x, dtype=np.float64) - self._mean) / self._std

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        features_fit: np.ndarray,
        correct_fit: np.ndarray,
        features_select: np.ndarray,
        correct_select: np.ndarray,
    ) -> Dict[str, Any]:
        features_fit = np.asarray(features_fit, dtype=np.float64)
        features_select = np.asarray(features_select, dtype=np.float64)
        if features_fit.shape[1] != features_select.shape[1]:
            raise ValueError("fit/select feature dimensions differ")
        self._input_dim = int(features_fit.shape[1])

        # Standardization statistics come from the FIT split only -- never the
        # select split and never test.
        self._mean = features_fit.mean(axis=0)
        self._std = np.maximum(features_fit.std(axis=0), 1e-8)

        x_fit = torch.tensor(self._standardize(features_fit), dtype=torch.float32, device=self._device)
        x_sel = torch.tensor(self._standardize(features_select), dtype=torch.float32, device=self._device)
        y_fit = torch.tensor(np.asarray(correct_fit, dtype=np.float64), dtype=torch.float32, device=self._device)
        correct_select = np.asarray(correct_select, dtype=np.float64)

        results: List[Dict[str, Any]] = []
        best: Optional[Dict[str, Any]] = None
        for weight_decay in self.weight_decay_grid:
            net = self._build_net(self._input_dim)
            optimizer = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=weight_decay)
            best_sel_auroc = -np.inf
            best_state = None
            no_improve = 0
            for _ in range(self.epochs):
                net.train()
                optimizer.zero_grad()
                loss = F.binary_cross_entropy_with_logits(net(x_fit), y_fit)
                loss.backward()
                optimizer.step()

                # Early stopping on the SELECT split's ranking quality: the
                # primary scientific metric is correctness ranking, so the
                # stopping rule optimizes AUROC rather than net flips.
                net.eval()
                with torch.no_grad():
                    sel_scores = net(x_sel).cpu().numpy()
                sel_auroc = _auroc(sel_scores, correct_select)
                if sel_auroc > best_sel_auroc + 1e-6:
                    best_sel_auroc = sel_auroc
                    best_state = {k: v.clone() for k, v in net.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    break
            if best_state is not None:
                net.load_state_dict(best_state)
            entry = {"weight_decay": float(weight_decay), "select_auroc": float(best_sel_auroc), "net": net}
            results.append(entry)
            if best is None or entry["select_auroc"] > best["select_auroc"]:
                best = entry

        assert best is not None
        self._net = best["net"]
        self._selected_weight_decay = float(best["weight_decay"])
        self._selection_results = [
            {k: v for k, v in r.items() if k != "net"} for r in results
        ]
        return {
            "selected_weight_decay": self._selected_weight_decay,
            "select_auroc": float(best["select_auroc"]),
            "selection_results": self._selection_results,
            "input_dim": self._input_dim,
            "trainable_params": self.parameter_count(self._input_dim),
            "seed": self.seed,
            "hyperparameter_trials": len(self.weight_decay_grid),
        }

    # -------------------------------------------------------------- predict
    def predict_confidence(self, features: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Call fit() before predict_confidence().")
        features = np.asarray(features, dtype=np.float64)
        if features.shape[1] != self._input_dim:
            raise ValueError(
                f"feature dim {features.shape[1]} != fitted dim {self._input_dim}"
            )
        x = torch.tensor(self._standardize(features), dtype=torch.float32, device=self._device)
        self._net.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self._net(x)).cpu().numpy().astype(np.float64)
        return np.clip(probs, 1e-6, 1.0 - 1e-6)

    def get_params(self) -> Dict[str, Any]:
        return {
            "hidden_dim": self.hidden_dim,
            "lr": self.lr,
            "epochs": self.epochs,
            "patience": self.patience,
            "weight_decay_grid": list(self.weight_decay_grid),
            "seed": self.seed,
            "input_dim": self._input_dim,
            "selected_weight_decay": self._selected_weight_decay,
            "selection_results": self._selection_results,
            "fitted": self._net is not None,
        }
