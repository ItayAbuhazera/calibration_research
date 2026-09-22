"""
GLAD-PI: Geometric Logit Additive Decision-calibrator with Permutation Invariance.

Architecture:
  Per-class inputs: [logit_k, distance_k]  (2 dims per class)
  Global context (permutation-invariant): [max_prob, entropy, top_margin]  (3 dims)
  Shared-weight per-class MLP h_theta([logit_k, distance_k, context]) -> delta_k

  Permutation equivariance comes from applying a shared per-class network
  with permutation-invariant global context (max, entropy, margin).
  Centering removes the softmax-unidentifiable degree of freedom:
  adding the same scalar to all logits does not change the probability vector.

  corrected_logits = (logits / T if use_temperature else logits) + weight * delta_centered
  q = softmax(corrected_logits)

Loss (asymmetric — margin term applied only to base-wrong samples):
  loss = nll + beta * mean_margin_loss_on_base_wrong_samples

  The asymmetric design avoids sharpening correct predictions (which would hurt ECE).

Beta grid selection:
  Train one model per beta on the fit split (with init_seed set, from a
  seeded initialization so a geometry/zero-geometry pair is exactly matched).
  Evaluate net_flips on the select split.
  Select the beta that maximises net_flips subject to NLL tolerance:
    selected_nll <= (1 + nll_tolerance) * reference_nll (from beta=0)
    and select_net_flips >= 0
  If no candidate passes: fallback to beta=0, report selected_fallback=True.
  Warn if selected beta is at the edge of the grid.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class _CorrectionNet(nn.Module):
    """
    Shared-weight per-class MLP producing a centered correction delta_k.

    Input per class k: [logit_k, distance_k, max_prob, entropy, top_margin] = 5 dims.
    Output per class k: delta_k (scalar).
    Correction is centered: delta -= delta.mean(dim=-1, keepdim=True).
    """

    INPUT_DIM = 5  # [logit_k, distance_k, max_prob, entropy, top_margin]

    def __init__(self, hidden_dim: int = 64, use_temperature: bool = False) -> None:
        super().__init__()
        self.use_temperature = use_temperature

        # Shared per-class MLP
        self.per_class_mlp = nn.Sequential(
            nn.Linear(self.INPUT_DIM, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        # Learnable correction scale (initialised small so model starts near identity)
        self.correction_weight = nn.Parameter(torch.tensor(0.1))

        if use_temperature:
            # Temperature net: [max_prob, entropy, top_margin] -> log_T -> T > 0
            self.temp_net = nn.Sequential(
                nn.Linear(3, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

    def forward(
        self,
        logits: torch.Tensor,   # [N, C]
        distances: torch.Tensor,  # [N, C]
    ) -> torch.Tensor:
        """Return corrected logits [N, C]."""
        N, C = logits.shape

        # Global context (permutation-invariant)
        probs = F.softmax(logits, dim=-1)
        max_prob = probs.max(dim=-1, keepdim=True)[0]            # [N, 1]
        entropy = -(probs * torch.log(probs.clamp(min=1e-12))).sum(dim=-1, keepdim=True)  # [N, 1]
        sorted_probs = probs.sort(dim=-1, descending=True)[0]
        top_margin = sorted_probs[:, 0:1] - sorted_probs[:, 1:2]  # [N, 1]
        global_ctx = torch.cat([max_prob, entropy, top_margin], dim=-1)  # [N, 3]

        # Per-class inputs: [N, C, 2]
        per_class_in = torch.stack([logits, distances], dim=-1)

        # Expand global context to [N, C, 3]
        global_ctx_exp = global_ctx.unsqueeze(1).expand(-1, C, -1)

        # Concatenate: [N, C, 5]
        inp = torch.cat([per_class_in, global_ctx_exp], dim=-1)

        # Apply shared MLP: reshape [N*C, 5] -> [N*C, 1] -> [N, C]
        delta = self.per_class_mlp(inp.reshape(N * C, self.INPUT_DIM)).reshape(N, C)

        # Bound the raw delta via tanh so out-of-distribution distance features cannot
        # produce corrections that swamp the base logits; then center to remove the
        # softmax-unidentifiable DOF. Centering after tanh is exact (sums to zero per sample).
        delta_bounded = torch.tanh(delta)
        delta_bounded = delta_bounded - delta_bounded.mean(dim=-1, keepdim=True)

        # Optional temperature scaling
        if self.use_temperature:
            log_T = self.temp_net(global_ctx).squeeze(-1)  # [N]
            T = F.softplus(log_T) + 0.05
            scaled_logits = logits / T.unsqueeze(-1)
        else:
            scaled_logits = logits

        corrected = scaled_logits + self.correction_weight * delta_bounded
        return corrected


def _asymmetric_margin_loss(
    corrected_logits: torch.Tensor,
    labels: torch.Tensor,
    base_pred: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """
    Margin term applied only to samples where base model is wrong.
    Encourages: corrected_logit[y] - corrected_logit[base_pred] >= margin.
    """
    base_wrong = (base_pred != labels)
    if not base_wrong.any():
        return corrected_logits.new_zeros(1).squeeze()

    cw_logits = corrected_logits[base_wrong]
    cw_labels = labels[base_wrong]
    cw_base = base_pred[base_wrong]
    idx = torch.arange(cw_logits.shape[0], device=cw_logits.device)

    true_logit = cw_logits[idx, cw_labels]
    pred_logit = cw_logits[idx, cw_base]
    return F.relu(margin - (true_logit - pred_logit)).mean()


def _net_flips(
    probs: np.ndarray,
    base_probs: np.ndarray,
    labels: np.ndarray,
) -> int:
    pred = np.argmax(probs, axis=1)
    base_pred = np.argmax(base_probs, axis=1)
    changed = pred != base_pred
    correct_to = pred == labels
    correct_from = base_pred == labels
    c2c = int(np.sum(changed & (~correct_from) & correct_to))
    c2w = int(np.sum(changed & correct_from & (~correct_to)))
    return c2c - c2w


def _train_one_beta(
    net: _CorrectionNet,
    logits_fit: torch.Tensor,
    distances_fit: torch.Tensor,
    labels_fit: torch.Tensor,
    base_pred_fit: torch.Tensor,
    beta: float,
    lr: float,
    weight_decay: float,
    epochs: int,
    patience: int,
    margin: float,
    device: torch.device,
) -> _CorrectionNet:
    """Train a fresh _CorrectionNet with the given beta. Returns the best-state net."""
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)

    best_loss = float("inf")
    best_state = None
    no_improve = 0

    net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        corrected = net(logits_fit, distances_fit)
        nll = F.cross_entropy(corrected, labels_fit)
        if beta > 0:
            margin_loss = _asymmetric_margin_loss(corrected, labels_fit, base_pred_fit, margin)
            loss = nll + beta * margin_loss
        else:
            loss = nll
        loss.backward()
        optimizer.step()

        val = float(loss.item())
        if val < best_loss - 1e-6:
            best_loss = val
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


class GLADPICalibrator:
    """
    GLAD-PI calibrator.

    fit(logits_fit, labels_fit, distance_matrix_fit,
        logits_select, labels_select, distance_matrix_select) -> dict

    calibrate(logits_test, distance_matrix_test) -> np.ndarray [N, C]
    get_params() -> dict
    """

    def __init__(
        self,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 100,
        patience: int = 10,
        beta_grid: Optional[List[float]] = None,
        nll_tolerance: float = 0.05,
        use_temperature: bool = False,
        margin: float = 0.5,
        device: Optional[str] = None,
        zero_geometry: bool = False,
        init_seed: Optional[int] = None,
    ) -> None:
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.patience = patience
        self.beta_grid = beta_grid if beta_grid is not None else [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
        self.nll_tolerance = nll_tolerance
        self.use_temperature = use_temperature
        self.margin = margin
        # Capacity-matched zero-geometry control (§8e of the benchmark plan):
        # masks distance_k to 0 before the MLP so architecture/parameter count
        # stay identical to the geometry-aware model by construction.
        self.zero_geometry = zero_geometry
        # Matched-ablation determinism. With init_seed set, every beta's network
        # is constructed immediately after torch.manual_seed(init_seed + beta
        # index), so the geometry arm and the zero-geometry arm start from
        # BIT-IDENTICAL parameters and the only difference between them is the
        # geometry input. With init_seed=None (the default) initialization comes
        # from the ambient RNG, which is what every already-fitted state in
        # results/studyAB was produced with -- leaving the default unchanged
        # keeps those artifacts reproducible.
        self.init_seed = init_seed
        self._device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._net: Optional[_CorrectionNet] = None
        self._selected_beta: Optional[float] = None
        self._selected_fallback: bool = False
        self._selection_results: Optional[List[Dict[str, Any]]] = None
        self._num_classes: Optional[int] = None
        # Distance standardization stats: fit from fit-split, applied globally.
        # Stored so calibrate() uses the same transform as training.
        self._dist_mean: Optional[float] = None
        self._dist_std: Optional[float] = None

    def _normalize_distances(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Standardize distances using fit-split statistics (must call fit first)."""
        if self._dist_mean is None or self._dist_std is None:
            raise RuntimeError("Distance normalization stats not set; call fit() first.")
        return (distance_matrix - self._dist_mean) / self._dist_std

    def _to_tensors(
        self,
        logits: np.ndarray,
        distance_matrix: np.ndarray,
        labels: Optional[np.ndarray] = None,
    ) -> Tuple:
        logits_t = torch.tensor(logits, dtype=torch.float32, device=self._device)
        dist_t = torch.tensor(distance_matrix, dtype=torch.float32, device=self._device)
        if self.zero_geometry:
            dist_t = torch.zeros_like(dist_t)
        if labels is not None:
            labels_t = torch.tensor(labels, dtype=torch.long, device=self._device)
            return logits_t, dist_t, labels_t
        return logits_t, dist_t

    def _eval_on_select(
        self,
        net: _CorrectionNet,
        logits_sel: torch.Tensor,
        dist_sel: torch.Tensor,
        labels_sel_np: np.ndarray,
        base_probs_sel: np.ndarray,
    ) -> Dict[str, Any]:
        with torch.no_grad():
            corrected = net(logits_sel, dist_sel)
            nll = float(F.cross_entropy(corrected, torch.tensor(labels_sel_np, dtype=torch.long, device=self._device)).item())
            probs_np = F.softmax(corrected, dim=-1).cpu().numpy().astype(np.float64)

        flips = _net_flips(probs_np, base_probs_sel, labels_sel_np)
        acc = float(np.mean(np.argmax(probs_np, axis=1) == labels_sel_np))
        correction_l2 = float(np.mean(np.linalg.norm(probs_np - base_probs_sel, axis=1)))
        return {
            "select_nll": nll,
            "select_net_flips": flips,
            "select_accuracy": acc,
            "mean_correction_l2": correction_l2,
        }

    def fit(
        self,
        logits_fit: np.ndarray,
        labels_fit: np.ndarray,
        distance_matrix_fit: np.ndarray,
        logits_select: np.ndarray,
        labels_select: np.ndarray,
        distance_matrix_select: np.ndarray,
    ) -> Dict[str, Any]:
        self._num_classes = logits_fit.shape[1]

        # Fit distance standardization on fit-split only; apply to all splits.
        # This makes the MLP input scale-invariant to absolute distance magnitude.
        self._dist_mean = float(np.mean(distance_matrix_fit))
        self._dist_std = max(float(np.std(distance_matrix_fit)), 1e-8)
        dist_fit_norm = self._normalize_distances(distance_matrix_fit)
        dist_sel_norm = self._normalize_distances(distance_matrix_select)

        logits_fit_t, dist_fit_t, labels_fit_t = self._to_tensors(logits_fit, dist_fit_norm, labels_fit)
        logits_sel_t, dist_sel_t = self._to_tensors(logits_select, dist_sel_norm)

        base_pred_fit = torch.tensor(
            np.argmax(
                F.softmax(torch.tensor(logits_fit, dtype=torch.float32), dim=-1).numpy(),
                axis=1,
            ),
            dtype=torch.long,
            device=self._device,
        )

        base_probs_sel = F.softmax(
            torch.tensor(logits_select, dtype=torch.float32), dim=-1
        ).numpy().astype(np.float64)

        # Train one model per beta and record select-set metrics
        results: List[Dict[str, Any]] = []
        reference_nll: Optional[float] = None

        for beta_index, beta in enumerate(self.beta_grid):
            if self.init_seed is not None:
                torch.manual_seed(int(self.init_seed) + beta_index)
            net = _CorrectionNet(hidden_dim=self.hidden_dim, use_temperature=self.use_temperature)
            net = _train_one_beta(
                net,
                logits_fit_t, dist_fit_t, labels_fit_t, base_pred_fit,
                beta=beta,
                lr=self.lr,
                weight_decay=self.weight_decay,
                epochs=self.epochs,
                patience=self.patience,
                margin=self.margin,
                device=self._device,
            )
            sel = self._eval_on_select(net, logits_sel_t, dist_sel_t, labels_select, base_probs_sel)
            entry = {"beta": beta, "net": net, **sel}
            results.append(entry)

            if beta == 0.0:
                reference_nll = sel["select_nll"]
            logger.debug(
                "GLAD-PI beta=%.4f: select_nll=%.4f, select_net_flips=%d",
                beta, sel["select_nll"], sel["select_net_flips"],
            )

        # Guard: at beta=0, select accuracy must be near the base model's select accuracy.
        # If it is not, the select-split inputs are mis-scaled or misaligned; proceeding
        # would silently produce a meaningless beta selection.
        _beta0_result = next((r for r in results if r["beta"] == 0.0), None)
        if _beta0_result is not None:
            _base_sel_acc = float(np.mean(np.argmax(base_probs_sel, axis=1) == labels_select))
            if _beta0_result["select_accuracy"] < _base_sel_acc - 0.05:
                raise RuntimeError(
                    f"GLAD-PI select-split inputs appear mis-scaled or misaligned: "
                    f"beta=0 select accuracy {_beta0_result['select_accuracy']:.4f} << "
                    f"base select accuracy {_base_sel_acc:.4f}. "
                    f"Ensure distance_matrix_select rows align with logits_select and labels_select."
                )

        if reference_nll is None:
            # Grid does not include 0; use minimum NLL across all betas
            reference_nll = min(r["select_nll"] for r in results)

        nll_budget = (1.0 + self.nll_tolerance) * reference_nll

        # Filter: NLL within tolerance AND net_flips >= 0
        candidates = [
            r for r in results
            if r["select_nll"] <= nll_budget and r["select_net_flips"] >= 0
        ]

        if candidates:
            # Primary: max net_flips; tie-break: accuracy, then lower NLL, then smaller L2
            best = max(
                candidates,
                key=lambda r: (
                    r["select_net_flips"],
                    r["select_accuracy"],
                    -r["select_nll"],
                    -r["mean_correction_l2"],
                ),
            )
            self._selected_fallback = False
        else:
            logger.warning(
                "GLAD-PI: no candidate passed NLL tolerance (%.4f) with non-negative net flips. "
                "Falling back to beta=0.",
                nll_budget,
            )
            # Fallback: beta=0 model
            best = next((r for r in results if r["beta"] == 0.0), results[0])
            self._selected_fallback = True

        self._net = best["net"]
        self._selected_beta = float(best["beta"])

        # Warn if selected beta is at the edge of the non-zero grid values
        non_zero = [b for b in self.beta_grid if b > 0]
        if non_zero and not self._selected_fallback:
            beta_val = self._selected_beta
            if beta_val == min(self.beta_grid) or beta_val == max(self.beta_grid):
                warnings.warn(
                    f"GLAD-PI: selected beta={beta_val} is at the edge of the grid {self.beta_grid}. "
                    "Consider extending the grid.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Strip the net objects from the stored results (keep only scalars)
        self._selection_results = [
            {k: v for k, v in r.items() if k != "net"} for r in results
        ]

        return {
            "selected_beta": self._selected_beta,
            "selected_fallback": self._selected_fallback,
            "reference_nll": reference_nll,
            "nll_budget": nll_budget,
            "best_select_net_flips": int(best["select_net_flips"]),
            "best_select_nll": float(best["select_nll"]),
            "selection_results": self._selection_results,
        }

    def calibrate(
        self,
        logits_test: np.ndarray,
        distance_matrix_test: np.ndarray,
    ) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Call fit() before calibrate()")
        dist_test_norm = self._normalize_distances(distance_matrix_test)
        logits_t, dist_t = self._to_tensors(logits_test, dist_test_norm)
        self._net.eval()
        with torch.no_grad():
            corrected = self._net(logits_t, dist_t)
            probs = F.softmax(corrected, dim=-1)
        return probs.cpu().numpy().astype(np.float64)

    def get_params(self) -> Dict[str, Any]:
        return {
            "fitted": self._net is not None,
            "selected_beta": self._selected_beta,
            "selected_fallback": self._selected_fallback,
            "hidden_dim": self.hidden_dim,
            "use_temperature": self.use_temperature,
            "zero_geometry": self.zero_geometry,
            "init_seed": self.init_seed,
            "nll_tolerance": self.nll_tolerance,
            "beta_grid": self.beta_grid,
            "selection_results": self._selection_results,
            "num_classes": self._num_classes,
            "dist_mean": self._dist_mean,
            "dist_std": self._dist_std,
        }
