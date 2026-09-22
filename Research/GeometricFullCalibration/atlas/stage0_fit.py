"""Penalized multinomial logistic regression for Stage 0c (docs/stage0_execution_spec.md Section 3).

    q(x) = softmax(X_tilde @ W + b),  X_tilde = per-coordinate standardized features
    objective = mean_i[-log q(y_i|x_i)] + lambda * ||W||_F^2   (b unpenalized, NO 1/2 factor)

Written fresh rather than reusing Calibrators/layer_readouts.py::fit_probe, which has a 0.5*lam
convention that would silently change the declared objective. All computation in float64.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

torch.set_default_dtype(torch.float64)

LAMBDA_GRID = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5)
TOL_GRAD = 1e-8
TOL_CHANGE = 1e-11
MAX_ITER = 2000
RETRY_MAX_ITER = 6000


def standardize_fit(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    zero_var = std == 0
    std_safe = np.where(zero_var, 1.0, std)
    return mean, std_safe


def standardize_apply(X: np.ndarray, mean: np.ndarray, std_safe: np.ndarray) -> np.ndarray:
    return (X - mean) / std_safe


def _nll_from_logits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logp = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    return -logp[torch.arange(logits.shape[0]), y].mean()


def fit_penalized_multinomial(
    X: np.ndarray, y: np.ndarray, lam: float, n_classes: int,
    max_iter: int = MAX_ITER, retry_max_iter: int = RETRY_MAX_ITER,
    tol_grad: float = TOL_GRAD, tol_change: float = TOL_CHANGE,
) -> Dict:
    n, d = X.shape
    Xt = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float64))
    yt = torch.from_numpy(np.ascontiguousarray(y, dtype=np.int64))
    W = torch.zeros((d, n_classes), requires_grad=True)
    b = torch.zeros((n_classes,), requires_grad=True)

    def make_opt(mi):
        return torch.optim.LBFGS([W, b], lr=1.0, max_iter=mi, history_size=20,
                                  line_search_fn="strong_wolfe",
                                  tolerance_grad=tol_grad, tolerance_change=tol_change)

    def closure():
        opt.zero_grad()
        logits = Xt @ W + b
        loss = _nll_from_logits(logits, yt) + lam * (W * W).sum()
        loss.backward()
        return loss

    # torch's LBFGS.step breaks the moment ANY of {grad tolerance, step-size tolerance,
    # loss-change tolerance, max_iter/max_eval} triggers. `state['n_iter'] < max_iter` (or, for
    # the sanity floor below, a small grad norm) means it stopped for a real numerical reason,
    # not merely because the iteration cap was reached -- that cap-hit case is exactly the
    # "150 iterations is not a scientific stopping criterion" failure mode the spec calls out,
    # and is what triggers the numerical-recovery retry with a larger cap.
    grad_sanity_ceiling = 1e-4

    def fresh_grad_inf_norm():
        closure()
        return max(W.grad.abs().max().item(), b.grad.abs().max().item())

    def ran_to_cap(opt_, mi):
        st = opt_.state[opt_.param_groups[0]['params'][0]]
        return st.get('n_iter', mi) >= mi

    opt = make_opt(max_iter)
    final_loss = opt.step(closure)
    grad_inf_norm = fresh_grad_inf_norm()
    hit_cap = ran_to_cap(opt, max_iter)
    converged = (not hit_cap) and grad_inf_norm <= grad_sanity_ceiling
    n_iter = max_iter
    retried = False
    if not converged:
        opt = make_opt(retry_max_iter)
        final_loss = opt.step(closure)
        grad_inf_norm = fresh_grad_inf_norm()
        hit_cap = ran_to_cap(opt, retry_max_iter)
        converged = (not hit_cap) and grad_inf_norm <= grad_sanity_ceiling
        n_iter = retry_max_iter
        retried = True

    return {
        "W": W.detach().numpy(), "b": b.detach().numpy(),
        "loss": float(final_loss.item()), "grad_inf_norm": float(grad_inf_norm),
        "converged": bool(converged), "hit_cap": bool(hit_cap), "n_iter_cap": n_iter, "retried": retried,
        "lambda": lam,
    }


def predict_probs(X: np.ndarray, mean: np.ndarray, std_safe: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    Xs = standardize_apply(X, mean, std_safe)
    logits = Xs @ W + b
    logits = logits - logits.max(axis=1, keepdims=True)
    ex = np.exp(logits)
    return ex / ex.sum(axis=1, keepdims=True)


def nll_np(probs: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    idx = np.arange(len(y))
    return float(-np.mean(np.log(np.clip(probs[idx, y], eps, 1.0))))


def select_from_table(table: List[Dict], tie_eps: float = 1e-9) -> Dict:
    """Min inner-val NLL; ties within tie_eps broken toward larger lambda, then grid order."""
    best_nll = min(t["val_nll"] for t in table)
    tied = [t for t in table if abs(t["val_nll"] - best_nll) < tie_eps]
    tied_sorted = sorted(tied, key=lambda t: -t["lambda"])  # larger lambda (stronger reg) wins ties
    return tied_sorted[0]


def select_lambda(
    X_fit: np.ndarray, y_fit: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    n_classes: int, lambda_grid: Tuple[float, ...] = LAMBDA_GRID,
) -> Dict:
    mean, std = standardize_fit(X_fit)
    Xf = standardize_apply(X_fit, mean, std)
    table = []
    for lam in lambda_grid:
        fit = fit_penalized_multinomial(Xf, y_fit, lam, n_classes)
        probs_val = predict_probs(X_val, mean, std, fit["W"], fit["b"])
        val_nll = nll_np(probs_val, y_val)
        table.append({"lambda": lam, "val_nll": val_nll, "converged": fit["converged"], "grad_inf_norm": fit["grad_inf_norm"]})

    best = select_from_table(table)
    return {"table": table, "selected_lambda": best["lambda"], "selected_val_nll": best["val_nll"],
            "scaler_mean_used": mean, "scaler_std_used": std}


def fit_arm(
    X_full_train: np.ndarray, y_full_train: np.ndarray,
    X_inner_fit: np.ndarray, y_inner_fit: np.ndarray,
    X_inner_val: np.ndarray, y_inner_val: np.ndarray,
    n_classes: int, lambda_grid: Tuple[float, ...] = LAMBDA_GRID,
) -> Dict:
    sel = select_lambda(X_inner_fit, y_inner_fit, X_inner_val, y_inner_val, n_classes, lambda_grid)
    lam = sel["selected_lambda"]
    mean, std = standardize_fit(X_full_train)
    Xf = standardize_apply(X_full_train, mean, std)
    final = fit_penalized_multinomial(Xf, y_full_train, lam, n_classes)
    return {
        "selection": sel, "selected_lambda": lam,
        "scaler_mean": mean, "scaler_std": std,
        "W": final["W"], "b": final["b"],
        "converged": final["converged"], "grad_inf_norm": final["grad_inf_norm"], "retried": final["retried"],
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n, d, k = 400, 10, 5
    Wtrue = rng.normal(size=(d, k)) * 0.5
    X = rng.normal(size=(n, d))
    logits = X @ Wtrue
    probs = np.exp(logits - logits.max(1, keepdims=True))
    probs /= probs.sum(1, keepdims=True)
    y = np.array([rng.choice(k, p=probs[i]) for i in range(n)])
    Xtr, ytr = X[:250], y[:250]
    Xf, yf = X[:180], y[:180]
    Xv, yv = X[180:250], y[180:250]
    Xte, yte = X[250:], y[250:]
    out = fit_arm(Xtr, ytr, Xf, yf, Xv, yv, k)
    print("selected lambda:", out["selected_lambda"], "converged:", out["converged"], "grad_inf_norm:", out["grad_inf_norm"])
    probs_te = predict_probs(Xte, out["scaler_mean"], out["scaler_std"], out["W"], out["b"])
    print("test NLL:", nll_np(probs_te, yte), "test acc:", (probs_te.argmax(1) == yte).mean())
