"""Decoder Panel v1 (docs/decoder_panel_v1_spec.md): six fixed decoder families with bounded, predeclared model selection.

Families (exactly six; never extended after outcomes):
    linear   L2-penalized (anchored) multinomial logistic / ridge on standardized inputs          exhaustive lambda grid
    poly2    [x~, std(TensorSketch_2(x~))] -> L2 linear head (degree fixed at 2)                     grid: sketch dim x lambda
    lgbm     LightGBM on raw coordinates (no PCA; anchored via init_score)                          Optuna (TPE)
    mlp      Linear(d,w)-GELU-Dropout-Linear(w,K) on standardized inputs (exactly one hidden layer)  Optuna (TPE)
    knn      standardize -> SparseRandomProjection (JL, eps=0.5, only if d > target) -> kNN         grid: k x weighting (x beta)
    rff      standardize -> RBF random Fourier features (1024) -> std -> L2 linear head             grid: gamma multiplier x lambda

Tasks: 'classification' (K >= 2 classes; binary is K = 2) with an optional anchor offset (logits = offset + g(x)), and 'regression'
(scalar, squared loss). Action selection is classification over Delta in {-1, 0, +1} with a caller-supplied selection objective.

Leakage discipline: every transform (scaler, projection, sketch, RFF gamma) is fitted inside `_fit_family` on the rows it is given and
is stored on the returned model; model selection sees only the rows of the supplied inner splits; `fit_decoder` never receives
evaluation rows. Objective: loss to MINIMIZE (G1-DP: inner-val NLL; N1a-DP: negative realized policy utility).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PANEL_VERSION = "decoder_panel_v1"
SPACE_VERSION = "dp1-space-1"
FAMILIES = ("linear", "poly2", "lgbm", "mlp", "knn", "rff")
OPTUNA_FAMILIES = ("lgbm", "mlp")
MASTER_SEED = 20260928

# ---- frozen grids / spaces (docs/decoder_panel_v1_spec.md sec. 3) -----------------------------------------------------------------
LAMBDA_GRID = (1e1, 1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7)   # strong -> weak (warm-start order)
POLY_DIMS = (512, 1024, 2048)
KNN_K = (5, 15, 30, 50, 100)
KNN_WEIGHTS = ("uniform", "distance")
KNN_BETA = (0.25, 0.5, 1.0, 2.0, 4.0)       # anchored classification only: logits = offset + beta * log p_knn
KNN_ALPHA = 1.0                             # Laplace smoothing of neighbour class frequencies
JL_EPS = 0.5
RFF_COMPONENTS = 1024
RFF_MULTS = (0.25, 0.5, 1.0, 2.0, 4.0)
N_TRIALS = 50                                # Optuna budget per study (fixed by the Stage-B convergence audit, docs/decoder_panel_v1_resource_plan.md)
LGBM_MAX_ROUNDS = 1000
LGBM_EARLY_STOP = 50
LGBM_MAX_DELTA_STEP = 2.0                    # caps |leaf output| (x learning rate); guards Newton steps under saturated anchor init_scores
MLP_MAX_EPOCHS = 200
MLP_PATIENCE = 20
MLP_BATCH = 256
AUDIT_ROWS = 500                             # rows for the label-free JL distortion audit / RFF median heuristic
SEED_OFFSETS = {"poly2": 11, "knn": 13, "rff": 17}   # fixed projection seeds: MASTER_SEED + offset (identical across arms/folds)


def derive_seed(*parts) -> int:
    """Deterministic 31-bit seed from identifiers (experiment, base, fold, cell/family, decoder, arm, ...)."""
    s = "|".join(str(p) for p in (MASTER_SEED,) + tuple(parts))
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def n_threads() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("OMP_NUM_THREADS", "1")))


# ---- context --------------------------------------------------------------------------------------------------------------------------
@dataclass
class HPOContext:
    """Model-selection context. `mode` distinguishes the two frozen HPO regimes:
    'target_pooled'  (G1-DP: one image-grouped inner split of the target-supervised training rows; accessibility diagnostic only)
    'shift_transfer' (N1a-DP: inner leave-one-TRAINING-environment-out splits; the held-out outer environment is never present).
    `forbidden_env` (shift_transfer) names the outer held-out environment; `env_train` gives each training row's environment and is
    asserted not to contain it."""
    experiment: str
    ids: Dict[str, object]
    mode: str
    inner_splits: List[Tuple[np.ndarray, np.ndarray]]
    objective: Callable[[np.ndarray, np.ndarray, np.ndarray], float]   # (pred, y_val, val_rows) -> loss to minimize
    forbidden_env: Optional[str] = None
    env_train: Optional[np.ndarray] = None

    def validate(self, n_train: int):
        assert self.mode in ("target_pooled", "shift_transfer"), self.mode
        assert len(self.inner_splits) >= 1
        for fi, vi in self.inner_splits:
            assert len(np.intersect1d(fi, vi)) == 0, "inner fit/val rows overlap"
            assert fi.max(initial=-1) < n_train and vi.max(initial=-1) < n_train
        if self.mode == "target_pooled":
            assert len(self.inner_splits) == 1 and self.forbidden_env is None, "target_pooled mode: one inner split, no environment holdout"
        else:
            assert self.forbidden_env is not None and self.env_train is not None and len(self.inner_splits) >= 2, \
                "shift_transfer mode requires the held-out environment name, per-row environments and >= 2 environment splits"
            assert self.forbidden_env not in set(np.asarray(self.env_train).tolist()), "held-out environment present in training rows"
            for fi, vi in self.inner_splits:
                ev = set(np.asarray(self.env_train)[vi].tolist()); ef = set(np.asarray(self.env_train)[fi].tolist())
                assert len(ev) == 1 and not (ev & ef), "shift_transfer inner split must hold out exactly one training environment"

    def study_seed(self, family: str) -> int:
        return derive_seed(self.experiment, *[f"{k}={self.ids[k]}" for k in sorted(self.ids)], family)


def nll_objective(pred, y, _rows=None):
    """Mean multiclass NLL from logits (classification) — the G1-DP selection objective."""
    z = pred - pred.max(1, keepdims=True); lse = np.log(np.exp(z).sum(1))
    return float(np.mean(lse - z[np.arange(len(y)), y]))


# ---- standardization ------------------------------------------------------------------------------------------------------------------
class Std:
    def fit(self, X):
        X = np.asarray(X, dtype=np.float64); self.m = X.mean(0); s = X.std(0); self.s = np.where(s == 0, 1.0, s); return self

    def __call__(self, X):
        return (np.asarray(X, dtype=np.float64) - self.m) / self.s


# ---- linear head (shared by linear / poly2 / rff) -----------------------------------------------------------------------------------
def _torch():
    import torch
    torch.set_num_threads(n_threads())
    return torch


def linear_path(Phi, y, offset, task, lambdas, K=None, max_iter=2000, retry_max_iter=6000, tol_grad=1e-8, tol_change=1e-11):
    """Fit the L2 head for each lambda (strong -> weak, warm-started; the problem is strictly convex so warm starts change numerics only).
    classification: mean CE(offset + Phi W + b) + lambda ||W||_F^2 (bias unpenalized; repository convention, no 1/2).
    regression:     mean (Phi w + b - y)^2 + lambda ||w||^2 (closed form)."""
    out = []
    if task == "regression":
        Xc = Phi - Phi.mean(0); yc = y - y.mean(); n = len(y); G = Xc.T @ Xc / n; r = Xc.T @ yc / n
        for lam in lambdas:
            w = np.linalg.solve(G + lam * np.eye(G.shape[0]), r)
            out.append({"lambda": lam, "W": w, "b": float(y.mean() - Phi.mean(0) @ w), "converged": True, "retried": False, "grad_inf": 0.0})
        return out
    torch = _torch(); import torch.nn.functional as F
    X = torch.from_numpy(np.ascontiguousarray(Phi, dtype=np.float64)); yt = torch.from_numpy(np.asarray(y, dtype=np.int64))
    O = torch.from_numpy(np.ascontiguousarray(offset, dtype=np.float64)) if offset is not None else None
    d = Phi.shape[1]; K = K if offset is None else offset.shape[1]
    W = torch.zeros((d, K), dtype=torch.float64, requires_grad=True); b = torch.zeros((K,), dtype=torch.float64, requires_grad=True)
    for lam in lambdas:
        def loss_fn():
            lg = X @ W + b
            if O is not None:
                lg = lg + O
            return F.cross_entropy(lg, yt) + lam * (W * W).sum()

        def run(mi):
            opt = torch.optim.LBFGS([W, b], lr=1.0, max_iter=mi, history_size=20, line_search_fn="strong_wolfe",
                                    tolerance_grad=tol_grad, tolerance_change=tol_change)

            def closure():
                opt.zero_grad(); l = loss_fn(); l.backward(); return l
            opt.step(closure)
            opt.zero_grad(); l = loss_fn(); l.backward()
            g = max(W.grad.abs().max().item(), b.grad.abs().max().item())
            cap = opt.state[opt.param_groups[0]["params"][0]].get("n_iter", mi) >= mi
            return g, cap
        g, cap = run(max_iter); conv, retried = (not cap) and g <= 1e-4, False
        if not conv:
            g, cap = run(retry_max_iter); conv, retried = (not cap) and g <= 1e-4, True
        out.append({"lambda": lam, "W": W.detach().numpy().copy(), "b": b.detach().numpy().copy(), "converged": bool(conv), "retried": retried, "grad_inf": g})
    return out


# ---- fitted model -----------------------------------------------------------------------------------------------------------------------
@dataclass
class Fitted:
    family: str
    task: str
    params: Dict
    transform: Callable[[np.ndarray], np.ndarray]
    head: Dict = field(default_factory=dict)
    meta: Dict = field(default_factory=dict)
    null: bool = False            # anchor-only candidate (g = 0)

    def predict(self, X, offset=None):
        """classification -> logits (offset added when anchored); regression -> scalar prediction."""
        if self.null:
            return np.asarray(offset, dtype=np.float64).copy()
        g = PREDICT[self.family](self, X, offset)
        return g


def _lin_predict(m, X, offset):
    Phi = m.transform(X); g = Phi @ m.head["W"] + m.head["b"]
    if m.task == "regression":
        return g
    return g if offset is None else np.asarray(offset, dtype=np.float64) + g


# ---- family: transforms ------------------------------------------------------------------------------------------------------------
def _poly_transform(X_fit, dim):
    from sklearn.kernel_approximation import PolynomialCountSketch
    s1 = Std().fit(X_fit); ts = PolynomialCountSketch(degree=2, gamma=1.0, coef0=0.0, n_components=dim,
                                                      random_state=MASTER_SEED + SEED_OFFSETS["poly2"]).fit(s1(X_fit))
    s2 = Std().fit(ts.transform(s1(X_fit)))
    return lambda X: np.concatenate([s1(X), s2(ts.transform(s1(X)))], 1)


def rff_gamma_median(Xs) -> float:
    """Label-free median heuristic on a fixed sample of the (standardized) fit rows: gamma = 1 / median ||x_i - x_j||^2."""
    rng = np.random.default_rng(MASTER_SEED + SEED_OFFSETS["rff"]); idx = rng.choice(len(Xs), min(AUDIT_ROWS, len(Xs)), replace=False)
    S = Xs[idx]; sq = (S * S).sum(1); D = sq[:, None] + sq[None, :] - 2 * S @ S.T
    iu = np.triu_indices(len(S), 1)
    return float(1.0 / np.median(np.maximum(D[iu], 1e-12)))


def _rff_transform(X_fit, mult):
    from sklearn.kernel_approximation import RBFSampler
    s1 = Std().fit(X_fit); Xs = s1(X_fit); gmed = rff_gamma_median(Xs)
    rbf = RBFSampler(gamma=mult * gmed, n_components=RFF_COMPONENTS, random_state=MASTER_SEED + SEED_OFFSETS["rff"]).fit(Xs)
    s2 = Std().fit(rbf.transform(Xs))
    return (lambda X: s2(rbf.transform(s1(X)))), {"gamma_median": gmed, "gamma": mult * gmed}


def jl_target(n: int, d: int) -> int:
    from sklearn.random_projection import johnson_lindenstrauss_min_dim
    t = int(johnson_lindenstrauss_min_dim(n_samples=n, eps=JL_EPS))
    return t if t < d else d


def jl_audit(Xs, P) -> Dict:
    """Label-free distortion audit ||R x_i - R x_j|| / ||x_i - x_j|| on a fixed sample of fit rows."""
    rng = np.random.default_rng(MASTER_SEED + SEED_OFFSETS["knn"] + 1); idx = rng.choice(len(Xs), min(AUDIT_ROWS, len(Xs)), replace=False)
    A, B = Xs[idx], P(Xs[idx])
    def pd(M):
        sq = (M * M).sum(1); return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * M @ M.T, 0))
    iu = np.triu_indices(len(idx), 1); r = pd(B)[iu] / np.maximum(pd(A)[iu], 1e-12)
    return {"median": float(np.median(r)), "p05": float(np.quantile(r, .05)), "p95": float(np.quantile(r, .95)),
            "min": float(r.min()), "max": float(r.max()), "n_pairs": int(len(r))}


def _knn_transform(X_fit):
    from sklearn.random_projection import SparseRandomProjection
    s1 = Std().fit(X_fit); Xs = s1(X_fit); t = jl_target(len(Xs), Xs.shape[1]); meta = {"jl_target": t, "input_dim": int(Xs.shape[1]), "projected": t < Xs.shape[1]}
    if t < Xs.shape[1]:
        rp = SparseRandomProjection(n_components=t, random_state=MASTER_SEED + SEED_OFFSETS["knn"]).fit(Xs)
        meta["jl_audit"] = jl_audit(Xs, rp.transform)
        return (lambda X: rp.transform(s1(X))), meta
    return s1, meta


# ---- family: kNN -------------------------------------------------------------------------------------------------------------------
def _knn_query(train_T, query_T, kmax, block=2048):
    """Exact brute-force Euclidean kNN in float64, blockwise NumPy (deterministic; ties -> lower training index).
    (sklearn's threaded brute-force path returned out-of-range indices when OpenMP runtimes were mixed in one process; not used.)"""
    A = np.asarray(train_T, np.float64); Q = np.asarray(query_T, np.float64); k = min(kmax, len(A))
    a2 = (A * A).sum(1); D_out = np.empty((len(Q), k)); I_out = np.empty((len(Q), k), np.int64)
    for s in range(0, len(Q), block):
        q = Q[s:s + block]; D = (q * q).sum(1)[:, None] + a2[None, :] - 2.0 * q @ A.T; np.maximum(D, 0, out=D)
        part = np.argpartition(D, k - 1, axis=1)[:, :k]
        dp_ = np.take_along_axis(D, part, 1); o = np.lexsort((part, dp_), axis=1)
        I_out[s:s + block] = np.take_along_axis(part, o, 1); D_out[s:s + block] = np.sqrt(np.take_along_axis(dp_, o, 1))
    assert I_out.min() >= 0 and I_out.max() < len(A)
    return D_out, I_out


def knn_scores(d, i, y_train, task, K, k, weighting):
    d, i = d[:, :k], i[:, :k]
    w = np.ones_like(d) if weighting == "uniform" else 1.0 / np.maximum(d, 1e-12)
    w = w / w.sum(1, keepdims=True)
    if task == "regression":
        return (w * y_train[i]).sum(1)
    cnt = np.zeros((len(d), K))
    np.add.at(cnt, (np.repeat(np.arange(len(d)), k), y_train[i].ravel()), (w * k).ravel())
    return np.log((cnt + KNN_ALPHA) / (k + K * KNN_ALPHA))


def _knn_predict(m, X, offset):
    d, i = _knn_query(m.head["train_T"], m.transform(X), m.params["k"])
    s = knn_scores(d, i, m.head["y"], m.task, m.head["K"], m.params["k"], m.params["weighting"])
    if m.task == "regression":
        return s
    return s * m.params.get("beta", 1.0) + (0 if offset is None else np.asarray(offset, dtype=np.float64))


# ---- family: LightGBM ----------------------------------------------------------------------------------------------------------------
def _lgbm_params(p, task, K, seed):
    q = {"objective": "multiclass" if (task == "classification" and K > 2) else ("binary" if task == "classification" else "regression"),
         "num_leaves": p["num_leaves"], "max_depth": p["max_depth"], "min_data_in_leaf": p["min_data_in_leaf"], "learning_rate": p["learning_rate"],
         "feature_fraction": p["feature_fraction"], "lambda_l1": p["lambda_l1"], "lambda_l2": p["lambda_l2"], "num_threads": n_threads(),
         "max_delta_step": LGBM_MAX_DELTA_STEP, "seed": seed, "deterministic": True, "force_col_wise": True, "verbose": -1}
    if q["objective"] == "multiclass":
        q["num_class"] = K
    return q


def _lgbm_init(offset, task, K):
    if offset is None:
        return None
    if task == "classification" and K == 2:
        return offset[:, 1] - offset[:, 0]
    return offset


def _lgbm_train(p, X, y, offset, task, K, seed, rounds, Xv=None, yv=None, offv=None):
    import lightgbm as lgb
    ds = lgb.Dataset(np.asarray(X, dtype=np.float32), label=y, init_score=_lgbm_init(offset, task, K), free_raw_data=True)
    kw = {}
    if Xv is not None:
        dv = lgb.Dataset(np.asarray(Xv, dtype=np.float32), label=yv, init_score=_lgbm_init(offv, task, K), reference=ds)
        kw = {"valid_sets": [dv], "callbacks": [lgb.early_stopping(LGBM_EARLY_STOP, verbose=False)]}
    return lgb.train(_lgbm_params(p, task, K, seed), ds, num_boost_round=rounds, **kw)


def _lgbm_raw(booster, X, task, K, iters=None):
    r = booster.predict(np.asarray(X, dtype=np.float32), raw_score=True, num_iteration=iters)
    if task == "classification" and K == 2:
        r = np.stack([np.zeros_like(r), r], 1)
    return r


def _lgbm_predict(m, X, offset):
    r = _lgbm_raw(m.head["booster"], X, m.task, m.head["K"])
    if m.task == "regression":
        return r
    return r + (0 if offset is None else np.asarray(offset, dtype=np.float64))


def lgbm_space(trial):
    return {"num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True), "max_depth": trial.suggest_categorical("max_depth", [-1, 4, 6, 8]),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 10, 200, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.1, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True), "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True)}


# ---- family: MLP --------------------------------------------------------------------------------------------------------------------
def mlp_space(trial):
    return {"width": trial.suggest_categorical("width", [64, 128, 256]), "lr": trial.suggest_float("lr", 1e-4, 3e-2, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-1, log=True), "dropout": trial.suggest_categorical("dropout", [0.0, 0.1, 0.2])}


def make_mlp(d, width, out, dropout):
    """Exactly one hidden layer. The output layer is zero-initialized so an anchored model starts at g = 0 (logits = offset)."""
    torch = _torch(); nn = torch.nn
    net = nn.Sequential(nn.Linear(d, width), nn.GELU(), nn.Dropout(dropout), nn.Linear(width, out))
    nn.init.zeros_(net[3].weight); nn.init.zeros_(net[3].bias)
    return net.float()          # float32 regardless of torch's global default (stage0_fit sets float64 at import)


def _mlp_train(p, Xs, y, offset, task, K, seed, epochs, Xv=None, yv=None, offv=None, device="cpu"):
    """AdamW, batch 256, loss = CE(offset + g) or MSE. With validation rows: early stopping on validation loss (patience 20), returns the
    best-epoch state; without: trains exactly `epochs` epochs."""
    torch = _torch(); import torch.nn.functional as F
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed)
    out = 1 if task == "regression" else K
    net = make_mlp(Xs.shape[1], p["width"], out, p["dropout"]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    T = lambda a, dt=torch.float32: torch.as_tensor(np.asarray(a), dtype=dt, device=device)  # noqa: E731
    X, O = T(Xs), (T(offset) if offset is not None else None)
    Y = T(y, torch.float32 if task == "regression" else torch.long)

    def loss_of(xb, yb, ob):
        o = net(xb)
        if task == "regression":
            return F.mse_loss(o[:, 0], yb)
        return F.cross_entropy(o if ob is None else o + ob, yb)
    if Xv is not None:
        XV, OV = T(Xv), (T(offv) if offv is not None else None); YV = T(yv, torch.float32 if task == "regression" else torch.long)
    best, best_ep, best_state, bad = math.inf, 0, None, 0
    n = len(y)
    for ep in range(1, epochs + 1):
        net.train(); perm = torch.randperm(n, generator=g).to(device)
        for s in range(0, n, MLP_BATCH):
            bi = perm[s:s + MLP_BATCH]
            opt.zero_grad(); l = loss_of(X[bi], Y[bi], None if O is None else O[bi]); l.backward(); opt.step()
        if Xv is None:
            continue
        net.eval()
        with torch.no_grad():
            vl = float(loss_of(XV, YV, OV))
        if vl < best - 1e-7:
            best, best_ep, bad = vl, ep, 0; best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= MLP_PATIENCE:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net, best_ep


def _mlp_predict(m, X, offset):
    torch = _torch()
    with torch.no_grad():
        o = m.head["net"](torch.as_tensor(m.transform(X), dtype=torch.float32)).double().numpy()
    if m.task == "regression":
        return o[:, 0]
    return o if offset is None else np.asarray(offset, dtype=np.float64) + o


PREDICT = {"linear": _lin_predict, "poly2": _lin_predict, "rff": _lin_predict, "knn": _knn_predict, "lgbm": _lgbm_predict, "mlp": _mlp_predict}


# ---- selection ----------------------------------------------------------------------------------------------------------------------
def _sub(a, idx):
    return None if a is None else a[idx]


def _grid_family_candidates(family, task, anchored):
    if family == "linear":
        return [{}]
    if family == "poly2":
        return [{"sketch_dim": m} for m in POLY_DIMS]
    if family == "rff":
        return [{"gamma_mult": g} for g in RFF_MULTS]
    if family == "knn":
        return [{}]
    raise ValueError(family)


def _grid_transform(family, X_fit, outer):
    if family == "linear":
        s = Std().fit(X_fit); return s, {}
    if family == "poly2":
        return _poly_transform(X_fit, outer["sketch_dim"]), {}
    if family == "rff":
        return _rff_transform(X_fit, outer["gamma_mult"])
    raise ValueError(family)


def _eval_grid(family, X, y, offset, task, K, ctx: HPOContext, anchored):
    """Every grid configuration on every inner split. Returns (trials, meta)."""
    trials, meta = [], {}
    if family == "knn":
        combos = [(k, w, b) for k in KNN_K for w in KNN_WEIGHTS for b in (KNN_BETA if (anchored and task == "classification") else (1.0,))]
        per = {c: [] for c in combos}; jl = []
        for si, (fi, vi) in enumerate(ctx.inner_splits):
            T, m = _knn_transform(X[fi]); jl.append(m)
            d, i = _knn_query(T(X[fi]), T(X[vi]), max(KNN_K))
            for (k, w, b) in combos:
                s = knn_scores(d, i, y[fi], task, K, k, w)
                pred = s if task == "regression" else s * b + (0 if offset is None else offset[vi])
                per[(k, w, b)].append(ctx.objective(pred, y[vi], vi))
        for (k, w, b), v in per.items():
            p = {"k": k, "weighting": w} | ({"beta": b} if (anchored and task == "classification") else {})
            trials.append({"params": p, "split_objectives": v, "objective": float(np.mean(v))})
        meta["jl"] = jl
        return trials, meta
    for outer in _grid_family_candidates(family, task, anchored):
        per = {lam: [] for lam in LAMBDA_GRID}; conv = {lam: [] for lam in LAMBDA_GRID}; tmeta = []
        for fi, vi in ctx.inner_splits:
            T, tm = _grid_transform(family, X[fi], outer); tmeta.append(tm)
            Phi_f, Phi_v = T(X[fi]), T(X[vi])
            for h in linear_path(Phi_f, y[fi], _sub(offset, fi), task, LAMBDA_GRID, K=K):
                g = Phi_v @ h["W"] + h["b"]
                pred = g if (task == "regression" or offset is None) else offset[vi] + g
                per[h["lambda"]].append(ctx.objective(pred, y[vi], vi)); conv[h["lambda"]].append(h["converged"])
        for lam in LAMBDA_GRID:
            trials.append({"params": dict(outer, **{"lambda": lam}), "split_objectives": per[lam], "objective": float(np.mean(per[lam])),
                           "converged": all(conv[lam]), "transform_meta": tmeta})
    return trials, meta


def _optuna_study(family, X, y, offset, task, K, ctx: HPOContext, n_trials, device="cpu"):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    seed = ctx.study_seed(family)
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=f"{ctx.experiment}:{family}:{ctx.ids}")
    records = []
    prep = []
    for fi, vi in ctx.inner_splits:
        if family == "mlp":
            s = Std().fit(X[fi]); prep.append((s(X[fi]), s(X[vi])))
        else:
            prep.append((X[fi], X[vi]))

    def obj(trial):
        p = lgbm_space(trial) if family == "lgbm" else mlp_space(trial)
        t0 = time.time(); vals, iters = [], []
        for si, ((fi, vi), (Xf, Xv)) in enumerate(zip(ctx.inner_splits, prep)):
            tseed = derive_seed(seed, trial.number, si)
            if family == "lgbm":
                bst = _lgbm_train(p, Xf, y[fi], _sub(offset, fi), task, K, tseed, LGBM_MAX_ROUNDS, Xv, y[vi], _sub(offset, vi))
                it = int(bst.best_iteration or bst.current_iteration()); r = _lgbm_raw(bst, Xv, task, K, it)
                pred = r if (task == "regression" or offset is None) else offset[vi] + r
            else:
                net, it = _mlp_train(p, Xf, y[fi], _sub(offset, fi), task, K, tseed, MLP_MAX_EPOCHS, Xv, y[vi], _sub(offset, vi), device=device)
                torch = _torch()
                with torch.no_grad():
                    o = net(torch.as_tensor(Xv, dtype=torch.float32, device=device)).double().cpu().numpy()
                pred = o[:, 0] if task == "regression" else (o if offset is None else offset[vi] + o)
            vals.append(ctx.objective(pred, y[vi], vi)); iters.append(max(int(it), 1))
        v = float(np.mean(vals))
        records.append({"number": trial.number, "params": p, "split_objectives": vals, "objective": v, "best_iters": iters, "seconds": time.time() - t0})
        trial.set_user_attr("best_iters", iters)
        print(f"    [{family} trial {trial.number}] objective={v:.6f} iters={iters} {time.time() - t0:.1f}s", flush=True)
        return v
    study.optimize(obj, n_trials=n_trials, n_jobs=1)
    return records, {"sampler": "TPESampler", "sampler_seed": seed, "n_trials": n_trials, "best_number": study.best_trial.number}


def fit_decoder(family: str, X, y, task: str, ctx: HPOContext, offset=None, K: Optional[int] = None, n_trials: Optional[int] = None,
                device: str = "cpu", allow_null: Optional[bool] = None, X_refit=None) -> Tuple[Fitted, Dict]:
    """Nested selection + refit. X, y, offset: ALL permitted training rows (outer-train); ctx.inner_splits index into them.
    X_refit (optional, same rows/shape as X) replaces X for the final refit only (used by shuffled controls whose refit rows carry a
    different within-partition permutation). Returns (fitted model on all training rows, study record). Never receives evaluation rows."""
    assert family in FAMILIES, family
    assert task in ("classification", "regression")
    X = np.asarray(X); y = np.asarray(y); ctx.validate(len(y))
    anchored = offset is not None
    if task == "classification":
        K = offset.shape[1] if anchored else (K or int(y.max()) + 1)
    allow_null = anchored if allow_null is None else allow_null
    n_trials = N_TRIALS if n_trials is None else n_trials
    t0 = time.time()
    if family in OPTUNA_FAMILIES:
        trials, smeta = _optuna_study(family, X, y, offset, task, K, ctx, n_trials, device)
        hpo = "optuna"
    else:
        trials, smeta = _eval_grid(family, X, y, offset, task, K, ctx, anchored)
        hpo = "grid"
    cands = list(trials)
    if allow_null:
        nv = [ctx.objective(offset[vi], y[vi], vi) for _, vi in ctx.inner_splits]
        null = {"params": {"null": True}, "split_objectives": nv, "objective": float(np.mean(nv))}
        cands = [null] + cands
    best = min(range(len(cands)), key=lambda j: (cands[j]["objective"], j))      # ties -> null, then earlier (stronger) candidate
    sel = cands[best]
    record = {"panel_version": PANEL_VERSION, "space_version": SPACE_VERSION, "family": family, "hpo": hpo, "mode": ctx.mode,
              "experiment": ctx.experiment, "ids": {k: str(v) for k, v in ctx.ids.items()}, "study_seed": ctx.study_seed(family),
              "n_inner_splits": len(ctx.inner_splits), "trials": trials, "null_candidate": cands[0] if allow_null else None,
              "selected": sel["params"], "selected_objective": sel["objective"], "study_meta": smeta, "hpo_seconds": time.time() - t0}
    record["grid_edge"] = _edge_flags(family, sel["params"])
    t1 = time.time()
    if sel["params"].get("null"):
        model = Fitted(family, task, {"null": True}, transform=lambda A: A, null=True)
    else:
        Xr = X if X_refit is None else np.asarray(X_refit)
        assert Xr.shape == X.shape
        model = _refit(family, Xr, y, offset, task, K, sel, ctx, device)
    record["refit_seconds"] = time.time() - t1
    record["refit_meta"] = model.meta
    return model, record


def _edge_flags(family, p):
    f = {}
    if "lambda" in p:
        f["lambda_at_edge"] = p["lambda"] in (LAMBDA_GRID[0], LAMBDA_GRID[-1])
    if family == "poly2" and "sketch_dim" in p:
        f["sketch_dim_at_edge"] = p["sketch_dim"] in (POLY_DIMS[0], POLY_DIMS[-1])
    if family == "rff" and "gamma_mult" in p:
        f["gamma_at_edge"] = p["gamma_mult"] in (RFF_MULTS[0], RFF_MULTS[-1])
    if family == "knn" and "k" in p:
        f["k_at_edge"] = p["k"] in (KNN_K[0], KNN_K[-1])
    return f


def _refit(family, X, y, offset, task, K, sel, ctx, device="cpu"):
    p = dict(sel["params"])
    if family in ("linear", "poly2", "rff"):
        outer = {k: v for k, v in p.items() if k != "lambda"}
        T, tm = _grid_transform(family, X, outer)
        # refit along the same strong->weak path up to the selected lambda (identical warm-start numerics as selection)
        lams = LAMBDA_GRID[:LAMBDA_GRID.index(p["lambda"]) + 1]
        h = linear_path(T(X), y, offset, task, lams, K=K)[-1]
        return Fitted(family, task, p, T, head={"W": h["W"], "b": h["b"]}, meta={"converged": h["converged"], "retried": h["retried"], "grad_inf": h["grad_inf"], **tm})
    if family == "knn":
        T, tm = _knn_transform(X)
        return Fitted(family, task, p, T, head={"train_T": T(X), "y": y, "K": K}, meta=tm)
    iters = int(round(float(np.mean(sel["best_iters"]))))
    seed = derive_seed(ctx.study_seed(family), "refit")
    if family == "lgbm":
        bst = _lgbm_train(p, X, y, offset, task, K, seed, iters)
        return Fitted(family, task, p, lambda A: A, head={"booster": bst, "K": K}, meta={"rounds": iters, "raw_coordinates": True})
    s = Std().fit(X)
    net, _ = _mlp_train(p, s(X), y, offset, task, K, seed, iters, device=device)
    net = net.cpu()
    return Fitted(family, task, p, s, head={"net": net}, meta={"epochs": iters, "hidden_layers": 1})


def save_record(path: str, record: Dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(record, f, indent=0, default=_jsonable)
    os.replace(tmp, path)


def _jsonable(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)
