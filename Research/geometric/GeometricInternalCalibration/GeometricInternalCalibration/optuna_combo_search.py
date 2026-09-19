#!/usr/bin/env python3
"""
Optuna search for multi-stage layer-selection combos (validation-only).
- Single-objective #1: maximize win-rate vs best post-hoc on held-out seeds
- Single-objective #2: minimize mean ECE gap on held-out seeds
- Multi-objective: (maximize win-rate, minimize mean ECE gap)

This file depends on EnhancedMetricAnalyzer already having loaded results.
We do NOT touch test data during selection; test is for evaluation only.
"""

from __future__ import annotations
import numpy as np
import optuna
from typing import Dict, Any, List, Tuple, Optional
from scipy.stats import kendalltau
from optuna.study import StudyDirection
from optuna.importance import get_param_importances
import os
import json
import time
import pandas as pd

# ---- Safe param importances helper (with timeout) ----------------------------

def _safe_param_importances(study, timeout_s=2.0):
    """
    Compute param importances with a timeout to prevent hangs.
    Returns empty dict if computation times out or fails.
    """
    import threading
    out = {}
    err = [None]
    
    def worker():
        try:
            imps = get_param_importances(study)
            out.update(imps)
        except Exception as e:
            err[0] = e
    
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    t.join(timeout_s)
    
    if t.is_alive():
        # Thread is still running - timed out
        return {}
    
    return out

# ---- Small helpers -----------------------------------------------------------

def _to_float(v, default=np.nan):
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default

def _val_ece_from_feats(feats: Dict[str, Any]) -> float:
    """Map your validation features to an absolute ECE estimate in [0,1]."""
    u = _to_float(feats.get('kfold_ece_utility', np.nan), np.nan)
    if np.isfinite(u) and 0.0 <= u <= 1.0:
        return float(1.0 - u)
    es = _to_float(feats.get('ece_score', np.nan), np.nan)  # negative ECE
    if np.isfinite(es):
        return float(np.clip(-es, 0.0, 1.0))
    return 1.0

# ---- Feature pool for progressive combos (validation-only metrics) -----------

METRIC_POOL = [
    # core & cheap
    "calibration_decisiveness",
    "confidence_distance_correlation",
    "boundary_proximity_correlation",
    "uncertainty_geometry_alignment",
    "avg_class_separation_ratio",
    "class_separation_uniformity",
    "local_intrinsic_dimensionality",
    "reliability_curve_quality",
    "prototype_softmax_ece",           # negative ECE internally by your convention
    "label_cka",
    "margin_tail_cvar",
    "multiscale_separation",
    "logistic_calibratability",

    # additional
    "kfold_ece_utility",               # [0,1], higher better
    "decisiveness_tail_mass",
    "confidence_entropy",
    "confident_error_rate_tau",
    "aurc_proxy",
    "class_balanced_margin_cvar",
    "impostor_gap_cvar",
    "margin_skewkurt_safety",
    "temperature_estimate_strength",

    # expensive
    "geometry_error_concordance",
    "label_cka_hsic",
    "spearman_stability_accuracy",
    "ece_score",                        # NEGATIVE ECE (higher better already)
]

def _borda_rank(layers_feats, spec):
    """ layers_feats: [(L, feats)], spec: [(feature_name, maximize_bool), ...] """
    n = len(layers_feats)
    scores = {L: 0 for L, _ in layers_feats}
    for (name, maximize) in spec:
        ranked = sorted(
            layers_feats,
            key=lambda t: _to_float((t[1] or {}).get(name, 0.0), 0.0),
            reverse=maximize
        )
        for r, (L, _) in enumerate(ranked):
            scores[L] += (n - 1 - r)
    return scores

def _eps_band(val_map: Dict[int, float], eps: float) -> List[int]:
    if not val_map:
        return []
    best = min(val_map.values())
    band = [L for L, e in val_map.items() if e <= best + eps]
    if not band:
        band = [min(val_map, key=val_map.get)]
    return sorted(set(band))

# ---- Fusion helper -----------------------------------------------------------

def _fusion_score(L: int, feats: Dict[str, Any], val_e: float) -> float:
    # normalize weights; if all zero, fallback to decisiveness
    w_dec = max(0.0, feats.get('w_dec', 0.0))
    w_cvar = max(0.0, feats.get('w_cvar', 0.0))
    w_imp = max(0.0, feats.get('w_imp', 0.0))
    s = w_dec + w_cvar + w_imp
    if s <= 1e-12:
        w_dec, w_cvar, w_imp = 1.0, 0.0, 0.0
    else:
        w_dec, w_cvar, w_imp = w_dec / s, w_cvar / s, w_imp / s

    dec = _to_float(feats.get('calibration_decisiveness', np.nan), -np.inf)
    cvar = _to_float(feats.get('margin_tail_cvar', np.nan), -np.inf)
    imp = _to_float(feats.get('impostor_gap_cvar', np.nan), -np.inf)
    # prefer high decisiveness/cvar/imp; penalize poor val ECE slightly
    return (w_dec * dec + w_cvar * cvar + w_imp * imp) - 0.05 * _to_float(val_e, 1.0)

# ---- DIAGNOSTICS: rank correlation between proxies and test ECE -------------

def _val_proxy(layer_feats: Dict[str, Any]) -> float:
    # same mapping as analyzer._val_ece_from_feats but local + returns utility (higher better)
    u = _to_float(layer_feats.get('kfold_ece_utility', np.nan), np.nan)
    if np.isfinite(u) and 0.0 <= u <= 1.0:
        return u
    es = _to_float(layer_feats.get('ece_score', np.nan), np.nan)
    if np.isfinite(es):
        # ece_score is negative ECE -> higher is better
        return es
    return np.nan

def compute_proxy_vs_test_corr(analyzer, split, part: str = 'train') -> Dict[str, Any]:
    """
    For each (tm,ds,mdl,seed), compute rank corr between:
      - validation proxy ranks (descending utility)
      - realized test ECE ranks (ascending ECE)
    Returns macro stats for logging.
    """
    rows = []
    for (tm, ds, mdl), parts in split.items():
        for seed in parts.get(part, []):
            sdata = analyzer.results.get(tm, {}).get(ds, {}).get(mdl, {}).get(seed, {})
            lm = sdata.get('layer_metrics') or {}
            te = sdata.get('layer_eces') or {}
            if not lm or not te:
                continue
            Ls = sorted(set(lm.keys()) & set(te.keys()))
            if len(Ls) < 3:
                continue
            proxy = []
            test = []
            for L in Ls:
                p = _val_proxy(lm[L])
                e = _to_float(te[L], np.nan)
                if np.isfinite(p) and np.isfinite(e):
                    proxy.append(p)
                    test.append(e)
            if len(proxy) >= 3:
                # proxy: higher is better; test: lower is better
                # invert test so both are "higher better" for tau
                try:
                    tau, _ = kendalltau(proxy, [-x for x in test])
                    if np.isfinite(tau):
                        rows.append(float(tau))
                except Exception:
                    pass
    if not rows:
        return {'tau_mean': np.nan, 'tau_median': np.nan, 'n': 0}
    return {'tau_mean': float(np.mean(rows)), 'tau_median': float(np.median(rows)), 'n': int(len(rows))}

# ---- A paramized cascade (search space is here) ------------------------------

def run_cascade(layer_metrics: Dict[int, Dict[str, Any]],
                params: Dict[str, Any]) -> Optional[int]:
    """
    Return chosen layer ID or None.
    params include:
      - eps: float
      - order: one of {'K→D→CVAR→IMP', 'IMP→D→K', 'Borda', 'K∩(CVAR∪IMP)→D'}
      - K1, K2, K3: integer quotas (after stages)
      - delta: validation ECE band (adds tolerance around best val)
      - decisiveness_tau: optional threshold (if not met, fallback to K winner)
      - weights for Borda / score fusion (optional)
    """
    Ls = sorted(layer_metrics.keys())
    if not Ls:
        return None

    # compute per-layer val ECE (validation-only)
    val_map = {L: _val_ece_from_feats(layer_metrics[L] or {}) for L in Ls}
    band = _eps_band(val_map, params['eps'] + params['delta'])

    # Adaptive: if the band is huge → tighten; if tiny → relax slightly
    if params.get('order') in ('K→Adaptive→Fusion',):
        if len(band) >= 10:
            band = _eps_band(val_map, max(0.0, params['eps'] - 5e-4))
        elif len(band) <= 3:
            band = _eps_band(val_map, params['eps'] + 5e-4)

    def _topk_by(feature, k, maximize=True, pool=None):
        pool = pool if pool is not None else band
        ranked = sorted(pool, key=lambda L: _to_float(layer_metrics[L].get(feature, np.nan), -np.inf), reverse=maximize)
        return ranked[:min(k, len(ranked))]

    def _best_kfold(pool):
        # prefer kfold utility; fallback to ece_score; final fallback val ece
        def score(L):
            u = _to_float(layer_metrics[L].get('kfold_ece_utility', np.nan), np.nan)
            if np.isfinite(u): return u
            es = _to_float(layer_metrics[L].get('ece_score', np.nan), np.nan)
            if np.isfinite(es): return -es
            return 1.0 - _to_float(val_map.get(L, 1.0), 1.0)
        return max(pool, key=score)

    order = params['order']
    K1, K2, K3 = params['K1'], params['K2'], params['K3']
    tau = params['decisiveness_tau']
    w_pack = {'w_dec': params.get('w_dec', 0.0), 'w_cvar': params.get('w_cvar', 0.0), 'w_imp': params.get('w_imp', 0.0)}

    # --- K: kfold filter
    def stage_K(pool):
        # band + top-K1 by kfold-ish score
        if not pool: return []
        cand = sorted(pool, key=lambda L: (
            _to_float(layer_metrics[L].get('kfold_ece_utility', -1.0), -1.0),
            -_to_float(layer_metrics[L].get('ece_score', np.nan), np.inf)  # ece_score is negative ECE
        ), reverse=True)[:min(K1, len(pool))]
        return cand

    # --- D: decisiveness
    def stage_D(pool):
        if not pool: return []
        cand = sorted(pool, key=lambda L: _to_float(layer_metrics[L].get('calibration_decisiveness', np.nan), -np.inf), reverse=True)
        return cand[:min(K2, len(cand))]

    # --- CVAR: margin tail CVaR
    def stage_CVAR(pool):
        if not pool: return []
        cand = sorted(pool, key=lambda L: _to_float(layer_metrics[L].get('margin_tail_cvar', np.nan), -np.inf), reverse=True)
        return cand[:min(K3, len(cand))]

    # --- IMP: impostor gap CVaR (final resolve)
    def stage_IMP_pick(pool):
        if not pool: return None
        # optional D threshold: if no item passes tau, fallback to best Kfold among pool
        if tau is not None:
            good = [L for L in pool if _to_float(layer_metrics[L].get('calibration_decisiveness', np.nan), -np.inf) >= tau]
            pool_use = good if good else pool
        else:
            pool_use = pool
        return max(pool_use, key=lambda L: _to_float(layer_metrics[L].get('impostor_gap_cvar', np.nan), -np.inf))

    # --- Borda fusion
    def stage_Borda_pick(pool):
        if not pool: return None
        lf = [(L, layer_metrics[L]) for L in pool]
        borda = _borda_rank(lf, [
            ('kfold_ece_utility', True),
            ('calibration_decisiveness', True),
            ('margin_tail_cvar', True),
            ('impostor_gap_cvar', True),
            ('boundary_proximity_correlation', True),
        ])
        return max(borda.items(), key=lambda kv: kv[1])[0] if borda else pool[0]

    # --- K∩(CVAR∪IMP)→D path
    def stage_intersection():
        # K banded shortlist
        kset = set(stage_K(band))
        # Best 5 by CVAR + best 5 by IMP
        c5 = set(_topk_by('margin_tail_cvar', 5, maximize=True, pool=band))
        i5 = set(_topk_by('impostor_gap_cvar', 5, maximize=True, pool=band))
        pool = sorted(kset & (c5 | i5))
        if not pool:
            pool = sorted(kset | c5 | i5) or band
        after_D = stage_D(pool)
        return stage_IMP_pick(after_D or pool)

    # --- FUSION: weighted fusion score
    def stage_FUSION_pick(pool):
        if not pool:
            return None
        return max(pool, key=lambda L: _fusion_score(L, {**layer_metrics[L], **w_pack}, _to_float(val_map[L], 1.0)))

    # Execute the order
    if order == 'K→D→CVAR→IMP':
        sK = stage_K(band)
        sD = stage_D(sK)
        sC = stage_CVAR(sD)
        picked = stage_IMP_pick(sC or sD or sK or band)
    elif order == 'IMP→D→K':
        # tail-first
        t5 = _topk_by('impostor_gap_cvar', 5, True, band)
        sD = stage_D(t5)
        picked = _best_kfold(sD or t5 or band)
    elif order == 'Borda':
        picked = stage_Borda_pick(band)
    elif order == 'K∩(CVAR∪IMP)→D':
        picked = stage_intersection()
    elif order == 'K→Adaptive→Fusion':
        sK = stage_K(band)
        sD = stage_D(sK)
        cand = sD or sK or band
        picked = stage_FUSION_pick(cand)
    else:
        picked = None

    return int(picked) if picked is not None else None

# ---- Evaluation over seeds ---------------------------------------------------

def evaluate_params_on_split(analyzer,
                             params: Dict[str, Any],
                             split: Dict[Tuple[str, str, str], Dict[str, List[str]]],
                             part: str = 'train',
                             mode: str = 'macro',           # 'macro' or 'micro'
                             seed_weighting: bool = True) -> Dict[str, float]:
    """
    mode='macro': average per (tm,ds,mdl), then average groups (less dominated by big groups)
    mode='micro': average across all seeds (higher power if groups are small)
    seed_weighting (macro only): weight groups by their #seeds
    """
    rows = []
    chosen_layers = []   # telemetry
    band_sizes = []      # telemetry
    for (tm, ds, mdl), parts in split.items():
        chosen_seeds = parts.get(part, [])
        if not chosen_seeds:
            continue
        model_data = analyzer.results.get(tm, {}).get(ds, {}).get(mdl, {})
        for seed in chosen_seeds:
            sdata = model_data.get(seed, {})
            lm = sdata.get('layer_metrics') or {}
            te = sdata.get('layer_eces') or {}
            if not lm or not te:
                continue
            candidate_layers = sorted(set(lm.keys()) & set(te.keys()))
            # compute band size based on local validation proxy
            if candidate_layers:
                val_map = {int(L): _val_ece_from_feats(lm[L] or {}) for L in candidate_layers}
                try:
                    band = _eps_band(val_map, params['eps'] + params['delta'])
                    band_sizes.append(len(band))
                except Exception:
                    pass
            chosen_L = run_cascade(lm, params)
            if chosen_L is None or chosen_L not in te:
                continue
            chosen_ece = _to_float(te[chosen_L], np.nan)
            if not np.isfinite(chosen_ece):
                continue
            chosen_layers.append(chosen_L)
            best_true = min(_to_float(te[L], np.inf) for L in te)
            gap = max(0.0, chosen_ece - best_true)
            best_m, best_posthoc = analyzer._best_baseline_ece(sdata.get('posthoc', {}) or {})
            beats = (best_posthoc is not None) and (chosen_ece < best_posthoc)
            rows.append({'group': (tm, ds, mdl), 'gap': gap, 'beats': float(beats)})

    if not rows:
        return {'win_rate': 0.0, 'mean_gap': 1.0}

    import pandas as pd
    df = pd.DataFrame(rows)

    if mode == 'micro':
        out = {'win_rate': float(df['beats'].mean()),
               'mean_gap': float(df['gap'].mean())}
        if chosen_layers:
            out['mean_chosen_layer'] = float(np.mean(chosen_layers))
        if band_sizes:
            out['mean_band_size'] = float(np.mean(band_sizes))
        return out

    # macro (grouped)
    out = []
    for g, grp in df.groupby('group'):
        out.append({
            'group': g,
            'n': len(grp),
            'win_rate': grp['beats'].mean(),
            'mean_gap': grp['gap'].mean()
        })
    G = pd.DataFrame(out)
    if seed_weighting:
        w = G['n'] / G['n'].sum()
        out = {
            'win_rate': float((G['win_rate'] * w).sum()),
            'mean_gap': float((G['mean_gap'] * w).sum())
        }
    else:
        out = {
            'win_rate': float(G['win_rate'].mean()),
            'mean_gap': float(G['mean_gap'].mean())
        }
    if chosen_layers:
        out['mean_chosen_layer'] = float(np.mean(chosen_layers))
    if band_sizes:
        out['mean_band_size'] = float(np.mean(band_sizes))
    return out

# ---- Cross-seed CV helpers ---------------------------------------------------

def _seed_folds(split, part: str = 'train', k: int = 3, rng_seed: int = 0):
    """
    Build k folds over seeds per (train_loss,dataset,model).
    Returns: List[ {(tm,ds,mdl): {'train': [...], 'val': [...]}} ]
    """
    rng = np.random.default_rng(rng_seed)
    folds = []
    for _ in range(max(1, int(k))):
        fold = {}
        for key, parts in split.items():
            seeds = list(parts.get(part, []))
            if not seeds:
                fold[key] = {'train': [], 'val': []}
                continue
            rng.shuffle(seeds)
            cut = max(1, int(round(0.8 * len(seeds))))
            inner_train = seeds[:cut]
            inner_val = seeds[cut:] if seeds[cut:] else [seeds[-1]]
            fold[key] = {'train': inner_train, 'val': inner_val}
        folds.append(fold)
    return folds

def _cv_objective(analyzer, split, params: Dict[str, Any], k: int = 3, rng_seed: int = 0):
    """
    Return (win_rate_mean, mean_gap_mean) averaged over k folds
    built from training seeds (80/20 per (tm,ds,mdl)).
    """
    folds = _seed_folds(split, part='train', k=k, rng_seed=rng_seed)
    wrs, gaps = [], []
    for f in folds:
        perf = evaluate_params_on_split(analyzer, params, f, part='val', mode='macro', seed_weighting=True)
        wrs.append(float(perf.get('win_rate', 0.0)))
        gaps.append(float(perf.get('mean_gap', 1.0)))
    return float(np.mean(wrs) if wrs else 0.0), float(np.mean(gaps) if gaps else 1.0)

# ---- Param expansion helpers -------------------------------------------------

def expand_quota_params(p: Dict[str, Any]) -> Dict[str, Any]:
    """Return params dict with effective K2/K3 materialized from raw, if needed."""
    p = dict(p or {})
    k1 = int(p.get('K1', 0))
    try:
        k2 = int(p['K2']) if 'K2' in p else int(min(int(p.get('K2_raw', k1)), k1))
    except Exception:
        k2 = int(k1)
    try:
        k3 = int(p['K3']) if 'K3' in p else int(min(int(p.get('K3_raw', k2)), k2))
    except Exception:
        k3 = int(k2)
    p['K1'], p['K2'], p['K3'] = int(k1), int(k2), int(k3)
    return p

def _effective_k2k3(params: Dict[str, Any]) -> Tuple[int, int]:
    p = expand_quota_params(params)
    return int(p.get('K2', 0)), int(p.get('K3', 0))

# ---- Optuna studies ----------------------------------------------------------

def make_search_space(trial: optuna.trial.Trial, focused: bool = True) -> Dict[str, Any]:
    """
    focused=True: narrow space for small trial budgets
    focused=False: wider exploratory space
    """
    if focused:
        order = trial.suggest_categorical('order', [
            'K→D→CVAR→IMP',
            'Borda',
            'K∩(CVAR∪IMP)→D',
        ])
        # keep eps in sensible bands
        eps = trial.suggest_categorical('eps', [5e-4, 1e-3, 2e-3, 5e-3])
        delta = trial.suggest_categorical('delta', [0.0, 5e-4, 1e-3, 2e-3])
        # constrained quotas: K1 >= K2 >= K3
        K1 = trial.suggest_int('K1', 3, 100, step=1)
        K2_raw = trial.suggest_int('K2_raw', 2, 100, step=1)
        K3_raw = trial.suggest_int('K3_raw', 1, 100, step=1)
        K2 = min(K2_raw, K1)
        K3 = min(K3_raw, K2)
        decisiveness_tau = trial.suggest_categorical('decisiveness_tau', [None, 0.1, 0.2, 0.3])
    else:
        order = trial.suggest_categorical('order', [
            'K→D→CVAR→IMP', 'IMP→D→K', 'Borda', 'K∩(CVAR∪IMP)→D', 'K→Adaptive→Fusion'
        ])
        eps = trial.suggest_categorical('eps', [5e-4, 1e-3, 1.5e-3, 2e-3, 5e-3])
        delta = trial.suggest_categorical('delta', [0.0, 5e-4, 1e-3, 2e-3])
        K1 = trial.suggest_int('K1', 3, 100, step=1)
        K2_raw = trial.suggest_int('K2_raw', 2, 100, step=1)
        K3_raw = trial.suggest_int('K3_raw', 1, 100, step=1)
        K2 = min(K2_raw, K1)
        K3 = min(K3_raw, K2)
        decisiveness_tau = trial.suggest_categorical('decisiveness_tau', [None, 0.0, 0.1, 0.2, 0.3])

    # optional fusion weights (for Fusion/Borda/adaptive paths)
    # we’ll normalize later; keep small space
    # coarse-grained weights (0.05 grid)
    w_dec = trial.suggest_float('w_dec', 0.0, 1.0, step=0.05)
    w_cvar = trial.suggest_float('w_cvar', 0.0, 1.0, step=0.05)
    w_imp = trial.suggest_float('w_imp', 0.0, 1.0, step=0.05)

    return {
        'order': order, 'eps': float(eps), 'delta': float(delta),
        'K1': int(K1), 'K2': int(K2), 'K3': int(K3),
        'decisiveness_tau': None if decisiveness_tau is None else float(decisiveness_tau),
        'w_dec': float(w_dec), 'w_cvar': float(w_cvar), 'w_imp': float(w_imp),
        'focused': bool(focused),
    }

def _sample_and_eval_invented(trial, analyzer, split, part='train'):
    """Sample an invented combo and evaluate it, returning key metrics.

    Ensures invented-combo search works across all studies when enabled.
    """
    try:
        max_layers_available = analyzer.num_layers(split, part=part) if hasattr(analyzer, 'num_layers') else 12
    except Exception:
        max_layers_available = 12

    combo = _sample_invented_combo(trial, max_layers_available)

    # Re-hydrate the progressive list for readability and evaluation
    prog = []
    for stage in combo.get('progressive', ()):
        d = dict(stage)
        prog.append({"metric": d.get("metric"), "keep_k": int(d.get("keep_k", 1))})

    try:
        res = analyzer.evaluate_combo(
            split=split,
            eps=combo["eps"],
            progressive=prog,
            strict_band=combo.get("strict_band"),
            refine=combo.get("refine"),
            decisiveness_topm=combo.get("decisiveness_topm"),
            kfold=combo.get("kfold"),
            weights=combo.get("weights"),
            part=part
        )
        beats_rate = float(res.get("beats_rate", 0.0))
        mean_gap   = float(res.get("mean_gap", 1.0))
        robustness = float(res.get("robustness", 0.0))
        within_eps = float(res.get("within_epsilon", 0.0))
    except Exception:
        beats_rate, mean_gap, robustness, within_eps = 0.0, 1.0, 0.0, 0.0

    # Store readable combo on the trial for exports/audits
    trial.set_user_attr("combo", {
        **combo,
        "progressive": prog,
    })

    return {
        "beats_rate": beats_rate,
        "mean_gap": mean_gap,
        "robustness": robustness,
        "within_epsilon": within_eps,
    }

def study_winrate(analyzer, split, n_trials=80, seed=777, mode: str = 'macro', invent_in_all: bool = False):
    def objective(trial):
        if invent_in_all:
            res = _sample_and_eval_invented(trial, analyzer, split, part='train')
            # Win-rate analogue = beats_rate from invented combo evaluation
            trial.set_user_attr('mean_gap', res['mean_gap'])
            trial.set_user_attr('robustness', res['robustness'])
            trial.set_user_attr('within_epsilon', res['within_epsilon'])
            return res['beats_rate']
        else:
            params = make_search_space(trial, focused=True)
            # CV over seeds to reduce overfitting
            wr_cv, gp_cv = _cv_objective(analyzer, split, params, k=3, rng_seed=seed)
            # also compute single-pass telemetry on train for diagnostics
            perf = evaluate_params_on_split(analyzer, params, split, part='train', mode=mode, seed_weighting=True)
            trial.set_user_attr('mean_gap', perf.get('mean_gap', np.nan))
            trial.set_user_attr('mean_band_size', perf.get('mean_band_size', np.nan))
            trial.set_user_attr('mean_chosen_layer', perf.get('mean_chosen_layer', np.nan))
            diag = compute_proxy_vs_test_corr(analyzer, split, part='train')
            trial.set_user_attr('tau_mean', diag['tau_mean'])
            trial.set_user_attr('tau_median', diag['tau_median'])
            trial.set_user_attr('tau_n', diag['n'])
            # maximize CV win-rate
            return wr_cv
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, n_startup_trials=40)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=20)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner, study_name='winrate')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    # Param importances can hang with many categorical params; disable by default
    if os.getenv("ENABLE_OPTUNA_IMPORTANCES", "0") == "1":
        try:
            imps = get_param_importances(study)
            print("[winrate] param importances:", imps)
        except Exception:
            pass
    return study

def study_gap(analyzer, split, n_trials=60, seed=778, mode: str = 'macro', invent_in_all: bool = False):
    def objective(trial):
        if invent_in_all:
            res = _sample_and_eval_invented(trial, analyzer, split, part='train')
            trial.set_user_attr('beats_rate', res['beats_rate'])
            trial.set_user_attr('robustness', res['robustness'])
            trial.set_user_attr('within_epsilon', res['within_epsilon'])
            # minimize mean gap
            return res['mean_gap']
        else:
            params = make_search_space(trial, focused=True)
            wr_cv, gp_cv = _cv_objective(analyzer, split, params, k=3, rng_seed=seed)
            perf = evaluate_params_on_split(analyzer, params, split, part='train', mode=mode, seed_weighting=True)
            trial.set_user_attr('win_rate', perf.get('win_rate', np.nan))
            trial.set_user_attr('mean_band_size', perf.get('mean_band_size', np.nan))
            trial.set_user_attr('mean_chosen_layer', perf.get('mean_chosen_layer', np.nan))
            diag = compute_proxy_vs_test_corr(analyzer, split, part='train')
            trial.set_user_attr('tau_mean', diag['tau_mean'])
            trial.set_user_attr('tau_median', diag['tau_median'])
            trial.set_user_attr('tau_n', diag['n'])
            # minimize CV mean gap
            return gp_cv
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, n_startup_trials=40)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=20)
    study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner, study_name='gap')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    # Param importances can hang with many categorical params; disable by default
    if os.getenv("ENABLE_OPTUNA_IMPORTANCES", "0") == "1":
        try:
            imps = get_param_importances(study)
            print("[gap] param importances:", imps)
        except Exception:
            pass
    return study

def study_multi(analyzer, split, n_trials=240, seed=779, mode: str = 'macro', invent_in_all: bool = False):
    def objective(trial):
        if invent_in_all:
            res = _sample_and_eval_invented(trial, analyzer, split, part='train')
            # keep original direction: maximize win_rate, minimize mean_gap
            # Here: win_rate analogue = beats_rate
            trial.set_user_attr('robustness', res['robustness'])
            trial.set_user_attr('within_epsilon', res['within_epsilon'])
            return res['beats_rate'], res['mean_gap']
        else:
            params = make_search_space(trial, focused=True)
            # multi-objective: CV win-rate and CV mean gap
            wr_cv, gp_cv = _cv_objective(analyzer, split, params, k=3, rng_seed=seed)
            perf = evaluate_params_on_split(analyzer, params, split, part='train', mode=mode, seed_weighting=True)
            trial.set_user_attr('mean_band_size', perf.get('mean_band_size', np.nan))
            trial.set_user_attr('mean_chosen_layer', perf.get('mean_chosen_layer', np.nan))
            diag = compute_proxy_vs_test_corr(analyzer, split, part='train')
            trial.set_user_attr('tau_mean', diag['tau_mean'])
            trial.set_user_attr('tau_median', diag['tau_median'])
            trial.set_user_attr('tau_n', diag['n'])
            return wr_cv, gp_cv
    sampler = optuna.samplers.NSGAIISampler(seed=seed, population_size=48)
    study = optuna.create_study(directions=['maximize','minimize'], sampler=sampler, study_name='multi')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study

# ---- Two-stage search (structure → finetune) ---------------------------------

def study_structure(analyzer, split, n_trials=150, seed=880, focused=True, mode='macro', invent_in_all: bool = False):
    def objective(trial):
        if invent_in_all:
            res = _sample_and_eval_invented(trial, analyzer, split, part='train')
            trial.set_user_attr('mean_gap', res['mean_gap'])
            trial.set_user_attr('robustness', res['robustness'])
            trial.set_user_attr('within_epsilon', res['within_epsilon'])
            return res['beats_rate']  # maximize
        else:
            p = make_search_space(trial, focused=focused)
            # freeze quotas to reasonable defaults for speed during structure search
            p['K1'], p['K2'], p['K3'] = 8, 4, 3
            perf = evaluate_params_on_split(analyzer, p, split, part='train', mode=mode, seed_weighting=True)
            trial.set_user_attr('mean_gap', perf['mean_gap'])
            trial.set_user_attr('mean_band_size', perf.get('mean_band_size', np.nan))
            trial.set_user_attr('mean_chosen_layer', perf.get('mean_chosen_layer', np.nan))
            diag = compute_proxy_vs_test_corr(analyzer, split, part='train')
            trial.set_user_attr('tau_mean', diag['tau_mean'])
            trial.set_user_attr('tau_median', diag['tau_median'])
            trial.set_user_attr('tau_n', diag['n'])
            return perf['win_rate']
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name='structure')
    # --- CALLBACK: early stop when stalled --------------------------------------
    class NoImprovementStopper:
        def __init__(self, patience=80):
            self.patience = patience
            self.best = None
            self.since = 0
        def __call__(self, study, trial):
            val = study.best_value if study.best_value is not None else None
            if self.best is None or (val is not None and val > (self.best if self.best is not None else -np.inf) + 1e-12):
                self.best = val
                self.since = 0
            else:
                self.since += 1
                if self.since >= self.patience:
                    study.stop()
    stopper = NoImprovementStopper(patience=max(100, n_trials//2))
    study.optimize(objective, n_trials=n_trials, callbacks=[stopper], show_progress_bar=False)
    return study

def study_finetune(analyzer, split, base_params, n_trials=120, seed=881, mode='macro', invent_in_all: bool = False):
    def objective(trial):
        if invent_in_all:
            res = _sample_and_eval_invented(trial, analyzer, split, part='train')
            trial.set_user_attr('robustness', res['robustness'])
            trial.set_user_attr('within_epsilon', res['within_epsilon'])
            return res['beats_rate'] - 0.5 * res['mean_gap']
        else:
            p = dict(base_params)
            # only tune quotas / tau / fusion weights
            K1 = trial.suggest_int('K1', 3, 100, step=1)
            K2_raw = trial.suggest_int('K2_raw', 2, 100, step=1)
            K3_raw = trial.suggest_int('K3_raw', 1, 100, step=1)
            p['K1'] = int(K1)
            p['K2'] = int(min(K2_raw, K1))
            p['K3'] = int(min(K3_raw, p['K2']))
            p['decisiveness_tau'] = trial.suggest_float('decisiveness_tau', 0.0, 0.75)
            p['w_dec'] = trial.suggest_float('w_dec', 0.0, 0.75)
            p['w_cvar'] = trial.suggest_float('w_cvar', 0.0, 0.75)
            p['w_imp'] = trial.suggest_float('w_imp', 0.0, 0.75)
            perf = evaluate_params_on_split(analyzer, p, split, part='train', mode=mode, seed_weighting=True)
            # a simple scalarized objective: win_rate − λ*gap with λ small
            trial.set_user_attr('mean_band_size', perf.get('mean_band_size', np.nan))
            trial.set_user_attr('mean_chosen_layer', perf.get('mean_chosen_layer', np.nan))
            diag = compute_proxy_vs_test_corr(analyzer, split, part='train')
            trial.set_user_attr('tau_mean', diag['tau_mean'])
            trial.set_user_attr('tau_median', diag['tau_median'])
            trial.set_user_attr('tau_n', diag['n'])
            return perf['win_rate'] - 0.5 * perf['mean_gap']
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction='maximize', sampler=sampler, study_name='finetune')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study

# ---- Trial export helpers (CSV + JSON) ---------------------------------------

def _format_number_7digits(num):
    try:
        if isinstance(num, (np.floating, float)):
            if not np.isfinite(num):
                return num
            return format(float(num), '.7g')
        if isinstance(num, (np.integer, int)):
            return int(num)
        return num
    except Exception:
        return num

def _format_df_7digits(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    df_out = df.copy()
    for col in df_out.columns:
        try:
            if pd.api.types.is_numeric_dtype(df_out[col]):
                df_out[col] = df_out[col].apply(_format_number_7digits)
        except Exception:
            pass
    return df_out

def _trial_to_row(trial: optuna.trial.FrozenTrial):
    vals = trial.values if trial.values is not None else ([trial.value] if trial.value is not None else [])
    row = {
        'number': trial.number,
        'state': str(trial.state),
        'values': vals,
        'params': trial.params,
        'user_attrs': trial.user_attrs,
        'datetime_start': str(trial.datetime_start),
        'datetime_complete': str(trial.datetime_complete),
        'duration_seconds': (trial.datetime_complete - trial.datetime_start).total_seconds() if (trial.datetime_start and trial.datetime_complete) else None,
    }
    return row

def trials_dataframe_with_meta(study: optuna.Study) -> pd.DataFrame:
    rows = [_trial_to_row(t) for t in study.trials]
    df = pd.DataFrame(rows)
    if not df.empty:
        try:
            p = pd.json_normalize(df['params']).add_prefix('p.')
            ua = pd.json_normalize(df['user_attrs']).add_prefix('ua.')
            df = pd.concat([df.drop(columns=['params','user_attrs']), p, ua], axis=1)
        except Exception:
            pass
    return df

def save_top_trials(study: optuna.Study, out_dir: str, top_k: int = 10, name: str = None, fast_mode: bool = False):
    """
    Save top trials to CSV and JSON.
    
    Args:
        fast_mode: If True, skip expensive JSON normalization (faster for checkpoints with many trials)
    """
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base = name or (study.study_name or "study")

    # For fast mode (checkpoints), build a minimal dataframe directly
    if fast_mode and len(study.trials) > 1000:
        print(f"[save_top_trials] Fast mode: building minimal dataframe for {len(study.trials)} trials...")
        rows = []
        for t in study.trials:
            vals = t.values if t.values is not None else ([t.value] if t.value is not None else [])
            rows.append({
                'number': t.number,
                'state': str(t.state),
                'value': vals[0] if vals and len(vals) > 0 else np.nan,
            })
        df = pd.DataFrame(rows)
    else:
        df = trials_dataframe_with_meta(study)
    if hasattr(study, "directions") and getattr(study, 'directions', None):
        dirs = study.directions
        # robust values extraction (handles None/empty)
        if 'values' in df:
            vals_list = df['values'].apply(lambda v: v if isinstance(v, (list, tuple)) else ([v] if v is not None else [])).tolist()
            vals = np.array(vals_list, dtype=float) if vals_list else np.empty((0, len(dirs)))
        else:
            vals = np.empty((0, len(dirs)))
        normed_cols = []
        for j, d in enumerate(dirs):
            col = vals[:, j] if vals.size else np.array([])
            if col.size == 0:
                normed_cols.append(col)
                continue
            mx = np.nanmax(col); mn = np.nanmin(col)
            if d == StudyDirection.MAXIMIZE:
                n = (1.0 - (col - mn) / max(1e-12, (mx - mn)))
            else:
                n = (col - mn) / max(1e-12, (mx - mn))
            normed_cols.append(n)
        normed = np.vstack(normed_cols).T if normed_cols else np.empty((0,0))
        if normed.size:
            df['utopia_distance'] = np.sqrt((normed**2).sum(axis=1))
            df_sorted = df.sort_values('utopia_distance', ascending=True)
        else:
            df_sorted = df.copy()
    else:
        # single-objective path
        ascending = (hasattr(study, "direction") and study.direction == StudyDirection.MINIMIZE)
        if 'values' in df:
            df['value'] = df['values'].apply(lambda v: v[0] if isinstance(v, (list, tuple)) and v else np.nan)
        df_sorted = df.sort_values('value', ascending=ascending) if 'value' in df else df

    top = df_sorted.head(top_k).copy()
    csv_path = os.path.join(out_dir, f"{base}_top{top_k}_{stamp}.csv")
    # Save top-k with 7-digit numeric formatting
    _format_df_7digits(top).to_csv(csv_path, index=False)

    json_path = os.path.join(out_dir, f"{base}_top{top_k}_params_{stamp}.json")
    with open(json_path, "w") as f:
        payload = []
        for _, r in top.iterrows():
            num = int(r['number']) if 'number' in r else None
            if num is not None and 0 <= num < len(study.trials):
                params = study.trials[num].params
            else:
                params = r.get('params', {}) if isinstance(r.get('params', {}), dict) else {}
            k2_eff, k3_eff = _effective_k2k3(params)
            payload.append({
                'number': num, 
                'params': params, 
                'K2_eff': int(k2_eff), 
                'K3_eff': int(k3_eff),
                'combo': study.trials[num].user_attrs.get('combo') if num is not None and 0 <= num < len(study.trials) else None
            })
        # Apply 7-digit formatting to numeric leaves in JSON payload
        def _fmt_leaf(x):
            if isinstance(x, dict):
                return {k: _fmt_leaf(v) for k, v in x.items()}
            if isinstance(x, list):
                return [_fmt_leaf(v) for v in x]
            return _format_number_7digits(x)
        json.dump(_fmt_leaf(payload), f, indent=2)

    full_csv = os.path.join(out_dir, f"{base}_alltrials_{stamp}.csv")
    # also add effective K2/K3 columns to CSV if params exist
    try:
        if not top.empty and 'number' in top.columns:
            k2_list, k3_list = [], []
            for _, r in top.iterrows():
                num = int(r['number']) if 'number' in r else None
                params = study.trials[num].params if num is not None and 0 <= num < len(study.trials) else {}
                k2_eff, k3_eff = _effective_k2k3(params)
                k2_list.append(int(k2_eff)); k3_list.append(int(k3_eff))
            top['p.K2_eff'] = k2_list
            top['p.K3_eff'] = k3_list
    except Exception:
        pass
    df.to_csv(full_csv, index=False)
    return {'top_csv': csv_path, 'top_params_json': json_path, 'all_trials_csv': full_csv}

def export_pareto_trials(study: optuna.Study, out_dir: str, name: str = "pareto"):
    """Export Pareto trials (multi-objective) to JSON and a light CSV."""
    os.makedirs(out_dir, exist_ok=True)
    trials = list(getattr(study, 'best_trials', []) or [])
    if not trials:
        return {}
    stamp = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(out_dir, f"{name}_trials_{stamp}.json")
    csv_path = os.path.join(out_dir, f"{name}_summary_{stamp}.csv")
    # JSON payload with full combos if present
    with open(json_path, 'w') as f:
        json.dump([
            {
                'number': t.number,
                'values': list(t.values) if t.values is not None else None,
                'params': t.params,
                'combo': t.user_attrs.get('combo')
            } for t in trials
        ], f, indent=2)
    # Lightweight CSV with 7-digit formatting
    try:
        import csv
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            # handle up to four objectives (obj0..obj3)
            w.writerow(["trial", "obj0", "obj1", "obj2", "obj3", "has_combo"])
            for t in trials:
                vals = list(t.values) if t.values is not None else []
                row = [
                    _format_number_7digits(t.number),
                    _format_number_7digits(vals[0] if len(vals)>0 else None),
                    _format_number_7digits(vals[1] if len(vals)>1 else None),
                    _format_number_7digits(vals[2] if len(vals)>2 else None),
                    _format_number_7digits(vals[3] if len(vals)>3 else None),
                    bool(t.user_attrs.get('combo'))
                ]
                w.writerow(row)
    except Exception:
        pass
    return {'pareto_json': json_path, 'pareto_csv': csv_path}

def run_two_stage_optuna(analyzer, split, n_struct=150, n_fine=120, focused=True, mode='macro', invent_in_all: bool = False):
    s_struct = study_structure(analyzer, split, n_trials=n_struct, focused=focused, mode=mode, invent_in_all=invent_in_all)
    
    if invent_in_all:
        # For invented combos, just use the best structure combo for finetuning
        t_struct = s_struct.best_trial
        combo_struct = t_struct.user_attrs.get('combo') or {}
        s_fine = study_finetune(analyzer, split, {}, n_trials=n_fine, mode=mode, invent_in_all=invent_in_all)
        t_fine = s_fine.best_trial
        combo_fine = t_fine.user_attrs.get('combo') or {}
        
        # Evaluate train/test using the fine-tuned combo
        prog = combo_fine.get('progressive')
        combo_params = {k: combo_fine.get(k) for k in ['eps','strict_band','refine','decisiveness_topm','kfold','weights']}
        tr = analyzer.evaluate_combo(split=split, part='train', progressive=prog, **combo_params)
        te = analyzer.evaluate_combo(split=split, part='test', progressive=prog, **combo_params)
        return {'structure_best': combo_struct, 'finetune_best': combo_fine, 'train_perf': tr, 'test_perf': te}
    else:
        base = {k: v for k, v in s_struct.best_params.items()}
        # ensure required keys exist
        base.setdefault('K1', 8); base.setdefault('K2', 4); base.setdefault('K3', 3)
        base.setdefault('decisiveness_tau', 0.1); base.setdefault('w_dec', 1.0); base.setdefault('w_cvar', 0.0); base.setdefault('w_imp', 0.0)
        s_fine = study_finetune(analyzer, split, base, n_trials=n_fine, mode=mode, invent_in_all=invent_in_all)
        best = s_fine.best_params
        # evaluate train/test
        tr = evaluate_params_on_split(analyzer, best, split, 'train', mode=mode, seed_weighting=True)
        te = evaluate_params_on_split(analyzer, best, split, 'test', mode=mode, seed_weighting=True)
        return {'structure_best': s_struct.best_params, 'finetune_best': best, 'train_perf': tr, 'test_perf': te}

# ---- Multi-objective utopia-distance pick -----------------------------------

def _pick_pareto_utopia(study):
    # assume two objectives: [win_rate (maximize), mean_gap (minimize)]
    pareto = []
    for t in study.best_trials:
        pareto.append({'params': t.params, 'win_rate': t.values[0], 'mean_gap': t.values[1]})
    if not pareto:
        return None
    wr = np.array([p['win_rate'] for p in pareto], float)
    gp = np.array([p['mean_gap'] for p in pareto], float)
    wr_n = (1.0 - wr) / max(1e-9, (1.0 - wr.min()))  # lower is better after 1-wr
    gp_n = (gp - gp.min()) / max(1e-9, (gp.max() - gp.min()))
    d = np.sqrt(wr_n**2 + gp_n**2)
    pick = int(np.argmin(d))
    return pareto[pick]

# ---- Invented combo search (discrete/continuous choices) ----------------------

def _sample_combo_only(trial: optuna.trial.Trial, max_layers_available: int,
                      eps_choices=(0.001, 0.0015, 0.002),
                      force_tail: bool = True):
    """
    Combo-only sampler: wide search over progressive combo structure only.
    No side knobs (strict_band, refine, weights) - pure combo exploration.
    """
    # ε: restricted, to avoid the winner being "just ε"
    eps = trial.suggest_categorical("eps", list(eps_choices))

    # length: explore
    n_steps = trial.suggest_int("n_metrics", 3, 6)

    # Build the full sequence via params (so TPE learns each position),
    # then enforce uniqueness deterministically.
    raw = [trial.suggest_categorical(f"metric_{i+1}", METRIC_POOL) for i in range(n_steps)]

    if force_tail:
        tail = ["margin_tail_cvar", "impostor_gap_cvar", "margin_skewkurt_safety"]
        tail = [m for m in tail if m in METRIC_POOL]
        head_len = max(n_steps - len(tail), 0)
        raw = raw[:head_len] + tail[:max(0, n_steps - head_len)]

    # Enforce uniqueness (first occurrence wins), fill remaining with first unused metrics.
    seen, metrics = set(), []
    for m in raw:
        if m not in seen:
            metrics.append(m); seen.add(m)
    if len(metrics) < n_steps:
        for m in METRIC_POOL:
            if m not in seen:
                metrics.append(m); seen.add(m)
                if len(metrics) == n_steps:
                    break

    # keep_k: strictly non-increasing using STATIC int range + clamp
    prev = int(max_layers_available)
    keeps = []
    for i, _ in enumerate(metrics, 1):
        raw = trial.suggest_int(f"keep_k_{i}", 1, int(max_layers_available))  # static bounds
        k = min(int(raw), prev)
        keeps.append(k)
        prev = k

    return {
        "eps": float(eps),
        "progressive": tuple({"metric": m, "keep_k": k} for m, k in zip(metrics, keeps)),
        "strict_band": None,
        "refine": "none",
        "decisiveness_topm": None,
        "kfold": None,
        "weights": None,
        # telemetry
        "n_metrics": int(n_steps),
        "metrics_sequence": tuple(metrics),
        "keep_ks": tuple(keeps),
    }

def _sample_focused_pipeline(trial: optuna.trial.Trial, max_layers_available: int):
    """
    Focused pipeline sampler that gravitates toward patterns that worked:
    - Tight ε (0.001-0.002)
    - Strong late tail metrics
    - Real monotone decreasing pruning
    - Tiny final tie-break
    """
    # --- eps: restrict to sweet spot ---
    eps = trial.suggest_categorical("eps", [0.001, 0.0015, 0.002])

    # --- refine / strict band / weights ---
    refine = trial.suggest_categorical("refine", ["decisiveness_topm", "none"])
    decisiveness_topm = trial.suggest_int("decisiveness_topm", 2, 3) if refine == "decisiveness_topm" else None

    use_strict_band = trial.suggest_categorical("use_strict_band", [False, False, False, True])  # 0.75 prob False
    strict_band = trial.suggest_float("strict_band", 5e-4, 1.5e-3) if use_strict_band else None

    use_weights = trial.suggest_categorical("use_weights", [False, False, True])  # mostly False
    if use_weights:
        w_last_metric = trial.suggest_float("w_last_metric", 0.4, 0.8)
        w_decis = trial.suggest_float("w_decis", 0.2, 0.6)
    else:
        w_last_metric = w_decis = None

    # --- fixed 5–6 stage scaffold ---
    n_metrics = trial.suggest_categorical("n_metrics", [5, 6])

    # Stage 1: broad filter
    m1 = trial.suggest_categorical("metric_1", ["local_intrinsic_dimensionality", "spearman_stability_accuracy"])
    
    # Stage 2: mid filter (bias toward CER@τ)
    m2 = trial.suggest_categorical("metric_2", ["confident_error_rate_tau", "confident_error_rate_tau", "confidence_entropy"])
    
    # Stage 3-5: fixed tail stack
    m3 = "margin_tail_cvar"
    m4 = "impostor_gap_cvar"
    m5 = "margin_skewkurt_safety"
    
    # Stage 6: optional polish
    m6 = trial.suggest_categorical("metric_6", ["ece_score", "temperature_estimate_strength"]) if n_metrics == 6 else None

    # --- keep_k monotone decreasing ---
    k1 = trial.suggest_int("keep_k_1", 24, 64)
    
    def next_k(name, prev, max_layers=64):
        raw = trial.suggest_int(name, 1, max_layers)  # static range
        return min(raw, prev)  # clamp to monotone
    
    k2 = next_k("keep_k_2", k1)
    k3 = next_k("keep_k_3", k2)
    k4 = next_k("keep_k_4", k3)
    k5 = next_k("keep_k_5", k4)
    k6 = next_k("keep_k_6", k5) if n_metrics == 6 else None

    metrics = [m1, m2, m3, m4, m5] + ([m6] if m6 else [])
    keeps = [k1, k2, k3, k4, k5] + ([k6] if k6 else [])

    # Build progressive stages for evaluate_combo
    progressive = []
    for i, (metric, keep_k) in enumerate(zip(metrics, keeps)):
        progressive.append({"metric": metric, "keep_k": int(keep_k)})

    return {
        "eps": float(eps),
        "progressive": tuple(tuple(d.items()) for d in progressive),  # immutable for storage
        "strict_band": None if strict_band is None else float(strict_band),
        "refine": str(refine),
        "decisiveness_topm": None if decisiveness_topm is None else int(decisiveness_topm),
        "kfold": None,  # not used in focused pipeline
        "weights": (w_last_metric, w_decis) if use_weights else None,
        # telemetry
        "n_metrics": int(n_metrics),
        "metrics_sequence": tuple(metrics),
        "keep_ks": tuple(keeps),
        "use_strict_band": use_strict_band,
        "use_weights": use_weights,
    }

def _sample_invented_combo(trial: optuna.trial.Trial, max_layers_available: int):
    # global banding epsilon
    eps = trial.suggest_categorical("eps", [0.0005, 0.001, 0.0015, 0.002, 0.005])

    # progressive metric schedule length
    n_steps = trial.suggest_int("n_metrics", 1, min(6, len(METRIC_POOL)))

    # 1) Sample EACH metric_i from a STATIC value space (full METRIC_POOL)
    picks = []
    for i in range(n_steps):
        picks.append(trial.suggest_categorical(f"metric_{i+1}", METRIC_POOL))

    # 2) Enforce uniqueness deterministically after sampling
    chosen = []
    seen = set()
    for m in picks:
        if m not in seen:
            chosen.append(m)
            seen.add(m)
    if len(chosen) < n_steps:
        for m in METRIC_POOL:
            if m not in seen:
                chosen.append(m)
                seen.add(m)
                if len(chosen) == n_steps:
                    break

    # Progressive keep_k's (static value space; enforce monotonicity after sampling)
    progressive = []
    prev_max = int(max(1, max_layers_available))
    for i, m_i in enumerate(chosen, start=1):
        # Suggest from a static upper bound, then clamp to previous maximum to maintain non-increasing sequence
        k_sample = trial.suggest_int(f"keep_k_{i}", 1, int(max_layers_available))
        k_i = int(min(int(k_sample), int(prev_max)))
        progressive.append({"metric": m_i, "keep_k": int(k_i)})
        prev_max = int(k_i)

    use_strict = trial.suggest_categorical("use_strict_band", [False, True])
    strict_band = trial.suggest_float("strict_band", 0.0005, 0.02, log=True) if use_strict else None

    refine = trial.suggest_categorical("refine", ["decisiveness_topm", "best_kfold", "none"])
    topm = trial.suggest_int("decisiveness_topm", 1, 5) if refine == "decisiveness_topm" else None
    kfold = trial.suggest_categorical("kfold", [3, 5]) if refine == "best_kfold" else None

    use_weights = trial.suggest_categorical("use_weights", [False, True])
    weights = None
    if use_weights:
        w_last = trial.suggest_float("w_last_metric", 0.0, 1.0)
        w_decis = trial.suggest_float("w_decis", 0.0, 1.0)
        weights = (w_last, w_decis)

    return {
        "eps": float(eps),
        "progressive": tuple(tuple(d.items()) for d in progressive),  # immutable for storage
        "strict_band": None if strict_band is None else float(strict_band),
        "refine": str(refine),
        "decisiveness_topm": None if topm is None else int(topm),
        "kfold": None if kfold is None else int(kfold),
        "weights": weights,
        # telemetry
        "n_metrics": int(n_steps),
        "metrics_sequence": tuple(chosen),
    }

def study_combo_only(analyzer, split, n_trials: int = 20000, seed: int = 1357,
                     objective: str = 'win_rate', force_tail: bool = True,
                     eps_choices=(0.001, 0.0015, 0.002), storage: str = None,
                     study_name: str = 'combo_only', output_dir: str = None):
    """
    Combo-only study: wide search over progressive combo structure only.
    No side knobs - pure combo exploration for 20k trials.
    """
    def objective_func(trial):
        max_layers = analyzer.num_layers(split, part='train') if hasattr(analyzer, 'num_layers') else 12
        combo = _sample_combo_only(trial, max_layers, eps_choices=eps_choices, force_tail=force_tail)
        
        # Evaluate on TRAIN only (no extras)
        prog = list(combo["progressive"])
        res = analyzer.evaluate_combo(
            split=split, part='train',
            eps=combo["eps"], progressive=prog,
            strict_band=None, refine="none",
            decisiveness_topm=None, kfold=None, weights=None
        )
        
        beats_rate = float(res.get("beats_rate", 0.0))
        mean_gap = float(res.get("mean_gap", 1.0))
        within_eps = float(res.get("within_epsilon", 0.0))
        robustness = float(res.get("robustness", 0.0))

        trial.set_user_attr("combo", combo)
        trial.set_user_attr("beats_rate", beats_rate)
        trial.set_user_attr("mean_gap", mean_gap)
        trial.set_user_attr("within_epsilon", within_eps)
        trial.set_user_attr("robustness", robustness)

        if objective == 'win_rate':
            return beats_rate
        elif objective == 'gap':
            return 1.0 - mean_gap
        elif objective == 'robustness':
            return robustness
        elif objective == 'within_epsilon':
            return within_eps
        else:
            return beats_rate

    # Static space → fine to keep multivariate TPE; silence the dynamic warning just in case
    sampler = optuna.samplers.TPESampler(
        seed=seed, multivariate=True, group=True,
        n_startup_trials=5000, warn_independent_sampling=False  # Higher startup for large studies
    )
    # No pruning since we don't call trial.report() with steps
    pruner = optuna.pruners.NopPruner()

    create_kwargs = dict(direction='maximize', sampler=sampler, pruner=pruner, study_name=study_name)
    if storage:
        create_kwargs.update(storage=storage, load_if_exists=True)

    study = optuna.create_study(**create_kwargs)
    
    # Periodic checkpoint callback
    def every(save_every=5000):
        def _cb(study, trial):
            if (trial.number + 1) % save_every == 0 and output_dir:
                try:
                    print(f"[combo_only] Checkpoint at trial {trial.number + 1}: saving top trials...")
                    from optuna_combo_search import save_top_trials
                    # Use fast_mode=True for checkpoints to avoid expensive JSON normalization with many trials
                    save_top_trials(study, output_dir, top_k=50, name=study_name, fast_mode=True)
                    print(f"[combo_only] Checkpoint saved successfully")
                except Exception as e:
                    print(f"[combo_only] Checkpoint save failed: {e}")
        return _cb
    
    callbacks = [every(5000)] if output_dir else []
    print(f"[combo_only] Starting optimization with {n_trials} trials...")
    study.optimize(objective_func, n_trials=n_trials, callbacks=callbacks, show_progress_bar=False)
    print(f"[combo_only] Optimization completed")
    
    # Param importances can hang with many categorical params; disable by default
    if os.getenv("ENABLE_OPTUNA_IMPORTANCES", "0") == "1":
        try:
            imps = get_param_importances(study)
            print(f"[combo_only_{objective}] param importances:", imps)
        except Exception:
            pass
    
    return study

def rescore_top_combos_robustness(study, analyzer, split, top_k=50, 
                                 eps_choices=(0.00075, 0.001, 0.0015, 0.002, 0.0025)):
    """
    Re-score top combos with different ε values for robustness checking.
    """
    import pandas as pd
    
    # Get top trials
    trials = sorted(study.trials, key=lambda t: t.value, reverse=True)[:top_k]
    
    results = []
    for i, trial in enumerate(trials):
        combo = trial.user_attrs.get('combo', {})
        if not combo:
            continue
            
        # Test with different ε values
        for eps in eps_choices:
            try:
                prog = list(combo.get('progressive', []))
                res = analyzer.evaluate_combo(
                    split=split, part='train',
                    eps=eps, progressive=prog,
                    strict_band=None, refine="none",
                    decisiveness_topm=None, kfold=None, weights=None
                )
                
                results.append({
                    'trial_rank': i + 1,
                    'trial_number': trial.number,
                    'original_value': trial.value,
                    'eps': eps,
                    'beats_rate': res.get('beats_rate', 0.0),
                    'mean_gap': res.get('mean_gap', 1.0),
                    'robustness': res.get('robustness', 0.0),
                    'within_epsilon': res.get('within_epsilon', 0.0),
                    'metrics_sequence': combo.get('metrics_sequence', ()),
                    'keep_ks': combo.get('keep_ks', ())
                })
            except Exception as e:
                print(f"Warning: Failed to evaluate trial {trial.number} with ε={eps}: {e}")
                continue
    
    return pd.DataFrame(results)

def study_focused_pipeline(analyzer, split, n_trials: int = 200, seed: int = 991, objective: str = 'win_rate'):
    """
    Focused pipeline study that gravitates toward patterns that worked:
    - Tight ε (0.001-0.002)
    - Strong late tail metrics
    - Real monotone decreasing pruning
    - Tiny final tie-break
    """
    def objective_func(trial):
        max_layers_available = analyzer.num_layers(split, part='train') if hasattr(analyzer, 'num_layers') else 12
        combo = _sample_focused_pipeline(trial, max_layers_available)
        
        try:
            # Re-hydrate the progressive list for readability and evaluation
            prog = []
            for stage in combo.get('progressive', ()):
                d = dict(stage)
                prog.append({"metric": d.get("metric"), "keep_k": int(d.get("keep_k", 1))})
            
            result = analyzer.evaluate_combo(
                split=split,
                eps=combo["eps"],
                progressive=prog,
                strict_band=combo.get("strict_band"),
                refine=combo["refine"],
                decisiveness_topm=combo.get("decisiveness_topm"),
                kfold=combo.get("kfold"),
                weights=combo.get("weights"),
                part='train'
            )
            
            beats_rate = float(result.get("beats_rate", 0.0))
            mean_gap = float(result.get("mean_gap", 1.0))
            robustness = float(result.get("robustness", 0.0))
            within_eps = float(result.get("within_epsilon", 0.0))
            
            # Pruning bonus: reward for real pruning
            keep_ks = combo.get("keep_ks", [])
            if len(keep_ks) >= 2:
                init_k = keep_ks[0]
                final_k = keep_ks[-1]
                shrink_ratio = final_k / max(1, init_k)
                prune_bonus = 0.03 * (1.0 - shrink_ratio)  # cap at ~3pp
            else:
                prune_bonus = 0.0
            
            # Store readable combo on the trial for exports/audits
            combo_readable = dict(combo)
            combo_readable['progressive'] = prog
            trial.set_user_attr("combo", combo_readable)
            trial.set_user_attr("prune_bonus", prune_bonus)
            trial.set_user_attr("shrink_ratio", shrink_ratio if len(keep_ks) >= 2 else 1.0)
            
            # Choose objective
            if objective == 'win_rate':
                base_score = beats_rate
            elif objective == 'gap':
                base_score = 1.0 - mean_gap  # convert to maximize
            elif objective == 'robustness':
                base_score = robustness
            elif objective == 'within_epsilon':
                base_score = within_eps
            else:
                base_score = beats_rate
            
            # Add pruning bonus
            final_score = base_score + prune_bonus
            
            # Store all metrics for analysis
            trial.set_user_attr('beats_rate', beats_rate)
            trial.set_user_attr('mean_gap', mean_gap)
            trial.set_user_attr('robustness', robustness)
            trial.set_user_attr('within_epsilon', within_eps)
            trial.set_user_attr('base_score', base_score)
            trial.set_user_attr('final_score', final_score)
            
            return final_score
            
        except Exception as e:
            # Invalid combo → worst scores
            trial.set_user_attr('error', str(e))
            return 0.0
    
    # Use TPE sampler for focused search
    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, n_startup_trials=40)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=20)
    
    study = optuna.create_study(
        direction='maximize', 
        sampler=sampler, 
        pruner=pruner, 
        study_name=f'focused_pipeline_{objective}'
    )
    
    study.optimize(objective_func, n_trials=n_trials, show_progress_bar=False)
    
    # Param importances can hang with many categorical params; disable by default
    if os.getenv("ENABLE_OPTUNA_IMPORTANCES", "0") == "1":
        try:
            imps = get_param_importances(study)
            print(f"[focused_pipeline_{objective}] param importances:", imps)
        except Exception:
            pass
    
    return study

def study_invented_combos(analyzer, split, n_trials: int = 400, seed: int = 990):
    def objective(trial):
        max_layers_available = analyzer.num_layers(split, part='train') if hasattr(analyzer, 'num_layers') else 12
        combo = _sample_invented_combo(trial, max_layers_available)
        try:
            # recover readable progressive list of dicts
            prog = []
            for stage in combo.get('progressive', ()):
                d = dict(stage)
                prog.append({"metric": d.get("metric"), "keep_k": int(d.get("keep_k", 1))})
            result = analyzer.evaluate_combo(
                split=split,
                eps=combo["eps"],
                progressive=prog,
                strict_band=combo.get("strict_band"),
                refine=combo["refine"],
                decisiveness_topm=combo.get("decisiveness_topm"),
                kfold=combo.get("kfold"),
                weights=combo.get("weights"),
                part='train'
            )
            beats_rate = float(result.get("beats_rate", 0.0))
            robustness = float(result.get("robustness", 0.0))
            within_eps = float(result.get("within_epsilon", 0.0))
            neg_mean_gap = float(-result.get("mean_gap", 1.0))
        except Exception:
            # invalid combo → worst scores
            beats_rate, robustness, within_eps, neg_mean_gap = 0.0, 0.0, 0.0, -1.0
        # store readable progressive too for auditability
        combo_readable = dict(combo)
        combo_readable['progressive'] = prog
        trial.set_user_attr("combo", combo_readable)
        # 4-objective: prioritize beats_rate while keeping original metrics
        return beats_rate, robustness, within_eps, neg_mean_gap

    sampler = optuna.samplers.TPESampler(seed=seed, multivariate=True, group=True, n_startup_trials=40)
    # Multi-objective pruning often provides little benefit; use NopPruner to avoid warnings
    pruner = optuna.pruners.NopPruner()
    study = optuna.create_study(directions=["maximize","maximize","maximize","maximize"], sampler=sampler, pruner=pruner, study_name='invented_combo')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    # Print a utopia-distance pick (weights emphasize beats_rate)
    try:
        pick = _pick_utopia_multi(study, weights=[3.0, 1.0, 1.0, 1.0])
        if pick and pick.get('trial') is not None:
            t = pick['trial']
            combo = t.user_attrs.get('combo')
            print(f"[invented_combo] utopia pick: trial #{t.number}")
            print(f"  values = {t.values}  (order: beats_rate, robustness, within_eps, -mean_gap)")
            print(f"  combo  = {combo}")
        else:
            print("[invented_combo] No utopia pick available (no completed trials).")
    except Exception:
        pass
    return study

def _pick_utopia_multi(study: optuna.Study, weights=None):
    """
    Generic utopia-distance pick for M objectives with directions.
    Returns dict with {'trial': FrozenTrial, 'values': [...]} or None.
    Optionally weight objectives via 'weights' (same length as #objectives).
    """
    try:
        dirs = study.directions
    except Exception:
        return None
    if not dirs:
        return None
    trials = [t for t in study.trials if t.values is not None and all(v is not None for v in t.values)]
    if not trials:
        return None
    V = np.array([t.values for t in trials], dtype=float)
    # Normalize per direction to [0,1] with 0 near utopia for each objective
    normed = []
    for j, d in enumerate(dirs):
        col = V[:, j]
        mx = np.nanmax(col); mn = np.nanmin(col)
        if d == StudyDirection.MAXIMIZE:
            n = (1.0 - (col - mn) / max(1e-12, (mx - mn)))
        else:
            n = (col - mn) / max(1e-12, (mx - mn))
        normed.append(n)
    N = np.vstack(normed).T
    if weights is not None:
        w = np.asarray(weights, dtype=float)
        if w.size != N.shape[1]:
            w = np.ones(N.shape[1], float)
        # normalize weights to unit norm for scale invariance
        w = w / max(1e-12, np.linalg.norm(w))
        dist = np.sqrt(((N * w)**2).sum(axis=1))
    else:
        dist = np.sqrt((N**2).sum(axis=1))
    k = int(np.argmin(dist))
    return {'trial': trials[k], 'values': trials[k].values}

# ---- Dry-run baselines & helpers --------------------------------------------

DRY_BASELINES = [
    {'name': 'combo_margin3_then_decisiveness', 'order': 'K→D→CVAR→IMP', 'eps': 1e-3, 'delta': 0.0, 'K1': 8, 'K2': 4, 'K3': 3, 'decisiveness_tau': 0.1, 'w_dec': 1.0, 'w_cvar': 0.0, 'w_imp': 0.0},
    {'name': 'combo_cvar5_decisiveness3_bestkfold', 'order': 'K→D→CVAR→IMP', 'eps': 2e-3, 'delta': 0.0, 'K1': 8, 'K2': 5, 'K3': 3, 'decisiveness_tau': 0.1, 'w_dec': 0.7, 'w_cvar': 0.75, 'w_imp': 0.0},
    {'name': 'borda_marg_decis_bound', 'order': 'Borda', 'eps': 1e-3, 'delta': 0.0, 'K1': 8, 'K2': 4, 'K3': 3, 'decisiveness_tau': None, 'w_dec': 0.5, 'w_cvar': 0.75, 'w_imp': 0.2},
]

def dry_run_check(analyzer, split, mode='macro'):
    rows = []
    for cfg in DRY_BASELINES:
        p = {k: v for k, v in cfg.items() if k != 'name'}
        tr = evaluate_params_on_split(analyzer, p, split, 'train', mode=mode, seed_weighting=True)
        te = evaluate_params_on_split(analyzer, p, split, 'test', mode=mode, seed_weighting=True)
        rows.append({'name': cfg['name'], 'train_win': tr['win_rate'], 'train_gap': tr['mean_gap'],
                     'test_win': te['win_rate'], 'test_gap': te['mean_gap']})
    return rows

def repeat_structure_runs(analyzer, split, seeds=(111, 222, 333), n_trials=150, focused=True, mode='macro'):
    out = []
    for s in seeds:
        sampler = optuna.samplers.TPESampler(seed=s)
        study = optuna.create_study(direction='maximize', sampler=sampler, study_name=f'structure_{s}')
        def objective(trial):
            p = make_search_space(trial, focused=focused)
            K1 = trial.suggest_int('K1', 3, 100, step=1)
            K2_raw = trial.suggest_int('K2_raw', 2, 100, step=1)
            K3_raw = trial.suggest_int('K3_raw', 1, 100, step=1)
            p['K1'] = int(K1)
            p['K2'] = int(min(K2_raw, K1))
            p['K3'] = int(min(K3_raw, p['K2']))
            perf = evaluate_params_on_split(analyzer, p, split, 'train', mode=mode, seed_weighting=True)
            return perf['win_rate']
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        out.append({'seed': s, 'best_value': study.best_value, 'best_params': study.best_params})
    return out

def summarize_study(study):
    try:
        print(f"[{study.study_name}] best value: {study.best_value:.4f}")
    except Exception:
        pass
    taus = [t.user_attrs.get('tau_mean') for t in study.trials if 'tau_mean' in t.user_attrs]
    bands = [t.user_attrs.get('mean_band_size') for t in study.trials if 'mean_band_size' in t.user_attrs]
    if taus:
        try:
            print(f"  diag Kendall tau (mean over trials): {np.nanmean(taus):.3f}")
        except Exception:
            pass
    if bands:
        try:
            print(f"  mean band size (over trials): {np.nanmean(bands):.2f}")
        except Exception:
            pass

# ---- Public entry ------------------------------------------------------------

def run_all_optuna(analyzer,
                   split,
                   n_winrate: int = 80,
                   n_gap: int = 60,
                   n_multi: int = 100,
                   n_combo: int = 0,
                   n_focused: int = 0,
                   focused_objective: str = 'win_rate',
                   n_combo_only: int = 0,
                   combo_only_objective: str = 'win_rate',
                   combo_only_force_tail: bool = True,
                   export_dir: Optional[str] = None,
                   invent_in_all: bool = False) -> Dict[str, Any]:
    """
    Returns {
      'winrate': {'best_params':..., 'train_perf':..., 'test_perf':...},
      'gap': {...},
      'multi': {'pareto': [...], 'pick': {...}}
    }
    """
    out = {}

    # Study 1: win-rate
    if n_winrate and n_winrate > 0:
        s1 = study_winrate(analyzer, split, n_trials=n_winrate, mode='macro', invent_in_all=invent_in_all)
        if invent_in_all:
            # Best trial uses invented combos → evaluate train/test via its stored combo
            t = s1.best_trial
            combo = t.user_attrs.get('combo') or {}
            prog = combo.get('progressive')
            combo_params = {k: combo.get(k) for k in ['eps','strict_band','refine','decisiveness_topm','kfold','weights']}
            tr1 = analyzer.evaluate_combo(split=split, part='train', progressive=prog, **combo_params)
            te1 = analyzer.evaluate_combo(split=split, part='test', progressive=prog, **combo_params)
            out['winrate'] = {'best_combo': combo, 'train_perf': tr1, 'test_perf': te1}
        else:
            p1 = s1.best_params
            # derive effective K2/K3 if only raw variants exist
            try:
                k1 = int(p1.get('K1'))
                k2 = int(min(p1.get('K2_raw', k1), k1)) if 'K2' not in p1 else int(p1['K2'])
                k3 = int(min(p1.get('K3_raw', k2), k2)) if 'K3' not in p1 else int(p1['K3'])
            except Exception:
                k1, k2, k3 = p1.get('K1'), p1.get('K2'), p1.get('K3')
            p1_eff = dict(p1); p1_eff['K1']=k1; p1_eff['K2']=k2; p1_eff['K3']=k3
            tr1 = evaluate_params_on_split(analyzer, p1_eff, split, 'train', mode='macro', seed_weighting=True)
            te1 = evaluate_params_on_split(analyzer, p1_eff, split, 'test', mode='macro', seed_weighting=True)
            out['winrate'] = {'best_params': p1_eff, 'train_perf': tr1, 'test_perf': te1}
        if export_dir:
            paths = save_top_trials(s1, export_dir, top_k=10, name='winrate')
            print("[winrate] saved:", paths)

    # Study 2: gap
    if n_gap and n_gap > 0:
        s2 = study_gap(analyzer, split, n_trials=n_gap, mode='macro', invent_in_all=invent_in_all)
        if invent_in_all:
            # Best trial uses invented combos → evaluate train/test via its stored combo
            t = s2.best_trial
            combo = t.user_attrs.get('combo') or {}
            prog = combo.get('progressive')
            combo_params = {k: combo.get(k) for k in ['eps','strict_band','refine','decisiveness_topm','kfold','weights']}
            tr2 = analyzer.evaluate_combo(split=split, part='train', progressive=prog, **combo_params)
            te2 = analyzer.evaluate_combo(split=split, part='test', progressive=prog, **combo_params)
            out['gap'] = {'best_combo': combo, 'train_perf': tr2, 'test_perf': te2}
        else:
            p2 = s2.best_params
            try:
                k1 = int(p2.get('K1'))
                k2 = int(min(p2.get('K2_raw', k1), k1)) if 'K2' not in p2 else int(p2['K2'])
                k3 = int(min(p2.get('K3_raw', k2), k2)) if 'K3' not in p2 else int(p2['K3'])
            except Exception:
                k1, k2, k3 = p2.get('K1'), p2.get('K2'), p2.get('K3')
            p2_eff = dict(p2); p2_eff['K1']=k1; p2_eff['K2']=k2; p2_eff['K3']=k3
            tr2 = evaluate_params_on_split(analyzer, p2_eff, split, 'train', mode='macro', seed_weighting=True)
            te2 = evaluate_params_on_split(analyzer, p2_eff, split, 'test', mode='macro', seed_weighting=True)
            out['gap'] = {'best_params': p2_eff, 'train_perf': tr2, 'test_perf': te2}
        if export_dir:
            paths = save_top_trials(s2, export_dir, top_k=10, name='gap')
            print("[gap] saved:", paths)

    # Study 3: multi (utopia-distance pick)
    if n_multi and n_multi > 0:
        s3 = study_multi(analyzer, split, n_trials=n_multi, mode='macro', invent_in_all=invent_in_all)
        if invent_in_all:
            # For invented combos, pick by utopia distance across objectives
            pick = _pick_utopia_multi(s3, weights=[1.0, 1.0])  # equal weight for beats_rate and mean_gap
            if pick and pick.get('trial') is not None:
                t = pick['trial']
                combo = t.user_attrs.get('combo') or {}
                prog = combo.get('progressive')
                combo_params = {k: combo.get(k) for k in ['eps','strict_band','refine','decisiveness_topm','kfold','weights']}
                tr3 = analyzer.evaluate_combo(split=split, part='train', progressive=prog, **combo_params)
                te3 = analyzer.evaluate_combo(split=split, part='test', progressive=prog, **combo_params)
                out['multi'] = {'pareto_sample': {'combo': combo, 'values': pick['values']}, 'train_perf': tr3, 'test_perf': te3}
            else:
                out['multi'] = {'pareto_sample': None}
        else:
            chosen = _pick_pareto_utopia(s3)
            if chosen:
                p3 = dict(chosen['params'])
                try:
                    k1 = int(p3.get('K1'))
                    k2 = int(min(p3.get('K2_raw', k1), k1)) if 'K2' not in p3 else int(p3['K2'])
                    k3 = int(min(p3.get('K3_raw', k2), k2)) if 'K3' not in p3 else int(p3['K3'])
                except Exception:
                    k1, k2, k3 = p3.get('K1'), p3.get('K2'), p3.get('K3')
                p3['K1']=k1; p3['K2']=k2; p3['K3']=k3
                tr3 = evaluate_params_on_split(analyzer, p3, split, 'train', mode='macro', seed_weighting=True)
                te3 = evaluate_params_on_split(analyzer, p3, split, 'test', mode='macro', seed_weighting=True)
                out['multi'] = {'pareto_sample': {'params': p3, 'win_rate': chosen.get('win_rate'), 'mean_gap': chosen.get('mean_gap')}, 'train_perf': tr3, 'test_perf': te3}
            else:
                out['multi'] = {'pareto_sample': None}
        if export_dir:
            paths = save_top_trials(s3, export_dir, top_k=10, name='multi')
            print("[multi] saved:", paths)

    # Study 4: invented combos (3-objective)
    if n_combo and n_combo > 0:
        s4 = study_invented_combos(analyzer, split, n_trials=n_combo, seed=8801)
        # Pick by weighted utopia distance across all 4 objectives (prioritize beats_rate)
        pick = _pick_utopia_multi(s4, weights=[3.0, 1.0, 1.0, 1.0])
        if pick and pick.get('trial') is not None:
            try:
                print(f"[invented_combo] utopia pick: trial #{pick['trial'].number} values={pick['values']} (beats_rate, robustness, within_eps, -mean_gap)")
            except Exception:
                pass
            params = dict(pick['trial'].user_attrs.get('combo') or {})
            # Rebuild progressive list for readability if stored as tuples
            prog = []
            for stage in params.get('progressive', ()):
                d = dict(stage)
                prog.append({"metric": d.get("metric"), "keep_k": int(d.get("keep_k", 1))})
            # Evaluate on train/test using the same progressive combo schema
            tr4 = analyzer.evaluate_combo(
                split=split,
                eps=params.get('eps'),
                progressive=prog,
                strict_band=params.get('strict_band'),
                refine=params.get('refine'),
                decisiveness_topm=params.get('decisiveness_topm'),
                kfold=params.get('kfold'),
                weights=params.get('weights'),
                part='train')
            te4 = analyzer.evaluate_combo(
                split=split,
                eps=params.get('eps'),
                progressive=prog,
                strict_band=params.get('strict_band'),
                refine=params.get('refine'),
                decisiveness_topm=params.get('decisiveness_topm'),
                kfold=params.get('kfold'),
                weights=params.get('weights'),
                part='test')
            out['invented_combo'] = {'best_combo': params, 'train_perf': tr4, 'test_perf': te4}
        else:
            out['invented_combo'] = {'best_combo': None}
        if export_dir:
            paths = save_top_trials(s4, export_dir, top_k=10, name='invented_combo')
            print("[invented_combo] saved:", paths)
            try:
                pareto = export_pareto_trials(s4, export_dir, name='invented_combo_pareto')
                print("[invented_combo] pareto exported:", pareto)
            except Exception:
                pass

    # Study 5: focused pipeline (new)
    if n_focused and n_focused > 0:
        s5 = study_focused_pipeline(analyzer, split, n_trials=n_focused, objective=focused_objective)
        # Best trial uses focused pipeline → evaluate train/test via its stored combo
        t = s5.best_trial
        combo = t.user_attrs.get('combo') or {}
        prog = combo.get('progressive')
        combo_params = {k: combo.get(k) for k in ['eps','strict_band','refine','decisiveness_topm','kfold','weights']}
        tr5 = analyzer.evaluate_combo(split=split, part='train', progressive=prog, **combo_params)
        te5 = analyzer.evaluate_combo(split=split, part='test', progressive=prog, **combo_params)
        out['focused_pipeline'] = {'best_combo': combo, 'train_perf': tr5, 'test_perf': te5}
        if export_dir:
            paths = save_top_trials(s5, export_dir, top_k=10, name=f'focused_pipeline_{focused_objective}')
            print(f"[focused_pipeline_{focused_objective}] saved:", paths)

    # Study 6: combo-only (new)
    if n_combo_only and n_combo_only > 0:
        s6 = study_combo_only(
            analyzer=analyzer, 
            split=split, 
            n_trials=n_combo_only, 
            objective=combo_only_objective,
            force_tail=combo_only_force_tail,
            seed=1357,
            storage=None,  # Could add storage parameter to run_all_optuna if needed
            study_name=f'combo_only_{combo_only_objective}',
            output_dir=export_dir
        )
        # Best trial uses combo-only → evaluate train/test via its stored combo
        t = s6.best_trial
        combo = t.user_attrs.get('combo') or {}
        prog = list(combo.get('progressive', []))
        tr6 = analyzer.evaluate_combo(split=split, part='train', progressive=prog,
                                     eps=combo["eps"], strict_band=None, refine="none",
                                     decisiveness_topm=None, kfold=None, weights=None)
        te6 = analyzer.evaluate_combo(split=split, part='test', progressive=prog,
                                     eps=combo["eps"], strict_band=None, refine="none",
                                     decisiveness_topm=None, kfold=None, weights=None)
        out['combo_only'] = {'best_combo': combo, 'train_perf': tr6, 'test_perf': te6}
        if export_dir:
            paths = save_top_trials(s6, export_dir, top_k=10, name=f'combo_only_{combo_only_objective}')
            print(f"[combo_only_{combo_only_objective}] saved:", paths)

    return out


