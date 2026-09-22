"""Properties of the residual-evidence study (docs/residual_evidence_study_spec.md)."""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Calibrators import residual_readout as RR  # noqa: E402
from Experiments import residual_evidence_study as S  # noqa: E402

K = RR.K


def _toy(n=600, seed=0, signal=0.0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    z = rng.normal(size=(n, K)); z[np.arange(n), y] += 2.0
    return z, y


def test_frozen_constants():
    assert RR.LAMBDAS == (1e-4, 1e-2, 1.0, 100.0) and RR.NLL_TOLERANCE == 0.01 and RR.K == 100
    assert S.K_C == 5 and S.LAYER == "layer3.22" and S.K_DAC == 200 and S.INNER_SEED == 123
    assert S.NATIVE_LAYERS == ("conv1", "layer1", "layer2", "layer3", "layer4")  # five-source variant, no logits
    assert RR.HIDDEN_FAMILIES == ("G", "S", "DG", "DS") and RR.OUTPUT_FAMILIES == ("O", "DL")


def test_random_maps_are_deterministic_and_data_independent():
    a, b = RR.gaussian_map(1024, 100, RR.SEED_MAP_G), RR.gaussian_map(1024, 100, RR.SEED_MAP_G)
    assert np.array_equal(a, b) and a.shape == (100, 1024)
    assert not np.array_equal(a[:, :100], RR.gaussian_map(4096, 100, RR.SEED_MAP_S)[:, :100])
    x = np.random.default_rng(1).normal(size=(50, 1024))
    ratio = np.linalg.norm(x @ a.T, axis=1) / np.linalg.norm(x, axis=1)
    assert 0.7 < ratio.mean() < 1.3  # norm-preserving in expectation (N(0,1/k) entries)
    B, c = RR.logit_nonlinear_map(RR.SEED_MAP_O)
    assert B.shape == (K, K) and c.shape == (K,) and abs(B.std() - 0.1) < 0.01


def test_all_feature_maps_give_100_standardized_dims_using_fit_statistics_only():
    rng = np.random.default_rng(0)
    raw = {"G": rng.normal(size=(300, 1024)), "S": rng.normal(size=(300, 4096)), "DG": rng.uniform(.1, .6, (300, K)),
           "DS": rng.uniform(.1, .6, (300, K)), "DL": rng.uniform(.1, .6, (300, K)), "O": rng.normal(size=(300, K))}
    for fam, x in raw.items():
        fm = RR.make_feature_map(fam, x)
        phi = fm(x)
        assert phi.shape == (300, K), fam
        np.testing.assert_allclose(phi.mean(0), 0, atol=1e-9)
        assert np.all(np.abs(phi.std(0) - 1) < 1e-3) or fam == "O"
        new = fm(rng.normal(size=(5, x.shape[1])) if fam in ("G", "S", "O") else rng.uniform(.1, .6, (5, K)))
        assert new.shape == (5, K)  # applies fit-only statistics to new rows without refitting


def test_zero_candidate_is_the_anchor_exactly_and_class_offset_removed():
    z, y = _toy()
    phi = np.random.default_rng(3).normal(size=(len(y), K))
    f = RR.fit_residual(z, phi, y, 1.0)
    assert f["converged"]
    np.testing.assert_allclose(f["W"].mean(0), 0.0, atol=1e-12)  # class rows centered
    c = RR.correction_logits(phi, f["W"])
    np.testing.assert_allclose(c.mean(1), 0.0, atol=1e-9)
    # W = 0 gives the anchor exactly
    np.testing.assert_array_equal(z + RR.correction_logits(phi, np.zeros((K, K))), z)


def test_residual_fit_uses_the_stated_penalty_and_strong_lambda_shrinks_to_zero():
    z, y = _toy(400)
    phi = np.random.default_rng(4).normal(size=(400, K))
    w_small = RR.fit_residual(z, phi, y, 1e-4)["W"]
    w_big = RR.fit_residual(z, phi, y, 100.0)["W"]
    assert np.linalg.norm(w_big) < 0.1 * np.linalg.norm(w_small)
    assert np.linalg.norm(w_big) < 0.5


def test_matrix_scaling_starts_at_identity_and_large_lambda_returns_identity():
    z, y = _toy(300)
    f = RR.fit_matrix_scaling(z, y, 1e6)
    np.testing.assert_allclose(f["A"], np.eye(K), atol=1e-3)
    assert f["converged"]


def test_logit_gauge_invariance_of_the_readout():
    z, y = _toy(200)
    phi = np.random.default_rng(5).normal(size=(200, K))
    W = RR.fit_residual(z, phi, y, 1.0)["W"]
    p1 = RR.softmax_np(z + RR.correction_logits(phi, W))
    p2 = RR.softmax_np(z + 7.3 + RR.correction_logits(phi, W))
    np.testing.assert_allclose(p1, p2, atol=1e-12)


def _cand(family, lam, nll, acc):
    return {"family": family, "lambda": lam, "select_nll": nll, "select_acc": acc}


def test_decision_policy_feasibility_ties_and_anchor_always_available():
    anchor_nll = 1.0
    zero = RR.zero_candidate(anchor_nll, 0.70)
    pool = [_cand("G", 1e-4, 1.005, 0.72), _cand("S", 1e-4, 0.99, 0.715), _cand("DG", 1.0, 1.02, 0.80), zero]
    i = RR.decision_policy(pool, anchor_nll)
    assert pool[i]["family"] == "G"  # DG is infeasible (NLL +0.02 > 0.01) even with the highest accuracy
    # tie on accuracy: lower NLL wins, then larger lambda, then family order
    pool = [_cand("G", 1e-4, 1.004, 0.72), _cand("S", 1e-4, 1.002, 0.72), zero]
    assert pool[RR.decision_policy(pool, anchor_nll)]["family"] == "S"
    pool = [_cand("G", 1e-2, 1.002, 0.72), _cand("G", 1.0, 1.002, 0.72), zero]
    assert pool[RR.decision_policy(pool, anchor_nll)]["lambda"] == 1.0
    pool = [_cand("S", 1.0, 1.002, 0.72), _cand("G", 1.0, 1.002, 0.72), zero]
    assert pool[RR.decision_policy(pool, anchor_nll)]["family"] == "G"
    # nothing feasible beats the anchor
    pool = [_cand("G", 1e-4, 1.5, 0.9), zero]
    assert pool[RR.decision_policy(pool, anchor_nll)]["family"] == "zero"
    # NLL policy: lowest NLL, may pick an arm the Decision policy would not
    pool = [_cand("G", 1e-4, 0.95, 0.70), _cand("S", 1e-4, 1.0, 0.75), zero]
    assert pool[RR.nll_policy(pool)]["family"] == "G"


def test_bin_utility_signed_identity():
    rng = np.random.default_rng(0)
    n = 500
    y = rng.integers(0, 5, n); base = rng.integers(0, 5, n); pred = np.where(rng.random(n) < .2, rng.integers(0, 5, n), base)
    v = rng.normal(size=n); edges = RR.quantile_edges(v)
    bins = RR.bin_utility(v, edges, base, pred, y)
    assert sum(b["n"] for b in bins) == n
    net = sum(b["signed_utility"] * b["n"] for b in bins)
    assert abs(net - (np.sum(pred == y) - np.sum(base == y))) < 1e-9


def test_evaluation_path_cannot_reach_fitting_and_corruption_loaded_after_state_written():
    for fn in ("evaluate_cell", "eval_logits", "per_sample"):
        tree = ast.parse(inspect.getsource(getattr(S, fn)))
        called = {n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
                  for n in ast.walk(tree) if isinstance(n, ast.Call)}
        bad = {c for c in called if c.startswith("fit") or c in {"step", "minimize", "make_feature_map", "select_anchor"}}
        assert not bad, (fn, bad)
    src = inspect.getsource(S.main)
    assert src.index('"frozen_state.json"') < src.index("load_corrupted(")
    assert src.index("fit_phase(") < src.index("load_corrupted(")


def test_condition_lists_are_frozen():
    dev = S.cells_for("dev12")
    assert len(dev) == 13 and dev[0][0] == "clean"
    assert {c for _, c, _ in dev[1:]} == {"gaussian_noise", "defocus_blur", "fog", "jpeg_compression"}
    assert {s for _, _, s in dev[1:]} == {1, 3, 5}
    full = S.cells_for("full75")
    assert len(full) == 76 and len({c for _, c, _ in full[1:]}) == 15


def test_jl_sufficient_dimension_is_reported_not_used_as_a_guarantee():
    assert RR.jl_required_dim(45000, 0.2) == 2473 and RR.jl_required_dim(45000, 0.5) == 515
