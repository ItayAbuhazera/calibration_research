"""Failure-risk tests for the fixed-gate study (engineering checks, not scientific validation)."""
import inspect
import numpy as np
from scipy import optimize

from atlas import fg_gate as FG, gate, stats


def _toy(n=400, c=12, seed=0):
    r = np.random.default_rng(seed)
    z = r.normal(size=(n, c)) * 2
    y = r.integers(0, c, n)
    i = z.argmax(1); j = (i + r.integers(1, 4, n)) % c
    pg = np.abs(r.normal(size=(n, c))) + 0.01; pg /= pg.sum(1, keepdims=True)
    return z, y, i, j, pg, r.uniform(0.5, 1.5, n), r.uniform(0.5, 3, n)


def test_feature_shapes_and_no_label_argument():
    z, y, i, j, pg, kth, qn = _toy(c=100)
    shapes = {f: FG.features(f, z, i, j, pg, kth, qn).shape[1] for f in FG.FAMILIES}
    assert shapes == {"C0": 5, "C1": 9, "Z0": 105, "Z1": 109}
    assert "y" not in inspect.signature(FG.features).parameters


def test_ridge_matches_brute_force_objective():
    z, y, i, j, pg, kth, qn = _toy(n=300)
    j = np.where(j == i, (j + 1) % 12, j)
    X = FG.features("C1", z, i, j, pg, kth, qn)
    fit = FG.fit_family(X, i, j, y, lambdas=(1.0,))
    m = fit["models"][1.0]
    D = gate.utility_target(i, j, y); dis = j != i
    Xs = (X[dis] - m["mean"]) / m["std"]
    obj = lambda p: np.mean((D[dis] - p[0] - Xs @ p[1:]) ** 2) + 1.0 * np.sum(p[1:] ** 2)
    r = optimize.minimize(obj, np.zeros(1 + Xs.shape[1]), method="BFGS", options={"gtol": 1e-10})
    assert np.allclose(r.x[1:], m["w"], atol=1e-4) and abs(r.x[0] - m["b"]) < 1e-6


def test_order_is_nested_class_stratified():
    y = np.repeat(np.arange(100), 25)
    o = FG.stratified_order(y, 5)
    assert sorted(o) == list(range(2500))
    for n in (625, 1250):
        cnt = np.bincount(y[o[:n]], minlength=100)
        assert cnt.max() - cnt.min() <= 1
    assert set(o[:625]) < set(o[:1250])
    assert not np.array_equal(o, FG.stratified_order(y, 6))


def test_select_never_when_gates_harm_and_exact_zero_intervention():
    z, y, i, j, pg, kth, qn = _toy(n=600)
    y = i.copy()                      # base always right => every intervention is a harm (D = -1)
    X = FG.features("Z0", z, i, j, pg, kth, qn)
    fit = FG.fit_family(X, i, j, y)
    best, table = FG.select_config(fit, X, i, j, y)
    assert best["kind"] == "never" and best["net"] == 0.0 and best["interventions"] == 0
    g, sc = FG.apply_config(best, None, X, i, j)
    assert g.sum() == 0
    q0 = stats.softmax(z); q = gate.build_q(q0, pg, g)
    assert np.array_equal(q, q0)


def test_g_zero_when_no_disagreement_and_wh_identity_and_argmax():
    z, y, i, j, pg, kth, qn = _toy(n=500)
    j2 = j.copy(); j2[:100] = i[:100]
    y = np.where(np.arange(500) % 3 == 0, j2, y)
    X = FG.features("C0", z, i, j2, pg, kth, qn)
    fit = FG.fit_family(X, i, j2, y)
    cfg = {"kind": "gate", "lambda": 0.01, "theta": -2.0}          # threshold below every score => all disagreements pass
    g, _ = FG.apply_config(cfg, fit["models"][0.01], X, i, j2)
    assert not g[:100].any() and g[100:].all()
    pg2 = pg.copy(); pg2[np.arange(500), j2] += 5.0               # make argmax(pg) = j2
    q = gate.build_q(stats.softmax(z), pg2, g)
    assert np.array_equal(q.argmax(1), np.where(g, j2, i))
    qT = gate.apply_temperature(q, 1.7)
    assert np.array_equal(qT.argmax(1), q.argmax(1))
    r = stats.wh_stats(i, q.argmax(1), y)
    assert abs(r["net_utility"] - ((q.argmax(1) == y).mean() - (i == y).mean())) < 1e-12


def test_degenerate_small_disagreement():
    z, y, i, j, pg, kth, qn = _toy(n=50)
    fit = FG.fit_family(FG.features("C0", z, i, i, pg, kth, qn), i, i, y)   # j == i everywhere
    assert fit["degenerate"] and fit["models"] == {}
