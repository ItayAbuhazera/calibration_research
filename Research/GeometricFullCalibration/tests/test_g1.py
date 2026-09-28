"""Structural tests for G1 (docs/g1_conditional_access_spec.md). No scientific outputs are read."""
import numpy as np

from atlas import g1_fit, g1_rules
from atlas.followup_anchored import INF


def test_kernel_projector_properties_and_equivalent_span():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((10, 40)); b = rng.standard_normal(10)
    Pk = g1_fit.kernel_projector(W)
    assert np.allclose(W @ Pk, 0, atol=1e-10)
    assert np.allclose(Pk @ Pk, Pk, atol=1e-10)
    h = rng.standard_normal((200, 40)); z = h @ W.T + b
    A = np.c_[z, h, np.ones(200)]; Bm = np.c_[z, h @ Pk.T, np.ones(200)]
    # (z, h) and (z, P_ker h) span the same affine column space
    assert np.linalg.matrix_rank(np.c_[A, Bm], tol=1e-8) == np.linalg.matrix_rank(A, tol=1e-8) == np.linalg.matrix_rank(Bm, tol=1e-8)


def _path(curve):
    return g1_fit.path_with_edges(lambda lam: {"lambda": lam, "val_nll": curve(lam), "converged": True, "retried": False}, curve(INF))


def test_grid_edge_rule_interior():
    table, sel, edge = _path(lambda l: 1.0 if l == INF else (np.log10(l) + 3) ** 2)
    assert sel == 1e-3 and edge == "interior_or_f0" and len(table) == 6


def test_grid_edge_rule_strong_edge_extends_and_resolves():
    # minimum at lambda = 1 (stronger than the grid): extension to 1e0 is selected, 1e1 is worse -> resolved
    table, sel, edge = _path(lambda l: 5.0 if l == INF else (np.log10(l) - 0) ** 2)
    assert sel == 1e0 and edge == "extended_resolved" and {1e0, 1e1} <= {t["lambda"] for t in table}


def test_grid_edge_rule_unresolved_strong_and_weak():
    _, sel, edge = _path(lambda l: 9.0 if l == INF else -np.log10(l))       # wants ever stronger ridge
    assert sel == 1e1 and edge == "unresolved_edge"
    _, sel, edge = _path(lambda l: 9.0 if l == INF else np.log10(l))        # wants ever weaker ridge
    assert sel == 1e-7 and edge == "unresolved_edge"


def test_grid_edge_rule_f0_selected():
    _, sel, edge = _path(lambda l: 0.0 if l == INF else 1.0)
    assert sel == INF and edge == "interior_or_f0"


def _q(**kw):
    q = dict(dcond8=0.0, dcond8_ci=(-0.1, 0.1), dcond25=0.0, dcond25_ci=(-0.1, 0.1), dE=0.0, dE_ci=(-0.1, 0.1), dDF=-2.0, dDF_ci=(-2.3, -1.7),
             s=0.2, nest=0.0, budget_drop=0.0, budget_drop_ci=(-0.1, 0.1), catchup=0.0, catchup_ci=(-0.1, 0.1), accB=55.0, accC=55.0, accG=55.2)
    q.update(kw); return q


V = {"v_extract": True, "v_converged": True, "v_edge": True, "shuf": 0.0}


def test_rules_outcomes():
    neg = _q()
    assert g1_rules.decide({2: neg, 4: neg}, {2: V, 4: V})["outcome"] == "B"
    acc = _q(dcond8=1.2, dcond8_ci=(0.9, 1.5), dE=1.1, dE_ci=(0.8, 1.4), dDF=1.0, dDF_ci=(0.6, 1.4))
    assert g1_rules.decide({2: acc, 4: acc}, {2: V, 4: V})["outcome"] == "A"
    div = dict(acc, dDF=-1.0, dDF_ci=(-1.4, -0.6))
    assert g1_rules.decide({2: div, 4: div}, {2: V, 4: V})["outcome"] == "D"
    cap_e = dict(acc, dE=0.2, dE_ci=(-0.1, 0.5))                 # redundant H_L summary explains it
    assert g1_rules.decide({2: cap_e, 4: cap_e}, {2: V, 4: V})["outcome"] == "C"
    e3 = _q(dcond8=0.3, dcond8_ci=(0.1, 0.5), budget_drop=1.0, budget_drop_ci=(0.6, 1.4), catchup=0.9, catchup_ci=(0.5, 1.3))
    assert g1_rules.decide({2: e3, 4: e3}, {2: V, 4: V})["outcome"] == "C"
    nest = _q(nest=-0.8)
    assert g1_rules.decide({2: nest, 4: nest}, {2: V, 4: V})["outcome"] == "C"
    assert g1_rules.decide({2: acc, 4: neg}, {2: V, 4: V})["outcome"] == "E"             # checkpoint disagreement is not pooled away
    mid = _q(dcond8=0.3, dcond8_ci=(0.05, 0.55))
    assert g1_rules.decide({2: mid, 4: mid}, {2: V, 4: V})["outcome"] == "E"
    small_vs_band = dict(acc, s=1.5)                                                    # effect not above the parameterization band
    assert g1_rules.label(small_vs_band) != "acc"
    notrecover = _q(accC=53.0, accG=53.2)                                               # negligible but H_L does not recover B
    assert g1_rules.label(notrecover) == "mid"


def test_rules_validity():
    for bad in ({"v_extract": False}, {"v_converged": False}, {"v_edge": False}, {"shuf": 0.2}):
        v = dict(V, **bad)
        assert g1_rules.decide({2: _q(), 4: _q()}, {2: v, 4: V})["outcome"] == "F"


def test_gather_shuffle_moves_only_p():
    rng = np.random.default_rng(1)
    cells = {c: {"z": rng.standard_normal((20, 100)), "p": rng.standard_normal((20, 100)), "labels": rng.integers(0, 100, 20),
                 "h": rng.standard_normal((20, 2048))} for c in g1_fit.spec.CELLS}
    ids = np.arange(12); ca = np.arange(12) % len(g1_fit.spec.CELLS); perm = ids[::-1].copy()
    inp, y = g1_fit.gather(cells, ids, ca, ("z", "h", "p"), None, p_ids=perm)
    for r in range(12):
        c = g1_fit.spec.CELLS[ca[r]]
        assert np.array_equal(inp["z"][r], cells[c]["z"][ids[r]]) and np.array_equal(inp["h"][r], cells[c]["h"][ids[r]])
        assert np.array_equal(inp["p"][r], cells[c]["p"][perm[r]]) and y[r] == cells[c]["labels"][ids[r]]


def test_feat_statistics_come_from_fit_rows_only():
    rng = np.random.default_rng(2)
    fit = {"z": rng.standard_normal((50, 3)) * 3 + 1}; other = {"z": rng.standard_normal((50, 3))}
    f = g1_fit.Feat(("z",)).fit(fit)
    assert np.allclose(f(fit).mean(0), 0) and np.allclose(f(fit).std(0), 1)
    assert not np.allclose(f(other).mean(0), 0, atol=0.2)
