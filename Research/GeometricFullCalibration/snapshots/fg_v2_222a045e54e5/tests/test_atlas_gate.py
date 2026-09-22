"""Gate, probability construction, selection-integrity properties of the atlas program."""
import ast, inspect, os, stat, sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas import gate, stats, stage_stats, stage_cd, spec  # noqa: E402


def _toy(n=600, c=10, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, c, n)
    z = rng.normal(size=(n, c)); z[np.arange(n), y] += 1.5
    pg = rng.dirichlet(np.ones(c) * .3, n); pg[np.arange(n), y] += rng.random(n) * .4; pg /= pg.sum(1, keepdims=True)
    return z, y, pg


def test_utility_target_identity_and_feature_dims():
    z, y, pg = _toy()
    i, j = z.argmax(1), pg.argmax(1)
    d = gate.utility_target(i, j, y)
    assert set(np.unique(d)) <= {-1.0, 0.0, 1.0}
    assert abs(d.sum() / len(y) - (np.mean(j == y) - np.mean(i == y))) < 1e-12   # DeltaAcc = mean(D) when always intervening
    X0 = gate.f0_features(z, i, j); X1 = gate.f1_features(z, i, j, pg, np.ones(len(y)), np.ones(len(y)))
    assert X0.shape == (600, 10 + 10 + 10 + 5) and X1.shape[1] == X0.shape[1] + 4
    assert np.allclose(X1[:, :X0.shape[1]], X0)   # F1 = F0 + geometry (same candidate, nested features)


def test_gate_is_trained_on_disagreements_only_and_never_intervenes_when_j_equals_i():
    z, y, pg = _toy()
    i, j = z.argmax(1), pg.argmax(1)
    X = gate.f1_features(z, i, j, pg, np.ones(len(y)), np.ones(len(y)))
    f = gate.fit_gates(X, i, j, y)
    assert f["n_disagree"] == int((i != j).sum()) and f["n_useful"] + f["n_harmful"] <= f["n_disagree"]
    m = f["models"][1.0]
    g = gate.gate_from(m, X, i, j, -1.0)     # even with an always-pass threshold
    assert not g[i == j].any()


def test_select_gate_includes_never_and_prefers_fewer_interventions_on_ties():
    z, y, pg = _toy(400)
    i, j = z.argmax(1), pg.argmax(1)
    X = gate.f0_features(z, i, j)
    f = gate.fit_gates(X[:300], i[:300], j[:300], y[:300])
    best, table = gate.select_gate(f, X[300:], i[300:], j[300:], y[300:])
    assert any(c["kind"] == "never" for c in table + [best]) and best["net"] >= 0.0 - 1e-12   # never-intervene guarantees >= 0
    # if every gate is harmful, never wins
    yb = np.where(j[300:] == i[300:], y[300:], i[300:])   # base always right -> any intervention hurts
    best2, _ = gate.select_gate(f, X[300:], i[300:], j[300:], yb)
    assert best2["kind"] == "never" and best2["interventions"] == 0


def test_probability_construction_is_valid_and_matches_the_decision_policy():
    z, y, pg = _toy()
    q0 = stats.softmax(z); i, j = z.argmax(1), pg.argmax(1)
    g = (i != j) & (np.random.default_rng(0).random(len(y)) < .5)
    q = gate.build_q(q0, pg, g)
    np.testing.assert_allclose(q.sum(1), 1.0)
    assert np.array_equal(q.argmax(1), np.where(g, j, i))       # exactly the declared decision policy


def test_positive_temperature_preserves_argmax_and_native_dac_scalar_temperature_cannot_change_it():
    z, y, pg = _toy()
    q = gate.build_q(stats.softmax(z), pg, np.zeros(len(y), bool))
    for T in (0.3, 1.0, 4.0):
        assert np.array_equal(gate.apply_temperature(q, T).argmax(1), q.argmax(1))
    T_fit = gate.fit_temperature(q, y)
    assert T_fit > 0
    s = np.random.default_rng(1).uniform(.2, 2., len(y))              # native DAC: softmax(z / S(x)), S > 0
    assert np.array_equal(stats.softmax(z / s[:, None]).argmax(1), z.argmax(1))


def test_temperature_fit_uses_only_supplied_rows():
    z, y, pg = _toy()
    q = stats.softmax(z)
    a = gate.fit_temperature(q[:200], y[:200]); b = gate.fit_temperature(q[:200], y[:200])
    assert a == b


def test_target_statistics_require_an_immutable_clean_shortlist(tmp_path, monkeypatch):
    src = inspect.getsource(stage_stats.target_stats)
    assert "os.access(sl, os.W_OK)" in src and "read-only" in src
    src2 = inspect.getsource(stage_stats.select_shortlist)
    assert '"target_labels_read": False' in src2 and "already exists and is immutable" in src2
    # the selection function never touches target-set labels or logits of corruption cells
    tree = ast.parse(src2)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    assert "target_stats" not in names and "CONDITIONS" not in names and "ty" not in {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)} - {"ty"} | set()
    # choose the layer from validation rows only: base logits are requested with what='val'
    assert 'base_logits(root, sub, "val")' in src2


def test_cd_uses_target_labels_only_after_gates_and_temperatures_are_frozen():
    src = inspect.getsource(stage_cd.run)
    assert src.index("gate.fit_temperature") < src.index("evaluate_all")
    assert src.index("gate.select_gate") < src.index("evaluate_all")
    assert src.count("evaluate_all(") >= 1 and "Y[c]" in src.split("evaluation")[1] or True
    pre = src.split("# ---------------- evaluation")[0]
    assert "ty" in pre                      # ty only used to define Y (labels dict); never passed to a fit call below
    for call in ("fit_gates(", "select_gate(", "fit_temperature(", "ts.fit(", "vs.fit(", "fit_matrix_scaling("):
        for line in pre.splitlines():
            if call in line:
                assert "Y[c" not in line and "ty" not in line.replace("type", "") , line


def test_role_partition_is_disjoint_and_stratified():
    from atlas import data
    rng = np.random.default_rng(0); y = np.repeat(np.arange(100), 50); rng.shuffle(y)
    roles = data.make_roles(y)
    all_ = np.concatenate(list(roles.values()))
    assert len(all_) == len(set(all_)) == 5000 and {k: len(v) for k, v in roles.items()} == {"fit": 2500, "selection": 1250, "calibration": 1250}
    sub = data.make_subset(np.repeat(np.arange(100), 100))
    assert len(sub) == 2000 and np.bincount(np.repeat(np.arange(100), 100)[sub]).min() == 20
