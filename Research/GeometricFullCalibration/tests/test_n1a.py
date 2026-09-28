"""Structural tests for N1a (docs/n1a_action_ambiguity_spec.md). No N1a outcome is read."""
import numpy as np

from atlas import n1a, n1a_rules, spec
from atlas.n1a_aggregate import knn_ambiguity


def _logits(pred, n_cls=5):
    z = np.zeros((len(pred), n_cls)); z[np.arange(len(pred)), pred] = 5.0; return z


def test_paired_action_outcomes():
    y = np.array([0, 1, 2, 3])
    zb = _logits(np.array([0, 0, 2, 4]))       # keep correct on rows 0, 2
    zo = _logits(np.array([0, 1, 3, 4]))       # route correct on rows 0, 1
    oc = n1a.outcomes(zb, zo, y)
    assert oc["d_route"].tolist() == [0, 1, -1, 0]          # both-correct, repair, harm, both-wrong
    assert set(np.unique(oc["d_ens"])) <= {-1, 0, 1}


def test_symmetric_routing_negates_delta():
    rng = np.random.default_rng(0); y = rng.integers(0, 5, 200)
    za, zb = rng.standard_normal((200, 5)), rng.standard_normal((200, 5))
    assert n1a.other(2) == 4 and n1a.other(4) == 2
    assert np.array_equal(n1a.outcomes(za, zb, y)["d_route"], -n1a.outcomes(zb, za, y)["d_route"])


def test_family_and_image_holdout_all_folds():
    for fam in n1a.FAMILIES:
        seen = set()
        for k in range(5):
            sp = n1a.split(k, fam)
            assert not set(sp["train_ids"].tolist()) & set(sp["eval_ids"].tolist())
            assert set(sp["fit_ids"].tolist()) | set(sp["val_ids"].tolist()) == set(sp["train_ids"].tolist())
            assert all(c.startswith(fam + "_s") for c in sp["eval_cells"]) and len(sp["eval_cells"]) == 3
            assert not any(c.startswith(fam + "_s") for c in sp["train_cells"]) and len(sp["train_cells"]) == 9
            assert not seen & set(sp["eval_ids"].tolist()); seen |= set(sp["eval_ids"].tolist())
        assert len(seen) == 10000                           # every image evaluated exactly once per held-out family


def test_output_evidence_excludes_other_model():
    rng = np.random.default_rng(1); ids = np.arange(30)
    data = {c: {"zb": rng.standard_normal((30, 100)), "zo": rng.standard_normal((30, 100)), "y": rng.integers(0, 100, 30)} for c in spec.CELLS}
    X1, D1, _ = n1a.rows(data, ids, spec.CELLS[:2], "Z")
    for c in spec.CELLS:
        data[c]["zo"] = rng.standard_normal((30, 100))       # change only the other model's logits
    X2, _, _ = n1a.rows(data, ids, spec.CELLS[:2], "Z")
    assert X1.shape[1] == 206 and np.array_equal(X1, X2)
    X3, _, _ = n1a.rows(data, ids, spec.CELLS[:2], "ZZo")
    assert X3.shape[1] == 412


def test_output_features_deterministic_from_single_logits():
    z = np.random.default_rng(2).standard_normal((10, 100))
    f = n1a.output_features(z)
    assert np.array_equal(f, n1a.output_features(z.copy())) and f.shape == (10, 206)
    assert np.allclose(f[:, 100 + 4], np.exp(z - z.max(1, keepdims=True)).max(1) / np.exp(z - z.max(1, keepdims=True)).sum(1))


def test_knn_ambiguity_excludes_same_image_and_is_deterministic():
    rng = np.random.default_rng(3)
    img = np.repeat(np.arange(40), 3); d = rng.integers(-1, 2, 120); z = rng.standard_normal((120, 5)).astype(np.float32)
    z[1::3] = z[0::3]; z[2::3] = z[0::3]                     # same-image rows are identical in Z
    P = {"d": d, "img": img, "z": z, "fold": np.zeros(120, int)}
    m1, m2 = knn_ambiguity(P), knn_ambiguity(P)
    assert np.array_equal(m1, m2) and ((m1 >= 0) & (m1 <= 0.5)).all()


def _q(**kw):
    q = {"h": 5.0, "MA": 3.0, "MA_ci": (2.5, 3.5), "MA_by_family": {f: 3.0 for f in n1a.FAMILIES}, "MB": 3.0, "Q": 1.0, "Q_ci": (0.6, 1.4),
         "support": {f: (2000, 2000) for f in n1a.FAMILIES}, "converged": True}
    q.update(kw); return q


def test_stop_go_rules():
    assert n1a_rules.decide({2: _q(), 4: _q()})["outcome"] == "GO"
    assert n1a_rules.decide({2: _q(h=0.5), 4: _q()})["outcome"] == "STOP_HET"
    assert n1a_rules.decide({2: _q(MA=0.5, MA_ci=(0.3, 0.8)), 4: _q(Q=0.1, Q_ci=(-0.1, 0.3))})["outcome"] == "STOP_Z"
    assert n1a_rules.decide({2: _q(), 4: _q(MB=0.5)})["outcome"] == "INC_PREC"
    fam_bad = {f: 3.0 for f in n1a.FAMILIES}; fam_bad.update({n1a.FAMILIES[0]: 0.2, n1a.FAMILIES[1]: 0.2})
    assert n1a_rules.decide({2: _q(MA_by_family=fam_bad), 4: _q()})["outcome"] == "INC_PREC"
    sup = {f: (100, 100) for f in n1a.FAMILIES}
    assert n1a_rules.decide({2: _q(support=sup), 4: _q()})["outcome"] == "INC_VALID"
    assert n1a_rules.decide({2: _q(converged=False), 4: _q()})["outcome"] == "INC_VALID"
