"""Structural tests for Decoder Panel v1, G1-DP and N1a-DP (docs/decoder_panel_v1_spec.md sec. 9). No scientific outcome is read or
computed: all targets are synthetic or the tests only inspect row sets / shapes / seeds."""
import json
import os

import numpy as np
import pytest

from atlas import decoder_panel as dp, g1dp, n1a, n1adp, spec


@pytest.fixture(autouse=True)
def _small(monkeypatch):
    monkeypatch.setattr(dp, "N_TRIALS", 3); monkeypatch.setattr(dp, "LGBM_MAX_ROUNDS", 20); monkeypatch.setattr(dp, "MLP_MAX_EPOCHS", 4)
    monkeypatch.setattr(dp, "LAMBDA_GRID", (1e0, 1e-2, 1e-4)); monkeypatch.setattr(dp, "POLY_DIMS", (32, 64)); monkeypatch.setattr(dp, "RFF_MULTS", (0.5, 1.0))
    monkeypatch.setattr(dp, "RFF_COMPONENTS", 64); monkeypatch.setattr(dp, "KNN_K", (3, 5)); monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")


def _toy(n=300, d=12, K=4, seed=0):
    rng = np.random.default_rng(seed); X = rng.standard_normal((n, d)); off = rng.standard_normal((n, K))
    y = (off + X[:, :K] + rng.gumbel(size=(n, K))).argmax(1)
    return X, y, off


def _ctx(n, ids=None):
    fi, vi = np.arange(0, int(n * .7)), np.arange(int(n * .7), n)
    return dp.HPOContext("test", ids or {"a": 1}, "target_pooled", [(fi, vi)], dp.nll_objective)


# 1, 13 ------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("family", dp.FAMILIES)
def test_preprocessing_sees_only_permitted_training_rows(family, monkeypatch):
    """Every Std.fit / transform-fit input consists of training rows only; a sentinel evaluation matrix never reaches any fit."""
    X, y, off = _toy(); SENT = 1e6; seen = []
    orig = dp.Std.fit
    def spy(self, A):
        seen.append(np.asarray(A)); return orig(self, A)
    monkeypatch.setattr(dp.Std, "fit", spy)
    ctx = _ctx(len(y)); m, rec = dp.fit_decoder(family, X, y, "classification", ctx, offset=off, allow_null=False)
    X_eval = np.full((20, X.shape[1]), SENT); m.predict(X_eval, np.zeros((20, 4)))
    fi, _ = ctx.inner_splits[0]
    for A in seen:
        assert np.abs(A).max() < SENT / 10
        assert A.shape[0] in (len(fi), len(y))            # inner-fit rows (selection) or all outer-train rows (refit)
    if family in ("linear", "poly2", "rff", "knn", "mlp"):
        assert any(A.shape[0] == len(y) for A in seen) and any(A.shape[0] == len(fi) for A in seen)


# 2 ----------------------------------------------------------------------------------------------------------------------------------
def test_objective_only_receives_inner_validation_rows():
    X, y, off = _toy(); calls = []
    def obj(pred, yv, rows):
        calls.append(np.asarray(rows)); return dp.nll_objective(pred, yv)
    fi, vi = np.arange(200), np.arange(200, 300)
    ctx = dp.HPOContext("test", {"a": 1}, "target_pooled", [(fi, vi)], obj)
    for fam in dp.FAMILIES:
        calls.clear(); dp.fit_decoder(fam, X, y, "classification", ctx, offset=off)
        assert calls and all(np.array_equal(r, vi) for r in calls)


def test_g1dp_fit_arm_never_uses_test_labels_for_selection(tmp_path, monkeypatch):
    """A fake bundle whose outer-test labels are out of range: selection/refit must succeed (they never see them); only evaluation reads them."""
    rng = np.random.default_rng(3); n, nt, C = 160, 30, len(spec.CONDITIONS)
    B = {"train_ids": np.arange(n), "inner_fit_mask": np.arange(n) < 120, "test_ids": np.arange(n, n + nt),
         "train_z": rng.standard_normal((n, 100)), "train_y": rng.integers(0, 100, n), "train_p": rng.standard_normal((n, 100)),
         "test_z": rng.standard_normal((C, nt, 100)), "test_p": rng.standard_normal((C, nt, 100)), "test_y": np.zeros((C, nt), np.int64)}
    seen_y = []
    orig = dp.fit_decoder
    def spy(family, X, y, *a, **k):
        seen_y.append(np.asarray(y)); return orig(family, X, y, *a, **k)
    monkeypatch.setattr(dp, "fit_decoder", spy)
    out, rec = g1dp.fit_arm(B, "linear", 2, 0, "B")
    assert len(seen_y) == 1 and np.array_equal(seen_y[0], B["train_y"]) and len(seen_y[0]) == n
    assert out["correct"].shape == (C, nt) and rec["mode"] == "target_pooled"


# 3 ----------------------------------------------------------------------------------------------------------------------------------
def test_deterministic_inner_splits():
    ctx = [g1dp.context(2, 0, "A", np.arange(10) < 7).inner_splits[0] for _ in range(2)]
    assert all(np.array_equal(a, b) for a, b in zip(*ctx))
    data = n1a.load(2)
    U1, U2 = n1adp.unit_rows(data, 2, "fog", 0), n1adp.unit_rows(data, 2, "fog", 0)
    for (a, b), (c, d) in zip(U1["splits"], U2["splits"]):
        assert np.array_equal(a, c) and np.array_equal(b, d)


# 4 ----------------------------------------------------------------------------------------------------------------------------------
def test_deterministic_study_seeds():
    a = dp.HPOContext("G1-DP", {"seed": 2, "fold": 0, "arm": "C"}, "target_pooled", [(np.arange(2), np.arange(2, 4))], dp.nll_objective)
    b = dp.HPOContext("G1-DP", {"arm": "C", "fold": 0, "seed": 2}, "target_pooled", [(np.arange(2), np.arange(2, 4))], dp.nll_objective)
    c = dp.HPOContext("G1-DP", {"seed": 2, "fold": 0, "arm": "D"}, "target_pooled", [(np.arange(2), np.arange(2, 4))], dp.nll_objective)
    assert a.study_seed("lgbm") == b.study_seed("lgbm") != c.study_seed("lgbm") and a.study_seed("lgbm") != a.study_seed("mlp")
    assert dp.derive_seed("x", 1) == dp.derive_seed("x", 1)


def test_optuna_study_reproducible():
    X, y, off = _toy()
    r1 = dp.fit_decoder("lgbm", X, y, "classification", _ctx(len(y)), offset=off)[1]
    r2 = dp.fit_decoder("lgbm", X, y, "classification", _ctx(len(y)), offset=off)[1]
    assert [t["params"] for t in r1["trials"]] == [t["params"] for t in r2["trials"]]
    assert [t["objective"] for t in r1["trials"]] == pytest.approx([t["objective"] for t in r2["trials"]])


# 5, 6 -------------------------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("family", dp.FAMILIES)
def test_matched_arms_get_identical_search_space_and_budget(family):
    X, y, off = _toy(d=20)
    ra = dp.fit_decoder(family, X[:, :5], y, "classification", _ctx(len(y), {"arm": "A"}), offset=off)[1]
    rb = dp.fit_decoder(family, X, y, "classification", _ctx(len(y), {"arm": "C"}), offset=off)[1]
    assert len(ra["trials"]) == len(rb["trials"]) and ra["space_version"] == rb["space_version"]
    ka = [tuple(sorted(t["params"])) for t in ra["trials"]]; kb = [tuple(sorted(t["params"])) for t in rb["trials"]]
    assert ka == kb
    if family not in dp.OPTUNA_FAMILIES:
        assert [t["params"] for t in ra["trials"]] == [t["params"] for t in rb["trials"]]     # identical exhaustive grid
    else:
        assert len(ra["trials"]) == dp.N_TRIALS


# 7, 8, 9 ------------------------------------------------------------------------------------------------------------------------------
def test_sketch_projection_rff_reproducible():
    X, _, _ = _toy(n=200, d=30)
    assert np.array_equal(dp._poly_transform(X, 64)(X), dp._poly_transform(X, 64)(X))
    T1, m1 = dp._knn_transform(np.tile(X, (1, 10))); T2, _ = dp._knn_transform(np.tile(X, (1, 10)))
    assert m1["projected"] and np.array_equal(T1(np.tile(X, (1, 10))), T2(np.tile(X, (1, 10))))
    assert np.array_equal(dp._rff_transform(X, 1.0)[0](X), dp._rff_transform(X, 1.0)[0](X))


def test_jl_target_and_distortion_audit():
    assert dp.jl_target(8000, 100) == 100                           # no projection when d is below the JL target
    t = dp.jl_target(6000, 2148); assert 300 < t < 2148
    rng = np.random.default_rng(0); X = rng.standard_normal((600, 3000))
    a = dp.jl_audit(X, lambda A: A)
    assert a["median"] == pytest.approx(1.0) and a["p05"] == pytest.approx(1.0)
    T, m = dp._knn_transform(X)
    assert m["projected"] and 0.8 < m["jl_audit"]["median"] < 1.2 and m["jl_audit"]["p05"] > 0.5 and m["jl_audit"]["p95"] < 1.5


# 10, 11, 12 ---------------------------------------------------------------------------------------------------------------------------
def test_lgbm_uses_raw_coordinates():
    X, y, off = _toy()
    m, _ = dp.fit_decoder("lgbm", X, y, "classification", _ctx(len(y)), offset=off, allow_null=False)
    assert m.transform(X) is X and m.head["booster"].num_feature() == X.shape[1] and m.meta["raw_coordinates"]


def test_mlp_has_exactly_one_hidden_layer():
    import torch
    net = dp.make_mlp(10, 64, 3, 0.1)
    lin = [mod for mod in net.modules() if isinstance(mod, torch.nn.Linear)]
    assert len(lin) == 2 and lin[0].out_features == 64 and isinstance(net[1], torch.nn.GELU)
    assert torch.count_nonzero(lin[1].weight) == 0          # anchored start: g = 0
    X, y, off = _toy(); m, _ = dp.fit_decoder("mlp", X, y, "classification", _ctx(len(y)), offset=off, allow_null=False)
    assert len([mod for mod in m.head["net"].modules() if isinstance(mod, torch.nn.Linear)]) == 2 and m.meta["hidden_layers"] == 1


def test_rff_gamma_uses_training_rows_only():
    X, _, _ = _toy(n=400, d=10); Xs = dp.Std().fit(X)(X)
    g1 = dp.rff_gamma_median(Xs)
    _, meta = dp._rff_transform(X, 1.0)
    X2 = X.copy(); _, meta2 = dp._rff_transform(X2, 1.0)
    assert meta["gamma_median"] == meta2["gamma_median"] == pytest.approx(g1)


# 14, 15, 16, 19 ------------------------------------------------------------------------------------------------------------------------
def test_n1adp_heldout_family_never_in_selection_and_images_disjoint():
    data = n1a.load(4)
    for fam in n1a.FAMILIES:
        U = n1adp.unit_rows(data, 4, fam, 1)
        assert fam not in set(U["env"].tolist())
        ctx = n1adp.context(4, fam, 1, U); ctx.validate(len(U["y"]))
        assert ctx.mode == "shift_transfer" and len(ctx.inner_splits) == 3
        for fi, vi in U["splits"]:
            assert len(set(U["env"][vi].tolist())) == 1 and not set(U["env"][vi]) & set(U["env"][fi])
            assert not set(U["img"][fi].tolist()) & set(U["img"][vi].tolist())      # inner image disjointness
        assert not set(U["me"][:, 0].tolist()) & set(U["img"].tolist())              # outer image disjointness
        assert U["X"].shape[1] == 206


def test_n1adp_evidence_excludes_other_model():
    data = n1a.load(2); U1 = n1adp.unit_rows(data, 2, "fog", 0)
    rng = np.random.default_rng(0)
    for c in spec.CELLS:
        data[c] = dict(data[c], zo=rng.standard_normal(data[c]["zo"].shape))
    U2 = n1adp.unit_rows(data, 2, "fog", 0)
    assert np.array_equal(U1["X"], U2["X"]) and np.array_equal(U1["Xe"], U2["Xe"])


def test_hpo_modes_cannot_be_mixed():
    fi, vi = np.arange(10), np.arange(10, 20); env = np.array(["a"] * 10 + ["b"] * 10)
    with pytest.raises(AssertionError):      # target_pooled refuses an environment holdout / multiple splits
        dp.HPOContext("x", {}, "target_pooled", [(fi, vi), (vi, fi)], dp.nll_objective).validate(20)
    with pytest.raises(AssertionError):
        dp.HPOContext("x", {}, "target_pooled", [(fi, vi)], dp.nll_objective, forbidden_env="c").validate(20)
    with pytest.raises(AssertionError):      # shift_transfer requires environments
        dp.HPOContext("x", {}, "shift_transfer", [(fi, vi), (vi, fi)], dp.nll_objective).validate(20)
    with pytest.raises(AssertionError):      # held-out environment present in training rows
        dp.HPOContext("x", {}, "shift_transfer", [(fi, vi), (vi, fi)], dp.nll_objective, forbidden_env="a", env_train=env).validate(20)
    dp.HPOContext("x", {}, "shift_transfer", [(fi, vi), (vi, fi)], dp.nll_objective, forbidden_env="c", env_train=env).validate(20)
    assert g1dp.context(2, 0, "A", np.arange(10) < 7).mode == "target_pooled"


# 17, 18 ------------------------------------------------------------------------------------------------------------------------------
def test_output_paths_unique():
    paths = [g1dp.out_paths(f, s, k, a)["npz"] for f in dp.FAMILIES for s in g1dp.SEEDS for k in g1dp.FOLDS for a in g1dp.ARM_ORDER]
    paths += [n1adp.out_paths(f, b, h, k)["npz"] for f in dp.FAMILIES for b, h, k in n1adp.units()]
    assert len(paths) == len(set(paths))
    assert len(g1dp.units(False)) == 10 and len(g1dp.units(True)) == 110 and len(n1adp.units()) == 40


def test_restart_skips_only_validated_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(g1dp, "FITS", str(tmp_path))
    p = g1dp.out_paths("linear", 2, 0, "A"); os.makedirs(p["dir"])
    assert not g1dp.is_complete("linear", 2, 0, "A")
    np.savez(p["npz"], a=1); open(p["study"], "w").write("{}")
    assert not g1dp.is_complete("linear", 2, 0, "A")                  # no marker
    from atlas import common
    json.dump({"npz_sha256": common.file_sha(p["npz"]), "study_sha256": common.file_sha(p["study"])}, open(p["done"], "w"))
    assert g1dp.is_complete("linear", 2, 0, "A")
    open(p["study"], "w").write('{"x": 1}')                           # corrupted after completion
    assert not g1dp.is_complete("linear", 2, 0, "A")


# G1-DP arms / shuffle -----------------------------------------------------------------------------------------------------------------
def test_g1dp_arms_preserve_g1_meaning_and_shuffle_rule():
    from atlas import g1_fit
    for a in ("A", "B", "C", "D", "E", "F"):
        assert g1dp.ARMS[a] == g1_fit.ARMS[a]
    for ctrl, real in (("Cs", "C"), ("Hs", "H"), ("Ds", "D")):
        assert len(g1dp.ARMS[ctrl]) == len(g1dp.ARMS[real]) and g1dp.ARMS[ctrl][0] == "z"
        assert [b.rstrip("~") for b in g1dp.ARMS[ctrl]] == list(g1dp.ARMS[real])
    assert g1dp.PERM_SEED == g1_fit.PERM_SEED
    P = g1dp._perms(np.arange(5), np.arange(5, 8), np.arange(8), np.arange(8, 12))
    assert sorted(P["fit"].tolist()) == list(range(5)) and sorted(P["test"].tolist()) == list(range(8, 12))


def test_regression_and_binary_tasks_supported():
    X, y, _ = _toy(); yr = X[:, 0] * X[:, 1]
    mse = lambda p, t, _r=None: float(np.mean((p - t) ** 2))  # noqa: E731
    ctx = dp.HPOContext("t", {"a": 1}, "target_pooled", [(np.arange(200), np.arange(200, 300))], mse)
    for fam in dp.FAMILIES:
        m, _ = dp.fit_decoder(fam, X, yr, "regression", ctx); assert m.predict(X[:7]).shape == (7,)
        m, _ = dp.fit_decoder(fam, X, (y > 1).astype(int), "classification", _ctx(len(y)), K=2); assert m.predict(X[:7]).shape == (7, 2)


# frozen decision rules -------------------------------------------------------------------------------------------------------------------
def test_g1dp_classify_and_verdict():
    from atlas import g1dp_rules as R
    assert R.classify(0.8, (0.3, 1.2)) == "POS" and R.classify(-0.8, (-1.2, -0.3)) == "REV"
    assert R.classify(0.1, (-0.2, 0.4)) == "NULL" and R.classify(0.4, (0.1, 0.7)) == "UNC"
    fams = dp.FAMILIES; allv = set(fams)
    lab = lambda d: {f: {2: d[f], 4: d[f]} for f in fams}  # noqa: E731
    assert R.verdict(lab({f: "POS" for f in fams}), allv)["outcome"] == "ROBUST+"
    assert R.verdict(lab({f: ("POS" if f == "linear" else "NULL") for f in fams}), allv)["outcome"] == "SPECIFIC"
    assert R.verdict(lab({f: "NULL" for f in fams}), allv)["outcome"] == "NEGATIVE"
    assert R.verdict(lab({f: "UNC" for f in fams}), allv)["outcome"] == "INC"
    assert R.verdict(lab({f: "POS" for f in fams}), set(fams[:4]))["outcome"] == "INC_VALID"
    mixed = {f: {2: "POS", 4: "NULL"} for f in fams}                       # checkpoint disagreement
    assert R.verdict(mixed, allv)["outcome"] == "INC"


def test_n1adp_decision_rule():
    from atlas import n1adp_rules as R
    res = {"phi": .6, "phi_ci": (.55, .65), "U": 3, "U_ci": (2.5, 3.5), "MA": 2, "MA_ci": (1.5, 2.5)}
    sub = {"phi": .15, "phi_ci": (.1, .2), "U": 7, "U_ci": (6.5, 7.5), "MA": 7, "MA_ci": (6.8, 7.2)}
    mid = {"phi": .4, "phi_ci": (.35, .45), "U": 5, "U_ci": (4.6, 5.4), "MA": 3, "MA_ci": (1.5, 4)}
    fams = dp.FAMILIES; allv = set(fams); MB = {2: 3.5, 4: 3.4}
    mk = lambda d: {f: {2: d[f], 4: d[f]} for f in fams}  # noqa: E731
    assert R.decide(mk({f: sub for f in fams}), MB, allv)["outcome"] == "ROBUST"
    assert R.decide(mk({f: sub for f in fams}), {2: 0.5, 4: 3.4}, allv)["outcome"] == "INC"
    assert R.decide(mk({f: (res if f in ("lgbm", "mlp") else sub) for f in fams}), MB, allv)["outcome"] == "LIMIT"
    assert R.decide(mk({f: (res if f == "lgbm" else sub) for f in fams}), MB, allv)["outcome"] == "SPECIFIC"
    assert R.decide(mk({f: (mid if f == "lgbm" else sub) for f in fams}), MB, allv)["outcome"] == "INC"
    assert R.decide(mk({f: sub for f in fams}), MB, set(fams[:4]))["outcome"] == "INC_VALID"


def test_g1dp_task_end_to_end_with_fake_bundle(tmp_path, monkeypatch):
    """Fake on-disk bundle -> stage to scratch (sha256 verified) -> fit two arms incl. a shuffled control -> atomic outputs -> restart skip."""
    from atlas import common
    monkeypatch.setattr(g1dp, "BUNDLES", str(tmp_path / "b")); monkeypatch.setattr(g1dp, "FITS", str(tmp_path / "f"))
    rng = np.random.default_rng(5); n, nt, C = 120, 20, len(spec.CONDITIONS); d = f"{tmp_path}/b/seed2/fold0"; os.makedirs(d)
    arrs = {"train_ids": np.arange(n), "inner_fit_mask": np.arange(n) < 90, "cell_assign": np.zeros(n, int), "test_ids": np.arange(n, n + nt),
            "train_y": rng.integers(0, 100, n), "test_y": rng.integers(0, 100, (C, nt)), "train_z": rng.standard_normal((n, 100)),
            "test_z": rng.standard_normal((C, nt, 100)), "train_p": rng.standard_normal((n, 100)), "test_p": rng.standard_normal((C, nt, 100)),
            "train_h": rng.standard_normal((n, 2048)).astype(np.float32), "test_h": rng.standard_normal((C, nt, 2048)).astype(np.float32),
            "train_p~inner": rng.standard_normal((n, 100)), "train_p~full": rng.standard_normal((n, 100)), "test_p~test": rng.standard_normal((C, nt, 100))}
    man = {}
    for k, v in arrs.items():
        np.save(f"{d}/{k}.npy", v); man[k] = {"sha256": common.file_sha(f"{d}/{k}.npy")}
    json.dump({"files": man}, open(f"{d}/manifest.json", "w"))
    g1dp.run("knn", 2, 0, ["B", "Ds"])
    for a in ("B", "Ds"):
        assert g1dp.is_complete("knn", 2, 0, a)
        x = np.load(g1dp.out_paths("knn", 2, 0, a)["npz"]); assert x["correct"].shape == (C, nt)
    assert _needed_ok(g1dp._needed(["Ds"]))
    mt = os.path.getmtime(g1dp.out_paths("knn", 2, 0, "B")["npz"]); g1dp.run("knn", 2, 0, ["B"])
    assert os.path.getmtime(g1dp.out_paths("knn", 2, 0, "B")["npz"]) == mt           # validated output skipped


def _needed_ok(need):
    return {"train_p~inner", "train_p~full", "test_p~test", "train_h", "test_h"} <= set(need) and "train_h3" not in need


def test_g1dp_tie_aware_gate():
    from atlas.g1dp_extract import tie_aware_agreement
    pc = np.array([[1.0, 3.0, 3.0], [5.0, 1.0, 0.0], [0.0, 2.0, 1.0]], np.float16)
    pr = np.array([[1.0, 3.0, 3.001], [5.0, 1.0, 0.0], [0.0, 1.0, 2.0]])   # row 0: tie -> consistent; row 2: non-tied disagreement
    assert tie_aware_agreement(pr, pc) == pytest.approx(2 / 3)
