"""Mechanical end-to-end test of clean statistics -> immutable shortlist -> target diagnostics on a synthetic neighbour tree
(no GPU). Verifies that the rigged best site is chosen from VALIDATION rows only, the artifact is read-only, re-selection is
refused, and target statistics refuse to run without the frozen artifact."""
import json, os, stat, sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas import data, spec, stage_stats as ss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def tree(tmp_path, monkeypatch):
    if not (ROOT / data.BENCH).exists():
        pytest.skip("benchmark arrays unavailable")
    os.chdir(ROOT)
    monkeypatch.setattr(data, "ROOT", str(tmp_path / "atlas")); monkeypatch.setattr(data, "SHARED", str(tmp_path / "atlas/shared"))
    os.makedirs(data.SHARED); 
    for f in ("subset_ids.npy", "test_labels.npy"):
        os.symlink(ROOT / "results/atlas/shared" / f, Path(data.SHARED) / f)
    seed = 2
    sd = Path(data.seed_dir(seed)); (sd / "knn_unit").mkdir(parents=True); (sd / "u0").mkdir(); (sd / "stage_a").mkdir()
    os.symlink(ROOT / f"results/atlas/seed{seed}/roles.json", sd / "roles.json")
    arrays = data.load_seed_arrays(seed); by = np.asarray(arrays["train"][1]); vy = np.asarray(arrays["val"][1])
    sub = np.load(Path(data.SHARED) / "subset_ids.npy"); ty = np.load(Path(data.SHARED) / "test_labels.npy")
    rng = np.random.default_rng(0)
    byc = [np.flatnonzero(by == c) for c in range(100)]
    yy = np.concatenate([vy] + [ty[sub] for _ in spec.CONDITIONS])
    logits = lambda n: rng.normal(size=(n, 100)).astype(np.float32)   # noqa: E731
    np.savez(sd / "u0/val.npz", logits=logits(5000))
    for c in spec.CONDITIONS:
        np.savez(sd / f"u0/{c}.npz", logits=logits(10000))
    rigged = ("layer2.1", "gap")
    files = {}
    for s in spec.SITE_NAMES:
        for p in spec.POOLS:
            idx = rng.integers(0, len(by), (len(yy), spec.K_NN)).astype(np.int32)
            if (s, p) == rigged:   # neighbours of the same class as the (validation AND target) query: high accuracy
                for r in range(len(yy)):
                    idx[r] = rng.choice(byc[yy[r]], spec.K_NN)
            np.savez(sd / f"knn_unit/{s}__{p}.npz", idx=idx, dist=np.sort(rng.random((len(yy), spec.K_NN)).astype(np.float32), 1), qnorm=np.ones(len(yy), np.float32))
            files[f"{s}__{p}"] = {"knn_ms_per_query_batch_amortized": 0.1, "dim": 1}
    json.dump({"files": files}, open(sd / "stage_a/group00.done.json", "w"))
    return seed, sd


def test_selection_is_validation_only_immutable_and_gates_target_stats(tree):
    seed, sd = tree
    with pytest.raises(SystemExit):
        ss.target_stats(seed)                       # no frozen shortlist yet
    ss.clean_stats(seed)
    art = ss.select_shortlist(seed)
    assert art["chosen"]["gap"]["site"] == "layer2.1" and art["provenance"]["target_labels_read"] is False
    path = sd / "shortlist.json"
    assert not os.access(path, os.W_OK) and (path.stat().st_mode & stat.S_IWUSR) == 0
    with pytest.raises(SystemExit):
        ss.select_shortlist(seed)                   # immutable
    out = ss.target_stats(seed)
    assert out["layer2.1"]["gap"]["clean"]["candidate_acc_among_base_errors"] == 1.0   # rigged neighbours
    orc = json.load(open(sd / "target_oracle_ranking_unit_l2.json"))
    assert "TARGET-LABELLED" in orc["label"] and orc["best_site_by_condition"]["gap"]["clean"] == "layer2.1"


def test_stage_cd_end_to_end_on_rigged_neighbours(tree):
    from atlas import stage_cd, stats
    seed, sd = tree
    ss.clean_stats(seed); art = ss.select_shortlist(seed)
    arrays = data.load_seed_arrays(seed); by = np.asarray(arrays["train"][1]); vy = np.asarray(arrays["val"][1])
    ty = np.load(Path(data.SHARED) / "test_labels.npy")
    rng = np.random.default_rng(1)
    byc = [np.flatnonzero(by == c) for c in range(100)]
    sets = ("val",) + spec.CONDITIONS
    json.dump({"native_weights": [0.2, 0.2, 0.2, 0.1, 0.1], "native_intercept": 0.5}, open(sd / "u0/done.json", "w"))
    metric = {"hidden": {p: {"metric": "unit_l2"} for p in spec.POOLS}, "output": {"metric": "unit_l2"}}
    json.dump(metric, open(sd / "metric_selection.json", "w"))
    (sd / "knn_F").mkdir()
    for s in sets:
        y = vy if s == "val" else ty
        n = len(y)
        lg = rng.normal(size=(n, 100)).astype(np.float32)
        payload = {"logits": lg, "s_glob": rng.uniform(.2, .8, (n, 5)).astype(np.float32)}
        for m in spec.METRICS:
            payload[f"{m}_idx"] = rng.integers(0, len(by), (n, spec.K_NN)).astype(np.int32); payload[f"{m}_dist"] = np.sort(rng.random((n, spec.K_NN)).astype(np.float32), 1)
        np.savez(sd / f"u0/{s}.npz", **payload)
        for p in spec.POOLS:
            site = art["chosen"][p]["site"]
            idx = rng.integers(0, len(by), (n, spec.K_NN)).astype(np.int32)
            if p == "gap":     # perfect candidate
                for r in range(n):
                    idx[r] = rng.choice(byc[y[r]], spec.K_NN)
            np.savez(sd / f"knn_F/{site}__{p}__unit_l2__{s}.npz", unit_l2_idx=idx, unit_l2_dist=np.sort(rng.random((n, spec.K_NN)).astype(np.float32), 1),
                     qnorm=rng.uniform(.5, 2, n).astype(np.float32))
    np.savez(Path(data.SHARED) / "dummy.npz", x=np.zeros(1))
    stage_cd.run(seed)
    res = json.load(open(sd / "cd/results.json")); meta = json.load(open(sd / "cd/meta.json"))
    for name in ("base", "ts", "vs", "ms", "native_dac", "gap_F0", "gap_F1", "out_F0", "out_F1", "gap_alone"):
        assert name in res and set(res[name]) == {"raw", "T_nll", "T_brier"}
    assert all(t["nll"] > 0 and t["brier"] > 0 for t in meta["temperatures"].values())
    # native DAC and every scalar-temperature output keep the base argmax
    for name in ("native_dac", "ts"):
        assert res[name]["raw"]["clean"]["wh_vs_base"]["total_flips"] == 0
    # perfect candidate alone is far better than base; the F1 gate on it selects interventions and beats base accuracy
    assert res["gap_alone"]["raw"]["clean"]["accuracy"] > 0.9
    assert res["gap_F1"]["raw"]["clean"]["accuracy"] > res["base"]["raw"]["clean"]["accuracy"] + 0.2
    # the temperature never changes decisions
    for name in ("gap_F1", "out_F0"):
        for c in spec.CONDITIONS:
            assert res[name]["T_nll"][c]["wh_vs_base"]["net"] == res[name]["raw"][c]["wh_vs_base"]["net"]


def test_metric_selection_uses_validation_rows_and_is_immutable(tree):
    from atlas import stage_bsel
    seed, sd = tree
    ss.clean_stats(seed); art = ss.select_shortlist(seed)
    arrays = data.load_seed_arrays(seed); by = np.asarray(arrays["train"][1]); vy = np.asarray(arrays["val"][1])
    rng = np.random.default_rng(2); byc = [np.flatnonzero(by == c) for c in range(100)]
    (sd / "knn_B").mkdir()
    n = 31000
    for p in spec.POOLS:
        site = art["chosen"][p]["site"]
        payload = {"qnorm": np.ones(n, np.float32)}
        for m in ("raw_l2", "mahalanobis"):
            idx = rng.integers(0, len(by), (n, spec.K_NN)).astype(np.int32)
            if p == "grid2" and m == "mahalanobis":     # rig: Mahalanobis neighbours are same-class on validation rows
                for r in range(5000):
                    idx[r] = rng.choice(byc[vy[r]], spec.K_NN)
            payload[f"{m}_idx"], payload[f"{m}_dist"] = idx, np.sort(rng.random((n, spec.K_NN)).astype(np.float32), 1)
        np.savez(sd / f"knn_B/{site}__{p}__atlas.npz", **payload)
        json.dump({"info": {"rank": 5}}, open(sd / f"knn_B/{site}__{p}.done.json", "w"))
    payload = {}
    for m in spec.METRICS:
        payload[f"{m}_idx"] = rng.integers(0, len(by), (5000, spec.K_NN)).astype(np.int32); payload[f"{m}_dist"] = np.sort(rng.random((5000, spec.K_NN)).astype(np.float32), 1)
    np.savez(sd / "u0/val.npz", logits=np.load(sd / "u0/val.npz")["logits"], **payload)
    json.dump({"logit_mahalanobis": {"rank": 100}}, open(sd / "u0/done.json", "w"))
    out = stage_bsel.run(seed)
    assert out["hidden"]["grid2"]["metric"] == "mahalanobis" and out["hidden"]["gap"]["metric"] == "unit_l2" and out["provenance"]["target_labels_read"] is False
    assert not os.access(sd / "metric_selection.json", os.W_OK)
    with pytest.raises(SystemExit):
        stage_bsel.run(seed)
