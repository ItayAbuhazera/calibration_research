"""
Tests for the Study-A incremental-geometry reliability experiment.

Adversarial where it matters: the arms must be genuinely matched, the
controls must actually be controls, and no arm may reach information it is
not entitled to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Calibrators.reliability_learner import ReliabilityLearner  # noqa: E402
from Experiments.reliability_experiment import (  # noqa: E402
    INNER_FIT_FRACTION,
    INNER_SPLIT_SEED,
    SHUFFLE_SEEDS,
    _class_centroids,
    _class_distance_matrix,
    build_arms,
    geometry_features,
    inner_split,
    recoverability_diagnostic,
    regular_features,
)


def _synthetic(n_train=600, n_val=300, n_test=400, c=6, hidden=32, seed=0):
    rng = np.random.default_rng(seed)
    train_y = rng.integers(0, c, n_train)
    centers = rng.normal(size=(c, hidden)) * 3.0
    mk = lambda y: centers[y] + rng.normal(size=(len(y), hidden))
    val_y = rng.integers(0, c, n_val)
    test_y = rng.integers(0, c, n_test)
    head = rng.normal(size=(hidden, c))
    feats = {"train": mk(train_y), "val": mk(val_y), "test": mk(test_y)}
    logits = {k: v @ head for k, v in feats.items()}
    sm = lambda z: np.exp(z - z.max(1, keepdims=True)) / np.exp(z - z.max(1, keepdims=True)).sum(1, keepdims=True)
    return {
        "train_features": feats["train"], "train_logits": logits["train"], "train_y": train_y,
        "val_features": feats["val"], "val_logits": logits["val"], "val_labels": val_y,
        "test_features": feats["test"], "test_logits": logits["test"], "test_labels": test_y,
        "base_probs_val": sm(logits["val"]), "base_probs_test": sm(logits["test"]),
        "per_sample_path": "synthetic",
    }


# ---------------------------------------------------------------- 1, 2, 12
def test_feature_names_always_match_feature_columns():
    """Guards the C < TOP_K case: names and columns must stay in lockstep."""
    rng = np.random.default_rng(0)
    for c in (3, 6, 10, 40):
        r, rn = regular_features(rng.normal(size=(25, c)))
        assert r.shape[1] == len(rn), f"regular block, C={c}"
        g, gn = geometry_features(rng.random((25, c)), rng.integers(0, c, 25), "geo")
        assert g.shape[1] == len(gn), f"geometry block, C={c}"
    built = build_arms(_synthetic(c=4))
    for name, arm in built["arms"].items():
        assert arm["val"].shape[1] == len(arm["names"]), name


def test_matched_arms_share_initial_parameters_and_param_count():
    data = _synthetic()
    built = build_arms(data)
    arms = built["arms"]
    learner = ReliabilityLearner(seed=123)

    widths = {n: a["val"].shape[1] for n, a in arms.items()}
    geometry_arms = [n for n in arms if n != "R"]
    assert len({widths[n] for n in geometry_arms}) == 1, (
        f"zero/shuffled/logit/hidden arms must share an input width: {widths}"
    )
    fingerprints = {n: learner.initial_parameter_fingerprint(widths[n]) for n in geometry_arms}
    assert len(set(fingerprints.values())) == 1, "matched arms must start identically"
    counts = {n: learner.parameter_count(widths[n]) for n in geometry_arms}
    assert len(set(counts.values())) == 1, f"parameter counts differ: {counts}"
    # R is narrower by exactly the geometry block, which is why R+zero exists.
    assert widths["R"] < widths["R_plus_G_H"]
    assert widths["R_plus_G_H"] - widths["R"] == len(built["geometry_names"])


def test_learner_is_deterministic_for_a_fixed_seed():
    data = _synthetic()
    arm = build_arms(data)["arms"]["R_plus_G_H"]
    correct = (np.argmax(data["base_probs_val"], 1) == data["val_labels"]).astype(float)
    fit_idx, sel_idx = inner_split(len(correct), INNER_SPLIT_SEED, INNER_FIT_FRACTION)
    outs = []
    for _ in range(2):
        m = ReliabilityLearner(seed=555, epochs=30)
        m.fit(arm["val"][fit_idx], correct[fit_idx], arm["val"][sel_idx], correct[sel_idx])
        outs.append(m.predict_confidence(arm["test"]))
    np.testing.assert_allclose(outs[0], outs[1])


# ------------------------------------------------------------------------ 3
def test_all_arms_use_the_same_fit_and_selection_indices():
    a = inner_split(500, INNER_SPLIT_SEED, INNER_FIT_FRACTION)
    b = inner_split(500, INNER_SPLIT_SEED, INNER_FIT_FRACTION)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
    assert set(a[0]).isdisjoint(set(a[1]))
    assert len(a[0]) + len(a[1]) == 500
    assert len(a[0]) == int(round(INNER_FIT_FRACTION * 500))


# ------------------------------------------------------------------------ 4
def test_target_is_always_base_model_correctness():
    import inspect

    from Experiments import reliability_experiment as rx

    src = inspect.getsource(rx.run_experiment)
    assert "correct_val = (val_pred == val_labels)" in src
    assert "correct_test = (test_pred == test_labels)" in src
    # No arm's own prediction may define the target.
    assert "np.argmax(conf" not in src
    for name, arm in build_arms(_synthetic())["arms"].items():
        assert arm["val"].shape[0] == 300 and arm["test"].shape[0] == 400


# ------------------------------------------------------------------------ 5
def test_regular_features_never_touch_internal_representation():
    import inspect

    from Experiments import reliability_experiment as rx

    src = inspect.getsource(rx.regular_features)
    body = src[src.index('"""', src.index('"""') + 3) + 3:]  # skip name + docstring
    for forbidden in ("features", "centroid", "distance", "train_"):
        assert forbidden not in body, f"regular_features must not reference {forbidden!r}"

    # Changing the hidden representation cannot change the R block.
    data = _synthetic()
    r1, _ = regular_features(data["val_logits"])
    data2 = dict(data)
    data2["val_features"] = data["val_features"] * 17.0 + 3.0
    r2, _ = regular_features(data2["val_logits"])
    np.testing.assert_allclose(r1, r2)

    arms = build_arms(data)["arms"]
    arms2 = build_arms(data2)["arms"]
    np.testing.assert_allclose(arms["R"]["val"], arms2["R"]["val"])
    assert not np.allclose(arms["R_plus_G_H"]["val"], arms2["R_plus_G_H"]["val"])


# ------------------------------------------------------------------------ 6
def test_zero_arm_carries_exactly_zero_geometry():
    built = build_arms(_synthetic())
    arms, n_geo = built["arms"], len(built["geometry_names"])
    for split in ("val", "test"):
        block = arms["R_plus_zero"][split][:, -n_geo:]
        assert np.all(block == 0.0)
        # ...and its regular half is byte-identical to the R arm's.
        np.testing.assert_allclose(
            arms["R_plus_zero"][split][:, :-n_geo], arms["R"][split]
        )


# ------------------------------------------------------------------------ 7
def test_shuffled_geometry_preserves_marginals_but_breaks_association():
    built = build_arms(_synthetic())
    arms, n_geo = built["arms"], len(built["geometry_names"])
    true_test = arms["R_plus_G_H"]["test"][:, -n_geo:]
    for seed in SHUFFLE_SEEDS:
        sh = arms[f"R_plus_shuffledG_H_s{seed}"]["test"][:, -n_geo:]
        assert sh.shape == true_test.shape
        # Identical multiset of rows (marginals, scale, dimensionality kept).
        np.testing.assert_allclose(np.sort(sh, axis=0), np.sort(true_test, axis=0))
        # Association destroyed.
        assert not np.allclose(sh, true_test)
        assert np.mean(np.all(np.isclose(sh, true_test), axis=1)) < 0.05
    # Distinct seeds give distinct permutations.
    a = arms[f"R_plus_shuffledG_H_s{SHUFFLE_SEEDS[0]}"]["test"][:, -n_geo:]
    b = arms[f"R_plus_shuffledG_H_s{SHUFFLE_SEEDS[1]}"]["test"][:, -n_geo:]
    assert not np.allclose(a, b)


def test_shuffle_permutation_does_not_depend_on_labels():
    """Permuting the labels must not change the permutation that is drawn."""
    data = _synthetic()
    tampered = dict(data)
    rng = np.random.default_rng(0)
    tampered["test_labels"] = rng.permutation(data["test_labels"])
    n_geo = len(build_arms(data)["geometry_names"])
    key = f"R_plus_shuffledG_H_s{SHUFFLE_SEEDS[0]}"
    np.testing.assert_allclose(
        build_arms(data)["arms"][key]["test"][:, -n_geo:],
        build_arms(tampered)["arms"][key]["test"][:, -n_geo:],
    )


# ------------------------------------------------------------------------ 8
def test_logit_geometry_never_touches_hidden_representation():
    data = _synthetic()
    built = build_arms(data)
    gz_val, gz_test = built["distance_matrices"]["G_Z"]

    scrambled = dict(data)
    rng = np.random.default_rng(3)
    for k in ("train_features", "val_features", "test_features"):
        scrambled[k] = rng.normal(size=data[k].shape) * 50.0
    built2 = build_arms(scrambled)
    np.testing.assert_allclose(gz_val, built2["distance_matrices"]["G_Z"][0])
    np.testing.assert_allclose(gz_test, built2["distance_matrices"]["G_Z"][1])
    np.testing.assert_allclose(
        built["arms"]["R_plus_G_Z"]["test"], built2["arms"]["R_plus_G_Z"]["test"]
    )
    # ...while the hidden arm does move.
    assert not np.allclose(
        built["arms"]["R_plus_G_H"]["test"], built2["arms"]["R_plus_G_H"]["test"]
    )


def test_hidden_and_logit_geometry_share_one_statistic_definition():
    """G_H and G_Z must differ only by the space they are computed in."""
    import inspect

    from Experiments import reliability_experiment as rx

    src = inspect.getsource(rx.build_arms)
    assert src.count("_class_distance_matrix(") >= 6
    assert "_class_centroids(data[\"train_features\"]" in src
    assert "_class_centroids(data[\"train_logits\"]" in src
    # Same summariser for every space -> identical feature width.
    built = build_arms(_synthetic())
    widths = {k: geometry_features(v[1], np.zeros(v[1].shape[0], dtype=int), "geo")[0].shape[1]
              for k, v in built["distance_matrices"].items()}
    assert len(set(widths.values())) == 1, widths


# ------------------------------------------------------------------------ 9
def test_reference_bank_uses_only_train_labels():
    """Centroids must come from the train split; test labels must not matter."""
    data = _synthetic()
    built = build_arms(data)
    tampered = dict(data)
    rng = np.random.default_rng(5)
    tampered["test_labels"] = rng.permutation(data["test_labels"])
    tampered["val_labels"] = rng.permutation(data["val_labels"])
    built2 = build_arms(tampered)
    for key in built["distance_matrices"]:
        for i in (0, 1):
            np.testing.assert_allclose(
                built["distance_matrices"][key][i], built2["distance_matrices"][key][i]
            )
    # Changing TRAIN labels does move the bank (so it is genuinely used).
    tampered2 = dict(data)
    tampered2["train_y"] = rng.permutation(data["train_y"])
    assert not np.allclose(
        built["distance_matrices"]["G_H"][1],
        build_arms(tampered2)["distance_matrices"]["G_H"][1],
    )


def test_class_centroids_reject_a_missing_class():
    feats = np.random.default_rng(0).normal(size=(20, 4))
    labels = np.zeros(20, dtype=int)
    with pytest.raises(ValueError, match="no examples of class 1"):
        _class_centroids(feats, labels, 3)


def test_class_distance_matrix_matches_direct_computation():
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(15, 7))
    cent = rng.normal(size=(4, 7))
    got = _class_distance_matrix(feats, cent)
    norm = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    expected = np.stack(
        [np.linalg.norm(norm - cent[c], axis=1) for c in range(4)], axis=1
    )
    np.testing.assert_allclose(got, expected, atol=1e-10)


# ----------------------------------------------------------------------- 10
def test_no_corruption_specific_fitting_anywhere_in_the_experiment():
    src = (Path(__file__).resolve().parents[1] / "Experiments" / "reliability_experiment.py").read_text()
    lowered = src.lower()
    for token in ("corruption_type", "cifar_c", "cifar-c", "severity"):
        assert token not in lowered.replace("corruption_specific_fitting", "").replace(
            "no corruption-specific fitting", ""
        ), f"{token!r} must not appear in the reliability experiment"


# ----------------------------------------------------------------------- 13
def test_recoverability_diagnostic_is_well_formed():
    rng = np.random.default_rng(0)
    n, c = 200, 5
    labels = rng.integers(0, c, n)
    base_pred = labels.copy()
    base_pred[:50] = (base_pred[:50] + 1) % c          # 50 forced errors
    d = rng.random((n, c)) + 1.0
    d[np.arange(n), labels] = 0.1                       # geometry always knows the truth
    out = recoverability_diagnostic(d, base_pred, labels)
    assert out["n_base_errors"] == 50
    assert out["geometry_recoverable_top1_rate"] == pytest.approx(1.0)
    assert out["median_gt_rank_on_errors"] == 0.0

    d2 = rng.random((n, c))                             # geometry knows nothing
    out2 = recoverability_diagnostic(d2, base_pred, labels)
    assert 0.0 <= out2["geometry_recoverable_top1_rate"] <= 0.5
    assert out2["geometry_recoverable_top5_rate"] == pytest.approx(1.0)  # c == 5
