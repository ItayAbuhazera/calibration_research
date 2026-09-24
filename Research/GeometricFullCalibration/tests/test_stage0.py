"""Structural/leakage properties for Stage 0 (docs/stage0_execution_spec.md).

House style follows tests/test_layer_selection_pilot.py: one assertion per named invariant,
synthetic data where a real forward pass isn't needed, real cached artifacts only where the
property is genuinely about those artifacts.
"""
import numpy as np

from atlas import stage0_data, stage0_fit, stage0_folds


def test_duplicate_groups_never_split_across_outer_folds():
    plan = stage0_folds.load_plan()
    fold_id = np.array(plan["fold_id"])
    for entry in stage0_data.test_duplicate_groups().values():
        idxs = entry["indices"]
        assert len({fold_id[i] for i in idxs}) == 1


def test_duplicate_groups_never_split_across_inner_split_or_nested_subset():
    plan = stage0_folds.load_plan()
    gid_full = np.array(plan["duplicate_group_id"])
    for o in plan["outer"]:
        train_idx = np.array(o["train_idx"])
        inner_fit_mask = np.array(o["inner_fit_mask"], dtype=bool)
        groups_here = gid_full[train_idx]
        for g in np.unique(groups_here):
            members = inner_fit_mask[groups_here == g]
            assert len(set(members.tolist())) == 1, f"group {g} split across inner-fit/inner-val in fold {o['fold']}"
        nested_local = np.array(o["nested_2500_local_positions"])
        nested_groups = set(groups_here[nested_local].tolist())
        for g in nested_groups:
            all_positions_of_g = np.where(groups_here == g)[0]
            in_nested = np.isin(all_positions_of_g, nested_local)
            assert in_nested.all() or (~in_nested).all(), f"group {g} split across the nested-2500 boundary in fold {o['fold']}"


def test_outer_folds_partition_all_10000_images_exactly_once():
    plan = stage0_folds.load_plan()
    fold_id = np.array(plan["fold_id"])
    assert fold_id.shape == (10000,)
    assert (fold_id >= 0).all() and (fold_id < plan["n_outer"]).all()
    for o in plan["outer"]:
        assert set(o["train_idx"]) | set(o["test_idx"]) == set(range(10000))
        assert set(o["train_idx"]) & set(o["test_idx"]) == set()


def test_b_zero_reproduces_q_z():
    rng = np.random.default_rng(0)
    n, k = 50, 5
    Xz = rng.normal(size=(n, k))
    mean, std = stage0_fit.standardize_fit(Xz)
    A = rng.normal(size=(k, k)) * 0.3
    b = rng.normal(size=(k,)) * 0.1
    q_z = stage0_fit.predict_probs(Xz, mean, std, A, b)

    B = np.zeros_like(A)
    Xp = rng.normal(size=(n, k))  # arbitrary P -- must not matter when B=0
    W_check = np.concatenate([A, B], axis=0)
    mean_c = np.concatenate([mean, np.zeros(k)])
    std_c = np.concatenate([std, np.ones(k)])
    Xzp = np.concatenate([Xz, Xp], axis=1)
    q_zp = stage0_fit.predict_probs(Xzp, mean_c, std_c, W_check, b)

    assert np.abs(q_z - q_zp).max() < 1e-12


def test_lambda_tie_break_prefers_larger_lambda_then_grid_order():
    table = [
        {"lambda": 1e-1, "val_nll": 1.000000001},
        {"lambda": 1e-2, "val_nll": 1.000000002},  # tied within 1e-9? no: diff=1e-9 exactly at boundary
        {"lambda": 1e-3, "val_nll": 1.0000000015},
        {"lambda": 1e-4, "val_nll": 2.0},
        {"lambda": 1e-5, "val_nll": 2.0},
    ]
    best = stage0_fit.select_from_table(table, tie_eps=1e-8)
    assert best["lambda"] == 1e-1  # among the ties, the largest lambda wins

    table2 = [{"lambda": 1e-1, "val_nll": 5.0}, {"lambda": 1e-2, "val_nll": 1.0}]
    assert stage0_fit.select_from_table(table2)["lambda"] == 1e-2  # unambiguous min, no tie


def test_scaler_for_final_refit_depends_only_on_full_training_rows_not_inner_split():
    rng = np.random.default_rng(1)
    n, k, c = 60, 4, 3
    X_full = rng.normal(size=(n, k))
    y_full = rng.integers(0, c, n)
    X_if, y_if = X_full[:40], y_full[:40]
    X_iv, y_iv = X_full[40:], y_full[40:]

    out_a = stage0_fit.fit_arm(X_full, y_full, X_if, y_if, X_iv, y_iv, c, lambda_grid=(1e-1,))
    # perturbing the inner-val FEATURES (not X_full) must not change the final scaler
    X_iv_perturbed = X_iv + rng.normal(size=X_iv.shape) * 100
    out_b = stage0_fit.fit_arm(X_full, y_full, X_if, y_if, X_iv_perturbed, y_iv, c, lambda_grid=(1e-1,))
    assert np.allclose(out_a["scaler_mean"], out_b["scaler_mean"])
    assert np.allclose(out_a["scaler_std"], out_b["scaler_std"])
    expected_mean, expected_std = stage0_fit.standardize_fit(X_full)
    assert np.allclose(out_a["scaler_mean"], expected_mean)
    assert np.allclose(out_a["scaler_std"], expected_std)


def test_zero_variance_coordinate_does_not_produce_nan():
    rng = np.random.default_rng(2)
    n, k = 30, 4
    X = rng.normal(size=(n, k))
    X[:, 0] = 7.0  # constant column
    mean, std = stage0_fit.standardize_fit(X)
    assert std[0] == 1.0  # zero variance substituted, not divided by zero
    Xt = stage0_fit.standardize_apply(X, mean, std)
    assert np.allclose(Xt[:, 0], 0.0)
    assert not np.isnan(Xt).any()


def test_deterministic_argmax_tie_goes_to_smaller_class_index():
    probs = np.array([[0.5, 0.5, 0.0]])
    assert probs.argmax(1)[0] == 0


def test_accuracy_identity_W_minus_H_over_N():
    from Calibrators import layer_readouts as LR
    rng = np.random.default_rng(3)
    n, c = 500, 10
    labels = rng.integers(0, c, n)
    base_probs = np.eye(c)[rng.integers(0, c, n)]
    new_probs = np.eye(c)[rng.integers(0, c, n)]
    flips = LR.flip_decomposition(base_probs, new_probs, labels)
    acc_base = (base_probs.argmax(1) == labels).mean()
    acc_new = (new_probs.argmax(1) == labels).mean()
    assert abs((acc_new - acc_base) - flips["delta_acc_from_flips"]) < 1e-12



def test_evidence_default_is_layer322_and_variants_select_the_declared_source():
    import numpy as np
    seed, cell = 2, "fog_s3"
    d0 = stage0_data.load_cell(seed, cell)
    d8 = stage0_data.load_cell(seed, cell, "L8")
    d11 = stage0_data.load_cell(seed, cell, "L11")
    dx = stage0_data.load_cell(seed, cell, "xckpt")
    raw = np.load(f"results/layer_pilot/checkpoint_seed{seed}/{cell}/per_sample.npz")["raw__probe_logits"]
    other = np.load(f"results/atlas/seed4/u0/{cell}.npz")["logits"].astype(np.float64)
    assert np.array_equal(d0["p"], d8["p"]) and np.array_equal(d0["p"], raw[:, 8, :].astype(np.float64))
    assert np.array_equal(d11["p"], raw[:, 11, :].astype(np.float64))
    assert np.array_equal(dx["p"], other)
    assert np.array_equal(d0["z"], d11["z"]) and np.array_equal(d0["z"], dx["z"])
    assert np.array_equal(d0["labels"], dx["labels"])


def test_regime_map_rules_follow_the_frozen_v2_table():
    from atlas.regime_aggregate import evaluate_rules, combine_seeds
    mk = lambda gap, lo, ci, cl: {"gap8": gap, "gap8_ci": [lo, gap + 0.3], "clean_inc": ci, "clean_inc_ci": [cl, ci + 0.3]}
    base = {"a": mk(0.5, 0.2, 2.0, 1.5), "b1": mk(1.5, 1.0, 1.0, 0.5), "b3": mk(2.2, 1.7, 0.4, 0.1), "b10": mk(3.0, 2.6, 0.0, -0.3)}
    assert evaluate_rules(base, [1.9, 3.0])["row"] == 5                                                  # supports
    bad = dict(base, a=mk(0.5, 0.2, 0.2, 0.1)); assert evaluate_rules(bad, [1.9, 3.0])["row"] == 1      # manipulation check 1
    bad = dict(base, a=mk(0.5, 0.2, 2.0, -0.1)); assert evaluate_rules(bad, [1.9, 3.0])["row"] == 1
    bad = dict(base, b10=mk(3.0, 2.6, 0.5, 0.2)); assert evaluate_rules(bad, [1.9, 3.0])["row"] == 2    # manipulation check 2 (b10 clean increment > +0.3)
    bad = dict(base, b10=mk(0.9, 0.5, 0.0, -0.3)); assert evaluate_rules(bad, [0.0, 1.0])["row"] == 3   # final gap < 1.0
    bad = dict(base, b10={"gap8": 3.0, "gap8_ci": [-0.1, 3.3], "clean_inc": 0.0, "clean_inc_ci": [-0.3, 0.3]}); assert evaluate_rules(bad, [1.9, 3.0])["row"] == 3
    bad = dict(base, a=mk(2.5, 2.0, 2.0, 1.5)); assert evaluate_rules(bad, [0.0, 1.0])["row"] == 4      # gap(a) >= 0.8 gap(b)
    assert evaluate_rules(dict(base, a=mk(1.8, 1.4, 2.0, 1.5)), [0.2, 2.0])["row"] == 6                  # between 0.5 and 0.8
    bad = dict(base, b3=mk(0.2, 0.0, 0.4, 0.1)); r = evaluate_rules(bad, [1.9, 3.0]); assert r["row"] == 6 and not r["dose_condition_ok"]
    assert evaluate_rules(base, [-0.2, 3.0])["row"] == 6                                                 # difference interval includes 0
    s1, s2 = evaluate_rules(base, [1.9, 3.0]), evaluate_rules(dict(base, b3=mk(0.2, 0.0, 0.4, 0.1)), [1.9, 3.0])
    assert combine_seeds({"seed1": s1, "seed2": s1})["row"] == 5 and combine_seeds({"seed1": s1, "seed2": s2})["row"] is None


def test_grid_edge_rule_extends_up_to_two_decades_toward_the_edge():
    import numpy as np, torch
    from atlas import regime_map as RM
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(size=(300, 20)), dtype=torch.float32); y = rng.integers(0, 3, 300)
    old, old_dt = RM.NC, torch.get_default_dtype(); RM.NC = 3; torch.set_default_dtype(torch.float32)   # stage0_fit sets float64 globally
    try:
        pr, info = RM.fit_linear(x, y, x[:100], y[:100], (1e-4, 1e-3, 1e-2), device="cpu")   # noise features: the best lambda is large -> extend upward
    finally:
        RM.NC = old; torch.set_default_dtype(old_dt)
    assert info["extensions_up"] <= 2 and info["extensions_down"] <= 2
    assert set(map(float, info["inner_fit_nll_by_lambda"])) >= {1e-4, 1e-3, 1e-2}
    assert info["selected_lambda"] >= 1e-2 and info["extensions_up"] >= 1
