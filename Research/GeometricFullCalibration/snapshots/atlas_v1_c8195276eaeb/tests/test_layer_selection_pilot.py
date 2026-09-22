"""Properties of the layer-selection pilot (docs/layer_selection_pilot_spec.md §12)."""

from __future__ import annotations

import ast
import inspect
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from Calibrators import layer_readouts as LR  # noqa: E402
from Calibrators.full_vector_dac import np_softmax  # noqa: E402
from Experiments import layer_selection_pilot as P  # noqa: E402


# ------------------------------------------------------------ candidates
@pytest.fixture(scope="module")
def resnet():
    from Net.resnet_cifar import resnet101

    m = resnet101(num_classes=100).eval()
    return m


def test_candidates_are_distinct_residual_block_outputs_and_not_logits():
    names = LR.CANDIDATE_NAMES
    assert len(names) == len(set(names)) >= 8
    assert not any("fc" in n or "logit" in n for n in names)
    for n in names:  # only Bottleneck blocks, `layerS.B`
        stage, block = n.split(".")
        assert stage in {"layer1", "layer2", "layer3", "layer4"} and block.isdigit()
    dims = [d for _, d in LR.CANDIDATE_LAYERS]
    assert dims == sorted(dims)  # deeper => never fewer channels in ResNet-101


def test_candidate_modules_and_shapes_exist_in_the_real_graph(resnet):
    mods = dict(resnet.named_modules())
    shapes = {}

    def mk(n):
        def h(_m, _i, out):
            shapes[n] = tuple(out.shape[1:])

        return h

    hs = [mods[n].register_forward_hook(mk(n)) for n in LR.CANDIDATE_NAMES]
    with torch.no_grad():
        resnet(torch.zeros(1, 3, 32, 32))
    for h in hs:
        h.remove()
    for n, d in LR.CANDIDATE_LAYERS:
        assert shapes[n][0] == d, (n, shapes[n])


def test_native_dac_layers_alias_candidates_exactly(resnet):
    """`layerS` (Sequential) output IS its last block's output: no duplicated
    source, and the alias map used to derive S_DAC is right."""
    mods = dict(resnet.named_modules())
    out = {}

    def mk(n):
        def h(_m, _i, o):
            out[n] = o.detach().clone()

        return h

    names = ["layer1", "layer2", "layer3", "layer4"] + [LR.CANDIDATE_NAMES[i] for i in LR.NATIVE_ALIAS_TO_CANDIDATE.values()]
    hs = [mods[n].register_forward_hook(mk(n)) for n in set(names)]
    torch.manual_seed(0)
    with torch.no_grad():
        resnet(torch.randn(2, 3, 32, 32))
    for h in hs:
        h.remove()
    for native, ci in LR.NATIVE_ALIAS_TO_CANDIDATE.items():
        assert torch.equal(out[native], out[LR.CANDIDATE_NAMES[ci]])


def test_feature_tap_is_gap_then_l2(resnet):
    tap = P.FeatureTap(resnet)
    torch.manual_seed(1)
    with torch.no_grad():
        z, f = P.forward_pass(resnet, tap, torch.randn(3, 3, 32, 32))
    tap.close()
    assert z.shape == (3, 100)
    for n, d in list(LR.CANDIDATE_LAYERS) + [("conv1", 64)]:
        assert f[n].shape == (3, d)
        np.testing.assert_allclose(f[n].norm(dim=1).numpy(), 1.0, atol=1e-5)


def test_depth_spaced_sets_are_as_declared():
    assert LR.depth_spaced_indices(1) == [11]
    assert LR.depth_spaced_indices(4) == [0, 4, 7, 11]
    assert LR.depth_spaced_indices(6) == [0, 2, 4, 7, 9, 11]
    assert LR.depth_spaced_indices(8) == [0, 2, 3, 5, 6, 8, 9, 11]
    for L in LR.LAYER_COUNTS:
        assert len(set(LR.depth_spaced_indices(L))) == L


# ------------------------------------------------------------- Family A
def _synth(n=40, c=6, layers=5, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, c))
    r = rng.uniform(0.05, 0.6, size=(n, layers, c))
    s = rng.uniform(0.3, 1.5, size=n)
    y = rng.integers(0, c, size=n)
    return z, r, s, y


def test_beta_zero_is_native_dac_and_aggregation_is_plain_mean():
    z, r, s, _ = _synth()
    ref = np_softmax(z / s[:, None])
    np.testing.assert_allclose(LR.family_a_probs(z, r, [0, 2, 4], 0.0, s), ref, atol=1e-15)
    np.testing.assert_allclose(LR.mean_correction(r, [1, 3]), (r[:, 1] + r[:, 3]) / 2)
    # duplicated / reordered layer sets do not change an equal-weight mean of a set
    np.testing.assert_allclose(LR.mean_correction(r, [3, 1]), LR.mean_correction(r, [1, 3]))


def test_pairwise_decision_condition_and_scalar_temperature_cannot_flip():
    """j beats i iff beta(R_i - R_j) > z_i - z_j; S_DAC alone cannot change argmax."""
    rng = np.random.default_rng(3)
    z, r, s, _ = _synth(n=200, c=5, layers=3, seed=3)
    R = LR.mean_correction(r, [0, 1, 2])
    base = z.argmax(1)
    q0 = LR.family_a_probs(z, r, [0, 1, 2], 0.0, s)
    assert np.array_equal(q0.argmax(1), base)  # scalar temperature: argmax invariant
    for beta in (0.5, 3.0, 40.0):
        q = LR.family_a_probs(z, r, [0, 1, 2], beta, s)
        i = base
        for row in rng.integers(0, len(z), 50):
            j = q[row].argmax()
            if j != i[row]:
                assert beta * (R[row, i[row]] - R[row, j]) > z[row, i[row]] - z[row, j] - 1e-12
        # brute force
        assert np.array_equal(q.argmax(1), (z - beta * R).argmax(1))


def test_fit_family_a_uses_nll_only_and_is_deterministic():
    z, r, s, y = _synth(n=300, c=6, seed=5)
    a = LR.fit_family_a(z, r, [0, 1], s, y)
    b = LR.fit_family_a(z, r, [0, 1], s, y)
    assert a == b and a["objective"] == "nll" and a["beta"] >= 0
    fn = ast.parse(inspect.getsource(LR.fit_family_a)).body[0]
    fn.body = fn.body[1:]  # drop the docstring, which may DISCUSS accuracy
    code = ast.unparse(fn)
    for forbidden in ("accuracy", "flip", "argmax"):
        assert forbidden not in code


def test_class_conditional_distance_matches_bruteforce():
    rng = np.random.default_rng(0)
    m, d, c = 400, 8, 4
    bank = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(m, d)), dtype=torch.float32), dim=1)
    labels = np.repeat(np.arange(c), m // c)
    perm = LR.permuted_labels(labels, seed=7)
    assert np.array_equal(np.bincount(perm), np.bincount(labels))  # counts preserved
    assert np.array_equal(perm, LR.permuted_labels(labels, seed=7))  # deterministic
    q = torch.nn.functional.normalize(torch.tensor(rng.normal(size=(5, d)), dtype=torch.float32), dim=1)
    # the class table is hard-wired to NUM_CLASSES=100 in the runner; emulate with c classes
    P.NUM_CLASSES, saved = c, P.NUM_CLASSES
    try:
        lb = P.LayerBank(bank, labels, perm, torch.device("cpu"))
        r_t, r_p, g = lb.query(q, global_k=10)
    finally:
        P.NUM_CLASSES = saved
    dist = torch.sqrt(torch.clamp(2 - 2 * q @ bank.T, min=0)).numpy()
    for k in range(c):
        expect = np.sort(dist[:, labels == k], axis=1)[:, LR.K_C - 1]
        np.testing.assert_allclose(r_t[:, k], expect, atol=1e-5)
        expect_p = np.sort(dist[:, perm == k], axis=1)[:, LR.K_C - 1]
        np.testing.assert_allclose(r_p[:, k], expect_p, atol=1e-5)
    np.testing.assert_allclose(g, np.sort(dist, axis=1)[:, 9], atol=1e-5)


# --------------------------------------------------------------- selection
def test_greedy_is_nested_deterministic_forced_and_traces_rejected():
    rng = np.random.default_rng(1)
    gains = rng.normal(size=12)

    def score(S):  # additive + a penalty growing with |S| so later steps WORSEN the score
        return 2.0 - sum(gains[list(S)]) * 0.1 + 0.05 * len(S) ** 2

    a = LR.greedy_forward(score, 12, 8)
    b = LR.greedy_forward(score, 12, 8)
    assert a == b
    order = a["order"]
    assert len(order) == len(set(order)) == 8
    sets = LR.nested_sets(order)
    for small, big in ((1, 4), (4, 6), (6, 8)):
        assert set(sets[small]) < set(sets[big])
    # every step records every remaining candidate, including rejected ones
    for t, step in enumerate(a["trace"], start=1):
        assert len(step["candidate_scores"]) == 12 - (t - 1)
        assert step["score"] == min(step["candidate_scores"].values())
    # growth is forced even when the score worsens
    assert any(s["score_change_vs_previous_step"] > 0 for s in a["trace"][1:])


def test_greedy_ties_go_to_smaller_index():
    a = LR.greedy_forward(lambda S: 1.0, 5, 3)
    assert a["order"] == [0, 1, 2]


# --------------------------------------------------------------- Family B
def test_probe_learns_and_param_counts():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    c, d, n = 5, 12, 400
    y = torch.tensor(rng.integers(0, c, size=n))
    means = torch.randn(c, d) * 3
    x = means[y] + torch.randn(n, d)
    p = LR.fit_probe(x, y, c, lam=1e-3)
    acc = (p.logits(x).argmax(1) == y).float().mean().item()
    assert acc > 0.95
    assert p.n_params == d * c + c + 1
    t = LR.fit_probe_temperature(p.logits(x).numpy(), y.numpy())
    assert t > 0


def test_family_b_average_is_a_probability_vector_and_equal_weight():
    rng = np.random.default_rng(2)
    lg = rng.normal(size=(30, 4, 7))
    temps = [0.7, 1.0, 1.5, 2.0]
    q = LR.family_b_probs(lg, temps, [0, 2, 3])
    np.testing.assert_allclose(q.sum(1), 1.0, atol=1e-12)
    assert (q >= 0).all()
    manual = sum(np_softmax(lg[:, l] / temps[l]) for l in (0, 2, 3)) / 3
    np.testing.assert_allclose(q, manual)


# ------------------------------------------------------ decision accounting
def test_flip_decomposition_identity_and_margin_matched_enrichment():
    rng = np.random.default_rng(4)
    n, c = 3000, 10
    y = rng.integers(0, c, n)
    base = np_softmax(rng.normal(size=(n, c)) * 2)
    # random re-ranking: flips carry no information about base errors beyond margin
    other = np_softmax(rng.normal(size=(n, c)) * 2)
    d = LR.flip_decomposition(base, other, y)
    assert d["W"] + d["H"] + d["U"] == d["flips"]
    assert abs(d["delta_acc_from_flips"] - (np.mean(other.argmax(1) == y) - np.mean(base.argmax(1) == y))) < 1e-12
    e = LR.margin_matched_enrichment(base, other, y)
    assert 0.8 < e["enrichment_ratio"] < 1.2  # ~1: explained by margin/prevalence, not detection


def test_gt_rank_bounds():
    p = np.array([[0.1, 0.6, 0.3], [0.5, 0.3, 0.2]])
    np.testing.assert_array_equal(LR.gt_rank(p, np.array([1, 2])), [1, 3])


# ----------------------------------------------- protocol / no-leak structure
def _fn_source(name):
    return inspect.getsource(getattr(P, name))


def test_evaluation_path_cannot_reach_fitting():
    for fn in ("evaluate_cell", "arm_probs", "per_sample_record"):
        tree = ast.parse(_fn_source(fn))
        called = {
            n.func.attr if isinstance(n.func, ast.Attribute) else getattr(n.func, "id", "")
            for n in ast.walk(tree)
            if isinstance(n, ast.Call)
        }
        forbidden = {c for c in called if c.startswith("fit_") or c in {"greedy_forward", "minimize", "minimize_scalar", "step"}}
        assert not forbidden, f"{fn} reaches fitting: {forbidden}"


def test_corruption_data_is_loaded_only_after_the_frozen_state_is_written():
    src = inspect.getsource(P.main)
    write = src.index('"frozen_state.json"')
    assert write < src.index("load_corrupted(")
    assert src.index("run_selection(") < write
    # the FIT phase never touches a corruption cell
    fit_part = src[: src.index("EVALUATE")]
    assert "load_corrupted" not in fit_part and "cifar_c_dir" not in fit_part.split("ap.add_argument")[-1]


def test_beta_zero_assertion_runs_per_cell_in_evaluate_cell():
    assert "beta0_dev < 1e-12" in _fn_source("evaluate_cell")


def test_inherited_constants_match_the_frozen_spec():
    assert LR.K_C == 5 and LR.K_DAC == 200 and LR.LAYER_COUNTS == (1, 4, 6, 8)
    assert LR.PROBE_LAMBDAS == (1e-4, 1e-3, 1e-2)
    assert P.INNER_SEED == 123 and P.CORRUPTIONS == ("gaussian_noise", "defocus_blur", "fog", "jpeg_compression")
    assert P.SEVERITIES == (1, 3, 5)
