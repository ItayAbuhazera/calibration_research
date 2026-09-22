"""Core properties of the representation-atlas program (docs/atlas_program_spec.md)."""
import ast, inspect, sys
from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atlas import spec, pooling, knn, stats  # noqa: E402


def test_sites_and_dims():
    assert len(spec.SITES) == 34 and spec.SITE_NAMES[0] == "stem" and spec.SITE_NAMES[-1] == "layer4.2"
    assert sum(1 for s in spec.SITE_NAMES if s.startswith("layer")) == 33
    assert spec.pool_dim("layer3.22", "gap") == 1024 and spec.pool_dim("layer3.22", "grid2") == 4096
    assert spec.pool_dim("layer3.22", "spp") == 1024 * 21 and spec.pool_dim("layer4.2", "spp") == 2048 * 21
    assert all(spec.valid_levels(s) == (1, 2, 4) for s in spec.SITE_SIZE.values())  # smallest map is 4x4
    assert spec.valid_levels(2) == (1, 2) and spec.valid_levels(1) == (1,)   # never request a grid larger than the map


def test_site_shapes_match_the_real_graph():
    from Net.resnet_cifar import resnet101
    from atlas import features
    m = resnet101(num_classes=100).eval()
    tap = features.SiteTap(m, spec.SITE_NAMES)
    with torch.no_grad():
        z, maps = features.forward_maps(m, tap, torch.randn(2, 3, 32, 32))
    for name, c, s in spec.SITES:
        assert tuple(maps[name].shape[1:]) == (c, s, s), name
        assert (maps[name] >= 0).all(), f"{name} is not post-activation"
    # the 'final pooled' representation equals the GAP of layer4.2 (marked control, not a distinct site)
    with torch.no_grad():
        final = F.avg_pool2d(maps["layer4.2"], 4).flatten(1)
    assert torch.allclose(final, pooling.pool(maps["layer4.2"], "gap"), atol=1e-6)


def test_groups_cover_every_site_once_within_budget():
    g = spec.plan_groups()
    flat = [s for grp in g for s in grp]
    assert flat == list(spec.SITE_NAMES)
    assert all(sum(spec.bank_bytes(s) for s in grp) <= spec.MEM_BUDGET_BYTES for grp in g)


def test_pooling_definitions_and_spp_weighting():
    x = torch.rand(3, 8, 8, 8)
    assert torch.allclose(pooling.pool(x, "gap"), x.mean((2, 3)))
    g2 = pooling.pool(x, "grid2")
    assert g2.shape == (3, 8 * 4) and torch.allclose(g2.reshape(3, 8, 2, 2)[:, :, 0, 0], x[:, :, :4, :4].mean((2, 3)))
    s = pooling.pool(x, "spp")
    assert s.shape == (3, 8 * 21)
    lv = [F.adaptive_avg_pool2d(x, (q, q)).reshape(3, -1) / q for q in (1, 2, 4)]
    assert torch.allclose(s, torch.cat(lv, 1) / 3 ** 0.5)
    sq = pooling.spp_level_sq_norms(x)
    assert torch.allclose(sq.sum(1) / 3, s.pow(2).sum(1), rtol=1e-5)
    # a feature map smaller than the largest level drops the invalid level
    y = torch.rand(2, 4, 2, 2)
    assert pooling.pool(y, "spp").shape == (2, 4 * (1 + 4))


def test_unit_and_zero_vectors():
    v, n = pooling.unit(torch.tensor([[3.0, 4.0], [0.0, 0.0]]))
    assert torch.allclose(v[0], torch.tensor([0.6, 0.8])) and float(n[1]) == 0.0 and torch.isfinite(v).all()


def test_knn_exact_matches_bruteforce_and_ties_break_by_index():
    torch.manual_seed(0)
    bank = F.normalize(torch.randn(500, 16), dim=1); q = F.normalize(torch.randn(7, 16), dim=1)
    d, i = knn.knn_unit(q, bank)
    ref = torch.cdist(q.double(), bank.double())
    ref_d, ref_i = torch.sort(ref, dim=1)
    assert torch.equal(i.cpu(), ref_i[:, :spec.K_NN]) and torch.allclose(d.double(), ref_d[:, :spec.K_NN], atol=1e-4)
    # duplicated bank rows => exactly equal distances => the smaller bank index must come first
    b2 = torch.cat([bank[:1], bank[:1], bank[:1], bank[1:]])
    _, i2 = knn.knn_unit(bank[:1], b2)
    assert i2[0, :3].tolist() == [0, 1, 2]


def test_raw_l2_is_norm_preserving_and_not_unit_l2():
    torch.manual_seed(1)
    bank = torch.randn(300, 10) * torch.rand(300, 1) * 5; q = torch.randn(4, 10) * 3
    d, i = knn.knn_raw(q, bank, bank.pow(2).sum(1))
    ref = torch.cdist(q, bank)
    assert torch.equal(i, torch.sort(ref, dim=1)[1][:, :spec.K_NN]) and torch.allclose(d, torch.sort(ref, 1)[0][:, :spec.K_NN], atol=1e-3)


def test_mahalanobis_lowrank_matches_dense_formula():
    torch.manual_seed(2)
    d, n, r = 12, 400, 5
    A = torch.randn(n, d) @ torch.randn(d, d) * 0.5
    m = knn.LowRankMahalanobis.fit(A.clone(), rank=r, niter=10)
    # dense reference with the SAME decomposition
    mu = A.mean(0); cov = torch.cov((A - mu).T)
    w, V = torch.linalg.eigh(cov); U = V[:, -r:].flip(1)
    Ur = m.U
    Sigma = m.U @ torch.diag(m.e_r) @ m.U.T + m.e_res * (torch.eye(d) - m.U @ m.U.T)
    q = torch.randn(3, d)
    pb = m.prepare_bank(A.clone())
    dist, idx = m.knn(q, pb)
    ref = torch.stack([torch.sqrt(((A - qi) @ torch.linalg.inv(Sigma) * (A - qi)).sum(1)) for qi in q])
    assert torch.allclose(dist[:, :10], torch.sort(ref, 1)[0][:, :10], rtol=1e-3, atol=1e-3)
    assert m.info["shrinkage"] == spec.MAH_SHRINK and m.e_res > 0
    # top eigenvalues agree with the exact ones (randomized SVD audit)
    assert torch.allclose(m.eig_r[:3], w[-3:].flip(0), rtol=1e-2)


def test_pgeo_smoothing_prior_and_tie_break():
    labels = np.array([0, 0, 1, 2, 2, 2]); prior = np.bincount(labels, minlength=4) / len(labels)
    idx = np.array([[0, 1, 2, 3]])
    p, c = stats.p_geo(idx, labels, prior, alpha=1.0)
    np.testing.assert_allclose(p.sum(1), 1.0)
    np.testing.assert_allclose(p[0], (np.array([2, 1, 1, 0]) + prior) / 5)
    assert p.argmax(1)[0] == 0
    tie = np.array([[0, 2]]); pt, _ = stats.p_geo(tie, labels, np.full(4, .25), 1.0)
    assert pt.argmax(1)[0] == 0   # lowest class index wins the tie


def test_wh_identities_and_undefined_ratios():
    rng = np.random.default_rng(0); y = rng.integers(0, 5, 400); base = np.where(rng.random(400) < .6, y, rng.integers(0, 5, 400))
    cand = np.where(rng.random(400) < .5, base, rng.integers(0, 5, 400))
    r = stats.wh_stats(base, cand, y)
    assert r["W"] + r["H"] + r["U"] == r["total_flips"] and abs(r["net_utility"] - (np.mean(cand == y) - np.mean(base == y))) < 1e-12
    assert r["decisive_precision"] == r["W"] / (r["W"] + r["H"]) and r["intervention_precision"] == r["W"] / r["total_flips"]
    r0 = stats.wh_stats(base, base, y)
    assert r0["total_flips"] == 0 and r0["intervention_precision"] is None and r0["decisive_precision"] is None   # undefined, not 0


def test_gt_rank_is_deterministic_with_ties():
    p = np.array([[0.25, 0.25, 0.25, 0.25]]); assert stats.gt_rank(p, np.array([2]))[0] == 3
    assert stats.brier(np.eye(3), np.arange(3)) == 0.0
