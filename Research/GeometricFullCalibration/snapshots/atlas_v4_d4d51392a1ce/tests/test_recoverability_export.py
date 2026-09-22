from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, TensorDataset

from Experiments.run_rgc_experiments import extract_and_aggregate_sgc_features
from scripts.export_recoverability import (
    K_VOTE,
    _assert_arm_equivalence,
    _global_neighbour_fields,
    _join_stage,
)


def test_global_vote_uses_mean_distance_tie_break_and_laplace_smoothing():
    assert K_VOTE == 200
    # Querying all 200 references yields a 100/100 vote tie. Class 1 has the
    # smaller mean distance and must win despite its higher class index.
    train = np.concatenate(
        [np.full((100, 2), 2.0), np.full((100, 2), 1.0)], axis=0
    )
    labels = np.concatenate(
        [np.zeros(100, dtype=np.int64), np.ones(100, dtype=np.int64)]
    )
    index = NearestNeighbors(n_neighbors=200, metric="l2").fit(train)
    result = _global_neighbour_fields(
        index,
        np.zeros((1, 2)),
        np.arange(200, dtype=np.int64),
        labels,
    )

    assert result["y_pred_globalknn"].tolist() == [1]
    assert result["knn_counts"][0, :2].tolist() == [100, 100]
    assert result["tie_stats"]["fraction_top1_count_tied_before_distance_break"] == 1.0
    expected = np.full(100, 1.0 / 300.0)
    expected[:2] = 101.0 / 300.0
    np.testing.assert_allclose(result["knn_distribution"][0], expected)
    assert result["neighbour_margin"].tolist() == [0.0]


def test_join_stage_realigns_payload_by_carried_ids():
    master = np.array([10, 20, 30])
    ids = np.array([30, 10, 20])
    joined_ids, payload = _join_stage(master, ids, {"value": np.array([3, 1, 2])})
    np.testing.assert_array_equal(joined_ids, master)
    np.testing.assert_array_equal(payload["value"], [1, 2, 3])


def test_arm_equivalence_allows_only_prediction_and_probability_omission():
    common = {"sample_id": np.arange(3), "x": np.arange(3)}
    perclass = {**common, "y_pred_knn": np.array([0, 1, 2])}
    globalknn = {
        **common,
        "y_pred_knn": np.array([2, 1, 0]),
        "head_probabilities": np.ones((3, 2)) / 2,
        "knn_distribution": np.ones((3, 2)) / 2,
    }
    _assert_arm_equivalence(perclass, globalknn, "toy")
    with pytest.raises(RuntimeError, match="shared arm field"):
        _assert_arm_equivalence(
            perclass,
            {**globalknn, "x": np.array([9, 9, 9])},
            "toy",
        )


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(1, 2, kernel_size=1, bias=False)
        self.fc = torch.nn.Linear(2, 2, bias=False)

    def forward(self, x):
        feature = self.conv(x)
        return self.fc(feature.mean(dim=(2, 3)))


def test_observer_does_not_change_rgcl_embedding():
    torch.manual_seed(7)
    model = _TinyModel().eval()
    images = torch.arange(4 * 4 * 4, dtype=torch.float32).reshape(4, 1, 4, 4)
    labels = torch.tensor([0, 1, 0, 1])
    ids = torch.arange(4)
    plain_loader = DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)
    indexed_loader = DataLoader(
        TensorDataset(images, labels, ids), batch_size=2, shuffle=False
    )

    _, _, plain, _ = extract_and_aggregate_sgc_features(
        model,
        ["conv"],
        None,
        None,
        plain_loader,
        torch.device("cpu"),
        target_dim=3,
        seed=11,
    )
    seen = []

    def observer(split_name, batch, logits, activations):
        seen.extend(batch[2].tolist())
        assert split_name == "test"
        assert logits.shape[1] == 2
        assert "conv" in activations

    _, _, observed, _ = extract_and_aggregate_sgc_features(
        model,
        ["conv"],
        None,
        None,
        indexed_loader,
        torch.device("cpu"),
        target_dim=3,
        seed=11,
        batch_observer=observer,
        observer_layer_names=["conv"],
    )
    np.testing.assert_array_equal(observed, plain)
    assert seen == [0, 1, 2, 3]

