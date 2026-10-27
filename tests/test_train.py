"""Tests for the Stage B loss and per-body weights (gnn/train.py)."""

import pytest
import torch

from gnn.split import load_split
from gnn.train import F, relative_error, weighted_graphs, weighted_loss


def test_zero_prediction_gives_relative_error_one():
    y = torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 10.0], [1e-3, 0.0, 0.0]])  # |a| = 5, 10, and 0.001 (below F)
    err = relative_error(torch.zeros_like(y), y)  # the "predict zero" baseline
    assert torch.allclose(err[:2], torch.ones(2))  # above F: exactly 1, whatever the size of a
    assert err[2].item() == pytest.approx(1e-3 / F)  # below F: |a| / F, so less than 1


def test_perfect_prediction_gives_zero_loss():
    y = torch.randn(5, 3)  # 5 random accelerations
    assert weighted_loss(y, y, torch.rand(5)).item() == 0  # predicting the truth costs nothing


def test_loss_unchanged_when_all_weights_scaled():
    torch.manual_seed(0)
    pred, y, w = torch.randn(5, 3), torch.randn(5, 3), torch.rand(5)
    assert weighted_loss(pred, y, w).item() == pytest.approx(weighted_loss(pred, y, 100 * w).item())  # dividing by the total weight cancels the 100


@pytest.mark.parametrize("scheme", ["kick", "time"])
def test_weighted_graphs(scheme):
    graphs = weighted_graphs("val", scheme)
    n_files = len(load_split()["val"])  # 36 validation files
    assert sum(g.w.sum() for g in graphs).item() == pytest.approx(n_files, rel=1e-4)  # each file sums to 1
    for g in graphs:
        assert g.w.shape == (g.y.shape[0],)  # one weight per body
        assert (g.w > 0).all()  # no negative or zero weights
        if scheme == "time":
            assert torch.allclose(g.w, g.w[0].expand_as(g.w))  # time: every body in a row gets the same weight
        else:
            ratio = g.w / g.y.norm(dim=-1)  # kick: weight / |a| should be the same for every body in a row
            assert torch.allclose(ratio, ratio[0].expand_as(ratio))



