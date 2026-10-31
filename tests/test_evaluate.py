"""Tests for the Stage B evaluation (gnn/evaluate.py)."""

import math
import os

import torch
from torch_geometric.data import Batch, Data

from gnn.evaluate import EPS, com_positions, conservation, law_fit, per_decade, per_graph_sum
from gnn.graph_data import complete_edge_index, file_to_graphs
from gnn.split import DATA_DIR


def snapshot(masses, positions):
    """A hand-made snapshot in the same format as the real graphs. Returns (graph, positions)."""
    x = torch.tensor(masses).float()[:, None]  # (N, 1) masses
    pos = torch.tensor(positions).float()  # (N, 3) positions
    edge_index = complete_edge_index(len(masses))  # every ordered pair
    send, recv = edge_index
    disp = pos[send] - pos[recv]  # (E, 3) r_sender - r_receiver
    edge_attr = torch.cat([disp, disp.norm(dim=-1, keepdim=True)], dim=-1)  # (E, 4) disp then dist
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr), pos


def test_com_positions():
    a, _ = snapshot([1.0, 3.0], [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])  # centre of mass at x = 3
    b, pos_b = snapshot([1.0, 1.0, 2.0], [[1.0, 2.0, 0.0], [-1.0, 0.5, 3.0], [2.0, -1.0, 1.0]])
    batch = Batch.from_data_list([a, b])  # two snapshots in one batch, as in collect()
    r = com_positions(batch)

    assert torch.allclose(r[:2], torch.tensor([[-3.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))  # worked out by hand
    assert torch.allclose(per_graph_sum(batch.x * r, batch.batch, 2), torch.zeros(2, 3), atol=1e-6)  # sum of m r = 0 in each snapshot
    assert torch.allclose(r[2:] - r[2], pos_b - pos_b[0], atol=1e-6)  # gaps between bodies unchanged


def test_conservation_floor_on_true_accelerations():
    batch = Batch.from_data_list(file_to_graphs(os.path.join(DATA_DIR, "cluster_01160.h5")))  # a real 8-body file
    R_p, R_L = conservation(batch.x, batch.y, com_positions(batch), batch.batch, batch.num_graphs)
    assert R_p.max() < 1e-5  # real gravity conserves momentum (only float32 rounding left)
    assert R_L.max() < 1e-5  # and angular momentum


def test_conservation_detects_net_force_and_torque():
    g, _ = snapshot([1.0, 2.0, 3.0], [[1.0, 0.0, 0.0], [-1.0, 1.0, 0.0], [0.0, -2.0, 0.0]])  # all in the x-y plane
    batch = Batch.from_data_list([g])
    r = com_positions(batch)

    push = torch.tensor([[1.0, 0.0, 0.0]]).repeat(3, 1)  # every body pushed the same way: nothing cancels
    R_p, _ = conservation(batch.x, push, r, batch.batch, 1)
    assert torch.allclose(R_p, torch.ones(1))  # worst case: R_p = 1

    spin = torch.linalg.cross(torch.tensor([[0.0, 0.0, 1.0]]).repeat(3, 1), r)  # every body pushed sideways, all turning the same way
    _, R_L = conservation(batch.x, spin, r, batch.batch, 1)
    assert torch.allclose(R_L, torch.ones(1))  # worst case: R_L = 1


def test_law_fit_recovers_exact_law():
    gen = torch.Generator().manual_seed(0)
    n = 5000  # made-up edges
    log_mj = -3 * torch.rand(n, generator=gen)
    log_mi = -3 * torch.rand(n, generator=gen)
    log_r = -4 + 8 * torch.rand(n, generator=gen)  # distances spanning about 7 decades
    edges = {"log_mj": log_mj, "log_mi": log_mi, "log_r": log_r,
             "log_msg": log_mj - 2 * log_r + 0.3,  # exact gravity, with an arbitrary constant
             "file": torch.randint(10, (n,), generator=gen)}  # spread over 10 pretend files

    fit = law_fit(edges, n_files=10, n_boot=20)  # few resamples: only the best-fit values are tested
    assert abs(fit["alpha"][0] - 1) < 1e-6
    assert abs(fit["beta"][0] + 2) < 1e-6
    assert abs(fit["gamma"][0]) < 1e-6


def test_law_fit_options_for_the_checks():
    gen = torch.Generator().manual_seed(0)
    n = 5000
    log_mj, log_mi = -3 * torch.rand(n, generator=gen), -3 * torch.rand(n, generator=gen)
    log_r = -4 + 8 * torch.rand(n, generator=gen)
    edges = {"log_mj": log_mj, "log_mi": log_mi, "log_r": log_r,
             "log_msg": torch.zeros(n),  # a model that learned nothing: must be ignored when target="log_true"
             "log_true": log_mj - 2 * log_r + 0.3,  # exact gravity
             "kick_w": torch.rand(n, generator=gen),  # arbitrary positive weights
             "file": torch.randint(10, (n,), generator=gen)}

    for options in [{"weighting": "none"}, {"weighting": "kick"}, {"mask": log_r > 0}]:  # an exact law survives any weighting or subset
        fit = law_fit(edges, n_files=10, n_boot=20, target="log_true", **options)
        assert abs(fit["beta"][0] + 2) < 1e-6
        assert abs(fit["alpha"][0] - 1) < 1e-6


def test_true_pair_forces_add_up_to_true_acceleration():
    graphs = file_to_graphs(os.path.join(DATA_DIR, "cluster_01160.h5"))[:5]  # a few rows of a real 8-body file
    batch = Batch.from_data_list(graphs)
    send, recv = batch.edge_index
    disp, dist = batch.edge_attr[:, :3].double(), batch.edge_attr[:, 3].double()
    pair = batch.x[send].double() * disp / (dist[:, None] ** 2 + EPS ** 2) ** 1.5  # the formula collect() uses, as vectors
    total = torch.zeros(batch.num_nodes, 3, dtype=torch.float64).index_add_(0, recv, pair)  # sum into each receiver
    assert torch.allclose(total.float(), batch.y, rtol=1e-4, atol=1e-6)  # G = 1 and ε match the stored data


def test_per_decade_medians_and_hidden_bins():
    amag = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 20.0, 30.0])  # five bodies in 1-10, two in 10-100
    values = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 9.0, 9.0])
    file = torch.tensor([0, 1, 2, 3, 4, 0, 1])  # 1-10 draws on 5 files, 10-100 on only 2

    rows = per_decade(values, amag, file, n_files=5, n_boot=50)
    assert len(rows) == 2  # one row per decade
    assert math.isclose(rows[0]["median"], 0.3, rel_tol=1e-6)  # middle of 0.1 ... 0.5
    assert rows[0]["n"] == 5 and rows[0]["n_files"] == 5
    assert rows[0]["med_lo"] <= rows[0]["median"] <= rows[0]["med_hi"]  # the band contains the median
    assert math.isnan(rows[1]["median"])  # fewer than 3 files: hidden
