"""Tests for the Stage B model (gnn/model.py)."""

import os

import pytest
import torch
from torch_geometric.data import Batch

from gnn.graph_data import complete_edge_index, file_to_graphs
from gnn.model import GravityGNN
from gnn.split import DATA_DIR

DESIGNS = [1, 2, 3]


def random_system(N: int, seed: int = 0):
    """A made-up system of N bodies in the same format as the real graphs: x, edge_index, edge_attr."""
    torch.manual_seed(seed)  # repeatable random numbers
    x = torch.rand(N, 1) + 0.5  # (N, 1) masses between 0.5 and 1.5
    pos = torch.randn(N, 3)  # (N, 3) positions
    edge_index = complete_edge_index(N)  # every ordered pair
    send, recv = edge_index  # row 0 = senders, row 1 = receivers
    disp = pos[send] - pos[recv]  # (E, 3) r_sender - r_receiver
    edge_attr = torch.cat([disp, disp.norm(dim=-1, keepdim=True)], dim=-1)  # (E, 4) disp then dist
    return x, edge_index, edge_attr


@pytest.mark.parametrize("design", DESIGNS)
def test_output_shape_real_batch(design):
    graphs = (file_to_graphs(os.path.join(DATA_DIR, "two_body_00000.h5"))[:3]  # 3 two-body rows
              + file_to_graphs(os.path.join(DATA_DIR, "cluster_01000.h5"))[:3])  # + 3 cluster rows
    batch = Batch.from_data_list(graphs)  # 6 graphs glued into one big graph, as training will do
    a = GravityGNN(design)(batch.x, batch.edge_index, batch.edge_attr)
    assert a.shape == batch.y.shape == (15, 3)  # one 3-vector per body: 3 x 2 + 3 x 3 = 15 bodies


@pytest.mark.parametrize("design", [2, 3])
def test_messages_parallel_to_disp(design):
    x, edge_index, edge_attr = random_system(5)
    send, recv = edge_index
    msg = GravityGNN(design).message(x[recv], x[send], edge_attr)  # (E, 3) call message directly: x_i = receivers, x_j = senders
    disp = edge_attr[:, :3]  # (E, 3) displacement towards the sender
    sin_angle = torch.linalg.cross(msg, disp).norm(dim=-1) / (msg.norm(dim=-1) * disp.norm(dim=-1))  # (E,) 0 when parallel
    assert (sin_angle < 1e-5).all()  # every message lies along the line between its two bodies
    assert ((msg * disp).sum(dim=-1) > 0).all()  # and points towards the sender (attractive), not away


def momentum_ratio(x, a):
    """R_p: size of the net force on the system, relative to the sum of the individual force sizes."""
    return ((x * a).sum(dim=0).norm() / (x * a).norm(dim=-1).sum()).item()


def test_design3_conserves_momentum():
    x, edge_index, edge_attr = random_system(6)
    assert momentum_ratio(x, GravityGNN(3)(x, edge_index, edge_attr)) < 1e-5  # cancels with untrained weights
    assert momentum_ratio(x, GravityGNN(2)(x, edge_index, edge_attr)) > 1e-2  # control: design 2 does not


def mixed_mass_graphs():
    """Graphs from two files with different masses, so every input column varies (std > 0)."""
    return (file_to_graphs(os.path.join(DATA_DIR, "two_body_00075.h5"))  # masses 0.909 and 0.091
            + file_to_graphs(os.path.join(DATA_DIR, "cluster_01160.h5")))  # eight masses of 0.125


@pytest.mark.parametrize("design", DESIGNS)
def test_input_stats_standardise(design):
    graphs = mixed_mass_graphs()
    model = GravityGNN(design)
    model.set_input_stats(graphs)

    inp = torch.cat([model.mlp_inputs(g.x[g.edge_index[1]], g.x[g.edge_index[0]], g.edge_attr) for g in graphs])
    z = (inp - model.in_mean) / model.in_std  # what the MLP will actually see

    others = [0, 4, 5] if design == 1 else list(range(z.shape[1]))  # every column except design 1's disp
    assert torch.allclose(z[:, others].mean(dim=0), torch.zeros(len(others)), atol=1e-4)  # mean 0
    assert torch.allclose(z[:, others].std(dim=0), torch.ones(len(others)), atol=1e-4)  # spread 1

    if design == 1:
        assert (model.in_mean[1:4] == 0).all()  # disp is not shifted
        assert model.in_std[1] == model.in_std[2] == model.in_std[3]  # one shared scale for x, y, z


def test_design3_conserves_momentum_with_stats():
    x, edge_index, edge_attr = random_system(6)
    model = GravityGNN(3)
    model.set_input_stats(mixed_mass_graphs())  # real, non-trivial shift and scale
    assert momentum_ratio(x, model(x, edge_index, edge_attr)) < 1e-5  # standardising keeps the third law
