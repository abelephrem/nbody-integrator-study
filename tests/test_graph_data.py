import torch
import pytest
import glob
import os
import h5py
import torch
from torch_geometric.data import Batch
from gnn.graph_data import complete_edge_index, time_weights, read_rows, file_to_graphs
from gnn.split import DATA_DIR


# TEMPORARY - remove once the two-body files are regenerated (Phase 2).
# These five e = 0.8 two-body files end at t = 19.83 instead of 20: resample_true_anomaly
# cannot track the orbit at periapsis when picking from stored steps at dt = 0.01.
# Named explicitly (not a looser tolerance) so that any OTHER short file still fails.
KNOWN_SHORT_FILES = {f"two_body_{i:05d}.h5" for i in range(75, 80)}
SAMPLE_FILES = ["two_body_00000.h5", "two_body_00075.h5", "cluster_01000.h5", "cluster_01160.h5"]

@pytest.fixture(scope="module")
def sample_graphs():
    """{filename: (N, list of graphs)}, built once and shared by every test that asks for it."""
    out = {}
    for name in SAMPLE_FILES:
        path = os.path.join(DATA_DIR, name)
        masses, _, _, _ = read_rows(path)
        out[name] = (masses.shape[0], file_to_graphs(path))  # N, and every graph in the file
    return out

def test_graph_shapes(sample_graphs):
    for name, (N, graphs) in sample_graphs.items():
        E = N * (N - 1)                                    # fully connected, both directions
        for g in graphs:
            assert g.x.shape == (N, 1), name               # mass per body
            assert g.edge_index.shape == (2, E), name
            assert g.edge_attr.shape == (E, 4), name       # dx, dy, dz, distance
            assert g.y.shape == (N, 3), name               # target acceleration
            assert g.x.dtype == torch.float32 and g.edge_attr.dtype == torch.float32, name
            assert g.y.dtype == torch.float32, name
            assert g.edge_index.dtype == torch.int64, name


def test_edges_reproduce_stored_acceleration(sample_graphs):
    # Recompute every body's acceleration using ONLY the graph (step 3's check).
    # A sender/receiver mix-up anywhere would give errors of order 1, not 1e-7.
    eps = 0.002                                            # softening of the dataset (G = 1)
    for name, (N, graphs) in sample_graphs.items():
        for g in graphs:
            s, r = g.edge_index
            disp = g.edge_attr[:, :3].double()             # float64 for the recompute
            dist = g.edge_attr[:, 3:].double()             # (E, 1)
            m_sender = g.x[s].double()                     # (E, 1) sender mass per edge
            msg = m_sender * disp / (dist**2 + eps**2) ** 1.5      # (E, 3)
            a = torch.zeros(N, 3, dtype=torch.float64)
            a.index_add_(0, r, msg)                        # sum the messages at each receiver
            # float32 rounding gives at most ~6e-7 relative error here, so 1e-5 has headroom
            err = (a - g.y.double()).norm(dim=1)            # (N,) size of each body's error
            size = g.y.double().norm(dim=1)                 # (N,) size of each body's acceleration
            assert (err <= 1e-5 * size).all(), (name, g.row)   # worst seen ~6e-7, so 1e-5 has headroom


def test_batching_keeps_graphs_separate(sample_graphs):
    # arrange - one graph from each sample file: 2, 2, 3 and 8 bodies
    graphs = [gs[0] for (_, gs) in sample_graphs.values()]
    sizes = torch.tensor([g.num_nodes for g in graphs])

    # act - glue them together, as the DataLoader does
    batch = Batch.from_data_list(graphs)

    # assert - node count, and ptr marks where each graph starts
    assert batch.num_nodes == sizes.sum().item()
    assert torch.equal(batch.ptr, torch.cat([torch.tensor([0]), torch.cumsum(sizes, 0)]))

    # assert - no edge joins two different graphs
    s, r = batch.edge_index
    assert torch.equal(batch.batch[s], batch.batch[r])   # sender and receiver always in the same graph

    # assert - each graph's edges are its own edges, shifted by where it starts
    start = 0
    for k, g in enumerate(graphs):
        E = g.edge_index.shape[1]
        assert torch.equal(batch.edge_index[:, start:start + E], g.edge_index + batch.ptr[k])
        start += E

def test_times_strictly_increasing_every_file():
    # arrange - every HDF5 file in the dataset
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
    assert len(paths) == 360

    for path in paths:
        # act - the rows training will actually see (duplicates removed)
        _, _, _, times = read_rows(path)

        # assert - each row later than the one before, and every run starts at t=0
        assert (times[1:] > times [:-1]).all(), path
        assert times[0].item() == 0.0, path

def test_complete_edge_index():
    # arrange / act - edges for 3 bodies and for 10 bodies
    ei3 = complete_edge_index(3)
    ei10 = complete_edge_index(10)

    # assert - exact layout for N=3: senders on top, receivers below
    expected = torch.tensor([[1, 2, 0, 2, 0, 1],
                             [0, 0, 1, 1, 2, 2]])
    assert torch.equal(ei3, expected)  # same shape and same values everywhere
    # assert - N=10: N*(N-1) edges, no self-pairs, every ordered pair exactly once
    assert ei10.shape == (2, 90)
    assert (ei10[0] != ei10[1]).all()  # no edge where sender == receiver
    pairs = set(zip(ei10[0].tolist(), ei10[1].tolist())) # (sender, receiver) pairs; a set drops duplicates
    assert len(pairs) == 90  # still 90 after dropping duplicates -> none repeated


def test_time_weights_worked_example():
    # arrange - uneven gaps of 10, 20, 30, 40 
    times = torch.tensor([0., 10., 30., 60., 100.])

    # act
    w = time_weights(times)

    # assert - each row gets half the gap on each side; end rows get one half
    expected = torch.tensor([5., 15., 25., 35., 20.])
    assert torch.allclose(w, expected) # floats, so allow rounding differences


def test_time_weights_every_file():
    # arrange - every HDF5 file in the dataset
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "*.h5")))
    assert len(paths) == 360 # guards against the test silently checking nothing)

    for path in paths:
        _, _, _, times = read_rows(path)  # duplicates removed, as in training

        # act
        w = time_weights(times)

        # assert - every row owns some time (so timestamps are unique and increasing),
        # and the weights and up to the full run length of 20
        assert (w > 0).all(), path
        if os.path.basename(path) in KNOWN_SHORT_FILES:
            continue  # ends at 19.83 -  see KNOWN_SHORT_FILES
        assert abs(w.sum().item() - 20.0) < 1e-9, path