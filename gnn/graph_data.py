"""Graph construction for the GNN: turns HDF5 trajectory into PyTorch Geometric graphs."""

import torch
import h5py
import os
from torch_geometric.data import Data
from gnn.split import DATA_DIR, load_split


def complete_edge_index(N: int) -> torch.Tensor:
    """Every ordered pair (sender, reciever) with sender != receiver.

    Returns a (2, N*(N-1)) int64 tensor: row 0 = senders, row 1 = receivers.
    PyG passes messages from row 0 to row 1, so edge k carries the influence of 
    body edge_index[0, k] on edge_index[1, k].
    """
    idx = torch.arange(N)  # [0, 1, ..., N-1]
    senders = idx.repeat(N)  # whole list repeated N times: [0,1,2, 0,1,2, ...]
    receivers = idx.repeat_interleave(N)  # each entry repeated N times: [0,0,0, 1,1,1, ...]
    keep = senders != receivers  # mask: False for self-pairs (a body doesn't act on itself)
    return torch.stack([senders[keep], receivers[keep]])  # (2, E)


def time_weights(times: torch.Tensor) -> torch.Tensor:
    """How much time each row represents (trapezoidal-rule weights).
    
    The weights are normalised by the run length, so they add up to 1 for every file.

    Example: times [0, 10, 30, 60, 100] -> weights [0.05, 0.15, 0.25, 0.35, 0.20], total 1.

    """

    gaps = times[1:] - times[:-1]  # (T-1,) gap between each row and the next
    w = torch.zeros_like(times)  # (T,), all zeros, same shape and dtype as times
    w[:-1] += gaps / 2  # half of each gap to the row on its left (the last row gets none)
    w[1:] += gaps / 2  # half of each gap to the row on its right (the first row gets none)
    return w / (times[-1] - times[0])  # divide by run length -> share of the run; every file sums to 1



def read_rows(path: str):
    """Read one HDF5 file: masses (N,), positions (T, N, 3), accelerations(T, N, 3), times (T,)."""
    with h5py.File(path, "r") as f:
        masses = torch.from_numpy(f["masses"][:])  # (N,)
        pos = torch.from_numpy(f["positions"][:])  # (T, N, 3)
        acc = torch.from_numpy(f["accelerations"][:])  # (T, N, 3)
        times = torch.from_numpy(f["times"][:])  # (T,)
    assert (times[1:] > times[:-1]).all(), f"{path}: times not strictly increasing"  # dataset is clean; fail loudly if not
    return masses, pos, acc, times  # rows exactly as stored


def file_to_graphs(path: str) -> list[Data]:
    """Turn every row of one HDF5 trajectory file into a PyG graph.
        x          (N, 1)  float32  mass of each body
        edge_index (2, E)  int64    sender -> receiver, from complete_edge_index
        edge_attr  (E, 4)  float32  displacement r_sender - r_receiver (3), distance (1)
        y          (N, 3)  float32  stored acceleration of each body (the target)
        weight     ()      float32  time this row represents (see time_weights)
        t          ()      float64  the row's time, for tracing back to the file
        row        int              the row's index in the file
    """

    # read every row once; float64 so the subtraction below is precise
    masses, pos, acc, times = read_rows(path)   # float64, duplicate rows already removed


    N = masses.shape[0]  # number of bodies = length of the masses list
    edge_index = complete_edge_index(N)  # same for every row of this file, so build it at once
    s, r = edge_index  # s = senders, r = receivers

    # edge features for all T rows at once
    disp = pos[:, s] - pos[:, r]  # (T, E, 3) r_sender - r_receiver, for every row
    dist = disp.norm(dim=-1, keepdim=True)  # (T, E, 1) dim=-1 = last axis(xyz)
    edge_attr = torch.cat([disp, dist], dim=-1)  # (T, E, 4)

    # convert to float32 now that the precise subtraction is done
    x = masses[:, None].float()  # (N, 1) masses as a column; the same for every row
    edge_attr = edge_attr.float() # (T, E, 4)
    y = acc.float()  # (T, N, 3)
    w = time_weights(times).float()  # (T,) weights worked out in float64, stored as float32

    # cut the big blocks into one graph per row
    graphs = []
    for k in range(len(times)):
        graphs.append(Data(
            x=x,  # shared by every graph from this file
            edge_index=edge_index,  # shared by every graph from this file
            edge_attr=edge_attr[k],  # (E, 4) this row's edge features
            y=y[k],  # (N, 3) this row's accelerationsq
            weight=w[k],  # time this row represents
            t=times[k],  # this row's time (kept float64, only for tracing)
            row=k,  # this row's index in the file
        ))
    return graphs


def load_graphs(split_name: str, data_dir: str = DATA_DIR) -> list[Data]:
    """Every graph from every file in one part of the saved split."""
    files = load_split()[split_name]  # the filenames in that part of the split
    graphs = []
    for name in files:
        graphs += file_to_graphs(os.path.join(data_dir, name))  # += adds every graph from this file
    return graphs