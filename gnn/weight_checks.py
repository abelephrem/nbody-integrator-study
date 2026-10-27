"""Stage B: how spread out the kick weights are, before any training."""

import os

import torch

from gnn.graph_data import file_to_graphs
from gnn.split import DATA_DIR, load_split

F = 2e-3  # loss floor on |a| (labbook 2026-10-25)


def n_eff(w: torch.Tensor) -> float:
    """Effective number of samples, (sum w)^2 / sum w^2. Equals len(w) when all weights are equal."""
    return (w.sum() ** 2 / (w ** 2).sum()).item()  #  .item() turns a one-number tensor into a plain float


def top_share(w: torch.Tensor, frac:float) -> float:
    """Share of the total weight held by the heaviest 'frac' of the entries."""
    k = max(1, int(frac * len(w)))  # how many entries are in the top fraction (at least one)
    return (w.sort(descending=True).values[:k].sum() / w.sum()).item()  # heaviest k summed, over the total


def file_weights(path:str):
    """|a| and per-file normalised kick weight of every body in every row of one file."""
    graphs = file_to_graphs(path)  # one PyG graph per stored row (Stage A)
    amag = torch.stack([g.y.norm(dim=-1) for g in graphs])  # (T, N) |a| of each body in each row
    w_time = torch.stack([g.weight for g in graphs])  # (T,) stored time weight of each row
    kick = amag * w_time[:, None]  # (T, N) raw kick weight = |a| x the row's time weight
    kick = kick / kick.sum()  #  normalise per file: all body-rows of the file now sum to 1
    return amag, kick, w_time


def main():
    files = load_split()["train"]  # the 208 training filenames
    body_amag, body_kick = [], []  # one tensor per file: every body in every row
    row_kick, row_time = [], []  # one tensor per file: every row
    file_kick = []  # one number per file

    for name in files:
        amag, kick, w_time = file_weights(os.path.join(DATA_DIR, name))
        body_amag.append(amag.flatten())  # (T*N,) this file's body-rows laid out in one line
        body_kick.append(kick.flatten())
        row_kick.append(kick.sum(dim=1))  # (T,) a row's weight = sum over its bodies
        row_time.append(w_time)  # (T,) time weighting normalised the same way: each body w_time/N, so a row sums back to w_time
        file_kick.append(kick.sum().item())  # should be 1 for every file

    body_amag = torch.cat(body_amag)  # join the 208 per-file tensors end to end into one long tensor
    body_kick = torch.cat(body_kick)
    row_kick = torch.cat(row_kick)
    row_time = torch.cat(row_time)
    file_kick = torch.tensor(file_kick)  # Python list of 208 floats -> tensor

    print(f"{len(files)} files, {len(row_kick)} rows, {len(body_kick)} body-rows")
    
    # 1. spread: kick weighting against time weighting
    for label, w in [("rows, time", row_time), ("rows, kick", row_kick), ("files, kick", file_kick)]:
        print(f"{label:12s} n_eff/n = {n_eff(w) / len(w):.3f}  "
              f"top 1% hold {top_share(w, 0.01):.3f}  top 10% hold {top_share(w, 0.10):.3f}")

    # 2. body-rows below the loss floor
    below = body_amag < F  # True where |a| < F, False elsewhere
    print(f"|a| < F: {below.float().mean():.4f} of body-rows, "
          f"{(body_kick[below].sum() / body_kick.sum()):.2e} of kick weight")

    # 3. two-body vs cluster share of the total kick weight
    is_two = torch.tensor([name.startswith("two_body") for name in files])  # True for two-body files
    print(f"two-body: {is_two.sum()} files, {(file_kick[is_two].sum() / file_kick.sum()):.3f} of kick weight")
    print(f"cluster:  {(~is_two).sum()} files, {(file_kick[~is_two].sum() / file_kick.sum()):.3f} of kick weight")

    

if __name__ == "__main__":
    main()

    

