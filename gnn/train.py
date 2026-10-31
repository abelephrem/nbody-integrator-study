"""GNN Stage B: weighted relative loss, per-body weights and the training loop."""

import os

import torch
import copy
import sys
import time
from torch_geometric.loader import DataLoader
from gnn.model import GravityGNN
from gnn.graph_data import file_to_graphs
from gnn.split import DATA_DIR, load_split

F = 2e-3  # loss floor on |a| (labbook 2026-10-25)
CLIP_K = 10  # clip a step whose gradient is more that CLIP_k x the recent median gradient size
CLIP_WINDOW = 100  # how many recent training steps that median is taken over
OUT_DIR = "C:/nbody_results/stage_b"  # trained models, kept outside OneDrive: its sync locked the file mid-save (error 1224)


def relative_error(pred: torch.Tensor, y: torch.Tensor, floor: float = F) -> torch.Tensor:
    """Per-body |pred - y| / max(|y|, floor). pred, y: (N, 3) -> (N,)."""
    return (pred - y).norm(dim=-1) / y.norm(dim=-1).clamp(min=floor)  # clamp lifts any |y| below the floor up to it


def weighted_loss(pred: torch.Tensor, y: torch.Tensor, w: torch.Tensor, floor: float = F):
    """Weighted mean of the squared relative error; dividing by the total weight keeps the scale fixed."""
    return (w * relative_error(pred, y, floor) ** 2).sum() / w.sum()


def weighted_graphs(split_name: str, scheme: str = "kick") -> list:
    """Every graph in one part of the split, each with g.w = (N,) per-body weight, normalised per file."""
    out = []
    for name in load_split()[split_name]:
        graphs = file_to_graphs(os.path.join(DATA_DIR, name))  # one graph per stored row
        amag = torch.stack([g.y.norm(dim=-1) for g in graphs])  # (T, N) |a| of each body in each row
        w_time = torch.stack([g.weight for g in graphs])  # (T,) stored time weight of each row
        if scheme == "kick":
            raw = amag * w_time[:, None]  # (T, N) |a| x the row's time weight
        else:
            raw = w_time[:, None].expand_as(amag)  # (T, N) time weighting: every body in a row gets the row's weight
        w = raw / raw.sum()  #  normalise per file: all body-rows of the file sum to 1
        for g, w_row in zip(graphs, w):  # pair each graph with its row of weights
            g.w = w_row  # (N,) attach this row's per-body weights to the graph
        out += graphs
    return out

def epoch_loss(model, loader, opt=None, grad_sizes=None) -> tuple:
    """One pass over loader. Trains if an optimiser is given, otherwise only measures. Returns the weighted loss of the whole pass."""
    training = opt is not None  # True when we were given an optimiser
    model.train(training)  # tell the model which mode it is in
    num, den, n_clipped = 0.0, 0.0, 0  # running sum of w x error^2, of w, and count of clipped steps
    for batch in loader:
        with torch.set_grad_enabled(training):  # only record the computation for gradients when training
            pred = model(batch.x, batch.edge_index, batch.edge_attr)  # (bodies in batch, 3) predicted accelerations
            loss = weighted_loss(pred, batch.y, batch.w)  # this batch's weighted mean
        if training:
            opt.zero_grad()  # clear the previous step's gradients
            loss.backward()  # work out the gradient of the loss for every weight
            recent = grad_sizes[-CLIP_WINDOW:]  # gradient sizes of the last 100 steps (fewer at the very start)
            limit = CLIP_K * torch.tensor(recent).median().item() if recent else float("inf")  # no history yet -> no limit
            size = torch.nn.utils.clip_grad_norm_(model.parameters(), limit).item()  # clips if over the limit; returns the size before clipping
            grad_sizes.append(size)  # remember it, for the next steps' median and for the training log
            n_clipped += size > limit  # True counts as 1, False counts as 0
            opt.step()  # nudge every weight using its (possibly clipped) gradient

        w_sum = batch.w.sum().item()  # this batch's total weight
        num += loss.item() * w_sum  # undo the per-batch division, so batches add up correctly
        den += w_sum
    return num / den, n_clipped # weighted mean over the whole pass, and how many steps were clipped


def train(design: int, scheme: str = "kick", n_epochs: int = 100, batch_size: int = 512,
          lr: float = 1e-3, seed: int = 0, out_dir: str = OUT_DIR) -> str:
    """Train one model. Saves final and best-validation weights, every setting, and a per-epoch log. Returns the file path."""
    torch.manual_seed(seed)  # same start weights and batch orders for a given seed

    # data
    train_graphs = weighted_graphs("train", scheme)  # kept as a list, so the input statistics can be measured from it
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)  # new random order every epoch
    val_loader = DataLoader(weighted_graphs("val", scheme), batch_size=2048)  # fixed order; batch size here only affects speed

    # model, optimiser and learning-rate schedule
    model = GravityGNN(design)  # random starting weights
    model.set_input_stats(train_graphs)  # standardise inputs with training-set statistics only (decision 12)
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # hand the weights to Adam
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.1  ** (1 / n_epochs)) # lr x this factor each epoch: 10x lower by the end

    # bookkeeping
    grad_sizes, log = [], [] # every step's gradient size; one dictionary of results per epoch
    best_val, best_state, best_epoch = float("inf"), None, None  # best validation loss so far, and the weights that got it

    # where to save
    os.makedirs(out_dir, exist_ok=True) # create the folder if it doesn't exist yet
    tag = f"_seed{seed}" if seed != 0 else ""  # seed 0 keeps the original file names
    path = os.path.join(out_dir, f"design{design}_{scheme}_{n_epochs}ep{tag}.pt")  # e.g. design2_kick_200ep_seed1.pt

    for epoch in range(n_epochs):
        start = time.time()  # to time the epoch
        lr_now = opt.param_groups[0]["lr"]  # the learning rate used this epoch

        # train, measure, then lower the learning rate
        train_loss, n_clipped = epoch_loss(model, train_loader, opt, grad_sizes)  # one training pass
        val_loss, _ = epoch_loss(model, val_loader)  # one measuring pass on validation data; _ = the clip count we don't need (always 0)
        sched.step()  # lower the learning rate for the next epoch

        # record this epoch
        sizes = torch.tensor(grad_sizes[-len(train_loader):])  # this epoch's step sizes (one per batch)
        log.append({"epoch": epoch, "lr": lr_now, "train": train_loss, "val": val_loss, "clipped": n_clipped,
                    "grad_p50": sizes.median().item(), "grad_p99": sizes.quantile(0.99).item(),
                    "seconds": time.time() - start})

        # keep a frozen copy of the best_validation weights
        if val_loss < best_val:
            best_val, best_state, best_epoch = val_loss, copy.deepcopy(model.state_dict()), epoch

        # save everything every epoch, so a crash loses nothing
        torch.save({"design": design, "scheme": scheme, "n_epochs": n_epochs, "batch_size": batch_size,
                    "lr": lr, "seed": seed, "clip_k": CLIP_K, "clip_window": CLIP_WINDOW,
                    "final_state": model.state_dict(), "best_state": best_state, "best_epoch": best_epoch,
                    "log": log}, path)

        # one line of progress per epoch
        e = log[-1]  # this epoch's entry
        print(f"epoch {epoch:3d}  lr {e['lr']:.2e}  train {e['train']:.3e}  val {e['val']:.3e}  "
              f"clipped {e['clipped']:3d}  grad p50 {e['grad_p50']:.2e} p99 {e['grad_p99']:.2e}  {e['seconds']:.0f}s",
              flush=True)

    return path


if __name__ == "__main__":
    design = int(sys.argv[1])  # e.g. python -m gnn.train 2
    n_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 100  # optional: python -m gnn.train 2 3 runs 3 epochs
    scheme = sys.argv[3] if len(sys.argv) > 3 else "kick"  # optional: "time" for the Stage C baseline
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0  # optional: python -m gnn.train 2 200 kick 1 runs seed 1

    print(train(design, scheme, n_epochs, seed=seed))


