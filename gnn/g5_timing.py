"""GNN Stage C, G5: time per force call, true force (NumPy) vs each GNN, at N = 2..10. Run on an otherwise idle machine."""

import os
import time

import matplotlib
matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
import torch

from gnn.evaluate import load_model
from gnn.g2_slopes import COLOURS, MODELS, RESULTS, write_csv
from gnn.rollout import gnn_force_func, true_force_func
from gnn.train import OUT_DIR as MODEL_DIR

SOFTENING = 0.002
WARMUP = 50  # calls discarded before timing
CALLS = 200  # calls per timed block
REPEATS = 15  # timed blocks; the median block is reported


def time_per_call(force, pos):
    """Median seconds per call over REPEATS blocks of CALLS calls, after a warm-up."""
    for _ in range(WARMUP):
        force(pos)
    blocks = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        for _ in range(CALLS):
            force(pos)
        blocks.append((time.perf_counter() - start) / CALLS)
    return float(np.median(blocks)), float(np.percentile(blocks, 25)), float(np.percentile(blocks, 75))


def main():
    torch.set_num_threads(1)  # one thread, as in the rollouts
    models = {name: load_model(os.path.join(MODEL_DIR, f"{name}.pt"), "final_state") for name in MODELS["test"]}
    rng = np.random.default_rng(0)
    rows = []
    for n in range(2, 11):
        masses = np.full(n, 1.0 / n)  # cost doesn't depend on the values, only on N
        pos = 0.3 * rng.normal(size=(n, 3))
        forces = {"true force": true_force_func(masses, SOFTENING)}
        forces.update({name: gnn_force_func(model, masses) for name, model in models.items()})
        for name, force in forces.items():
            median, q25, q75 = time_per_call(force, pos)
            rows.append({"N": n, "force": name, "seconds_per_call": median, "q25": q25, "q75": q75})
        true = rows[-len(forces)]["seconds_per_call"]
        print(f"N {n:2d}: " + "  ".join(f"{r['force']} {r['seconds_per_call'] * 1e6:7.1f} us" for r in rows[-len(forces):])
              + f"  (true / kick-D3 = {true / rows[-len(forces) + 1]['seconds_per_call']:.3f})", flush=True)
    write_csv(f"{RESULTS}/g5_timing.csv", rows)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name in ["true force"] + list(models):
        r = [x for x in rows if x["force"] == name]
        ax.plot([x["N"] for x in r], [x["seconds_per_call"] * 1e6 for x in r], "o-", ms=5, lw=1.5,
                color=COLOURS.get(name, "0.2"), label=name)
    ax.set_yscale("log")
    ax.set_xlabel("number of bodies N")
    ax.set_ylabel("time per force call (μs, median)")
    ax.set_title("G5: force-call cost, single thread", fontsize=10)
    ax.grid(color="0.92", lw=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/g5_timing.png", dpi=150)


if __name__ == "__main__":
    main()
