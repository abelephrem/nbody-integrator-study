"""Stage 10: the test-cluster panels of g3_test.png redrawn at single-column size (3.5 in) for the paper.

Same numbers as gnn/g3_conservation.py: per-file running max of |dE/U0| and of the angular momentum error,
median over files. Same colours. Plot only; the median max |dE/U0| of each method is checked against g3_test.csv.
Run with: python -m gnn.g3_column_figure
"""

import csv

import matplotlib
matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np

from gnn.g2_slopes import COLOURS, MODELS, RESULTS
from gnn.g3_conservation import STYLE, TOLERANCE
from gnn.metrics import angular_momentum_error, energy_errors, load_rollout
from gnn.rollout import read_run, rollout_files

LABELS = {"design3_kick_200ep": "kick-D3", "design3_time_200ep": "time-D3", "design2_kick_200ep": "D2"}  # paper names


def main(split="test"):
    jobs = ["reference", "rk4"] + MODELS[split]
    files = [name for name in rollout_files(split) if name.startswith("cluster")]
    with open(f"{RESULTS}/g3_{split}.csv") as f:
        g3 = [r for r in csv.DictReader(f) if r["kind"] == "cluster"]

    plt.rcParams.update({"font.size": 8, "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
                         "legend.fontsize": 7, "font.family": "serif", "mathtext.fontset": "dejavuserif"})
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.4), sharex=True)
    for job in jobs:
        curves = []
        for name in files:
            run = read_run(name)
            pos, vel, times = load_rollout(job, name)
            dE_U, _ = energy_errors(run["masses"], pos, vel, run["softening"], run["G"])
            curves.append((times, np.abs(dE_U), angular_momentum_error(run["masses"], pos, vel)))
        n = min(len(c[0]) for c in curves)  # runs share checkpoints; trim to the shortest
        colour, label = STYLE[job] if job in STYLE else (COLOURS[job], LABELS[job])
        for row, k in enumerate((1, 2)):  # row 0 energy, row 1 angular momentum
            envelope = np.array([np.maximum.accumulate(c[k][:n]) for c in curves])  # running max of each file
            axes[row].plot(curves[0][0][1:n], np.median(envelope, axis=0)[1:], color=colour, lw=1.2, label=label)
        mine = np.median([c[1].max() for c in curves])
        theirs = np.median([float(r["max_abs_dE_U0"]) for r in g3 if r["job"] == job])
        print(f"  [{'PASS' if np.isclose(mine, theirs, rtol=1e-9) else 'FAIL'}] {job}: median max|dE/U0| {mine:.3e} vs g3_{split}.csv {theirs:.3e}")

    axes[0].axhline(TOLERANCE["cluster"], color="0.5", ls="--", lw=0.9)  # dataset tolerance, for context only
    axes[0].set_ylabel(r"$|\Delta E|\,/\,|U_0|$")
    axes[1].set_ylabel(r"$|\Delta \mathbf{L}|\,/\,\Sigma_i |\mathbf{L}_i(0)|$")  # \Sigma_i: mathtext's \sum puts i underneath
    axes[1].set_xlabel(r"time ($N$-body units)")
    for ax in axes:
        ax.set_yscale("log")
        ax.grid(color="0.92", lw=0.5)
        ax.set_xlim(0, 20)
        ax.set_xticks([0, 5, 10, 15, 20])
    axes[0].set_yticks([1e-15, 1e-12, 1e-9, 1e-6, 1e-3])
    axes[1].set_yticks([1e-16, 1e-13, 1e-10, 1e-7, 1e-4, 1e-1])
    axes[0].legend(loc="lower right", framealpha=0.9, handlelength=1.6, borderpad=0.4, labelspacing=0.3)
    fig.tight_layout(h_pad=0.6)
    fig.savefig(f"{RESULTS}/g3_{split}_clusters_column.png", dpi=300)


if __name__ == "__main__":
    main()
