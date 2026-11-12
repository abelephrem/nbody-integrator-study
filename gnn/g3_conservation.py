"""GNN Stage C, G3: energy and angular momentum error of GNN rollouts beside the true-force reference and RK4."""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np

from analysis import drift_growth_exponent
from gnn.g2_slopes import COLOURS, MODELS, RESULTS, write_csv
from gnn.metrics import angular_momentum_error, energy_errors, late_early_ratio, load_rollout
from gnn.rollout import read_run, rollout_files

TOLERANCE = {"cluster": 1e-2, "two_body": 1e-4}  # dataset energy check on |dE/U0| (labbook 09-25)
STYLE = {"reference": ("0.2", "true force, Leapfrog"), "rk4": ("#eda100", "true force, RK4")}


def main(split):
    jobs = ["reference", "rk4"] + MODELS[split]
    rows, series = [], {}  # series[(kind, job)] = list of (times, |dE/U0|, dL) per file
    for name in rollout_files(split):
        run = read_run(name)
        kind = "cluster" if name.startswith("cluster") else "two_body"
        for job in jobs:
            pos, vel, times = load_rollout(job, name)
            dE_U, dE_E = energy_errors(run["masses"], pos, vel, run["softening"], run["G"])
            dL = angular_momentum_error(run["masses"], pos, vel)
            q, q_points, saturated = drift_growth_exponent(times, np.abs(dE_U))  # Stage 8 Q2: fit below 1% only
            rows.append({"file": name, "kind": kind, "job": job, "max_abs_dE_U0": np.abs(dE_U).max(), "final_dE_U0": dE_U[-1],
                         "max_abs_dE_E0": np.abs(dE_E).max(), "late_early": late_early_ratio(dE_U), "growth_q": q,
                         "q_points": q_points, "reaches_1pct": saturated, "over_tolerance": bool(np.abs(dE_U).max() > TOLERANCE[kind]),
                         "max_dL": dL.max(), "final_dL": dL[-1]})
            series.setdefault((kind, job), []).append((times, np.abs(dE_U), dL))
    write_csv(f"{RESULTS}/g3_{split}.csv", rows)

    print(f"{split}: medians over files (max |dE/U0|, late/early, q, max dL) and files over tolerance")
    for kind in ("cluster", "two_body"):
        for job in jobs:
            r = [x for x in rows if x["kind"] == kind and x["job"] == job]
            med = lambda key: np.nanmedian([x[key] for x in r])
            print(f"  {kind:8s} {job:20s} max|dE/U0| {med('max_abs_dE_U0'):.1e}  late/early {med('late_early'):5.2f}  q {med('growth_q'):5.2f}"
                  f"  max dL {med('max_dL'):.1e}  over tol {sum(x['over_tolerance'] for x in r)}/{len(r)}  reach 1% {sum(x['reaches_1pct'] for x in r)}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex="col")
    for col, kind in enumerate(("cluster", "two_body")):
        for job in jobs:
            colour, label = STYLE.get(job, (COLOURS.get(job), job))
            curves = series[(kind, job)]
            n = min(len(c[0]) for c in curves)  # runs share checkpoints; trim to the shortest
            times = curves[0][0][:n]
            for row, k in enumerate((1, 2)):
                envelope = np.array([np.maximum.accumulate(c[k][:n]) for c in curves])  # running max of each file
                axes[row, col].plot(times[1:], np.median(envelope, axis=0)[1:], color=colour, lw=1.5, label=label)
        axes[0, col].axhline(TOLERANCE[kind], color="0.5", ls="--", lw=1)
        axes[0, col].set_title(f"{split} {kind.replace('_', '-')}: median running max over files", fontsize=10)
        axes[1, col].set_xlabel("time")
    axes[0, 0].set_ylabel("|ΔE| / |U₀|  (dashed: dataset tolerance)")
    axes[1, 0].set_ylabel("|ΔL| / Σ|L_i(0)|")
    for ax in axes.flat:
        ax.set_yscale("log")
        ax.grid(color="0.92", lw=0.5)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/g3_{split}.png", dpi=150)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
