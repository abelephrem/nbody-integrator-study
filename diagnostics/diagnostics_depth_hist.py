"""Diagnostic: the encounter-depth distribution of the regenerated dataset.

Before this session the true closest approach was known for three runs out of 260 - every
other figure came from saved snapshots, which the close-encounter investigation proved
understate depth by up to 11.7x. `min_sep_run` (tracked over every substep) and the
closest-approach rows now put the real distribution on every file.

Three views, because they answer different questions:
    per-run      how deep does each simulation get?          (260 values, from attrs)
    per-encounter  how deep is a typical encounter?          (local minima of the saved series)
    per-row      what depths does the GNN actually train on?  (every saved cluster row)

Also reports how many DISTINCT runs contribute rows below each depth: 100% coverage of the
acceleration peak achieved by one extreme run is a much weaker result than the same number
spread across a hundred runs.

READ-ONLY with respect to the dataset.

Outputs:
    results/closest_approach_depth.csv
    figures/closest_approach_depth.png
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]  # repo root for `from bodies import`,
#                                                 this dir for sibling diagnostics
import glob
import os

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = "C:/nbody_data"
EPS = 0.002
EVENT_FACTOR = 3.0  # the episode trigger; per-encounter minima are only sought below this
THRESHOLDS = (3.0, 2.0, 1.0, 0.707, 0.5, 0.2, 0.1, 0.05, 0.02)  # in units of eps
# 0.707 = 1/sqrt(2): the Plummer acceleration peak sits at r = eps/sqrt(2), so rows below
# this are the "inner branch" that was entirely absent before the closest-approach row.


def min_sep_series(positions):
    """(T,) minimum separation over all pairs at each saved row.

    Uses upper-triangle pair indexing rather than the full (T,N,N) distance matrix - at
    T=48705, N=10 the full matrix is ~1.2 GB, the pair form ~53 MB.
    """
    N = positions.shape[1]
    iu = np.triu_indices(N, k=1)
    if len(iu[0]) == 0:
        return np.full(len(positions), np.inf)
    d = positions[:, iu[0], :] - positions[:, iu[1], :]  # (T, n_pairs, 3)
    return np.linalg.norm(d, axis=2).min(axis=1)


def local_minima(series, ceiling):
    """Indices where the middle of three consecutive rows is below both neighbours.

    The same bracket test the recorder uses, applied to the SAVED rows - so it recovers the
    closest-approach rows plus any shell row that happens to sit at a turning point. The
    ceiling restricts it to the encounter regime, matching the recorder's gate.
    """
    if len(series) < 3:
        return np.array([], dtype=int)
    mid = series[1:-1]
    hit = (mid < series[:-2]) & (mid < series[2:]) & (mid < ceiling)
    return np.nonzero(hit)[0] + 1


def main():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "cluster_*.h5")))
    if not paths:
        raise SystemExit(f"no cluster files in {DATA_DIR}")

    per_run, per_enc, per_row, per_row_run = [], [], [], []
    for p in paths:
        with h5py.File(p, "r") as f:
            pos = f["positions"][:]
            per_run.append(float(f.attrs["min_sep_run"]))
        s = min_sep_series(pos)
        per_row.append(s)
        per_row_run.append(np.full(len(s), len(per_run) - 1))  # which run each row came from
        per_enc.append(s[local_minima(s, EVENT_FACTOR * EPS)])

    per_run = np.array(per_run) / EPS
    per_enc = np.concatenate(per_enc) / EPS
    per_row = np.concatenate(per_row) / EPS
    per_row_run = np.concatenate(per_row_run)
    n_runs = len(paths)

    print(f"{n_runs} cluster runs, {len(per_row):,} saved rows, "
          f"{len(per_enc):,} per-encounter minima\n")
    for name, v in (("per-run minimum", per_run), ("per-encounter", per_enc)):
        print(f"{name:16s} min {v.min():.4f}  p10 {np.percentile(v,10):.3f}  "
              f"median {np.median(v):.3f}  p90 {np.percentile(v,90):.2f} eps")

    print("\ndepth      rows below   runs contributing   per-run minima below")
    table = []
    for t in THRESHOLDS:
        rows_below = int((per_row < t).sum())
        runs_contrib = len(np.unique(per_row_run[per_row < t]))
        runs_reaching = int((per_run < t).sum())
        enc_below = int((per_enc < t).sum())
        print(f"{t:6.3f} eps  {rows_below:8,d}   {runs_contrib:4d} / {n_runs}"
              f"        {runs_reaching:4d} / {n_runs}")
        table.append(dict(depth_over_eps=t, rows_below=rows_below,
                          frac_rows=rows_below / len(per_row),
                          runs_contributing=runs_contrib,
                          runs_reaching=runs_reaching, encounters_below=enc_below))

    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    with open("results/closest_approach_depth.csv", "w", encoding="utf-8") as fh:
        cols = list(table[0])
        fh.write(",".join(cols) + "\n")
        for r in table:
            fh.write(",".join(f"{r[c]:.6g}" for c in cols) + "\n")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    bins = np.logspace(np.log10(max(per_row.min(), 1e-3)), np.log10(50), 60)
    ax1.hist(per_row, bins=bins, alpha=0.55, label=f"all saved rows (n={len(per_row):,})")
    ax1.hist(per_enc, bins=bins, alpha=0.75,
             label=f"per-encounter minima (n={len(per_enc):,})")
    ax1.hist(per_run, bins=bins, alpha=0.9, label=f"per-run minimum (n={n_runs})")
    ax1.axvline(1 / np.sqrt(2), color="k", ls="--", lw=1,
                label=r"$\varepsilon/\sqrt{2}$ (accel. peak)")
    ax1.axvline(0.2, color="r", ls=":", lw=1, label="old innermost shell")
    ax1.set_xscale("log"); ax1.set_yscale("log")
    ax1.set_xlabel(r"minimum separation $/\varepsilon$"); ax1.set_ylabel("count")
    ax1.set_title("Encounter depth distribution")
    ax1.legend(fontsize=7)

    ts = np.logspace(np.log10(0.01), np.log10(3), 80)
    ax2.plot(ts, [(per_row < t).sum() for t in ts], label="rows below")
    ax2.plot(ts, [len(np.unique(per_row_run[per_row < t])) for t in ts],
             label="distinct runs contributing")
    ax2.plot(ts, [(per_run < t).sum() for t in ts], label="runs reaching")
    ax2.axvline(1 / np.sqrt(2), color="k", ls="--", lw=1)
    ax2.axvline(0.2, color="r", ls=":", lw=1)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel(r"depth threshold $/\varepsilon$"); ax2.set_ylabel("count below threshold")
    ax2.set_title("Cumulative coverage")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig("figures/closest_approach_depth.png", dpi=150)
    print("\nwrote results/closest_approach_depth.csv and "
          "figures/closest_approach_depth.png")


if __name__ == "__main__":
    main()
