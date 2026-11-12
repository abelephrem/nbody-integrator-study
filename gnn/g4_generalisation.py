"""GNN Stage C, G4: test vs holdout gap, with holdout clusters grouped by the reference's closest approach and by category.

Reads the CSVs written by g2_slopes.py and g3_conservation.py, so run those on both splits first.
"""

import csv
import os
import sys

import h5py
import numpy as np

from gnn.g2_slopes import RESULTS, write_csv
from gnn.rollout import DATA_DIR

MODELS = ["design3_kick_200ep", "design3_time_200ep"]  # holdout has kick-D3 and time-D3 only
DEPTH = 0.2  # closest approach in units of softening; training coverage is thin below this (09-25)


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def file_info(name):
    """Category and reference closest approach (in softening lengths) from the stored dataset file."""
    with h5py.File(os.path.join(DATA_DIR, name), "r") as f:
        depth = float(f.attrs["min_sep_run"]) / float(f.attrs["epsilon"])
        return {"category": str(f.attrs["split"]), "depth": depth,
                "depth_group": f"below {DEPTH} eps" if depth < DEPTH else f"above {DEPTH} eps"}


def group_stats(split, model, files):
    """G2 and G3 numbers for one model over one group of files."""
    files = set(files)
    ratios = [r for r in read_csv(f"{RESULTS}/g2_ratios_{split}.csv") if r["model"] == model and r["window"] == "main" and r["file"] in files]
    horizons = [float(r["horizon"]) for r in read_csv(f"{RESULTS}/g2_horizons_{split}.csv") if r["job"] == model and r["file"] in files]
    g3 = [r for r in read_csv(f"{RESULTS}/g3_{split}.csv") if r["job"] == model and r["file"] in files]
    ratio = np.array([float(r["ratio"]) for r in ratios])
    horizons = np.array(horizons)
    energy = np.array([float(r["max_abs_dE_U0"]) for r in g3])
    return {"n_files": len(files), "n_voting": len(ratio), "yes": int((ratio >= 1.5).sum()),
            "yes_share": (ratio >= 1.5).mean() if len(ratio) else np.nan,
            "median_ratio": np.nanmedian(ratio) if len(ratio) else np.nan,
            "reach_0.1": int(np.isfinite(horizons).sum()),
            "median_horizon": np.median(horizons[np.isfinite(horizons)]) if np.isfinite(horizons).any() else np.inf,
            "median_max_dE_U0": np.median(energy) if len(energy) else np.nan,
            "over_tolerance": sum(r["over_tolerance"] == "True" for r in g3),
            "median_max_dL": np.median([float(r["max_dL"]) for r in g3]) if g3 else np.nan}


def main(test_split="test", holdout_split="holdout"):
    files = {s: sorted({r["file"] for r in read_csv(f"{RESULTS}/g3_{s}.csv")}) for s in (test_split, holdout_split)}
    rows = []
    for model in MODELS:
        for kind in ("cluster", "two_body"):
            for s in (test_split, holdout_split):
                group = [n for n in files[s] if n.startswith(kind)]
                rows.append({"model": model, "split": s, "group": f"all {kind}", **group_stats(s, model, group)})
        clusters = [n for n in files[holdout_split] if n.startswith("cluster")]
        info = {n: file_info(n) for n in clusters}
        for key in ("depth_group", "category"):
            for value in sorted({i[key] for i in info.values()}):
                group = [n for n in clusters if info[n][key] == value]
                rows.append({"model": model, "split": holdout_split, "group": value, **group_stats(holdout_split, model, group)})
    write_csv(f"{RESULTS}/g4_{holdout_split}.csv", rows)

    print(f"{'model':20s} {'split':8s} {'group':18s} files voting yes  share  ratio  reach  horizon  max|dE/U0| over_tol  max dL")
    for r in rows:
        print(f"{r['model']:20s} {r['split']:8s} {r['group']:18s} {r['n_files']:5d} {r['n_voting']:6d} {r['yes']:3d} {r['yes_share']:6.2f} "
              f"{r['median_ratio']:6.2f} {r['reach_0.1']:6d} {r['median_horizon']:8.2f} {r['median_max_dE_U0']:10.1e} {r['over_tolerance']:8d} {r['median_max_dL']:7.1e}")


if __name__ == "__main__":
    main(*sys.argv[1:3])  # e.g. "test holdout"; "test test" is a dry run that gives a zero gap
