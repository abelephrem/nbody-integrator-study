"""Stage C supplementary diagnostic: one-step error of kick-D3 and time-D3 on chosen two-body orbits, by decade of |a|.

Asks whether time-D3's collapse at mass ratio 15:1 is mass-ratio extrapolation (error at a given |a| much worse than
in-range orbits) or a sustained moderate error (error at a given |a| matches in-range). No rollout, no rule change.
"""

import csv
import math
import os

import h5py
import torch

from gnn.evaluate import collect, load_model
from gnn.g2_slopes import write_csv
from gnn.rollout import DATA_DIR, rollout_files
from gnn.train import OUT_DIR as MODEL_DIR

MODELS = {"design3_kick_200ep": "kick-D3", "design3_time_200ep": "time-D3"}
ORBITS = [("test", 7, 0.4), ("test", 10, 0.4), ("holdout", 5, 0.4), ("holdout", 15, 0.4)]  # (split, mass ratio, e)
STAGE_B = "results/stage_b/weighting_compare.csv"
OUT_CSV = "results/stage_c/one_step_check.csv"


def orbit_files(split, mass_ratio, e):
    """Every two-body file in the split with this mass ratio and eccentricity (holdout: 5 orientations of one orbit)."""
    names = []
    for name in rollout_files(split):
        if not name.startswith("two_body"):
            continue
        with h5py.File(os.path.join(DATA_DIR, name), "r") as f:
            if round(float(f.attrs["mass_ratio"])) == mass_ratio and abs(float(f.attrs["e"]) - e) < 1e-6:
                names.append(name)
    return names


def stage_b_two_body():
    """Stage B test-set two-body median error per decade: {(label, decade exponent): median}. Kick = seed 0 only."""
    out = {}
    with open(STAGE_B) as f:
        for r in csv.DictReader(f):
            if r["quantity"] != "rel two-body" or r["run"] not in ("kick seed 0", "time seed 0"):
                continue
            label = "kick-D3" if r["run"].startswith("kick") else "time-D3"
            out[(label, round(math.log10(float(r["range"].split(" to ")[0]))))] = float(r["value"])
    return out


def main():
    torch.set_num_threads(1)
    reference = stage_b_two_body()
    rows = []
    for job, label in MODELS.items():
        model = load_model(os.path.join(MODEL_DIR, f"{job}.pt"), "final_state")
        for split, mr, e in ORBITS:
            names = orbit_files(split, mr, e)
            bodies, _, _ = collect(model, names)
            decade = torch.floor(torch.log10(bodies["amag"])).long()  # e.g. |a| = 300 -> 2
            for d in sorted(set(decade.tolist())):
                err = bodies["rel"][decade == d]
                row = {"model": label, "split": split, "mass_ratio": mr, "e": e, "files": len(names),
                       "decade": f"1e{d}-1e{d + 1}", "n_bodies": len(err),
                       "median_rel": float(err.median()), "p90_rel": float(err.quantile(0.9)),
                       "stage_b_test_median": reference.get((label, d), float("nan"))}
                rows.append(row)
                print(f"{label:8s} {split:7s} {mr:2d}:1  |a| {row['decade']:9s} n {len(err):6d}  "
                      f"median {row['median_rel']:.2e}  p90 {row['p90_rel']:.2e}  Stage B {row['stage_b_test_median']:.2e}")
    write_csv(OUT_CSV, rows)


if __name__ == "__main__":
    main()
