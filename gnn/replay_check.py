"""GNN Stage C: check that a true-force rollout reproduces the stored checkpoints exactly."""

import csv
import os
import sys
import time

import numpy as np

from gnn.rollout import adaptive_rollout, read_run, rollout_files

OUT_DIR = "results/stage_c"


def check_file(name):
    """Replay one stored file and compare it with every stored checkpoint row."""
    run = read_run(name)

    start = time.perf_counter()  # to time the replay
    all_pos, all_vel, times, n_sub = adaptive_rollout(run["masses"], run["pos"][0], run["vel"][0], run["dt"],
                                                      run["n_steps"], run["softening"], G=run["G"])
    seconds = time.perf_counter() - start

    rows = np.isin(run["times"], times)  # True for stored rows whose time is exactly a checkpoint time
    steps = np.rint(run["times"][rows] / run["dt"]).astype(int)  # which rollout row each of those stored rows matches

    return {
        "file": name,
        "rows_stored": len(run["times"]),
        "rows_compared": int(rows.sum()),  # the rest are event rows captured mid-substep
        "n_sub": n_sub,
        "n_capped": run["n_capped"],
        "max_dpos": float(np.abs(all_pos[steps] - run["pos"][rows]).max()),  # worst position difference; 0 = exact
        "max_dvel": float(np.abs(all_vel[steps] - run["vel"][rows]).max()),  # worst velocity difference; 0 = exact
        "seconds": seconds,
    }


if __name__ == "__main__":
    split_name = sys.argv[1]  # e.g. python -m gnn.replay_check test
    os.makedirs(OUT_DIR, exist_ok=True)  # create the folder if it doesn't exist yet

    results = []
    for name in rollout_files(split_name):
        results.append(check_file(name))
        r = results[-1]  # the result just added
        print(f"{r['file']}  compared {r['rows_compared']}/{r['rows_stored']}  capped {r['n_capped']}  "
              f"max dpos {r['max_dpos']:.1e}  max dvel {r['max_dvel']:.1e}  {r['seconds']:.0f}s", flush=True)

    with open(os.path.join(OUT_DIR, f"replay_check_{split_name}.csv"), "w", newline="") as f:  # newline="": no blank rows on Windows
        writer = csv.DictWriter(f, fieldnames=results[0].keys())  # column names = the dictionary's keys
        writer.writeheader()
        writer.writerows(results)

    exact = sum(r["max_dpos"] == 0 and r["max_dvel"] == 0 for r in results)  # True counts as 1
    capped = sum(r["n_capped"] > 0 for r in results)
    print(f"{split_name}: {exact}/{len(results)} files exact; cap fired in {capped} files")
