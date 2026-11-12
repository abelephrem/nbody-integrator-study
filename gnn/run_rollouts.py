"""GNN Stage C: run rollouts and save their checkpoints. Metrics are worked out later from the saved files."""

import os
import sys
import time

import numpy as np
import torch

from gnn.evaluate import load_model
from gnn.rollout import adaptive_rollout, gnn_force_func, kdk_substep, read_run, rk4_substep, rollout_files
from gnn.train import OUT_DIR as MODEL_DIR

OUT_DIR = "C:/nbody_results/stage_c"  # outside OneDrive, as in Stage B
NUDGE = 1e-6  # size of the chaos baseline's starting shift
BASELINE_SEEDS = {"baseline2": 2, "baseline3": 3}  # extra chaos baselines; the original baseline keeps its own seed


def file_number(name):
    """The number in a filename: cluster_01001.h5 -> "01001.h5" -> "01001" -> 1001."""
    return int(name.split("_")[-1].split(".")[0])


def nudged_start(masses, pos0, seed):
    """Starting positions shifted randomly, with no centre-of-mass shift and an RMS of exactly NUDGE."""
    rng = np.random.default_rng(seed)  # same seed gives the same nudge
    nudge = rng.normal(size=pos0.shape)  # (N, 3) random direction and size for each body
    nudge -= (masses[:, None] * nudge).sum(axis=0) / masses.sum()  # remove the centre-of-mass shift
    nudge *= NUDGE / np.sqrt((nudge**2).sum(axis=1).mean())  # rescale so sqrt(mean_i |nudge_i|²) = NUDGE
    return pos0 + nudge


def run_and_save(path, run, start_pos, kick_force=None, substep=kdk_substep):
    """Run one rollout and save its checkpoints, unless the file already exists."""
    if os.path.exists(path):  # finished earlier, so a restart doesn't repeat it
        print(f"{path} exists - skipped", flush=True)
        return

    os.makedirs(os.path.dirname(path), exist_ok=True)  # create the folder if it doesn't exist yet

    start = time.perf_counter()  # to time the rollout
    all_pos, all_vel, times, n_sub = adaptive_rollout(run["masses"], start_pos, run["vel"][0], run["dt"],
                                                      run["n_steps"], run["softening"], kick_force=kick_force, G=run["G"], substep=substep)
    seconds = time.perf_counter() - start

    tmp = path + f".{os.getpid()}.tmp.npz"  # process id keeps two processes on one file apart; .npz stops np.savez adding its own
    np.savez(tmp, pos=all_pos, vel=all_vel, times=times, n_sub=n_sub, seconds=seconds)  # each keyword names a saved array
    os.replace(tmp, path)  # rename only when complete, so a crash can't leave a half-written file at path

    print(f"{path}  substeps {n_sub}  {seconds:.0f}s", flush=True)


if __name__ == "__main__":
    split_name = sys.argv[1]  # "test" or "holdout"
    job = sys.argv[2]  # "reference", "rk4", "baseline2", "baseline3" or a model name such as design3_kick_200ep
    torch.set_num_threads(1)  # one thread per process, because several run side by side

    model = None  # the reference, rk4 and extra baseline jobs use no model
    if job not in ("reference", "rk4") and job not in BASELINE_SEEDS:  # only model jobs load a model
        model = load_model(os.path.join(MODEL_DIR, f"{job}.pt"), "final_state")  # final weights, chosen explicitly

    names = rollout_files(split_name)
    if len(sys.argv) > 3 and sys.argv[3] == "reverse":  # a second process works from the other end of the list
        names = names[::-1]

    for name in names:
        run = read_run(name)
        start_pos = run["pos"][0]  # row 0 of the stored file
        file = name.replace(".h5", ".npz")

        if job == "reference":
            run_and_save(os.path.join(OUT_DIR, "reference", file), run, start_pos)

            if name.startswith("cluster"):  # two-body has no chaos baseline
                seed = file_number(name)  # same seed as before, so the saved baseline stays valid
                run_and_save(os.path.join(OUT_DIR, "baseline", file), run, nudged_start(run["masses"], start_pos, seed))
        
        elif job == "rk4":
            run_and_save(os.path.join(OUT_DIR, "rk4", file), run, start_pos, substep=rk4_substep)

        elif job in BASELINE_SEEDS:
            if name.startswith("cluster"):  # two-body has no chaos baseline
                seed = [file_number(name), BASELINE_SEEDS[job]]  # a list gives a separate random stream for each baseline
                run_and_save(os.path.join(OUT_DIR, job, file), run, nudged_start(run["masses"], start_pos, seed))

        else:
            run_and_save(os.path.join(OUT_DIR, job, file), run, start_pos, gnn_force_func(model, run["masses"]))
