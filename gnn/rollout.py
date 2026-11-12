"""GNN Stage C: adaptive Leapfrog rollouts with a swappable kick force (true or GNN)."""
import os

import h5py
import numpy as np
import torch

from bodies import SystemState
from forces import compute_accelerations
from gnn.split import DATA_DIR, load_split
from gnn.graph_data import complete_edge_index

FLAGGED = {"cluster_01216.h5"}  # failed the energy check (C = 2.7e-2), so never used as a rollout reference



def true_force_func(masses, softening, G=1.0):
    """Build a force function for true softened gravity: positions (N, 3) -> accelerations (N, 3)."""

    def force(pos):
        state = SystemState(masses, pos, None)  # forces only read masses and positions, so velocities can be None
        return compute_accelerations(state, G=G, softening=softening)

    return force  # the inner function remembers masses, softening and G


def kdk_substep(pos, vel, acc, force, h):
    """One kick-drift-kick substep of size h. acc is the acceleration at pos, already known."""
    v_half = vel + 0.5 * h * acc  # first half kick, reusing the known acceleration
    new_pos = pos + h * v_half  # full drift, using the half-updated velocity

    new_acc = force(new_pos)  # the only force call in this substep
    new_vel = v_half + 0.5 * h * new_acc  # second half kick

    return new_pos, new_vel, new_acc  # new_acc becomes acc in the next substep


def core_step_cap(masses, softening, G=1.0, n_core=100):
    """Largest step allowed during a close encounter: a fixed fraction of the tightest possible core orbit."""
    heaviest_pair = np.sort(masses)[-2:].sum()  # sort ascending, take the last two: the shortest orbit period
    t_core = 2 * np.pi * np.sqrt(softening**3 / (G * heaviest_pair))  # orbit period inside the softening core
    return t_core / n_core  # constant for a run, so it only needs working out once


def choose_substep(pos, acc_true, softening, cap, time_left, pairs, n_resolve=96, event_factor=3.0):
    """Size of the next substep, by the dataset's rule. acc_true is always the TRUE acceleration at pos."""
    a_max = np.max(np.linalg.norm(acc_true, axis=1))  # strongest acceleration on any body
    h = np.sqrt(softening / a_max) / n_resolve if a_max > 0 else time_left  # zero force: nothing limits the step

    min_sep = np.linalg.norm(pos[pairs[0]] - pos[pairs[1]], axis=1).min()  # distance between the closest pair
    if min_sep < event_factor * softening and h > cap:  # only inside an encounter, and only if h is too big
        h = cap

    return min(h, time_left)  # never step past the next checkpoint


def rk4_substep(pos, vel, acc, force, h):
    """One RK4 substep of size h, same inputs and outputs as kdk_substep. acc is the acceleration at pos, already known."""
    kx1, kv1 = vel, acc  # stage 1: slopes where we are, reusing the known acceleration

    kx2 = vel + 0.5 * h * kv1  # stage 2: slopes half a step ahead, using stage 1
    kv2 = force(pos + 0.5 * h * kx1)

    kx3 = vel + 0.5 * h * kv2  # stage 3: half a step ahead again, using stage 2
    kv3 = force(pos + 0.5 * h * kx2)

    kx4 = vel + h * kv3  # stage 4: a full step ahead, using stage 3
    kv4 = force(pos + h * kx3)

    new_pos = pos + (h / 6) * (kx1 + 2 * kx2 + 2 * kx3 + kx4)  # weighted average; the middle stages count double
    new_vel = vel + (h / 6) * (kv1 + 2 * kv2 + 2 * kv3 + kv4)

    return new_pos, new_vel, force(new_pos)  # this acceleration becomes acc in the next substep



def adaptive_rollout(masses, pos0, vel0, dt, n_steps, softening, kick_force=None, G=1.0,
                     n_resolve=96, n_core=100, event_factor=3.0, substep=kdk_substep):
    """Adaptive Leapfrog run, saved at every outer checkpoint. Steps always come from the true force.

    kick_force = None kicks with the true force; otherwise it's a function positions (N, 3) -> accelerations (N, 3).
    Returns positions (n_steps + 1, N, 3), velocities (same shape), times (n_steps + 1,) and the substep count.
    """
    true_force = true_force_func(masses, softening, G)  # always needed for the step rule
    kick = true_force if kick_force is None else kick_force  # the force that actually moves the bodies
    cap = core_step_cap(masses, softening, G, n_core)  # constant, so worked out once
    pairs = np.triu_indices(len(masses), k=1)  # each pair of bodies once, built once

    all_pos = np.zeros((n_steps + 1, *pos0.shape))  # *pos0.shape unpacks (N, 3), giving (n_steps + 1, N, 3)
    all_vel = np.zeros((n_steps + 1, *vel0.shape))
    all_pos[0], all_vel[0] = pos0, vel0  # row 0 is the starting state

    pos, vel = pos0, vel0
    acc = kick(pos)  # starting acceleration, reused by the first substep
    n_sub = 0  # substeps taken across the whole run

    for i in range(1, n_steps + 1):  # one pass per outer checkpoint
        elapsed = 0.0  # time covered so far within this dt

        while dt - elapsed > 1e-12:  # same stopping test as run_simulation, so replays match exactly
            acc_true = acc if kick_force is None else true_force(pos)  # true-force runs reuse acc; GNN runs need one extra call
            h = choose_substep(pos, acc_true, softening, cap, dt - elapsed, pairs, n_resolve, event_factor)
            pos, vel, acc = substep(pos, vel, acc, kick, h)  # Leapfrog by default; RK4 if asked for
            elapsed += h
            n_sub += 1

        all_pos[i], all_vel[i] = pos, vel  # save only the checkpoint; substeps are discarded

    times = np.arange(n_steps + 1) * dt  # same i * dt as run_simulation
    return all_pos, all_vel, times, n_sub


def rollout_files(split_name):
    """Filenames in one part of the split, leaving out flagged references."""
    return [name for name in load_split()[split_name] if name not in FLAGGED]


def read_run(name):
    """Everything a rollout needs from one stored HDF5 file, as a dictionary."""
    with h5py.File(os.path.join(DATA_DIR, name), "r") as f:  # the file closes itself at the end of this block
        return {
            "masses": f["masses"][:],  # (N,); [:] copies the dataset into a NumPy array
            "pos": f["positions"][:],  # (rows, N, 3) float64
            "vel": f["velocities"][:],  # (rows, N, 3) float64
            "times": f["times"][:],  # (rows,) checkpoints, plus event rows in clusters
            "dt": float(f.attrs["dt"]),  # the outer step between checkpoints
            "n_steps": int(f.attrs["N_steps"]),  # outer steps taken, not rows stored
            "softening": float(f.attrs["epsilon"]),
            "G": float(f.attrs["G"]),
            "n_capped": int(f.attrs["n_capped"]),  # substeps where the encounter cap fired
        }


def gnn_force_func(model, masses):
    """Build a force function from a trained model: float64 positions (N, 3) -> float64 accelerations (N, 3)."""
    model.eval()  # evaluation mode, as the plan requires
    x = torch.from_numpy(masses[:, None]).float()  # (N, 1) float32 masses, same every call
    edge_index = complete_edge_index(len(masses))  # (2, E) every ordered pair, same every call
    send, recv = edge_index  # row 0 = senders, row 1 = receivers

    def force(pos):
        pos_t = torch.from_numpy(pos)  # (N, 3) a float64 tensor sharing pos's memory
        disp = pos_t[send] - pos_t[recv]  # (E, 3) r_sender - r_receiver, subtracted in float64 as in Stage A
        dist = disp.norm(dim=-1, keepdim=True)  # (E, 1) length of each displacement
        edge_attr = torch.cat([disp, dist], dim=-1).float()  # (E, 4) joined, then cast to float32

        with torch.no_grad():  # no gradients needed: saves time and memory
            acc = model(x, edge_index, edge_attr)  # (N, 3) float32 predicted accelerations

        return acc.double().numpy()  # back to float64 NumPy for the integrator

    return force
