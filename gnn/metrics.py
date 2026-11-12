"""GNN Stage C: G2 and G3 metrics, worked out from the saved rollout files."""

import os

import numpy as np

from gnn.run_rollouts import OUT_DIR


def load_rollout(job, name):
    """Saved checkpoints of one rollout. name is the .h5 filename from the split."""
    path = os.path.join(OUT_DIR, job, name.replace(".h5", ".npz"))
    with np.load(path) as saved:  # the file closes itself at the end of this block
        return saved["pos"], saved["vel"], saved["times"]  # (T, N, 3), (T, N, 3), (T,)


def about_com(masses, pos):
    """Positions (T, N, 3) measured from the centre of mass at each checkpoint."""
    com = (masses[:, None] * pos).sum(axis=1) / masses.sum()  # (T, 3) weighted mean over bodies
    return pos - com[:, None]  # com[:, None] is (T, 1, 3), so it subtracts from every body


def cluster_divergence(masses, pos, pos_ref):
    """delta(t): RMS distance between matching bodies, each run about its own centre of mass."""
    diff = about_com(masses, pos) - about_com(masses, pos_ref)  # (T, N, 3)
    return np.sqrt((diff**2).sum(axis=2).mean(axis=1))  # sum over xyz, mean over bodies: (T,)


def two_body_divergence(masses, pos, pos_ref):
    """|change in the relative orbit| / a, where a = m1 * m2 in this dataset."""
    rel = pos[:, 0] - pos[:, 1]  # (T, 3) body 0 minus body 1 at each checkpoint
    rel_ref = pos_ref[:, 0] - pos_ref[:, 1]
    return np.linalg.norm(rel - rel_ref, axis=1) / (masses[0] * masses[1])  # (T,)


def horizon(times, div, threshold=0.1):
    """First saved time div reaches the threshold, or infinity if it never does."""
    above = np.nonzero(div >= threshold)[0]  # indices of the rows at or above the threshold
    return times[above[0]] if above.size > 0 else np.inf  # above[0] is the earliest


def slope_window(div, lo=1e-4, hi=0.1):
    """Rows (first, last) of the G2 slope fit: first row >= lo after the last row below lo, to the first row >= hi."""
    reached = np.nonzero(div >= hi)[0]
    if reached.size == 0:  # never reaches hi, so there is no window
        return None
    last = reached[0]
    below = np.nonzero(div[:last] < lo)[0]  # rows before the end that are still below lo
    return (below[-1] + 1 if below.size else 0), last


def fit_slope(times, div, window):
    """Straight-line fit of log(div) against time over the window. Returns (slope, R²)."""
    first, last = window
    t, y = times[first:last + 1], np.log(div[first:last + 1])
    slope, intercept = np.polyfit(t, y, 1)  # degree 1: slope first, then intercept
    r2 = 1 - ((y - (slope * t + intercept)) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    return slope, r2


def energies(masses, pos, vel, softening, G=1.0):
    """Kinetic and softened potential energy at every checkpoint: (T,), (T,)."""
    kinetic = 0.5 * (masses * (vel**2).sum(axis=2)).sum(axis=1)  # sum over xyz, then over bodies
    disp = pos[:, None, :, :] - pos[:, :, None, :]  # (T, N, N, 3) every pair at every checkpoint
    inv_dist = ((disp**2).sum(axis=3) + softening**2) ** -0.5  # (T, N, N); diagonal is 1/softening, removed below
    pair = masses[:, None] * masses[None, :] * inv_dist
    pair[:, np.arange(len(masses)), np.arange(len(masses))] = 0.0  # drop the self terms
    potential = -0.5 * G * pair.sum(axis=(1, 2))  # 0.5 because each pair is counted twice
    return kinetic, potential


def energy_errors(masses, pos, vel, softening, G=1.0):
    """(E - E0)/|U0| and (E - E0)/|E0| at every checkpoint."""
    kinetic, potential = energies(masses, pos, vel, softening, G)
    total = kinetic + potential
    return (total - total[0]) / abs(potential[0]), (total - total[0]) / abs(total[0])


def body_angular_momenta(masses, pos, vel):
    """Each body's angular momentum about the centre of mass, in the centre-of-mass frame: (T, N, 3)."""
    com_vel = (masses[:, None] * vel).sum(axis=1) / masses.sum()  # (T, 3)
    return masses[:, None] * np.cross(about_com(masses, pos), vel - com_vel[:, None])


def angular_momentum_error(masses, pos, vel):
    """|L(t) - L(0)| / sum_i |L_i(0)| at every checkpoint."""
    L_i = body_angular_momenta(masses, pos, vel)
    L = L_i.sum(axis=1)  # (T, 3) total
    return np.linalg.norm(L - L[0], axis=1) / np.linalg.norm(L_i[0], axis=1).sum()


def late_early_ratio(drift):
    """Largest |drift| in the second half of the run divided by the largest in the first half (Stage 8 Q2)."""
    half = len(drift) // 2
    first = np.abs(drift[:half]).max()
    return np.abs(drift[half:]).max() / first if first > 0 else np.inf
