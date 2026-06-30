"""Pairwise gravitational force and acceleration calculations."""

import numpy as np


def compute_accelerations(state, G=1.0, softening=0.0):
    positions = state.positions  # (N, 3)
    masses = state.masses  # (N,)
    disp = positions[None, :, :] - positions[:, None, :]  # (N, N, 3), disp[i,j] = r_j - r_i
    dist_sq = np.sum(disp**2, axis=2) + softening**2  # (N, N)
    with np.errstate(divide="ignore"):
        inv_dist_cubed = dist_sq ** -1.5  # diagonal is infinite here, zeroed next line
    np.fill_diagonal(inv_dist_cubed, 0.0)  # point mass exerts no force on itself
    weights = masses[None, :] * inv_dist_cubed # (N, N), weights[i,j] = m_j / r_ij^3
    weighted_disp = weights[:, :, None] * disp  # (N, N, 3)
    accelerations = G * np.sum(weighted_disp, axis=1)  # (N, 3)
    return accelerations
