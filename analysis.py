"""Analysis and metrics — energy drift, convergence order fitting."""
import numpy as np


def total_energy(state, G=1.0, softening=0.0):
    speeds_sq = np.sum(state.velocities**2, axis=1)  # (N,) : |v_i|^2, sum over x,y,z
    T = 0.5 * np.sum(state.masses * speeds_sq)  # scalar

    disp = state.positions[None, :, :] - state.positions[:, None, :]  # (N,N,3)
    dist_sq = np.sum(disp**2, axis=2) + softening**2  # (N, N)
    with np.errstate(divide="ignore"):
        inv_dist = dist_sq**-0.5  # 1/sqrt(r^2+eps^2); diagonal may be inf
    mass_products = state.masses[:, None] * state.masses[None, :]  # (N,N): m_i*m_j
    pair_energy = mass_products * inv_dist  # (N,N)
    np.fill_diagonal(pair_energy, 0.0)  # drop self_terms (i=j)
    U = -G * 0.5 * np.sum(pair_energy)  # 0.5 because the full sum counts i<j and j<i
    
    return T + U


def angular_momentum(state):
    cross = np.cross(state.positions, state.velocities)  # (N,3): r_i * v_i per body
    L = np.sum(state.masses[:, None] * cross, axis=0)  # (3,): total L vector
    return L


