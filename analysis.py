"""Analysis and metrics — energy drift, convergence order fitting."""
import numpy as np
from bodies import SystemState
from simulation import run_simulation


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
    M = np.sum(state.masses)
    R_com = np.sum(state.masses[:, None] * state.positions, axis=0) / M  # (3,) COM position
    V_com = np.sum(state.masses[:, None] * state.velocities, axis=0) / M
    rel_pos = state.positions - R_com  # (N, 3)
    rel_vel = state.velocities - V_com  # (N, 3)
    cross = np.cross(rel_pos, rel_vel)  # (N, 3)
    L = np.sum(state.masses[:, None] * cross, axis=0)  # (3,)
    return L

def energy_drift(traj):
    """Relative energy drift (E(t) - E0)/|E0| at every saved snapshot.
    Returns a (T,) array, one value per timestep"""
    E = np.array([
        total_energy(SystemState(traj.masses, traj.positions[t], traj.velocities[t]),
                     G=traj.G, softening=traj.softening)
        for t in range(len(traj.times))
    ])
    return (E - E[0]) / abs(E[0])

def angular_momentum_drift(traj):
    """|L(t) - L0| / |L0| at every saved snapshot - the norm of the vector difference, not the change in magnitude.
    Returns a (T,) array"""
    L = np.array([
        angular_momentum(SystemState(traj.masses, traj.positions[t], traj.velocities[t]))
        for t in range(len(traj.times))
    ])
    L0 = L[0]  # (3,) initial angular momentum vector
    return np.linalg.norm(L - L0, axis=1) / np.linalg.norm(L0)


def kepler_solve(mean_anom, e, tol=1e-12, max_iter=100):
    """Solve Kepler's equation mean_anom = E - e*sin(E) for the eccentric anomly E, by Newton-Raphson"""
    E = np.array(mean_anom, dtype=float)  # initial guess E = mean_anom; copy so input isn't mutated
    for _ in range(max_iter):
        f = E - e * np.sin(E) - mean_anom  # how far E is from solving the equation
        if np.max(np.abs(f)) < tol:
            break
        f_prime = 1 - e * np.cos(E)  # derivative df/dE
        E = E - f / f_prime  # Newton step, applied to the whole array at once
    return E

def two_body_reference(times, a , e, m1, m2, G=1.0):
    """Exact relative-seperation vector r_rel(t) = r1 - r2 of the UNSOFTENED two_body Kepler orbit"""
    Mtot = m1 + m2
    n = np.sqrt(G * Mtot / a**3)  # K's 3rd law
    mean_anom = n * times  # (T,) 
    E = kepler_solve(mean_anom, e)  # (T,) eccentric anomaly at each timestamp

    r = a * (1 - e * np.cos(E))  # (T,) seperation distance, recovered from E
    nu = 2 * np.arctan2(np.sqrt(1 + e) * np.sin(E / 2),
                        np.sqrt(1 - e) * np.cos(E / 2))  # (T,) true anomaly, quadrant-safe
    
    x = r * np.cos(nu)
    y = r * np.sin(nu)
    z = np.zeros_like(x)
    return np.stack([x, y, z], axis=1) # (T, 3)

def position_error(traj, a, e):
    r_sim = traj.positions[:, 0, :] - traj.positions[:, 1, :]  # (T, 3) simulated seperation r1 - r2
    r_ref = two_body_reference(traj.times, a, e, traj.masses[0], traj.masses[1], G=traj.G)  # (T, 3) exact orbit
    return np.linalg.norm(r_sim - r_ref, axis=1)  # (T,) distance per timestep


def run_convergence_sweep(state, integrator, step_sizes, t_final, a, e, G=1.0):
    """Run the two_body orbit to t_final at each step size, with softening off and adaptive substepping off,
    amd return the global position error at t_final for each step."""
    errors=[]
    for h in step_sizes:
        n_steps= round(t_final/h)  # steps needed to reach t_final at this step
        traj = run_simulation(state, integrator, dt=h, n_steps=n_steps, scenario_name="two_body", G=G, softening=0.0, adaptive=False)
        assert traj.metadata["max_n_sub"] == 1  # substepping stayed dormant
        errors.append(position_error(traj, a, e)[-1])  # global error at the final time
    return np.array(errors)


def convergence_order(step_sizes, errors, fit_slice=None):
    """Fit log(error) = p*log(h) + c and return (p, c). p is the measured order."""
    h = np.asarray(step_sizes, dtype=float)
    err = np.asarray(errors, dtype=float)
    if fit_slice is not None:
        h = h[fit_slice]
        err = err[fit_slice]
    p, c = np.polyfit(np.log(h), np.log(err), 1)  # slope, intercept of the log-log line
    return p, c