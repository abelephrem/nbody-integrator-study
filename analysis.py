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

def angular_momentum_signed_drift(traj):
    """Signed relative change in L, projected onto the INITIAL L direction.

    angular_momentum_drift returns |L(t)-L0|/|L0| - a norm, so it is always
    >= 0 and cannot tell shrinking from growing. Here we take the component of
    L(t) along the unit vector u = L0/|L0| and compare it against |L0|, which
    keeps the sign: negative means the orbit is losing angular momentum.
    Returns a (T,) array.
    """
    L = np.array([
        angular_momentum(SystemState(traj.masses, traj.positions[t], traj.velocities[t]))
        for t in range(len(traj.times))
    ])
    L0_mag = np.linalg.norm(L[0])
    u = L[0] / L0_mag  # unit vector along the initial angular momentum
    return (L @ u - L0_mag) / L0_mag  # (T,) signed, relative to |L0|



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


def local_slopes(step_sizes, errors):
    """Slope of the log-log curve between each ADJACENT pair of points.
    
    Returns an (n-1,) array for n points.
    """
    return np.diff(np.log(errors)) / np.diff(np.log(step_sizes))


def fit_region(step_sizes, errors, tol=0.3):
    """Indicies of the straight middle of the log-log curve (integrator-dominated).
    Trims both ends: large-h(not-yet-assympototic) and small-h (round-off floor)."""
    logh = np.log10(step_sizes)
    loge = np.log10(errors)
    local = np.diff(loge) / np.diff(logh)  # slope between each adjacent pair
    med = np.median(local)  # robust estimate of the true order p, ignores the outliers
    clean = np.abs(local - med) < tol * abs(med)  # whch local slopes sit near the median

    # longest contiguous run of clean slops -> the straight region
    padded = np.concatenate(([False], clean, [False]))
    diffs = np.diff(padded.astype(int))  # +1 where a run starts, -1 where it ends
    starts = np.where(diffs == 1)[0]  # run start indices
    ends = np.where(diffs == -1)[0]  # run end indices (exclusive)
    k = np.argmax(ends - starts)  # the longest run
    return np.arange(starts[k], ends[k] + 1)  # +1: n slopes span n+1 points


def drift_growth_exponent(times, drift, threshold=1e-2):
    """Exponent q of the running-max envelope of |drift| against time.

    q ~ 0 means the error is bounded; q ~ 1 means it grows linearly in t.

    Two details, both load-bearing:
    - The RUNNING MAX is fitted, not |drift| itself. Leapfrog's drift oscillates
      through zero, so fitting the raw series would mostly measure where in its
      cycle each sample happened to land.
    - Only the part of the run BELOW `threshold` is fitted. Once the relative
      energy error reaches ~1% the trajectory is no longer the orbit that was
      asked for, so it has stopped measuring the integrator. Without this,
      Euler - which saturates at order-unity error within a fraction of an
      orbit - reads as 'bounded', the exact opposite of the truth.

    Returns (q, n_points, saturated).
    """
    envelope = np.maximum.accumulate(np.abs(drift))  # monotone: oscillation can't confuse it
    saturated = bool(envelope[-1] >= threshold)

    # t=0 has no logarithm, and neither does a still-zero envelope; drop those
    # along with everything past the threshold
    usable = (times > 0) & (envelope > 0) & (envelope < threshold)
    if usable.sum() < 2:  # nothing left to fit a line through
        return float("nan"), int(usable.sum()), saturated

    q, _ = np.polyfit(np.log(times[usable]), np.log(envelope[usable]), 1)
    return q, int(usable.sum()), saturated



def convergence_order(step_sizes, errors, fit_slice=None):
    """Fit log(error) = p*log(h) + c and return (p, c). p is the measured order."""
    h = np.asarray(step_sizes, dtype=float)
    err = np.asarray(errors, dtype=float)
    if fit_slice is not None:
        h = h[fit_slice]
        err = err[fit_slice]
    p, c = np.polyfit(np.log(h), np.log(err), 1)  # slope, intercept of the log-log line
    return p, c