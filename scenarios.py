"""Initial condition scenarios — 2-body, figure-8, chaotic cluster."""
import numpy as np
from bodies import Body, SystemState, bodies_to_state
from analysis import total_energy
from dataclasses import dataclass, field

# Sweep axis definitions (one-axis-at-a-time holdout design)
E_TRAINED = [0.0, 0.2, 0.4, 0.8]
E_INTERP, E_EXTRAP = 0.6, 0.9

RATIO_TRAINED = [1, 3, 7, 10]  # mass ratio
RATIO_INTERP, RATIO_EXTRAP = 5,15

N_TRAINED = [3, 4, 5, 7, 8]
N_INTERP, N_EXTRAP = 6, 10

# Q is continous - trained region is two intervals, sampled by LHS (cluster only)
Q_TRAINED_INTERVALS = [(0.3, 0.7), (1.1, 1.5)]
Q_INTERP_INTERVAL = (0.7, 1.1)
Q_EXTRAP_INTERVAL = (1.6, 1.8)

# Baseline: the (trained) value an axis holds while a different axis is varied
E_BASELINE = 0.4  # mid-range trained value
RATIO_BASELINE = 1  # simplest mass case
N_BASELINE = 5  # mid-range trained value
Q_BASELINE = 1.3  # mid-range trained value


def _kepler_two_body(m1, m2, a, e, G=1.0):
    """Two-body orbit built at periapsis (v_r=0), returned in the COM frame.
    Orbit lies in the xy-plane. Requires 0<= e< 1 (bound orbit)"""

    M = m1 + m2
    if not (0 <= e < 1):
        raise ValueError(f"e must be in [0, 1); got {e}")
    r_peri = a * (1 - e)
    v_peri = np.sqrt(G * M / a * (1 + e) / (1 - e))  # from vis-visa eq

    r_rel = np.array([r_peri, 0.0, 0.0])  # position on +x axis
    v_rel = np.array([0.0, v_peri, 0.0])  # velocity purely tangential (+y)

    r1 = (m2 / M) * r_rel
    r2 = -(m1 / M) * r_rel
    v1 = (m2 / M) * v_rel
    v2 = -(m1 / M) * v_rel

    b1 = Body(m1, r1, v1)
    b2 = Body(m2, r2, v2)
    return bodies_to_state([b1, b2])


def two_body_circular(m1=1.0, m2=1.0, r=1.0, G=1.0):
    """Circular orbit: e=0, orbital radius (seperation) = r."""
    return _kepler_two_body(m1, m2, a=r, e=0.0, G=G)


def two_body_eccentric(m1=1.0, m2=1.0, a=1.0, e=0.5, G=1.0):
    """Eccentric orbit, started at periapsis."""
    return _kepler_two_body(m1, m2, a=a, e=e, G=G)


def figure_eight():
    """Three equal masses on the figure-eight choreography (Moore 1993).
    G=1, m=1; published initial conditions, planar (z=0)."""
    r1 = np.array([-0.97000436, 0.24308753, 0.0])
    r3 = np.array([0.0,0.0, 0.0])
    v3 = np.array([0.93240737, 0.86473146, 0.0])

    b1 = Body(1.0, r1, -v3 / 2)
    b2 = Body(1.0, -r1, -v3 / 2)
    b3 = Body(1.0, r3, v3)
    return bodies_to_state([b1, b2, b3])


def _zero_com(state):
    """Return a copy of state shifted into the center-of-mass rest frame:
    COM at the origin, total momentum zero."""
    
    M = np.sum(state.masses)
    com_pos = np.sum(state.masses[:, None] * state.positions, axis=0) / M
    com_vel = np.sum(state.masses[:, None] * state.velocities, axis=0) / M
    new_pos = state.positions - com_pos
    new_vel = state.velocities - com_vel
    
    return SystemState(state.masses, new_pos, new_vel)


def chaotic_cluster(N=5, seed=None, pos_scale=1.0, vel_scale=0.5, G=1.0):
    """N equal-mass bodies with random positions/velocities, in the COM frame.
    Rejection-samples until the system is bound (E < 0)"""

    rng = np.random.default_rng(seed)  # seeded, reproducible for tests
    while True:
        masses = np.ones(N)
        positions = rng.normal(0.0, pos_scale, size=(N, 3))
        velocities = rng.normal(0.0, vel_scale, size=(N, 3))  # slow --> more often bound
        state = _zero_com(SystemState(masses, positions, velocities))
        if total_energy(state, G=G) < 0:  # keep only bound draws
            return state
        

def virial_radius(state, G=1.0):
    """Virial length scale R = GM^2 / 2(|E|), using the conserved total energy.
    Requires a bound system (E < 0). Two body check: R = M^2*a / (m1*m2)"""

    M = np.sum(state.masses)
    E = total_energy(state, G=G)
    if E>= 0:
        raise ValueError(f"system is unbound (E={E} >= 0); virial R undefined")
    return G * M**2 / (2 * abs(E))


def nondimensionalise(state, G=1.0):
    """Rescale to natual units: G=1, total mass=1, length unit R, time unit sqrt(R^3/GM). Assumes state is already in the COM rest frame."""
    
    M = np.sum(state.masses)
    R = virial_radius(state, G=G)
    v_unit = np.sqrt(G * M / R)  # velocity scales by L_unit / T_unit = sqrt(GM/R)
    
    masses = state.masses / M
    positions = state.positions / R 
    velocities = state.velocities / v_unit
    return SystemState(masses, positions, velocities)


@dataclass
class TrajectoryConfig:
    scenario_type: str  # "two_body" | "cluster"
    params: dict  # physical axis values, e.g. {"e": 0.4, "mass_ratio": 3}
    split: str  # "train" | "e_interp" | "e_extrap" | "ratio_interp" | ...
    seed: int  # RNG seed for this trajectory's randomness (orientation/geometry)


def two_body_configs(n_orient=5, seed0=0):
    """ALl two-body trajectory configs: trained cross-product + per-axis holdout.
    n_orient = random-orientation replicates per (e, ratio) cell."""
    configs = []
    s = seed0

    # Training set: full cross product of TRAINED e x TRAINED mass_ratio
    for e in E_TRAINED:
        for ratio in RATIO_TRAINED:
            for _ in range(n_orient):
                configs.append(TrajectoryConfig(
                    "two_body", {"e": e, "mass_ratio": ratio}, "train", s))
                s += 1
    
    # Generalisation: vary ONE axis off-distribution, hold the other at baseline
    for e, split in [(E_INTERP, "e_interp"), (E_EXTRAP, "e_extrap")]:
        for _ in range (n_orient):
            configs.append(TrajectoryConfig(
                "two_body", {"e": e, "mass_ratio": RATIO_BASELINE}, split, s
            ))
            s += 1
    
    for ratio, split in [(RATIO_INTERP, "ratio_interp"), (RATIO_EXTRAP, "ratio_extrap")]:
        for _ in range(n_orient):
            configs.append(TrajectoryConfig(
                "two_body", {"e": E_BASELINE, "mass_ratio": ratio}, split, s
            ))
            s += 1

    return configs


def _stratified_over_intervals(intervals, n, rng):
    """n evenly-spread samples across disjoint intervals (1-D Latin hypercube).
    Intervals laid end-to-end, stratified into n bins, one jittered sample per bin, then each matched back to its real value (skipping the gaps).
    """
    lengths = [hi - lo for lo, hi in intervals]
    total = sum(lengths)
    edges = np.linspace(0.0, total, n + 1)  # n equal bins along [0, total]
    u = edges[:-1] + rng.random(n) * np.diff(edges)  # one jittered sample per bin

    values = []
    for ui in u:
        pos = ui
        for (lo, hi), L in zip(intervals, lengths):
            if pos <= L:  # sample falls inside this interval's length
                values.append(lo + pos)  # real value = interval's start + how far in
                break  # found it - stop scanning intervals
            pos -= L  # else step past it into the next interval
    return np.array(values)
            


def cluster_configs(n_draws=10, seed0=1000):
    """All cluster trajectory configs: trained cross-product + per-axis holdouts.
    n_draws = replicate draws per discrete (N, ratio) cell (Q via LHS, geometry via seed)."""
    configs = []
    s = seed0
    rng = np.random.default_rng(seed0)  # drives the Q sampling

    # Training: N x ratio_cells; per cell, n_draws replicates each with an LHS Q
    for N in N_TRAINED:
        for ratio in RATIO_TRAINED:
            for Q in _stratified_over_intervals(Q_TRAINED_INTERVALS, n_draws, rng):
                configs.append(TrajectoryConfig(
                    "cluster", {"N": N, "Q": Q, "mass_ratio": ratio}, "train", s
                ))
                s+=1

    # N holdout: vary N, hold Q and ratio at baseline
    for N, split in [(N_INTERP, "N_interp"), (N_EXTRAP, "N_extrap")]:
        for _ in range(n_draws):
            configs.append(TrajectoryConfig(
                "cluster", {"N": N, "Q": Q_BASELINE, "mass_ratio": RATIO_BASELINE}, split, s
            ))
            s += 1
    
    # Q holdout: sample Q from the interp/extrap interval, hold N and ratio at baseline
    for interval, split in [(Q_INTERP_INTERVAL, "Q_interp"), (Q_EXTRAP_INTERVAL, "Q_extrap")]:
        for Q in _stratified_over_intervals([interval], n_draws, rng):
            configs.append(TrajectoryConfig(
                "cluster", {"N": N_BASELINE, "Q": Q, "mass_ratio": RATIO_BASELINE}, split, s
            ))
            s+=1
    
    # mass_ratio holdout: vary ratio, hold N and Q at baseline
    for ratio, split in [(RATIO_INTERP, "ratio_interp"), (RATIO_EXTRAP, "ratio_extrap")]:
        for _ in range(n_draws):
            configs.append(TrajectoryConfig(
                "cluster", {"N": N_BASELINE, "Q": Q_BASELINE, "mass_ratio": ratio}, split, s))
            s += 1

    return configs  
