"""Initial condition scenarios — 2-body, figure-8, chaotic cluster."""
import numpy as np
import glob
import h5py
from bodies import Body, SystemState, bodies_to_state
from analysis import total_energy
from dataclasses import dataclass, field, replace
from scipy.spatial.transform import Rotation
import os 
from simulation import run_simulation, save_trajectory
from integrators import leapfrog_step


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


def chaotic_cluster(N=5, Q=1.0, seed=None, mass_ratio=1.0, pos_scale=1.0, G=1.0):
    """N equal-mass bodies, random geometry, COM frame, scaled to target virial ratio Q = 2T / |U|. Q < 2 --> bound (no rejection needed)"""
    rng = np.random.default_rng(seed)
    masses = np.ones(N)
    masses[0] = mass_ratio  # one heavy body; rest stay at mass 1
    positions = rng.normal(0.0, pos_scale, size=(N, 3))
    velocities = rng.normal(0.0, 1.0, size=(N, 3))  # random directions; magnitude set by Q below
    state = _zero_com(SystemState(masses, positions, velocities))

    T = 0.5 * np.sum(state.masses * np.sum(state.velocities**2, axis=1))
    U = total_energy(state, G=G) - T
    lam = np.sqrt(Q / (2 * T / abs(U)))  # lambda = sqrt(Q_target / Q_current)
    velocities = state.velocities * lam
    return SystemState(state.masses, state.positions, velocities)

    
        

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


def random_orientation(state, seed=None):
    """Roatate positions and velocities by a uniformly-random 3D rotation.
    Gravity is rotation-invariant, so the physics is unchanged - this only varies 
    the orbital plane's orientation."""
    rng = np.random.default_rng(seed)
    R = Rotation.random(random_state=rng).as_matrix()  # (3,3) Haar-uniform rotation
    positions = state.positions @ R.T  # applies the rotation to every body's position vector at once
    velocities = state.velocities @ R.T  # same R for positions and velocities
    return SystemState(state.masses, positions, velocities)


def build_initial_state(config):
    """Turn a TrajectoryConfig into a nondimenionalised initial SystemState"""
    p = config.params
    if config.scenario_type == "two_body":
        state = two_body_eccentric(m1=p["mass_ratio"], m2=1.0, a=1.0, e=p["e"])  # a=1.0 is arbitary, always produces same nondimensional system
        state = random_orientation(state, seed=config.seed)  # seed.config drives random 3D orientation
    elif config.scenario_type == "cluster":
        state = chaotic_cluster(N=p["N"], Q=p["Q"], mass_ratio=p["mass_ratio"], seed=config.seed)  # seed.config drives random geometry
    else: raise ValueError(f"unknown scenario_type: {config.scenario_type}")

    return nondimensionalise(state)  # applied to both scenarios at end


def resample_uniform(traj, n_keep):
    """Keep n_keep evenly-spaced (uniform in time) snapshots. Placeholder cadence."""  
    idx = np.linspace(0, len(traj.times) - 1, n_keep, dtype=int)
    return replace(
        traj,
        positions=traj.positions[idx],
        velocities=traj.velocities[idx],
        accelerations=traj.accelerations[idx],
        times=traj.times[idx]
    )


def generate_dataset(configs, out_dir, duration=20.0, dt=0.01, n_keep=200, n_baseline=100, softening=0.002):
    """Run every config through Leapfrog; save a tagged, resampled HDF5 trajectory each."""
    
    os.makedirs(out_dir, exist_ok=True)  # creates the output folder if it doesn't exist
    n_steps = round(duration/dt)  # duration is in nondimensional time units
    for config in configs:
        state = build_initial_state(config)
        traj = run_simulation(state, leapfrog_step, dt=dt, n_steps=n_steps, scenario_name=config.scenario_type, 
                              G=1.0, softening=softening, adaptive=True)
        # choose the cadence that fits the scenario
        if config.scenario_type == "two_body":
            traj = resample_true_anomaly(traj, n_keep)
        elif config.scenario_type == "cluster":
            traj = resample_events(traj, n_baseline)
        else:
            raise ValueError(f"unknown scensrio_type: {config.scenario_type}")
        traj.metadata = {
            **traj.metadata,  # preserve max_n_sub set by run_simulation
            "scenario_type": config.scenario_type,
            "split": config.split,
            "seed": config.seed,
            **config.params,  # spills in every key from params
        }
        path = os.path.join(out_dir, f"{config.scenario_type}_{config.seed:05d}.h5")
        save_trajectory(traj, path)


def resample_true_anomaly(traj, n_keep):
    """Select n_keep snapshots evenely spaced in true anomaly (two-body).
    Denser near periapsis, at any eccentricity."""
    rel = traj.positions[:, 0, :] - traj.positions[:, 1, :]  # (T, 3) relative position
    vel = traj.velocities[:, 0, :] - traj.velocities[:, 1, :]  # (T, 3) relative velocity

    u = rel[0] / np.linalg.norm(rel[0])  # periapsis unit vector (orbit starts here)
    w = np.cross(rel[0], vel[0])
    w = w / np.linalg.norm(w)  # orbit normal (constant)
    p = np.cross(w, u)  # in-plane axis perpendicular to u
    x = rel @ u  # (T,) component along periapsis, x-axis pinned to peripasis direction
    y = rel @ p  # (T,) in-plane perpendicular component, y-axis pinned to this directio 
    theta = np.unwrap(np.arctan2(y, x))  # continuous true anomaly, 0 -> 2pi -> ...

    grid = np.linspace(theta[0], theta[-1], n_keep)
    idx = np.searchsorted(theta, grid)  # returns the thetas we want, orders them correctly
    idx = np.clip(idx, 0, len(theta) - 1)  # makes sure the last theta safely maps to the final real timestep
    
    return replace(traj, positions=traj.positions[idx], velocities=traj.velocities[idx], accelerations=traj.accelerations[idx], times=traj.times[idx])


def resample_events(traj, n_baseline, event_factor=3.0):
    """Cluster cadence: a course uniform baseline PLUS every snapshot where the two closest bodies come within
    event_factor * softening of each other. Close encounters are where accelerations are largest, so we sample them densely."""

    pos = traj.positions  # (T, N, 3): every body's position at every timestep    
    disp = pos[:, :, None, :] - pos[:, None, :, :]  # build seperation distances
    dist =  np.linalg.norm(disp, axis=3)  # the length of each pair's seperation

    # need to set the diagonal [t, i, i] to infinity so it is never mistaken as the closest pair, as distance would be 0
    self_idx = np.arange(pos.shape[1])  # indexes the diagonal (0,0), (1,1), ... at every timestep
    dist[:, self_idx, self_idx] = np.inf

    min_sep = dist.min(axis=(1, 2))  # (T,): distance between the 2 closest bodies, per timestep

    baseline_idx = np.linspace(0, len(min_sep) - 1, n_baseline, dtype=int)
    threshold = event_factor * traj.softening  # threshold scales with the run's epilson
    event_idx = np.nonzero(min_sep < threshold)[0]  # indices where the < condition is true

    # merge the 2 index sets while removing duplicates and sorting them in chronological order
    idx = np.union1d(baseline_idx, event_idx)

    return replace(traj, positions=traj.positions[idx], velocities=traj.velocities[idx], accelerations=traj.accelerations[idx], times=traj.times[idx])


def validate_dataset(directory, energy_tol=1e-2, cluster_energy_tol=3e-2, param_rtol=0.05):
    """Sanity-check a generated dataset. Returns a list of human_readable problem
    strings - an empty list means every file passed.

    Clusters get a looser energy bound: even with adaptive sub-stepping the extrap
    mass_ratio=15 tail drifts ~2e-2 (a resolved, physical close encounter, not corruption),
    so the tight two-body bound would false-flag it. The looser tol still catches real
    blow-ups (unresolved encounters drift ~10-250x)."""
    issues = []
    for path in sorted(glob.glob(os.path.join(directory, "*.h5"))):
        with h5py.File(path,"r") as f:
            positions = f["positions"][:]  # [:] copies the dataset into a NumPy array
            velocities = f["velocities"][:]
            masses = f["masses"][:]
            G = f.attrs["G"]
            softening = f.attrs["epsilon"]
            tags = {k:f.attrs[k] for k in f.attrs}  # all metadata into a plain Python dict
        name = os.path.basename(path)  # just the filename, for readable messages

        if not np.all(np.isfinite(positions)):
            issues.append(f"{name}: non-finite positions (trajectory blew up)")
            continue  # nothing else is meaningful once it's blown up, so skips the rest of the checks for that file

        # Rebuild the state at each saved snapshot and evaluate total energy
        E = np.array([
            total_energy(SystemState(masses, positions[t], velocities[t]),
                         G=G, softening=softening)
            for t in range(len(positions))
        ])
        drift = np.max(np.abs((E - E[0]) / E[0]))
        tol = cluster_energy_tol if tags["scenario_type"] == "cluster" else energy_tol
        if drift > tol:
            issues.append(f"{name}: energy drift {drift:.2e} exceeds {tol:.0e}")

        # check parameters landed where sweep intended
        if tags["scenario_type"] == "cluster":
            if len(masses) != tags["N"]:
                issues.append(f"{name}: N={len(masses)} but tagged {tags['N']}")
            # Re measure Q from the first snapshot (unsoftened, exactly at generation)
            v0 = velocities[0]
            T = 0.5 * np.sum(masses * np.sum(v0**2, axis=1))
            U = total_energy(SystemState(masses, positions[0], v0), G=G) - T
            Q_measured =  2 * T / abs(U)
            if not np.isclose(Q_measured, tags["Q"], rtol=param_rtol):
                issues.append(f"{name}: Q={Q_measured:.3f} but tagged {tags['Q']:.3f}")

        # mass ratio (both scenarios): heaviest / lightest should equal the tag.
        ratio_measured = masses.max() / masses.min()
        if not np.isclose(ratio_measured, tags["mass_ratio"], rtol=param_rtol):
            issues.append(f"{name}: mass_ratio={ratio_measured:.3f} but tagged {tags['mass_ratio']}")

    return issues


    


