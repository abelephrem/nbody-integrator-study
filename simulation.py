"""Time-stepping loop and history logging."""

from dataclasses import dataclass, field  # for the trajectory bundle
import numpy as np  # allocating/filling the history arrays
from forces import compute_accelerations, reset_force_count, get_force_count  # to build forces_func inside run_simulation
from time import perf_counter  
import h5py


@dataclass
class Trajectory:
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray
    times: np.ndarray
    masses: np.ndarray
    G: float
    softening: float
    dt: float
    integrator: str
    scenario: str
    N_bodies: int
    N_steps: int
    metadata: dict = field(default_factory=dict)  # provenance tags for the sweep
    events: dict = field(default_factory=dict)  # close-encounter rows captured mid-substep


def run_simulation(
        initial_state,  # a SystemState
        integrator,  # the step function 
        dt,
        n_steps,  # number of steps TAKEN
        scenario_name,
        G=1.0,
        softening=0.0,
        adaptive=False,  # opt-in: off by default so softening=0 callers are untouched
        n_resolve=24,  # steps to place across each close-encounter crossing time
        record_events=False,  # opt-in: capture close-encounter rows from inside the substep loop
        event_factor=3.0,  # an episode starts when min_sep drops below event_factor*softening
        n_shells=8,  # rows per episode is capped at 2*n_shells (in and out)
        n_core=100,  # substeps per core orbit the step-size cap guarantees inside an encounter
):
    """Step the system forward n_steps and record the full trajectory.
    
    The result has n_steps + 1 rows (row 0 is the initial state)
    
    Returns a Trajectory with positions, velocities, accelerations, times, masses, and the run metadata.
    """
    N = len(initial_state.masses)
    forces_func = lambda s: compute_accelerations(s, G=G, softening=softening)  # one-arg expected so this line needed
    positions = np.zeros((n_steps + 1, N, 3))
    velocities = np.zeros((n_steps + 1, N, 3))
    accelerations = np.zeros((n_steps + 1, N, 3))
    times = np.zeros((n_steps + 1,))
    reset_force_count()
    loop_time = 0.0
    integrator_name = integrator.__name__.replace("_step", "")

    state = initial_state
    positions[0] = initial_state.positions
    velocities[0] = initial_state.velocities
    accelerations[0] = forces_func(initial_state)
    times[0] = 0.0
    max_n_sub = 1  # largest substep count any interval needed - the deepest-encounter diagnostic

    # Close-encounter recording. The pair separation is only visible INSIDE the substep
    # loop: at eps=0.002 a tight pair orbits ~7 times per outer dt, so an approach that
    # begins and ends within one interval leaves no checkpoint row at all. min_sep_run is
    # tracked unconditionally (it costs no force evaluation and fixes the long-standing
    # under-reporting of min_sep in the HDF5 attrs); the row capture is opt-in.
    min_sep_run = np.inf
    iu = np.triu_indices(N, k=1)  # upper triangle: each pair once, no self-pairs
    shells = np.geomspace(event_factor, 0.2, n_shells) * softening if softening > 0 else None

    # Step-size floor for the tightest orbit the system can form. Inside the softening
    # radius the Plummer core is harmonic (a ~ r), so the pair period saturates at t_core
    # while a -> 0 - and the adaptive rule, which tracks sqrt(eps/a_max), lets the step
    # GROW as the pair plunges. t_core uses the two heaviest masses: shortest period, so
    # the bound is conservative for every other pair. Constant, so computed once here.
    if softening > 0 and N > 1:
        t_core = 2 * np.pi * np.sqrt(softening**3 / (G * np.sort(initial_state.masses)[-2:].sum()))
        dt_core_cap = t_core / n_core
    else:
        dt_core_cap = np.inf  # no pairs, or no softening: nothing to bound
    n_capped = 0  # substeps where the cap actually bound
    min_sep_capped = np.inf  # closest separation at which it bound - where the tail is
    ev_t, ev_pos, ev_vel, ev_acc = [], [], [], []
    in_episode = False
    fired_down = np.zeros(n_shells, dtype=bool)
    fired_up = np.zeros(n_shells, dtype=bool)
    n_episodes = 0
    older = None  # (time, s_min, state, accelerations), two substeps back
    middle = None  # one substep back, what we are capturing
    n_minima = 0

    for i in range(1, n_steps + 1):
        t_elapsed = 0.0  # time covered so far within this dt_base interval
        n_sub = 0  # substeps taken this interval (for the diagnostic)

        t0 = perf_counter()
        while dt - t_elapsed > 1e-12:  # keep substepping until the full dt_base is covered
            a_sub = None
            if adaptive and softening > 0:
                # live a_max: re-sampled every substep so dt_inner tracks the plunge as it deepens
                a_sub = forces_func(state)  # reused for event rows below - no extra force eval
                a_max = np.max(np.linalg.norm(a_sub, axis=1))
                dt_inner = np.sqrt(softening / a_max) / n_resolve if a_max > 0 else dt
            else: 
                dt_inner = dt  # non-adaptive (or softening=0): one full step, substepping off

            # Separation bookkeeping, read-only: never touches state, never calls forces_func,
            # so positions, max_n_sub and force_evals are bit-for-bit what they were without it.
            sep = np.linalg.norm(state.positions[iu[0]] - state.positions[iu[1]], axis=1)
            min_sep_now = sep.min() if len(sep) else np.inf
            if min_sep_now < min_sep_run:
                min_sep_run = min_sep_now

            # Encounter-gated step-size cap. The gate is essential: ungated it binds on
            # ~100% of substeps and costs 4.3x, since t_core is the period of a config
            # (two heaviest masses, r -> 0) that almost never occurs. Measured at
            # n_resolve=96 on cluster_01078: bound 346 times, never deeper than 1.8 eps,
            # cost within chaos noise - so it fires in the wide-separation branch, not the
            # core. Kept as a BOUND on the deep tail, unmeasured across the set.
            # Condition restated because the gate needs min_sep_now, computed just above.
            if adaptive and softening > 0 and min_sep_now < event_factor * softening:
                if dt_inner > dt_core_cap:
                    dt_inner = dt_core_cap
                    n_capped += 1
                    if min_sep_now < min_sep_capped:
                        min_sep_capped = min_sep_now
            if record_events and shells is not None:
                if not in_episode and min_sep_now < shells[0]:
                    in_episode = True
                    n_episodes += 1
                    fired_down[:] = False
                    fired_up[:] = False
                if in_episode:
                    captured = False  # at most one row per substep, however many shells it crosses
                    for kk in range(n_shells):
                        if not captured and min_sep_now < shells[kk] and not fired_down[kk]:
                            fired_down[kk] = True
                            captured = True
                        elif (not captured and min_sep_now > shells[kk] and fired_down[kk]
                              and not fired_up[kk]):
                            fired_up[kk] = True
                            captured = True
                    if captured:
                        if a_sub is None:
                            a_sub = forces_func(state)
                        ev_t.append((i - 1) * dt + t_elapsed)
                        ev_pos.append(state.positions.copy())
                        ev_vel.append(state.velocities.copy())
                        ev_acc.append(a_sub.copy())
                    if min_sep_now >= shells[0]:
                        in_episode = False

            # Three samples bracket a minimum when the middle beats both neigbours.
            if record_events and older is not None and middle is not None:
                t_mid, min_sep_mid, state_mid, a_mid = middle
                if (min_sep_mid < event_factor * softening
                    and min_sep_mid < older[1] and min_sep_mid < min_sep_now):
                    if a_mid is None:  # non-adaptive path: state_mid is still alive
                        a_mid = forces_func(state_mid)
                    ev_t.append(t_mid)
                    ev_pos.append(state_mid.positions.copy())
                    ev_vel.append(state_mid.velocities.copy())
                    ev_acc.append(a_mid.copy())
                    n_minima += 1

            if record_events:
                older = middle
                middle = ((i - 1) * dt + t_elapsed, min_sep_now, state, a_sub)

            dt_inner = min(dt_inner, dt - t_elapsed)  # clamp so the last substep lands exactly on the checkpoint
            state = integrator(state, forces_func, dt_inner)  # substeps discarded, only the checkpoint is kept
            t_elapsed += dt_inner
            n_sub +=1

        max_n_sub = max(max_n_sub, n_sub)      
        loop_time += perf_counter() - t0  
        
        positions[i] = state.positions
        velocities[i] = state.velocities
        accelerations[i] = forces_func(state)
        times[i] = i * dt
    force_evals = get_force_count() - (n_steps + 1)

    return Trajectory(
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        times=times,
        masses=initial_state.masses,
        G=G,
        softening=softening,
        dt=dt,
        integrator=integrator_name,
        scenario=scenario_name,
        N_bodies=N,
        N_steps=n_steps,
        metadata={"max_n_sub": max_n_sub,
                  "force_evals": force_evals,
                  "loop_time": loop_time,
                  "min_sep_run": float(min_sep_run),  # true minimum over every substep
                  "n_episodes": n_episodes,
                  "n_minima": n_minima,
                  "n_capped": n_capped,  # substeps the cap bound; 0 = it never fired
                  "min_sep_capped": float(min_sep_capped),  # inf if it never fired
                  "t_core": float(dt_core_cap * n_core)},
        events={"times": np.array(ev_t),
                "positions": np.array(ev_pos).reshape(len(ev_t), N, 3),
                "velocities": np.array(ev_vel).reshape(len(ev_t), N, 3),
                "accelerations": np.array(ev_acc).reshape(len(ev_t), N, 3)},
    )


def save_trajectory(traj, path):
    """Write a Trajectory to a HDF5 file at `path`."""
    with h5py.File(path, "w") as f:  
        # write attrs and datasets into f here
        f.attrs["G"] = traj.G
        f.attrs["epsilon"] = traj.softening
        f.attrs["dt"] = traj.dt
        f.attrs["integrator"] = traj.integrator
        f.attrs["scenario"] = traj.scenario
        f.attrs["N_bodies"] = traj.N_bodies
        f.attrs["N_steps"] = traj.N_steps
        for key, value in traj.metadata.items():
            f.attrs[key] = value
        f.create_dataset("positions", data=traj.positions, chunks=True, compression="gzip")
        f.create_dataset("velocities", data=traj.velocities, chunks=True, compression="gzip")
        f.create_dataset("accelerations", data=traj.accelerations, chunks=True, compression="gzip")
        f.create_dataset("masses", data=traj.masses)
        f.create_dataset("times", data=traj.times)


# HDF5 attr name -> Trajectory field name. It names the fields to pull out,
# and by exclusion marks everything else as metadata.
_ATTR_TO_FIELD = {
    "G": "G",
    "epsilon": "softening",
    "dt": "dt",
    "integrator": "integrator",
    "scenario": "scenario",
    "N_bodies": "N_bodies",
    "N_steps": "N_steps",
}


def load_trajectory(path):
    """Read an HDF5 file written by 'save_trajectory' back into a Trajectory."""
    with h5py.File(path, "r") as f:
        positions = f["positions"][:]
        velocities = f["velocities"][:]
        accelerations = f["accelerations"][:]
        masses = f["masses"][:]
        times = f["times"][:]

        fields = {}
        for attr_name, field_name in _ATTR_TO_FIELD.items():
            fields[field_name] = f.attrs[attr_name]   

        metadata = {}
        for key in f.attrs:
            if key in _ATTR_TO_FIELD:
                continue
            metadata[key] = f.attrs[key]

    return Trajectory(
        positions=positions,
        velocities=velocities,
        accelerations=accelerations,
        masses=masses,
        times=times,
        metadata=metadata,
        **fields,
    )