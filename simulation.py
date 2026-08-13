"""Time-stepping loop and history logging."""

from dataclasses import dataclass, field  # for the trajectory bundle
import numpy as np  # allocating/filling the history arrays
from forces import compute_accelerations  # to build forces_func inside run_simulation
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


def run_simulation(
        initial_state,  # a SystemState
        integrator,  # the step function 
        dt,
        n_steps,  # number of steps TAKEN
        scenario_name,
        G=1.0,
        softening=0.0,
        save_every_k=1,  # keep the knob, defaults to 1
        adaptive=False,  # opt-in: off by default so softening=0 callers are untouched
        n_resolve=24,  # steps to place across each close-encounter crossing time
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
    integrator_name = integrator.__name__.replace("_step", "")

    state = initial_state
    positions[0] = initial_state.positions
    velocities[0] = initial_state.velocities
    accelerations[0] = forces_func(initial_state)
    times[0] = 0.0
    max_n_sub = 1  # largest substep count any interval needed - the deepest-encounter diagnostic

    for i in range(1, n_steps + 1):
        t_elapsed = 0.0  # time covered so far within this dt_base interval
        n_sub = 0  # substeps taken this interval (for the diagnostic)

        while dt - t_elapsed > 1e-12:  # keep substepping until the full dt_base is covered
            if adaptive and softening > 0:
                # live a_max: re-sampled every substep so dt_inner tracks the plunge as it deepens
                a_max = np.max(np.linalg.norm(forces_func(state), axis=1))
                dt_inner = np.sqrt(softening / a_max) / n_resolve if a_max > 0 else dt
            else:
                dt_inner = dt  # non-adaptive (or softening=0): one full step, substepping off
            
            dt_inner = min(dt_inner, dt - t_elapsed)  # clamp so the last substep lands exactly on the checkpoint
            state = integrator(state, forces_func, dt_inner)  # substeps discarded, only the checkpoint is kept
            t_elapsed += dt_inner
            n_sub +=1

        max_n_sub = max(max_n_sub, n_sub)        
        
        positions[i] = state.positions
        velocities[i] = state.velocities
        accelerations[i] = forces_func(state)
        times[i] = i * dt

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
        metadata={"max_n_sub": max_n_sub}
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