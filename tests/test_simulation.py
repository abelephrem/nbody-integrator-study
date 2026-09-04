import numpy as np
from bodies import Body, bodies_to_state
from integrators import leapfrog_step
from simulation import run_simulation, save_trajectory, load_trajectory
import h5py


def make_circular_orbit():
    b1 = Body(1.0, [1.0, 0.0, 0.0], [0.0, 0.5, 0.0])
    b2 = Body(1.0, [-1.0, 0.0, 0.0], [0.0, -0.5, 0.0]) 

    return bodies_to_state([b1, b2])


def test_trajectory_shapes():
    state = make_circular_orbit()
    traj = run_simulation(state, leapfrog_step, dt=0.01, n_steps=10, scenario_name="two_body")
    assert traj.positions.shape == (11, 2, 3)
    assert traj.velocities.shape == (11, 2, 3)
    assert traj.accelerations.shape == (11, 2, 3)
    assert traj.times.shape == (11,)
    assert traj.masses.shape == (2,)


def test_row_zero_is_initial_state():
    state = make_circular_orbit()
    traj = run_simulation(state, leapfrog_step, dt=0.01, n_steps=10, scenario_name="two_body")
    assert np.array_equal(traj.positions[0], state.positions)


def test_circular_orbit_stays_bounded():
    state = make_circular_orbit()
    dt = 0.001
    n_steps = round(5 * 4 * np.pi / dt)
    traj = run_simulation(state, leapfrog_step, dt=dt, n_steps=n_steps, scenario_name="two_body")
    final_radius = np.linalg.norm(traj.positions[-1, 0])
    assert abs(final_radius -1.0) < 1e-3


def test_save_and_reload(tmp_path):
    state = make_circular_orbit()
    traj = run_simulation(state, leapfrog_step, dt=0.01, n_steps=10, scenario_name="two_body")
    
    path = tmp_path / "test.h5"  # temp file, auto_cleaned by pytest
    save_trajectory(traj, path)

    with h5py.File(path, "r") as f:  # "r" = read mode
        # check datasets came though at the right shapes 
        assert f["positions"].shape == (11, 2, 3)
        assert f["velocities"].shape == (11, 2, 3)
        assert f["accelerations"].shape == (11, 2, 3)
        assert f["times"].shape == (11,)
        assert f["masses"].shape == (2,)

        # metadata round-tripped (incl. the softening → epsilon name mapping)
        assert f.attrs["scenario"] == "two_body"
        assert f.attrs["integrator"] == "leapfrog"  # proves stripping worked
        assert f.attrs["N_steps"] == 10
        assert f.attrs["epsilon"] == 0.0  # proves softening --> epsilon name mapping worked


def test_load_trajectory_round_trip(tmp_path):
    state = make_circular_orbit()
    traj = run_simulation(state, leapfrog_step, dt=0.01, n_steps=10, 
                          scenario_name="two_body", G=1, softening=0.02)
    traj.metadata = {**traj.metadata, "split": "train", "seed": 42}

    path = tmp_path / "roundtrip.h5"
    save_trajectory(traj, path)
    loaded = load_trajectory(path)

    assert np.array_equal(loaded.positions, traj.positions)
    assert np.array_equal(loaded.velocities, traj.velocities)
    assert np.array_equal(loaded.accelerations, traj.accelerations)
    assert np.array_equal(loaded.times, traj.times)

    assert loaded.softening == 0.02
    assert set(loaded.metadata) == set(traj.metadata)


