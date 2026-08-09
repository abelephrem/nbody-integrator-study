import numpy as np
from scenarios import (two_body_circular, two_body_eccentric, figure_eight, chaotic_cluster, 
                       virial_radius, nondimensionalise, cluster_configs, random_orientation, build_initial_state, N_TRAINED, 
                       RATIO_TRAINED, N_BASELINE, RATIO_BASELINE, TrajectoryConfig, resample_uniform, two_body_configs, generate_dataset, resample_true_anomaly,
                       resample_events, validate_dataset)
from simulation import run_simulation
from integrators import leapfrog_step
from analysis import total_energy
import h5py


def test_circular_returns_after_one_period():
    G, m1, m2, r = 1.0, 1.0, 1.0, 1.0
    state = two_body_circular(m1,m2, r, G)
    a = r
    T = 2*np.pi*np.sqrt(a**3 / (G*(m1+m2)))
    dt = 1e-3
    n_steps = round(T / dt)  # rounds to nearest whole step
    traj = run_simulation(state, leapfrog_step, dt=dt, n_steps=n_steps, scenario_name="two_body_circular", G=G)
    start = traj.positions[0]
    end = traj.positions[-1]
    assert np.allclose(start, end, atol=1e-2)  # must match with a  1e-2 max discrepancy


def test_eccentric_returns_after_one_period():
    G, m1, m2, a, e = 1.0, 1.0, 1.0, 1.0, 0.5
    state = two_body_eccentric(m1, m2, a, e)
    T = 2*np.pi*np.sqrt(a**3 / (G*(m1+m2)))  # uses a, NOT r_peri
    dt = 1e-3
    n_steps = round(T / dt)
    traj = run_simulation(state, leapfrog_step, dt=dt, n_steps=n_steps, scenario_name="two_body_eccentric", G=G)
    start = traj.positions[0]
    end = traj.positions[-1]
    assert np.allclose(start, end, atol=1e-2)


def test_figure_eight_retraces():
    T = 6.3259  # published period (Moore 1993)
    dt = 1e-4
    n_steps = round(T / dt)
    state = figure_eight()
    traj = run_simulation(state,leapfrog_step, dt=dt, n_steps=n_steps, scenario_name="figure_eight", G=1.0)
    assert np.allclose(traj.positions[0], traj.positions[-1], atol=1e-2)


def test_cluster_is_bound_and_stays_finite():
    state = chaotic_cluster(N=5, seed=42)  # fixed seed --> deterministic
    assert total_energy(state) < 0  # built bound

    traj = run_simulation(state, leapfrog_step, dt=1e-3,n_steps=5000, scenario_name="chaotic_cluster", G=1.0)
    assert np.all(np.isfinite(traj.positions))  # no body escaped to inf/nan


def test_virial_radius_two_body():
    m1, m2, a = 2.0, 1.0, 1.5
    state = two_body_eccentric(m1, m2, a=a, e=0.3)
    R = virial_radius(state)
    assert np.isclose(R, (m1+m2)**2 * a / (m1*m2))


def test_non_dimensionalise_sets_unit_scales():
    state = chaotic_cluster(N=5, seed=1)
    nd = nondimensionalise(state)
    assert np.isclose(np.sum(nd.masses), 1.0)  # total mass --> 1
    assert np.isclose(virial_radius(nd), 1.0)  # length unit --> R=1


def test_cluster_configs_one_axis_holdout():
    cfgs = cluster_configs(n_draws=5)  # generate full list of configs
    for c in cfgs:
        p = c.params
        if c.split == "train":
            assert p["N"] in N_TRAINED and p["mass_ratio"] in RATIO_TRAINED  # makes sure axes are drawn from trained sets
        elif c.split.startswith("Q"):
            assert p["N"] == N_BASELINE and p["mass_ratio"] == RATIO_BASELINE
    # no training Q ever lands in the interpolation gap (0.7, 1.1)
    train_Q = [c.params["Q"] for c in cfgs if c.split == "train"]  
    assert all(not (0.7 < q < 1.1) for q in train_Q)  # asserts no training Q fell in the interpolation gap 
        

def test_cluster_hits_target_Q():
    Q_target = 1.3
    state = chaotic_cluster(N=5, Q=Q_target, seed=7)
    T = 0.5 * np.sum(state.masses * np.sum(state.velocities**2, axis=1))
    U = total_energy(state) - T
    assert np.isclose(2 * T / abs(U), Q_target)
    assert total_energy(state) < 0  # Q < 2 --> bound


def test_cluster_mass_ratio():
    state = chaotic_cluster(N=5, mass_ratio=3.0, seed=2)
    assert np.isclose(state.masses[0], 3.0)
    assert np.allclose(state.masses[1:], 1.0)   # rest are light
    assert total_energy(state) < 0              # still bound


def test_random_orientation_preserves_physics():
    state = two_body_eccentric(1.0, 1.0, a=1.0, e=0.3)
    rotated = random_orientation(state, seed=5)
    assert np.isclose(total_energy(state), total_energy(rotated))  # physics should be unchanged
    d0 = np.linalg.norm(state.positions[0] - state.positions[1])
    d1 = np.linalg.norm(rotated.positions[0] - rotated.positions[1])
    assert np.isclose(d0, d1)  # rigid - seperation preserved
    assert not np.allclose(state.positions, rotated.positions)  # check actually rotated


def test_build_initial_state():
    c =  TrajectoryConfig("two_body", {"e":0.4, "mass_ratio": 3}, "train", 0)
    s = build_initial_state(c)
    assert len(s.masses) == 2
    assert np.isclose(np.sum(s.masses), 1.0)  # nondimensionalised

    c2 = TrajectoryConfig("cluster", {"N": 5, "Q": 1.2, "mass_ratio":1}, "train", 1000)
    s2 = build_initial_state(c2)
    assert len(s2.masses) == 5
    assert np.isclose(virial_radius(s2), 1.0)  # length unit landed at R=1


def test_resample_uniform_keeps_n():
    state = chaotic_cluster(N=4, seed=3)
    traj = run_simulation(state, leapfrog_step, dt=0.01, n_steps=500, scenario_name="cluster", G=1.0)
    r = resample_uniform(traj, 50)
    assert r.positions.shape == (50, 4, 3)
    assert r.times.shape == (50,)
    assert r.times[0] == traj.times[0] and r.times[-1] == traj.times[-1]  # endpoints kept


def test_generate_dataset(tmp_path):
    configs = two_body_configs(n_orient=1)[:2]  # slice the list, only incude elements 0 and 1
    generate_dataset(configs, tmp_path, duration=2.0, dt=0.05, n_keep=20)
    files = sorted(tmp_path.glob("*.h5")) # list of HDF5 files the driver wrote
    assert len(files) == 2  # 2 configs in should produce 2 files out
    with h5py.File(files[0], "r") as f:
        assert f["positions"].shape[0] == 20  # resampled to n_keep
        assert f.attrs["split"] == "train"
        assert "scenario_type" in f.attrs and "e" in f.attrs  # tags written


def test_true_anomaly_uniform_in_angle():
    state = two_body_eccentric(1.0, 1.0, a=1.0, e=0.6)
    T = 2*np.pi*np.sqrt(1.0**3 / 2.0)
    traj = run_simulation(state, leapfrog_step, dt=1e-3, n_steps=round(T/1e-3), scenario_name="two_body", G=1.0)
    r = resample_true_anomaly(traj, 20)
    rel = r.positions[:, 0, :] - r.positions[:, 1, :]
    theta = np.unwrap(np.arctan2(rel[:, 1], rel[:, 0]))  # planar: xy is the orbital plane, no 3D tilt
    d = np.diff(theta)  # angular gaps between consecutive kept snapshots
    assert np.std(d) < 0.3 * np.mean(d)  # angular spacing ~ uniform


def test_events_capture_close_encounters():
    state = chaotic_cluster(N=4, Q=0.3, seed=11)  # cold -> collapses -> close encounters
    traj = run_simulation(state, leapfrog_step, dt=1e-3, n_steps=3000, scenario_name="cluster", G=1.0, softening=0.05)
    r = resample_events(traj, n_baseline=20)

    # Independently recompute the closest-pair distance on the full trajectory
    pos = traj.positions
    dist = np.linalg.norm(pos[:, :, None,:] - pos[:, None, :, :], axis=3)
    ii = np.arange(4)
    dist[:, ii, ii] = np.inf
    min_sep = dist.min(axis=(1, 2))
    close_times = traj.times[min_sep < 3.0 * traj.softening]  # times of close encounters

    assert len(close_times) > 0  # confirm the cold cluster did collapse
    assert np.all(np.isin(close_times, r.times))  # every close encounter survived resampling


def test_validate_dataset_clean(tmp_path):
    configs = cluster_configs(n_draws=1)[:3] + two_body_configs(n_orient=1)[:3]
    generate_dataset(configs, tmp_path, duration=5.0, dt=1e-3, softening=0.05, n_keep=50, n_baseline=30)
    issues = validate_dataset(tmp_path)
    assert issues == [], f"dataset issues: {issues}"