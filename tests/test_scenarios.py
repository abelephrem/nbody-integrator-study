import numpy as np
from scenarios import (two_body_circular, two_body_eccentric, figure_eight, chaotic_cluster, 
                       virial_radius, nondimensionalise, cluster_configs, N_TRAINED, 
                       RATIO_TRAINED, N_BASELINE, RATIO_BASELINE,)
from simulation import run_simulation
from integrators import leapfrog_step
from analysis import total_energy


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
        

