import numpy as np
from scenarios import two_body_circular, two_body_eccentric
from simulation import run_simulation
from integrators import euler_step, leapfrog_step, rk4_step
from analysis import (total_energy, angular_momentum, energy_drift,
                      angular_momentum_drift, kepler_solve, two_body_reference,
                      position_error, run_convergence_sweep, convergence_order)
from bodies import SystemState


def test_energy_drift_bounded_on_circular():
    """checking relative drift is correctly computed and stays bounded."""
    state = two_body_circular(m1=1.0, m2=1.0, r=1.0)
    period = 2 * np.pi * np.sqrt(1.0**3 / 2.0)
    dt = 1e-3
    n_steps = round(2 * period / dt)  # two orbits - enough to expose drift
    traj = run_simulation(state, leapfrog_step, dt, n_steps, "circular", G=1.0)
    drift = energy_drift(traj)
    assert drift[0] == 0.0  # zero at t=0 by construction
    assert np.max(np.abs(drift)) < 1e-4  # bounded, no drift


def test_energy_drift_detects_euler_growth():
    """Guard against a metric that trivially returns ~0: Euler must show 
    clearly larger drift then Leapfrog on the same orbit."""
    state = two_body_circular(1.0, 1.0, 1.0)
    leap = run_simulation(state, leapfrog_step, dt=1e-3, n_steps=5000, scenario_name="c", G=1.0)
    eul = run_simulation(state, euler_step, dt=1e-3, n_steps=5000, scenario_name="c", G=1.0)
    assert np.max(np.abs(energy_drift(eul))) > np.max(np.abs(energy_drift(leap)))


def test_angular_momentum_conserved_on_circular():
    state = two_body_circular(1.0, 1.0, 1.0)
    traj = run_simulation(state, leapfrog_step, dt=1e-3, n_steps=5000, scenario_name="c", G=1.0)
    drift = angular_momentum_drift(traj)
    assert drift[0] == 0.0
    assert np.max(np.abs(drift)) < 1e-10   # L conserved to ~machine precision


def test_angular_momentum_boost_invariant():
    """Validates the COM subtraction: L must be unchanged by a common translation
    of all positions and a common boost of all velocities."""
    state = two_body_eccentric(2.0, 1.0, a=1.0, e=0.3)
    L = angular_momentum(state)
    boosted = SystemState(state.masses,
                          state.positions + np.array([5.0, -2.0, 1.0]),
                          state.velocities + np.array([3.0, 1.0, -4.0]))
    assert np.allclose(L, angular_momentum(boosted))


def test_kepler_solve_roundtrip():
    """BUild M from a know E, solve back, and confirm we recover E."""
    e = 0.6
    E_true = np.linspace(0, 2 * np.pi, 50)
    mean_anom = E_true - e * np.sin(E_true)  # forward direction
    assert np.allclose(kepler_solve(mean_anom, e), E_true, atol=1e-10)


def test_kepler_solve_circular():
    mean_anom = np.linspace(0, 2 * np.pi, 20)
    assert np.allclose(kepler_solve(mean_anom, 0.0), mean_anom)  # e=0 → E == M


def test_two_body_reference_peri_apo():
    """At t=0 the orbit is at periapsis on +x; half a period later, apoapsis on -x."""
    a, e, m1, m2 = 1.0, 0.5, 1.0, 1.0
    period = 2 * np.pi * np.sqrt(a**3 / (m1 + m2))
    ref = two_body_reference(np.array([0.0, period / 2]), a, e, m1, m2)
    assert np.allclose(ref[0], [a * (1 - e), 0, 0])    # periapsis
    assert np.allclose(ref[1], [-a * (1 + e), 0, 0])   # apoapsis


def test_position_error_small_for_leapfrog():
    a, e = 1.0, 0.5
    state = two_body_eccentric(1.0, 1.0, a=a, e=e)
    period = 2 * np.pi * np.sqrt(a**3 / 2.0)
    dt = 1e-4
    traj = run_simulation(state, leapfrog_step, dt=dt, n_steps=round(period / dt),
                          scenario_name="two_body", G=1.0)
    err = position_error(traj, a, e)
    assert err[0] < 1e-12          # sim and reference both start at periapsis
    assert np.max(err) < 1e-2      # a well-resolved orbit stays close to exact


def test_convergence_orders():
    """The headline Q1 check: measured order matches theeory for all three integrators."""
    state = two_body_circular(1.0, 1.0, 1.0)
    a, e = 1.0, 0.0
    step_sizes = [0.04, 0.02, 0.01, 0.005]  # all divide t_final evenly
    t_final = 4.0
    expected = {euler_step: 1.0, leapfrog_step: 2.0, rk4_step:4.0}
    for integrator, p_theory in expected.items():
        errors = run_convergence_sweep(state, integrator, step_sizes, t_final, a, e)
        p, c = convergence_order(step_sizes, errors)
        assert abs(p - p_theory) < 0.3, f"{integrator.__name__}: p={p:.2f}, expected {p_theory}"

