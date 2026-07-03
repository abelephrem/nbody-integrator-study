import numpy as np
from bodies import Body, bodies_to_state
from forces import compute_accelerations
from integrators import euler_step, leapfrog_step, rk4_step


def make_circular_orbit():
    b1 = Body(1.0, [1.0, 0.0, 0.0], [0.0, 0.5, 0.0])
    b2 = Body(1.0, [-1.0, 0.0, 0.0], [0.0, -0.5, 0.0])
    
    return bodies_to_state([b1, b2])


def run(step, dt, n_steps):
    state = make_circular_orbit()
    forces = lambda s: compute_accelerations(s, G=1.0)
    for _ in range(n_steps):
        state = step(state, forces, dt)
    
    return state


def radius(state):
    return np.linalg.norm(state.positions[0])  # distance of body 0 from origin


def test_leapfrog_conserves_orbit():
    dt = 0.001
    n_steps = round(5 * 4 * np.pi / dt)
    final = run(leapfrog_step, dt, n_steps)
    assert abs(radius(final) - 1.0) < 1e-3


def test_rk4_conserves_orbit():
    dt = 0.001
    n_steps = round(5 * 4 * np.pi / dt)
    final = run(rk4_step, dt, n_steps)
    assert abs(radius(final) - 1.0) < 1e-3


def test_euler_spirals_outward():
    dt = 0.001
    n_steps = round(5 * 4 * np.pi / dt)
    final = run(euler_step, dt, n_steps)
    assert radius(final) > 1.02