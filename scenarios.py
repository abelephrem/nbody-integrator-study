"""Initial condition scenarios — 2-body, figure-8, chaotic cluster."""
import numpy as np
from bodies import Body, SystemState, bodies_to_state
from analysis import total_energy


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
    
