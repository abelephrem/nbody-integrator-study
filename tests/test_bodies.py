import numpy as np
from bodies import Body, bodies_to_state


def test_two_body_state():
    # arrange - build 2 bodies
    b1 = Body(mass=1.0, position=[1.0, 0.0, 0.0], velocity=[0.0, 0.5, 0.0])
    b2 = Body(mass=1.0, position=[-1.0, 0.0, 0.0], velocity=[0.0, -0.5, 0.0])

    # act - convert to a SystemState
    state = bodies_to_state([b1, b2])

    #assert - shape, then every value of both bodies
    assert state.positions.shape == (2, 3)
    np.testing.assert_array_equal(state.masses, [1.0, 1.0])
    np.testing.assert_array_equal(state.positions, [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(state.velocities, [[0.0, 0.5, 0.0], [-0.0, -0.5, 0.0]])
