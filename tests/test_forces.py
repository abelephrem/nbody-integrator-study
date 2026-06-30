import numpy as np
from bodies import Body, bodies_to_state
from forces import compute_accelerations


def test_two_body_acceleration():
    # arrange - 2 bodies on the x-axis, seperation r = 2, unequal masses
    b1 = Body(mass=1.0, position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0])
    b2 = Body(mass=2.0, position=[2.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0])
    state = bodies_to_state([b1, b2])
    
    # act - compute accelerations
    accelerations = compute_accelerations(state)

    # assert - shape, then each body against hand calc
    assert accelerations.shape == (2,3)
    expected = np.array([[0.5, 0.0, 0.0], [-0.25, 0.0, 0.0]])
    np.testing.assert_allclose(accelerations, expected)

def test_three_body_superposition():
    # arrange - three bodies in a line on the x-axis, equal masses, spacing 1
    b1 = Body(mass=1.0, position=[-1.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0])
    b2 = Body(mass=1.0, position=[0.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0])
    b3 = Body(mass=1.0, position=[1.0, 0.0, 0.0], velocity=[0.0, 0.0, 0.0]) 
    state = bodies_to_state([b1, b2, b3])

    # act
    accelerations = compute_accelerations(state)

    # assert
    assert accelerations.shape == (3,3)
    expected = np.array([[1.25, 0, 0], [0, 0, 0], [-1.25, 0, 0]])
    np.testing.assert_allclose(accelerations, expected)
