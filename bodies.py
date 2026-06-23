"""State representation — mass, position, velocity arrays."""

from dataclasses import dataclass
import numpy as np

@dataclass
class Body:
    mass: float
    position: list[float]
    velocity: list[float]

@dataclass
class SystemState:
    """System state as parallel NumPy arrays.
    N = number of bodies
    masses: shape (N,) - one mass per body
    positions: shape (N, 3) - row i is body i's [x, y, z]
    velocities: shape (N, 3) - row i is body i's [vx, vy, vz]
    """
    masses: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray


def bodies_to_state(list_of_bodies):
    """Convert a list of Body objects into a System State (row i = body i)"""
    masses = np.array([body.mass for body in list_of_bodies])
    positions = np.array([body.position for body in list_of_bodies])
    velocities = np.array([body.velocity for body in list_of_bodies])
    return SystemState(masses, positions, velocities)