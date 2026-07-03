import numpy as np
import matplotlib.pyplot as plt
from bodies import Body, bodies_to_state
from forces import compute_accelerations
from integrators import euler_step, leapfrog_step, rk4_step


def make_circular_orbit():
    b1 = Body(1.0, [1.0, 0.0, 0.0], [0.0, 0.5, 0.0])
    b2 = Body(1.0, [-1.0, 0.0, 0.0], [0.0, -0.5, 0.0])
    
    return bodies_to_state([b1, b2])

def run_with_history(step, dt, n_steps):
    state = make_circular_orbit()
    forces = lambda s: compute_accelerations(s, G=1.0) 

    history = [state.positions.copy()]  # record the starting positions
    for _ in range(n_steps):
        state = step(state, forces, dt)
        history.append(state.positions.copy())  # record after each step

    return np.array(history)  # shape (n_steps+1, N, 3)


if __name__ == "__main__":
    dt = 0.001
    n_orbits = 5
    n_steps = round(n_orbits * 4 * np.pi / dt)

    integrators = [("Euler", euler_step),
                   ("Leapfrog", leapfrog_step),
                   ("RK4", rk4_step)]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (name, step) in zip(axes, integrators):
        history = run_with_history(step, dt, n_steps)
        n_bodies = history.shape[1]
        for i in range(n_bodies):
            ax.plot(history[:, i, 0], history[:, i, 1], linewidth=0.1)       
        ax.set_title(name)
        ax.set_aspect("equal")
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()
    plt.savefig("figures/integrator_comparison.png", dpi=150)
    plt.show()