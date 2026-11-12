"""Demo (outside the frozen G2-G4 rules): figure-eight orbit rolled out with the true force, RK4 and each GNN."""

import os

import numpy as np
import torch

from gnn.evaluate import load_model
from gnn.g2_slopes import MODELS
from gnn.rollout import gnn_force_func, rk4_substep
from gnn.run_rollouts import OUT_DIR, run_and_save
from gnn.train import OUT_DIR as MODEL_DIR
from scenarios import _zero_com, figure_eight, nondimensionalise, virial_radius

NAME = "figure_eight.npz"
PERIOD_RAW = 6.3259  # published period for G = 1, m = 1 (labbook: full-period retrace)


def period_in_dataset_units():
    """Period after nondimensionalising: time unit = sqrt(R^3 / (G M)) in the original units."""
    raw = _zero_com(figure_eight())
    return PERIOD_RAW / np.sqrt(virial_radius(raw) ** 3 / raw.masses.sum())


def main():
    torch.set_num_threads(1)
    state = nondimensionalise(_zero_com(figure_eight()))
    run = {"masses": state.masses, "vel": state.velocities[None], "dt": 0.01, "n_steps": 2000, "softening": 0.002, "G": 1.0}  # as generate_dataset
    pos0 = state.positions
    print(f"figure-eight period in dataset units: {period_in_dataset_units():.4f}; run covers t = 0-20")
    run_and_save(os.path.join(OUT_DIR, "reference", NAME), run, pos0)
    run_and_save(os.path.join(OUT_DIR, "rk4", NAME), run, pos0, substep=rk4_substep)
    for job in MODELS["test"]:
        model = load_model(os.path.join(MODEL_DIR, f"{job}.pt"), "final_state")
        run_and_save(os.path.join(OUT_DIR, job, NAME), run, pos0, gnn_force_func(model, run["masses"]))


if __name__ == "__main__":
    main()
