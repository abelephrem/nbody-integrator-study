import os
import experiments
import glob
import numpy as np
from experiments import run_path, RAW_DIR
from simulation import load_trajectory

def test_run_path_is_unique_per_run():
    """The three fields together must separate every run in the suite - in
    particular Set C's runs from the Set B run with the same scenario+integrator."""
    b = run_path("B", "circular", "leapfrog", 0.011107)   # 400 steps/orbit
    c = run_path("C", "circular", "leapfrog", 0.022214)   # 200 steps/orbit
    assert b != c

    assert run_path("B", "circular", "euler", 0.011107) != b


def test_energy_scaling_runs_respect_the_constraints(tmp_path, monkeypatch):
    monkeypatch.setattr(experiments, "RAW_DIR", str(tmp_path))
    experiments.run_energy_scaling_runs(n_orbits=1, n_h=2, h_max=1e-1, h_min=1e-2)

    paths = sorted(glob.glob(os.path.join(str(tmp_path), "D", "*.h5")))
    assert len(paths) == 2
    for path in paths:
        traj = load_trajectory(path)
        assert traj.metadata["max_n_sub"] == 1  # effective step == nominal dt
        assert traj.softening == 0.0            # Kepler reference stays valid
        assert traj.integrator == "rk4"
        assert traj.metadata["e"] == 0.5


def test_energy_scaling_reduce_wiring(tmp_path, monkeypatch):
    """Plumbing only: the reduce reads runs back, fits a positive slope, and
    writes one row per run. The real exponent comes from the full sweep."""
    monkeypatch.setattr(experiments, "RAW_DIR", str(tmp_path))
    monkeypatch.setattr(experiments, "RESULTS_DIR", str(tmp_path / "results"))
    experiments.run_energy_scaling_runs(n_orbits=1, n_h=3, h_max=1e-1, h_min=1e-2)

    s, rows = experiments.reduce_energy_scaling()
    assert len(rows) == 3
    assert s > 0                            # error must grow with step size
    assert rows[-1]["local_slope"] == ""    # last point has no forward slope
    assert os.path.exists(os.path.join(str(tmp_path / "results"), "energy_scaling.csv"))


def test_sign_structure_detects_a_crossing():
    t = np.linspace(0, 1, 5)
    s = np.array([0.0, -1.0, -2.0, 1.0, 2.0])   # flips sign between index 2 and 3
    row = experiments._sign_structure(t, s, "test")
    assert row["n_crossings"] == 1
    assert row["single_signed"] is False


def test_sign_structure_ignores_the_zero_at_t0():
    t = np.linspace(0, 1, 4)
    s = np.array([0.0, -1.0, -2.0, -3.0])       # index 0 is zero by construction
    row = experiments._sign_structure(t, s, "test")
    assert row["n_crossings"] == 0
    assert row["sign"] == "-"
    assert row["monotone_frac"] == 1.0


