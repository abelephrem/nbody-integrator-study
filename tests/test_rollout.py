"""Tests for the Stage C rollout loop (gnn/rollout.py)."""

import numpy as np
import torch

from bodies import SystemState
from forces import compute_accelerations
from gnn.graph_data import file_to_graphs
from gnn.model import GravityGNN
from gnn.rollout import (FLAGGED, adaptive_rollout, choose_substep, core_step_cap, gnn_force_func, kdk_substep,
                         read_run, rollout_files, true_force_func, rk4_substep)
from gnn.split import DATA_DIR
from integrators import leapfrog_step, rk4_step
from simulation import run_simulation
from gnn.run_rollouts import file_number, nudged_start, run_and_save
import os



def make_system():
    rng = np.random.default_rng(0)  # same random numbers every run
    masses = np.array([0.5, 0.3, 0.2])
    return masses, rng.normal(size=(3, 3)), rng.normal(size=(3, 3))  # masses, positions, velocities


def test_true_force_func_matches_compute_accelerations():
    masses, pos, vel = make_system()
    force = true_force_func(masses, softening=0.002)
    expected = compute_accelerations(SystemState(masses, pos, vel), softening=0.002)
    assert np.array_equal(force(pos), expected)  # exactly equal, not just close


def test_kdk_substep_matches_leapfrog_step_with_one_call_per_substep():
    masses, pos, vel = make_system()
    force = true_force_func(masses, softening=0.002)
    state = SystemState(masses, pos, vel)
    steps = [0.01, 0.003, 0.02, 0.0005, 0.01]  # uneven sizes, like the adaptive rule produces

    calls = []  # one entry per force call

    def counted_force(p):
        calls.append(1)
        return force(p)

    acc = counted_force(pos)  # the one starting force call

    for h in steps:
        state = leapfrog_step(state, lambda s: force(s.positions), h)  # leapfrog_step's force takes a state
        pos, vel, acc = kdk_substep(pos, vel, acc, counted_force, h)
        assert np.array_equal(pos, state.positions)  # exact match, because reuse changes no arithmetic
        assert np.array_equal(vel, state.velocities)

    assert len(calls) == len(steps) + 1  # one per substep, plus the starting call


EPS = 0.002  # the dataset's softening length


def two_bodies(sep):
    """Two equal masses, sep apart along x: masses, positions, and the single pair of indices."""
    masses = np.array([0.5, 0.5])
    pos = np.array([[sep / 2, 0.0, 0.0], [-sep / 2, 0.0, 0.0]])
    return masses, pos, np.triu_indices(2, k=1)


def test_core_step_cap_value():
    masses, _, _ = two_bodies(1.0)  # _ = values this test doesn't need
    expected = 2 * np.pi * np.sqrt(EPS**3 / 1.0) / 100  # the two masses add up to 1
    assert np.isclose(core_step_cap(masses, EPS), expected)


def test_wide_pair_uses_base_rule_and_ignores_cap():
    masses, pos, pairs = two_bodies(1.0)  # 500 softening lengths apart: no encounter
    acc = true_force_func(masses, EPS)(pos)  # build the force function, then call it straight away
    cap = core_step_cap(masses, EPS)
    expected = np.sqrt(EPS / np.linalg.norm(acc, axis=1).max()) / 96

    h = choose_substep(pos, acc, EPS, cap, 1.0, pairs)

    assert h == expected
    assert h > cap  # the cap would have shrunk this step, so this shows the gate kept it off


def test_deep_encounter_is_capped():
    masses, pos, pairs = two_bodies(1e-5)  # far inside the softening core
    acc = true_force_func(masses, EPS)(pos)
    cap = core_step_cap(masses, EPS)
    base = np.sqrt(EPS / np.linalg.norm(acc, axis=1).max()) / 96

    assert base > cap  # the core's weak force would allow a step that is too big
    assert choose_substep(pos, acc, EPS, cap, 1.0, pairs) == cap


def test_step_stops_at_the_checkpoint():
    masses, pos, pairs = two_bodies(1.0)
    acc = true_force_func(masses, EPS)(pos)
    cap = core_step_cap(masses, EPS)
    assert choose_substep(pos, acc, EPS, cap, 1e-5, pairs) == 1e-5  # only 1e-5 left before the checkpoint


def test_zero_force_takes_all_the_time_left():
    masses, pos, pairs = two_bodies(1.0)
    cap = core_step_cap(masses, EPS)
    assert choose_substep(pos, np.zeros((2, 3)), EPS, cap, 0.25, pairs) == 0.25  # no divide-by-zero


def test_true_force_rollout_matches_run_simulation_exactly():
    masses, pos, vel = make_system()
    pos[1] = pos[0] + np.array([3e-3, 0.0, 0.0])  # put two bodies close together, so many substeps are needed
    traj = run_simulation(SystemState(masses, pos, vel), leapfrog_step, dt=0.01, n_steps=5, scenario_name="test",
                          softening=EPS, adaptive=True, n_resolve=96, n_core=100)

    all_pos, all_vel, times, n_sub = adaptive_rollout(masses, pos, vel, 0.01, 5, EPS)

    assert np.array_equal(all_pos, traj.positions)  # exactly equal, not just close
    assert np.array_equal(all_vel, traj.velocities)
    assert np.array_equal(times, traj.times)
    assert n_sub > 5  # it really did substep


def test_passing_the_true_force_as_kick_changes_nothing():
    masses, pos, vel = make_system()
    expected = adaptive_rollout(masses, pos, vel, 0.01, 5, EPS)  # kick_force = None
    result = adaptive_rollout(masses, pos, vel, 0.01, 5, EPS, kick_force=true_force_func(masses, EPS))

    assert np.array_equal(result[0], expected[0])  # [0] = positions
    assert np.array_equal(result[1], expected[1])  # [1] = velocities
    assert result[3] == expected[3]  # [3] = substep count


def test_kick_moves_bodies_but_true_force_sets_the_steps():
    masses, pos, vel = make_system()
    zero_kick = lambda p: np.zeros_like(p)  # a fake model that predicts no force at all

    all_pos, all_vel, times, n_sub = adaptive_rollout(masses, pos, vel, 0.01, 5, EPS, kick_force=zero_kick)

    assert np.array_equal(all_vel[-1], vel)  # no force, so velocities never change
    assert np.allclose(all_pos[-1], pos + vel * times[-1], rtol=0, atol=1e-12)  # bodies move in straight lines
    assert n_sub > 5  # a rule using the zero kick would take exactly one step per checkpoint


def untrained_model():
    torch.manual_seed(0)  # same starting weights every run
    return GravityGNN(3)  # the match with Stage A doesn't depend on training, so no checkpoint is needed


def test_gnn_force_matches_stage_a_graphs_exactly():
    name = rollout_files("test")[0]
    run = read_run(name)
    graphs = file_to_graphs(os.path.join(DATA_DIR, name))  # the graphs Stage A builds from this file
    model = untrained_model()
    force = gnn_force_func(model, run["masses"])

    for k, g in enumerate(graphs):  # k = row number, g = that row's graph
        with torch.no_grad():
            expected = model(g.x, g.edge_index, g.edge_attr).double().numpy()
        assert np.array_equal(force(run["pos"][k]), expected)  # same inputs must give exactly the same output


def test_gnn_force_returns_float64_in_eval_mode():
    masses, pos, _ = make_system()
    model = untrained_model()
    acc = gnn_force_func(model, masses)(pos)

    assert acc.shape == pos.shape
    assert acc.dtype == np.float64  # the integrator stays in float64
    assert not model.training  # model.eval() switched this off


def test_gnn_force_plugs_into_the_rollout():
    masses, pos, vel = make_system()
    force = gnn_force_func(untrained_model(), masses)

    all_pos, all_vel, times, n_sub = adaptive_rollout(masses, pos, vel, 0.01, 2, EPS, kick_force=force)

    assert all_pos.shape == (3, 3, 3)  # 3 checkpoints, 3 bodies, xyz
    assert np.isfinite(all_pos).all()
    assert n_sub > 2  # the true-force rule still substeps


def test_flagged_file_is_left_out_of_rollouts():
    files = rollout_files("holdout")
    assert len(files) == 79  # 80 holdout files minus cluster_01216
    assert not FLAGGED & set(files)  # & = what the two sets share; it must be empty


def test_read_run_shapes():
    run = read_run(rollout_files("test")[0])
    N = len(run["masses"])
    assert run["pos"].shape == (len(run["times"]), N, 3)
    assert run["vel"].shape == run["pos"].shape
    assert run["pos"].dtype == np.float64  # the integrator must start from full precision


def test_nudged_start_has_the_right_size_and_no_com_shift():
    masses, pos, _ = make_system()
    nudge = nudged_start(masses, pos, seed=7) - pos

    assert np.isclose(np.sqrt((nudge**2).sum(axis=1).mean()), 1e-6)  # RMS shift is exactly the nudge size
    assert np.allclose((masses[:, None] * nudge).sum(axis=0), 0.0, atol=1e-15)  # centre of mass stays put
    assert np.array_equal(nudged_start(masses, pos, seed=7) - pos, nudge)  # same seed, same nudge


def test_run_and_save_writes_a_complete_file(tmp_path):  # tmp_path = a temporary folder pytest provides
    masses, pos, vel = make_system()
    run = {"masses": masses, "vel": vel[None], "dt": 0.01, "n_steps": 2, "softening": EPS, "G": 1.0}  # vel[None] adds the row axis
    path = str(tmp_path / "run.npz")  # / joins paths on a pytest path object

    run_and_save(path, run, pos)

    with np.load(path) as saved:  # "with" closes the file afterwards, which Windows needs
        assert saved["pos"].shape == (3, 3, 3)  # 3 checkpoints, 3 bodies, xyz
        assert saved["n_sub"] > 2
    assert not os.path.exists(path + ".tmp.npz")  # the temporary file was renamed away


def test_run_and_save_skips_a_finished_file(tmp_path):
    path = tmp_path / "run.npz"
    path.write_bytes(b"done")  # a stand-in for a finished file

    run_and_save(str(path), run=None, start_pos=None)  # run=None would crash if it tried to run

    assert path.read_bytes() == b"done"  # untouched


def test_rk4_substep_matches_rk4_step_with_four_calls_per_substep():
    masses, pos, vel = make_system()
    force = true_force_func(masses, softening=EPS)
    state = SystemState(masses, pos, vel)
    steps = [0.01, 0.003, 0.02, 0.0005, 0.01]  # uneven sizes, like the adaptive rule produces

    calls = []  # one entry per force call

    def counted_force(p):
        calls.append(1)
        return force(p)

    acc = counted_force(pos)  # the one starting force call

    for h in steps:
        state = rk4_step(state, lambda s: force(s.positions), h)  # rk4_step's force takes a state
        pos, vel, acc = rk4_substep(pos, vel, acc, counted_force, h)
        assert np.array_equal(pos, state.positions)  # exact match with the existing integrator
        assert np.array_equal(vel, state.velocities)

    assert len(calls) == 4 * len(steps) + 1  # 3 stages + 1 for the next substep, plus the starting call


def test_rollout_uses_the_substep_it_is_given():
    masses, pos, vel = make_system()
    leapfrog = adaptive_rollout(masses, pos, vel, 0.01, 5, EPS)
    rk4 = adaptive_rollout(masses, pos, vel, 0.01, 5, EPS, substep=rk4_substep)

    assert not np.array_equal(rk4[0], leapfrog[0])  # [0] = positions: a different integrator gives different numbers
    assert rk4[3] > 5  # [3] = substep count: the step rule still applies


def test_run_and_save_passes_the_substep_through(tmp_path):
    masses, pos, vel = make_system()
    run = {"masses": masses, "vel": vel[None], "dt": 0.01, "n_steps": 2, "softening": EPS, "G": 1.0}
    path = str(tmp_path / "rk4.npz")

    run_and_save(path, run, pos, substep=rk4_substep)
    expected = adaptive_rollout(masses, pos, vel, 0.01, 2, EPS, substep=rk4_substep)

    with np.load(path) as saved:
        assert np.array_equal(saved["pos"], expected[0])


def test_file_number_reads_the_number_from_a_filename():
    assert file_number("cluster_01001.h5") == 1001


def test_extra_baseline_seeds_give_different_nudges_of_the_right_size():
    masses, pos, _ = make_system()
    starts = [nudged_start(masses, pos, seed) for seed in (1001, [1001, 2], [1001, 3])]  # the three baseline seed forms

    assert not np.array_equal(starts[0], starts[1])
    assert not np.array_equal(starts[0], starts[2])
    assert not np.array_equal(starts[1], starts[2])

    for start in starts:
        nudge = start - pos
        assert np.isclose(np.sqrt((nudge**2).sum(axis=1).mean()), 1e-6)  # still exactly the nudge size



