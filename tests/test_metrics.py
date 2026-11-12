"""Tests for the Stage C metrics (gnn/metrics.py)."""

import numpy as np

from gnn.metrics import (about_com, angular_momentum_error, cluster_divergence, energies, energy_errors, fit_slope, horizon,
                         late_early_ratio, load_rollout, slope_window, two_body_divergence)
from gnn.rollout import read_run, rollout_files


def make_pair():
    masses = np.array([0.75, 0.25])
    pos = np.zeros((2, 2, 3))  # 2 checkpoints, 2 bodies, xyz
    pos[:, 0, 0] = 0.25  # body 0 on the x axis
    pos[:, 1, 0] = -0.75  # body 1 on the x axis, so the centre of mass is at the origin
    return masses, pos


def test_about_com_ignores_a_shift_of_the_whole_system():
    masses, pos = make_pair()
    assert np.allclose(about_com(masses, pos + 5.0), pos)


def test_cluster_divergence_of_a_shifted_system_is_zero():
    masses, pos = make_pair()
    assert np.allclose(cluster_divergence(masses, pos + 5.0, pos), 0.0)


def test_divergences_match_a_hand_worked_stretch():
    masses, ref = make_pair()
    pos = ref.copy()
    pos[1, 0, 0] += 0.075  # stretch the separation by 0.1 at checkpoint 1
    pos[1, 1, 0] -= 0.025  # these two shifts leave the centre of mass where it was

    delta = cluster_divergence(masses, pos, ref)
    two_body = two_body_divergence(masses, pos, ref)

    assert np.allclose(delta, [0.0, np.sqrt((0.075**2 + 0.025**2) / 2)])
    assert np.allclose(two_body, [0.0, 0.1 / (0.75 * 0.25)])


def test_horizon_is_the_first_time_at_or_above_the_threshold():
    times = np.array([0.0, 1.0, 2.0, 3.0])
    assert horizon(times, np.array([0.0, 0.1, 0.05, 0.5])) == 1.0  # exactly 0.1 counts, and a later dip doesn't matter


def test_horizon_is_infinite_if_never_reached():
    assert horizon(np.array([0.0, 1.0]), np.array([0.0, 0.05])) == np.inf


def test_saved_rollouts_start_where_they_should():
    name = next(n for n in rollout_files("test") if n.startswith("cluster"))  # first cluster in the test set
    masses = read_run(name)["masses"]
    ref, _, times = load_rollout("reference", name)
    model, _, model_times = load_rollout("design3_kick_200ep", name)
    baseline, _, _ = load_rollout("baseline", name)

    assert np.array_equal(times, model_times)  # same checkpoints, so rows can be compared directly
    assert cluster_divergence(masses, model, ref)[0] == 0  # the model starts exactly on the reference
    assert np.isclose(cluster_divergence(masses, baseline, ref)[0], 1e-6)  # the baseline starts one nudge away


def test_slope_window_and_fit_recover_an_exponential():
    times = np.linspace(0, 10, 101)
    div = 1e-6 * np.exp(2.0 * times)  # crosses 1e-4 near t = 2.3 and 0.1 near t = 5.8
    div[0] = 0.0  # rollouts start with zero divergence
    first, last = slope_window(div)
    assert div[first - 1] < 1e-4 <= div[first] and div[last - 1] < 0.1 <= div[last]
    slope, r2 = fit_slope(times, div, (first, last))
    assert np.isclose(slope, 2.0) and np.isclose(r2, 1.0)


def test_slope_window_is_none_when_never_reached():
    assert slope_window(np.full(10, 1e-3)) is None


def test_energy_of_a_circular_two_body_orbit():
    masses = np.array([1.0, 1.0])
    pos = np.array([[[0.5, 0, 0], [-0.5, 0, 0]]])  # separation 1
    vel = np.array([[[0, 1 / np.sqrt(2), 0], [0, -1 / np.sqrt(2), 0]]])  # circular: v² = G M / (4 r) per body with r = 0.5
    kinetic, potential = energies(masses, pos, vel, softening=0.0)
    assert np.isclose(kinetic[0], 0.5) and np.isclose(potential[0], -1.0)


def test_unchanged_run_has_zero_errors_and_a_velocity_boost_does_not_change_L():
    masses = np.array([1.0, 2.0, 3.0])
    rng = np.random.default_rng(0)
    pos0, vel0 = rng.normal(size=(3, 3)), rng.normal(size=(3, 3))
    pos, vel = np.stack([pos0, pos0]), np.stack([vel0, vel0])
    dE_U, dE_E = energy_errors(masses, pos, vel, softening=0.01)
    assert np.allclose(dE_U, 0) and np.allclose(dE_E, 0)
    assert np.allclose(angular_momentum_error(masses, pos, vel), 0)
    vel_shift = np.stack([vel0, vel0 + 5.0])  # a uniform velocity boost changes nothing in the centre-of-mass frame
    assert np.allclose(angular_momentum_error(masses, pos, vel_shift), 0)


def test_late_early_ratio():
    assert np.isclose(late_early_ratio(np.array([0.0, 1.0, -2.0, 4.0])), 4.0)
