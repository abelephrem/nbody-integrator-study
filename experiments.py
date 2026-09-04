"""Stage 8 - the experiment suite.

Two halves, deliberately seperate: generate(run sims, write HDF5 to disk) and
reduce (read those files back, produce the tables and plots)."""

import os
import numpy as np
import glob
import csv
import matplotlib.pyplot as plt

from scenarios import two_body_circular, two_body_eccentric, circular_period, resample_uniform
from simulation import run_simulation, save_trajectory, load_trajectory
from integrators import euler_step, leapfrog_step, rk4_step
from analysis import position_error, fit_region, convergence_order, energy_drift, angular_momentum_drift, local_slopes, angular_momentum_signed_drift, drift_growth_exponent

INTEGRATORS = {"euler": euler_step, "leapfrog": leapfrog_step, "rk4": rk4_step}
SCENARIOS = {
    "circular": (two_body_circular, 0.0),
    "eccentric": (two_body_eccentric, 0.5),
}

RAW_DIR = "C:/nbody_raw"
RESULTS_DIR = "results"


def run_path(set_name, scenario, integrator_name, dt):
    """Where the HDF5 file for one run lives."""
    filename = f"{scenario}_{integrator_name}_dt{dt:.6e}.h5"

    return os.path.join(RAW_DIR, set_name, filename)


def run_convergence_runs(n_orbits=5, n_h=15, n_keep=200, skip_existing=True):
    """Set A - the Q1 convergence sweep. 3 integrators x n_h step sizes = 45 runs.
    Every run is the same circular orbit integrated to the same physical end time;
    only the step size changes. 
    
    The measured order is then the slope of log(final position error) against log(h),
    fitted in the reduce half.
    """
    state = two_body_circular()
    period = circular_period()
    t_final = n_orbits * period

    step_sizes = np.geomspace(1e-1, 1e-4, n_h)

    for name, integrator in INTEGRATORS.items():
        for dt in step_sizes:
            path = run_path("A", "circular", name, dt)
            if skip_existing and os.path.exists(path):
                print(f"A {name:9s} dt={dt:.3e} exists - skipped")
                continue

            n_steps = round(t_final / dt)
            traj = run_simulation(state, integrator, dt, n_steps=n_steps, scenario_name="circular", G=1, softening=0.0)
            assert traj.metadata["max_n_sub"] == 1, "substepping ran - effective step != dt"

            traj.metadata = {**traj.metadata, "set": "A", "n_orbits": n_orbits, "a":1.0, "e":0.0}

            traj = resample_uniform(traj, n_keep)

            
            os.makedirs(os.path.dirname(path), exist_ok=True)  # side effect kept out of run_path
            save_trajectory(traj, path)
            print(f"A {name:9s} dt={dt:.3e} n_steps={n_steps:7d} -> {path}")


def long_run_specs():
    """Every Set B and Set C run, as (set, scenario, integrator, steps_per_orbit)."""
    specs = []

    # Set B = Q2/3/4. Both scenarios x all 3 integrators at one shared step size
    for scenario in SCENARIOS:
        for name in INTEGRATORS:
            specs.append(("B", scenario, name, 400))

    # Set C - the Q4 scaling test (circular and eccentric))
    for scenario in ("circular", "eccentric"):
        for name in ("leapfrog", "rk4"):
            for spo in (200, 800):
                specs.append(("C", scenario, name, spo))

    return specs


def run_long_runs(n_orbits=300, skip_existing=True):
    """Sets B and C - the long-run comparison. 14 runs, all to the same 300 orbits."""
    period = circular_period()  # same for both scenarios

    for set_name, scenario, name, spo in long_run_specs():
        builder, e = SCENARIOS[scenario]
        integrator = INTEGRATORS[name]

        dt = period / spo
        n_steps = n_orbits * spo
        path = run_path(set_name, scenario, name, dt)

        # skips already existing runs if new runs are added, instead of recomputing
        if skip_existing and os.path.exists(path):
            print(f"{set_name} {scenario:9s} {name:9s} spo={spo:4d} exists - skipped")
            continue

        state = builder()
        traj = run_simulation(state, integrator, dt, n_steps=n_steps, scenario_name=scenario, G=1, softening=0.0)
        assert traj.metadata["max_n_sub"] == 1, "substepping ran - effective step is != dt"
        traj.metadata = {**traj.metadata, "set": set_name, "n_orbits": n_orbits,
                         "steps_per_orbit": spo, "a": 1.0, "e": e}

        os.makedirs(os.path.dirname(path), exist_ok=True)
        save_trajectory(traj, path)
        print(f"{set_name} {scenario:9s} {name:9s} spo={spo:4d} n_steps={n_steps:7d} -> {path}")


def reduce_convergence(set_name="A"):
    """Set A --> measured convergence order per integrator (Q1).
    
    Reads the saved runs back off disk, takes each one's final position error, and fits log(error) against log(dt)."""
    runs = {}  # integrator name -> list of (dt, final error)
    for path in sorted(glob.glob(os.path.join(RAW_DIR, set_name, "*.h5"))):
        traj = load_trajectory(path)
        err = position_error(traj, traj.metadata["a"], traj.metadata["e"])
        runs.setdefault(traj.integrator, []).append((traj.dt, err[-1]))

    orders = {}
    rows = []
    for name, pairs in runs.items():
        pairs.sort()
        h = np.array([dt for dt, _ in pairs])
        errors = np.array([e for _, e in pairs])
        
        region = fit_region(h, errors) 
        p, c = convergence_order(h, errors, fit_slice=region)
        orders[name] = p

        for i in range(len(h)):
            rows.append({"integrator": name, "dt": h[i], "error": errors[i],
                         "in_fit": i in region})  # so a later plot can mark the fitted points

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "convergence.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["integrator", "dt", "error", "in_fit"])
        writer.writeheader()
        writer.writerows(rows)

    for name, p in sorted(orders.items()):
        print(f"{name:9s} measured p = {p:.2f}")
    return orders, rows


def reduce_long_runs():
    """Sets B and C -> one summary row per run (Q2, Q3).
    """
    rows = []
    for set_name in ("B", "C"):
        for path in sorted(glob.glob(os.path.join(RAW_DIR, set_name, "*.h5"))):
            traj = load_trajectory(path)

            e_drift = np.abs(energy_drift(traj))  # at every saved step
            l_drift = np.abs(angular_momentum_drift(traj))

            half = len(e_drift) // 2  # split the run into 2 to compare earluy vs late
            first, second = e_drift[:half].max(), e_drift[half:].max()
            growth = second / first if first > 0 else np.inf 
            q, q_n, saturated = drift_growth_exponent(traj.times, e_drift)

            rows.append({
                "set": set_name,
                "scenario": traj.scenario,
                "integrator": traj.integrator,
                "steps_per_orbit": int(traj.metadata["steps_per_orbit"]),
                "dt": traj.dt,
                "max_E_drift": e_drift.max(),
                "final_E_drift": e_drift[-1],
                "growth_ratio": growth,
                "q": q,
                "q_n_points": q_n,
                "saturated": saturated,
                "max_L_drift": l_drift.max(),        # Q3
                "force_evals": int(traj.metadata["force_evals"]),  # cost
                "loop_time": float(traj.metadata["loop_time"]),
            })

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "long_runs.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))  # column order = insertion order above
        writer.writeheader()
        writer.writerows(rows)

    for r in rows:
        print(f"{r['set']} {r['scenario']:9s} {r['integrator']:9s} spo={r['steps_per_orbit']:4d} "
            f"maxE={r['max_E_drift']:.2e} growth={r['growth_ratio']:6.2f} q={r['q']:.3f}"
            f"{' SAT' if r['saturated'] else '   '} maxL={r['max_L_drift']:.2e}")

    return rows


def run_energy_scaling_runs(n_orbits=5, n_h=10, h_max=1e-1, h_min=1e-4, skip_existing=True):
    """Set D - Run 1: eccentric, RK4, no resampling."""
    state = two_body_eccentric()
    period = circular_period()
    t_final = n_orbits * period

    step_sizes = np.geomspace(h_max, h_min, n_h)

    for dt in step_sizes:
        path = run_path("D", "eccentric", "rk4", dt)
        if skip_existing and os.path.exists(path):
            print(f"D rk4 dt={dt:.3e} exists - skipped")
            continue

        n_steps = round(t_final / dt)
        traj = run_simulation(state, rk4_step, dt, n_steps=n_steps, scenario_name="eccentric", G=1, softening=0.0)
        assert traj.metadata["max_n_sub"] == 1, "substepping ran - effective step != dt"

        traj.metadata = {**traj.metadata, "set": "D", "n_orbits": n_orbits, "a": 1.0, "e": 0.5}

        
        os.makedirs(os.path.dirname(path), exist_ok=True)  # side effect kept out of run_path
        save_trajectory(traj, path)
        print(f"D rk4 dt={dt:.3e} n_steps={n_steps:7d} -> {path}")


def reduce_energy_scaling(set_name="D"):
    """Set D -> the measured energy-error scaling exponent s (Run 1).

    Takes each run's max|dE/E0| over the whole run and fits log(max|dE/E0|)
    against log(h). The slope is s.
    """
    pairs = []          # (dt, max|dE/E0|), one per run
    end_times = set()   # every run must share an end time or the fit mixes regimes

    for path in sorted(glob.glob(os.path.join(RAW_DIR, set_name, "*.h5"))):
        traj = load_trajectory(path)
        drift = np.abs(energy_drift(traj))  # signed series -> magnitude of each excursion
        pairs.append((traj.dt, drift.max()))  # the max over the run, not the final value
        end_times.add(int(traj.metadata["n_orbits"]))
        print(f"read dt={traj.dt:.3e}  max|dE/E0|={drift.max():.3e}")

    assert len(end_times) == 1, f"runs have different end times: {end_times}"

    pairs.sort()  # ascending in dt, so index 0 is the smallest step size
    h = np.array([dt for dt, _ in pairs])
    errors = np.array([e for _, e in pairs])

    region = fit_region(h, errors)
    s, _ = convergence_order(h, errors, fit_slice=region)
    local = local_slopes(h, errors)

    rows = []
    for i in range(len(h)):
        rows.append({"dt": h[i], "max_E_drift": errors[i], "in_fit": i in region,
                     "local_slope": local[i] if i < len(local) else ""})  # last point has no forward slope

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "energy_scaling.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dt", "max_E_drift", "in_fit", "local_slope"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nmeasured s = {s:.2f}  (fitted on {len(region)}/{len(h)} points)")
    print("local slopes (small h -> large h): " + "  ".join(f"{v:.2f}" for v in local))
    return s, rows#


def plot_energy_scaling(rows, s, out_path="figures/energy_scaling.png"):
    """Run 1 figure. Left: max|dE/E0| against h, log-log, with the fitted slope
    s and slope-4/slope-5 guides. Right: the local slope at each adjacent pair,
    which is where the curve's bend actually shows.

    Takes the reduce half's output rather than re-fitting, so this figure and
    results/energy_scaling.csv can never disagree with each other.
    """
    h = np.array([r["dt"] for r in rows])
    errors = np.array([r["max_E_drift"] for r in rows])
    in_fit = np.array([r["in_fit"] for r in rows])

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    # --- left panel: the error curve itself ---
    ax.loglog(h[in_fit], errors[in_fit], "o", color="C0",
              label=f"RK4, e = 0.5:  s = {s:.2f}")
    # points fit_region excluded, drawn hollow so the trimming is visible, not silent
    ax.loglog(h[~in_fit], errors[~in_fit], "o", mfc="none", color="C0")

    # Rebuild the fitted line in real space. A least-squares line passes exactly
    # through the centroid of its points, so given the slope s the intercept is
    # fixed - no need for the reduce half to hand it over as well.
    h_fit = h[in_fit]
    c = np.mean(np.log(errors[in_fit])) - s * np.mean(np.log(h_fit))
    ax.loglog(h_fit, np.exp(c) * h_fit**s, "--", color="C0")

    # Guides for the two candidate exponents, both anchored at the largest fitted
    # point so they emanate from real data rather than floating at some offset.
    h_ref, e_ref = h_fit[-1], errors[in_fit][-1]
    for order, style in ((4, ":"), (5, "-.")):
        ax.loglog(h_fit, e_ref * (h_fit / h_ref)**order, style, color="0.5",
                  lw=1, label=f"slope {order}")

    ax.set_xlabel("step size  h")
    ax.set_ylabel(r"$\max|\Delta E / E_0|$ over the run")
    ax.set_title("Energy-error scaling")
    ax.legend()

    # --- right panel: how the slope varies along that curve ---
    local = np.array([r["local_slope"] for r in rows[:-1]])  # last row has no forward slope
    # each local slope lives BETWEEN two points, so place it at their geometric
    # midpoint - the natural centre of a pair on a log axis
    h_mid = np.sqrt(h[:-1] * h[1:])
    # a slope is part of the fit only if BOTH of its endpoints survived trimming
    slope_in_fit = in_fit[:-1] & in_fit[1:]

    ax2.semilogx(h_mid[slope_in_fit], local[slope_in_fit], "o-", color="C0")
    ax2.semilogx(h_mid[~slope_in_fit], local[~slope_in_fit], "o", mfc="none", color="C0")
    for order, style in ((4, ":"), (5, "-.")):
        ax2.axhline(order, ls=style, color="0.5", lw=1)

    # the two round-off-floor slopes sit near 0 and would squash the band that
    # matters into a fifth of the panel; the left panel already shows the trimming
    ax2.set_ylim(3.5, 5.5)

    ax2.set_xlabel("step size  h  (geometric midpoint of each pair)")
    ax2.set_ylabel("local slope")
    ax2.set_title("Local slope along the curve")

    fig.suptitle("RK4 energy-error scaling — two-body, e = 0.5, 5 orbits")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def _sign_structure(times, series, name):
    """Describe one signed drift series: which sign, how monotone, any crossings.

    Index 0 is skipped throughout: the drift is identically zero there by
    construction, and a naive sign test would report that as a zero-crossing
    in every single series.
    """
    x = series[1:]
    t = times[1:]

    # A crossing is where two consecutive NON-ZERO values have opposite signs.
    # Filtering the zeros out first matters: sign() returns 0 for them, so a
    # single exact zero sitting in a one-signed series would read as two flips.
    nonzero = x[x != 0]
    crossings = np.where(np.diff(np.sign(nonzero)) != 0)[0]

    direction = np.sign(x[-1])  # the overall direction of travel
    steps = np.diff(series)
    # fraction of steps moving the same way as the overall change:
    # 1.0 = strictly monotone, ~0.5 = the ripple is as large as the trend
    monotone_frac = np.mean(np.sign(steps) == direction)

    if len(crossings) == 0:
        first_crossing_t = ""
    else:
        # crossings indexes the FILTERED array, so map it back: idx[k] is where
        # the k-th non-zero value sits in x, and +1 takes the first value on
        # the far side of the flip rather than the last one before it
        idx = np.where(x != 0)[0]
        first_crossing_t = t[idx[crossings[0] + 1]]

    print(f"  {name}: sign={'+' if direction > 0 else '-'}  "
          f"crossings={len(crossings)}  monotone_frac={monotone_frac:.4f}")
    print(f"    min={series.min():.4e}  max={series.max():.4e}  final={series[-1]:.4e}")

    return {"quantity": name, "sign": "+" if direction > 0 else "-",
            "single_signed": len(crossings) == 0, "n_crossings": len(crossings),
            "monotone_frac": monotone_frac, "min": series.min(),
            "max": series.max(), "final": series[-1],
            "first_crossing_t": first_crossing_t}


def reduce_signed_drift(set_name="B", integrator="rk4", scenarios=("circular", "eccentric")):
    """Set B -> the SIGN structure of RK4's energy and angular momentum error.

    reduce_long_runs reports max|dE/E0|: a running max of an absolute value,
    which destroys the sign. This reads the same files back and keeps it.
    No new simulation - Set B stores every step, unresampled.
    """
    period = circular_period()
    rows, series = [], {}

    for scenario in scenarios:
        matches = glob.glob(os.path.join(RAW_DIR, set_name, f"{scenario}_{integrator}_*.h5"))
        assert len(matches) == 1, f"expected one {scenario} {integrator} run, got {matches}"
        traj = load_trajectory(matches[0])

        dE = energy_drift(traj)                   # already signed
        dL = angular_momentum_signed_drift(traj)  # signed, projected onto L0

        print(f"{scenario} {integrator} (dt={traj.dt:.3e}, {len(traj.times)} snapshots):")
        for name, s in (("dE/E0", dE), ("dL/L0", dL)):
            row = _sign_structure(traj.times, s, name)
            row["scenario"] = scenario
            rows.append(row)
        # do E and L shrink together? the inward-spiral picture requires it
        print(f"    sign(dE) == sign(dL): {np.sign(dE[-1]) == np.sign(dL[-1])}")

        series[scenario] = (traj.times / period, dE, dL)  # x-axis in orbits

    os.makedirs(RESULTS_DIR, exist_ok=True)
    fields = ["scenario", "quantity", "sign", "single_signed", "n_crossings",
              "monotone_frac", "min", "max", "final", "first_crossing_t"]
    with open(os.path.join(RESULTS_DIR, "signed_drift.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    return rows, series


def plot_signed_drift(series, out_path="figures/signed_drift.png", zoom_orbits=5):
    """Run 2 figure: signed dE/E0 and dL/L0 against time, on a LINEAR y-axis.

    Left column is the whole 300 orbits (the trend); right column zooms the
    first few orbits at full time resolution, which is the only place the
    per-orbit ripple is resolvable at this figure size.

    Linear y is not a style preference: signed data cannot go on a log axis -
    the negative half would silently vanish, taking the answer with it.
    """
    scenarios = list(series)
    fig, axes = plt.subplots(2 * len(scenarios), 2, figsize=(11, 9), sharex="col")

    for i, scenario in enumerate(scenarios):
        orbits, dE, dL = series[scenario]
        zoom = orbits <= zoom_orbits  # boolean mask picking out the first few orbits

        for j, (label, data) in enumerate(((r"$\Delta E/E_0$", dE),
                                           (r"$\Delta L/L_0$", dL))):
            row = 2 * i + j
            for col, mask in enumerate((slice(None), zoom)):  # full run, then the zoom
                ax = axes[row, col]
                ax.plot(orbits[mask], data[mask], lw=0.6, color="C0")
                ax.axhline(0.0, color="0.5", lw=0.8, ls="--")  # the line a sign change must cross
                ax.set_ylabel(f"{scenario}\n{label}", fontsize=8)
                ax.grid(alpha=0.3)
                ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[0, 0].set_title("full run", fontsize=9)
    axes[0, 1].set_title(f"first {zoom_orbits} orbits", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel("orbits")

    fig.suptitle("Signed drift — RK4, 400 steps/orbit, 300 orbits (linear y-axis)",
                 fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def plot_convergence_set(rows, orders, out_path="figures/convergence_set_a.png"):
    """Set A figure: final position error against h per integrator, log-log,
    with the fitted order p and the trimmed points drawn hollow.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for i, (name, p) in enumerate(sorted(orders.items())):
        sel = sorted([r for r in rows if r["integrator"] == name], key=lambda r: r["dt"])
        h = np.array([r["dt"] for r in sel])
        err = np.array([r["error"] for r in sel])
        in_fit = np.array([r["in_fit"] for r in sel])

        colour = f"C{i}"  # one colour per integrator, shared by its points and its fit line
        ax.loglog(h[in_fit], err[in_fit], "o", color=colour, label=f"{name}:  p = {p:.2f}")
        # excluded points hollow: same series, not in the fit
        ax.loglog(h[~in_fit], err[~in_fit], "o", mfc="none", color=colour)

        # fitted line rebuilt through the centroid of its own points, as in Run 1
        h_fit = h[in_fit]
        c = np.mean(np.log(err[in_fit])) - p * np.mean(np.log(h_fit))
        ax.loglog(h_fit, np.exp(c) * h_fit**p, "--", color=colour)

    ax.set_xlabel("step size  h")
    ax.set_ylabel("position error at $t_{final}$")
    ax.set_title("Convergence order — Set A, two-body circular, 5 orbits")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def plot_long_runs_scaling(rows, out_path="figures/long_runs_scaling.png"):
    """Sets B+C figure: max|dE/E0| against step size, log-log, per scenario.

    Plotted against h rather than steps-per-orbit so the slope comes out as +s
    instead of -s. Euler is left out: it only ran at one step size, so it has
    no slope, and its drift of ~1 would stretch the axis by six decades.
    """
    scenarios = sorted({r["scenario"] for r in rows})
    fig, axes = plt.subplots(1, len(scenarios), figsize=(12, 5.5), sharey=True)

    for ax, scenario in zip(axes, scenarios):
        for i, integrator in enumerate(("leapfrog", "rk4")):
            sel = sorted([r for r in rows
                          if r["scenario"] == scenario and r["integrator"] == integrator],
                         key=lambda r: r["dt"])
            h = np.array([r["dt"] for r in sel])
            err = np.array([r["max_E_drift"] for r in sel])

            # only three points, and all three are wanted - no floor to trim here,
            # so this fits the lot rather than going through fit_region
            s, c = convergence_order(h, err)

            colour = f"C{i}"
            ax.loglog(h, err, "o", color=colour, label=f"{integrator}:  s = {s:.2f}")
            ax.loglog(h, np.exp(c) * h**s, "--", color=colour)

        ax.set_xlabel("step size  h")
        ax.set_title(scenario)
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)

    axes[0].set_ylabel(r"$\max|\Delta E / E_0|$ over 300 orbits")
    fig.suptitle("Energy-error scaling with step size — Sets B + C, 300 orbits")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def main(skip_existing=True):
    """Regenerate every Stage 8 table and figure in one run.
    """
    print("=== Set A: convergence sweep (Q1) ===")
    run_convergence_runs(skip_existing=skip_existing)
    orders, rows_a = reduce_convergence()
    plot_convergence_set(rows_a, orders)

    print("\n=== Sets B and C: long runs (Q2, Q3, Q4) ===")
    run_long_runs(skip_existing=skip_existing)
    rows_bc = reduce_long_runs()
    plot_long_runs_scaling(rows_bc)

    print("\n=== Set D: energy-error scaling at 5 orbits (addendum Run 1) ===")
    run_energy_scaling_runs(skip_existing=skip_existing)
    s, rows_d = reduce_energy_scaling()
    plot_energy_scaling(rows_d, s)

    print("\n=== Set B revisited: signed drift (addendum Run 2) ===")
    rows_signed, series = reduce_signed_drift()
    plot_signed_drift(series)

    print("\nall Stage 8 tables and figures regenerated")


if __name__ == "__main__":
    main()






