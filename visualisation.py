"""Stage 7 visualisation - research-evidence plots from fresh fixed-dt runs."""
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.colors 
from matplotlib.animation import FuncAnimation

from scenarios import two_body_circular, two_body_eccentric, chaotic_cluster, nondimensionalise
from simulation import run_simulation
from analysis import energy_drift, run_convergence_sweep, convergence_order
from integrators import euler_step, leapfrog_step, rk4_step
INTEGRATORS = [("Euler", euler_step), ("Leapfrog", leapfrog_step), ("RK4", rk4_step)]


def circular_period(m1=1.0, m2=1.0,r=1.0, G=1.0):
    """Period of the circular two-body orbit (K3)"""
    return 2 * np.pi * np.sqrt(r**3 / (G * (m1 + m2)))


def energy_drift_series(n_periods=200, steps_per_period=400):
    T = circular_period()
    dt = T / steps_per_period
    n_steps = n_periods * steps_per_period

    series = {}
    for name, step in INTEGRATORS:
        state = two_body_circular()  # fresh identical IC each loop
        traj = run_simulation(
            state, step, dt=dt, n_steps=n_steps,
            scenario_name="two_body", G=1.0, softening=0.0, adaptive=False,
        )
        series[name] = (traj.times / T, energy_drift(traj))  # stores an (x, y) pair under the integrator's name
    return series


def plot_energy_drift(series, out_path="figures/energy_drift.png"):
    fig, (ax_log, ax_lin) = plt.subplots(1, 2, figsize=(12, 5))

    # Euler and RK4 - monotonic, span many decades
    for name in ("Euler", "RK4"):
        t, d = series[name]
        ax_log.plot(t, np.abs(d), label=name)
    ax_log.set_yscale("log")
    ax_log.set_xlabel("time [orbits]")
    ax_log.set_ylabel(r"$|\Delta E / E_0|$")
    ax_log.set_title("Euler & RK4 (log scale)")
    ax_log.legend()
    
    # Leapfrog
    t, d = series["Leapfrog"]
    ax_lin.plot(t, d)  # signed d, not abs
    ax_lin.axhline(0.0, color="grey", linewidth=0.8)  # zero reference line
    ax_lin.set_xlabel("time [orbits]")
    ax_lin.set_ylabel(r"$\Delta E / E_0$")
    ax_lin.set_title("Leapfrog (linear scale, bounded)")

    # inset: zoom the first few orbits so the per-orbit oscillation is visible
    # (the full panel is 200 cycles packed solid, the inset shows the actual shape)
    axins = ax_lin.inset_axes([0.4, 0.4, 0.55, 0.5])  # [x0, y0, w, h] in axes fractions
    axins.plot(t, d)
    axins.axhline(0.0, color="grey", linewidth=0.8)
    axins.set_xlim(0, 5)  # first 5 orbits only
    axins.set_title("first 5 orbits", fontsize=9)
    ax_lin.indicate_inset_zoom(axins, edgecolor="grey")  # draws the connector to the zoomed region

    fig.suptitle("Energy conservation by integrator — circular two-body, 200 orbits")
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # auto-adjust spacing so labels don't overlap
    plt.savefig(out_path, dpi=150)  # render the whole figure to the .png file


def convergence_series(step_sizes, e=0.0, t_final=None):
    """Global position error at t_final for each integrator, across a range of h.
    Returns {name: errors_array} aligned to step_sizes."""
    a = 1.0
    if t_final is None:
        t_final = 2 * circular_period()  # enough error to fit, short enough to stay stable
    
    series = {}
    for name, step in INTEGRATORS:
        state = two_body_eccentric(m1=1.0, m2=1.0, a=a, e=e)
        series[name] = run_convergence_sweep(
            state, step, step_sizes, t_final=t_final, a=a, e=e, G=1.0,
        )
    return series


def fit_region(step_sizes, errors, tol=0.3):
    """Indicies of the straight middle of the log-log curve (integrator-dominated).
    Trims both ends: large-h(not-yet-assympototic) and small-h (round-off floor)."""
    logh = np.log10(step_sizes)
    loge = np.log10(errors)
    local = np.diff(loge) / np.diff(logh)  # slope between each adjacent pair
    med = np.median(local)  # robust estimate of the true order p, ignores the outliers
    clean = np.abs(local - med) < tol * abs(med)  # whch local slopes sit near the median

    # longest contiguous run of clean slops -> the straight region
    padded = np.concatenate(([False], clean, [False]))
    diffs = np.diff(padded.astype(int))  # +1 where a run starts, -1 where it ends
    starts = np.where(diffs == 1)[0]  # run start indices
    ends = np.where(diffs == -1)[0]  # run end indices (exclusive)
    k = np.argmax(ends - starts)  # the longest run
    return np.arange(starts[k], ends[k] + 1)  # +1: n slopes span n+1 points


def plot_convergence(step_sizes, series, out_path="figures/convergence.png"):
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, errors in series.items():
        r = fit_region(step_sizes, errors)
        p, c = convergence_order(step_sizes, errors, fit_slice=r)

        h_fit   = step_sizes[r]
        err_fit = np.exp(c) * h_fit**p                 # rebuild the fitted line in real space

        line, = ax.loglog(step_sizes, errors, "o", label=f"{name}:  p = {p:.2f}")
        ax.loglog(h_fit, err_fit, "--", color=line.get_color())

    ax.set_xlabel("step size  h")
    ax.set_ylabel("global position error at $t_{final}$")
    ax.set_title("Convergence order — two-body, e = 0")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def phase_space_series(e=0.5, n_periods=8, steps_per_period=2000):
    """Radial phase-space track (r, v_r) per integrator on an eccentric two_body orbit.
    A bound orbit is a closed loop; energy drift shows up as the loop spiralling."""
    T = circular_period()  # period depends on a & Mtot only, not e
    dt = T / steps_per_period
    n_steps = n_periods * steps_per_period

    series = {}
    for name, step in INTEGRATORS:
        state = two_body_eccentric(m1=1.0, m2=1.0, a=1.0, e=e)
        traj = run_simulation(state, step, dt, n_steps=n_steps, 
                              scenario_name="two_body", G=1.0, softening=0.0, adaptive=False)
        r_rel = traj.positions[:, 0, :] - traj.positions[:, 1, :]  # (T, 3) seperation vector
        v_rel = traj.velocities[:, 0, :] - traj.velocities[:, 1, :]  # (T, 3) relative velocity
        r = np.linalg.norm(r_rel, axis=1)  # (T,) seperation distance
        v_r = np.sum(v_rel * r_rel, axis=1) / r  # (T,) radial velocity
        series[name] = (r, v_r)
    return series


def plot_phase_space(series, out_path="figures/phase_space.png"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    for ax, name in zip(axes, series):
        r, v_r = series[name]
        ax.plot(r, v_r, linewidth=0.5)
        ax.set_title(name)
        ax.set_xlabel("separation $r$")
    axes[0].set_ylabel(r"radial velocity $v_r$")

    fig.suptitle("Radial phase-space portrait - two-body, e=0.5, 8 orbits")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150)


def cluster_trajectory(N=6, Q=0.5, seed=5, duration=15.0, dt=0.01, softening=0.002):
    """One chaotic-cluster trajectory, generated in-memory (no dataset on disk)."""
    state = chaotic_cluster(N=N, Q=Q, seed=seed)
    state = nondimensionalise(state)
    n_steps = round(duration / dt)
    return run_simulation(state, leapfrog_step, dt=dt, n_steps=n_steps, 
                          scenario_name="cluster", G=1.0, softening=softening, adaptive=True)


def plot_cluster_3d(traj, out_path="figures/cluster_3d.html"):
    """Rotatable 3D scene of a chaotic cluster: one line per body's path through space, 
    plus a marker at each body's final position."""
    fig = go.Figure()
    N = traj.positions.shape[1]
    palette = plotly.colors.qualitative.Plotly 

    for i in range(N):
        xs = traj.positions[:, i, 0]
        ys = traj.positions[:, i, 1]
        zs = traj.positions[:, i, 2]
        c = palette[i % len(palette)]
        # the body's whole path
        fig.add_trace(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines",
                                   line=dict(width=3, color=c), name=f"body {i}"))
        # a dot at where it ends up
        fig.add_trace(go.Scatter3d(x=[xs[-1]], y=[ys[-1]], z=[zs[-1]], mode="markers",
                                   marker=dict(size=4, color=c), showlegend=False))
    
    fig.update_layout(
        scene=dict(aspectmode="data",
                   xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                   title="Chaotic cluster - Leapfrog trajectory (rotatable)",

    )
    fig.write_html(out_path)


def animation_series(e=0.5, n_periods=6, steps_per_period=300):
    """Precomputed positions for each integrator on the same eccentric two-body IC.
    Simulated once here; the animation only reads frames from these arrays - it never steps the integrator itself."""   
    T = circular_period()
    dt = T / steps_per_period
    n_steps = n_periods * steps_per_period
    data = {}
    for name, step in INTEGRATORS:
        state = two_body_eccentric(m1=1.0, m2=1.0, a=1.0, e=e)
        traj = run_simulation(state, step, dt, n_steps=n_steps, 
                              scenario_name="two_body", G=1.0, softening=0.0, adaptive=False)
        data[name] = traj.positions
    return data


def animate_integrators(data, out_path="figures/integrators.mp4", stride=3, save=False):
    """Side-by-side Euler/Leapfrog/RK4 animation, fixed top-down 2D camera."""
    names = list(data)  # ["Euler", "Leapfrog", "RK4"]
    F_total = data[names[0]].shape[0]  # number of frames simulated
    N = data[names[0]].shape[1]  # bodies

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    lim = 2.0
    trails, heads = {}, {}
    for ax, name in zip(axes, names):
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")  # circles look circular
        ax.set_title(name)
        ax.set_xlabel("x")
        tl, hd = [], []
        for i in range(N):
            (trail_line,) = ax.plot([], [], linewidth=0.8)
            (head_dot,) = ax.plot([], [], "o", markersize=6,
                                color=trail_line.get_color())
            tl.append(trail_line)
            hd.append(head_dot)
            trails[name] = tl
            heads[name] = hd
            axes[0].set_ylabel("y")
    
    def update(frame):
        f = frame * stride  # maps animation frame to a real simulation index
        updated = []
        for name in names:
            pos = data[name]
            for i in range(N):
                trails[name][i].set_data(pos[:f+1, i, 0], pos[:f+1, i, 1])  # redraws the path from start up to now
                heads[name][i].set_data([pos[f, i, 0]], [pos[f, i, 1]])  # current point
                updated += [trails[name][i], heads[name][i]]
        return updated  # blit needs the changed artists
        
    anim = FuncAnimation(fig, update, frames=F_total // stride, 
                        interval=30, blit=True)

    if save:
        anim.save(out_path, writer="ffmpeg", fps=30, dpi=150)
    else:
        plt.show()
    return anim



if __name__ == "__main__":
    plot_energy_drift(energy_drift_series())  # default n_periods=200
    hs = np.geomspace(1e-1, 1e-4, 12)
    plot_convergence(hs, convergence_series(hs))
    plot_phase_space(phase_space_series())
    plot_cluster_3d(cluster_trajectory())



