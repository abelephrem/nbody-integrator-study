"""GNN Stage C: 3D animations of a kick-D3 rollout overlaid on the true-force reference (one two-body, one cluster)."""

import sys

import matplotlib
matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from gnn.metrics import about_com, cluster_divergence, horizon, load_rollout, two_body_divergence
from gnn.rollout import read_run

MODEL = "design3_kick_200ep"
CHOSEN = {"two_body_00062.h5": None, "cluster_01073.h5": 4.0}  # file -> end time (None = whole run); the cluster's encounter (t 0.36) and horizon (0.89) are early
FRAMES = 600  # about 20 s at 30 fps, whatever the number of checkpoints
BODY_COLOURS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cc4", "#555555"]
TRAIL = 60  # frames of trail behind each body
TURN = 90  # degrees the camera turns over the whole video, so depth is visible


def outside_to_nan(points, centre, half):
    """Copy of points (..., 3) with anything outside the plotted cube set to NaN, since 3D axes don't clip lines."""
    points = points.copy()
    points[(np.abs(points - centre) > half).any(axis=-1)] = np.nan
    return points


def animate(name, out_path, t_end=None):
    run = read_run(name)
    masses = run["masses"]
    ref, _, times = load_rollout("reference", name)
    gnn, _, _ = load_rollout(MODEL, name)
    if t_end is not None:  # keep only the start, so the early events get most of the video
        keep = times <= t_end + 1e-9
        ref, gnn, times = ref[keep], gnn[keep], times[keep]
    stride = max(1, len(times) // FRAMES)  # checkpoints per frame
    ref, gnn = about_com(masses, ref), about_com(masses, gnn)  # both about their own centre of mass, as in G2
    is_cluster = name.startswith("cluster")
    div = cluster_divergence(masses, gnn, ref) if is_cluster else two_body_divergence(masses, gnn, ref)
    t_hor = horizon(times, div)
    period = 2 * np.pi * np.sqrt((masses[0] * masses[1]) ** 3 / masses.sum()) if not is_cluster else None  # Kepler, a = m1 m2, G = 1

    both = np.concatenate([ref.reshape(-1, 3), gnn.reshape(-1, 3)])
    lo, hi = np.percentile(both, 1, axis=0), np.percentile(both, 99, axis=0)  # ignore far escapers so the core stays visible
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.15  # one half-width for all axes: a cube, so shapes aren't stretched
    ref_draw, gnn_draw = outside_to_nan(ref, centre, half), outside_to_nan(gnn, centre, half)

    fig = plt.figure(figsize=(12, 5.4))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax_div = fig.add_subplot(1, 2, 2)
    for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
        setter(c - half, c + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.tick_params(labelsize=7)
    ax.set_title(f"{name[:-3]} (centre-of-mass frame)", fontsize=10)

    n_bodies = len(masses)
    sizes = 4 + 6 * masses / masses.max()  # marker diameter grows with mass
    ref_trails = [ax.plot([], [], [], color=BODY_COLOURS[i], lw=1.3)[0] for i in range(n_bodies)]
    gnn_trails = [ax.plot([], [], [], color=BODY_COLOURS[i], lw=1.1, ls="--")[0] for i in range(n_bodies)]
    ref_dots = [ax.plot([], [], [], "o", color=BODY_COLOURS[i], ms=sizes[i])[0] for i in range(n_bodies)]
    gnn_dots = [ax.plot([], [], [], "o", mfc="none", mec=BODY_COLOURS[i], mew=1.5, ms=sizes[i] + 2)[0] for i in range(n_bodies)]
    clock = ax.text2D(0.0, 0.97, "", transform=ax.transAxes, fontsize=9, va="top")

    key = [Line2D([], [], color="0.3", lw=1.3, marker="o", label="true force (solid)"),
           Line2D([], [], color="0.3", lw=1.1, ls="--", marker="o", mfc="none", label=f"{MODEL} (dashed, hollow)")]
    key += [Line2D([], [], color=BODY_COLOURS[i], marker="o", ls="none", ms=sizes[i], label=f"m = {m:.3g}") for i, m in enumerate(masses)]
    ax.legend(handles=key, fontsize=7, loc="upper right", bbox_to_anchor=(1.12, 1.0), frameon=False)

    x_scale = period if period else 1.0  # two-body time axis in orbits
    t_plot = times / x_scale
    ax_div.semilogy(t_plot[1:], div[1:], color="0.8", lw=1)  # the whole curve, faint
    shown, = ax_div.semilogy([], [], color="#2a78d6", lw=1.5)
    marker, = ax_div.semilogy([], [], "o", color="#2a78d6", ms=5)
    ax_div.axhline(0.1, color="0.2", lw=0.8)
    if np.isfinite(t_hor):
        ax_div.axvline(t_hor / x_scale, color="0.2", lw=0.8, ls=":")
        ax_div.text(t_hor / x_scale, ax_div.get_ylim()[0], f" horizon t = {t_hor:.2f}", fontsize=8, va="bottom", ha="left")
    ax_div.set_xlabel("orbits" if period else "time")
    ax_div.set_ylabel("δ(t), RMS body separation" if is_cluster else "|Δr_rel| / a")
    ax_div.set_title(f"divergence of {MODEL} from the true-force reference (line at 0.1)", fontsize=9)
    ax_div.grid(color="0.92", lw=0.5)
    fig.tight_layout()

    def update(frame):
        k = frame * stride  # checkpoint index
        start = max(0, k - TRAIL * stride)
        for i in range(n_bodies):
            ref_trails[i].set_data_3d(*ref_draw[start:k + 1, i].T)  # .T gives x, y and z rows
            gnn_trails[i].set_data_3d(*gnn_draw[start:k + 1, i].T)
            ref_dots[i].set_data_3d(*ref_draw[k:k + 1, i].T)
            gnn_dots[i].set_data_3d(*gnn_draw[k:k + 1, i].T)
        ax.view_init(elev=22, azim=-60 + TURN * frame / FRAMES)
        clock.set_text(f"t = {times[k]:.2f}" + (f"  ({times[k] / period:.1f} orbits)" if period else ""))
        shown.set_data(t_plot[1:k + 1], div[1:k + 1])
        marker.set_data([t_plot[k]], [max(div[k], 1e-16)])

    anim = FuncAnimation(fig, update, frames=len(times) // stride)  # no blitting: the 3D view turns every frame
    anim.save(out_path, writer="ffmpeg", fps=30, dpi=110)
    plt.close(fig)
    print(out_path, f"{len(times) // stride} frames", flush=True)


def animate_side_by_side(name, out_path, t_end=None):
    """Cluster version: reference and GNN in separate 3D panels with one shared camera. The GNN panel also shows
    each reference body as a grey ghost, joined to its GNN body by a line whose length is that body's error."""
    masses = read_run(name)["masses"]
    ref, _, times = load_rollout("reference", name)
    gnn, _, _ = load_rollout(MODEL, name)
    if t_end is not None:  # keep only the start, so the early events get most of the video
        keep = times <= t_end + 1e-9
        ref, gnn, times = ref[keep], gnn[keep], times[keep]
    stride = max(1, len(times) // FRAMES)
    ref, gnn = about_com(masses, ref), about_com(masses, gnn)
    div = cluster_divergence(masses, gnn, ref)
    t_hor = horizon(times, div)

    both = np.concatenate([ref.reshape(-1, 3), gnn.reshape(-1, 3)])
    lo, hi = np.percentile(both, 3, axis=0), np.percentile(both, 97, axis=0)  # tighter than two-body: late escapers may leave the box
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.15
    ref_draw, gnn_draw = outside_to_nan(ref, centre, half), outside_to_nan(gnn, centre, half)

    fig = plt.figure(figsize=(12, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[2, 1], hspace=0.25)
    ax_ref = fig.add_subplot(grid[0, 0], projection="3d")
    ax_gnn = fig.add_subplot(grid[0, 1], projection="3d")
    ax_div = fig.add_subplot(grid[1, :])  # graph spans the full width
    n_bodies = len(masses)
    sizes = 4 + 6 * masses / masses.max()

    panels = []  # (axis, positions, trail lines, body markers)
    for ax, pos, title in ((ax_ref, ref_draw, "true force (reference)"), (ax_gnn, gnn_draw, f"{MODEL}")):
        for setter, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), centre):
            setter(c - half, c + half)
        ax.set_box_aspect((1, 1, 1), zoom=1.1)  # zoom fills more of the panel
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.tick_params(labelsize=7)
        ax.set_title(title, fontsize=10)
        trails = [ax.plot([], [], [], color=BODY_COLOURS[i], lw=1.3)[0] for i in range(n_bodies)]
        dots = [ax.plot([], [], [], "o", color=BODY_COLOURS[i], ms=sizes[i])[0] for i in range(n_bodies)]
        panels.append((ax, pos, trails, dots))
    ghosts = [ax_gnn.plot([], [], [], "o", mfc="none", mec="0.55", mew=1.2, ms=sizes[i] + 1)[0] for i in range(n_bodies)]
    links = [ax_gnn.plot([], [], [], color="0.35", lw=1.0)[0] for i in range(n_bodies)]
    clock = fig.text(0.5, 0.965, "", ha="center", fontsize=11)
    fig.suptitle(f"{name[:-3]}, centre-of-mass frame", fontsize=11, y=0.995)

    key = [Line2D([], [], color=BODY_COLOURS[i], marker="o", ls="none", ms=sizes[i], label=f"m = {m:.3g}") for i, m in enumerate(masses)]
    ax_ref.legend(handles=key, fontsize=7, loc="upper left", frameon=False)
    ax_gnn.legend(handles=[Line2D([], [], color="0.35", marker="o", mfc="none", mec="0.55", label="where the reference body is; line = that body's error")],
                  fontsize=7, loc="upper left", frameon=False)

    ax_div.semilogy(times[1:], div[1:], color="0.8", lw=1)
    shown, = ax_div.semilogy([], [], color="#2a78d6", lw=1.5)
    marker, = ax_div.semilogy([], [], "o", color="#2a78d6", ms=5)
    ax_div.axhline(0.1, color="0.2", lw=0.8)
    if np.isfinite(t_hor):
        ax_div.axvline(t_hor, color="0.2", lw=0.8, ls=":")
        ax_div.text(t_hor, ax_div.get_ylim()[0], f" horizon t = {t_hor:.2f}", fontsize=8, va="bottom", ha="left")
    ax_div.set_xlabel("time")
    ax_div.set_ylabel("δ(t), RMS body separation")
    ax_div.set_title(f"divergence of {MODEL} from the reference (line at 0.1)", fontsize=9)
    ax_div.grid(color="0.92", lw=0.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    def update(frame):
        k = frame * stride
        start = max(0, k - TRAIL * stride)
        azim = -60 + TURN * frame / FRAMES
        for ax, pos, trails, dots in panels:
            for i in range(n_bodies):
                trails[i].set_data_3d(*pos[start:k + 1, i].T)
                dots[i].set_data_3d(*pos[k:k + 1, i].T)
            ax.view_init(elev=22, azim=azim)  # same camera in both panels
        for i in range(n_bodies):
            ghosts[i].set_data_3d(*ref_draw[k:k + 1, i].T)
            links[i].set_data_3d(*np.stack([gnn_draw[k, i], ref_draw[k, i]]).T)  # from the GNN body to the reference body
        clock.set_text(f"t = {times[k]:.2f}")
        shown.set_data(times[1:k + 1], div[1:k + 1])
        marker.set_data([times[k]], [max(div[k], 1e-16)])

    anim = FuncAnimation(fig, update, frames=len(times) // stride)
    anim.save(out_path, writer="ffmpeg", fps=30, dpi=100)
    plt.close(fig)
    print(out_path, f"{len(times) // stride} frames", flush=True)


if __name__ == "__main__":
    for name, t_end in CHOSEN.items():
        make = animate_side_by_side if name.startswith("cluster") else animate  # clusters are clearer in separate panels
        if len(sys.argv) > 1 and sys.argv[1] not in name:  # optional filter, e.g. "cluster" to redo only the cluster video
            continue
        make(name, f"results/stage_c/animation_{name[:-3]}.mp4", t_end)
