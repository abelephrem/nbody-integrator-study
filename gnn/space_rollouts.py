"""GNN Stage C: full-screen 3D "space" views of the rollouts (three.js), kick-D3 overlaid on the true-force reference."""

import json
import sys

import numpy as np

from gnn.animate_rollouts import BODY_COLOURS, CHOSEN, MODEL
from gnn.metrics import about_com, cluster_divergence, horizon, load_rollout, two_body_divergence
from gnn.rollout import read_run

TEMPLATE_PATH = "gnn/space_template.html"
SCENE_RADIUS = 10.0  # scene units the system is scaled to fit
MAX_POINTS = 6000  # checkpoints kept per run; the page interpolates between them


def page_data(name, t_end):
    masses = read_run(name)["masses"]
    ref, ref_vel, times = load_rollout("reference", name)
    gnn, _, _ = load_rollout(MODEL, name)
    if t_end is not None:  # keep only the start, as in the videos
        keep = times <= t_end + 1e-9
        ref, gnn, times = ref[keep], gnn[keep], times[keep]
    ref, gnn = about_com(masses, ref), about_com(masses, gnn)
    vel0 = ref_vel[0] - (masses[:, None] * ref_vel[0]).sum(axis=0) / masses.sum()  # start velocities in the centre-of-mass frame
    spin = (masses[:, None] * np.cross(ref[0], vel0)).sum(axis=0)  # total angular momentum: the camera looks down it
    spin /= np.linalg.norm(spin)
    is_cluster = name.startswith("cluster")
    div = cluster_divergence(masses, gnn, ref) if is_cluster else two_body_divergence(masses, gnn, ref)
    t_hor = horizon(times, div)
    period = None if is_cluster else 2 * np.pi * np.sqrt((masses[0] * masses[1]) ** 3 / masses.sum())  # Kepler, a = m1 m2

    step = max(1, -(-len(times) // MAX_POINTS))
    ref, gnn, times, div = ref[::step], gnn[::step], times[::step], div[::step]
    scale = SCENE_RADIUS / np.abs(np.concatenate([ref, gnn])).max()  # whole run fits in the scene

    r = lambda a: np.round(a, 4).tolist()
    return {
        "title": name[:-3],
        "model": MODEL,
        "masses": masses.tolist(),
        "colours": BODY_COLOURS[:len(masses)],
        "times": r(times),
        "ref": r(ref * scale),  # (T, N, 3) scene units
        "gnn": r(gnn * scale),
        "div": [float(f"{d:.4g}") for d in div],
        "divLabel": "δ(t), RMS body separation" if is_cluster else "|Δr_rel| / a",
        "horizon": None if not np.isfinite(t_hor) else float(t_hor),
        "period": period,
        "trailTime": 0.8 * period if period else 0.6,  # trail length in simulation time
        "defaultRate": 1.0 * period if period else 0.25,  # simulation time per second at speed x1 (two-body: 1 orbit/s)
        "sceneRadius": SCENE_RADIUS,
        "viewRadius": float(np.percentile(np.linalg.norm(ref, axis=2), 70 if is_cluster else 95) * scale),  # frame where bodies usually are; cluster escapers may leave view
        "spin": spin.tolist(),
    }


def build(name, t_end, out_path):
    data = page_data(name, t_end)
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__TITLE__", f"{data['title']} in space").replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(out_path, f"{len(data['times'])} checkpoints, {len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    for name, t_end in CHOSEN.items():
        if len(sys.argv) > 1 and sys.argv[1] not in name:  # optional filter, e.g. "cluster"
            continue
        build(name, t_end, f"results/stage_c/space_{name[:-3]}.html")
