"""GNN Stage C: one page to pick any test or holdout orbit (two-body, cluster, figure-eight) and watch a GNN rollout against the reference."""

import base64
import json
import os

import h5py
import numpy as np

from gnn.animate_rollouts import BODY_COLOURS
from gnn.figure8_rollouts import NAME as FIG8_FILE, period_in_dataset_units
from gnn.g2_slopes import MODELS
from gnn.metrics import about_com, cluster_divergence, horizon, two_body_divergence
from gnn.rollout import DATA_DIR, read_run, rollout_files
from gnn.run_rollouts import OUT_DIR

TEMPLATE_PATH = "gnn/explorer_template.html"
OUT_PATH = "results/stage_c/orbit_explorer.html"
VIEW = 10.0  # scene units for the typical size of each system
SAMPLES_PER_ORBIT = 64  # two-body and figure-eight; the page draws cubic curves between samples using the velocities
CLUSTER_STRIDE = 2  # clusters keep every 2nd checkpoint (dt 0.02)
MODEL_LABELS = {"design3_kick_200ep": "kick-D3", "design3_time_200ep": "time-D3", "design2_kick_200ep": "D2 (no third law)"}


def load(job, file):
    with np.load(os.path.join(OUT_DIR, job, file)) as saved:
        return saved["pos"], saved["vel"], saved["times"]


def pack(pos, vel, p_scale, v_scale):
    """(T, N, 3) positions and velocities -> base64 int16, interleaved per body as x y z vx vy vz."""
    both = np.concatenate([pos / p_scale, vel / v_scale], axis=2)  # (T, N, 6), each within [-1, 1]
    return base64.b64encode(np.round(both * 32000).astype("<i2").tobytes()).decode("ascii")


def run_entry(kind, file, masses, info, split="test"):
    ref, ref_vel, times = load("reference", file)
    runs = {"reference": (ref, ref_vel)}
    models = MODELS[split]  # holdout: kick-D3 and time-D3 only (D2 holdout cut 2026-11-12)
    for model in models:
        pos, vel, _ = load(model, file)
        runs[model] = (pos, vel)

    com_vel = lambda v: v - ((masses[:, None] * v).sum(axis=1) / masses.sum())[:, None]
    runs = {k: (about_com(masses, p), com_vel(v)) for k, (p, v) in runs.items()}
    ref, ref_vel = runs["reference"]

    divergence = two_body_divergence if kind == "two_body" else cluster_divergence  # figure-eight is 3 bodies: use δ, as for clusters
    div = {m: divergence(masses, runs[m][0], ref) for m in models}
    hor = {m: horizon(times, d) for m, d in div.items()}

    period = info.get("period")
    step = CLUSTER_STRIDE if period is None else max(1, int(round(period / (times[1] - times[0]) / SAMPLES_PER_ORBIT)))
    keep = np.arange(0, len(times), step)
    early = ref[: max(2, len(ref) // 4)]  # first quarter, before escapers wander off
    scale = VIEW / np.percentile(np.linalg.norm(early, axis=2), 70)

    scaled = {k: (p[keep] * scale, v[keep] * scale) for k, (p, v) in runs.items()}
    p_scale = max(np.abs(p).max() for p, _ in scaled.values()) * 1.001
    v_scale = max(np.abs(v).max() for _, v in scaled.values()) * 1.001

    spin = (masses[:, None] * np.cross(ref[0], ref_vel[0])).sum(axis=0)
    if np.linalg.norm(spin) < 1e-6 * np.abs(ref[0]).max() * np.abs(ref_vel[0]).max():  # figure-eight: total L = 0
        spin = np.linalg.svd(early.reshape(-1, 3), full_matrices=False)[2][-1]  # normal of the plane the bodies move in
    spin = spin / np.linalg.norm(spin)
    return {
        "file": file.replace(".npz", ""), "kind": kind, "info": info, "split": split,
        "masses": masses.tolist(), "colours": BODY_COLOURS[:len(masses)],
        "t0": float(times[0]), "dt": float(times[keep[1]] - times[keep[0]]), "n": len(keep),
        "pScale": float(p_scale), "vScale": float(v_scale),
        "runs": {k: pack(p, v, p_scale, v_scale) for k, (p, v) in scaled.items()},
        "div": {m: [float(f"{x:.3g}") for x in d[keep]] for m, d in div.items()},
        "horizon": {m: (None if not np.isfinite(h) else float(h)) for m, h in hor.items()},
        "period": period, "spin": spin.tolist(),
        "trailTime": 0.8 * period if period else 0.6,
        "defaultRate": period if period else 0.25,
    }


def main():
    entries = []
    for split, name in [(s, n) for s in ("test", "holdout") for n in rollout_files(s)]:
        run = read_run(name)
        with h5py.File(os.path.join(DATA_DIR, name), "r") as f:
            attrs = dict(f.attrs)
        category = str(attrs.get("split", split))  # holdout files say which setting is new, e.g. ratio_extrap
        mr = int(round(float(attrs["mass_ratio"])))
        file = name.replace(".h5", ".npz")
        if name.startswith("two_body"):
            m = run["masses"]
            period = 2 * np.pi * np.sqrt((m[0] * m[1]) ** 3 / m.sum())  # Kepler with a = m1 m2, G = M = 1
            info = {"massRatio": mr, "e": round(float(attrs["e"]), 2), "period": period, "category": category}
            entries.append(run_entry("two_body", file, m, info, split))
        else:
            info = {"massRatio": mr, "N": int(attrs["N"]), "Q": round(float(attrs["Q"]), 2),
                    "closest": round(float(attrs["min_sep_run"]) / float(attrs["epsilon"]), 2), "category": category}
            entries.append(run_entry("cluster", file, run["masses"], info, split))
        print(split, name, "packed", flush=True)

    masses = np.full(3, 1 / 3)
    entries.append(run_entry("figure_eight", FIG8_FILE, masses, {"massRatio": 1, "N": 3, "period": period_in_dataset_units()}))

    data = {"models": {k: MODEL_LABELS[k] for k in MODELS["test"]}, "entries": entries}
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        html = f.read()
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":")))
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html)
    print(OUT_PATH, f"{len(entries)} runs, {len(html) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
