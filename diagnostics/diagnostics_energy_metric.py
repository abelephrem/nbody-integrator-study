"""Diagnostic: is the Q-ordering of dataset energy-drift failures a property of the
METRIC rather than of the integration?

`validate_dataset` scores a run by max |(E(t)-E_0)/E_0|. With Q = 2T_0/|U_0|,

    E_0 = T_0 + U_0 = |U_0| * (Q/2 - 1)   ==>   |E_0| = |U_0| * |1 - Q/2|

so the DENOMINATOR collapses as Q -> 2. If Leapfrog's absolute energy error scales
with the interaction energy |U_0| (which is what the integrator actually sees - it
never computes E_0), then identical integration quality C = |dE|/|U_0| is amplified
by A(Q) = |1 - Q/2|^-1 when expressed relative to E_0.

This script recomputes both metrics per run and re-bins the failure rates. It is
READ-ONLY with respect to the dataset: nothing is regenerated, deleted or retuned.

Outputs (results/):
    energy_metric_diagnostic.csv   per-run table
    energy_metric_bins.csv         failure counts, old vs new, binned three ways
    energy_metric_threshold.csv    failure rate vs threshold on C
    figures/severe_runs_drift.png  dE/E vs t for the severe runs
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]  # repo root for `from bodies import`,
#                                                 this dir for sibling diagnostics
import glob
import os

import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bodies import SystemState
from analysis import total_energy

DATA_DIR = "C:/nbody_data"
CLUSTER_TOL = 3e-2  # the existing tolerance on the OLD metric - reported, never changed
SEVERE = 1e-1  # "severe" band from the Stage 8 write-up


def kinetic(masses, velocities):
    return 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))


def load_run(path):
    """Everything one cluster run contributes, in a flat dict."""
    with h5py.File(path, "r") as f:
        pos = f["positions"][:]
        vel = f["velocities"][:]
        masses = f["masses"][:]
        times = f["times"][:]
        attrs = {k: f.attrs[k] for k in f.attrs}

    G = attrs["G"]
    eps = attrs["epsilon"]

    # Energy series exactly as validate_dataset builds it: SOFTENED potential, at the
    # saved snapshots only. Any conversion has to sit on this same series.
    E = np.array([
        total_energy(SystemState(masses, pos[t], vel[t]), G=G, softening=eps)
        for t in range(len(times))
    ])
    dE = E - E[0]

    T0 = kinetic(masses, vel[0])
    U0_soft = E[0] - T0  # negative by the code's sign convention
    # Unsoftened U_0 is what validate_dataset's own Q check uses; kept for cross-check.
    E0_unsoft = total_energy(SystemState(masses, pos[0], vel[0]), G=G, softening=0.0)
    U0_unsoft = E0_unsoft - T0

    # Minimum pair separation over the saved snapshots (diagonal excluded).
    disp = pos[:, :, None, :] - pos[:, None, :, :]
    dist = np.linalg.norm(disp, axis=3)
    i = np.arange(pos.shape[1])
    dist[:, i, i] = np.inf
    min_sep_series = dist.min(axis=(1, 2))

    # Energy-error timescale. In these nondimensional units M=1 and the virial radius
    # is 1, so dt is identical everywhere; the crossing time is not, because it is
    # built from |E_0| which collapses near Q=2. t_U uses |U_0| instead - the
    # Q-independent comparison.
    M = np.sum(masses)
    t_cross_E = G * M**2.5 / (2 * abs(E[0]))**1.5
    t_cross_U = G * M**2.5 / (2 * abs(U0_soft))**1.5

    # How the loss arrives: share of the total |dE| carried by the single biggest
    # snapshot-to-snapshot jump. ~1 means one event, ~0 means steady accumulation.
    steps = np.abs(np.diff(dE))
    jump_share = float(steps.max() / steps.sum()) if steps.sum() > 0 else np.nan

    return {
        "run": os.path.basename(path),
        "seed": int(attrs["seed"]),
        "split": str(attrs["split"]),
        "N": int(attrs["N"]),
        "mass_ratio": float(attrs["mass_ratio"]),
        "Q_tag": float(attrs["Q"]),
        "T0": T0,
        "U0_soft": U0_soft,
        "U0_unsoft": U0_unsoft,
        "E0_soft": E[0],
        "dt": float(attrs["dt"]),
        "max_n_sub": int(attrs["max_n_sub"]),
        "force_evals": int(attrs["force_evals"]),
        "n_snapshots": len(times),
        "t_cross_E": t_cross_E,
        "t_cross_U": t_cross_U,
        "min_sep": float(min_sep_series.min()),
        "epsilon": eps,
        "jump_share": jump_share,
        "dE_max_abs": float(np.max(np.abs(dE))),
        "times": times,
        "dE_over_E0": dE / E[0],
    }


def build_table(paths):
    rows = []
    for path in paths:
        r = load_run(path)
        Q_soft = 2 * r["T0"] / abs(r["U0_soft"])  # consistent with the drift series
        Q_unsoft = 2 * r["T0"] / abs(r["U0_unsoft"])  # what validate_dataset measures
        old = r["dE_max_abs"] / abs(r["E0_soft"])
        new = r["dE_max_abs"] / abs(r["U0_soft"])
        amp = abs(1 - Q_soft / 2)
        rows.append({
            **{k: v for k, v in r.items() if k not in ("times", "dE_over_E0")},
            "Q_soft": Q_soft,
            "Q_unsoft": Q_unsoft,
            "amp_factor": amp,
            "old_metric": old,
            "new_metric": new,
            # exact-conversion cross-check: C should equal |dE/E_0| * |1 - Q/2|
            "conv_residual": abs(old * amp - new) / new,
            "old_fail": old > CLUSTER_TOL,
            "h_over_tcrossE": r["dt"] / r["t_cross_E"],
            "h_over_tcrossU": r["dt"] / r["t_cross_U"],
            "min_sep_over_eps": r["min_sep"] / r["epsilon"],
            "series": r,  # kept in memory for the severe-run plot; not written to CSV
        })
    return rows


def q_bin(q):
    """The sampling intervals, kept separate. The Stage 8 write-up quoted three bins
    (7/12/40%), which folded the Q_interp band 0.7-1.1 into the 1.1-1.5 one; split
    out here so the middle rate isn't an average of two different populations."""
    if q < 0.7:
        return "Q<0.7"
    if q < 1.1:
        return "0.7<=Q<1.1"
    if q < 1.5:
        return "1.1<=Q<1.5"
    return "Q>=1.5"


def bin_report(rows, key, binner, threshold_new):
    """Failure counts under both metrics, grouped by binner(row[key])."""
    out = []
    labels = []
    for r in rows:
        lab = binner(r[key])
        if lab not in labels:
            labels.append(lab)
    for lab in labels:
        sel = [r for r in rows if binner(r[key]) == lab]
        n = len(sel)
        old_f = sum(r["old_fail"] for r in sel)
        new_f = sum(r["new_metric"] > threshold_new for r in sel)
        out.append({
            "axis": key, "bin": str(lab), "n": n,
            "old_fail": old_f, "old_rate": old_f / n,
            "new_fail": new_f, "new_rate": new_f / n,
        })
    return out


def write_csv(path, rows, columns):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write(",".join(columns) + "\n")
        for r in rows:
            f.write(",".join(_fmt(r[c]) for c in columns) + "\n")


def _fmt(v):
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def plot_severe(rows, out_path="figures/severe_runs_drift.png"):
    """dE/E vs t for the runs above the severe band - one event or steady creep?"""
    severe = sorted((r for r in rows if r["old_metric"] > SEVERE),
                    key=lambda r: -r["old_metric"])
    if not severe:
        return severe
    fig, ax = plt.subplots(figsize=(8, 5))
    for r in severe:
        s = r["series"]
        ax.plot(s["times"], s["dE_over_E0"],
                label=f"{r['run']} Q={r['Q_soft']:.2f} sep/eps={r['min_sep_over_eps']:.1f}")
    ax.set_xlabel("t (nondimensional)")
    ax.set_ylabel(r"$\Delta E / E_0$")
    ax.set_title(f"Severe runs (old metric > {SEVERE:g})")
    ax.legend(fontsize=6)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return severe


def spearman(x, y):
    """Rank correlation - the relationships here are monotone, not linear."""
    def rank(a):
        order = np.argsort(a)
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        return r
    rx, ry = rank(np.asarray(x)), rank(np.asarray(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    paths = sorted(glob.glob(os.path.join(DATA_DIR, "cluster_*.h5")))
    rows = build_table(paths)
    print(f"{len(rows)} cluster runs")

    cols = ["run", "split", "N", "mass_ratio", "Q_tag", "Q_soft", "Q_unsoft",
            "amp_factor", "old_metric", "new_metric", "old_fail", "conv_residual",
            "dt", "h_over_tcrossE", "h_over_tcrossU", "max_n_sub", "force_evals",
            "min_sep", "min_sep_over_eps", "jump_share", "n_snapshots"]
    write_csv("results/energy_metric_diagnostic.csv", rows, cols)

    # --- conversion cross-check -------------------------------------------------
    res = np.array([r["conv_residual"] for r in rows])
    print(f"conversion residual  max={res.max():.3e}  median={np.median(res):.3e}")

    # --- Q range ----------------------------------------------------------------
    qs = np.array([r["Q_soft"] for r in rows])
    qu = np.array([r["Q_unsoft"] for r in rows])
    print(f"Q_soft   min={qs.min():.4f} max={qs.max():.4f}  #(Q>2)={int((qs > 2).sum())}")
    print(f"Q_unsoft min={qu.min():.4f} max={qu.max():.4f}  #(Q>2)={int((qu > 2).sum())}")
    print(f"amp factor |1-Q/2|: min={min(r['amp_factor'] for r in rows):.4f} "
          f"max={max(r['amp_factor'] for r in rows):.4f}")

    # --- threshold sweep on C ---------------------------------------------------
    # Overall, and split by Q bin: a threshold choice is only defensible if the Q
    # ordering it produces is stable across the range, not just at one cut.
    qlabels = ["Q<0.7", "0.7<=Q<1.1", "1.1<=Q<1.5", "Q>=1.5"]
    sweep = []
    for thr in np.geomspace(1e-3, 1e-1, 21):
        n_fail = sum(r["new_metric"] > thr for r in rows)
        entry = {"threshold": thr, "n_fail": n_fail, "rate": n_fail / len(rows)}
        for lab in qlabels:
            sel = [r for r in rows if q_bin(r["Q_soft"]) == lab]
            entry[f"rate_{lab}"] = sum(r["new_metric"] > thr for r in sel) / len(sel)
        sweep.append(entry)
    write_csv("results/energy_metric_threshold.csv", sweep,
              ["threshold", "n_fail", "rate"] + [f"rate_{l}" for l in qlabels])
    print("\nthreshold on C -> failure rate (overall | " + " | ".join(qlabels) + ")")
    for e in sweep[::4]:
        print(f"  {e['threshold']:.2e}  {e['rate']:6.1%}  |  "
              + "  ".join(f"{e['rate_' + l]:6.1%}" for l in qlabels))

    # Match the OLD metric's overall failure count, so the bin comparison is
    # rate-for-rate rather than an artefact of a looser cut.
    n_old = sum(r["old_fail"] for r in rows)
    new_sorted = np.sort([r["new_metric"] for r in rows])[::-1]
    thr_matched = float(new_sorted[n_old]) if n_old < len(rows) else 0.0
    print(f"old failures: {n_old}/{len(rows)}; count-matched threshold on C = {thr_matched:.3e}")

    bins = []
    bins += bin_report(rows, "Q_soft", q_bin, thr_matched)
    bins += bin_report(rows, "N", lambda v: v, thr_matched)
    bins += bin_report(rows, "mass_ratio", lambda v: v, thr_matched)
    write_csv("results/energy_metric_bins.csv", bins,
              ["axis", "bin", "n", "old_fail", "old_rate", "new_fail", "new_rate"])
    for b in bins:
        print(f"  {b['axis']:>11} {b['bin']:>10}  n={b['n']:3d}  "
              f"old {b['old_fail']:3d} ({b['old_rate']:5.1%})  "
              f"new {b['new_fail']:3d} ({b['new_rate']:5.1%})")

    # --- resolution correlations ------------------------------------------------
    for name in ["Q_soft", "h_over_tcrossE", "h_over_tcrossU", "max_n_sub",
                 "force_evals", "min_sep_over_eps", "N"]:
        v = [r[name] for r in rows]
        print(f"  spearman(C, {name:>16}) = {spearman(v, [r['new_metric'] for r in rows]):+.3f}"
              f"   |  old: {spearman(v, [r['old_metric'] for r in rows]):+.3f}")

    # --- severe runs ------------------------------------------------------------
    severe = plot_severe(rows)
    print(f"\nsevere runs (old > {SEVERE:g}): {len(severe)}")
    for r in severe:
        print(f"  {r['run']}  Q={r['Q_soft']:.3f}  old={r['old_metric']:.3e} "
              f"C={r['new_metric']:.3e}  min_sep/eps={r['min_sep_over_eps']:.2f} "
              f"max_n_sub={r['max_n_sub']}  jump_share={r['jump_share']:.3f}")


if __name__ == "__main__":
    main()
