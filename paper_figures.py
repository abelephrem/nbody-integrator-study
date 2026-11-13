"""Stage 10: every paper figure, read from data already on disk. Plot only: no simulation, no training."""

import csv
import glob
import os
import pickle

import h5py
import matplotlib
matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from analysis import convergence_order, energy_drift
from experiments import RAW_DIR
from scenarios import circular_period
from simulation import load_trajectory
from gnn.metrics import angular_momentum_error, cluster_divergence, energy_errors, load_rollout
from gnn.rollout import DATA_DIR, read_run, rollout_files

OUT = "figures/paper"
STAGE_C = "results/stage_c"
CACHE = "C:/nbody_results/paper_cache"  # Fig. 2 evaluation cache, outside the repo and OneDrive
SINGLE, FULL = 3.4, 7.0  # column widths in inches
F = 2e-3  # Stage B loss floor
SEEDS = ["baseline", "baseline2", "baseline3"]

# one name, colour, line style and width per method, the same in every figure
STYLE = {
    "leapfrog": ("Leapfrog", "#000000", "-", 1.0),
    "rk4": ("RK4", "#E69F00", "--", 1.0),
    "euler": ("Euler", "#999999", ":", 1.0),
    "design3_kick_200ep": ("kick-D3", "#0072B2", "-", 1.0),
    "design3_time_200ep": ("time-D3", "#D55E00", "-.", 1.0),
    "design2_kick_200ep": ("D2", "#009E73", "--", 1.0),
    "baseline": ("nudged baselines", "#BBBBBB", "-", 0.6),
}
STYLE["reference"] = STYLE["leapfrog"]  # Stage C's reference run is true-force Leapfrog
MARKER = {"design3_kick_200ep": "o", "design3_time_200ep": "s", "design2_kick_200ep": "^"}
CHECKS = []  # (figure, check, pass?, numbers) for the end-of-run report


def setup_style():
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,  # cmr10 has no minus glyph; mathtext supplies it
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4, "ytick.minor.width": 0.4, "lines.linewidth": 1.0,
        "legend.frameon": False, "axes.grid": False, "pdf.fonttype": 42,  # embed TrueType fonts in the PDF
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def check(fig, name, ok, detail):
    CHECKS.append((fig, name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {fig}: {name} -- {detail}")


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def style_kw(job, **extra):
    label, colour, ls, lw = STYLE[job]
    return {"color": colour, "ls": ls, "lw": lw, "label": label, **extra}


def panel_label(ax, text, x=0.02, y=0.97):
    ax.text(x, y, text, transform=ax.transAxes, ha="left", va="top", fontsize=8)


def light_log_grid(ax, axis="y"):
    ax.grid(True, axis=axis, which="major", color="0.92", lw=0.4)
    ax.set_axisbelow(True)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=300)
    plt.close(fig)
    print(f"  saved {OUT}/{name}.pdf and .png")


# ---------- Fig. 1: energy error of the three integrators, Set B ----------

def binned_max(x, y, size):
    """Max of y within consecutive bins of `size` points, plotted at each bin's last x (keeps the envelope)."""
    n = len(y) // size * size
    return x[size - 1:n:size], y[:n].reshape(-1, size).max(axis=1)


def fig1(bin_size=10):
    period = circular_period()
    table = {(r["scenario"], r["integrator"]): float(r["max_E_drift"]) for r in read_csv("results/long_runs.csv") if r["set"] == "B"}
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=True)
    for ax, scenario, tag in zip(axes, ("circular", "eccentric"), ("(a) circular", "(b) eccentric, $e = 0.5$")):
        for name in ("leapfrog", "rk4", "euler"):  # Leapfrog's band first, so RK4 stays visible on top
            path, = glob.glob(os.path.join(RAW_DIR, "B", f"{scenario}_{name}_*.h5"))  # exactly one run per pair
            traj = load_trajectory(path)
            drift = np.abs(energy_drift(traj))
            x, y = binned_max(traj.times[1:] / period, drift[1:], bin_size)  # skip t = 0, where the drift is exactly 0
            ax.plot(x, y, **style_kw(name, rasterized=False))
            ok = np.isclose(drift.max(), table[(scenario, name)], rtol=1e-12) and np.isclose(y.max(), drift[1:].max(), rtol=1e-12)
            check("Fig. 1", f"{scenario} {name} max matches long_runs.csv", ok, f"{drift.max():.6e} vs {table[(scenario, name)]:.6e}")
        ax.set_yscale("log")
        ax.set_xlim(0, 300)
        ax.set_xlabel(r"$t$ (orbits)")
        light_log_grid(ax)
        panel_label(ax, tag)
    axes[0].set_ylabel(r"$|\Delta E/E_0|$")
    axes[0].set_ylim(1e-12, 1e3)  # headroom above Euler for the panel labels
    axes[1].legend(loc="lower right", ncol=3)
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig1_energy_integrators")


# ---------- Fig. 2: one-step acceleration error by decade of |a| ----------

FIG2_MODELS = ["design2_kick_200ep", "design3_kick_200ep", "design3_time_200ep"]


def fig2_decades():
    """Per-decade median relative error and bootstrap band per model and group, as gnn/evaluate.py computes it."""
    path = f"{CACHE}/fig2_decades.pkl"
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    from gnn.evaluate import collect, load_model, per_decade  # torch only needed here
    from gnn.split import load_split
    from gnn.train import OUT_DIR as MODEL_DIR
    files = load_split()["test"]
    out = {}
    for model in FIG2_MODELS:
        bodies, _, _ = collect(load_model(os.path.join(MODEL_DIR, f"{model}.pt"), "final_state"), files)
        two = bodies["two_body"]
        for group, mask in (("two-body", two), ("clusters", ~two)):
            out[(model, group)] = per_decade(bodies["rel"][mask], bodies["amag"][mask], bodies["file"][mask], len(files))
        print(f"  evaluated {model}", flush=True)
    os.makedirs(CACHE, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(out, f)
    return out


def fig2():
    decades = fig2_decades()
    fig, axes = plt.subplots(1, 2, figsize=(FULL, 2.4), sharey=True)
    for ax, group, tag in zip(axes, ("clusters", "two-body"), ("(a) clusters", "(b) two-body")):
        ax.axvspan(1e-4, F, color="0.92", lw=0, zorder=0)
        ax.text(1.3e-4, 2.2e-4, "below\nloss floor", fontsize=6.5, color="0.35", va="bottom")
        ax.axhline(1.0, color="0.35", lw=0.6, ls=":", zorder=1)
        ax.text(2e4, 1.12, "zero-prediction baseline", fontsize=6.5, color="0.35", ha="right", va="bottom")
        for model in FIG2_MODELS:
            rows = [r for r in decades[(model, group)] if not np.isnan(r["median"])]
            x = [r["centre"] for r in rows]
            _, colour, _, _ = STYLE[model]
            ax.fill_between(x, [r["med_lo"] for r in rows], [r["med_hi"] for r in rows], color=colour, alpha=0.2, lw=0, zorder=2)
            ax.plot(x, [r["median"] for r in rows], marker=MARKER[model], ms=3, zorder=3, **style_kw(model))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-4, 1e5)
        ax.set_xlabel(r"$|\mathbf{a}|$")
        light_log_grid(ax)
        panel_label(ax, tag)
    axes[0].set_ylim(1e-4, 3)
    axes[0].set_ylabel("median relative error")
    axes[1].legend(loc="center right")
    fig.tight_layout(w_pad=0.8)
    save(fig, "fig2_one_step_error")


# ---------- Fig. 3: divergence curves and the slope test ----------

def pick_clusters():
    """(typical, strongest) test clusters by the fixed rule on kick-D3 main-window ratios."""
    rows = [r for r in read_csv(f"{STAGE_C}/g2_ratios_test.csv") if r["model"] == "design3_kick_200ep" and r["window"] == "main"]
    ratio = {r["file"]: float(r["ratio"]) for r in rows}
    med = np.median(list(ratio.values()))
    typical = min(ratio, key=lambda n: (abs(ratio[n] - med), ratio[n]))  # tie (even count): take the lower median
    strongest = max(ratio, key=ratio.get)
    print(f"  {len(ratio)} voting clusters, median ratio {med:.3f}; typical {typical} ({ratio[typical]:.3f}), "
          f"strongest {strongest} ({ratio[strongest]:.3f})")
    return typical, strongest, ratio


def fig3():
    typical, strongest, _ = pick_clusters()
    fig = plt.figure(figsize=(FULL, 2.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25], wspace=0.38)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharey=ax_a)
    for ax, name, tag in ((ax_a, typical, "(a) typical"), (ax_b, strongest, "(b) strongest model effect")):
        run = read_run(name)
        ref, _, times = load_rollout("reference", name)
        ax.axhspan(1e-4, 0.1, color="0.95", lw=0, zorder=0)
        ax.axhline(0.1, color="0.3", lw=0.6, zorder=1)
        for k, job in enumerate(SEEDS + ["design3_time_200ep", "design3_kick_200ep"]):
            pos, _, _ = load_rollout(job, name)
            kw = style_kw("baseline" if job in SEEDS else job)
            if job in SEEDS and k > 0:
                kw["label"] = None  # one legend entry for the three baselines
            ax.plot(times[1:], cluster_divergence(run["masses"], pos, ref)[1:], zorder=2 if job in SEEDS else 3, **kw)
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 1e3)  # headroom above the saturated curves for the panel label
        ax.set_xlim(0, times[-1])
        ax.set_xlabel(r"$t$ ($N$-body units)")
        panel_label(ax, tag)
        ax.text(0.98, 0.03, name.replace(".h5", "").replace("_", " "), transform=ax.transAxes,  # cmr10 has no underscore
                ha="right", va="bottom", fontsize=6, color="0.35")
    ax_a.set_ylabel(r"$\delta$")
    plt.setp(ax_b.get_yticklabels(), visible=False)
    ax_a.legend(loc="lower right", bbox_to_anchor=(1.0, 0.08), fontsize=6.5)

    ax = fig.add_subplot(gs[2])
    ratios = {s: read_csv(f"{STAGE_C}/g2_ratios_{s}.csv") for s in ("test", "holdout")}
    votes = {s: {r["model"]: int(r["yes"]) for r in read_csv(f"{STAGE_C}/g2_vote_{s}.csv") if r["window"] == "main"} for s in ratios}
    columns = {"test": ["design3_kick_200ep", "design3_time_200ep", "design2_kick_200ep"],
               "holdout": ["design3_kick_200ep", "design3_time_200ep"]}
    data, rk4_present = [], True
    for split, models in columns.items():
        for model in models:
            r = np.array([float(x["ratio"]) for x in ratios[split] if x["model"] == model and x["window"] == "main"])
            above = int((r >= 1.5).sum())
            check("Fig. 3", f"{STYLE[model][0]} {split}: points >= 1.5 equal vote yes", above == votes[split][model],
                  f"{above} vs {votes[split][model]} of {len(r)}")
            data.append((STYLE[model][0], split, r, STYLE[model][1]))
        # RK4 control: RK4 main-window slope / the same baseline mean, voting clusters only
        fits = {x["file"]: float(x["slope"]) for x in read_csv(f"{STAGE_C}/g2_fits_{split}.csv") if x["job"] == "rk4" and x["window"] == "main"}
        base = {x["file"]: float(x["baseline_mean"]) for x in ratios[split] if x["model"] == "design3_kick_200ep" and x["window"] == "main"}
        rk4_present &= bool(fits)
        r = np.array([fits.get(n, np.nan) / b for n, b in base.items()])
        print(f"  RK4 {split}: {np.isfinite(r).sum()} of {len(r)} voting clusters have a slope; {int((r >= 1.5).sum())} at or above 1.5; median {np.nanmedian(r):.2f}")
        data.append(("RK4", split, r, "0.55"))
    for k, (label, split, r, colour) in enumerate(data):
        r = r[np.isfinite(r)]
        jitter = np.random.default_rng(k).uniform(-0.18, 0.18, len(r))  # spread the dots sideways
        ax.scatter(k + jitter, r, s=7, color=colour, edgecolor="none", zorder=3)
    ax.axhline(1.0, color="0.3", lw=0.6, ls="--", zorder=1)
    ax.axhline(1.5, color="0.3", lw=0.6, zorder=1)
    n_test = len(columns["test"]) + 1
    ax.axvline(n_test - 0.5, color="0.6", lw=0.4)
    ax.set_xticks(range(len(data)), [lbl for lbl, _, _, _ in data], rotation=45, ha="right", rotation_mode="anchor")
    for k, (_, _, _, colour) in enumerate(data):
        ax.get_xticklabels()[k].set_color(colour if colour != "0.55" else "0.4")
    ax.set_xlim(-0.6, len(data) - 0.4)
    ax.set_yscale("log")
    ax.set_ylim(top=60)  # headroom for the panel and group labels
    for x, text in ((n_test / 2 - 0.5, "test"), ((n_test + len(data)) / 2 - 0.5, "holdout")):
        ax.text(x, 0.97, text, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color="0.35")
    ax.set_ylabel("slope ratio")
    light_log_grid(ax)
    panel_label(ax, "(c)")
    save(fig, "fig3_divergence_slopes")
    return typical, strongest, rk4_present


# ---------- Fig. 4: conservation in rollouts ----------

TOLERANCE = {"cluster": 1e-2, "two_body": 1e-4}


def g3_series(split, jobs):
    """series[(kind, job)] = list of (times, |dE/U0|, dL) per file, exactly as gnn/g3_conservation.py builds it."""
    series = {}
    for name in rollout_files(split):
        run = read_run(name)
        kind = "cluster" if name.startswith("cluster") else "two_body"
        for job in jobs:
            pos, vel, times = load_rollout(job, name)
            dE_U, _ = energy_errors(run["masses"], pos, vel, run["softening"], run["G"])
            series.setdefault((kind, job), []).append((times, np.abs(dE_U), angular_momentum_error(run["masses"], pos, vel)))
    return series


def fig4(split="test"):
    jobs = ["reference", "rk4", "design3_kick_200ep", "design3_time_200ep"] + (["design2_kick_200ep"] if split == "test" else [])
    series = g3_series(split, jobs)
    g3 = read_csv(f"{STAGE_C}/g3_{split}.csv")
    fig, axes = plt.subplots(2, 2, figsize=(FULL, 4.0), sharex="col")
    tags = iter(["(a) clusters, energy", "(b) two-body, energy", "(c) clusters, angular momentum", "(d) two-body, angular momentum"])
    for row, k in enumerate((1, 2)):  # row 0 energy, row 1 angular momentum
        for col, kind in enumerate(("cluster", "two_body")):
            ax = axes[row, col]
            if row == 0:
                ax.axhline(TOLERANCE[kind], color="0.6", lw=0.6, ls=":", zorder=1, label="dataset tolerance")
            for job in jobs:
                curves = series[(kind, job)]
                n = min(len(c[0]) for c in curves)  # runs share checkpoints; trim to the shortest
                env = np.array([np.maximum.accumulate(c[k][:n]) for c in curves])  # running max of each file
                ax.plot(curves[0][0][1:n], np.median(env, axis=0)[1:], **style_kw(job))
                if row == 0:  # median of each file's max over the run, against g3_*.csv
                    mine = np.median([c[1].max() for c in curves])
                    theirs = np.median([float(r["max_abs_dE_U0"]) for r in g3 if r["kind"] == kind and r["job"] == job])
                    check(f"Fig. 4 {split}", f"{kind} {STYLE[job][0]} median max|dE/U0| matches g3_{split}.csv",
                          np.isclose(mine, theirs, rtol=1e-9), f"{mine:.3e} vs {theirs:.3e}")
            ax.set_yscale("log")
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * 100)  # two decades of headroom for the panel label
            ax.set_xlim(left=0)
            ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
            light_log_grid(ax)
            panel_label(ax, next(tags))
            if row == 1:
                ax.set_xlabel(r"$t$ ($N$-body units)")
    axes[0, 0].set_ylabel(r"$|\Delta E|/|U_0|$")
    axes[1, 0].set_ylabel(r"$|\Delta \mathbf{L}|/\Sigma_i|\mathbf{L}_i(0)|$")  # \Sigma_i: mathtext's \sum puts i underneath
    handles, labels = axes[0, 0].get_legend_handles_labels()
    order = [labels.index(STYLE[j][0]) for j in jobs] + [labels.index("dataset tolerance")]
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=len(order), bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=0.4, w_pad=0.8)
    save(fig, "fig4_conservation" if split == "test" else "fig4_holdout")


# ---------- Fig. 5: two-body horizons in orbits ----------

def two_body_info(name):
    with h5py.File(os.path.join(DATA_DIR, name), "r") as f:
        return {"mr": int(f.attrs["mass_ratio"]), "e": float(f.attrs["e"]), "period": float(f.attrs["period"]),
                "split": str(f.attrs["split"])}


def fig5():
    models = ["design3_kick_200ep", "design3_time_200ep", "design2_kick_200ep"]
    orbits = []  # (label, split tag, {model: horizon in orbits})
    medians = {}
    for split in ("test", "holdout"):
        rows = [r for r in read_csv(f"{STAGE_C}/g2_horizons_{split}.csv") if r["file"].startswith("two_body")]
        groups = {}
        for r in rows:
            info = two_body_info(r["file"])
            key = (info["mr"], info["e"], info["split"])
            groups.setdefault(key, {}).setdefault(r["job"], {})[r["file"]] = float(r["horizon"]) / info["period"]
        if split == "holdout":  # equivariance check: all 5 orientations of an orbit give the same horizon
            for key, by_job in sorted(groups.items()):
                for job in ("design3_kick_200ep", "design3_time_200ep"):
                    values = np.array(list(by_job[job].values()))
                    same = len(values) == 5 and np.all(values == values[0])
                    check("Fig. 5", f"holdout {key[0]}:1 e {key[1]} {STYLE[job][0]}: 5 orientations identical",
                          same, f"{len(values)} files, horizons {np.unique(values).round(4).tolist()} orbits")
                    if not same:
                        raise SystemExit("equivariance check failed: stop and report")
        for key in sorted(groups, key=lambda k: (k[0], k[1])):
            hz = {job: next(iter(v.values())) if split == "holdout" else list(v.values())[0] for job, v in groups[key].items()}
            tag = "" if split == "test" else (" (extrap.)" if "extrap" in key[2] else " (interp.)")
            orbits.append((f"{key[0]}:1, {key[1]:g}{tag}", split, hz))
        for job in models:
            vals = [h[job] for _, s, h in orbits if s == split and job in h and np.isfinite(h[job])]  # labbook: median of orbits that reach 0.1
            if vals:
                medians[(split, job)] = np.median(vals)
    for job, expect in (("design3_kick_200ep", 3.9), ("design3_time_200ep", 2.5), ("design2_kick_200ep", 6.5)):
        got = medians[("test", job)]
        check("Fig. 5", f"test median {STYLE[job][0]} = {expect} orbits", round(got, 1) == expect, f"{got:.3f}")
    expected = {(1, 0.6): (1.9, 0.8), (1, 0.9): (0.9, 0.2), (5, 0.4): (2.9, 3.9), (15, 0.4): (6.0, 1.0)}
    for label, split, hz in orbits:
        if split != "holdout":
            continue
        mr, e = label.split(" (")[0].split(", ")
        exp = expected[(int(mr.split(":")[0]), float(e))]
        got = (hz["design3_kick_200ep"], hz["design3_time_200ep"])
        check("Fig. 5", f"holdout {label} matches G4 table (kick, time)",
              all(round(g, 1) == x for g, x in zip(got, exp)), f"{got[0]:.2f}, {got[1]:.2f} vs {exp}")

    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    finite = [v for _, _, h in orbits for v in h.values() if np.isfinite(v)]
    top = 10 ** np.ceil(np.log10(max(finite)) + 0.15)
    for job, size, alpha, offset in (("design2_kick_200ep", 8, 0.45, 0.0), ("design3_time_200ep", 14, 1.0, 0.12),
                                     ("design3_kick_200ep", 14, 1.0, -0.12)):
        label, colour, _, _ = STYLE[job]
        xs = [k + offset for k, (_, _, h) in enumerate(orbits) if job in h]
        ys = [h[job] for _, _, h in orbits if job in h]
        fin = [(x, y) for x, y in zip(xs, ys) if np.isfinite(y)]
        ax.scatter(*zip(*fin), s=size, marker=MARKER[job], color=colour, alpha=alpha, edgecolor="none", label=label, zorder=3)
        for x, y in zip(xs, ys):
            if not np.isfinite(y):  # never reached 0.1: arrow at the top edge
                ax.annotate("", xy=(x, top), xytext=(x, top / 2.2), arrowprops=dict(arrowstyle="->", color=colour, lw=0.8, alpha=alpha))
    n_test = sum(s == "test" for _, s, _ in orbits)
    ax.axvline(n_test - 0.5, color="0.4", lw=0.5)
    ax.text(n_test - 0.3, 0.97, "holdout", transform=ax.get_xaxis_transform(), fontsize=6.5, color="0.35", va="top")
    ax.set_yscale("log")
    ax.set_ylim(top=top)
    ax.set_xlim(-0.6, len(orbits) - 0.4)
    ax.set_xticks(range(len(orbits)), [o[0] for o in orbits], rotation=90, fontsize=6)
    ax.set_xlabel("mass ratio, $e$")
    ax.set_ylabel("horizon (orbits)")
    light_log_grid(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handletextpad=0.2, columnspacing=1.2)
    save(fig, "fig5_two_body_horizons")
    return medians


# ---------- Appendix ----------

def figA1():
    rows = read_csv("results/convergence.csv")
    expected = {"euler": 1.0, "leapfrog": 2.00, "rk4": 4.37}
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    for name in ("euler", "leapfrog", "rk4"):
        sel = sorted([r for r in rows if r["integrator"] == name], key=lambda r: float(r["dt"]))
        h = np.array([float(r["dt"]) for r in sel])
        err = np.array([float(r["error"]) for r in sel])
        fit = np.array([r["in_fit"] == "True" for r in sel])
        p, c = convergence_order(h, err, fit_slice=np.nonzero(fit)[0])
        label, colour, ls, lw = STYLE[name]
        ax.loglog(h[fit], err[fit], "o", ms=3, color=colour, mec=colour)
        ax.loglog(h[~fit], err[~fit], "o", ms=3, mfc="none", mec=colour, mew=0.6)
        ax.loglog(h[fit], np.exp(c) * h[fit] ** p, color=colour, ls=ls, lw=lw, label=f"{label}, $p = {p:.2f}$")
        check("Fig. A1", f"{label} fitted order ~ {expected[name]}", abs(p - expected[name]) < (0.15 if name == "euler" else 0.01), f"p = {p:.3f}")
    ax.set_xlabel(r"step size $h$")
    ax.set_ylabel("position error at $t_{\\mathrm{final}}$")
    light_log_grid(ax)
    ax.legend(loc="lower right")
    save(fig, "figA1_convergence")


def figA2(h_lo=1e-3, h_hi=0.0465):
    rows = read_csv("results/energy_scaling.csv")
    h = np.array([float(r["dt"]) for r in rows])
    err = np.array([float(r["max_E_drift"]) for r in rows])
    floor = np.array([r["in_fit"] == "False" for r in rows])  # round-off floor points, as marked by fit_region
    win = (h >= h_lo) & (h <= h_hi)

    def model(params, x):
        A, B = np.exp(params)  # fit in log A, log B so both stay positive
        return A * x ** 4 + B * x ** 5

    res = least_squares(lambda p: np.log(model(p, h[win])) - np.log(err[win]), x0=np.log([5.0, 500.0]))
    A, B = np.exp(res.x)
    rms_ln = np.sqrt(np.mean(res.fun ** 2))
    rms_log10 = rms_ln / np.log(10)
    print(f"  two-term fit on {win.sum()} points: A = {A:.3f}, B = {B:.1f}, rms log residual {rms_ln:.4f} (ln) / {rms_log10:.4f} (log10)")
    check("Fig. A2", "A ~ 4.60", abs(A - 4.60) < 0.05, f"A = {A:.3f}")
    check("Fig. A2", "B ~ 772", abs(B - 772) < 8, f"B = {B:.1f}")
    check("Fig. A2", "rms log residual ~ 0.001", rms_ln < 0.002, f"{rms_ln:.4f} (ln), {rms_log10:.4f} (log10)")
    if not (abs(A - 4.60) < 0.05 and abs(B - 772) < 8 and rms_ln < 0.002):
        raise SystemExit("two-term fit differs from the expected A, B, rms: stop and report")

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(SINGLE, 3.6), sharex=True, gridspec_kw={"height_ratios": [1.5, 1]})
    hh = np.geomspace(h[win].min() / 1.5, h.max() * 1.2, 200)
    ax.loglog(hh, model(res.x, hh), color=STYLE["rk4"][1], lw=1.0, label=r"$A h^4 + B h^5$")
    ax.loglog(hh, A * hh ** 4, color="0.4", lw=0.6, ls=":", label=r"$A h^4$, $B h^5$ separately")
    ax.loglog(hh, B * hh ** 5, color="0.4", lw=0.6, ls=":")
    ax.loglog(h[win], err[win], "o", ms=3.5, color="k", label="RK4, measured (fitted)")
    out = ~win & ~floor
    ax.loglog(h[out], err[out], "o", ms=3.5, mfc="none", mec="k", mew=0.6, label="outside fit window")
    ax.loglog(h[floor], err[floor], "o", ms=3.5, mfc="none", mec="0.6", mew=0.6, label="round-off floor")
    ax.set_ylabel(r"$\max|\Delta E/E_0|$")
    light_log_grid(ax)
    ax.legend(loc="upper left", bbox_to_anchor=(0.08, 1.0), fontsize=6.5)
    panel_label(ax, "(a)")
    ax.text(0.97, 0.05, f"$A = {A:.2f}$, $B = {B:.0f}$", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)

    local = np.array([float(r["local_slope"]) for r in rows[:-1]])
    h_mid = np.sqrt(h[:-1] * h[1:])
    keep = ~floor[:-1] & ~floor[1:]  # slopes with a round-off endpoint are off the scale
    ax2.semilogx(h_mid[keep], local[keep], "o", ms=3.5, color="k", label="measured (adjacent pairs)")
    ax2.semilogx(hh, (4 * A * hh ** 4 + 5 * B * hh ** 5) / model(res.x, hh), color=STYLE["rk4"][1], lw=1.0, label="fitted model")
    for order in (4, 5):
        ax2.axhline(order, color="0.6", lw=0.5, ls=":")
    ax2.set_ylim(3.8, 5.2)
    ax2.set_xlabel(r"step size $h$")
    ax2.set_ylabel("local slope")
    ax2.legend(loc="upper left", bbox_to_anchor=(0.08, 1.0), fontsize=6.5)
    panel_label(ax2, "(b)")
    fig.tight_layout(h_pad=0.4)
    save(fig, "figA2_rk4_energy_scaling")
    return A, B, rms_ln


def slope_triangle(ax, x0, y0, order, width=1.6, colour="0.3"):
    """Right-angle triangle with log-log slope -order, anchored at (x0, y0), going right."""
    x1 = x0 * width
    y1 = y0 * width ** (-order)
    ax.plot([x0, x1, x0, x0], [y0, y1, y1, y0], color=colour, lw=0.5)
    ax.text(x0 / 1.04, np.sqrt(y0 * y1), f"$h^{order}$", ha="right", va="center", fontsize=6.5, color=colour)


def figA3():
    rows = read_csv("results/long_runs.csv")
    fig, ax = plt.subplots(figsize=(SINGLE, 2.6))
    for name in ("leapfrog", "rk4"):
        for scenario, marker, mfc in (("circular", "o", None), ("eccentric", "s", "none")):
            sel = sorted([r for r in rows if r["integrator"] == name and r["scenario"] == scenario], key=lambda r: int(r["steps_per_orbit"]))
            spo = np.array([int(r["steps_per_orbit"]) for r in sel])
            err = np.array([float(r["max_E_drift"]) for r in sel])
            label, colour, ls, lw = STYLE[name]
            ax.loglog(spo, err, color=colour, ls=ls, lw=lw, marker=marker, ms=3.5, mfc=mfc or colour, mec=colour, mew=0.7,
                      label=f"{label}, {scenario}")
            check("Fig. A3", f"{label} {scenario} has 200/400/800", spo.tolist() == [200, 400, 800], str(spo.tolist()))
    slope_triangle(ax, 330, 1.2e-6, 2)
    slope_triangle(ax, 330, 3e-9, 4)
    slope_triangle(ax, 330, 1.2e-10, 5)
    ax.set_xlim(170, 950)
    ax.set_xticks([200, 400, 800], ["200", "400", "800"])
    ax.minorticks_off()
    ax.set_xlabel("steps per orbit")
    ax.set_ylabel(r"$\max|\Delta E/E_0|$ over 300 orbits")
    light_log_grid(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=6.5)
    save(fig, "figA3_long_run_scaling")


def main():
    setup_style()
    print("Fig. 1"); fig1()
    print("Fig. 2"); fig2()
    print("Fig. 3"); picks = fig3()
    print("Fig. 4"); fig4("test"); fig4("holdout")
    print("Fig. 5"); fig5()
    print("Fig. A1"); figA1()
    print("Fig. A2"); figA2()
    print("Fig. A3"); figA3()
    print(f"\n{sum(c[2] for c in CHECKS)} of {len(CHECKS)} checks passed")
    for fig, name, ok, detail in CHECKS:
        if not ok:
            print(f"  FAIL {fig}: {name} -- {detail}")
    print("Fig. 3 clusters:", picks[:2], "RK4 slopes present:", picks[2])


if __name__ == "__main__":
    main()
