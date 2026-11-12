"""GNN Stage C, G2: chaos-baseline false-vote test, model slope ratios and the vote, horizons. Rules: stage-c-decisions.md."""

import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")  # files only, no window
import matplotlib.pyplot as plt
import numpy as np

from gnn.metrics import cluster_divergence, fit_slope, horizon, load_rollout, slope_window, two_body_divergence
from gnn.rollout import read_run, rollout_files

RESULTS = "results/stage_c"
SEEDS = ["baseline", "baseline2", "baseline3"]
MODELS = {"test": ["design3_kick_200ep", "design3_time_200ep", "design2_kick_200ep"],
          "holdout": ["design3_kick_200ep", "design3_time_200ep"]}  # D2 holdout cut 2026-11-12
WINDOWS = {"main": 1e-4, "1e-3": 1e-3}  # lower edge of the fit; the top is always 0.1
VOTE = 1.5  # model error if GNN slope >= 1.5 x mean baseline slope
FALSE_VOTE_LIMIT = 0.25  # gate passes if no pretend model gets more than a quarter of the voting clusters
COLOURS = {"design3_kick_200ep": "#2a78d6", "design3_time_200ep": "#eb6834", "design2_kick_200ep": "#1baf7a"}


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def slope_row(times, div, lo):
    """Slope, R² and window length for one divergence curve, or NaNs if it never reaches 0.1."""
    window = slope_window(div, lo=lo)
    if window is None:
        return np.nan, np.nan, 0
    slope, r2 = fit_slope(times, div, window)
    return slope, r2, window[1] - window[0] + 1


def verdict(yes, n):
    if yes > n / 2:
        return "model error"
    return "model error in some clusters" if yes >= 3 else "within noise"


def main(split):
    os.makedirs(RESULTS, exist_ok=True)
    names = rollout_files(split)
    clusters = [n for n in names if n.startswith("cluster")]
    models = MODELS[split]

    curves, fit, horizon_rows, fit_rows = {}, {}, [], []  # curves[(job, name)] = (times, div); fit[(job, window, name)] = (slope, r2, n)
    for name in names:
        run = read_run(name)
        ref, _, times = load_rollout("reference", name)
        is_cluster = name.startswith("cluster")
        for job in (SEEDS if is_cluster else []) + models + ["rk4"]:
            pos, _, _ = load_rollout(job, name)
            div = cluster_divergence(run["masses"], pos, ref) if is_cluster else two_body_divergence(run["masses"], pos, ref)
            curves[(job, name)] = (times, div)
            horizon_rows.append({"file": name, "job": job, "horizon": horizon(times, div), "peak": div.max()})
            if is_cluster:
                for wname, lo in WINDOWS.items():
                    fit[(job, wname, name)] = slope_row(times, div, lo)
                    fit_rows.append({"file": name, "job": job, "window": wname, "slope": fit[(job, wname, name)][0],
                                     "r2": fit[(job, wname, name)][1], "n_points": fit[(job, wname, name)][2]})
    write_csv(f"{RESULTS}/g2_horizons_{split}.csv", horizon_rows)
    write_csv(f"{RESULTS}/g2_fits_{split}.csv", fit_rows)

    voting = [n for n in clusters if all(np.isfinite(fit[(s, "main", n)][0]) for s in SEEDS)]  # all 3 seeds reach 0.1
    print(f"{split}: {len(voting)} voting clusters of {len(clusters)}; non-voting: {[n for n in clusters if n not in voting]}")

    def baseline_table(wname):  # (voting clusters, 3 seeds) of slopes
        return np.array([[fit[(s, wname, n)][0] for s in SEEDS] for n in voting])

    false_rows, gate = [], True
    for wname in WINDOWS:
        table = baseline_table(wname)
        noise = table.max(axis=1) / table.min(axis=1)
        votes_15 = []
        for i, seed in enumerate(SEEDS):
            ratio = table[:, i] / np.delete(table, i, axis=1).mean(axis=1)  # pretend model vs mean of the other two
            for thr in (1.25, 1.5, 2.0, 2.5):
                false_rows.append({"window": wname, "pretend_model": seed, "threshold": thr,
                                   "false_votes": int((ratio >= thr).sum()), "n_voting": len(voting)})
            votes_15.append(int((ratio >= VOTE).sum()))
        shares = {thr: sum(r["false_votes"] for r in false_rows if r["window"] == wname and r["threshold"] == thr) / (3 * len(voting))
                  for thr in (1.25, 1.5, 2.0, 2.5)}
        gate &= max(votes_15) <= FALSE_VOTE_LIMIT * len(voting)
        print(f"  {wname}: max/min noise mean {noise.mean():.2f} median {np.median(noise):.2f} max {noise.max():.2f}; "
              f"false votes at 1.5x {votes_15} of {len(voting)}; share by threshold "
              + ", ".join(f"{t}: {s:.1%}" for t, s in shares.items()))
    write_csv(f"{RESULTS}/g2_false_votes_{split}.csv", false_rows)
    print(f"  gate (at most {FALSE_VOTE_LIMIT:.0%} false votes per pretend model, both windows): {'PASS' if gate else 'FAIL'}")
    if not gate:
        print("  gate failed: stopping before model slopes")
        return

    ratio_rows, vote_rows = [], []
    for model in models:
        for wname in WINDOWS:
            base_mean = baseline_table(wname).mean(axis=1)
            for name, b in zip(voting, base_mean):
                slope, r2, npts = fit[(model, wname, name)]
                ratio_rows.append({"model": model, "window": wname, "file": name, "gnn_slope": slope, "gnn_r2": r2,
                                   "gnn_points": npts, "baseline_mean": b, "ratio": slope / b,
                                   "votes_model_error": bool(slope / b >= VOTE)})  # NaN (never reaches 0.1) is not a yes
            r = np.array([row["ratio"] for row in ratio_rows if row["model"] == model and row["window"] == wname])
            yes = int((r >= VOTE).sum())
            vote_rows.append({"model": model, "window": wname, "yes": yes, "n_voting": len(r), "median_ratio": np.nanmedian(r),
                              "never_reached": int(np.isnan(r).sum()), "verdict": verdict(yes, len(r))})
    write_csv(f"{RESULTS}/g2_ratios_{split}.csv", ratio_rows)
    write_csv(f"{RESULTS}/g2_vote_{split}.csv", vote_rows)
    for row in vote_rows:
        print(f"  {row['model']:20s} {row['window']:5s} yes {row['yes']:2d}/{row['n_voting']}  median ratio {row['median_ratio']:.2f}  "
              f"never reached {row['never_reached']}  -> {row['verdict']}")

    low = [r for r in fit_rows if r["file"] in voting and r["r2"] < 0.9]
    counts = {}
    for r in low:
        counts[(r["job"], r["window"])] = counts.get((r["job"], r["window"]), 0) + 1
    print(f"  fits with R2 < 0.9 on voting clusters (all listed in g2_fits_{split}.csv):", counts or "none")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, wname in zip(axes, WINDOWS):
        for k, model in enumerate(models):
            r = [row["ratio"] for row in ratio_rows if row["model"] == model and row["window"] == wname]
            jitter = np.random.default_rng(k).uniform(-0.15, 0.15, len(r))  # spread the dots sideways so they don't overlap
            ax.scatter(k + jitter, r, s=36, color=COLOURS[model], edgecolor="white", linewidth=1, zorder=3)
        ax.axhline(1.0, color="0.5", lw=1, ls="--")
        ax.axhline(VOTE, color="0.2", lw=1)
        ax.set_xticks(range(len(models)), [m.replace("_200ep", "").replace("design", "D") for m in models])
        ax.set_title(f"{split}, window {'1e-4' if wname == 'main' else wname} to 0.1 (lines at 1 and 1.5)", fontsize=10)
        ax.set_yscale("log")
        ax.grid(axis="y", color="0.9", lw=0.5)
    axes[0].set_ylabel("GNN slope / mean baseline slope")
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/g2_ratios_{split}.png", dpi=150)

    rows = -(-len(clusters) // 5)  # ceiling division: enough rows of 5 for every cluster
    fig, axes = plt.subplots(rows, 5, figsize=(16, 2.75 * rows), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, clusters):
        for seed in SEEDS:
            ax.plot(*curves[(seed, name)], color="0.6", lw=1)
        for model in models:
            ax.plot(*curves[(model, name)], color=COLOURS[model], lw=1.2)
        ax.set_yscale("log")
        ax.set_ylim(1e-8, 10)
        ax.axhline(0.1, color="0.2", lw=0.6)
        ax.set_title(name.replace(".h5", "") + ("" if name in voting else " (non-voting)"), fontsize=9)
    for model in models:
        axes.flat[0].plot([], [], color=COLOURS[model], label=model)
    axes.flat[0].plot([], [], color="0.6", label="chaos baselines")
    axes.flat[0].legend(fontsize=7)
    fig.supxlabel("time")
    fig.supylabel("δ(t)")
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/g2_curves_{split}.png", dpi=120)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
