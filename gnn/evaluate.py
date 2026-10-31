"""GNN Stage B: evaluate trained models on the test set (per-decade errors, conservation, law recovery)."""

import os
import math

import csv
import sys

import matplotlib
matplotlib.use("Agg")  # draw straight to a file, no window
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import NullFormatter



import torch
from torch_geometric.data import Batch

from gnn.graph_data import file_to_graphs
from gnn.model import GravityGNN
from gnn.split import DATA_DIR, load_split
from gnn.train import F, relative_error, OUT_DIR

EPS = 2e-3  # softening length used to generate the data
COLOURS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}  # blue, orange, aqua: checked distinguishable, incl. colour-blind
MARKERS = {1: "o", 2: "s", 3: "^"}  # circle, square, triangle: a second cue besides colour
INK, INK_2, GRID, FLOOR_SHADE, PAPER = "#0b0b0b", "#52514e", "#e4e3df", "#efeeea", "#fcfcfb"


def load_model(path: str, which: str = "final_state") -> GravityGNN:
    """Rebuild a saved model. which = "final_state" or "best_state"."""
    ckpt = torch.load(path)  # the dictionary train() saved
    model = GravityGNN(ckpt["design"])  # empty model of the right design
    model.load_state_dict(ckpt[which])  # trained weights + standardisation stats
    model.eval()  # evaluation mode
    return model


def per_graph_sum(values: torch.Tensor, graph_of_node: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Add up per-body rows within each snapshot: (bodies, k) -> (snapshots, k)."""
    return torch.zeros(n_graphs, values.shape[1]).index_add_(0, graph_of_node, values)


def com_positions(batch) -> torch.Tensor:
    """Each body's position relative to its snapshot's centre of mass, rebuilt from the edge displacements."""
    send, recv = batch.edge_index
    first = batch.ptr[:-1][batch.batch]  # for every body: the index of the first body in its snapshot

    pos = torch.zeros(batch.num_nodes, 3)  # positions relative to the first body (the first body itself stays at 0)
    into_first = recv == first[recv]  # edges whose receiver is the first body of its snapshot
    pos[send[into_first]] = batch.edge_attr[into_first, :3]  # disp = r_sender - r_first

    m = batch.x  # (bodies, 1) masses
    com = per_graph_sum(m * pos, batch.batch, batch.num_graphs) / per_graph_sum(m, batch.batch, batch.num_graphs)
    return pos - com[batch.batch]  # shift every body by its own snapshot's centre of mass 


def conservation(x, acc, pos, graph_of_node, n_graphs):
    """R_p and R_L for every snapshot: net force and net torque, each relative to the sum of their sizes."""
    force = x * acc  # (bodies, 3) m a
    torque = x * torch.linalg.cross(pos, acc)  # (bodies, 3) m (r x a)

    R_p = (per_graph_sum(force, graph_of_node, n_graphs).norm(dim=-1)
           / per_graph_sum(force.norm(dim=-1, keepdim=True), graph_of_node, n_graphs)[:, 0])
    R_L = (per_graph_sum(torque, graph_of_node, n_graphs).norm(dim=-1)
           / per_graph_sum(torque.norm(dim=-1, keepdim=True), graph_of_node, n_graphs)[:, 0])
    return R_p, R_L


def collect(model, files: list) -> tuple:
    """Run the model on every snapshot of every file. Returns three dicts of 1-D tensors: per body, per snapshot, per far edge."""
    bodies = {k: [] for k in ["amag", "rel", "angle", "file", "two_body"]}
    snaps = {k: [] for k in ["Rp", "RL", "Rp_true", "RL_true", "file", "two_body"]}
    edges = {k: [] for k in ["log_msg", "log_true", "log_mj", "log_mi", "log_r", "kick_w", "file"]}

    for k, name in enumerate(files):
        batch = Batch.from_data_list(file_to_graphs(os.path.join(DATA_DIR, name)))  # every snapshot of this file
        send, recv = batch.edge_index
        two = name.startswith("two_body")  # True for two-body files

        # predictions: summed accelerations, and the individual messages before summing
        with torch.no_grad():
            pred = model(batch.x, batch.edge_index, batch.edge_attr)  # (bodies, 3)
            msg = model.message(batch.x[recv], batch.x[send], batch.edge_attr)  # (edges, 3)
        y = batch.y  # (bodies, 3) true accelerations

        # per body (E1, E2)
        bodies["amag"].append(y.norm(dim=-1))  # |a|, for binning
        bodies["rel"].append(relative_error(pred, y))  # |pred - y| / max(|y|, F)
        cross, dot = torch.linalg.cross(pred, y).norm(dim=-1), (pred * y).sum(dim=-1)
        bodies["angle"].append(torch.rad2deg(torch.atan2(cross, dot)))  # direction error in degrees
        bodies["file"].append(torch.full((batch.num_nodes,), k))  # which file each body came from (for the bootstrap)
        bodies["two_body"].append(torch.full((batch.num_nodes,), two))

        # per snapshot (E3): the model, and the truth as the float32 floor
        pos = com_positions(batch)
        R_p, R_L = conservation(batch.x, pred, pos, batch.batch, batch.num_graphs)
        R_p_true, R_L_true = conservation(batch.x, y, pos, batch.batch, batch.num_graphs)
        snaps["Rp"].append(R_p)
        snaps["RL"].append(R_L)
        snaps["Rp_true"].append(R_p_true)
        snaps["RL_true"].append(R_L_true)
        snaps["file"].append(torch.full((batch.num_graphs,), k))
        snaps["two_body"].append(torch.full((batch.num_graphs,), two))

        # per far edge (E4): pairs more than 10 softening lengths apart
        dist = batch.edge_attr[:, 3]
        far = dist > 10 * EPS  # True/False per edge
        edges["log_msg"].append(msg[far].norm(dim=-1).log())
        edges["log_mj"].append(batch.x[send[far], 0].log())  # sender mass
        edges["log_mi"].append(batch.x[recv[far], 0].log())  # receiver mass
        edges["log_r"].append(dist[far].log())
        m_j, r = batch.x[send[far], 0].double(), dist[far].double()
        edges["log_true"].append((m_j * r / (r ** 2 + EPS ** 2) ** 1.5).log().float())  # true pair |a| = G m_j r / (r² + ε²)^1.5, G = 1
        kick = y.norm(dim=-1) * batch.weight[batch.batch]  # (bodies,) |a| x the row's time weight, as in training
        edges["kick_w"].append((kick / kick.sum())[recv[far]])  # normalised per file; each edge gets its receiver's weight
        edges["file"].append(torch.full((int(far.sum()),), k))

    join = lambda d: {key: torch.cat(parts) for key, parts in d.items()}  # glue each list of per-file tensors into one
    return join(bodies), join(snaps), join(edges)


def per_decade(values, amag, file, n_files: int, n_boot: int = 1000, min_files: int = 3, seed: int = 0) -> list:
    """Median, p90 and a bootstrap 95% range for the median, in each decade of |a|. One dict per decade."""
    lo_exp = math.floor(math.log10(amag.min().item()))  # e.g. |a| min 0.0003 -> decade 10^-4
    hi_exp = math.ceil(math.log10(amag.max().item()))  # e.g. |a| max 80000 -> up to 10^5

    # the same 1000 random file picks for every decade (repeatable thanks to the seed)
    gen = torch.Generator().manual_seed(seed)
    picks = torch.randint(n_files, (n_boot, n_files), generator=gen)  # (1000, 36): each row = 36 file numbers, repeats allowed

    rows = []
    for e in range(lo_exp, hi_exp):
        in_bin = (amag >= 10.0 ** e) & (amag < 10.0 ** (e + 1))  # bodies in this decade
        v, f = values[in_bin], file[in_bin]
        row = {"lo": 10.0 ** e, "hi": 10.0 ** (e + 1), "centre": 10.0 ** (e + 0.5),  # centre = midpoint on a log axis
               "n": len(v), "n_files": len(f.unique())}

        if row["n_files"] < min_files:  # too few independent trajectories: don't trust or plot this bin
            rows.append({**row, "median": math.nan, "p90": math.nan, "med_lo": math.nan, "med_hi": math.nan})
            continue

        # bootstrap: rebuild the bin from 36 randomly picked files, 1000 times, and take each median
        by_file = [v[f == j] for j in range(n_files)]  # this bin's values, split by file
        boot = []
        for pick in picks:
            sample = torch.cat([by_file[j] for j in pick.tolist()])  # a picked-twice file appears twice
            if len(sample) > 0:  # skip the rare draw where none of the picked files has bodies in this bin
                boot.append(sample.median())
        boot = torch.stack(boot)

        rows.append({**row, "median": v.median().item(), "p90": v.quantile(0.9).item(),
                     "med_lo": boot.quantile(0.025).item(), "med_hi": boot.quantile(0.975).item()})
    return rows


def law_fit(edges: dict, n_files: int, n_boot: int = 1000, seed: int = 0,
            target: str = "log_msg", weighting: str = "decade", mask=None) -> dict:
    """Fit log|message| = alpha log m_j + beta log r + gamma log m_i + c on far edges, each decade of r given equal say.
    Returns {name: (value, low, high)} with a bootstrap 95% range over files.
    Law-fit checks: target="log_true" fits the true pair forces; weighting="none" or "kick"; mask = a subset of edges."""
    if mask is not None:
        edges = {k: v[mask] for k, v in edges.items()}  # keep only the chosen edges, e.g. one decade of r

    # the fit's inputs as columns: (edges, 4); the column of ones gives the constant c
    X = torch.stack([edges["log_mj"], edges["log_r"], edges["log_mi"], torch.ones_like(edges["log_r"])], dim=1).double()
    y = edges[target].double()  # (edges,) what we're fitting

    if weighting == "decade":  # equal say per decade of r: each edge's weight = 1 / (number of edges in its decade)
        decade = torch.floor(edges["log_r"] / math.log(10)).long()  # log_r is a natural log; dividing by ln 10 gives log10
        shifted = decade - decade.min()  # 0, 1, 2 ... so it can index a list
        w = 1.0 / torch.bincount(shifted)[shifted].double()
    elif weighting == "kick":  # each edge counts as much as its receiver did in training
        w = edges["kick_w"].double()
    else:  # "none": every edge counts once, so dense regions of r dominate
        w = torch.ones_like(y)

    def fit(rows):
        """Weighted least squares on the chosen edges; returns (alpha, beta, gamma)."""
        sw = w[rows].sqrt()  # weighting = scaling each row by the square root of its weight
        return torch.linalg.lstsq(X[rows] * sw[:, None], (y[rows] * sw)[:, None]).solution[:3, 0]

    best = fit(torch.arange(len(y)))  # the fit on all edges

    # bootstrap over files, as in per_decade
    rows_of = [torch.nonzero(edges["file"] == j)[:, 0] for j in range(n_files)]  # each file's edge numbers
    gen = torch.Generator().manual_seed(seed)
    picks = torch.randint(n_files, (n_boot, n_files), generator=gen)
    boot = torch.stack([fit(torch.cat([rows_of[j] for j in pick.tolist()])) for pick in picks])  # (1000, 3)
    lo, hi = boot.quantile(0.025, dim=0), boot.quantile(0.975, dim=0)

    return {name: (best[i].item(), lo[i].item(), hi[i].item()) for i, name in enumerate(["alpha", "beta", "gamma"])}


def evaluate_model(path: str, which: str, files: list) -> tuple:
    """Everything for one model: (summary row, per-decade results, truth conservation row)."""
    bodies, snaps, edges = collect(load_model(path, which), files)
    n_files = len(files)
    two = bodies["two_body"]

    # per-decade results for the figure: 2 groups x 2 metrics (E1, E2)
    decades = {}
    for group, mask in [("two-body", two), ("clusters", ~two)]:
        for metric in ["rel", "angle"]:
            decades[(group, metric)] = per_decade(bodies[metric][mask], bodies["amag"][mask], bodies["file"][mask], n_files)

    # summary numbers
    above = bodies["amag"] >= F  # table numbers use bodies above the loss floor only
    clusters = ~snaps["two_body"]  # R_L on clusters only (E3)
    p90s = [r["p90"] for group in ["two-body", "clusters"] for r in decades[(group, "rel")]
            if r["lo"] >= F and not math.isnan(r["p90"])]  # decades fully above F that have enough files

    row = {
        "median_rel": bodies["rel"][above].median().item(),
        "p90_rel_worst_decade": max(p90s),
        "angle_median": bodies["angle"][above].median().item(),
        "angle_p90": bodies["angle"][above].quantile(0.9).item(),
        "Rp_median": snaps["Rp"].median().item(),
        "Rp_p90": snaps["Rp"].quantile(0.9).item(),
        "Rp_two_body_median": snaps["Rp"][~clusters].median().item(),
        "Rp_clusters_median": snaps["Rp"][clusters].median().item(),
        "RL_median": snaps["RL"][clusters].median().item(),
        "RL_p90": snaps["RL"][clusters].quantile(0.9).item(),
    }
    for name, (value, lo, hi) in law_fit(edges, n_files).items():  # alpha, beta, gamma with their ranges (E4)
        row[name], row[name + "_lo"], row[name + "_hi"] = value, lo, hi

    # the same conservation numbers on the true accelerations: the float32 floor (E3)
    truth = {"Rp_median": snaps["Rp_true"].median().item(), "Rp_p90": snaps["Rp_true"].quantile(0.9).item(),
             "RL_median": snaps["RL_true"][clusters].median().item(), "RL_p90": snaps["RL_true"][clusters].quantile(0.9).item()}

    return row, decades, truth


def plot_per_decade(results: dict, path: str) -> None:
    """2 x 2 figure: rows = two-body / clusters, columns = relative / direction error. results = {design: decades}."""
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
                         "xtick.color": INK_2, "ytick.color": INK_2})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)  # 4 panels sharing one |a| axis

    for i, group in enumerate(["two-body", "clusters"]):
        for j, metric in enumerate(["rel", "angle"]):
            ax = axes[i, j]

            # background: the untrained region below F, and the predict-zero baseline
            ax.axvspan(1e-5, F, color=FLOOR_SHADE, linewidth=0, zorder=0)
            if metric == "rel":
                ax.axhline(1.0, color=INK_2, linewidth=1, linestyle=":", zorder=1)

            # one design at a time: band, then p90, then median on top
            for design, decades in results.items():
                rows = decades[(group, metric)]
                x = [r["centre"] for r in rows]
                c = COLOURS[design]
                ax.fill_between(x, [r["med_lo"] for r in rows], [r["med_hi"] for r in rows],
                                color=c, alpha=0.18, linewidth=0, zorder=2)  # 95% bootstrap band
                ax.plot(x, [r["p90"] for r in rows], color=c, linewidth=1, linestyle="--", zorder=3)
                ax.plot(x, [r["median"] for r in rows], color=c, linewidth=2,
                        marker=MARKERS[design], markersize=6, zorder=4)

            # axes, grid and labels
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(1e-4, 1e5)
            ax.grid(True, color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.yaxis.set_minor_formatter(NullFormatter())  # label only the powers of ten (the in-between ticks stay unlabelled)

            ax.set_title(f"{group}: " + ("relative error" if metric == "rel" else "direction error"),
                         loc="left", color=INK, fontsize=11)
            ax.set_ylabel("relative error  |â − a| / max(|a|, F)" if metric == "rel" else "angle between â and a (degrees)")
            if i == 1:
                ax.set_xlabel("true acceleration |a|")

    # one legend for the whole figure
    handles = [Line2D([], [], color=COLOURS[d], marker=MARKERS[d], linewidth=2,
                      label=f"Design {d} median (band: 95% bootstrap)") for d in results]
    handles += [Line2D([], [], color=INK_2, linestyle="--", linewidth=1, label="p90 (same colour per design)"),
                Line2D([], [], color=INK_2, linestyle=":", linewidth=1, label="predict-zero baseline (= 1)"),
                Patch(color=FLOOR_SHADE, label="|a| < F: below loss floor, untrained")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)

    fig.tight_layout(rect=(0, 0.08, 1, 1))  # leave the bottom 8% for the legend
    fig.savefig(path, dpi=200, facecolor=PAPER)
    plt.close(fig)


def main(designs: list, n_epochs: int = 200) -> None:
    """Evaluate final and best-validation weights of each design; save the summary CSV and the figure."""
    files = load_split()["test"]
    rows, finals = [], {}

    for design in designs:
        path = os.path.join(OUT_DIR, f"design{design}_kick_{n_epochs}ep.pt")  # e.g. design2_kick_200ep.pt
        for which in ["final_state", "best_state"]:
            row, decades, truth = evaluate_model(path, which, files)  # about 1.5 minutes each
            rows.append({"design": design, "weights": which.split("_")[0], **row})  # "final" or "best"
            if which == "final_state":
                finals[design] = decades  # the figure uses final weights only
            print(f"design {design} {which.split('_')[0]:5s}  median rel {row['median_rel']:.3g}  "
                  f"angle {row['angle_median']:.3g} deg  R_p {row['Rp_median']:.2g}  "
                  f"alpha {row['alpha']:.3f}  beta {row['beta']:.3f}", flush=True)

    rows.append({"design": "truth", "weights": "-", **truth})  # same for every model: the float32 floor

    # summary table
    csv_path = os.path.join("results", "stage_b", "summary.csv")  # the table stays in the project (written once, no sync clash)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), restval="")  # columns from the first row; blanks where missing
        writer.writeheader()
        writer.writerows(rows)

    # figure
    os.makedirs("figures", exist_ok=True)
    plot_per_decade(finals, os.path.join("figures", "stage_b_per_decade.png"))
    print("saved", csv_path, "and figures/stage_b_per_decade.png")


def seed_spread(n_epochs: int = 200) -> None:
    """Final weights of every seed of designs 2 and 3, plus time-weighted design 3; one CSV row each."""
    files = load_split()["test"]
    runs = [(d, "kick", s) for d in [2, 3] for s in [0, 1, 2]] + [(3, "time", 0)]  # (design, scheme, seed)
    rows = []

    for design, scheme, seed in runs:
        tag = f"_seed{seed}" if seed != 0 else ""  # same naming as train.py
        path = os.path.join(OUT_DIR, f"design{design}_{scheme}_{n_epochs}ep{tag}.pt")
        row, _, _ = evaluate_model(path, "final_state", files)  # _ = per-decade and truth results, not needed here
        rows.append({"design": design, "scheme": scheme, "seed": seed, **row})
        print(f"design {design} {scheme} seed {seed}  median rel {row['median_rel']:.3g}  "
              f"p90 worst {row['p90_rel_worst_decade']:.3g}  R_p {row['Rp_median']:.2g}  "
              f"alpha {row['alpha']:.3f}  beta {row['beta']:.3f}", flush=True)

    csv_path = os.path.join("results", "stage_b", "seeds.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("saved", csv_path)


def law_checks(designs: list = [2, 3], n_epochs: int = 200) -> None:
    """The three law-fit checks (Project, after Stage B): truth control, beta per decade of r, reweighted refits."""
    files = load_split()["test"]
    n_files = len(files)
    rows = []

    def record(design, check, r_range, fit):
        rows.append({"design": design, "check": check, "r_range": r_range,
                     **{f"{name}{end}": v for name, vals in fit.items() for end, v in zip(["", "_lo", "_hi"], vals)}})
        a, b, g = fit["alpha"], fit["beta"], fit["gamma"]
        print(f"{design!s:6s} {check:22s} {r_range:14s}  alpha {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}]  "
              f"beta {b[0]:.3f} [{b[1]:.3f}, {b[2]:.3f}]  gamma {g[0]:.3f} [{g[1]:.3f}, {g[2]:.3f}]", flush=True)

    for design in designs:
        path = os.path.join(OUT_DIR, f"design{design}_kick_{n_epochs}ep.pt")
        _, _, edges = collect(load_model(path), files)  # _ = per-body and per-snapshot results, not needed here

        # check 1: the same fit on the true pair forces (same edges, same weighting); only needed once
        if design == designs[0]:
            record("truth", "control", "all r", law_fit(edges, n_files, target="log_true"))

        # check 2: beta in each decade of r, model and truth side by side
        record(design, "as in summary", "all r", law_fit(edges, n_files))
        decade = torch.floor(edges["log_r"] / math.log(10)).long()
        for e in decade.unique().tolist():
            mask = decade == e
            if len(edges["file"][mask].unique()) < 3:  # same rule as the figure: fewer than 3 files is not trusted
                continue
            r_range = f"1e{e} to 1e{e + 1}"
            record(design, "per decade", r_range, law_fit(edges, n_files, mask=mask))
            if design == designs[0]:
                record("truth", "per decade", r_range, law_fit(edges, n_files, target="log_true", mask=mask))

        # check 3: refit weighted by data density (every edge once) and by kick weight
        record(design, "density-weighted", "all r", law_fit(edges, n_files, weighting="none"))
        record(design, "kick-weighted", "all r", law_fit(edges, n_files, weighting="kick"))

    csv_path = os.path.join("results", "stage_b", "law_checks.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("saved", csv_path)


def weighting_compare(n_epochs: int = 200) -> None:
    """Time- vs kick-weighted Design 3: per-decade relative error (all kick seeds) and per-decade beta (Project, after Stage B)."""
    files = load_split()["test"]
    n_files = len(files)
    runs = [("kick", s) for s in [0, 1, 2]] + [("time", 0)]  # (scheme, seed)
    rows = []

    for scheme, seed in runs:
        tag = f"_seed{seed}" if seed != 0 else ""  # same naming as train.py
        label = f"{scheme} seed {seed}"
        bodies, _, edges = collect(load_model(os.path.join(OUT_DIR, f"design3_{scheme}_{n_epochs}ep{tag}.pt")), files)

        # (1) per-decade relative error, two-body and clusters
        two = bodies["two_body"]
        for group, mask in [("two-body", two), ("clusters", ~two)]:
            for r in per_decade(bodies["rel"][mask], bodies["amag"][mask], bodies["file"][mask], n_files):
                if r["lo"] >= F and not math.isnan(r["median"]):  # decades fully above F with enough files
                    rows.append({"run": label, "quantity": f"rel {group}", "range": f"{r['lo']:.0e} to {r['hi']:.0e}",
                                 "value": r["median"], "lo": r["med_lo"], "hi": r["med_hi"], "p90": r["p90"]})

        # (2) per-decade beta, time-weighted model only
        if scheme == "time":
            decade = torch.floor(edges["log_r"] / math.log(10)).long()
            for e in decade.unique().tolist():
                mask = decade == e
                if len(edges["file"][mask].unique()) < 3:  # same rule as law_checks
                    continue
                beta = law_fit(edges, n_files, mask=mask)["beta"]
                rows.append({"run": label, "quantity": "beta", "range": f"1e{e} to 1e{e + 1}",
                             "value": beta[0], "lo": beta[1], "hi": beta[2], "p90": ""})
        print("done", label, flush=True)

    # print side by side: one line per decade, one column per run
    for quantity in ["rel two-body", "rel clusters", "beta"]:
        print(quantity)
        for rng in dict.fromkeys(r["range"] for r in rows if r["quantity"] == quantity):  # decades in order, no repeats
            cells = [f"{r['run']}: {r['value']:.2e} [{r['lo']:.2e}, {r['hi']:.2e}]" for r in rows
                     if r["quantity"] == quantity and r["range"] == rng]
            print(f"  {rng}   " + "   ".join(cells))

    csv_path = os.path.join("results", "stage_b", "weighting_compare.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("saved", csv_path)


def doubling_test(path_short: str, path_long: str, files: list) -> None:
    """Decision 8: is the long run clearly better, decade by decade? Prints the comparison and the verdict."""
    # per-decade median relative error for both models, two-body and clusters
    results = {}
    for label, path in [("short", path_short), ("long", path_long)]:
        bodies, _, _ = collect(load_model(path), files)  # _ = the per-snapshot and per-edge results, not needed here
        two = bodies["two_body"]
        results[label] = {group: per_decade(bodies["rel"][mask], bodies["amag"][mask], bodies["file"][mask], len(files))
                          for group, mask in [("two-body", two), ("clusters", ~two)]}

    # compare decade by decade
    n_better, n_total = 0, 0
    for group in ["two-body", "clusters"]:
        print(group)
        for s, l in zip(results["short"][group], results["long"][group]):  # same test data, so the same decades in the same order
            if s["lo"] < F or math.isnan(s["median"]) or math.isnan(l["median"]):
                continue  # only decades fully above F with enough files
            ratio = s["median"] / l["median"]  # above 1 = the long run is more accurate
            clearly_better = ratio > 1.25 and l["med_hi"] < s["med_lo"]  # at least 20% lower AND bands don't overlap
            n_better += clearly_better
            n_total += 1
            print(f"  {s['lo']:.0e} to {s['hi']:.0e}  short {s['median']:.2e} [{s['med_lo']:.2e}, {s['med_hi']:.2e}]  "
                  f"long {l['median']:.2e} [{l['med_lo']:.2e}, {l['med_hi']:.2e}]  ratio {ratio:.2f}"
                  + ("  CLEARLY BETTER" if clearly_better else ""))

    # the verdict, by the rule fixed in the labbook before the result
    if n_better == 0:
        verdict = "no decade clearly better: 100 epochs is enough"
    elif n_better > n_total / 2:
        verdict = "most decades clearly better: rerun all three designs at 200 epochs"
    else:
        verdict = "some decades clearly better: report it and decide with the Project"
    print(f"{n_better} of {n_total} decades clearly better -> {verdict}")


if __name__ == "__main__":
    if sys.argv[1:] == ["doubling"]:  # python -m gnn.evaluate doubling
        doubling_test(os.path.join(OUT_DIR, "design3_kick_100ep.pt"),
                      os.path.join(OUT_DIR, "design3_kick_200ep.pt"), load_split()["test"])
    elif sys.argv[1:] == ["seeds"]:  # python -m gnn.evaluate seeds
        seed_spread()
    elif sys.argv[1:] == ["law"]:  # python -m gnn.evaluate law
        law_checks()
    elif sys.argv[1:] == ["weighting"]:  # python -m gnn.evaluate weighting
        weighting_compare()
    else:  # python -m gnn.evaluate [designs]
        designs = [int(d) for d in sys.argv[1:]] or [1, 2, 3]
        main(designs)

