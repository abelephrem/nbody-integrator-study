"""GNN Stage C: interactive (rotatable, playable) HTML versions of the two rollout animations."""

import sys

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gnn.animate_rollouts import BODY_COLOURS, CHOSEN, MODEL, TRAIL, outside_to_nan
from gnn.metrics import about_com, cluster_divergence, horizon, load_rollout, two_body_divergence
from gnn.rollout import read_run

PLAYBACK = {"two_body": (1200, 15), "cluster": (150, 20)}  # (frames, ms per frame): ~47 frames per orbit for two-body; the cluster plays slowed down
GREY = "#8c8c8c"

# rotating one 3D panel turns the other the same way (plotly keeps each scene's camera separate)
SYNC_CAMERAS = """
var gd = document.getElementById('{plot_id}');
var busy = false;
gd.on('plotly_relayout', function (e) {
  if (busy) return;
  var pairs = [['scene.camera', 'scene2.camera'], ['scene2.camera', 'scene.camera']];
  for (var i = 0; i < pairs.length; i++) {
    if (e[pairs[i][0]] && gd.layout.scene2) {
      busy = true;
      var update = {}; update[pairs[i][1]] = e[pairs[i][0]];
      Plotly.relayout(gd, update).then(function () { busy = false; });
      return;
    }
  }
});
"""


def clean(a):
    """Rounded list with NaN as None, so plotly leaves a gap instead of failing."""
    return [None if not np.isfinite(v) else round(float(v), 5) for v in np.ravel(a)]


def xyz(points):
    """dict(x=..., y=..., z=...) from points (..., 3)."""
    points = np.atleast_2d(points)
    return {"x": clean(points[:, 0]), "y": clean(points[:, 1]), "z": clean(points[:, 2])}


def build(name, t_end, out_path):
    masses = read_run(name)["masses"]
    ref, _, times = load_rollout("reference", name)
    gnn, _, _ = load_rollout(MODEL, name)
    if t_end is not None:
        keep = times <= t_end + 1e-9
        ref, gnn, times = ref[keep], gnn[keep], times[keep]
    ref, gnn = about_com(masses, ref), about_com(masses, gnn)
    is_cluster = name.startswith("cluster")
    div = cluster_divergence(masses, gnn, ref) if is_cluster else two_body_divergence(masses, gnn, ref)
    t_hor = horizon(times, div)
    period = None if is_cluster else 2 * np.pi * np.sqrt((masses[0] * masses[1]) ** 3 / masses.sum())
    side_by_side = is_cluster  # as in the videos: clusters in two panels, two-body overlaid

    both = np.concatenate([ref.reshape(-1, 3), gnn.reshape(-1, 3)])
    lo, hi = both.min(axis=0), both.max(axis=0)  # every position shown, so no body leaves the box
    centre, half = (lo + hi) / 2, (hi - lo).max() / 2 * 1.1
    ref_d, gnn_d = outside_to_nan(ref, centre, half), outside_to_nan(gnn, centre, half)

    frames, ms = PLAYBACK["cluster" if is_cluster else "two_body"]
    stride = max(1, -(-len(times) // frames))  # ceiling division: at most `frames` frames
    ks = np.arange(0, len(times), stride)  # checkpoint shown in each frame
    trail = TRAIL * max(1, len(times) // 600)  # same trail length in time as the videos
    n = len(masses)
    sizes = 4 + 6 * masses / masses.max()
    x_scale = period or 1.0

    if side_by_side:
        fig = make_subplots(rows=2, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}], [{"type": "xy", "colspan": 2}, None]],
                            row_heights=[0.66, 0.34], vertical_spacing=0.08,
                            subplot_titles=("true force (reference)", f"{MODEL}  (grey: where the reference body is; line = its error)", ""))
    else:
        fig = make_subplots(rows=2, cols=1, specs=[[{"type": "scene"}], [{"type": "xy"}]], row_heights=[0.66, 0.34],
                            vertical_spacing=0.08, subplot_titles=(f"solid = true force, dashed / open = {MODEL}", ""))
    gnn_scene = (1, 2) if side_by_side else (1, 1)
    graph = (2, 1)

    def body_traces(k):
        """Every animated trace at checkpoint k, always in the same order."""
        start = max(0, k - trail)
        rows = np.arange(k, start - 1, -max(1, trail // 60))[::-1]  # about 60 trail points, always ending at k
        out = []
        for i in range(n):
            m = f"m = {masses[i]:.3g}"
            out.append(go.Scatter3d(**xyz(ref_d[rows, i]), mode="lines", line=dict(color=BODY_COLOURS[i], width=4),
                                    name=m, legendgroup=m, showlegend=False, hoverinfo="skip"))
            out.append(go.Scatter3d(**xyz(ref_d[k, i]), mode="markers", marker=dict(color=BODY_COLOURS[i], size=sizes[i] * 1.2),
                                    name=m, legendgroup=m, showlegend=True, hovertemplate=f"reference, {m}<extra></extra>"))
            dash = "solid" if side_by_side else "dash"
            symbol = "circle" if side_by_side else "circle-open"
            out.append(go.Scatter3d(**xyz(gnn_d[rows, i]), mode="lines", line=dict(color=BODY_COLOURS[i], width=4, dash=dash),
                                    legendgroup=m, showlegend=False, hoverinfo="skip"))
            out.append(go.Scatter3d(**xyz(gnn_d[k, i]), mode="markers", marker=dict(color=BODY_COLOURS[i], size=sizes[i] * 1.2, symbol=symbol),
                                    legendgroup=m, showlegend=False, hovertemplate=f"{MODEL}, {m}<extra></extra>"))
            if side_by_side:
                out.append(go.Scatter3d(**xyz(ref_d[k, i]), mode="markers", marker=dict(color=GREY, size=sizes[i] * 1.2, symbol="circle-open"),
                                        legendgroup=m, showlegend=False, hoverinfo="skip"))
                out.append(go.Scatter3d(**xyz(np.stack([gnn_d[k, i], ref_d[k, i]])), mode="lines", line=dict(color=GREY, width=3),
                                        legendgroup=m, showlegend=False, hoverinfo="skip"))
        return out

    def scenes_of_traces():
        """Which scene each trace in body_traces belongs to."""
        per_body = [(1, 1), (1, 1), gnn_scene, gnn_scene] + ([gnn_scene, gnn_scene] if side_by_side else [])
        return per_body * n

    idx = ks  # divergence points shown: one per frame, so the growing curve stays light
    def graph_traces(j):
        k = ks[j]
        upto = idx[:j + 1]
        upto = upto[upto > 0]  # log axis: skip t = 0 where div = 0
        return [go.Scatter(x=clean(times[upto] / x_scale), y=clean(div[upto]), mode="lines", line=dict(color="#2a78d6", width=2),
                           showlegend=False, hovertemplate="t %{x:.2f}, %{y:.2e}<extra></extra>"),
                go.Scatter(x=clean([times[k] / x_scale]), y=clean([max(div[k], 1e-16)]), mode="markers",
                           marker=dict(color="#2a78d6", size=8), showlegend=False, hoverinfo="skip")]

    for trace, (r, c) in zip(body_traces(ks[0]), scenes_of_traces()):
        fig.add_trace(trace, row=r, col=c)
    fig.add_trace(go.Scatter(x=clean(times[1:] / x_scale), y=clean(div[1:]), mode="lines", line=dict(color="#d0d0d0", width=1),
                             showlegend=False, hoverinfo="skip"), row=2, col=1)  # whole curve, faint, never animated
    for trace in graph_traces(0):
        fig.add_trace(trace, row=2, col=1)
    n_body = len(scenes_of_traces())
    animated = list(range(n_body)) + [n_body + 1, n_body + 2]  # skip the faint static curve

    def label(k):
        return f"t = {times[k]:.2f}" + (f" ({times[k] / period:.1f} orbits)" if period else "")

    fig.frames = [go.Frame(data=body_traces(k) + graph_traces(j), traces=animated, name=str(j),
                           layout=dict(title_text=f"{name[:-3]}, centre-of-mass frame — {label(k)}"))
                  for j, k in enumerate(ks)]

    axis = lambda c: dict(range=[c - half, c + half], autorange=False)
    scene = dict(xaxis=axis(centre[0]), yaxis=axis(centre[1]), zaxis=axis(centre[2]), aspectmode="cube", uirevision="keep",
                 camera=dict(eye=dict(x=1.4, y=-1.4, z=0.8)))
    fig.update_layout(scene=scene, **({"scene2": scene} if side_by_side else {}))
    t_max = times[-1] / x_scale  # reference lines as static traces (add_hline fails on figures with 3D panels)
    fig.add_trace(go.Scatter(x=[0, t_max], y=[0.1, 0.1], mode="lines", line=dict(color="#333", width=1),
                             showlegend=False, hoverinfo="skip"), row=2, col=1)
    if np.isfinite(t_hor):
        y_lo = max(div[1:][div[1:] > 0].min(), 1e-12)
        fig.add_trace(go.Scatter(x=[t_hor / x_scale] * 2, y=[y_lo, div.max()], mode="lines+text", line=dict(color="#333", width=1, dash="dot"),
                                 text=[f"  horizon t = {t_hor:.2f}", ""], textposition="middle right", showlegend=False,
                                 hoverinfo="skip"), row=2, col=1)
    fig.update_yaxes(type="log", title_text="δ(t), RMS body separation" if is_cluster else "|Δr_rel| / a", row=2, col=1)
    fig.update_xaxes(title_text="orbits" if period else "time", row=2, col=1)

    play = dict(frame=dict(duration=ms, redraw=True), fromcurrent=True, transition=dict(duration=0), mode="immediate")
    fig.update_layout(
        title_text=f"{name[:-3]}, centre-of-mass frame — {label(ks[0])}", height=900, template="plotly_white",
        legend=dict(title="body mass", x=0.0, y=1.0),
        updatemenus=[dict(type="buttons", direction="left", x=0.0, y=-0.04, xanchor="left", yanchor="top", showactive=False,
                          buttons=[dict(label="▶ play", method="animate", args=[None, play]),
                                   dict(label="❚❚ pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])])],
        sliders=[dict(active=0, x=0.12, y=-0.02, len=0.88, currentvalue=dict(visible=False),
                      steps=[dict(label=f"{times[k]:.2f}", method="animate",
                                  args=[[str(j)], dict(frame=dict(duration=0, redraw=True), mode="immediate", transition=dict(duration=0))])
                             for j, k in enumerate(ks)])])
    fig.write_html(out_path, include_plotlyjs="cdn", post_script=SYNC_CAMERAS if side_by_side else None, auto_play=False)
    print(out_path, f"{len(ks)} frames")


if __name__ == "__main__":
    for name, t_end in CHOSEN.items():
        if len(sys.argv) > 1 and sys.argv[1] not in name:  # optional filter, e.g. "two_body"
            continue
        build(name, t_end, f"results/stage_c/interactive_{name[:-3]}.html")
