"""Step 2: acceleration distribution across the CURRENT 360 trajectories.

Read-only. Sets the event trigger threshold from the measured distribution rather than
inheriting resample_events' 3*eps, and gives the before/after baseline for the recording
fix.

Outputs:
    results/accel_histogram.csv
    figures/accel_histogram.png
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

DATA_DIR = "C:/nbody_data"
EPS = 0.002
G = 1.0


def plummer_accel(m_partner, r, eps=EPS):
    """|a| on a body from a single partner at separation r, Plummer-softened."""
    return G * m_partner * r / (r**2 + eps**2)**1.5


def collect(pattern):
    mags, masses_seen = [], []
    for p in sorted(glob.glob(os.path.join(DATA_DIR, pattern))):
        with h5py.File(p, "r") as f:
            a = f["accelerations"][:]
            m = f["masses"][:]
        mags.append(np.linalg.norm(a, axis=2).ravel())
        masses_seen.append(m)
    return np.concatenate(mags), masses_seen


def main():
    mags, masses_seen = collect("cluster_*.h5")
    mags = mags[mags > 0]
    print(f"{len(mags)} per-particle acceleration samples from cluster runs")
    print(f"  min {mags.min():.3e}  median {np.median(mags):.3e}  max {mags.max():.3e}")
    for q in (50, 90, 99, 99.9, 99.99, 100):
        print(f"  p{q:<6} {np.percentile(mags, q):.4e}")

    # Reference accelerations for the pair masses actually present.
    m_all = np.concatenate(masses_seen)
    m_light, m_heavy = np.median(m_all), np.max(m_all)
    print(f"\nmasses in the set: min {m_all.min():.4f} median {m_light:.4f} "
          f"max {m_heavy:.4f}")
    print("reference |a| from ONE partner at separation r (Plummer, eps=0.002):")
    rows = []
    for label, r in (("3 eps", 3 * EPS), ("2 eps", 2 * EPS), ("1 eps", EPS),
                     ("eps/sqrt2 (peak)", EPS / np.sqrt(2)), ("0.5 eps", 0.5 * EPS),
                     ("0.2 eps", 0.2 * EPS)):
        a_med, a_hvy = plummer_accel(m_light, r), plummer_accel(m_heavy, r)
        pct_med = float((mags < a_med).mean() * 100)
        rows.append({"r_label": label, "r_over_eps": r / EPS,
                     "a_median_partner": a_med, "a_heaviest_partner": a_hvy,
                     "pct_samples_below_a_median": pct_med})
        print(f"  r={label:<18} median partner {a_med:.4e} (p{pct_med:6.3f}) "
              f"| heaviest partner {a_hvy:.4e}")

    # Where does the measured tail actually stop?
    print(f"\nmeasured max |a| = {mags.max():.4e}; "
          f"peak physically available (heaviest partner at eps/sqrt2) = "
          f"{plummer_accel(m_heavy, EPS/np.sqrt(2)):.4e}")
    print(f"  ratio = {plummer_accel(m_heavy, EPS/np.sqrt(2)) / mags.max():.1f}x "
          f"above the largest recorded value")

    os.makedirs("results", exist_ok=True)
    with open("results/accel_histogram.csv", "w", newline="") as f:
        f.write("r_label,r_over_eps,a_median_partner,a_heaviest_partner,"
                "pct_samples_below_a_median\n")
        for r in rows:
            f.write(f"{r['r_label']},{r['r_over_eps']:.4f},{r['a_median_partner']:.6e},"
                    f"{r['a_heaviest_partner']:.6e},{r['pct_samples_below_a_median']:.4f}\n")
        f.write("\npercentile,value\n")
        for q in (1, 10, 25, 50, 75, 90, 99, 99.9, 99.99, 100):
            f.write(f"{q},{np.percentile(mags, q):.6e}\n")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.geomspace(mags.min(), max(mags.max(), plummer_accel(m_heavy, EPS/np.sqrt(2))) * 1.1, 90)
    ax.hist(mags, bins=bins, color="#2980b9", alpha=0.85)
    for label, r, c in (("3ε (current trigger)", 3 * EPS, "#c0392b"),
                        ("1ε", EPS, "#e67e22"),
                        ("ε/√2 (peak available)", EPS / np.sqrt(2), "#8e44ad")):
        ax.axvline(plummer_accel(m_light, r), color=c, ls="--", lw=1.5,
                   label=f"{label}, median partner")
    ax.axvline(mags.max(), color="k", ls=":", lw=1.5, label="max recorded")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"per-particle $|a|$")
    ax.set_ylabel("count")
    ax.set_title("Stored acceleration distribution, 260 cluster runs (current dataset)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs("figures", exist_ok=True)
    fig.savefig("figures/accel_histogram.png", dpi=150)
    plt.close(fig)
    print("\nplot written")
    np.save("results/_accel_mags_cluster.npy", mags)


if __name__ == "__main__":
    main()
