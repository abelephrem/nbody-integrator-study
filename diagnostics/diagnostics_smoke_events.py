"""Step 4: smoke test of the close-encounter recording change. ONE cluster, scratch path.

cluster_01078 - true min_sep 0.122 eps against a stored 1.430 eps, the largest known
recording gap in the set.

Writes to a scratch directory, never to C:/nbody_data/. Nothing is regenerated and no
existing file is touched.

The load-bearing check is that recording is READ-ONLY with respect to the dynamics:
checkpoint positions must be bitwise identical to a run with record_events=False, and
force_evals / max_n_sub must be unchanged.
"""
import os
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(_HERE), _HERE]  # repo root for `from bodies import`,
#                                                 this dir for sibling diagnostics
import os
import sys

import numpy as np
import h5py

from bodies import SystemState
from integrators import leapfrog_step
from simulation import run_simulation, save_trajectory
from scenarios import (cluster_configs, build_initial_state, resample_events,
                       merge_event_rows)
from analysis import total_energy

SEED = 1078
EPS = 0.002
OUTER_DT = 0.01
DURATION = 20.0
N_RESOLVE = int(sys.argv[1]) if len(sys.argv) > 1 else 96  # section D production value
SCRATCH = os.path.join(
    os.environ.get("TEMP", "."), "nbody_smoke")
G = 1.0


def drift(traj):
    E = np.array([total_energy(SystemState(traj.masses, traj.positions[t],
                                           traj.velocities[t]),
                               G=traj.G, softening=traj.softening)
                  for t in range(len(traj.times))])
    return float(np.max(np.abs((E - E[0]) / E[0])))


def min_sep_series(pos):
    d = np.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=3)
    i = np.arange(pos.shape[1])
    d[:, i, i] = np.inf
    return d.min(axis=(1, 2))


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    cfg = [c for c in cluster_configs() if c.seed == SEED][0]
    state0 = build_initial_state(cfg)
    n_steps = round(DURATION / OUTER_DT)
    print(f"cluster_{SEED:05d} {cfg.params} split={cfg.split}, scratch={SCRATCH}\n")

    ctl = run_simulation(state0, leapfrog_step, dt=OUTER_DT, n_steps=n_steps,
                         scenario_name="cluster", G=G, softening=EPS, adaptive=True,
                         n_resolve=N_RESOLVE, record_events=False)
    rec = run_simulation(state0, leapfrog_step, dt=OUTER_DT, n_steps=n_steps,
                         scenario_name="cluster", G=G, softening=EPS, adaptive=True,
                         n_resolve=N_RESOLVE, record_events=True)

    # ---- 1. recording must not perturb the dynamics -----------------------------
    pos_same = np.array_equal(ctl.positions, rec.positions)
    vel_same = np.array_equal(ctl.velocities, rec.velocities)
    acc_same = np.array_equal(ctl.accelerations, rec.accelerations)
    print("NO-PERTURBATION CHECK")
    print(f"  checkpoint positions bitwise identical : {pos_same}")
    print(f"  checkpoint velocities bitwise identical: {vel_same}")
    print(f"  checkpoint accelerations identical     : {acc_same}")
    print(f"  force_evals {ctl.metadata['force_evals']} vs "
          f"{rec.metadata['force_evals']}  "
          f"({'same' if ctl.metadata['force_evals'] == rec.metadata['force_evals'] else 'DIFFERENT'})")
    print(f"  max_n_sub   {ctl.metadata['max_n_sub']} vs {rec.metadata['max_n_sub']}  "
          f"({'same' if ctl.metadata['max_n_sub'] == rec.metadata['max_n_sub'] else 'DIFFERENT'})")
    print(f"  energy drift {drift(ctl):.6e} vs {drift(rec):.6e}")
    print("")
    print(f"STEP-SIZE CAP (n_resolve={N_RESOLVE})")
    print(f"  t_core {rec.metadata['t_core']:.4e}  cap {rec.metadata['t_core']/100:.4e}")
    print(f"  n_capped {rec.metadata['n_capped']} (0 = never fired), "
          f"min_sep_capped {rec.metadata['min_sep_capped']:.4e}")
    print(f"  control n_capped {ctl.metadata['n_capped']}")

    with h5py.File(f"C:/nbody_data/cluster_{SEED:05d}.h5", "r") as f:
        stored_fe = int(f.attrs["force_evals"])
        stored_mns = int(f.attrs["max_n_sub"])
        stored_pos = f["positions"][:]
    print(f"  vs STORED production run: force_evals {stored_fe} "
          f"({'match' if stored_fe == rec.metadata['force_evals'] else 'DIFFER'}), "
          f"max_n_sub {stored_mns} "
          f"({'match' if stored_mns == rec.metadata['max_n_sub'] else 'DIFFER'})")

    # ---- 2. min_sep_run --------------------------------------------------------
    mrun = rec.metadata["min_sep_run"]
    stored_min = float(min_sep_series(stored_pos).min())
    ckpt_min = float(min_sep_series(rec.positions).min())
    print(f"\nMIN_SEP")
    print(f"  min_sep_run (every substep)      : {mrun:.6e} = {mrun/EPS:.4f} eps")
    # Pre-cap anchor from 07 Sep: n_resolve=24 with NO step-size cap. Kept because it
    # is the figure the whole investigation rests on, but it will NOT match a current
    # run - the cap changes the branch, and 24 now gives 0.3298 eps.
    print(f"  pre-cap reference (07 Sep, n_resolve=24, no cap): 2.44e-04 = 0.122 eps")
    print(f"  min over CHECKPOINT rows         : {ckpt_min:.6e} = {ckpt_min/EPS:.4f} eps")
    print(f"  min over STORED saved snapshots  : {stored_min:.6e} = {stored_min/EPS:.4f} eps")
    print(f"  understatement factor, stored vs true: {stored_min/mrun:.2f}x")
    print(f"  control run min_sep_run          : {ctl.metadata['min_sep_run']:.6e} "
          f"({'same' if ctl.metadata['min_sep_run'] == mrun else 'DIFFERENT'})")

    # ---- 3. merge --------------------------------------------------------------
    ev = rec.events
    print(f"\nEVENT ROWS")
    print(f"  episodes detected : {rec.metadata['n_episodes']}")
    print(f"  event rows        : {len(ev['times'])} "
          f"({len(ev['times'])/max(rec.metadata['n_episodes'],1):.1f} per episode, cap 16)")
    nmin = rec.metadata['n_minima']
    print(f"  closest-approach rows : {nmin} "
          f"({nmin/max(rec.metadata['n_episodes'],1):.2f} per episode)")
    print(f"  control run n_minima  : {ctl.metadata['n_minima']} (must be 0)")
    print(f"  ev_t sorted at source : {bool(np.all(np.diff(ev['times']) > 0))} "
          f"(expected False - rows emitted one substep late)")

    base = resample_events(rec, 100)
    merged = merge_event_rows(base, ev)
    t = merged.times
    print(f"  baseline rows {len(base.times)} + events -> merged {len(t)}")
    print(f"  strictly time-ordered : {bool(np.all(np.diff(t) > 0))}")
    print(f"  duplicate timestamps  : {len(t) - len(np.unique(t))}")

    # rows inside encounters, before vs after
    ms_base = min_sep_series(base.positions)
    ms_merged = min_sep_series(merged.positions)
    for f_ in (3, 2, 1, 0.5):
        print(f"  rows with min_sep < {f_}eps : baseline {int((ms_base < f_*EPS).sum()):4d}"
              f"  ->  merged {int((ms_merged < f_*EPS).sum()):4d}")

    # ---- 4. acceleration tail ---------------------------------------------------
    a_base = np.linalg.norm(base.accelerations, axis=2).ravel()
    a_merged = np.linalg.norm(merged.accelerations, axis=2).ravel()
    print(f"\nACCELERATION TAIL (this run)")
    for q in (99, 99.9, 100):
        print(f"  p{q:<6} baseline {np.percentile(a_base, q):.4e}"
              f"  ->  merged {np.percentile(a_merged, q):.4e}")
    m = rec.masses
    m_pair_max = np.sort(m)[-1]
    peak = G * m_pair_max * (EPS/np.sqrt(2)) / ((EPS/np.sqrt(2))**2 + EPS**2)**1.5
    print(f"  peak available (heaviest partner at eps/sqrt2): {peak:.4e}")
    print(f"  baseline reaches {a_base.max()/peak:.1%} of it, merged {a_merged.max()/peak:.1%}")

    # ---- 5. size ---------------------------------------------------------------
    p_base = os.path.join(SCRATCH, f"baseline_{SEED:05d}.h5")
    p_merged = os.path.join(SCRATCH, f"merged_{SEED:05d}.h5")
    base.metadata = {**rec.metadata, "scenario_type": "cluster", "split": cfg.split,
                     "seed": SEED, **cfg.params}
    merged.metadata = dict(base.metadata)
    save_trajectory(base, p_base)
    save_trajectory(merged, p_merged)
    sb, sm = os.path.getsize(p_base), os.path.getsize(p_merged)
    stored_sz = os.path.getsize(f"C:/nbody_data/cluster_{SEED:05d}.h5")
    print(f"\nSIZE")
    print(f"  stored production file : {stored_sz/1024:8.1f} KiB ({len(stored_pos)} rows)")
    print(f"  baseline (this run)    : {sb/1024:8.1f} KiB ({len(base.times)} rows)")
    print(f"  merged                 : {sm/1024:8.1f} KiB ({len(t)} rows)  "
          f"x{sm/stored_sz:.2f} vs stored")
    print(f"  extrapolated full set (260 clusters + 100 two-body, current set 21 MB):")
    print(f"     clusters ~{sm*260/1024/1024:.1f} MiB  (+ two-body unchanged)")


if __name__ == "__main__":
    main()
