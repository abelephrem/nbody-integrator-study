## 2026-11-04 — GNN Stage C pre-work: design decisions (revised after Claude Code's codebase checks; G2 slope rules added 2026-11-06; noise gate replaced and scope cut for speed 2026-11-12)

Supersedes earlier versions of this doc. Revised against the full labbook; where an earlier version
contradicted a labbook decision, the labbook wins (noted inline).

### Finish-fast mode (2026-11-12)
- All rules below are frozen. For any new edge case, follow the decision policy in CLAUDE.md: check the record,
  prefer an option that keeps past decisions intact, decide, note it in the labbook, carry on. Only stop if a
  result can't be interpreted at all.
- Scope cut: no D2 holdout runs; G5 = time per force call only. Animations kept (one two-body, one cluster).

### Handling codebase surprises
- **Pure code issue** (signatures, shapes, file formats, dtypes): fix with the smallest change that keeps the
  intent below. Note it in labbook.md.
- **Changes what is measured or what a result means**: apply the CLAUDE.md decision policy; report in the
  end-of-session summary.

### Checks before step 1 (done 2026-11-04)
1. Energy-flagged clusters (09-25) excluded as rollout references: cluster_01216.
2. adaptive_rollout reproduces all 115 test and holdout files exactly (two-body and clusters).

### Decision 1 — reference and rollouts: adaptive replay, step size from the true force in every run
- Every run (reference, GNN, chaos baseline), two-body and clusters, uses the dataset's own adaptive Leapfrog
  with exactly `generate_dataset`'s settings (ε = 0.002, n_resolve = 96, encounter-gated step floor).
- At each substep the step size is chosen by the same rule from the **true** force at that run's current
  state (a_max for √(ε/a_max)/n_resolve; separations for the floor). The kick uses the GNN in GNN runs and the
  true force in reference runs. So the step rule is identical as a function of state; only the kick force differs.
- Cost: one extra true-force call per GNN substep. N from 2 to 10; a true-force call costs ~9% of a GNN call
  at every N timed (scratchpad timing).
- Why the old "fixed dt only" rule is retired: it existed to keep energy error bounded. 09-05
  (`diagnostics_symplectic.py`) showed adaptive Leapfrog at sufficient resolution gives flat energy error;
  drift came from under-resolution, not from varying the step. Fixed dt fails on clusters (C = 4.6 median even
  at dt; 0.084 at dt/64). 08-13 already planned to reuse the adaptive wrapper for GNN rollouts.
- Compare at the stored outer checkpoints (dt_base).
- No separate fixed-dt two-body runs needed.

### Models (per labbook 10-31)
- **Kick-D3 and time-D3 rolled out as equals** (test and holdout); rollouts choose the weighting. Tests the
  recorded prediction: kick wins cluster rollouts, time wins or ties two-body and quiet orbits.
- D2 = third-law comparison (equal one-step accuracy to D3). **Test set only** (holdout cut 2026-11-12).
  D1 not rolled out.
- Seed 0 for all. Seeds 1–2 already trained; use only if a result needs a spread.
- **Final weights** for every model (labbook decision 8, as in all Stage B tables). Time-D3 best-val
  (val 8.0e-6 vs final 2.1e-5): one-step evaluation only, reported beside the Stage B table as a side check.

### Precision and graph
- Integrator in float64. Edge features: displacement computed in float64, then cast to float32 (matches Stage A).
- GNN in eval mode, no gradients. Graph identical to Stage A: fully connected, disp = r_sender − r_receiver,
  log masses, stored standardisation buffers.
- Own rollout loop reusing the end-of-substep acceleration; tested against `leapfrog_step`.
  Load models with final weights explicitly.

### G2 — position divergence
**Clusters**
- Per body: d_i(t) = |r_i^A(t) − r_i^B(t)|, positions about each run's own centre of mass.
- δ(t) = sqrt(mean_i d_i²). R = 1 (virial radius, fixed by nondimensionalisation, 07-13), so δ/R = δ.
- Chaos baseline: true-force runs from a start nudged by 1e-6 in position, **three nudge seeds per cluster**.
- Horizon = first time δ ≥ 0.1.
- Compare slopes, not horizons (the GNN starts from a larger initial error, so its horizon is earlier even at
  the same slope).

**Cluster slope rules** (fixed 2026-11-06, before any model slope was computed; noise gate replaced 2026-11-12,
still before any model slope)
- Window: fit log δ from the first row with δ ≥ 1e-4 after the last row below 1e-4, to the first time δ ≥ 0.1.
  Same δ range for GNN and baseline. (2026-11-12: starting at the first row ≥ 1e-4 rather than the last row
  below it keeps fitted points in range and avoids log(0) at t = 0; mean noise 1.62 vs 1.61.)
  - Rejected: per-curve 25–75% of log rise (GNN curve starts higher, so its window covers a different δ range);
    1e-4 to 1e-2 (too short; single encounters dominate, slopes up to 23.7).
- Sensitivity check: also report the verdict for 1e-3 to 0.1, main rule unchanged. Reason: the GNN adds error
  every step, so its early curve can rise steeply before chaos-driven growth takes over; 1e-3 skips that phase.
- Baseline slope per cluster = mean of the 3 seeds' slopes.
- Voting clusters: only those where all 3 seeds reach δ = 0.1 → 18 of 20. cluster_01043 (peak 0.019) and
  cluster_01099 (peak 0.0007) reach 0.1 on no seed; reported separately like two-body (horizons only), noting
  that models reach 0.1 where chaos alone doesn't.
- Vote: a cluster votes "model error" if GNN slope ≥ 1.5 × its mean baseline slope. Overall verdict: model error
  if more than half of the voting clusters say yes (10+ of 18). Done separately for kick-D3, time-D3 and D2.
  Single-cluster ratios are not verdicts.
- Why 1.5: a convention fixed in advance, not derived. Ratio 1 = chaos-limited; the threshold must sit clearly
  above 1 to allow for fit noise, but low enough to catch a noticeable model contribution. Slope ratio 1.5 means
  the same growth (1e-4 → 0.1) in 2/3 of the time. Justified by the false-vote test below.

**Noise check (step 4a, done 2026-11-12)**
- Original gate (2026-11-06): mean over clusters of max ÷ min slope across the 3 seeds; stop if ≥ 1.5.
  **Failed:** mean 1.62, max 5.64 (cluster_01062), median 1.33, mean without cluster_01062 1.39, 6/18 clusters
  ≥ 1.5; 1e-3 window mean 2.02, max 8.75. Three fits with R² < 0.9.
- Why it was replaced, not acted on: it measured the spread between the two most extreme seeds and was dominated
  by one cluster. The question that matters is how often noise alone produces a "model error" vote. Replaced
  before any model slope was computed, so the change cannot be steered towards a result. Stated openly in the
  labbook and write-up.
- **Replacement: false-vote test.** Each true-force seed acts as a pretend model against the mean of the other two
  seeds (perfect physics, so any "model error" vote is false).
  - Results (scratch): false-vote share 18.5% at 1.25×, 7.4% at 1.5×, 1.9% at 2×, 0% at 2.5×. At 1.5×: 2, 1, 1
    false votes of 18 for the three pretend models. Ratios median 1.00, 10th–90th percentile 0.76–1.34, max 2.42.
    1e-3 window at 1.5×: 11.1%, max 3 of 18.
  - Caveat: three seeds that share data, so an indication only.
- **Pass rule (fixed 2026-11-12):** pass if no pretend model gets more than a quarter of the voting clusters as
  false votes at 1.5× (at most 4 of 18), on both windows. Currently passes (2/18 main, 3/18 on 1e-3).
- Kept as they were: 1.5× threshold, majority vote, all 18 voting clusters (cluster_01062 not dropped after
  seeing it). Not needed: higher threshold (loses sensitivity) or more seeds (margin of 2 false votes vs 10
  needed is too large for a sharper estimate to change anything).

**Reporting**
- Each model's yes-count against the noise level as well as the majority line:
  0–2 = within noise; 3–9 = model error in some clusters (more than noise explains, not a majority);
  10+ = model error in most clusters.
- Every voting cluster's ratio (GNN slope ÷ mean baseline slope) for each model, as a table and a strip/dot plot
  with 1 and 1.5 marked.
- False-vote shares at 1.25×/1.5×/2×/2.5× with the caveat. R² for every fit (list any R² < 0.9).

**Two-body** (revised 2026-11-04)
- Measure the relative orbit: divergence = |Δr_rel(t)| / a, where r_rel = r_1 − r_2 and a is the
  semi-major axis of the relative orbit. Horizon = first time divergence ≥ 0.1. No chaos baseline.
- Why not δ with R = 1: the virial radius is mass-weighted (a = m_1 m_2 when R = 1), so the real orbit shrinks
  with mass ratio and δ has a ceiling below 0.1 at mass ratio 15 (horizon undefined however wrong the rollout).
- Why not δ ≥ 0.1a: δ carries a mass factor sqrt((m_1² + m_2²)/2)/M (0.35 at mr 1, 0.66 at mr 15), so the
  threshold would still mean different things at different mass ratios.
- Meaning: on a circular orbit, 0.1 ≈ 0.1 rad (~6°) ahead or behind in phase. Ceiling ~2(1 + e), always reachable.
- Plots: two-body curves on |Δr_rel|/a, so all mass ratios share one axis. Raw δ(t) still saved.
- Two-body and cluster horizons reported separately (different measures; same spirit: error reaches 10% of
  the system's own size).

**Interpretation notes**
- Closest approach (reference runs, min_sep_run / ε): test median 0.68ε, deepest 0.18ε (1/20 below 0.2ε);
  holdout median 0.60ε, deepest 0.054ε (9/59 below 0.2ε). Training coverage is thin below 0.2ε (09-25).
  Typical encounters in both splits sit inside the covered region; only the 9 deepest holdout clusters go well
  below it. See G4.
  - Corrected 2026-11-12 (factual error, not a rule change): the earlier "test ~0.18ε, holdout ~0.05ε" quoted the
    deepest encounter in each split as if typical.

### G3 — conservation
- Energy: **ΔE/|U₀|** (the 09-25 dataset metric), so GNN, reference and dataset tolerance share one footing.
  Also show ΔE/E₀ for continuity with Stage 8.
- Angular momentum: |ΔL| / Σ|L_i(0)| (avoids small-L₀ inflation; same idea as Stage B's R_L).
- GNN alongside the true-force reference and RK4 on the same axes. RK4 run with the same step rule.
- Predictions (written in advance):
  - D3: energy error bounded, no secular trend (the integrator conserves the learned energy; size = learned vs
    true mismatch). Test with the Stage 8 Q2 measures (1% window fit and late/early ratio).
    L at float32 round-off, wandering slowly.
  - D2: energy and L may drift.
  - Kick vs time: per the 10-31 prediction.

### G4 — generalisation
- Same G2/G3 metrics on the holdout files (minus cluster_01216), **kick-D3 and time-D3 only**.
  Gap = holdout minus test.
- Separate two possible causes of the gap, no new runs: group holdout clusters by the reference run's true
  closest approach (above vs below 0.2ε). New settings (mass ratio, Q) vs deeper encounters.
  Use the reference's depth, since the GNN run follows its own chaotic branch.
- Notes: slowest holdout escapers below F (untrained); mr = 15 holdout two-body orbits precess (~−23°, 10-24),
  not quite Keplerian, so part of their error is the softened orbit.

### G5 — speed (cut to the minimum 2026-11-12)
- Time per force call only: true force (NumPy) vs each GNN, at each N from 2 to 10.
- Exclude warm-up; median of repeats. Frame as "when would a surrogate pay off" (expected: GNN slower;
  true force ~9% of a GNN call).
- Cut: wall-clock vs trajectory error curves (same conclusion, extra session).

### Animations (kept)
- One two-body and one cluster, GNN (kick-D3) vs true-force reference overlaid, .mp4.
- Pick a cluster with a visible encounter and a clear divergence.

### Compute
- Estimates per model: test clusters ~40 min, holdout clusters ~2.6 h, all two-body ~15–20 min.
- Run long jobs as detached processes in parallel, outside OneDrive (C:/nbody_results), as in Stage B.

### Session plan (finish-fast)
1. ~~Checks, rollout loop, replay check, noise check~~ (done).
2. **Session 1:** step 4b (model slopes, per-cluster ratios, vote; main and 1e-3 windows) + G3 on test, for
   kick-D3, time-D3 and D2. Launch holdout rollouts for kick-D3 and time-D3 as detached overnight jobs.
3. **Session 2:** G4 results, G5 timing, the two animations.
4. **Session 3:** Stage C labbook entry. Then the final write-up (Stage 9 stays deferred).
- End each session with one short summary: verdict per model (yes-count of 18), key G3 plots, decisions taken
  under the CLAUDE.md policy, anything against the predictions.

### Further cuts if still short of time
1. D2 test rollouts
(Kick-D3 and time-D3 are never cut: they answer the weighting question. Animations are kept.)
