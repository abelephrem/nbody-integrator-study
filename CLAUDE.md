# Gravitational N-Body Simulation and GNN Surrogate Modelling — Claude Code Context

## Role split
This project runs across three tracks:
- **VS Code + Claude Code (here)**: writing, debugging, refactoring, testing, committing — for both simulator and GNN stages
- **Claude.ai Project ("N-Body Project")**: concept explanations, physics and ML reasoning, stage pre-work, result interpretation
- **GNN learning track**: parallel reading (3B1B, DeepMind paper, Distill.pub) running alongside simulator work — see PROJECT_PLAN.md Part 4

Rule: if the question needs the running code or error in front of it → stay here.
If it needs connecting code behaviour to the physics, numerics, or research questions → go to the Project.

## Project structure
- `bodies.py` — state representation (mass, position, velocity arrays)
- `forces.py` — pairwise gravitational force/acceleration (vectorised, no Python loops)
- `integrators.py` — Euler, RK4, Leapfrog behind a common interface
- `simulation.py` — time-stepping loop, history logging, HDF5 trajectory output (`h5py`)
- `analysis.py` — energy/angular momentum drift, convergence-order fitting
- `scenarios.py` — initial conditions and data generation pipeline (2-body, figure-8, chaotic cluster, varied-IC sweep)
- GNN Stages A–C add new files after Stage 8 — structure to be decided at that point

## Key dependencies
- `numpy`, `matplotlib`, `scipy` — simulator and analysis
- `h5py` — HDF5 trajectory storage (Stage 4); must be installed before Stage 4 begins
- `torch`, `torch_geometric` — GNN implementation (Stages A–C); not needed until then

## Research questions (short form)
Q1: Do measured integrator convergence orders match theoretical orders?
Q2: Do symplectic integrators bound energy error while non-symplectic ones drift?
Q3: Angular momentum conservation across integrators?
Q4: Accuracy-order vs long-term correctness trade-off (RK4 vs Leapfrog head-to-head)?
Q5–Q6: Softening sensitivity, chaos signatures (optional stretch goals).
— Leapfrog confirmed by Q2 + Q3 + Q4 as ground-truth generator for GNN
  training data; argument is bounded/unbiased error, not smaller error.
  RK4's energy error is smaller on the eccentric case — do not claim
  otherwise. See PROJECT_PLAN.md Part 1 bridge. —
G1: Does the GNN learn to predict per-particle accelerations accurately?
G2: How does rollout trajectory error grow over time vs Leapfrog ground truth?
G3: Does the GNN rollout conserve energy and angular momentum comparably to Leapfrog?
G4: Does the model generalise to unseen initial conditions?
G5: Speed vs accuracy trade-off across all methods (integrators + GNN on shared axes)?

Note: original Q7 (accuracy-per-compute across integrators only) is absorbed into G5.

## Full stage plan
See `PROJECT_PLAN.md` in the repo root.

## Lab notebook
Append session notes to `labbook.md` — what was done, decisions made, anything that broke.
## Decision policy (finish-fast mode, from 2026-11-12)

When an edge case or design question comes up, decide it yourself. Don't stop to ask. Follow these steps:

1. **Check the record first.** Search labbook.md, PROJECT_PLAN.md and the decisions docs
   (stage-b/stage-c-decisions.md) for any earlier decision, test or finding on the same issue. Past diagnostics
   often already answer it (e.g. 09-05 showed adaptive Leapfrog is fine when resolved).
2. **List the options, including one that keeps every past decision and rule intact.** If such an option exists,
   take it. Only accept a trade-off if no downside-free option exists.
3. **Prefer the option that keeps the comparison fair.** Same data, same windows and same rules for every model
   and baseline; only the thing being tested should differ.
4. **If a choice changes what a result means and both versions are cheap (< ~30 min compute), compute both,**
   report both, and headline the one the pre-set rules in stage-c-decisions.md favour.
5. **Never change a rule, threshold or window after seeing model results.** Rule changes are only allowed before
   model data exists, and must be stated openly in the labbook.
6. **Don't drop data after seeing it** (clusters, files, seeds) unless an existing rule already says to.
7. **Record every decision in the labbook:** what came up, the options, which was taken and why (one or two lines).
8. **Only stop** if a result cannot be interpreted under any option. Otherwise put open questions in the
   end-of-session summary.
9. **End each session with one short summary:** results, decisions taken under this policy, open questions.
