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
— Leapfrog confirmed by Q2 + Q4 as ground-truth generator for GNN training data —
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