# N-Body Gravitational Simulator — Claude Code Context

## Role split
This project runs across two tools:
- **VS Code + Claude Code (here)**: writing, debugging, refactoring, testing, committing
- **Claude.ai Project ("N-Body Project")**: concept explanations, physics reasoning, stage pre-work, result interpretation

Rule: if the question needs the running code or error in front of it → stay here.
If it needs connecting code behaviour to the physics or research questions → go to the Project.

## Project structure
- `bodies.py` — state representation (mass, position, velocity arrays)
- `forces.py` — pairwise gravitational force/acceleration (vectorised, no Python loops)
- `integrators.py` — Euler, RK4, Leapfrog behind a common interface
- `simulation.py` — time-stepping loop, history logging
- `analysis.py` — energy/angular momentum drift, convergence-order fitting
- `scenarios.py` — initial conditions (2-body circular/eccentric, figure-8, chaotic cluster)

## Research questions (short form)
Q1: Do measured integrator convergence orders match theoretical orders?
Q2: Do symplectic integrators bound energy error while non-symplectic ones drift?
Q3: Angular momentum conservation across integrators?
Q4: Accuracy-order vs long-term correctness trade-off (RK4 vs Leapfrog head-to-head)?
Q5–Q7: Softening sensitivity, chaos signatures, accuracy-per-compute (stretch goals).

## Full stage plan
See `PROJECT_PLAN.md` in the repo root.

## Lab notebook
Append session notes to `labbook.md` — what was done, decisions made, anything that broke.