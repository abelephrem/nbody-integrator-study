# Gravitational N-Body Simulation and GNN Surrogate Modelling — Project Stage Plan

## Part 0: Workflow & Roles (Three-Track System)

This project runs across three separate streams with three distinct jobs. None has live visibility into the others — so this section exists to make the division explicit rather than relying on any side guessing.

### The three tracks

**Claude.ai, this Project — the "why" track**
- Role: concept explanations before code is written, stage planning and scope decisions, interpreting results against the physics, weekly progress check-ins, GNN conceptual pre-work before each GNN stage.
- What it has: continuity of *reasoning* across sessions (this Project's memory), and on-request access to the GitHub repo (single-file fetch or full clone) when asked.
- What it does NOT have: live awareness of what's been coded, what's currently broken, or how far along a stage you are. It must be told, or asked to go fetch.

**VS Code + Claude Code extension — the "how" track**
- Role: writing, running, debugging, refactoring, testing, committing. Anything that needs to see the actual file, the actual error, the actual diagnostics. Applies equally to simulator stages and GNN stages.
- What it has: live file and diagnostic context this Project doesn't have.
- What it does NOT have: the conceptual throughline — why a design choice was made, how a stage connects to the research questions, the physics or ML reasoning behind a result. That context lives here.

**GNN learning track — the parallel reading stream**
- Role: building the conceptual foundations for GNN implementation before any GNN code is written. Runs at roughly 2 hrs/week alongside the simulator work.
- What it covers: neural network concepts (3Blue1Brown), the direct research precedent (DeepMind paper), and graph neural network mechanics (Distill.pub). See Part 4 for the week-by-week schedule.
- What it is NOT: a full ML curriculum. PyTorch syntax, PyTorch Geometric, and the training loop are all learned in-context during GNN Stages A–C, the same way NumPy is learned during the simulator stages. The reading track covers concepts only; implementation knowledge comes from doing.
- When it connects back here: before GNN Stage A begins, there is a conceptual pre-work session in this Project (same as the pre-work done before each simulator stage) covering training loops, loss functions, and backpropagation at a conceptual level — enough to interpret what the GNN is doing, not to implement it from scratch.

### The decision rule

> Does answering this need to see the actual running code/error in front of it? → **VS Code.**
> Does answering this need to connect code behaviour to the physics, numerics, or research questions? → **this Project.**

| Stay in VS Code | Come back to this Project |
|---|---|
| "This broadcasting throws a shape mismatch on line 12 — why?" | "Why does Leapfrog conserve energy better than RK4 even though RK4 has higher formal order?" |
| "Explain this traceback" | "Does this energy-drift plot look like what we'd expect, or is something wrong?" |
| "Refactor this to use a consistent interface" | "I'm starting Stage 3 — walk me through why a shared interface matters here" |
| "Write a pytest for this function" | "I'm starting Stage 6 — what should the convergence-order fit actually look like conceptually?" |
| Day-to-day syntax / NumPy / PyTorch questions tied to a specific file | Weekly check-in: progress vs. the timeline table, adjust scope if needed |
| "Why is `r` going negative here" (debugging a value) | "Is this energy spike physics or a bug?" (interpreting what it means) |
| "My training loss isn't decreasing — fix it" | "My training loss isn't decreasing — is this a learning rate problem or a data problem?" |

### The bridge (since there's no automatic sync)

- **Repo**: push to a public GitHub repo. This Project can fetch a single file (raw URL) or clone the whole thing on request — but only when asked, and only the state as of the last push. Treat it as pull-on-request, not live.
- **Lab notebook** (`labbook.md`, see Notes below): your dated session log doubles as the bridge. A one-line status when you come back here — e.g. *"Finished Stage 2, energy roughly conserved, took longer than estimated due to a shape-mismatch bug"* — is usually all the recalibration needed.
- **`CLAUDE.md`**: unlike this document, `CLAUDE.md` is read automatically by Claude Code at the start of *every* session — this file isn't. So `CLAUDE.md` should be short: the condensed role split and decision rule from this section, plus an explicit pointer ("see `PROJECT_PLAN.md` for the full stage plan, research questions, and timeline"). This document is committed to the repo as `PROJECT_PLAN.md` for reference — the extension *can* read it when relevant, but only `CLAUDE.md` is guaranteed to load without being asked.

### The ritual

1. **Before starting a new stage** → come here for the conceptual pre-work (as we did for symplectic integration before any code existed; same pattern will apply before GNN Stage A).
2. **Finishing a stage, or stuck on a concept (not a bug)** → come here with a one-line status.
3. **Debugging, syntax, refactoring** → stay in VS Code; don't route these here.
4. **Roughly weekly** → check progress against the Part 4 timeline table here, adjust scope if running over.

---

## Part 1: Research Questions (the "why")

Everything below should serve answering these. If a piece of code doesn't help answer one of these, it's scope creep — cut it or defer it.

**Primary question:**
> Can a neural network learn to propagate a gravitational N-body system as faithfully as the best physics-preserving numerical integrator — and at what computational cost?

This framing makes the simulator work *prerequisite methodology* for the GNN work, not a separate investigation. You cannot answer the primary question without first establishing what the best physics-preserving integrator actually does — that is what Q1–Q4 deliver. The GNN is then held to that standard.

**Simulator sub-questions (answered in Stages 6–8):**

| # | Question | What you measure |
|---|----------|-------------------|
| Q1 | Does each integrator's *empirical* convergence order match its *theoretical* order? | Slope of log(global error) vs log(step size) |
| Q2 | Do symplectic integrators bound energy error while non-symplectic ones drift unboundedly? | Energy error ΔE/E₀ vs time, for many orbital periods |
| Q3 | Is angular momentum conserved differently across integrator types? | ΔL/L₀ vs time |
| Q4 | Is there a trade-off between formal accuracy (order) and long-term qualitative correctness? | Compare RK4 (order 4, non-symplectic) vs Leapfrog (order 2, symplectic) head-to-head |
| Q5 | How does softening affect energy conservation during close encounters? | Energy spikes at closest approach, with/without softening, across ε values |
| Q6 | At what point does chaos (not integrator error) dominate position error in the 3-body problem? | Divergence of two near-identical initial conditions vs time (Lyapunov-like estimate) |

Q1–Q3 are the core deliverables. Q4 is the "so what" that makes this more than a benchmark table — and critically, it is also the justification for selecting Leapfrog as the GNN training data generator (see bridge below). Q5 and Q6 are optional extensions (Stage 9); treat them as the first things to cut if time is short.

*Note: the original Q7 (accuracy-per-unit-compute trade-off across integrators) is not dropped — it is expanded and absorbed into G5, where the same analysis is applied to all methods including the GNN surrogate.*

**The bridge — from simulator results to GNN:**

At the end of Stage 8, Leapfrog is formally selected as the ground-truth generator for GNN training data, justified by the Q2 and Q4 results. This is not an arbitrary choice — it follows directly from the evidence. Training the GNN on RK4-generated trajectories would mean training it to learn slightly corrupted physics; the simulator results are the justification for the GNN methodology.

**GNN sub-questions (answered in GNN Stages A–C):**

| # | Question | What you measure |
|---|----------|-------------------|
| G1 | Does the GNN learn to predict per-particle accelerations accurately? | MSE between predicted and true accelerations on a held-out test set |
| G2 | When the trained GNN is rolled out autonomously, how does trajectory error grow over time? | Position error vs time, compared against Leapfrog ground truth |
| G3 | Does the GNN rollout conserve energy and angular momentum comparably to Leapfrog? | ΔE/E₀ and ΔL/L₀ vs time — GNN alongside Leapfrog and RK4 on the same axes |
| G4 | Does the model generalise to initial conditions it was not trained on? | Rollout error on out-of-distribution initial conditions vs in-distribution |
| G5 | What is the speed vs accuracy trade-off across all methods? | Wall-clock time vs trajectory error — all integrators and GNN on shared axes |

G1 is the training validation: it tells you the model learned something. G2 and G3 are the physically meaningful tests — a model that predicts accelerations well locally can still accumulate catastrophic error over long rollouts, and energy conservation is the physics check. G4 tests whether the model learned the underlying rule or merely memorised training cases. G5 is the "so what" — the whole point of a surrogate is speed, so the trade-off must be quantified.

G3 directly connects the GNN results back to the simulator investigation: the GNN is added as a fourth row to the integrator comparison table, evaluated on identical metrics. This is the result that makes the project one coherent study rather than two separate ones.

---

## Part 2: Quantification Framework

This is the analysis machinery that turns simulation and training output into evidence. It gets built incrementally, but it is worth seeing the whole picture now.

**Core metrics (computed for every simulator run):**
- **Energy drift**: `(E(t) - E0) / |E0|` over time
- **Angular momentum drift**: `(L(t) - L0) / |L0|` over time
- **Position error**: deviation from an analytic reference (two-body Kepler orbit) at matching timestamps
- **Global convergence order**: fit `log(error) = p·log(h) + c` via linear regression across a range of step sizes; compare fitted `p` to theory (Euler ≈ 1, Leapfrog ≈ 2, RK4 ≈ 4)
- **Compute cost**: wall-clock time or step count, paired with achieved accuracy

**GNN-specific metrics (computed during GNN Stages A–C):**
- **Acceleration MSE**: mean squared error between GNN-predicted and true per-particle accelerations on the held-out test set (G1)
- **Rollout position error**: deviation of GNN-generated trajectory from Leapfrog ground truth at matching timestamps (G2)
- **GNN energy and angular momentum drift**: same formulas as core metrics above, applied to GNN rollout trajectories (G3)
- **Generalisation gap**: difference in rollout error between in-distribution and out-of-distribution initial conditions (G4)

**Unified comparison table (the headline result of the whole project):**

| Method | Symplectic? | Theoretical order | Measured order | Energy drift behaviour | Trajectory error | Cost per step |
|---|---|---|---|---|---|---|
| Euler | No | 1 | | | | |
| Leapfrog | Yes | 2 | | | | |
| RK4 | No | 4 | | | | |
| GNN surrogate | N/A | N/A | N/A | | | |

The GNN row has no theoretical order (it is not a numerical integrator) but is evaluated on the same energy drift and cost metrics as the integrators. This is what makes the table a unified result rather than two separate tables.

**Secondary analyses:**
- Energy error vs softening parameter ε (heatmap or line plot) — Q5
- Trajectory divergence rate for two infinitesimally separated 3-body initial conditions (chaos signature) — Q6
- Wall-clock time vs trajectory error across all methods, with GNN as a single point on integrator curves — G5

---

## Part 3: Stage-by-Stage Plan

Each stage lists: **physics/numerics focus**, **deliverable**, **validation built in**, and **Python skills you'll practice**. Throughout, the pairing model is: I scaffold interfaces and explain anything mathematically subtle, you write the core logic, we debug together. You're not meant to type out boilerplate alone — the goal is that you *understand every line*, not that you typed every line.

### Stage 0 — Environment & Project Skeleton ✓
- **Time estimate**: ~1 session (2 hrs)
- **Questions addressed**: none directly — pure infrastructure.
- **Goal**: working repo, not yet doing physics.
- **Deliverable**: folder structure (`bodies.py`, `integrators.py`, `forces.py`, `simulation.py`, `analysis.py`, `scenarios.py`), virtual environment, git repo with first commit, `requirements.txt`.
- **Python skills**: venv/pip basics, project structure, git fundamentals (init, add, commit, .gitignore), why modular files matter.
- **Validation**: none yet — this is just plumbing.

### Stage 1 — Core Data Structures (`bodies.py`) ✓
- **Time estimate**: ~2–3 sessions (4–6 hrs)
- **Questions addressed**: none directly — but the array-vs-object decision here determines whether Q1–G5 are even computationally feasible later (vectorized force/energy calcs depend on it).
- **Goal**: represent a system of bodies (mass, position, velocity) in a way that's both readable and fast.
- **Deliverable**: a `Body` representation and a `System`/`State` representation. Key design decision: object-per-body (readable, slow) vs NumPy arrays of shape `(N, 3)` (fast, vectorizable) — and why N-body work always ends up using the latter.
- **Python skills**: `dataclasses`, type hints, NumPy array basics (shapes, broadcasting), the readability-vs-performance trade-off.
- **Validation**: unit test — can you create a 2-body system and read back correct positions/velocities?

### Stage 2 — Force Calculation (`forces.py`)
- **Time estimate**: ~2–3 sessions (4–6 hrs) — broadcasting is one of the trickier NumPy concepts; don't rush it if shape errors are confusing
- **Questions addressed**: builds the mechanism for **Q5** (softening parameter ε goes in here, even though you won't analyse its effect until Stage 9). No question is answerable yet.
- **Goal**: Newtonian gravity, vectorized across all pairs.
- **Deliverable**: pairwise force/acceleration calculator using NumPy broadcasting (no nested Python loops), with a softening parameter ε built in from day one (not bolted on later).
- **Python skills**: NumPy broadcasting for pairwise operations, avoiding O(N²) Python loops, why vectorization matters for performance.
- **Validation**: known analytic case — two bodies, check the force magnitude matches `Gm₁m₂/r²` by hand calculation.

### Stage 3 — Integrators (`integrators.py`)
- **Time estimate**: ~3–4 sessions (6–8 hrs) — three algorithms, each needs its own implementation + sanity check
- **Questions addressed**: this is the object under study for **Q1–Q4** — without multiple integrators behind a common interface, there's nothing to compare.
- **Goal**: implement Euler (explicit), RK4, and Leapfrog/Velocity-Verlet behind a common interface.
- **Deliverable**: each integrator takes a state + forces function + step size, returns the next state. Common interface (e.g. matching function signatures, or a shared abstract base class) so `simulation.py` can swap integrators without caring which one it's using.
- **Python skills**: function design / interface consistency, optionally abstract base classes (`abc` module) — this is a good place to learn *why* a consistent interface matters, not just OOP syntax for its own sake.
- **Validation**: simple harmonic oscillator or two-body circular orbit, sanity-check each integrator visually before trusting it on harder cases.

### Stage 4 — Simulation Engine (`simulation.py`)
- **Time estimate**: ~3 sessions (6 hrs) — one extra session beyond the original estimate for HDF5 instrumentation
- **Questions addressed**: none answered yet, but this is the last piece of plumbing needed before any of Q1–G5 become measurable. It also produces the training dataset the GNN depends on.
- **Goal**: the time-stepping loop that ties bodies + forces + integrator together, logs history, and saves trajectories to disk in a format the GNN can consume.
- **Deliverable**: run a scenario for N steps, store position/velocity/energy/angular-momentum history at each step (or every k steps, for memory); trajectories saved in HDF5 format via `h5py` (positions, velocities, accelerations at every saved step). The HDF5 output is the bridge between the simulator and the GNN — it must be built in here, not retrofitted later.
- **Python skills**: writing clean loops, pre-allocating NumPy arrays for history (vs. Python list appends — another performance lesson), basic logging/progress reporting, `h5py` for structured numerical file I/O.
- **Validation**: run a known 2-body circular orbit, confirm it visually looks like a circle and doesn't immediately blow up; confirm HDF5 file contains the expected arrays at the expected shapes.

### Stage 5 — Scenario Library, Validation Suite & Data Generation Pipeline (`scenarios.py`)
- **Time estimate**: ~4–5 sessions (8–10 hrs) — expanded from the original estimate; the data generation pipeline adds meaningful scope beyond a handful of named scenarios
- **Questions addressed**: provides the test cases each simulator question needs, and generates the training dataset the GNN needs. The two-body orbits give the analytic reference Q1 requires; the figure-8 and chaotic cluster are what Q6 will later run on. The varied initial conditions sweep is what G1–G5 depend on.
- **Goal**: a set of well-defined initial conditions with known correct behaviour for simulator validation, plus a systematic data generation pipeline for GNN training.
- **Deliverable**: functions that build initial conditions for (a) two-body circular orbit, (b) two-body eccentric orbit, (c) figure-8 three-body orbit (a known stable analytic solution), (d) a small chaotic cluster (4–5 bodies); plus a data generation script that sweeps across varied initial conditions (different particle counts, masses, separations, orbital configurations), runs each through the Leapfrog integrator, and saves trajectories to HDF5. The output of this script is the GNN training dataset.
- **Python skills**: function design with sensible defaults, returning structured data, writing a configurable data generation loop, saving structured output.
- **Validation**: each named scenario should have a known expected behaviour you can check against (e.g. figure-8 should retrace itself); the data pipeline should produce HDF5 files with the correct shapes and physically reasonable values before any GNN training begins.

### Stage 6 — Analysis & Metrics (`analysis.py`)
- **Time estimate**: ~3 sessions (6 hrs) — the convergence-order regression is a new concept, give it room to actually click rather than just running someone else's recipe
- **Questions addressed**: this is where **Q1, Q2, Q3** become computable for the first time.
- **Goal**: turn raw simulation history into the quantified outputs from Part 2.
- **Deliverable**: functions for energy calculation, angular momentum calculation, position error vs analytic reference, and convergence-order fitting (log-log regression).
- **Python skills**: `numpy.polyfit` or `scipy.stats.linregress` for the convergence fit, careful array indexing/slicing, writing functions that are reusable across scenarios.
- **Validation**: known case first — e.g. confirm energy is *exactly* conserved (to floating point) for an analytically circular orbit with infinitesimal step size, before trusting the metric on harder cases.

### Stage 7 — Visualization
- **Time estimate**: ~2–3 sessions (4–6 hrs)
- **Questions addressed**: makes **Q1–Q4** legible rather than answering anything new — the convergence plot is the visual evidence for Q1, the energy-drift plot is the visual evidence for Q2/Q4.
- **Goal**: make the results legible — this is what makes the project CV-worthy rather than just a folder of numbers.
- **Deliverable**: trajectory plots, energy-drift-vs-time plots (log scale), convergence plots (log-log error vs step size with fitted slope), phase space portraits for the 2-body case; animated visualisations using `matplotlib.animation.FuncAnimation` showing bodies moving in real time with orbital trails — produced for each scenario (two-body orbits, figure-8, chaotic cluster) and saved as `.mp4`. A particularly effective animation is the same orbit run under all three integrators side by side — Leapfrog staying stable, Euler spiralling outward — making the integrator comparison visible rather than just a table of numbers.
- **Python skills**: `matplotlib` fundamentals (subplots, log scales, legends, saving figures), `FuncAnimation` for frame-by-frame animation, saving video output via `ffmpeg`, the discipline of "every plot needs a clear question it's answering."
- **Validation**: implicit — if your plots don't visually match the physics you expect (e.g. RK4 energy drift should look roughly linear/unbounded, Leapfrog should look bounded/oscillatory), something upstream is wrong. Animations should look physically plausible (orbits closed, no bodies flying off to infinity on a stable scenario) before being used in the portfolio.

### Stage 8 — Experiment Suite (full comparison runs)
- **Time estimate**: ~3–4 sessions (6–8 hrs) — includes time to actually look at and interpret results, not just generate them
- **Questions addressed**: **Q1, Q2, Q3, Q4** all get directly and fully answered here.
- **Goal**: systematically run every integrator × every scenario × a range of step sizes, and generate the actual research dataset.
- **Deliverable**: a script that produces the comparison table and all key plots in one run, results saved to disk (CSV or similar) so you're not re-running expensive simulations every time you tweak a plot.
- **Python skills**: nested loops over configurations, saving/loading data (CSV via `csv` or `pandas`), separating "generate data" from "plot data" (a real research-software practice).
- **Validation**: the comparison table is itself a built-in check — results should make physical sense (e.g. measured orders roughly matching theory) before you trust them as findings.
- **Bridge**: at the end of Stage 8, Leapfrog is formally confirmed as the ground-truth generator for GNN training data, justified by the Q2 and Q4 results. The data generation pipeline from Stage 5 is run using Leapfrog only, producing the final training dataset.

### Stage 9 — Extensions (Q5–Q6, time permitting)
- **Time estimate**: ~3–4 sessions (6–8 hrs) total if doing both sub-studies (~1.5–2 sessions each for Q5 and Q6) — easy to scale down to just one
- **Questions addressed**: **Q5, Q6** — directly and exclusively. Note that Q7 from the original plan (accuracy-per-unit-compute trade-off) is not included here — it has been expanded and absorbed into G5 in GNN Stage C, where the same analysis covers all methods including the GNN surrogate.
- **Goal**: the parts that elevate the simulator section from "I built an N-body sim" to "I investigated a real numerical-physics question in depth."
- **Deliverable**: softening sensitivity study (Q5), chaos/divergence study on the 3-body case (Q6).
- **Python skills**: whatever's needed by then — likely more `matplotlib`, maybe `scipy` for curve fitting.
- **Note**: treat these as optional/stretch — better to have Stages 0–8 and GNN Stages A–C solid than Stage 9 half-done. If time is short, Q6 (the chaos study) is the more interesting standalone result.

### GNN Stage A — Graph Construction
- **Time estimate**: ~2–3 sessions (4–6 hrs)
- **Questions addressed**: sets up G1–G5 — without graph construction there is no GNN to train. No question is answerable yet.
- **Goal**: convert saved HDF5 simulation snapshots into graph-structured tensors suitable for GNN input.
- **Deliverable**: a data loading pipeline that reads HDF5 trajectory files and produces per-snapshot graphs — node feature tensors (position, velocity, mass per particle), edge feature tensors (displacement vector, scalar distance per pair within a cutoff radius), dataset split into train/validation/test sets.
- **Python skills**: PyTorch tensors and PyTorch Geometric data structures (both learned in context), HDF5 reading via `h5py`.
- **Validation**: inspect a sample graph — correct node count, edge count for a given cutoff radius, feature dimensions match what the model will expect.

### GNN Stage B — Model Architecture & Training
- **Time estimate**: ~3–4 sessions (6–8 hrs) — the most conceptually dense GNN stage; come back to this Project if training misbehaves and the cause isn't obvious from the code
- **Questions addressed**: **G1** — acceleration MSE on the held-out test set is measurable at the end of this stage.
- **Goal**: implement a message-passing GNN that predicts per-particle accelerations from the current graph state, and train it on the Leapfrog-generated dataset.
- **Deliverable**: GNN model using PyTorch Geometric message passing, training loop with train/validation loss curves, saved model weights, G1 measurement (acceleration MSE on test set versus a naive baseline).
- **Python skills**: PyTorch Geometric message passing layer, training loops, loss functions, model checkpointing (all learned in context).
- **Validation**: training loss should decrease and validation loss should track it without divergence. G1 MSE should be meaningfully below a naive baseline (e.g. predicting zero acceleration for all particles). If training is unstable, come back to this Project to interpret what it means before debugging in VS Code.

### GNN Stage C — Rollout Validation & Unified Comparison
- **Time estimate**: ~2–3 sessions (4–6 hrs)
- **Questions addressed**: **G2, G3, G4, G5** — all answered here. This is also where the unified comparison table (Euler, Leapfrog, RK4, GNN) is completed and the primary research question is answered.
- **Goal**: couple the trained GNN to the existing integrator and run autonomous rollouts; validate against Leapfrog ground truth; produce the unified comparison across all methods.
- **Deliverable**: GNN rollout script that feeds GNN-predicted accelerations into the Stage 3 integrator; G2 position error vs time plot; G3 energy and angular momentum drift vs time (GNN alongside Leapfrog and RK4 on shared axes); G4 generalisation test on unseen initial conditions; G5 wall-clock time vs trajectory error plot (all integrators plus GNN as a single point); completed unified comparison table; animated comparison of GNN rollout vs Leapfrog ground truth from identical initial conditions (bodies moving in real time, both trajectories overlaid), saved as `.mp4`. This animation — two trajectories starting identical and potentially diverging over time — directly visualises G2 and is the strongest single portfolio asset in the project.
- **Python skills**: coupling model inference to the existing simulation loop, benchmarking wall-clock time.
- **Validation**: G3 energy drift should fall somewhere between Leapfrog and random. If the GNN rollout conserves energy nearly as well as Leapfrog without being explicitly told to, that is a strong positive result. If not, that is an honest finding worth reporting — not a bug to fix.

### Stage 10 — Write-up & Packaging
- **Time estimate**: ~2–3 sessions (4–6 hrs)
- **Questions addressed**: none new — this stage presents the answers to **Q1–Q4 and G1–G5** (or however many you reached) as a coherent narrative, rather than producing any new evidence.
- **Goal**: package this for a CV/portfolio.
- **Deliverable**: README with the research question, method, and headline results up front (the unified comparison table and energy-drift plots are the visual anchors); clean repo structure; possibly a short report or notebook walking through the comparison table and what it shows. CV line: *"Developed a gravitational N-body simulator as a data-generating engine; trained a Graph Neural Network surrogate to learn particle dynamics from simulation trajectories; validated rollout accuracy and energy conservation against ground-truth integration across varied initial conditions."*
- **Python skills**: writing clear documentation, possibly a Jupyter notebook for narrative presentation of results.
- **Optional post-completion:** if time and inclination allow after the project is otherwise finished, the `.mp4` animations can be replaced with a browser-based visualisation using `p5.js` (a JavaScript sketching library). The position arrays are exported from Python as JSON; p5.js reads them and renders bodies with glow effects, smooth antialiasing, and a starfield background. Estimated 1–2 sessions for someone new to JavaScript. This is a cosmetic upgrade only — not a research one — and should not be attempted before everything else is complete.

---

## Part 4: Timeline Summary

The project runs two parallel tracks. The **simulator track** runs at 8 hrs/week (2 hrs × 4 sessions). The **GNN learning track** runs at 2 hrs/week (1 hr × 2 sessions) alongside it. GNN implementation takes over the 8 hr/week block after Stage 8. Stages 0 and 1 are already complete.

### Simulator track

| Stage | Sessions (2 hr) | Hours | Notes |
|---|---|---|---|
| 0 — Setup ✓ | 1 | 2 | Complete |
| 1 — Data structures ✓ | 2–3 | 4–6 | Complete |
| 2 — Force calculation | 2–3 | 4–6 | In progress |
| 3 — Integrators | 3–4 | 6–8 | |
| 4 — Simulation engine + HDF5 | 3 | 6 | +1 session for HDF5 instrumentation |
| 5 — Scenario library + data pipeline | 4–5 | 8–10 | +2 sessions for varied-IC generation |
| 6 — Analysis & metrics | 3 | 6 | |
| 7 — Visualization | 2–3 | 4–6 | |
| 8 — Experiment suite | 3–4 | 6–8 | Q1–Q4 answered; Leapfrog formally confirmed |
| **Simulator backbone (2–8)** | **~23–28** | **~46–56** | **~10 weeks from now** |
| 9 — Extensions (optional) | 3–4 | 6–8 | Q5, Q6 only; cut first if time is short |
| GNN Stage A — Graph construction | 2–3 | 4–6 | |
| GNN Stage B — Model & training | 3–4 | 6–8 | G1 answered at end |
| GNN Stage C — Rollout & comparison | 2–3 | 4–6 | G2–G5 answered; unified table complete |
| 10 — Write-up | 2–3 | 4–6 | |
| **Full project total (from Stage 2)** | **~35–45** | **~70–90** | **~16 weeks from now** |

### GNN learning track (parallel, 2 hr/week)

| Week | Resource | Notes |
|---|---|---|
| 1 | 3Blue1Brown neural network series (~1 hr) | Nodes, weights, gradient descent — concepts only |
| 2–3 | DeepMind paper: Sanchez-Gonzalez et al. 2020 (~2 hr) | Direct precedent; note what GNN inputs/outputs they use |
| 7 | Distill.pub GNN article (~1 hr) | Message passing — read close to implementation, not weeks before |
| 8 | Conceptual pre-work session here (~1 hr) | Training loop, loss, backprop — concepts only, no code |
| 11–15 | In-context learning during GNN Stages A–C | PyTorch, PyTorch Geometric, training loops learned as-you-go |

### Timeline shifts to expect

- **Stages 2–3 may run a session over estimate** while NumPy fluency is still developing — this is expected and front-loaded.
- **Stages 4 onward should speed up** relative to estimate as the patterns repeat.
- **Stage 9 is the primary buffer** — cut to Q6 only (the more interesting result) before cutting anything else.
- **GNN Stage B is the highest-risk stage** — training instability is hard to predict. Build in slack here rather than elsewhere in the GNN block.

**Approach: loose timeboxing.** These estimates are a planning anchor, not a contract. Check progress against the table roughly once a week and adjust scope (not understanding) if running consistently over.

---

## Notes on the learning approach

- At each stage, the conceptual pre-work happens here before any code is written — so the concept sticks to a reason, not just syntax. This applies equally to simulator stages and GNN stages.
- Validation is baked into every stage rather than left to the end — partly good research practice, partly because bugs caught near their source are how you actually learn to debug.
- Stages 0–6 are the simulator backbone and should not be rushed. The GNN work is only as good as the simulator it builds on — errors in the training data corrupt the model.
- **GNN learning approach**: the 2 hr/week parallel track covers concepts only (what a neural network is, what message passing does). Implementation knowledge — PyTorch syntax, training loops, PyTorch Geometric — is learned in-context during GNN Stages A–C, the same way NumPy is learned during the simulator stages. If training misbehaves and the cause is not obvious from the code, come back here rather than staying in VS Code.
- **Lab notebook**: keep a dated, append-only log (`labbook.md` in the repo) — a few bullet points per session covering what you did, decisions made and why, anything that broke and how you debugged it, and results worth remembering. Lighter than git history (which shows *what* changed but not *why*), and it directly feeds Stage 10 — you will be reconstructing the project's narrative from this rather than from memory.