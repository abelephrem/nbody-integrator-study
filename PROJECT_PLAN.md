# N-Body Gravitational Simulator — Project Stage Plan

## Part 0: Workflow & Roles (Two-Track System)

This project runs across two separate tools with two distinct jobs. Neither has live visibility into the other — there's no automatic sync — so this section exists to make the division explicit rather than relying on either side guessing.

### The two tracks

**Claude.ai, this Project — the "why" track**
- Role: concept explanations before code is written, stage planning and scope decisions, interpreting results against the physics, weekly progress check-ins.
- What it has: continuity of *reasoning* across sessions (this Project's memory), and on-request access to the GitHub repo (single-file fetch or full clone) when asked.
- What it does NOT have: live awareness of what's been coded, what's currently broken, or how far along a stage you are. It must be told, or asked to go fetch.

**VS Code + Claude Code extension — the "how" track**
- Role: writing, running, debugging, refactoring, testing, committing. Anything that needs to see the actual file, the actual error, the actual diagnostics.
- What it has: live file and diagnostic context this Project doesn't have.
- What it does NOT have: the conceptual throughline — why a design choice was made, how a stage connects to the research questions, the physics reasoning behind a result. That context lives here, in this document and in this Project's memory, not in the extension's session-local memory.

### The decision rule

> Does answering this need to see the actual running code/error in front of it? → **VS Code.**
> Does answering this need to connect code behavior to the physics or research questions? → **this Project.**

| Stay in VS Code | Come back to this Project |
|---|---|
| "This broadcasting throws a shape mismatch on line 12 — why?" | "Why does Leapfrog conserve energy better than RK4 even though RK4 has higher formal order?" |
| "Explain this traceback" | "Does this energy-drift plot look like what we'd expect, or is something wrong?" |
| "Refactor this to use a consistent interface" | "I'm starting Stage 3 — walk me through why a shared interface matters here" |
| "Write a pytest for this function" | "I'm starting Stage 6 — what should the convergence-order fit actually look like conceptually?" |
| Day-to-day syntax / NumPy questions tied to a specific file | Weekly check-in: progress vs. the timeline table, adjust scope if needed |
| "Why is `r` going negative here" (debugging a value) | "Is this energy spike physics or a bug?" (interpreting what it means) |

### The bridge (since there's no automatic sync)

- **Repo**: push to a public GitHub repo. This Project can fetch a single file (raw URL) or clone the whole thing on request — but only when asked, and only the state as of the last push. Treat it as pull-on-request, not live.
- **Lab notebook** (`labbook.md`, see Notes below): your dated session log doubles as the bridge. A one-line status when you come back here — e.g. *"Finished Stage 2, energy roughly conserved, took longer than estimated due to a shape-mismatch bug"* — is usually all the recalibration needed.
- **`CLAUDE.md`**: unlike this document, `CLAUDE.md` is read automatically by Claude Code at the start of *every* session — this file isn't. So `CLAUDE.md` should be short: the condensed role split and decision rule from this section, plus an explicit pointer ("see `PROJECT_PLAN.md` for the full stage plan, research questions, and timeline"). This document is committed to the repo as `PROJECT_PLAN.md` for reference — the extension *can* read it when relevant, but only `CLAUDE.md` is guaranteed to load without being asked. Worth drafting once Stage 0 starts.

### The ritual

1. **Before starting a new stage** → come here for the conceptual pre-work (as we did for symplectic integration before any code existed).
2. **Finishing a stage, or stuck on a concept (not a bug)** → come here with a one-line status.
3. **Debugging, syntax, refactoring** → stay in VS Code; don't route these here.
4. **Roughly weekly** → check progress against the Part 4 timeline table here, adjust scope if running over.

---

## Part 1: Research Questions (the "why")

Everything below should serve answering these. If a piece of code doesn't help answer one of these, it's scope creep — cut it or defer it.

**Primary question:**
> How well do different numerical integrators preserve the true physics of a gravitational N-body system over long timescales, and why?

**Sub-questions, each mapped to a measurable output:**

| # | Question | What you measure |
|---|----------|-------------------|
| Q1 | Does each integrator's *empirical* convergence order match its *theoretical* order? | Slope of log(global error) vs log(step size) |
| Q2 | Do symplectic integrators bound energy error while non-symplectic ones drift unboundedly? | Energy error ΔE/E₀ vs time, for many orbital periods |
| Q3 | Is angular momentum conserved differently across integrator types? | ΔL/L₀ vs time |
| Q4 | Is there a trade-off between formal accuracy (order) and long-term qualitative correctness? | Compare RK4 (order 4, non-symplectic) vs Leapfrog (order 2, symplectic) head-to-head |
| Q5 | How does softening affect energy conservation during close encounters? | Energy spikes at closest approach, with/without softening, across ε values |
| Q6 | At what point does chaos (not integrator error) dominate position error in the 3-body problem? | Divergence of two near-identical initial conditions vs time (Lyapunov-like estimate) |
| Q7 | What's the accuracy-per-unit-compute trade-off? | Wall-clock time vs achieved energy error, across integrators and step sizes |

Q1–Q3 are your core deliverables. Q4 is the "so what" that makes this more than a benchmark table. Q5–Q7 are what elevate it from "I implemented some integrators" to "I investigated a real numerical-physics question."

---

## Part 2: Quantification Framework

This is the analysis machinery that turns simulation output into evidence. It gets built incrementally, but it's worth seeing the whole picture now.

**Core metrics (computed for every run):**
- **Energy drift**: `(E(t) - E0) / |E0|` over time
- **Angular momentum drift**: `(L(t) - L0) / |L0|` over time
- **Position error**: deviation from an analytic reference (two-body Kepler orbit) at matching timestamps
- **Global convergence order**: fit `log(error) = p·log(h) + c` via linear regression across a range of step sizes; compare fitted `p` to theory (Euler ≈ 1, Leapfrog ≈ 2, RK4 ≈ 4)
- **Compute cost**: wall-clock time or step count, paired with achieved accuracy

**Comparison table (your headline result):**

| Integrator | Symplectic? | Theoretical order | Measured order | Energy drift behaviour | Cost per step |
|---|---|---|---|---|---|

**Secondary analyses:**
- Energy error vs softening parameter ε (heatmap or line plot)
- Trajectory divergence rate for two infinitesimally separated 3-body initial conditions (chaos signature)

---

## Part 3: Stage-by-Stage Plan

Each stage lists: **physics/numerics focus**, **deliverable**, **validation built in**, and **Python skills you'll practice**. Throughout, the pairing model is: I scaffold interfaces and explain anything mathematically subtle (e.g. vectorized force calculations), you write the core logic, we debug together. You're not meant to type out boilerplate alone — the goal is that you *understand every line*, not that you typed every line.

### Stage 0 — Environment & Project Skeleton
- **Time estimate**: ~1 session (2 hrs)
- **Questions addressed**: none directly — pure infrastructure. Nothing here is testable physics yet.
- **Goal**: working repo, not yet doing physics.
- **Deliverable**: folder structure (`bodies.py`, `integrators.py`, `forces.py`, `simulation.py`, `analysis.py`, `scenarios.py`), virtual environment, git repo with first commit, `requirements.txt`.
- **Python skills**: venv/pip basics, project structure, git fundamentals (init, add, commit, .gitignore), why modular files matter.
- **Validation**: none yet — this is just plumbing.

### Stage 1 — Core Data Structures (`bodies.py`)
- **Time estimate**: ~2–3 sessions (4–6 hrs)
- **Questions addressed**: none directly — but the array-vs-object decision here determines whether Q1–Q7 are even computationally feasible later (vectorized force/energy calcs depend on it).
- **Goal**: represent a system of bodies (mass, position, velocity) in a way that's both readable and fast.
- **Deliverable**: a `Body` representation and a `System`/`State` representation. Key design decision you'll need to actually understand: object-per-body (readable, slow) vs NumPy arrays of shape `(N, 3)` (fast, vectorizable) — and why N-body work always ends up using the latter.
- **Python skills**: `dataclasses`, type hints, NumPy array basics (shapes, broadcasting), the readability-vs-performance trade-off.
- **Validation**: unit test — can you create a 2-body system and read back correct positions/velocities?

### Stage 2 — Force Calculation (`forces.py`)
- **Time estimate**: ~2–3 sessions (4–6 hrs) — broadcasting is one of the trickier NumPy concepts; don't rush it if shape errors are confusing
- **Questions addressed**: builds the mechanism for **Q5** (softening parameter ε goes in here, even though you won't analyze its effect until Stage 9). No question is answerable yet — this just makes Q5 possible later.
- **Goal**: Newtonian gravity, vectorized across all pairs.
- **Deliverable**: pairwise force/acceleration calculator using NumPy broadcasting (no nested Python loops), with a softening parameter ε built in from day one (not bolted on later).
- **Python skills**: NumPy broadcasting for pairwise operations, avoiding O(N²) Python loops, why vectorization matters for performance.
- **Validation**: known analytic case — two bodies, check the force magnitude matches `Gm₁m₂/r²` by hand calculation.

### Stage 3 — Integrators (`integrators.py`)
- **Time estimate**: ~3–4 sessions (6–8 hrs) — three algorithms, each needs its own implementation + sanity check
- **Questions addressed**: this is the object under study for **Q1–Q4** — without multiple integrators behind a common interface, there's nothing to compare. Still no answers yet (that needs Stage 6 + 8), but this is where the *subjects* of the whole investigation get built.
- **Goal**: implement Euler (explicit), RK4, and Leapfrog/Velocity-Verlet behind a common interface.
- **Deliverable**: each integrator takes a state + forces function + step size, returns the next state. Common interface (e.g. matching function signatures, or a shared abstract base class) so `simulation.py` can swap integrators without caring which one it's using.
- **Python skills**: function design / interface consistency, optionally abstract base classes (`abc` module) — this is a good place to learn *why* a consistent interface matters, not just OOP syntax for its own sake.
- **Validation**: simple harmonic oscillator or two-body circular orbit, sanity-check each integrator visually before trusting it on harder cases.

### Stage 4 — Simulation Engine (`simulation.py`)
- **Time estimate**: ~2 sessions (4 hrs) — mostly assembly of pieces you've already built and understood
- **Questions addressed**: none answered yet, but this is the last piece of plumbing needed before any of Q1–Q7 become measurable — it's what actually produces a time history to analyze.
- **Goal**: the time-stepping loop that ties bodies + forces + integrator together and logs history.
- **Deliverable**: run a scenario for N steps, store position/velocity/energy/angular-momentum history at each step (or every k steps, for memory).
- **Python skills**: writing clean loops, pre-allocating NumPy arrays for history (vs. Python list appends — another performance lesson), basic logging/progress reporting.
- **Validation**: run a known 2-body circular orbit, confirm it visually looks like a circle and doesn't immediately blow up.

### Stage 5 — Scenario Library & Validation Suite (`scenarios.py`)
- **Time estimate**: ~2–3 sessions (4–6 hrs) — the figure-8 initial conditions are fiddly (known constants, not derived by you) but quick once you have them
- **Questions addressed**: provides the actual test cases each question needs — the two-body circular/eccentric orbits give the analytic reference **Q1** needs; the figure-8 and chaotic cluster are what **Q6** will later run on. Still no measurements yet, just the inputs.
- **Goal**: a set of well-defined initial conditions with known correct behaviour, used throughout the rest of the project.
- **Deliverable**: functions that build initial conditions for: (a) two-body circular orbit, (b) two-body eccentric orbit, (c) figure-8 three-body orbit (a known stable analytic solution — excellent validation case), (d) a small chaotic cluster (4–5 bodies).
- **Python skills**: function design with sensible defaults, returning structured data, organising "config-like" code.
- **Validation**: this stage *is* validation infrastructure — each scenario should have a known expected behaviour you can check against (e.g. figure-8 should retrace itself).

### Stage 6 — Analysis & Metrics (`analysis.py`)
- **Time estimate**: ~3 sessions (6 hrs) — the convergence-order regression is a new concept, give it room to actually click rather than just running someone else's recipe
- **Questions addressed**: this is where **Q1, Q2, Q3** become computable for the first time — convergence-order fitting directly produces the Q1 answer mechanism, energy/angular-momentum drift functions directly produce Q2/Q3. Still single-run outputs at this point, not yet the full comparison (that's Stage 8).
- **Goal**: turn raw simulation history into the quantified outputs from Part 2.
- **Deliverable**: functions for energy calculation, angular momentum calculation, position error vs analytic reference, and convergence-order fitting (log-log regression).
- **Python skills**: `numpy.polyfit` or `scipy.stats.linregress` for the convergence fit, careful array indexing/slicing, writing functions that are reusable across scenarios.
- **Validation**: known case first — e.g. confirm energy is *exactly* conserved (to floating point) for an analytically circular orbit with infinitesimal step size, before trusting the metric on harder cases.

### Stage 7 — Visualization
- **Time estimate**: ~2–3 sessions (4–6 hrs)
- **Questions addressed**: makes **Q1–Q4** legible rather than answering anything new — the convergence plot is the visual evidence for Q1, the energy-drift plot is the visual evidence for Q2/Q4. This is presentation, not discovery.
- **Goal**: make the results legible — this is what makes the project CV-worthy rather than just a folder of numbers.
- **Deliverable**: trajectory plots, energy-drift-vs-time plots (log scale), convergence plots (log-log error vs step size with fitted slope), phase space portraits for the 2-body case.
- **Python skills**: `matplotlib` fundamentals (subplots, log scales, legends, saving figures), the discipline of "every plot needs a clear question it's answering."
- **Validation**: implicit — if your plots don't visually match the physics you expect (e.g. RK4 energy drift should look roughly linear/unbounded, Leapfrog should look bounded/oscillatory), something upstream is wrong.

### Stage 8 — Experiment Suite (full comparison runs)
- **Time estimate**: ~3–4 sessions (6–8 hrs) — includes time to actually look at and interpret results, not just generate them
- **Questions addressed**: **Q1, Q2, Q3, Q4** all get directly and fully answered here — this is the stage that actually produces the comparison table and the systematic evidence, not just single-case checks.
- **Goal**: systematically run every integrator × every scenario × a range of step sizes, and generate the actual research dataset.
- **Deliverable**: a script that produces the comparison table and all key plots in one run, results saved to disk (CSV or similar) so you're not re-running expensive simulations every time you tweak a plot.
- **Python skills**: nested loops over configurations, saving/loading data (CSV via `csv` or `pandas`), separating "generate data" from "plot data" (a real research-software practice).
- **Validation**: the comparison table is itself a built-in check — results should make physical sense (e.g. measured orders roughly matching theory) before you trust them as findings.

### Stage 9 — Extensions (Q5–Q7, time permitting)
- **Time estimate**: ~4–6 sessions (8–12 hrs) total if doing all three sub-studies (~1.5–2 sessions each for Q5, Q6, Q7) — easy to scale down by doing only one or two
- **Questions addressed**: **Q5, Q6, Q7** — directly and exclusively. This is the only stage that touches these three, which is why it's the first thing to cut if time is short.
- **Goal**: the parts that elevate this from "I built an N-body sim" to "I investigated a numerical-physics question in depth."
- **Deliverable**: softening sensitivity study (Q5), chaos/divergence study on the 3-body case (Q6), accuracy-vs-cost analysis (Q7).
- **Python skills**: whatever's needed by then — likely more `matplotlib`, maybe `scipy` for curve fitting.
- **Note**: treat these as optional/stretch — better to have Stages 0–8 solid than all of Stage 9 half-done.

### Stage 10 — Write-up & Packaging
- **Time estimate**: ~2–3 sessions (4–6 hrs)
- **Questions addressed**: none new — this stage presents the answers to **Q1–Q7** (or however many you reached) as a coherent narrative, rather than producing any new evidence.
- **Goal**: package this for a CV/portfolio.
- **Deliverable**: README with the research question, method, and headline results/plots up front; clean repo structure; possibly a short report or notebook walking through the comparison table and what it shows.
- **Python skills**: writing clear documentation, possibly a Jupyter notebook for narrative presentation of results.

---

## Part 4: Timeline Summary

At 8 hrs/week (2 hrs × 4 days), here's roughly where each stage lands. These are estimates with the *pairing model already factored in* — they assume time for explanation and understanding, not just typing code, which is why some "simple" stages (like Stage 0) still get a full session. Treat them as a planning anchor, not a deadline: if a concept needs longer to actually click, that's the methodical approach working as intended, not falling behind.

| Stage | Sessions (2 hr) | Hours | Cumulative hours | Cumulative weeks |
|---|---|---|---|---|
| 0 — Setup | 1 | 2 | 2 | ~0.25 |
| 1 — Data structures | 2–3 | 4–6 | 6–8 | ~1 |
| 2 — Force calculation | 2–3 | 4–6 | 10–14 | ~1.5–1.75 |
| 3 — Integrators | 3–4 | 6–8 | 16–22 | ~2–2.75 |
| 4 — Simulation engine | 2 | 4 | 20–26 | ~2.5–3.25 |
| 5 — Scenario library | 2–3 | 4–6 | 24–32 | ~3–4 |
| 6 — Analysis & metrics | 3 | 6 | 30–38 | ~3.75–4.75 |
| 7 — Visualization | 2–3 | 4–6 | 34–44 | ~4.25–5.5 |
| 8 — Experiment suite | 3–4 | 6–8 | 40–52 | ~5–6.5 |
| **Backbone total (0–8)** | **~21–26** | **~42–52** | | **~5–6.5 weeks** |
| 9 — Extensions (optional) | 4–6 | 8–12 | 50–64 | ~6.25–8 |
| 10 — Write-up | 2–3 | 4–6 | 54–70 | ~6.75–8.75 |
| **Full project total** | **~27–35** | **~54–70** | | **~7–9 weeks** |

So: the core research result (Stages 0–8, which fully answers Q1–Q4) is roughly **6–7 weeks** at your pace. Adding the extensions and a proper write-up brings the whole thing to roughly **7–9 weeks**, call it two months.

A few things likely to shift this in practice:
- **Stages 0–2 will probably run a bit over estimate** while you're still building NumPy fluency — that's expected and front-loaded, not a sign anything's wrong.
- **Stages 4 onward should speed up** relative to estimate as the patterns repeat (you'll have written one integrator-shaped thing, one analysis-function-shaped thing, etc., before).
- If you fall behind, Stage 9 is the buffer — cut it to one sub-question (Q6, the chaos one, is probably the most interesting standalone result) rather than dropping it entirely.

**Approach: loose timeboxing.** These estimates are a planning anchor, not a contract. Check progress against the table roughly once a week and adjust scope (not understanding) if you're consistently running over — i.e. if Stage 2 is eating 3 weeks instead of 1, that's a signal to simplify the remaining scope (maybe trim Stage 9 further) rather than to rush the concept to hit the original number.

---

## Notes on the learning approach

- At each stage, I'll explain *why* a design choice is made (e.g. why arrays over objects, why a common integrator interface) before you write code — so the Python concept sticks to a reason, not just syntax.
- Validation is baked into every stage rather than left to the end — partly good research practice, partly because it means bugs get caught near their source, which is how you actually learn to debug.
- Stages 0–6 are the backbone and shouldn't be rushed. Stages 7–10 are where the "research piece" character actually becomes visible.
- **Lab notebook**: keep a dated, append-only log (e.g. `labbook.md` in the repo) — a few bullet points per session covering what you did, decisions made and why, anything that broke and how you debugged it, and results worth remembering. Lighter than git history (which shows *what* changed but not *why*), and it directly feeds Stage 10 — you'll be reconstructing the project's narrative from this rather than from memory.

---

*Next: go through this stage by stage and flag anything that needs reordering, merging, splitting, or cutting.*