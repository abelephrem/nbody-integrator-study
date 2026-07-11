# Lab Notebook

Dated, append-only log. A few notes per session: what was done, why decisions were made, anything that broke.

---

## 2026-06-22 — Stage 0: Setup

- Project skeleton created (`bodies.py`, `forces.py`, `integrators.py`, `simulation.py`, `analysis.py`, `scenarios.py`).
- Set up Python 3.12.10 venv — 3.14 had package compatibility issues.
- Installed numpy, matplotlib, scipy.

---

## 2026-06-23 — Stage 1: Core data structures (`bodies.py`)

### What I built
- `Body` dataclass — readable initial conditions (mass, position, velocity).
- `SystemState` dataclass — three NumPy arrays for computation: `masses (N,)`, `positions (N,3)`, `velocities (N,3)`.
- `bodies_to_state()` — converts a list of `Body` objects into one `SystemState`.
- `tests/test_bodies.py`, passing.

### Key points
- **Two formats, one conversion.** `Body` is readable, arrays are fast/vectorizable. Convert once before the sim loop — nothing inside ever sees a `Body`.
- **Test whole arrays, not samples** (`np.testing.assert_array_equal`) — reports the exact `[row, col]` of any mismatch.
- **Run pytest from the project root** so `from bodies import ...` resolves.

### What broke
- A test failed on a velocity value — turned out the *test* had a typo (a position value typed into velocity), not the code. Lesson: a failing test means code and test *disagree* — check which side is right first.

### Next
- Stage 2 — vectorized pairwise gravity in `forces.py`, softening built in from the start.

---

## 2026-06-30 — Stage 2: Force calculation (`forces.py`)

### What I built
- `compute_accelerations(state, G=1.0, softening=0.0)` — Newtonian gravity, fully vectorised over all pairs, no Python loops. Returns `(N, 3)`.
- Softening built in from the start (for Q5), defaulting to 0.
- `tests/test_forces.py`, passing.

### Key points
- **How the vectorisation works:** `positions[None,:,:] - positions[:,None,:]` builds an `(N,N,3)` pairwise displacement grid; square+sum over the xyz axis → `(N,N)` distances (softening added here); `dist_sq ** -1.5` gives the `1/r^3` weights; weight by source mass and sum over `j`.
- **Return acceleration, not force.** The target's own mass cancels (`a_i = G·m_j/r^2`), so accel depends only on the source mass — exactly what the integrators need.
- **The diagonal trap.** Self-pairs give `inf`, and `inf*0 = nan` poisons everything. Fix: `np.fill_diagonal(inv_dist_cubed, 0.0)` — a point mass exerts no force on itself. Wrapped the power line in `np.errstate(divide="ignore")` to silence the known, handled warning.

### Validation
- Two bodies, separation `r=2`, unequal masses — checked accel against `G·m/r^2` by hand (`0.5`, `-0.25`). Used `assert_allclose` (float arithmetic, not exact equality).
- Added a 3-body superposition test (middle body's pulls cancel to 0; end bodies sum two unequal pulls to `1.25`) — exercises the `sum over j` the 2-body case doesn't.

### Next
- Stage 3 — integrators (Euler, RK4, Leapfrog) behind a common interface.

---

## 2026-07-03 — Stage 3: Integrators (`integrators.py`)

### What I built
- `euler_step`, `leapfrog_step`, `rk4_step`, all behind one interface: `step(state, forces_func, dt) -> new SystemState`.
- `tests/test_integrators.py` — 3 tests on a two-body circular orbit, passing.
- `validate_integrators_plot.py` — three trajectory panels, saved to `figures/`.

### Key points
- **`forces_func` passed in, not called directly.** Caller pre-binds `G`/softening via a lambda, so integrators stay ignorant of the physics — lets a GNN swap in later with no code change.
- **Each step returns a fresh `SystemState`, never mutates the input.** RK4 evaluates forces at intermediate positions, so mutating would corrupt values still needed.
- **Euler is deliberately the explicit (bad) version** — both updates use old values, so it spirals. Kept as the negative control.
- **Leapfrog / RK4 need an intermediate throwaway state** to evaluate `forces_func` at new positions (accel is position-only, so the velocity in it doesn't matter).

### Validation
- Derived circular-orbit IC from force balance: `v = sqrt(G·m/(2d))` → bodies at `x=±1`, `v=±0.5`, period `T=4π`.
- Over 5 orbits: Leapfrog & RK4 hold radius to `<1e-3`; Euler spirals out to ~1.031. Plot shows Euler spiral vs clean Leapfrog/RK4 circles.

### Next
- Stage 4 — simulation engine (`simulation.py`) + HDF5 output (`h5py` needs installing first).

---

## 2026-07-04 — Stage 4: Simulation engine (`simulation.py`)

### What I built
- `Trajectory` dataclass — a whole-run bundle: 5 arrays (`positions`, `velocities`, `accelerations` `(T,N,3)`; `times` `(T,)`; `masses` `(N,)`) + 7 scalar metadata fields.
- `run_simulation(...)` — pre-allocates history arrays, records row 0 as the initial state, steps/records `n_steps` times, returns a `Trajectory`.
- `save_trajectory(traj, path)` — HDF5 write: 7 attrs, big three datasets gzip-compressed, `masses`/`times` plain.
- `tests/test_simulation.py` — 4 tests. Full suite 10/10.

### Key points
- **Return, don't write.** `run_simulation` returns arrays; `save_trajectory` is separate — keeps the loop testable without disk.
- **Row 0 = initial state**, so `n_steps + 1` rows and `times[0] == 0`. Record *before* stepping; uniform "step-then-record" loop after.
- **`forces_func` closure** bakes `G`/softening into a one-arg function the integrators expect. **`times[i] = i*dt`** (not accumulated) to avoid float drift.
- **Name mapping:** `softening` → attr `epsilon`; integrator name from `__name__` with `_step` stripped.

### Validation
- Shapes correct; row 0 exactly equals initial positions; circular orbit stays bounded over 5 periods (`<1e-3`); HDF5 round-trip (save → reopen via `tmp_path`) with right shapes and attrs.

### What broke / env
- Ran on **Python 3.14.6** (`C:\Python314`, no active venv) — Stage 0's 3.12 venv note no longer matches. `h5py`/`pytest` were missing, installed into 3.14 user site; all tests now pass. TODO: pick a canonical env.
- Added `h5py==3.16.0` to `requirements.txt`.

### Next
- Stage 5 — scenario library + data generation pipeline (`scenarios.py`).

---

## 2026-07-04 — GNN learning track: 3Blue1Brown Neural Networks series

### What I did
- Watched all 4 videos (~1 hr) — neuron/layer basics, gradient descent, backprop intuition, backprop calculus.

### Key points
- Core idea that stuck: backprop = repeatedly applying the chain rule backward through the network, each layer only needs the local derivative and the gradient handed to it from the layer after.
- Gradient descent = loss as a landscape, weights move opposite the gradient to go downhill.

### Still fuzzy
- Exactly how gradients combine when a value/weight feeds into more than one path — expecting this to click properly in micrograd when I have to code it by hand.

### Next
- Karpathy micrograd, split across two sessions per Claude's advice.

---

## 2026-07-04 — GNN learning track: Karpathy micrograd (Part 1, up to ~1:22:27+)

### What I did
- Built the `Value` class by hand: wraps a number, remembers the op/children that made it, visualized with `draw_dot`. Did backprop manually on a small expression and a full neuron (`x1*w1 + x2*w2 + b -> tanh -> o`), then automated it by giving each op (`+`, `*`, `tanh`) a `_backward()` local-derivative rule.

### What I learnt
- Each node's gradient = local derivative x incoming downstream gradient — chain rule, node by node (matches 3B1B calculus video).
- Per-op rules: `+` passes grad through unchanged; `*` gives each input `other.data x incoming grad`; `tanh` scales by `(1 - tanh(x)^2)`.
- Grads must use `+=` not `=`: a `Value` used on more than one path has to accumulate contributions from all of them (multivariable chain rule); `=` silently breaks this.
- Topological sort puts every node before its dependents; walking it in reverse and calling `_backward()` backprops the whole graph in one `o.backward()` call — this is what PyTorch's `.backward()` does under the hood (`Value` = scalar toy version of a tensor), the foundation for the later `torch`/`torch_geometric` GNN stages.

### Still fuzzy
- Was coding along while watching — don't fully remember the syntax, but mostly get the logic.

### Next
- Second half of the micrograd video (neuron/layer/MLP classes, training on a toy dataset), as a separate session.

---

## 2026-07-11 — Stage 5 Part 1: named scenarios + validation suite (`scenarios.py`, `analysis.py`)

### What I built
- `analysis.py`: `total_energy(state, G, softening)`, `angular_momentum(state)`.
- `scenarios.py`: `_kepler_two_body` + `two_body_circular`/`two_body_eccentric` wrappers, `figure_eight`, `_zero_com`, `chaotic_cluster`.
- `tests/test_scenarios.py` — 4 tests. Full suite 14/14.
- **Split Stage 5 in two:** Part 1 = named scenarios + validation (here); Part 2 = the sweep/pipeline. The pipeline's new infra (cadence resampling, metadata tagging, 3D orientation) all lives in Part 2, so Part 1 stands alone.

### Key decisions (why)
- **`total_energy`/`angular_momentum` pulled forward into `analysis.py` (a Stage 6 file)** — the cluster generator rejection-samples on `E<0`, so it can't be built without them. Put in their eventual home to avoid a later duplicate that could drift.
- **Softening-consistent potential:** `U ∝ -1/sqrt(r²+ε²)` (`**-0.5`) is the potential of the softened force in `forces.py` (`**-1.5`). Mismatch would make the `E<0` check inconsistent with the actual dynamics. (Plummer softening.)
- **Two-body built at periapsis:** there `v_r=0`, so velocity is purely tangential (`[0, v_peri, 0]`, `v_peri` from vis-viva) — no vector decomposition, and dodges Kepler's transcendental equation. Mass split `r1=(m2/M)r`, `r2=-(m1/M)r` preserves `r=r1-r2` and puts COM at origin *at rest* (period-return test needs no drift).
- **`figure_eight` = published Moore 1993 IC, full 8 decimals** — orbit is stable but the ICs are knife-edge; truncating breaks the retrace. Wrote `r2=-r1`, `v1=v2=-v3/2` via negation to guarantee zero momentum and avoid a mistyped digit.
- **Cluster via rejection sampling:** can't easily *construct* a bound random cluster but can cheaply *test* it (`E<0`) — draw Gaussian pos/vel, `_zero_com`, keep if bound. `vel_scale<pos_scale` biases toward bound so the loop accepts fast; seeded for reproducibility.
- **Everything embedded 3D with `z=0`** — `SystemState` is 3D; two-body and figure-8 are planar. One code path.

### Validation (differs per scenario)
- **Two-body — period return** over `T=2π√(a³/GM)`. Eccentric is the sharp test: `T` is independent of `e`, so a wrong `v_peri` closes an orbit at the wrong period and fails. Passed at `atol=1e-2`, no tuning.
- **figure-8 — full-period retrace** (bodies swap at sub-period symmetries, so only a full `T` returns each to its own start). Finer `dt=1e-4`; `T≈6.3259` is empirical, so `1e-2` sits near the error floor.
- **Cluster — no retrace exists:** assert `E<0` + `np.all(np.isfinite(positions))` only. Deliberately *not* energy conservation — chaotic close encounters spike it transiently (Q5 softening story), so a tight assert would be flaky; drift is quantified in Part 2.

### What broke / env
- `.venv` was missing `h5py` (the Stage 4 env drift — that stage ran on a 3.14 user site). Importing `run_simulation` failed at its top-level `import h5py`. Fixed via `pip install -r requirements.txt` (already pinned). Env still not canonical — nail down before Part 2's longer runs.

### Next
- Stage 5 Part 2 — data pipeline. Resolve first: (1) snapshot cadence as post-hoc resampling (`run_simulation` only records fixed `dt`; `save_every_k` is a no-op); (2) metadata dict on `Trajectory`/`save_trajectory` for scenario-type + train/interp/extrap tags; (3) 3D orientation + non-dimensionalization (virial `R=GM²/2|E|`).
