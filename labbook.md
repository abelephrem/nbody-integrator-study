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

---

## 2026-07-13 — Stage 5 Part 2: pipeline infrastructure (metadata, nondimensionalisation, sweep spec)

### What I built
- `simulation.py`: `metadata: dict` field on `Trajectory`; `save_trajectory` writes it generically into HDF5 attrs.
- `scenarios.py`: sweep axis constants + baselines; `TrajectoryConfig`; `virial_radius`, `nondimensionalise`; `_stratified_over_intervals` (1-D LHS); `two_body_configs`, `cluster_configs`.
- `tests/test_scenarios.py` — 3 new tests. Full suite green.
- **Re-split the pipeline:** Part 2 = infrastructure (nothing runs yet); Part 3 = driver + `chaotic_cluster` extensions + cadence + dataset validation. Each Part 2 piece is tested and standalone, so committing here keeps the diff reviewable.

### Key decisions (why)
- **Metadata as a generic `dict`, not named fields** — the tags are experiment provenance, not physics, so `save_trajectory` loops the dict into attrs and `simulation.py` never learns the keys. Driver owns the schema. `field(default_factory=dict)` (not `={}`) avoids the shared-mutable-default trap. Populated by the driver, since `run_simulation` can't know the split.
- **Nondimensionalise via virial `R = GM²/2|E|`, using conserved `E` not `U`** — `U` breathes over an orbit so a length built from it fluctuates; the virial relation `⟨U⟩=2E` swaps in the conserved `E`, giving one fixed `R`. Collapses every system onto the same dimensionless equation so the GNN learns physics, not units.
- **Velocity scales by `√(GM/R)`, not `R`** — it's length/time, so `L_unit/T_unit`. Split `virial_radius` out so the two-body form `R=M²a/(m₁m₂)` is a direct test. Softening stays out of the function: as a fixed fraction of `R` it's just `ε'=f` once `R→1`, passed straight to the run.
- **Sweep crossing = one-axis-at-a-time holdout** (over full factorial). Training = cross product of trained values; each generalisation test moves *one* axis off-distribution, others at a trained baseline. Full factorial is exponential and mixes interp+extrap → uninterpretable for G4. Interp and extrap kept as separate labels (genuinely different tests).
- **Continuous `Q` via 1-D LHS over two disjoint intervals** — `uniform` clumps; stratified sampling (one jittered point per bin) covers evenly. Lay `[0.3,0.7]∪[1.1,1.5]` end-to-end, stratify, map back — the interpolation gap isn't on the concatenated line, so no training `Q` can land in it.
- **Dataset size is fine because the GNN trains on snapshots, not trajectories** — hundreds of trajectories × hundreds of snapshots × N particles ≈ 10⁵⁺ examples. Coarse parameter grid is OK: it learns a local rule, not a param lookup. `n_orient`/`n_draws` are the knobs if G1 looks data-limited.

### Validation
- `virial_radius`: two-body `R=M²a/(m₁m₂)` with unequal masses + nonzero `e` (general formula, not the `4a` special case).
- `nondimensionalise`: `Σm=1` **and** `virial_radius(nd)≈1` — the latter only holds if positions and velocities scaled consistently, so it catches the velocity-scaling mistake specifically.
- `cluster_configs`: one-axis property per split + no training `Q` in the gap `(0.7,1.1)` (validates the map-back).

### What broke
- pytest runs the file *on disk* — an unsaved buffer runs stale code (bit us as `NoneType is not iterable` from a `return` that was typed but not saved). Habit: save before every run.
- One-axis-holdout bugs caught in review before running: training block hardcoding `Q_BASELINE` instead of the sampled `Q`; a mislabeled "Q holdout" that was really the ratio holdout, leaving the real Q holdout unwritten and a stale `ratio=10` leaking two axes off-distribution.

### Next
- Stage 5 Part 3 — the driver. Extend `chaotic_cluster` (target `Q` by scaling velocities to `Q=2T/|U|`; mass-ratio definition for N>2); two-body random 3D orientation; cadence resampling (uniform stride first, then true-anomaly / event-triggered); driver (config → IC → nondimensionalise → Leapfrog → resample → tag → save); dataset-level validation (energy-drift spot-check + confirm realized `Q`/`N`/ratio).

---

## 2026-08-09 — Stage 5 Part 3: data-generation driver + dataset validation

### What I built
- `scenarios.py`: `chaotic_cluster` extended (target `Q`, `mass_ratio`); `random_orientation`; `build_initial_state`; resamplers (`resample_uniform`, `resample_true_anomaly`, `resample_events`); `generate_dataset` (driver, cadence dispatch); `validate_dataset`.
- Tests for each; full suite green. Stage 5 now code-complete (Parts 1–3).

### Key decisions (why)
- **`Q`-targeting replaces the rejection loop.** `Q=2T/|U|`, `T ∝ v²`, so scaling velocities by `λ=√(Q_target/Q_current)` hits any `Q` exactly; `Q<2 ⟺ E<0` so it's bound by construction. Dropped `vel_scale` (washed out by `λ`).
- **Cluster `mass_ratio` = one heavy body, rest equal** — crisp scalar (like two-body `m1/m2`), probes the test-particle regime; chosen over "spread" because a clean generalisation axis beats per-snapshot mass diversity.
- **Random 3D orientation for two-body only** (clusters already isotropic). Safe: gravity is rotation-invariant, so energy/`Q`/distances are unchanged. Kills the planar `z=0` artifact.
- **Cadence = post-hoc resampling** on the finished fixed-`dt` trajectory, not baked into `run_simulation`. True anomaly (two-body, denser at periapsis via uniform-in-angle); min-separation event triggers (cluster: uniform baseline + every close encounter). Kept `resample_uniform` as an ablation baseline.

### Softening finding (important)
- `validate_dataset` immediately caught **~200% energy drift on clusters** at `ε'=0.002, dt=0.01` — the validation working, not a bug. Cause: **unresolved close encounters** (crossing time `~ε'/v ≪ dt`, so Leapfrog's energy bound breaks).
- Swept `(ε', dt)` on the real heavy-body configs: clean (`~1e-3`) needs **cluster `ε'=0.05, dt=1e-3`** (a heavy body digs a deeper well); small `ε'` is erratically catastrophic.
- **Two-body wants the opposite** — small `ε'≈0.002` for `e=0.9` periapsis fidelity, and it has no encounter problem. → **per-scenario softening**, run as separate `generate_dataset` calls; each file records its own `epsilon`.

### Open question → Project
- Two-body `ε'=0.002` vs cluster `ε'=0.05` is a **25× difference in the softened force law**. Can one GNN learn both, or should `ε'` be a model input feature? G1/G4 design question — flag before GNN Stage A.

### Next
- Commit Stage 5; raise the softening question with the Project (point it here).
- Stage 6 — `analysis.py`: energy/angular-momentum drift, position error vs analytic reference, convergence-order log-log fit (Q1–Q3).

---

## 2026-08-13 — Addressing the open softening question (post-Stage 5)

### Decision
- **One ε′ for the whole dataset, unified at ε′=0.002.** ε′ lives *inside* the force law, so two values = two contradictory laws; a GNN can only learn one. 0.002 is two-body's real physical floor (periapsis fidelity at e=0.9, r_peri=0.1 ≫ ε′); cluster's 0.05 was only a dt-crutch. dt never enters the force law, so the cluster fix has to come from dt → **adaptive sub-stepping**, substeps discarded, only dt_base checkpoints kept (preserves the pre-allocated `(n_steps+1,N,3)` arrays). Supersedes the per-scenario-softening plan from 08-09.

### What I built
- `run_simulation`: opt-in `adaptive`/`n_resolve` params; a per-substep while-loop that advances dt_base in inner steps `dt_inner = √(ε′/a_max)/n_resolve`, clamped to land exactly on the checkpoint. Logs `max_n_sub` per trajectory into `metadata` (→ HDF5 attr).
- `generate_dataset`: `adaptive=True`, unified `softening=0.002`, metadata merged (`**traj.metadata`) so `max_n_sub` survives the resample.

### What broke (pre-flight earned its keep)
- The Project's **per-interval** algorithm (pick one `n_sub` from `a_max` at the interval *start*) was wrong: that a_max is **stale** — blind to encounters that develop *within* a dt_base step. Pre-flight showed drift stuck at ~250× and `max_n_sub` capping ~500. Fix: **per-substep** adaptivity, recompute a_max every substep. Cost: a fresh force eval per substep (lost the free `accelerations[i-1]` reuse — the price of correctness).
- **n_resolve=12 too coarse** (worst-case drift 0.37). Sweeping it: 24→2e-3, 48→2e-4, 100→4e-5 — falls monotonically, so it's under-resolution, *not* broken symplecticity. Chose **n_resolve=24**. Adaptive also beats uniform tiny dt on accuracy-per-compute (~1e-3 in 15s vs dt=3e-5 giving 4e-2 in 61s).

### Verified
- Worst config (mass_ratio=15, Q=0.3): drift **250 → ~1.6e-3**, ~15–23 s/traj, `max_n_sub`≈300.
- Full-pipeline smoke test (build→adaptive run→resample→save→`validate_dataset`) **passes**, `max_n_sub` read back from HDF5.
- Bonus: two-body e=0.9 substeps at periapsis too (`max_n_sub`=129) → sharper periapsis. (τ formula is softened-core-calibrated, so it slightly over-resolves the Keplerian periapsis — conservative, and N=2 is cheap.)

### Open
- Extrap `mass_ratio=15` tail drifts ~2e-2, just over `validate_dataset`'s `energy_tol=1e-2` (not corruption — 2% is fine training data). Plan: loosen cluster tol to ~3e-2 (still catches real blow-ups like the 250× baseline) rather than raise n_resolve globally for one rare seed.
- GNN learning challenges unchanged/sharpened by small ε′: heavy-tailed acceleration targets, rollout stability (G1/G4) — the adaptive wrapper is integrator-agnostic, so it can be reused on GNN rollouts later.

---

## 2026-08-16 — Stage 6: Analysis & metrics (`analysis.py`)

### Built
- `energy_drift` / `angular_momentum_drift` — relative-drift series; L drift is the norm of the *vector* difference (rotating L still registers).
- `angular_momentum` — now in the COM frame.
- `kepler_solve` — vectorised Newton–Raphson for Kepler's equation.
- `two_body_reference` — exact unsoftened orbit r_rel(t); `position_error` vs it.
- `run_convergence_sweep` + `convergence_order` — log-log slope = measured order.
- `tests/test_analysis.py` — 9 tests, all passing.

### Decisions
- `energy_drift` reads `traj.softening` — wrong ε manufactures fake drift.
- Convergence sweep runs its own clean sims (softening=0, adaptive off): the Kepler reference is unsoftened, so any ε floors the error and reads a fake-low order. `assert max_n_sub == 1` guards effective step == h.
- COM subtraction removes a silent precondition (no numerical change on our zero-momentum ICs).

### Validation
- Energy bounded `<1e-4` on a circular Leapfrog orbit; L conserved `<1e-10`.
- `kepler_solve` by round-trip; `two_body_reference` at peri/apoapsis.
- Orders within ±0.3 of theory: Euler≈1, Leapfrog≈2, RK4≈4 → **Q1 confirmed**; Q2/Q3 now computable.

### Broke
- Two transcription typos caught in review: misplaced paren in `energy_drift`; `positions(t)` instead of `positions[t]`.

### Next
- Stage 7 — visualization (trajectory, energy-drift log plots, convergence log-log with fitted slope).

---

## 2026-08-20 — Stage 7: Visualisation (`visualisation.py`)

### Built
- New module `visualisation.py` — 5 figures, each a data function + a plot function:
  - **Energy drift** (`energy_drift_series` / `plot_energy_drift`) — |ΔE/E₀| vs orbits.
  - **Convergence** (`convergence_series` / `fit_region` / `plot_convergence`) — log-log error vs h, fitted order p annotated.
  - **Phase-space** (`phase_space_series` / `plot_phase_space`) — radial plane (r, v_r), eccentric orbit.
  - **3D cluster** (`cluster_trajectory` / `plot_cluster_3d`) — rotatable Plotly HTML.
  - **Animation** (`animation_series` / `animate_integrators`) — side-by-side Euler/Leapfrog/RK4 .mp4.
- Helpers: `circular_period` (K3), `INTEGRATORS` table (name→step fn).

### Decisions
- **Diagnostics run fresh fixed-dt in-memory sims, NOT the HDF5 dataset.** The dataset is softened + adaptive + Leapfrog-only + resampled (true-anomaly/event) + short (20 units) — every one of those is wrong for measuring *integrator* behaviour. Only the 3D cluster scene uses the dataset regime (softening+adaptive ON), because there the goal is to *show* a real chaotic trajectory, not diagnose a method.
- **Energy drift:** split panels — Euler/RK4 on log-y (|drift|), Leapfrog on linear-y (signed) + inset zoom. Leapfrog's drift oscillates *through zero*, so log(|·|) spikes to −∞ at every crossing → spurious downward artefact. Linear axis shows the bounded oscillation honestly.
- **Convergence:** `fit_region` trims BOTH ends — large-h (not-yet-asymptotic, e.g. Euler) *and* small-h (round-off floor, e.g. RK4) — via longest-contiguous-run of local slopes near the median. Fit line drawn only over the fitted region (never through excluded points).
- **3D cluster:** cold collapse Q=0.5 (sub-virial → dramatic close encounters). Picked `seed=5` after a numeric probe (rmax/r95 ratio ≈ 1.0 = compact, no ejection stretching the axes). One colour per body, shared by its line and end marker (Plotly assigns per-*trace*, and the markers are separate traces, so without explicit colour the dots don't match their paths).
- **Animation:** precompute-once rule — `update(frame)` only reads `positions[frame]`, never steps an integrator. Frame `stride` thins 1801 sim frames → 600 for a sane video. Same code does live `plt.show()` (no ffmpeg) or saved `.mp4` (needs ffmpeg).

### Results
- **Q1 reconfirmed** from the plot: measured p = Euler 0.93, Leapfrog 2.00, RK4 4.20.
- **Q2/Q4 visualised three ways:** Euler drifts/spirals out, Leapfrog bounded/closed loop, RK4 slow secular drift. RK4 crosses Leapfrog's energy bound at ~15 orbits (more accurate per step, not conservative).

### Environment
- Installed **plotly 6.9.0** (+ narwhals) and **ffmpeg 9.0** (`winget install Gyan.FFmpeg`). Added plotly/narwhals to `requirements.txt`; noted ffmpeg as a system (non-pip) dep.

### Broke
- Transcription bugs caught in review/runs: phase-space `r_rel` built from `velocities` not `positions`; `v_r` used `norm(v·r)` (always ≥0) instead of the signed dot `sum(v·r)`; animation `return updated` indented *inside* the `for name` loop → only Euler animated; `figsize(12,5)` missing `=`; `Q=0.5` hardcoded instead of `Q=Q`; plot called with the function object, not `func()`.
- **ffmpeg PATH gotcha (Windows/VS Code):** winget updates the registry PATH, but VS Code terminal *tabs* inherit the PATH from when VS Code launched — a new tab is not enough, needs a full VS Code restart. First `.mp4` was truncated (`moov atom not found`) because matplotlib silently fell back to the Pillow writer when ffmpeg wasn't found. Fixed after restart → valid 20 s H.264.

### Next
- Optional: pytest guard asserting fitted orders stay ≈ 1/2/4.
- Stage 8, then GNN stages A–C.
