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

## 2026-09-04 — Stage 8 results interpretation (Claude.ai Project session)

No code written. Resolved the unexplained energy-drift scaling from Stage 8:
Leapfrog gave h⁴ on circular / h² on eccentric, RK4 gave h⁵ on both, against
an expectation of h² and h⁴ throughout.

### Leapfrog — hypothesis confirmed
Backward error analysis: the numerical trajectory is the exact solution of a
"shadow" system whose energy is conserved exactly. So measured energy =
shadow energy − h²·H₂(state), i.e. ΔE is a *difference of a state function*,
not an accumulation. That is the mechanism behind q = 0.02 — no drift term
exists in the structure, rather than merely a small one.

H₂ = α|F|² + β(v·dF/dt), and both pieces depend only on r, ṙ and angular
speed. A circular orbit is a rotation, so H₂ is constant along it and cancels
in the difference. Note the correct framing: not "the state is invariant"
(the bodies move, dF/dt ≠ 0) but "H₂ is rotationally invariant so it cannot
see orientation."

Velocity Verlet is exactly reversible ⇒ only even powers of h in the error
expansion ⇒ killing h² exposes h⁴, a jump of two orders. Ratios 16.0 and
4.0 confirm.

### RK4 — hypothesis correctly falsified, different mechanism entirely
Not reversible, no conserved shadow energy, so none of the above applies.

State = (position, velocity) in phase space; energy ∝ length². One step
multiplies the state by amplification factor R = truncated series of e^{iθ},
θ = ωh. Energy multiplier per step = |R|² = 1 − θ⁶/72 + θ⁸/576: the θ² and
θ⁴ terms cancel identically, because RK4's leading error −iθ⁵/120 is purely
imaginary — a pure phase error, which rotates the state without shortening
it. Only at θ⁶ does the error acquire a component along the state vector.

Over t/h steps: |ΔE/E₀| ≈ ω⁶h⁵t/72 ⇒ s = 5, q = 1. Scenario-independent
because it comes from RK4's Butcher coefficients, not the orbit; eccentricity
changes the prefactor, not the exponent (32.0 vs 31.9).

### Two quantitative confirmations
- Predicted θ⁶/72 sits a constant factor 2.00 below measurement at 200/400/
  800 spo. Constant across 4× in h ⇒ scaling exact; the 2 is geometric
  (two coupled modes vs one oscillator).
- |ΔE/E| = 2|ΔL/L| for RK4 circular: 5.01e-8 / 2.5e-8 = 2.00, the Kepler
  prediction for a slow inward spiral at fixed shape. Two independently
  computed quantities hitting the predicted ratio ⇒ the spiral is literal.
  Also explains RK4's h⁵ scaling in L.

### Notation for the write-up
Three exponents, all simultaneously correct; name them separately or the
results read as self-contradictory.
  p — position error vs h   (convergence order)  LF 2.00, RK4 4.37
  q — max|ΔE/E₀| vs t       (growth in time)     LF 0.02, RK4 1.00
  s — max|ΔE/E₀| vs h       (amplitude scaling)  LF 4 / 2, RK4 5

### Plan amendments
Part 1 bridge rewritten, CLAUDE.md one-liner updated. The old "RK4 would
teach corrupted physics" claim was wrong — accelerations are stored exactly,
so G1 labels are correct regardless of integrator; the integrator determines
which *states* get visited. Real argument: directional bias in the training
distribution (RK4's orbits all systematically contracting), validity of the
reference for G2/G3, exact L conservation, cost. Now explicitly recorded that
RK4's energy error is *smaller* on the eccentric case at equal step size and
at equal cost — Leapfrog wins on error character, not magnitude.

### Outstanding
Two diagnostic runs queued (see Stage 8 addendum prompt); neither blocks
Stage 9. Also flagged for Stage 9: plot |ΔE/E₀| and 2|ΔL/L₀| on shared axes
for RK4 circular — they should coincide across all 300 orbits.

---

## 2026-09-04 — Stage 8: experiment suite (build, results, extra checks, dataset)

Suite was built across earlier sessions and never written up; this covers it retrospectively,
plus two further diagnostic runs, closure work and the training dataset.

### Decision — separate generating the data from reducing it
Simulations go to HDF5 outside the repo; reduction writes small tables into `results/`. The
point is that tweaking a plot must never mean re-running expensive simulations — the Stage 8
deliverable line asked for exactly this. Built `run_path` first, before any run, so 55 files
could not end up inconsistently named.

Three sets, each shaped by the question it answers: **Set A** for Q1 (circular, three
integrators across four decades of step size, only the final state matters, so it resamples
from ~150 MB down to nothing); **Sets B and C** for Q2–Q4 (300 orbits, every step kept).
Euler is deliberately out of Set C because the crossover being tested is a Leapfrog-vs-RK4
question. The step sizes are a factor of four apart, and 300 orbits is chosen to be long
enough for the predicted crossovers to actually happen.

**Set C was specified circular-only; I added the eccentric half mid-build as an extension.**
That turned out to matter — it is what produced Leapfrog's different energy-scaling behaviour
on eccentric orbits, the anomaly the whole interpretation session exists to explain.

### Results — Q1 to Q4
- **Q1 (convergence orders): confirmed.** Euler ~1, Leapfrog 2.00, RK4 4.37 against theory of
  1, 2, 4. Euler only fits over part of the range because at large steps its error is already
  as big as the answer, so there is no order left to measure; RK4's smallest steps sit on the
  round-off floor.
- **Q2 (bounded vs drifting energy error): confirmed, but the original measure was broken.**
  Comparing the second half of a run to the first says Leapfrog is bounded and RK4 grows
  linearly — correct — but it reads **Euler backwards**, calling it bounded. The reason is that
  Euler's error saturates near 100%: it has nowhere left to grow. Fix is a *window*, not a
  different ratio — fit the growth only over the part of the run still below 1% drift.
- **Q3 (angular momentum): expectation was wrong.** I expected all three integrators to
  conserve it at round-off level. Only Leapfrog does, and it does so *exactly*, not
  approximately. RK4 and Euler both lose it. Central forces conserve angular momentum in the
  continuous equations; whether a given integrator inherits that is a separate question.
- **Q4 (accuracy vs long-term correctness): the trade-off is real but not where I expected.**
  On eccentric orbits RK4's energy error is *smaller* than Leapfrog's at the same step size,
  and still smaller at equal cost. Leapfrog wins because its error stays bounded while RK4's
  grows, so RK4 eventually crosses it. **The argument is about the character of the error, not
  its size** — recorded explicitly so the write-up cannot overclaim.

### The agreed Q2 fix existed only in a chat log
The 1%-window fix had been agreed in an earlier session and never implemented, so `main()`
regenerated every Stage 8 result *except* the agreed Q2 answer. Now in `analysis.py`.
Kept the old ratio alongside it, since it is valid whenever a run hasn't saturated.

Measuring the two eccentric cases that had never been run exposed a fitting artefact: RK4 came
out below its expected value, ordered by step size, because the early flat part of the curve
drags the fit down. Refitting later in the run recovers the expected answer. **Left the
definition alone** — a per-scenario window would look better and be less honest.

### Set D — an answer that fits neither prediction
Ran RK4 alone on a short eccentric orbit across ten step sizes to pin down how its energy
error scales with step size. Result sits **between** the two candidate answers, and the local
slope *changes across the range* — which neither candidate describes.

Not a falsification: the expected behaviour is there at the small-step end. But the scaling
varies with step size within a single run length, which is a genuine open question rather than
a measurement problem. Checked the obvious artefact — the smallest steps are contaminated by
round-off and were trimmed — and confirmed the correction is far too small to explain the
trend. No higher-precision float is available on Windows to cross-check against.

### The shape of RK4's error — re-analysis, no new runs
Set B already stored every step, and energy drift was already signed; only angular momentum
needed new code, since the existing function returns a magnitude.

**RK4's error is single-signed and never once crosses zero**, in either energy or angular
momentum, in both scenarios, and the two share a sign. On circular orbits the energy loss is
exactly twice the angular momentum loss — the textbook signature of a slow inward spiral at
fixed orbit shape. Two independently computed quantities landing on the predicted ratio means
**the spiral is literal, not a metaphor**. On eccentric orbits the loss arrives in per-orbit
events at periapsis: angular momentum steps down cleanly, energy drops then partly recovers.

### Closure work
Added the plotting functions the stored columns were written for, and a `main()` so
`python experiments.py` regenerates everything — the Stage 8 deliverable line, previously
unmet. The long-run plot independently recovers the interpretation session's exponents, refit
by code that did not know them, which is the closest thing to a blind check available here.
Euler is excluded from that plot because its drift of ~100% stretches the axis six decades.

### Bridge — dataset generated, and NOT clean
Ran the Stage 5 pipeline, Leapfrog only, 360 trajectories in about 46 minutes. **28 of 260
cluster runs exceed the energy tolerance**, the worst at 125% — which is no longer physics.

**It is not the failure mode I predicted.** I expected the heavy-mass-ratio tail; failures are
spread across mass ratio and N, and plenty are in the training split. They *do* lean towards
hot systems. And the substep count is essentially the same for passing and failing runs, so
this is not "ran out of substeps". Nothing regressed either — the Stage 5 smoke test only
covered one configuration, so this is the first end-to-end validation of the whole set.

### Other decisions
- New CSVs rather than extending existing ones — neither schema fitted.
- Plots of *saved* data live in `experiments.py`; `visualisation.py` keeps its rule of only
  plotting fresh simulations.
- **Left the fit-trimming logic untouched.** It removes the round-off floor correctly, but its
  tolerance is too wide to see Set D's bend, so Set D's answer comes from local slopes instead.
  Retuning it would risk changing Q1's published orders.
- Kept the two reduce functions separate; merging them would rewrite a published CSV's schema.

### Broke / gotchas
- The circular-period helper is the full Kepler expression, not 2π, so early cost estimates
  came out 40% high.
- Every drift series starts at exactly zero, so a naive sign test counts that as a crossing.
- CSV booleans read back as the strings `"True"`/`"False"`, both truthy — so every point looks
  as though it was included in the fit. Both plot functions take the reduce step's return value
  rather than re-reading the file.

### Outstanding
- **Dataset quality** — accept a few percent drift and address only the worst handful; or raise
  the substep resolution and regenerate the failures; or check whether the worst are physical
  ejections rather than errors. Next session.
- Shorter orbit sweeps dropped once Set D showed the scaling varies with step size within a
  single run length, which isn't what those sweeps test.
- For the Project: does that varying scaling fit the phase-error account, given the relevant
  angle isn't constant around an eccentric orbit?

---

## 2026-09-05 — Stage 8 diagnostic: is the dataset drift metric unfair to hot systems?

Read-only. Nothing regenerated, no tolerance changed. New `diagnostics_energy_metric.py`.

### The question
28 of 260 cluster runs failed the energy check, and the failures leaned towards high-Q
(hot) systems. Suspicion: the metric itself is at fault, not the integration. It scores drift
as a fraction of **total** energy, and total energy is kinetic plus potential — which nearly
cancel when Q approaches 2. So the thing we divide by shrinks towards zero, and identical
integration quality looks worse and worse. Retested against potential energy alone, which
is always well away from zero.

### What I found
- **The bias is real but not the whole story.** Rescaling amplifies error by up to 9× on the
  hottest runs. Yet after removing it the Q ordering weakens rather than disappearing — the
  hottest bin still fails ~3× more often than the middle. So something physical remains.
- **Closeness predicts drift far better than anything else.** Rank correlation with closest
  approach is about −0.8 under either metric; Q, N and mass ratio are all weak by comparison.
  **Every severe run comes inside the softening length**, against a set-wide median of ~13×
  softening.
- Nine of the ten worst runs bleed energy gradually across many encounters rather than losing
  it in one event, so this is accumulated error, not a single catastrophic step.
- No run has Q above 2, so the old metric was at least always well-defined.

### Gotchas
- Closest-approach values come from **saved snapshots**, so they are upper bounds — the true
  minimum between two saved rows could be smaller. The ordering is safe; the numbers are not.
- The hottest bin holds only 10 runs, so one run moves it 10 percentage points. It cannot
  carry an argument on its own.

### Open
- No decision taken, dataset untouched. Interpretation goes to the Project.

---

## 2026-09-05 — Stage 8 diagnostic: how well resolved is the deepest part of an encounter?

Read-only. Nothing changed. New `diagnostics_substeps.py`.

### The question
If closeness predicts drift, the natural suspect is the substep rule under-resolving deep
encounters. Substep sizes are not stored, so I reconstructed them by replaying the rule
against the saved states.

### What I found
- **The accuracy knob already existed.** `n_resolve` (default 24) was in the code all along;
  the incoming estimate of the effective resolution was off by about 4.5× because it ignored it.
- **The predicted resolution was wrong, but the ordering was right.** Predicted 5–20 steps per
  orbit at closest approach; measured 57–172. Severe runs are still about 5× less resolved
  than quiet ones, so the hypothesis survives in direction if not in magnitude.
- **The important find: resolution gets *worse* the closer bodies get.** Softening makes the
  force weaken again below ε/√2, so the rule sees a smaller acceleration and *relaxes* the
  step — while the pair's orbit keeps getting faster. Going deeper is penalised twice over.
  This is the mechanism behind everything that follows.
- **Most severe runs are trapped pairs, not single passes.** Seven of ten spend thousands of
  close orbits together across dozens of separate episodes.
- **Substepping never idles** — the criterion is tied to softening rather than actual
  separation, so it is always active and cost scales linearly with `n_resolve`.

### Gotchas
- Fitting the error curve without trimming the round-off floor gave the wrong slope and
  shifted the resolution targets by ~2×.
- Time inside an encounter has to be weighted by the gap between snapshots, since the saved
  cadence is deliberately non-uniform. A raw row count over-weights encounters.

### Open
- Untested: whether error over ~10,000 close orbits follows the short-run scaling at all.

---

## 2026-09-05 — Stage 8 diagnostic: is adaptive stepping itself the problem? (cluster_01246)

Diagnostic only. New `diagnostics_symplectic.py`, carrying its own loops so fixed-step runs
bypass the substep machinery entirely. Gate: replay matched the stored run exactly.

### The question
Leapfrog conserves energy well because of its structure, and that guarantee assumes a
**constant** step. We vary the step. So: does varying it break the guarantee?

### The answer — no
Ran the same cluster four ways: adaptive coarse, adaptive fine, and fixed-step at two sizes.
The coarse adaptive run ramps away; the fine adaptive run is **flat**. Both use the same
variable-step machinery. If varying the step were the cause, both would ramp and only the
size would differ.

**So the cause is under-resolution and adaptivity is incidental.** That is the finding that
redirected everything afterwards — towards resolving encounters better, not towards
abandoning adaptive stepping.

Supporting: halving the step quarters the error for *both* the adaptive pair and the fixed
pair, the textbook behaviour, measured before the chaotic trajectories separate.

### Also worth recording
The close pair hardens or gets ejected in **every** run, including the best-resolved one. So
that is real physics, not an integration artefact — what changes with resolution is only
which branch the trajectory takes afterwards.

### Gotchas
- Had to sample every few substeps. At the stored 100-snapshot cadence the error band is
  invisible — the pair completes ~29,000 orbits over the run.
- Plotted on log axes; on linear axes the three well-behaved runs vanish against the bad one.

---

## 2026-09-07 — Stage 8 diagnostic: does more resolution fix the other flagged runs?

Diagnostic only. Three flagged clusters at the old and new resolution. New
`diagnostics_generalisation.py`. All three gates reproduced the stored runs exactly, so the
pipeline is deterministic.

### What I found
- **All three flatten at the higher resolution**, landing two orders below the tolerance.
- **But only one of the three is a fair comparison.** Chaos means changing the step changes
  the trajectory. In one run the close pair never formed at all at the higher resolution, so
  its flatness proves nothing; a second had far less encounter exposure. Only `cluster_01038`
  forms the same binary, holds it as long, ends *more* tightly bound — and is still flat.
  The case rests on that one run.

### The find that changed the project
**Stored closest-approach values are upper bounds, and can be wrong by 11.7×.** Saved
snapshots sit on outer-step boundaries, so an approach that begins and ends inside one step
is invisible. Tracking every substep instead, `cluster_01078`'s true closest approach is
0.122ε, not the 1.43ε on file.

Two consequences. It was picked as the run that fails *despite* adequate resolution — it
isn't; it is the deepest of the three. **That test was therefore never performed, and no run
in the sample fails while genuinely well resolved.** And every earlier depth-based conclusion
rests on the same biased numbers: the direction is safe, the boundaries are not.

Also: the steps-per-orbit formula matches measurement to 0.1% except in the deepest case,
where it is 25% out — because the rule watches the largest acceleration anywhere in the
system, and a very close pair's own acceleration falls towards zero. It fails precisely in
the regime the fix targets.

### Open
- No fix implemented. A substep-level pass over all 260 runs is needed before the depth
  evidence can be trusted.

---

## 2026-09-07 — Stage 8: close-encounter recording gap, measured and fixed

Regeneration NOT done and not authorised. Changed `simulation.py` (opt-in event recording,
always-on true minimum tracking) and `scenarios.py` (new `merge_event_rows`). 46/46 tests pass.

### Decision
**Record close-encounter rows from inside the substep loop, triggered on separation, at
log-spaced shells.** Three parts, each with a reason:

- **Inside the loop**, because separations are invisible outside it. At ε=0.002 a tight pair
  orbits about seven times per outer step, so an entire encounter can begin and end between
  two saved rows. This also fixes the 11.7× under-reporting as a side effect, by tracking the
  true minimum unconditionally.
- **On separation, not acceleration.** Softened acceleration peaks at ε/√2 and *falls* below
  it, so it is two-valued in radius — a(0.2ε) ≈ a(2ε). It cannot tell an approach from a deep
  plunge, so it cannot be the trigger. Kept the existing 3ε threshold; the measured
  distribution gave no reason to move it.
- **Log-spaced shells (3ε down to 0.2ε, once in and once out), not every substep.** Row count
  is then set by geometry rather than by how many substeps the solver chose — which keeps file
  size independent of `n_resolve`, the property the 08-13 "substeps discarded" decision was
  protecting. Caps at 16 rows per encounter.

### Two findings that qualified the premise
- **There is no missing extreme acceleration.** Softening imposes a ceiling, and the existing
  dataset already reached it. What was missing was **density**: under 10% of stored rows were
  inside 3ε and under 0.5% inside 1ε.
- Hence the goal is coverage inside encounters, not reaching larger values.

### Verified
- **Recording does not touch the physics** — positions, velocities and accelerations bitwise
  identical to a control run, force evaluations and substep counts unchanged. Event rows reuse
  an acceleration the solver had already computed, so they cost nothing.
- 263 rows below 1ε on the test cluster where there were previously none. File ×4.4.
- Merged output strictly ordered with no duplicate timestamps.

### Open
- Regeneration of the 360, and `generate_dataset` is deliberately **not** wired to use any of
  this yet.

---

## 2026-09-23 — Stage 8: closest-approach row, step-size floor, old dataset deleted

Changed `simulation.py` and `scenarios.py`. 46/46 tests pass. Dataset deleted, regeneration
not yet run.

### Decision 1 — record the closest approach itself
The innermost shell sits at 0.2ε but encounters go deeper, so the deepest part of the deepest
encounters still produced no row. Rather than guess a lower floor — which would be the same
unjustified constant one step down — **record the turning point itself, so the floor stops
mattering at any depth.**

Detected by watching three consecutive substeps and taking the middle when it is closer than
both neighbours. **Saved as the real integrated substep, never an interpolation**: an
interpolated state's acceleration would not satisfy the force law, and exact accelerations are
the entire value of this dataset. The cost of taking the middle rather than the exact turning
point is about 1% in separation, because separation flattens out near its minimum.

Capture is limited to the encounter region. The tracked separation is a minimum over *all*
pairs, so it wobbles constantly as different pairs take turns being closest; ungated, the test
would fire thousands of times a run instead of about once per encounter.

### Decision 2 — a step-size floor, but only inside encounters
Below the softening radius the pair's orbital period stops shrinking while the adaptive rule
keeps *growing* the step (the 09-05 mechanism). A floor fixes that.

**Rejected the ungated version on measurement.** It is built on the period of the tightest
orbit the system could possibly form — a configuration that almost never occurs — so applied
everywhere it binds on essentially every substep and costs 4.3× for no accuracy gain. Gated to
encounters it costs nothing measurable.

**Kept anyway, as a bound rather than a working mechanism.** Justification is *not* "no run
goes that deep": true minima were known for only 3 runs out of 260, the distribution was
unmeasured, and regenerated runs take different chaotic branches. If it ever fires that is
information about encounter depth, not a failure.

### What contradicted the plan
The argument for skipping the floor rested on a "deepest approach in the set" figure taken
from **saved snapshots** — the very measurement this investigation had already shown to be an
upper bound wrong by up to 11.7×. Disregarded.

Also: the analytic estimate of where the floor would engage is unreliable, because resolution
is not monotonic in depth. Replaced it with direct measurement.

### Also decided
- `n_resolve` raised 24 → 96, passed from `generate_dataset` rather than changed as a default,
  so the Q1–Q4 experiment code is untouched.
- Old dataset deleted, and **resume capability added first** — `skip_existing` is
  presence-based, so deleting beforehand is what makes it mean "resume" rather than "silently
  keep stale files".

---

## 2026-09-25 — Stage 8: regeneration complete, stage closed

360 files regenerated overnight, about 3.5 hours. No failures, no bad timestamps, no tag
mismatches. Six times as many rows as before.

### Verified first
**Every saved acceleration matches the force law exactly** — recomputed across thousands of
rows, zero difference. This is the check the whole GNN stage rests on: a row-alignment slip in
the merge would have left every other measure looking fine while the training targets were
quietly wrong.

### Decision — new tolerance, set from the data
Switched the energy check to score against potential energy rather than total, removing the
hot-system bias found on 09-05, and kept the old number alongside so the change stays
auditable.

**Clusters 1e-2, two-body 1e-4.** The cluster value is the one that flags the *same number* of
runs as the old tolerance did — so the switch changes *which* runs are flagged, not how strict
the check is, which is the cleanest way to justify it. Anything from 5e-3 to 1e-2 flags the
same three runs, so it is not finely tuned. Two-body is four orders cleaner than clusters and
needed its own, far tighter, value; the old shared tolerance could never have fired on it.

Comparing old and new data on the same footing, failures fell by roughly an order of magnitude
and the hot-system bias is gone. Drift now correlates almost entirely with encounter depth.

### Decision — keep all 360 for training
Three runs exceed the tolerance. They are kept, because **energy drift means the trajectory
wandered, not that the targets are wrong** — their accelerations are exact like everything
else. Only the rollout comparisons in Stage C need a trustworthy trajectory, so those draw
references from the other 357.

### Decision — no finer regeneration
Tested one flagged run across four resolutions. Halving the step quarters the error, textbook
behaviour, so the error *is* resolution-limited and could be reduced. But the flagged run is
**chaotic bad luck, not a systematic fault**: at the production setting it happened to fall
into a branch reaching three times deeper with ten times the encounters. Regenerating finer
would reshuffle which runs are awkward rather than fix these, at roughly 54 hours. Not worth it.

**The new rows also make errors look worse than before.** They land on the exact instant of
peak energy error, which the old dataset structurally could never see — so the improvement
above is understated, and the new data is being marked on a harder exam.

### What the data now shows
The encounter-depth distribution exists for the first time: the deepest run is far deeper than
the old snapshot-based estimate had suggested, and the strong-force region is covered by about
half the runs rather than a lucky few. Strong-force samples are roughly 70× denser than before,
though the *maximum* barely moved — the old set already held one lucky snapshot near the
physical ceiling. Below 0.2ε coverage is genuinely thin, so expect the GNN to be weakly
constrained there; that is a property of the initial conditions, not of the recording.

### Open
- Row counts per file vary by ~500×, because event sampling deliberately favours encounters.
  Handled at the data loader by weighting each row by the time it represents — which is why
  unique, increasing timestamps were a hard requirement. Deferred to Stage A.
- Whether to raise the step-size floor so it engages at moderate depths: Project decision.
- Promote the smoke test's bitwise checks into `tests/`.

### State
**Stage 8 complete** — Q1–Q4 answered and unaffected, Leapfrog confirmed as ground-truth
generator, final training dataset regenerated and verified. Ready for GNN Stage A.
