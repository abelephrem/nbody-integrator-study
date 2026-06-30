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
- `Body` dataclass — readable format for initial conditions (mass, position, velocity).
- `SystemState` dataclass — three NumPy arrays for computation: `masses (N,)`, `positions (N,3)`, `velocities (N,3)`.
- `bodies_to_state()` — converts a list of `Body` objects into one `SystemState`.
- First unit test: `tests/test_bodies.py`. Passing.

### Key decisions (and why)
- **Two formats, one conversion.** `Body` objects are readable; arrays are fast and vectorizable. Convert once with `bodies_to_state` before the sim loop — nothing inside the loop ever sees a `Body`. Arrays are what Stage 2's pairwise force calc needs.
- **Shape is inferred, not set.** `np.array` reads shape from list nesting. A list-comprehension gathering each body's 3-element position into an outer list of N gives `(N, 3)` for free.
- **Test whole arrays, not samples.** Used `np.testing.assert_array_equal` to check every value of both bodies at once; it also reports the exact `[row, col]` of any mismatch.
- **`tests/` folder.** Expecting ~4–5 test files by project end. Rule: always run `python -m pytest` from the project root so `from bodies import ...` resolves.

### What broke
- First test run failed at velocity `[1, 0]`: expected `-1.0`, got `0.0`. The `-1.0` was body 1's *position* x, accidentally typed into the *velocity*. Code was correct; the test had the typo.
- **Lesson:** a failing test means code and test *disagree* — check which side is right before changing anything.

### Tooling
- Installed `pytest` 9.1.1 into `.venv`.

### Cleanup still open
- `requirements.txt` is UTF-16 encoded (can trip `pip install -r`) and doesn't list `pytest` yet.
- Optional: docstrings on `SystemState` / `bodies_to_state` documenting the `(N,)` and `(N,3)` shapes.

### Next
- Stage 2 — vectorized pairwise gravity in `forces.py`, with softening ε built in from the start.

### Note to self
- The Stage 0 entry above got wiped to an empty file during the `f6aa6dd` "Stage 0: update files" commit and was recovered from git history (commit `77c3d0d`). Watch that "update files" steps don't overwrite the labbook.

---

## 2026-06-30 — Stage 2: Force calculation (`forces.py`)

### What I built
- `compute_accelerations(state, G=1.0, softening=0.0)` — Newtonian gravity, fully vectorised over all pairs, no Python loops.
- Returns accelerations of shape `(N, 3)`, ready to hand straight to the Stage 3 integrators.
- Softening parameter `eps` built in from the start (for Q5 later), defaulting to 0 so the function is plain Newtonian unless asked otherwise.
- New unit test `tests/test_forces.py`: two-body analytic case. Passing.

### How the vectorisation works (so I remember the broadcasting)
- **Pairwise displacement grid.** `positions[None, :, :] - positions[:, None, :]` builds an `(N, N, 3)` array where `disp[i, j] = r_j - r_i`. The `None` inserts a length-1 axis; broadcasting stretches the two copies crosswise so the grid coordinates `(i, j)` *are* the body pair. Same data used twice, laid out perpendicular.
- **Collapse to distance.** Square and sum over `axis=2` (the x/y/z axis) → `(N, N)` squared distances. Softening (`eps^2`) is added here, *inside* the denominator.
- **`1/r^3` weights.** `dist_sq ** -1.5` does the power and reciprocal in one go.
- **Assemble.** Weight each pair by `m_j` (source mass, on the column/`j` axis) and `1/r^3`, multiply onto the displacement vectors (`weights[:, :, None]` to line up shapes), then `np.sum(..., axis=1)` to sum over `j` — each body collects the pull from every source.

### Key decisions (and why)
- **Return acceleration, not force.** Acceleration `= F/m_i`, and the target's own mass `m_i` cancels (`a_i = G*m_j/r^2`). So body `i`'s acceleration depends only on the *source* `j`'s mass. Returning acceleration directly drops `m_i` entirely and is exactly what the integrators need.
- **The diagonal trap.** Self-pairs `[i, i]` have zero displacement and zero distance, so `dist_sq ** -1.5` is `inf` when `eps=0`, and `inf * 0 = nan` poisons the whole array. Fix: `np.fill_diagonal(inv_dist_cubed, 0.0)` zeroes the self-term's weight so it drops out of the row-sum. A point mass exerts no force on itself — this removes a non-physical artifact, not a real force.

### Validation (Stage 2's built-in check)
- Two bodies on the x-axis, separation `r=2`, unequal masses (1.0 and 2.0). Checked computed accelerations against `G*m/r^2` by hand: body 0 → `[0.5, 0, 0]`, body 1 → `[-0.25, 0, 0]`. Passes via `np.testing.assert_allclose`.
- **Test design choices:** separation `r != 1` (so a "forgot to divide by r^2" bug can't pass), unequal masses (so the source-vs-target mass logic is actually exercised — the heavier body accelerates *less*).
- Used `assert_allclose` not `assert_array_equal` — the force calc involves float division and a fractional power, so exact bit-for-bit equality is too strict (unlike `test_bodies`, which just reads values back with no arithmetic).

### Added after the analytic check
- **3-body superposition test.** Three equal masses in a line on the x-axis (`-1, 0, 1`). The middle body's two pulls cancel to exactly zero; each end body is a sum of two *unequal* pulls (`1.25`). This exercises the `np.sum(..., axis=1)` superposition that the 2-body test never does (one source only). Caught a sign typo in my `expected` array — the code was right, the test was wrong (Stage 1 lesson again: check which side is right before changing code).
- **Row order follows insertion order.** `bodies_to_state([b1, b2, b3])` → row 0 is `b1`, not "the middle one." Mixed this up thinking geometrically instead of by list position.
- **Silenced an expected divide-by-zero warning.** `dist_sq ** -1.5` warns on the zero diagonal even though `fill_diagonal` neutralises it on the next line. Wrapped just that line in `with np.errstate(divide="ignore"):` — local suppression of a known, handled case, not a global silence.

### Still open / next
- Optional NaN-safety test still unwritten (single body / `softening=0`) — would guard the diagonal trap directly.
- Carry-over from Stage 1: `requirements.txt` still UTF-16 encoded and missing `pytest`.
- Next stage: Stage 3 — integrators (Euler, RK4, Leapfrog) behind a common interface.
