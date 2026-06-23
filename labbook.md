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
