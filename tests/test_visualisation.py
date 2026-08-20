import numpy as np
from visualisation import fit_region
from analysis import convergence_order


import numpy as np
from visualisation import fit_region
from analysis import convergence_order


def test_fit_region_recovers_slope_despite_contaminated_ends():
    """fit_region must drop both the saturated large-h head and the round-off
    small-h floor, leaving only the straight middle so the true order is recovered."""
    h = np.geomspace(1e-1, 1e-5, 12)      # large -> small, like the real sweep
    err = h**2                            # a clean order-2 power law
    err = np.maximum(err, 1e-9)           # small-h tail clamped to a round-off floor
    err[:2] = [0.5, 0.4]                  # large-h head saturated (not-yet-asymptotic)

    region = fit_region(h, err)
    p, _ = convergence_order(h, err, fit_slice=region)
    assert abs(p - 2.0) < 0.1                     # slope recovered from the clean middle
    assert 0 not in region and 1 not in region    # saturated head was dropped


def test_fit_region_keeps_all_when_clean():
    """A pure power law with no contamination: nothing to trim, keep every point."""
    h = np.geomspace(1e-1, 1e-4, 8)
    region = fit_region(h, h**2)
    assert len(region) == len(h)
