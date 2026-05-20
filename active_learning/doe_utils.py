import numpy as np
from scipy.stats.qmc import LatinHypercube


def lhs_design(n_points, bounds, seed=42):
    """Latin Hypercube Sample scaled to bounds. Returns (n_points, d)."""
    d = bounds.shape[0]
    sampler = LatinHypercube(d=d, seed=seed)
    unit_samples = sampler.random(n=n_points)
    lb, ub = bounds[:, 0], bounds[:, 1]
    return lb + unit_samples * (ub - lb)
