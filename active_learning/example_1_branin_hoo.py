"""
Example 1 - Branin-Hoo
======================
Adaptive sequential DOE using directional derivative observations.

The reusable adaptive DOE implementation lives in adaptive_doe.py.
The project-specific figures live in plotting_utils/example_1_plotting_utils.py.
"""

from pathlib import Path

import numpy as np

from adaptive_doe import AdaptiveDirectionalGP
from plotting_utils.example_1_plotting_utils import (
    configure_plotting,
    save_eigen_spectrum_figures,
    save_final_design_figure,
    save_initial_doe_figure,
    save_initial_enrichment_figure,
    save_post_enrichment_figure,
    save_iteration_figures,
)


# ---------------------------------------------------------------------------
# Branin-Hoo test function
# ---------------------------------------------------------------------------

_a = 1.0
_b = 5.1 / (4 * np.pi**2)
_c = 5.0 / np.pi
_r = 6.0
_s = 10.0
_t = 1.0 / (8.0 * np.pi)

BRANIN_BOUNDS = np.array([[-5.0, 10.0], [0.0, 15.0]])


def branin(X):
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    term = x2 - _b * x1**2 + _c * x1 - _r
    return (_a * term**2 + _s * (1 - _t) * np.cos(x1) + _s).reshape(-1, 1)


def branin_grad(X):
    """Returns full gradient as (n, 2) array."""
    X = np.atleast_2d(X)
    x1, x2 = X[:, 0], X[:, 1]
    term = x2 - _b * x1**2 + _c * x1 - _r
    df1 = 2 * _a * term * (-2 * _b * x1 + _c) - _s * (1 - _t) * np.sin(x1)
    df2 = 2 * _a * term
    return np.column_stack([df1, df2])


def print_summary(history):
    print("\nIteration summary:")
    print(f"  {'step':>4}  {'x1':>8}  {'x2':>8}  {'sigma2_f':>10}  "
          f"{'n_derivs':>8}  {'lambda_2/lambda_1':>17}")
    for rec in history:
        ratios = rec["variance_ratios"]
        second_ratio = ratios[1] if len(ratios) > 1 else np.nan
        print(f"  {rec['step']:>4}  {rec['x_new'][0]:8.3f}  "
              f"{rec['x_new'][1]:8.3f}  {rec['mpv']:10.5f}  "
              f"{rec['n_selected']:>8}  {second_ratio:17.4f}")


if __name__ == "__main__":
    configure_plotting()

    d = BRANIN_BOUNDS.shape[0]
    al = AdaptiveDirectionalGP(
        func=branin,
        grad_func=branin_grad,
        bounds=BRANIN_BOUNDS,
        n_init=8,
        tau=0.05,
        n_iter=15,
        kernel="SE",
        kernel_type="anisotropic",
        seed=42,
    )
    al.enable_initial_derivative_enrichment(True)

    history = al.run()
    print_summary(history)

    figure_dir = Path("example_1_figures")
    figure_dir.mkdir(parents=True, exist_ok=True)
    save_initial_doe_figure(al, figure_dir)
    save_initial_enrichment_figure(al, figure_dir)
    save_post_enrichment_figure(al, figure_dir)
    save_iteration_figures(al, history, figure_dir)
    save_eigen_spectrum_figures(history, al.tau, figure_dir)
    save_final_design_figure(al, history, figure_dir, true_func=branin)
    print(f"\nSaved project figures to: {figure_dir.resolve()}")
