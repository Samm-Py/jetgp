"""
================================================================================
DEGP Tutorial: Learning a Complex Contour with Directional Derivatives
================================================================================

This tutorial demonstrates an advanced strategy for Directional-Derivative GPs
(DD-GP) where the derivative information is intelligently chosen to improve the
model's understanding of a specific, complex feature—in this case, a contour
line that encloses two separate peaks.

The directional derivatives (rays) are chosen to always point *toward* a target
contour line, forcing the model to learn the shape and location of this feature
with high fidelity, even from sparse training data.

Key concepts covered:
-   Intelligent, feature-focused ray generation for complex topologies.
-   Using Latin Hypercube Sampling (LHS) for efficient 2D sampling.
-   The specific pointwise hypercomplex AD workflow for the `gddegp` model.
-   Advanced visualization comparing the predicted vs. true contour lines.
"""

import itertools
import numpy as np
import pyoti.sparse as oti
from scipy.stats import qmc
from matplotlib import pyplot as plt
from full_gddegp.gddegp import gddegp
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from shapely.geometry import LineString, box
from dataclasses import dataclass
from typing import Dict, Callable
import utils

plt.rcParams.update({'font.size': 12})


@dataclass
class DualPeakConfig:
    """Configuration for the Contour-Seeking DEGP tutorial."""
    n_order: int = 10
    n_bases: int = 2
    num_training_pts: int = 16
    domain_box: tuple = ((-4, 4), (-4, 4))
    contour_threshold: float = 1.0
    test_grid_resolution: int = 100

    # Function shape parameters
    peak_amplitude: float = 3.0
    peak_width_sq: float = 0.8**2
    peak_center: float = 2.0

    # Model & Optimizer
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"
    n_restarts: int = 15
    swarm_size: int =100
    random_seed: int = 42


class DualPeakContourTutorial:
    """
    Manages and executes the tutorial on using DEGP with contour-seeking rays
    to learn the shape of a two-peak function.
    """

    def __init__(self, config: DualPeakConfig):
        self.config = config
        self.training_data: Dict = {}
        self.gp_model = None
        self.params = None
        self.results: Dict = {}
        np.random.seed(config.random_seed)

    def true_function(self, X, alg=np):
        """Two Gaussian bumps function defined by config parameters."""
        cfg = self.config
        x, y = X[:, 0], X[:, 1]
        g1 = cfg.peak_amplitude * \
            alg.exp(-((x - cfg.peak_center)**2 +
                    (y - cfg.peak_center)**2) / cfg.peak_width_sq)
        g2 = cfg.peak_amplitude * \
            alg.exp(-((x + cfg.peak_center)**2 +
                    (y + cfg.peak_center)**2) / cfg.peak_width_sq)
        return g1 + g2 - cfg.contour_threshold

    def true_gradient(self, x, y):
        """Analytical gradient of the two Gaussian bumps function."""
        cfg = self.config
        e1 = np.exp(-((x - cfg.peak_center)**2 +
                    (y - cfg.peak_center)**2) / cfg.peak_width_sq)
        e2 = np.exp(-((x + cfg.peak_center)**2 +
                    (y + cfg.peak_center)**2) / cfg.peak_width_sq)
        pref = -2 * cfg.peak_amplitude / cfg.peak_width_sq
        gx = pref * ((x - cfg.peak_center) * e1 + (x + cfg.peak_center) * e2)
        gy = pref * ((y - cfg.peak_center) * e1 + (y + cfg.peak_center) * e2)
        return np.array([gx, gy])

    def _generate_training_data(self):
        """Generates training data with rays pointing towards the target contour."""
        print("\n" + "="*50 +
              "\nGenerating Training Data with Contour-Seeking Rays\n" + "="*50)
        cfg = self.config

        sampler = qmc.LatinHypercube(d=cfg.n_bases, seed=cfg.random_seed)
        unit_samples = sampler.random(n=cfg.num_training_pts)
        X_train = qmc.scale(unit_samples, [b[0] for b in cfg.domain_box], [
                            b[1] for b in cfg.domain_box])

        rays_list, rays_plot, tag_map = [], [], []
        for i, (x, y) in enumerate(X_train):
            f_val = self.true_function(np.array([[x, y]]))[0]
            g = self.true_gradient(x, y)
            direction = g if f_val < cfg.contour_threshold else -g
            norm = np.linalg.norm(direction)
            v = direction / norm if norm > 1e-12 else np.array([1.0, 0.0])
            rays_list.append(v.reshape(2, 1))
            rays_plot.append(v.reshape(2, 1) * 0.4)
            tag_map.append(i + 1)

        print(
            f"  Generated {len(X_train)} training points with rays pointing toward f(x)={cfg.contour_threshold}.")

        X_pert = oti.array(X_train)
        for i, ray in enumerate(rays_list):
            e_tag = oti.e(1, order=cfg.n_order)
            perturbation = oti.array(ray) * e_tag
            X_pert[i, :] += perturbation.T

        f_hc = self.true_function(X_pert, alg=oti)
        for combo in itertools.combinations(tag_map, 2):
            f_hc = f_hc.truncate(combo)

        y_train_list = [f_hc.real.reshape(-1, 1)]
        der_indices_to_extract = [[[1, i+1]] for i in range(cfg.n_order)]
        for idx in der_indices_to_extract:
            y_train_list.append(f_hc.get_deriv(idx).reshape(-1, 1))

        self.training_data = {'X_train': X_train, 'y_train_list': y_train_list,
                              'rays_list': rays_list, 'rays_plot': rays_plot}
        print(
            f"  Extracted function values and {cfg.n_order} directional derivatives.")

    def _train_model(self):
        """Initializes and trains the GD-DEGP model."""
        print("\n" + "="*50 + "\nTraining GD-DEGP Model\n" + "="*50)
        cfg, data = self.config, self.training_data
        rays_array = np.hstack(data['rays_list'])

        self.gp_model = gddegp(data['X_train'], data['y_train_list'], n_order=cfg.n_order, rays_array=rays_array,
                               normalize=cfg.normalize_data, kernel=cfg.kernel, kernel_type=cfg.kernel_type)
        print("  Model initialization: SUCCESS")

        print("  Optimizing hyperparameters...")
        self.params = self.gp_model.optimize_hyperparameters(
            n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size)
        print("  Hyperparameter optimization: SUCCESS")

    def _evaluate_model(self):
        """Creates a test grid and evaluates the model's performance."""
        print("\n" + "="*50 + "\nModel Prediction and Evaluation\n" + "="*50)
        cfg = self.config

        gx = np.linspace(
            cfg.domain_box[0][0], cfg.domain_box[0][1], cfg.test_grid_resolution)
        gy = np.linspace(
            cfg.domain_box[1][0], cfg.domain_box[1][1], cfg.test_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(gx, gy)
        X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

        dummy_ray = np.array([[1.0], [0.0]])
        rays_pred = np.hstack([dummy_ray] * X_test.shape[0])

        y_pred = self.gp_model.predict(
            X_test, rays_pred, self.params, calc_cov=False, return_deriv=False)
        y_true = self.true_function(X_test, alg=np)

        self.results = {'y_pred': y_pred, 'y_true': y_true,
                        'X1_grid': X1_grid, 'X2_grid': X2_grid}
        print(
            f"  Evaluation complete. NRMSE: {utils.nrmse(y_true, y_pred):.6f}")

    def visualize_results(self):
        """Generates the 2-panel plot comparing the predicted and true contours."""
        print("\n" + "="*50 + "\nGenerating Visualizations\n" + "="*50)
        res, cfg = self.results, self.config
        X_train, rays_plot = self.training_data['X_train'], self.training_data['rays_plot']
        X1, X2 = res['X1_grid'], res['X2_grid']

        fig, (ax_gp, ax_true) = plt.subplots(
            1, 2, figsize=(13, 5), sharex=True, sharey=True)

        Zp, Zt = res['y_pred'].reshape(
            X1.shape), res['y_true'].reshape(X1.shape)
        levels = np.linspace(min(Zp.min(), Zt.min()),
                             max(Zp.max(), Zt.max()), 40)

        # Panel 1: GP Mean and Contour Comparison
        ax_gp.contourf(X1, X2, Zp, levels=levels, cmap="viridis")
        gp_line = ax_gp.contour(
            X1, X2, Zp, levels=[cfg.contour_threshold], colors="red", linewidths=1.8)
        true_line = ax_gp.contour(X1, X2, Zt, levels=[
                                  cfg.contour_threshold], colors="black", linewidths=1.8, linestyles="--")
        ax_gp.scatter(X_train[:, 0], X_train[:, 1],
                      c="red", edgecolor="k", s=35, zorder=3)
        for pt, v in zip(X_train, rays_plot):
            clipped_arrow(ax_gp, pt, v.flatten(), color="white")
        ax_gp.set_title("GP Mean Prediction")
        ax_gp.legend([gp_line.collections[0], true_line.collections[0]], [
                     f"GP f={cfg.contour_threshold}", f"True f={cfg.contour_threshold}"], loc="upper right")

        # Panel 2: True Function
        ax_true.contourf(X1, X2, Zt, levels=levels, cmap="viridis")
        ax_true.contour(X1, X2, Zt, levels=[
                        cfg.contour_threshold], colors="black", linewidths=1.8, linestyles="--")
        ax_true.scatter(X_train[:, 0], X_train[:, 1],
                        c="red", edgecolor="k", s=35, zorder=3)
        for pt, v in zip(X_train, rays_plot):
            clipped_arrow(ax_true, pt, v.flatten(), color="white")
        ax_true.set_title("True Function")

        for ax in (ax_gp, ax_true):
            ax.set_aspect("equal")
            ax.set(xlabel="$x_1$", ylabel="$x_2$")

        plt.tight_layout()
        plt.show()

    def run(self):
        """Executes the complete tutorial workflow."""
        print("DEGP Tutorial: Learning a Complex Contour with Directional Derivatives")
        print("=" * 75)

        self._generate_training_data()
        self._train_model()
        self._evaluate_model()
        self.visualize_results()

        print("\nTutorial Complete.")

# --- Helper Function for Plotting ---


def clipped_arrow(ax, start, vec, color="white", **kwargs):
    """
    Draw an arrow clipped to the axes limits.
    """
    x0, y0 = start
    dx, dy = vec
    x1, y1 = x0 + dx, y0 + dy

    # Axes limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create line segment and bounding box
    line = LineString([(x0, y0), (x1, y1)])
    bounds = box(xlim[0], ylim[0], xlim[1], ylim[1])
    clipped = line.intersection(bounds)

    if clipped.is_empty:
        return

    if clipped.geom_type == 'LineString':
        x_start, y_start = clipped.coords[0]
        x_end, y_end = clipped.coords[-1]
        ax.add_patch(FancyArrowPatch((x_start, y_start),
                                     (x_end, y_end),
                                     arrowstyle='->',
                                     color=color,
                                     linewidth=1.5,
                                     mutation_scale=10,
                                     **kwargs))


def main():
    """Main execution block."""
    config = DualPeakConfig()
    # The tutorial class defines the true function and its gradient internally
    tutorial = DualPeakContourTutorial(config)
    tutorial.run()


if __name__ == "__main__":
    main()
