"""
================================================================================
DEGP Tutorial: Contour-Seeking Directional Derivatives
================================================================================

This tutorial demonstrates an advanced strategy for Directional-Derivative GPs
(DD-GP) where the derivative information is intelligently chosen to improve the
model's understanding of a specific feature—in this case, a contour line.

Instead of using random or gradient-aligned rays, the directional derivatives
are chosen to always point *toward* a target contour line (e.g., f(x,y)=4).
This forces the model to learn the shape and location of this feature with
high fidelity, even from sparse training data.

Key concepts covered:
-   Intelligent, feature-focused directional derivative (ray) generation.
-   Using Latin Hypercube Sampling (LHS) for efficient 2D sampling.
-   The specific pointwise hypercomplex AD workflow for the `gddegp` model.
-   Advanced visualization comparing the predicted vs. true contour lines.
-   Using the Shapely library for robustly plotting clipped arrows.
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

plt.rcParams.update({'font.size': 12})


@dataclass
class ContourConfig:
    """Configuration for the Contour-Seeking DEGP tutorial."""
    n_order: int = 2
    n_bases: int = 2
    num_training_pts: int = 16
    domain_box: tuple = ((-2, 2), (-2, 2))
    contour_threshold: float = 4.0
    test_grid_resolution: int = 50
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"
    n_restarts: int = 15
    swarm_size: int = 500
    random_seed: int = 42


class ContourSeekingDEGPTutorial:
    """
    Manages and executes the tutorial on using DEGP with contour-seeking rays.
    """

    def __init__(self, config: ContourConfig, true_function: Callable, true_gradient: Callable):
        self.config = config
        self.true_function = true_function
        self.true_gradient = true_gradient
        self.training_data: Dict = {}
        self.gp_model = None
        self.params = None
        self.results: Dict = {}
        np.random.seed(config.random_seed)

    def _generate_training_data(self):
        """
        Generates training data with rays pointing towards a specific contour.
        """
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

        self.training_data = {
            'X_train': X_train,
            'y_train_list': y_train_list,
            'rays_list': rays_list,
            'rays_plot': rays_plot
        }
        print(
            f"  Extracted function values and {cfg.n_order} directional derivatives.")

    def _train_model(self):
        """Initializes and trains the GD-DEGP model."""
        print("\n" + "="*50 + "\nTraining GD-DEGP Model\n" + "="*50)
        cfg, data = self.config, self.training_data
        rays_array = np.hstack(data['rays_list'])

        self.gp_model = gddegp(
            data['X_train'], data['y_train_list'], n_order=cfg.n_order, rays_array=rays_array,
            normalize=cfg.normalize_data, kernel=cfg.kernel, kernel_type=cfg.kernel_type
        )
        print("  Model initialization: SUCCESS")

        print("  Optimizing hyperparameters...")
        self.params = self.gp_model.optimize_hyperparameters(
            n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size
        )
        print("  Hyperparameter optimization: SUCCESS")

    def _evaluate_model(self):
        """Creates a test grid and evaluates the model's performance."""
        print("\n" + "="*50 + "\nModel Prediction and Evaluation\n" + "="*50)
        cfg = self.config

        gx = np.linspace(
            cfg.domain_box[0][0] - 0.5, cfg.domain_box[0][1] + 0.5, cfg.test_grid_resolution)
        gy = np.linspace(
            cfg.domain_box[1][0] - 0.5, cfg.domain_box[1][1] + 0.5, cfg.test_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(gx, gy)
        X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

        dummy_ray = np.array([[1.0], [0.0]])
        rays_pred = np.hstack([dummy_ray] * X_test.shape[0])

        y_pred, y_var = self.gp_model.predict(
            X_test, rays_pred, self.params, calc_cov=True, return_deriv=False)
        y_true = self.true_function(X_test, alg=np)

        self.results = {
            'y_pred': y_pred, 'y_var': y_var, 'y_true': y_true,
            'X1_grid': X1_grid, 'X2_grid': X2_grid
        }
        print(f"  Evaluation complete.")

    def visualize_results(self):
        """Generates the 3-panel plot comparing contours and showing variance."""
        print("\n" + "="*50 + "\nGenerating Visualizations\n" + "="*50)
        res, cfg = self.results, self.config
        # ===================== Plotting ===============================
        threshold = 4.0
        fig, (ax_gp, ax_true, ax_var) = plt.subplots(1, 3, figsize=(19, 5),
                                                     sharex=True, sharey=True)
        y_pred = self.results['y_pred']
        y_true = self.results['y_true']
        y_var = self.results['y_var']
        X1 = self.results['X1_grid']
        X2 = self.results['X2_grid']
        X_train = self.training_data['X_train']
        rays_plot = self.training_data['rays_plot']
        Zp = y_pred.reshape(X1.shape)      # GP mean grid
        Zt = y_true.reshape(X1.shape)      # true grid
        Zv = y_var.reshape(X1.shape)       # GP variance grid

        # common filled-contour levels for mean/true
        levels = np.linspace(min(Zp.min(), Zt.min()),
                             max(Zp.max(), Zt.max()), 40)

        # variance levels (separate since variance has different scale)
        var_levels = np.linspace(Zv.min(), Zv.max(), 40)

        # ===== GP panel ==================================================
        cf1 = ax_gp.contourf(X1, X2, Zp, levels=levels, cmap="viridis")
        fig.colorbar(cf1, ax=ax_gp, fraction=0.046)
        ax_gp.set_title("GP mean")
        gp_line = ax_gp.contour(X1, X2, Zp, levels=[threshold],
                                colors="red", linewidths=1.8, linestyles="-")
        true_line = ax_gp.contour(X1, X2, Zt, levels=[threshold],
                                  colors="black", linewidths=1.8, linestyles="--")
        ax_gp.scatter(X_train[:, 0], X_train[:, 1],
                      c="red", edgecolor="k", s=35, zorder=3)
        for pt, v in zip(X_train, rays_plot):
            clipped_arrow(ax_gp, pt, v.flatten(), color="white")
        handles = [Line2D([], [], color="red", lw=2, label="GP  f=4"),
                   Line2D([], [], color="black", lw=2, ls="--", label="True f=4")]
        ax_gp.legend(handles=handles, loc="upper right")

        # ===== True panel =================================================
        cf2 = ax_true.contourf(X1, X2, Zt, levels=levels, cmap="viridis")
        fig.colorbar(cf2, ax=ax_true, fraction=0.046)
        ax_true.set_title("True function")
        ax_true.contour(X1, X2, Zt, levels=[threshold],
                        colors="black", linewidths=1.8, linestyles="--")
        ax_true.contour(X1, X2, Zp, levels=[threshold],
                        colors="red", linewidths=1.8, linestyles="-")
        ax_true.scatter(X_train[:, 0], X_train[:, 1],
                        c="red", edgecolor="k", s=35, zorder=3)
        for pt, v in zip(X_train, rays_plot):
            clipped_arrow(ax_true, pt, v.flatten(), color="white")

        # ===== Variance panel =============================================
        cf3 = ax_var.contourf(X1, X2, Zv, levels=var_levels, cmap="plasma")
        fig.colorbar(cf3, ax=ax_var, fraction=0.046)
        ax_var.set_title("GP variance")
        ax_var.scatter(X_train[:, 0], X_train[:, 1],
                       c="white", edgecolor="k", s=35, zorder=3)
        for pt, v in zip(X_train, rays_plot):
            clipped_arrow(ax_var, pt, v.flatten(), color="black")

        # ===== Axis Formatting ============================================
        for ax in (ax_gp, ax_true, ax_var):
            ax.set_xlim([-2.5, 2.5])
            ax.set_ylim([-2.5, 2.5])
            ax.set_aspect("equal")
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")

        plt.tight_layout()
        plt.show()

    def run(self):
        """Executes the complete tutorial workflow."""
        print("DEGP Tutorial: Contour-Seeking Directional Derivatives")
        print("=" * 75)

        self._generate_training_data()
        self._train_model()
        self._evaluate_model()
        self.visualize_results()

        print("\nTutorial Complete.")

# --- Helper Functions ---


def true_function(X, alg=np):
    """Quadratic-plus-linear toy function with oscillations."""
    x, y = X[:, 0], X[:, 1]
    return 3*x**2 + 2*y**2 + x + 2*alg.sin(2*x)*alg.cos(1.5*y)


def true_gradient(x, y):
    """Analytical gradient of the true function."""
    gx = 6*x + 1 + 4*np.cos(2*x)*np.cos(1.5*y)
    gy = 4*y - 3*np.sin(2*x)*np.sin(1.5*y)
    return np.array([gx, gy])


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
    config = ContourConfig()
    tutorial = ContourSeekingDEGPTutorial(config, true_function, true_gradient)
    tutorial.run()


if __name__ == "__main__":
    main()
