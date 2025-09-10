"""
================================================================================
DEGP Tutorial: Directional Derivative-Enhanced Gaussian Processes (GD-DEGP)
================================================================================

This tutorial demonstrates an advanced application of DEGP that utilizes
directional derivatives instead of standard partial derivatives. This model,
a Gradient-Directional DEGP (GD-DEGP), is particularly powerful for functions
where the direction of greatest change (the gradient) is the most informative.

This script follows a specific and powerful methodology for pointwise directional
derivatives where each training point's directional derivative is handled via a
unique hypercomplex perturbation that is later isolated via truncation.

Key concepts covered:
-   Using directional derivatives ("rays") to constrain a GP.
-   Generating training points with Latin Hypercube Sampling for good coverage.
-   Calculating the gradient of the true function to define optimal ray directions.
-   Performing pointwise directional automatic differentiation using pyoti.
-   Training and visualizing a 2D GD-DEGP model.
"""

import numpy as np
import pyoti.sparse as oti
import itertools
from full_gddegp.gddegp import gddegp
from scipy.stats import qmc
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from dataclasses import dataclass
from typing import Dict, Callable
import utils

plt.rcParams.update({'font.size': 12})


@dataclass
class DirectionalDEGPConfig:
    """Configuration for the Directional DEGP tutorial."""
    n_order: int = 2
    n_bases: int = 2
    num_training_pts: int = 15
    domain_bounds: tuple = ((-5.0, 10.0), (0.0, 15.0))
    test_grid_resolution: int = 100
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"
    n_restarts: int = 15
    swarm_size: int = 200
    random_seed: int = 1


class DirectionalDEGPTutorial:
    """
    Manages and executes a tutorial on Directional Derivative-Enhanced GPs,
    replicating the exact logic for pointwise directional AD.
    """

    def __init__(self, config: DirectionalDEGPConfig, true_function: Callable, true_gradient: Callable):
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
        Generates all training data using the specific pointwise directional AD method.
        This method encapsulates the logic from the original procedural script.
        """
        print("\n" + "="*50 + "\nGenerating Training Data and Rays\n" + "="*50)
        cfg = self.config

        # 1. Generate Points and Rays (from `generate_pointwise_rays` and `gradient_angles_lhs`)
        sampler = qmc.LatinHypercube(d=cfg.n_bases, seed=cfg.random_seed)
        unit_samples = sampler.random(n=cfg.num_training_pts)
        X_train = qmc.scale(unit_samples, [b[0] for b in cfg.domain_bounds], [
                            b[1] for b in cfg.domain_bounds])

        rays_list, tag_map = [], []
        for i, (x, y) in enumerate(X_train):
            gx, gy = self.true_gradient(x, y)
            theta = np.arctan2(gy, gx)
            ray = np.array([[np.cos(theta)], [np.sin(theta)]])
            rays_list.append(ray)
            tag_map.append(i + 1)  # Tags are 1-indexed

        print(
            f"  Generated {len(X_train)} training points and gradient-aligned rays.")

        # 2. Apply Pointwise Perturbation (from `apply_pointwise_perturb`)
        # This logic is specific and crucial: it uses e(1) for every point.
        X_pert = oti.array(X_train)
        for i, ray in enumerate(rays_list):
            e_tag = oti.e(1, order=cfg.n_order)
            perturbation = oti.array(ray) * e_tag
            X_pert[i, :] += perturbation.T

        # 3. Evaluate, Truncate, and Extract Derivatives (from `generate_training_data`)
        f_hc = self.true_function(X_pert, alg=oti)
        for combo in itertools.combinations(tag_map, 2):
            f_hc = f_hc.truncate(combo)

        y_train_list = [f_hc.real.reshape(-1, 1)]
        # These indices refer to the e(1) basis used in the perturbation step.
        der_indices_to_extract = [[[1, 1]], [[1, 2]]]
        for idx in der_indices_to_extract:
            y_train_list.append(f_hc.get_deriv(idx).reshape(-1, 1))

        self.training_data = {
            'X_train': X_train,
            'y_train_list': y_train_list,
            'rays_list': rays_list
        }
        print(
            f"  Extracted function values and {len(der_indices_to_extract)} directional derivatives.")

    def _train_model(self):
        """Initializes and trains the GD-DEGP model."""
        print("\n" + "="*50 + "\nTraining GD-DEGP Model\n" + "="*50)
        cfg = self.config
        data = self.training_data

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
            cfg.domain_bounds[0][0], cfg.domain_bounds[0][1], cfg.test_grid_resolution)
        gy = np.linspace(
            cfg.domain_bounds[1][0], cfg.domain_bounds[1][1], cfg.test_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(gx, gy)
        X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
        N_test = X_test.shape[0]

        dummy_ray = np.array([[1.0], [0.0]])
        rays_pred = np.hstack([dummy_ray for _ in range(N_test)])

        y_pred_full = self.gp_model.predict(
            X_test, rays_pred, self.params, calc_cov=False, return_deriv=True)
        y_pred = y_pred_full[:N_test]  # Extract only the function predictions

        y_true = self.true_function(X_test, alg=np)

        self.results = {
            'X_test': X_test, 'y_pred': y_pred, 'y_true': y_true,
            'X1_grid': X1_grid, 'X2_grid': X2_grid,
            'nrmse': utils.nrmse(y_true.flatten(), y_pred.flatten())
        }
        print(
            f"  Evaluation complete. Final NRMSE: {self.results['nrmse']:.6f}")

    def visualize_results(self):
        """Generates the 3-panel contour plot comparing prediction, truth, and error."""
        print("\n" + "="*50 + "\nGenerating Visualizations\n" + "="*50)
        res, cfg = self.results, self.config
        X_train, rays_list = self.training_data['X_train'], self.training_data['rays_list']
        X1, X2 = res['X1_grid'], res['X2_grid']

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))

        # Panel 1: GP Prediction with Rays
        cf1 = axs[0].contourf(X1, X2, res['y_pred'].reshape(
            X1.shape), levels=30, cmap='viridis')
        axs[0].scatter(X_train[:, 0], X_train[:, 1], c='red',
                       s=40, edgecolors='k', zorder=5)
        xlim, ylim = (cfg.domain_bounds[0], cfg.domain_bounds[1])
        for pt, ray in zip(X_train, rays_list):
            clipped_arrow(axs[0], pt, ray.flatten(), length=.5,
                          bounds=(xlim, ylim), color="black")
        axs[0].set_title("GD-DEGP Prediction")
        fig.colorbar(cf1, ax=axs[0])

        # Panel 2: True Function
        cf2 = axs[1].contourf(X1, X2, res['y_true'].reshape(
            X1.shape), levels=30, cmap='viridis')
        axs[1].set_title("True Function")
        fig.colorbar(cf2, ax=axs[1])

        # Panel 3: Absolute Error
        abs_error = np.abs(res['y_pred'].flatten() -
                           res['y_true'].flatten()).reshape(X1.shape)
        abs_error_clipped = np.clip(abs_error, 1e-6, None)  # for log scale
        # For more control, define the exact level boundaries
        log_levels = np.logspace(
            np.log10(abs_error_clipped.min()),
            np.log10(abs_error_clipped.max()),
            num=100  # The number of levels you want
        )

        cf3 = axs[2].contourf(X1, X2, abs_error_clipped,
                              levels=log_levels,
                              norm=LogNorm(), cmap="magma_r")
        fig.colorbar(cf3, ax=axs[2])
        axs[2].set_title("Absolute Error (log scale)")

        for ax in axs:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_aspect("equal")

        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
                   markeredgecolor='k', markersize=8, label='Train Points'),
            Line2D([0], [0], color='black', lw=2,
                   label='Gradient Ray Direction'),
        ]
        fig.legend(handles=custom_lines, loc='lower center', ncol=2,
                   frameon=False, fontsize=12, bbox_to_anchor=(0.5, -0.02))
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()

    def run(self):
        """Executes the complete tutorial workflow."""
        print("DEGP Tutorial: Directional Derivative-Enhanced Gaussian Processes")
        print("=" * 75)

        self._generate_training_data()
        self._train_model()
        self._evaluate_model()
        self.visualize_results()

        print("\nTutorial Complete.")

# --- Helper Functions ---


def true_function(X, alg=np):
    """2D Branin function."""
    x, y = X[:, 0], X[:, 1]
    a, b, c, r, s, t = 1.0, 5.1/(4*np.pi**2), 5.0 / \
        np.pi, 6.0, 10.0, 1.0/(8*np.pi)
    return a * (y - b*x**2 + c*x - r)**2 + s*(1 - t)*alg.cos(x) + s


def true_gradient(x, y):
    """Analytical gradient of the Branin function."""
    a, b, c, r, s, t = 1.0, 5.1/(4*np.pi**2), 5.0 / \
        np.pi, 6.0, 10.0, 1.0/(8*np.pi)
    gx = 2*a*(y - b*x**2 + c*x - r)*(-2*b*x + c) - s*(1 - t)*np.sin(x)
    gy = 2*a*(y - b*x**2 + c*x - r)
    return gx, gy


def clipped_arrow(ax, origin, direction, length, bounds, color="black"):
    """Helper function to draw an arrow clipped to the plot bounds."""
    x0, y0 = origin
    dx, dy = direction * length
    xlim, ylim = bounds
    tx = np.inf if dx == 0 else (
        xlim[1] - x0)/dx if dx > 0 else (xlim[0] - x0)/dx
    ty = np.inf if dy == 0 else (
        ylim[1] - y0)/dy if dy > 0 else (ylim[0] - y0)/dy
    t = min(1.0, tx, ty)
    ax.arrow(x0, y0, dx*t, dy*t, head_width=0.25,
             head_length=0.35, fc=color, ec=color)


def main():
    """Main execution block."""
    config = DirectionalDEGPConfig()
    tutorial = DirectionalDEGPTutorial(config, true_function, true_gradient)
    tutorial.run()


if __name__ == "__main__":
    main()
