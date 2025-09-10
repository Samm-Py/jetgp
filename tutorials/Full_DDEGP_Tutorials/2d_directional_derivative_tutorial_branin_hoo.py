"""
================================================================================
DEGP Tutorial: Global Directional GP on the Branin Function (with Ray Visualization)
================================================================================

This tutorial demonstrates how to apply a Directional-Derivative Enhanced
Gaussian Process (D-DEGP) to the 2D Branin function. This model uses a
common set of directional derivatives, or "rays," applied uniformly to all
training points.

This approach is useful for functions with known global anisotropies or when you
want to enforce consistent behavior along specific axes. The training points
are generated using Latin Hypercube Sampling (LHS) for efficient coverage of
the input space.

Key concepts covered:
-   Using a **global basis of directional derivatives** for all training points.
-   **Latin Hypercube Sampling (LHS)** for efficient data generation.
-   Training the `ddegp` model, which is specialized for this structure.
-   Visualizing the GP prediction, true function, and absolute error,
    **including the representation of the global directional rays at each
    training point.**
"""

import numpy as np
import pyoti.sparse as oti
import itertools
from full_ddegp.ddegp import ddegp
import utils
from scipy.stats import qmc
from dataclasses import dataclass, field
from typing import Dict, Callable
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm


@dataclass
class BraninConfig:
    """Configuration for the Branin D-DEGP tutorial."""
    n_order: int = 1
    n_bases: int = 2
    num_training_pts: int = 16
    domain_bounds: tuple = ((-5.0, 10.0), (0.0, 15.0))
    test_grid_resolution: int = 50

    # Define the global set of directional derivatives (rays)
    rays: np.ndarray = field(default_factory=lambda: np.array([
        [np.cos(np.pi/4), np.cos(np.pi/2), np.cos(5*np.pi/4)],
        [np.sin(np.pi/4), np.sin(np.pi/2), np.sin(5*np.pi/4)]
    ]))

    # Model & Optimizer
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"
    n_restarts: int = 15
    swarm_size: int = 200
    random_seed: int = 1


class BraninGlobalDirectionalTutorial:
    """
    Manages and executes the D-DEGP tutorial on the Branin function.
    """

    def __init__(self, config: BraninConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.training_data: Dict = {}
        self.gp_model = None
        self.params = None
        self.results: Dict = {}
        np.random.seed(config.random_seed)

    def _generate_training_data(self):
        """
        Generates training data using LHS and applies global directional perturbations.
        """
        print("\n" + "="*50 + "\nGenerating Training Data\n" + "="*50)
        cfg = self.config

        sampler = qmc.LatinHypercube(d=cfg.n_bases, seed=cfg.random_seed)
        unit_samples = sampler.random(n=cfg.num_training_pts)
        X_train = qmc.scale(unit_samples, [b[0] for b in cfg.domain_bounds], [
                            b[1] for b in cfg.domain_bounds])
        print(
            f"  Generated {len(X_train)} training points using Latin Hypercube Sampling.")

        e_bases = [oti.e(i + 1, order=cfg.n_order)
                   for i in range(cfg.rays.shape[1])]
        perturbations = np.dot(cfg.rays, e_bases)

        X_pert = oti.array(X_train)
        for j in range(cfg.n_bases):
            X_pert[:, j] += perturbations[j]

        print(
            f"  Applied {cfg.rays.shape[1]} directional perturbations to all points.")

        f_hc = self.true_function(X_pert, alg=oti)
        for combo in itertools.combinations(range(1, cfg.rays.shape[1] + 1), 2):
            f_hc = f_hc.truncate(combo)

        y_train_list = [f_hc.real]
        der_indices = [[[[1, 1]], [[2, 1]], [[3, 1]]]]

        for group in der_indices:
            for sub_group in group:
                y_train_list.append(f_hc.get_deriv(sub_group).reshape(-1, 1))

        self.training_data = {
            'X_train': X_train, 'y_train_list': y_train_list, 'der_indices': der_indices}
        print(
            f"  Extracted {len(y_train_list) - 1} types of directional derivatives.")

    def _train_model(self):
        """Initializes and trains the ddegp model."""
        print("\n" + "="*50 + "\nTraining D-DEGP Model\n" + "="*50)
        cfg, data = self.config, self.training_data

        self.gp_model = ddegp(
            data['X_train'], data['y_train_list'], n_order=cfg.n_order,
            der_indices=data['der_indices'], rays=cfg.rays, normalize=cfg.normalize_data,
            kernel=cfg.kernel, kernel_type=cfg.kernel_type
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

        x_lin = np.linspace(
            cfg.domain_bounds[0][0], cfg.domain_bounds[0][1], cfg.test_grid_resolution)
        y_lin = np.linspace(
            cfg.domain_bounds[1][0], cfg.domain_bounds[1][1], cfg.test_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
        X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

        y_pred = self.gp_model.predict(
            X_test, self.params, calc_cov=False, return_deriv=False)
        y_true = self.true_function(X_test, alg=np)

        self.results = {
            'X_test': X_test, 'y_pred': y_pred, 'y_true': y_true,
            'X1_grid': X1_grid, 'X2_grid': X2_grid,
            'nrmse': utils.nrmse(y_true, y_pred)
        }
        print(
            f"  Evaluation complete. Final NRMSE: {self.results['nrmse']:.6f}")

    def visualize_results(self):
        """
        Generates a 3-panel plot showing the GP prediction, true function,
        and absolute error, including directional rays.
        """
        print("\n" + "="*50 + "\nGenerating Visualizations\n" + "="*50)
        res, cfg = self.results, self.config
        X_train = self.training_data['X_train']

        fig, axs = plt.subplots(1, 3, figsize=(19, 5), constrained_layout=True)

        # Prepare data for plotting
        gp_map = res['y_pred'].reshape(res['X1_grid'].shape)
        true_map = res['y_true'].reshape(res['X1_grid'].shape)
        abs_err = np.abs(gp_map - true_map)
        abs_err_clipped = np.clip(abs_err, 1e-8, None)

        # --- Panel 1: GP Prediction ---
        levels1 = np.linspace(gp_map.min(), gp_map.max(), 40)
        cf1 = axs[0].contourf(res['X1_grid'], res['X2_grid'],
                              gp_map, levels=levels1, cmap='viridis')
        fig.colorbar(cf1, ax=axs[0], label="GP Mean")
        axs[0].scatter(X_train[:, 0], X_train[:, 1], c='red', s=50,
                       edgecolors='black', zorder=3, label='Training Points')
        axs[0].set_title("GP Prediction")
        axs[0].legend()

        # --- Panel 2: True Function ---
        levels2 = np.linspace(true_map.min(), true_map.max(), 40)
        cf2 = axs[1].contourf(res['X1_grid'], res['X2_grid'],
                              true_map, levels=levels2, cmap='viridis')
        fig.colorbar(cf2, ax=axs[1], label="True Function Value")
        axs[1].scatter(X_train[:, 0], X_train[:, 1], c='red',
                       s=50, edgecolors='black', zorder=3)
        axs[1].set_title("True Branin Function")

        # --- Panel 3: Absolute Error ---
        log_levels = np.logspace(
            np.log10(abs_err_clipped.min()),
            np.log10(abs_err_clipped.max()),
            num=100  # The number of levels you want
        )

        cf3 = axs[2].contourf(res['X1_grid'], res['X2_grid'],
                              abs_err_clipped, levels=log_levels, norm=LogNorm(), cmap="magma_r")
        fig.colorbar(cf3, ax=axs[2], label="Absolute Error (log scale)")
        axs[2].scatter(X_train[:, 0], X_train[:, 1], c='white',
                       s=50, edgecolors='black', zorder=3)
        axs[2].set_title("Absolute Error")

        # --- Draw rays on all plots ---
        ray_length = 0.8  # Adjust for visibility
        for ax, color in zip(axs, ['white', 'white', 'black']):
            for pt in X_train:
                for i in range(cfg.rays.shape[1]):
                    direction = cfg.rays[:, i]
                    ax.arrow(pt[0], pt[1], direction[0] * ray_length, direction[1] * ray_length,
                             head_width=0.3, head_length=0.4, fc=color, ec=color, zorder=4)

        # --- Common Axes Settings ---
        for ax in axs:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_aspect("equal")
            ax.set_xlim(res['X1_grid'].min(), res['X1_grid'].max())
            ax.set_ylim(res['X2_grid'].min(), res['X2_grid'].max())

        print("  Visualization created with directional rays and error plot.")
        plt.show()

    def run(self):
        """Executes the complete tutorial workflow."""
        print("DEGP Tutorial: Global Directional GP on the Branin Function")
        print("=" * 75)

        self._generate_training_data()
        self._train_model()
        self._evaluate_model()
        self.visualize_results()

        print(f"\nTutorial Complete. Final NRMSE: {self.results['nrmse']:.6f}")

# --- Helper Function ---


def true_function(X, alg=np):
    """2D Branin function."""
    x1, x2 = X[:, 0], X[:, 1]
    a, b, c, r, s, t = 1.0, 5.1 / \
        (4.0 * np.pi**2), 5.0 / np.pi, 6.0, 10.0, 1.0 / (8.0 * np.pi)
    return a * (x2 - b * x1**2 + c * x1 - r)**2 + s * (1 - t) * alg.cos(x1) + s


def main():
    """Main execution block."""
    config = BraninConfig()
    tutorial = BraninGlobalDirectionalTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
