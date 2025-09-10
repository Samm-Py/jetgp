"""
================================================================================
DEGP Tutorial: 2D GP with a Single Global Directional Derivative
================================================================================

This tutorial demonstrates a specific application of a Directional-Derivative
Enhanced Gaussian Process (D-DEGP). Unlike models that use multiple or unique
rays for each point, this example uses a single, global directional derivative
that is applied uniformly to all training points.

This approach is effective for introducing a general directional constraint to
the model without the complexity of pointwise or multi-directional setups.

Key concepts covered:
-   Using a single, global directional derivative for all training points.
-   Applying a uniform hypercomplex perturbation.
-   Training the `ddegp` model with a simple directional structure.
-   Visualizing the GP prediction, true function, and the relative error.
"""

import numpy as np
import pyoti.sparse as oti
import itertools
from full_ddegp.ddegp import ddegp
import utils
import plotting_helper
from matplotlib import pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Callable


@dataclass
class SimpleDirectionalConfig:
    """Configuration for the Single Directional DEGP tutorial."""
    n_order: int = 1
    n_bases: int = 2
    num_pts_per_axis: int = 3
    domain_bounds: tuple = ((-1, 1), (-1, 1))
    test_grid_resolution: int = 100

    # Define a single, global directional derivative (ray)
    # This example uses a ray pointing purely in the x₂ direction.
    ray: np.ndarray = field(default_factory=lambda: np.array([[1.0], [1.0]]))

    # Model & Optimizer
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"
    n_restarts: int = 15
    swarm_size: int = 100
    random_seed: int = 0


class SimpleDirectionalDEGPTutorial:
    """
    Manages and executes a tutorial for a DEGP with a single global
    directional derivative.
    """

    def __init__(self, config: SimpleDirectionalConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.training_data: Dict = {}
        self.gp_model = None
        self.params = None
        self.results: Dict = {}
        np.random.seed(config.random_seed)

    def _generate_training_data(self):
        """
        Generates training data by applying a single directional perturbation
        to all training points.
        """
        print("\n" + "="*50 + "\nGenerating Training Data\n" + "="*50)
        cfg = self.config

        # 1. Create Training Grid
        x_vals = np.linspace(
            cfg.domain_bounds[0][0], cfg.domain_bounds[0][1], cfg.num_pts_per_axis)
        y_vals = np.linspace(
            cfg.domain_bounds[1][0], cfg.domain_bounds[1][1], cfg.num_pts_per_axis)
        X_train = np.array(list(itertools.product(x_vals, y_vals)))

        # 2. Apply the Global Directional Perturbation
        perturbation = np.dot(cfg.ray, oti.e(1, order=cfg.n_order))

        X_pert = oti.array(X_train)
        for j in range(cfg.n_bases):
            X_pert[:, j] += perturbation[j, 0]

        print(
            f"  Applied a single directional perturbation to all {len(X_train)} training points.")

        # 3. Evaluate and Extract Data
        f_hc = self.true_function(X_pert, alg=oti)
        y_train_list = [f_hc.real]

        # We are only using the first derivative along our single ray (basis e(1))
        der_indices = [[[[1, 1]]]]
        for group in der_indices:
            for sub_group in group:
                y_train_list.append(f_hc.get_deriv(sub_group).reshape(-1, 1))

        self.training_data = {
            'X_train': X_train, 'y_train_list': y_train_list, 'der_indices': der_indices}
        print(f"  Extracted function values and 1 directional derivative per point.")

    def _train_model(self):
        """Initializes and trains the ddegp model."""
        print("\n" + "="*50 + "\nTraining D-DEGP Model\n" + "="*50)
        cfg, data = self.config, self.training_data

        self.gp_model = ddegp(
            data['X_train'], data['y_train_list'], n_order=cfg.n_order,
            der_indices=data['der_indices'], rays=cfg.ray, normalize=cfg.normalize_data,
            kernel=cfg.kernel, kernel_type=cfg.kernel_type
        )
        print("  Model initialization: SUCCESS")

        print("  Optimizing hyperparameters...")
        self.params = self.gp_model.optimize_hyperparameters(
            n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size
        )
        print("  Hyperparameter optimization: SUCCESS")

    def _evaluate_model(self):
        """Creates a test grid, evaluates performance, and calculates relative error."""
        print("\n" + "="*50 + "\nModel Prediction and Evaluation\n" + "="*50)
        cfg = self.config

        gx = np.linspace(
            cfg.domain_bounds[0][0] - 0.5, cfg.domain_bounds[0][1] + 0.5, cfg.test_grid_resolution)
        gy = np.linspace(
            cfg.domain_bounds[1][0] - 0.5, cfg.domain_bounds[1][1] + 0.5, cfg.test_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(gx, gy)
        X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

        y_pred = self.gp_model.predict(X_test, self.params, calc_cov=False)
        y_true = self.true_function(X_test, alg=np)

        # Calculate relative error, handling potential division by zero
        nonzero_mask = y_true.flatten() != 0
        rel_err = np.abs(y_pred.flatten() - y_true.flatten())
        rel_err[nonzero_mask] = np.abs(
            (y_pred.flatten()[nonzero_mask] - y_true[nonzero_mask].flatten()) / y_true[nonzero_mask].flatten())

        self.results = {
            'X_test': X_test, 'y_pred': y_pred, 'y_true': y_true,
            'X1_grid': X1_grid, 'X2_grid': X2_grid, 'rel_err': rel_err,
            'nrmse': utils.nrmse(y_true, y_pred)
        }
        print(
            f"  Evaluation complete. Final NRMSE: {self.results['nrmse']:.6f}")

    def visualize_results(self):
        """Generates the 3-panel plot comparing prediction, truth, and relative error."""
        print("\n" + "="*50 + "\nGenerating Visualizations\n" + "="*50)
        res, cfg = self.results, self.config
        X_train = self.training_data['X_train']
        X1, X2 = res['X1_grid'], res['X2_grid']

        fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        # Panel 1: GP Prediction
        cf1 = axs[0].contourf(X1, X2, res['y_pred'].reshape(
            X1.shape), levels=30, cmap='viridis')
        axs[0].scatter(X_train[:, 0], X_train[:, 1],
                       c='red', s=40, label='Train Points')
        fig.colorbar(cf1, ax=axs[0])
        axs[0].set_title("GP Prediction")

        # Draw arrows indicating the single global direction
        xlim, ylim = axs[0].get_xlim(), axs[0].get_ylim()
        for pt in X_train:
            clipped_arrow(axs[0], pt, cfg.ray.flatten(),
                          length=0.5, bounds=(xlim, ylim))

        # Panel 2: True Function
        cf2 = axs[1].contourf(X1, X2, res['y_true'].reshape(
            X1.shape), levels=30, cmap='viridis')
        fig.colorbar(cf2, ax=axs[1])
        axs[1].set_title("True Function")

        # Panel 3: Relative Error
        cf3 = axs[2].contourf(X1, X2, res['rel_err'].reshape(
            X1.shape), levels=30, cmap='magma')
        fig.colorbar(cf3, ax=axs[2])
        axs[2].set_title("Absolute Relative Error")

        for ax in axs:
            ax.set(xlabel="$x_1$", ylabel="$x_2$", aspect="equal")

        plt.show()

    def run(self):
        """Executes the complete tutorial workflow."""
        print("DEGP Tutorial: 2D GP with a Single Global Directional Derivative")
        print("=" * 75)

        self._generate_training_data()
        self._train_model()
        self._evaluate_model()
        self.visualize_results()

        print("\nTutorial Complete.")

# --- Helper Functions ---


def true_function(X, alg=np):
    """Simple quadratic function for demonstration."""
    x1, x2 = X[:, 0], X[:, 1]
    return 3*x1**2 + 2*x2**2 + x1


def clipped_arrow(ax, origin, direction, length, bounds):
    """Draw an arrow clipped at axis bounds."""
    x0, y0 = origin
    dx, dy = direction * length
    xlim, ylim = bounds
    tx = np.inf if dx == 0 else (
        xlim[1] - x0)/dx if dx > 0 else (xlim[0] - x0)/dx
    ty = np.inf if dy == 0 else (
        ylim[1] - y0)/dy if dy > 0 else (ylim[0] - y0)/dy
    t = min(.5, tx, ty)
    ax.arrow(x0, y0, dx*t, dy*t, head_width=0.1,
             head_length=0.15, fc='white', ec='white')


def main():
    """Main execution block."""
    config = SimpleDirectionalConfig()
    tutorial = SimpleDirectionalDEGPTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
