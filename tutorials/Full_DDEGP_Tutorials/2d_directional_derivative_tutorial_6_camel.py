"""
================================================================================
DEGP Tutorial: Global Directional Derivatives for 2D Functions
================================================================================

This tutorial demonstrates an advanced DEGP model that uses a common set of
directional derivatives, or "rays," applied uniformly to all training points.
Unlike pointwise directional models where each point has its own unique ray,
this approach constrains the GP's behavior along the same directions
everywhere in the input space.

This is useful for functions with known global anisotropies or when you want to
enforce consistent behavior along specific axes (e.g., spatial or temporal).

Key concepts covered:
-   Using a **global basis of directional derivatives** for all training points.
-   Applying simultaneous hypercomplex perturbations for multiple directions.
-   Training the `ddegp` model, which is specialized for this structure.
-   Visualizing the resulting 2D function approximation.
"""

import numpy as np
import pyoti.sparse as oti
import itertools
from full_ddegp.ddegp import ddegp
import utils
import plotting_helper
from dataclasses import dataclass, field
from typing import List, Dict, Callable


@dataclass
class GlobalDirectionalConfig:
    """Configuration for the Global Directional DEGP tutorial."""
    n_order: int = 2
    n_bases: int = 2
    num_pts_per_axis: int = 4
    domain_bounds: tuple = ((-1, 1), (-1, 1))
    test_grid_resolution: int = 25

    # Define the global set of directional derivatives (rays)
    # Each column is a 2D unit vector [cos(theta), sin(theta)]
    rays: np.ndarray = field(default_factory=lambda: np.array([
        [np.cos(np.pi/4), np.cos(np.pi/2), np.cos(5*np.pi/4)],
        [np.sin(np.pi/4), np.sin(np.pi/2), np.sin(5*np.pi/4)]
    ]))

    # Model & Optimizer
    normalize_data: bool = True
    kernel: str = "RQ"
    kernel_type: str = "isotropic"
    n_restarts: int = 15
    swarm_size: int = 250
    random_seed: int = 0


class GlobalDirectionalDEGPTutorial:
    """
    Manages and executes the tutorial for a DEGP with a global set of
    directional derivatives.
    """

    def __init__(self, config: GlobalDirectionalConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.training_data: Dict = {}
        self.gp_model = None
        self.params = None
        self.results: Dict = {}
        np.random.seed(config.random_seed)

    def _generate_training_data(self):
        """
        Generates training data by applying the global set of directional
        perturbations to all training points.
        """
        print("\n" + "="*50 + "\nGenerating Training Data\n" + "="*50)
        cfg = self.config

        # 1. Create Training Grid
        x_vals = np.linspace(
            cfg.domain_bounds[0][0], cfg.domain_bounds[0][1], cfg.num_pts_per_axis)
        y_vals = np.linspace(
            cfg.domain_bounds[1][0], cfg.domain_bounds[1][1], cfg.num_pts_per_axis)
        X_train = np.array(list(itertools.product(x_vals, y_vals)))

        # 2. Apply Global Directional Perturbations
        # Each direction (ray) is associated with a hypercomplex basis e(1), e(2), ...
        e_bases = [oti.e(i + 1, order=cfg.n_order)
                   for i in range(cfg.rays.shape[1])]
        perturbations = np.dot(cfg.rays, e_bases)

        X_pert = oti.array(X_train)
        # Add the same perturbation vector to each dimension of every training point
        for j in range(cfg.n_bases):
            X_pert[:, j] += perturbations[j]

        print(
            f"  Applied {cfg.rays.shape[1]} directional perturbations to all {len(X_train)} training points.")

        # 3. Evaluate, Truncate, and Extract Data
        f_hc = self.true_function(X_pert, alg=oti)
        # Truncate cross-terms between the different directional bases
        for combo in itertools.combinations(range(1, cfg.rays.shape[1] + 1), 2):
            f_hc = f_hc.truncate(combo)

        y_train_list = [f_hc.real]

        # Define which directional derivatives to extract for training
        # e.g., [[1,1]] is the 1st derivative along the 1st ray's direction
        der_indices = [[
            [[1, 1]], [[1, 2]],  # 1st & 2nd derivative along ray 1
            [[2, 1]], [[2, 2]],  # 1st & 2nd derivative along ray 2
            [[3, 1]], [[3, 2]],  # 1st & 2nd derivative along ray 3
        ]]

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
        """Generates 2D surface plots comparing the prediction with the true function."""
        print("\n" + "="*50 + "\nGenerating Visualizations\n" + "="*50)
        plotting_helper.make_plots(
            self.training_data['X_train'], self.training_data['y_train_list'],
            self.results['X_test'], self.results['y_pred'], self.true_function,
            X1_grid=self.results['X1_grid'], X2_grid=self.results['X2_grid'],
            n_order=self.config.n_order, plot_derivative_surrogates=False,
            der_indices=self.training_data['der_indices']
        )
        print("  Visualization created.")

    def run(self):
        """Executes the complete tutorial workflow."""
        print("DEGP Tutorial: Global Directional Derivatives")
        print("=" * 75)

        self._generate_training_data()
        self._train_model()
        self._evaluate_model()
        self.visualize_results()

        print("\nTutorial Complete.")

# --- Helper Function ---


def true_function(X, alg=np):
    """2D Six-Hump Camel function."""
    x1, x2 = X[:, 0], X[:, 1]
    return ((4 - 2.1 * x1**2 + (x1**4) / 3.0) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2)


def main():
    """Main execution block."""
    config = GlobalDirectionalConfig()
    tutorial = GlobalDirectionalDEGPTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
