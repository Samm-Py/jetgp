"""
================================================================================
DEGP Tutorial: 3D Function Approximation with 2D Slice Visualization
================================================================================

This tutorial demonstrates how to apply a Directional-Derivative Enhanced
Gaussian Process (D-DEGP) to a 3-dimensional function. It uses a global set of
directional derivatives (rays) that correspond to the standard Cartesian axes.

A key challenge in high-dimensional modeling is visualization. This script shows
how to analyze the performance of the 3D model by taking a 2D slice (fixing the
value of x₃) and plotting the GP's predictions on that plane.

Key concepts covered:
-   Applying a D-DEGP to a 3D function.
-   Using a global basis of directional derivatives (standard axes).
-   Training the `ddegp` model on 3D data.
-   Visualizing high-dimensional model performance using 2D slicing.
"""

import numpy as np
import pyoti.sparse as oti
import itertools
from full_ddegp.ddegp import ddegp
import utils
import plotting_helper
from dataclasses import dataclass, field
from typing import Dict, Callable


@dataclass
class ThreeDimConfig:
    """Configuration for the 3D D-DEGP tutorial."""
    n_order: int = 4
    n_bases: int = 3
    num_pts_per_axis: int = 3
    domain_bounds: tuple = ((-1, 1),) * 3

    # Slice configuration for visualization
    # Index of the dimension to fix (0-indexed, so 2 is x₃)
    slice_dimension_index: int = 2
    slice_dimension_value: float = 0.5  # Value at which to fix the dimension
    test_grid_resolution: int = 25

    # Model & Optimizer
    normalize_data: bool = True
    kernel: str = "RQ"
    kernel_type: str = "isotropic"
    n_restarts: int = 15
    swarm_size: int = 50
    random_seed: int = 0


class ThreeDimDirectionalDEGPTutorial:
    """
    Manages and executes the D-DEGP tutorial on a 3D function,
    with visualization on a 2D slice.
    """

    def __init__(self, config: ThreeDimConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.training_data: Dict = {}
        self.gp_model = None
        self.params = None
        self.results: Dict = {}
        np.random.seed(config.random_seed)

    def _generate_training_data(self):
        """
        Generates 3D training data and applies global directional perturbations.
        """
        print("\n" + "="*50 + "\nGenerating 3D Training Data\n" + "="*50)
        cfg = self.config

        # 1. Create 3D Training Grid
        axis_points = [np.linspace(b[0], b[1], cfg.num_pts_per_axis)
                       for b in cfg.domain_bounds]
        X_train = np.array(list(itertools.product(*axis_points)))

        # 2. Define Rays as Standard Basis Vectors
        rays = np.eye(cfg.n_bases)
        self.training_data['rays'] = rays  # Store for later use

        # 3. Apply Global Directional Perturbations
        e_bases = [oti.e(i + 1, order=cfg.n_order)
                   for i in range(rays.shape[1])]
        perturbations = np.dot(rays.T, e_bases)

        X_pert = oti.array(X_train)
        for j in range(cfg.n_bases):
            X_pert[:, j] += perturbations[j]

        print(
            f"  Applied {rays.shape[1]} standard basis directional perturbations to all {len(X_train)} training points.")

        # 4. Evaluate, Truncate, and Extract Data
        f_hc = self.true_function(X_pert, alg=oti)
        for combo in itertools.combinations(range(1, rays.shape[1] + 1), 2):
            f_hc = f_hc.truncate(combo)

        y_train_list = [f_hc.real]

        # Define which derivatives to extract for each of the 3 directions
        der_indices = [[
            [[1, 1]], [[1, 2]], [[1, 3]], [[1, 4]],  # Derivatives along x₁
            [[2, 1]], [[2, 2]], [[2, 3]], [[2, 4]],  # Derivatives along x₂
            [[3, 1]], [[3, 2]], [[3, 3]], [[3, 4]],  # Derivatives along x₃
        ]]

        for group in der_indices:
            for sub_group in group:
                y_train_list.append(f_hc.get_deriv(sub_group).reshape(-1, 1))

        self.training_data.update(
            {'X_train': X_train, 'y_train_list': y_train_list, 'der_indices': der_indices})
        print(
            f"  Extracted {len(y_train_list) - 1} types of directional derivatives.")

    def _train_model(self):
        """Initializes and trains the ddegp model."""
        print("\n" + "="*50 + "\nTraining 3D D-DEGP Model\n" + "="*50)
        cfg, data = self.config, self.training_data

        self.gp_model = ddegp(
            data['X_train'], data['y_train_list'], n_order=cfg.n_order,
            der_indices=data['der_indices'], rays=data['rays'], normalize=cfg.normalize_data,
            kernel=cfg.kernel, kernel_type=cfg.kernel_type
        )
        print("  Model initialization: SUCCESS")

        print("  Optimizing hyperparameters...")
        self.params = self.gp_model.optimize_hyperparameters(
            n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size
        )
        print("  Hyperparameter optimization: SUCCESS")

    def _evaluate_on_slice(self):
        """Creates a 2D test slice and evaluates the model's performance."""
        print("\n" + "="*50 + "\nModel Prediction on 2D Slice\n" + "="*50)
        cfg = self.config

        # Define the two active dimensions for the slice plot
        active_dims = [i for i in range(
            cfg.n_bases) if i != cfg.slice_dimension_index]

        # Create grid for the active dimensions
        x_lin = np.linspace(cfg.domain_bounds[active_dims[0]][0],
                            cfg.domain_bounds[active_dims[0]][1], cfg.test_grid_resolution)
        y_lin = np.linspace(cfg.domain_bounds[active_dims[1]][0],
                            cfg.domain_bounds[active_dims[1]][1], cfg.test_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)

        # Create the full 3D test matrix for prediction
        X_test = np.full((X1_grid.size, cfg.n_bases),
                         cfg.slice_dimension_value)
        X_test[:, active_dims[0]] = X1_grid.ravel()
        X_test[:, active_dims[1]] = X2_grid.ravel()

        y_pred = self.gp_model.predict(
            X_test, self.params, calc_cov=False, return_deriv=False)
        y_true = self.true_function(X_test, alg=np)

        self.results = {
            'X_test': X_test, 'y_pred': y_pred, 'y_true': y_true,
            'X1_grid': X1_grid, 'X2_grid': X2_grid,
            'nrmse': utils.nrmse(y_true, y_pred)
        }
        print(
            f"  Evaluation on slice (x₃={cfg.slice_dimension_value}) complete. NRMSE: {self.results['nrmse']:.6f}")

    def visualize_slice_results(self):
        """Generates 2D surface plots for the evaluated slice."""
        print("\n" + "="*50 + "\nGenerating Slice Visualization\n" + "="*50)
        plotting_helper.make_plots(
            self.training_data['X_train'], self.training_data['y_train_list'],
            self.results['X_test'], self.results['y_pred'], self.true_function,
            X1_grid=self.results['X1_grid'], X2_grid=self.results['X2_grid'],
            n_order=self.config.n_order, plot_derivative_surrogates=False,
            der_indices=self.training_data['der_indices']
        )
        print(
            f"  Visualization of slice at x₃={self.config.slice_dimension_value} created.")

    def run(self):
        """Executes the complete tutorial workflow."""
        print("DEGP Tutorial: 3D Function Approx with 2D Slice Visualization")
        print("=" * 75)

        self._generate_training_data()
        self._train_model()
        self._evaluate_on_slice()
        self.visualize_slice_results()

        print(
            f"\nTutorial Complete. Final NRMSE on slice: {self.results['nrmse']:.6f}")

# --- Helper Function ---


def true_function(X, alg=np):
    """True 3D polynomial function of degree 4."""
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return (x1**2 * x2 + x2**2 * x3 + x3**2 * x1 + x1**2 * x2**2)


def main():
    """Main execution block."""
    config = ThreeDimConfig()
    tutorial = ThreeDimDirectionalDEGPTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
