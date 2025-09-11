"""
================================================================================
DEGP Tutorial: High-Dimensional Function Approximation with 2D Visualization
================================================================================

This tutorial demonstrates how to apply DEGP to high-dimensional functions
(4D in this case) and visualize the results through 2D slices. This approach
is crucial for real-world applications where functions have many inputs but
we need to understand their behavior through lower-dimensional views.

Key concepts covered:
- High-dimensional DEGP with a selective "main derivatives" strategy.
- Sobol sequence sampling for efficient high-dimensional space coverage.
- Dimensionality reduction for analysis and visualization via 2D slicing.
- Quantifying the computational savings of a selective derivative approach.
- Performance analysis on multiple 2D slices of the high-dimensional space.
================================================================================
"""
import numpy as np
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
import sobol as sb
import time
import plotting_helper
from dataclasses import dataclass, field
from typing import List, Dict, Callable
from matplotlib import pyplot as plt


@dataclass
class HighDimConfig:
    """Configuration for the high-dimensional DEGP tutorial."""
    n_bases: int = 4
    n_order: int = 2
    num_training_pts: int = 25
    slice_grid_resolution: int = 25
    lower_bounds: List[float] = field(default_factory=lambda: [-5.0] * 4)
    upper_bounds: List[float] = field(default_factory=lambda: [5.0] * 4)
    normalize_data: bool = True
    kernel: str = "RQ"
    kernel_type: str = "isotropic"
    n_restarts: int = 15
    swarm_size: int = 250
    random_seed: int = 1354


class HighDimDEGPTutorial:
    """
    Manages a high-dimensional DEGP experiment, including derivative strategy
    analysis, training, and 2D slice-based evaluation.
    """

    def __init__(self, config: HighDimConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.training_data = {}
        self.gp_model = None
        self.params = None
        self.results = {}
        np.random.seed(config.random_seed)

    def _analyze_derivative_complexity(self):
        """Analyzes and defines the selective derivative strategy."""
        print("\n" + "="*55 +
              "\nHigh-Dimensional Derivative Complexity Analysis\n" + "="*55)
        cfg = self.config

        complete_indices = utils.gen_OTI_indices(cfg.n_bases, cfg.n_order)
        complete_count = sum(len(group) for group in complete_indices)

        # Strategy: Use only main 1st and 2nd order derivatives
        der_indices = [
            [[[i + 1, 1]] for i in range(cfg.n_bases)],  # All 1st order
            [[[i + 1, 2]] for i in range(cfg.n_bases)],  # All 2nd order
        ]
        selective_count = sum(len(group) for group in der_indices)

        print(
            f"Complete derivative set would contain: {complete_count} derivatives (incl. cross-terms)")
        print(
            f"Our selective strategy uses: {selective_count} derivatives (main derivatives only)")
        print(
            f"This is a {complete_count / selective_count:.1f}x reduction in derivative terms.")

        self.training_data['der_indices'] = der_indices

    def _generate_training_data(self):
        """Generates high-dimensional training data using Sobol sampling."""
        print("\n" + "="*50 + "\nGenerating High-Dimensional Training Data\n" + "="*50)
        start_time = time.time()
        cfg = self.config

        sobol_train = sb.create_sobol_samples(
            cfg.num_training_pts, cfg.n_bases, 1).T
        X_train = utils.scale_samples(
            sobol_train, cfg.lower_bounds, cfg.upper_bounds)
        print(f"  Sampling method: Sobol sequences for efficient space coverage.")

        X_train_pert = oti.array(X_train)
        for i in range(cfg.n_bases):
            X_train_pert[:, i] += oti.e(i + 1, order=cfg.n_order)

        y_train_hc = self.true_function(X_train_pert)
        y_train_list = [y_train_hc.real]
        for group in self.training_data['der_indices']:
            for sub_group in group:
                y_train_list.append(y_train_hc.get_deriv(sub_group))

        self.training_data.update(
            {'X_train': X_train, 'y_train_list': y_train_list})
        print(
            f"  Total observations created: {sum(d.shape[0] for d in y_train_list)}")
        print(f"  Data generation time: {time.time() - start_time:.3f}s")

    def _train_model(self):
        """Initializes, trains, and optimizes the high-dimensional DEGP model."""
        print("\n" + "="*45 + "\nTraining High-Dimensional DEGP Model\n" + "="*45)
        training_start = time.time()
        cfg = self.config
        data = self.training_data

        try:
            self.gp_model = degp(
                data['X_train'], data['y_train_list'], cfg.n_order, cfg.n_bases,
                data['der_indices'], normalize=cfg.normalize_data,
                kernel=cfg.kernel, kernel_type=cfg.kernel_type
            )
            print("  Model initialization: SUCCESS")

            print("  Optimizing hyperparameters...")
            self.params = self.gp_model.optimize_hyperparameters(
                n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size
            )
            print("  Hyperparameter optimization: SUCCESS")
            print(
                f"  Total training time: {time.time() - training_start:.2f}s")
        except Exception as e:
            print(f"  Training FAILED\n  Error: {e}")
            raise

    def _evaluate_on_slices(self):
        """Evaluates model performance on various 2D slices of the 4D space."""
        print("\n" + "="*60 +
              "\n2D Slice Analysis: Visualizing High-Dimensional Behavior\n" + "="*60)
        cfg = self.config

        slice_strategies = {
            'zero': {'vals': [0.0] * (cfg.n_bases - 2), 'desc': 'x₃=0, x₄=0'},
            'center': {'vals': [(l+u)/2 for l, u in zip(cfg.lower_bounds[2:], cfg.upper_bounds[2:])], 'desc': 'domain center'},
            'random': {'vals': [np.random.uniform(l, u) for l, u in zip(cfg.lower_bounds[2:], cfg.upper_bounds[2:])], 'desc': 'random point'}
        }

        print("Evaluating performance on different 2D slices...")
        for name, strategy in slice_strategies.items():
            result = self._evaluate_single_slice(strategy['vals'])
            self.results[name + '_slice'] = result
            print(
                f"  {name.capitalize()} slice ({strategy['desc']}) NRMSE: {result['nrmse']:.6f}")

    def _evaluate_single_slice(self, fixed_values: List[float]) -> Dict:
        """Helper function to evaluate performance on one specific slice."""
        cfg = self.config
        x_lin = np.linspace(
            cfg.lower_bounds[0], cfg.upper_bounds[0], cfg.slice_grid_resolution)
        y_lin = np.linspace(
            cfg.lower_bounds[1], cfg.upper_bounds[1], cfg.slice_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)

        X_test = np.zeros((X1_grid.size, cfg.n_bases))
        X_test[:, 0] = X1_grid.ravel()
        X_test[:, 1] = X2_grid.ravel()
        for i, val in enumerate(fixed_values):
            X_test[:, i + 2] = val

        y_pred, y_var = self.gp_model.predict(
            X_test, self.params, calc_cov=True)
        y_true = self.true_function(X_test, alg=np)

        return {
            'nrmse': utils.nrmse(y_true, y_pred),
            'X1_grid': X1_grid, 'X2_grid': X2_grid,
            'y_true_grid': y_true.reshape(X1_grid.shape),
            'y_pred_grid': y_pred.reshape(X1_grid.shape)
        }

    def visualize_slice_results(self, slice_name: str = 'zero_slice'):
        """Generates contour plots for a specified 2D slice."""
        print(f"\nGenerating 2D Visualization for the '{slice_name}'...")
        if slice_name not in self.results:
            print(f"  Error: Results for '{slice_name}' not found.")
            return

        res = self.results[slice_name]
        fig, axes = plt.subplots(1, 3, figsize=(
            18, 5), sharex=True, sharey=True)
        X_train_proj = self.training_data['X_train'][:, :2]

        c1 = axes[0].contourf(res['X1_grid'], res['X2_grid'],
                              res['y_true_grid'], levels=50, cmap="viridis")
        fig.colorbar(c1, ax=axes[0])
        axes[0].set_title(f"True Function ({slice_name})")
        axes[0].scatter(X_train_proj[:, 0], X_train_proj[:, 1], c="red",
                        edgecolor="k", s=50, label="Training Points (projection)")

        c2 = axes[1].contourf(res['X1_grid'], res['X2_grid'],
                              res['y_pred_grid'], levels=50, cmap="viridis")
        fig.colorbar(c2, ax=axes[1])
        axes[1].set_title(f"GP Prediction ({slice_name})")
        axes[1].scatter(X_train_proj[:, 0], X_train_proj[:, 1],
                        c="red", edgecolor="k", s=50)

        error_grid = np.abs(res['y_true_grid'] - res['y_pred_grid'])
        c3 = axes[2].contourf(res['X1_grid'], res['X2_grid'],
                              error_grid, levels=50, cmap="magma")
        fig.colorbar(c3, ax=axes[2])
        axes[2].set_title(f"Absolute Error ({slice_name})")
        axes[2].scatter(X_train_proj[:, 0], X_train_proj[:, 1],
                        c="red", edgecolor="k", s=50)

        for ax in axes:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
        axes[0].legend()
        plt.tight_layout()
        plt.show()

    def run(self):
        """Executes the complete tutorial workflow."""
        start_time = time.time()
        print("High-Dimensional DEGP Tutorial: 4D Function with 2D Visualization")
        print("=" * 75)
        print(
            f"Configuration: {self.config.n_bases}D, {self.config.num_training_pts} training points, order {self.config.n_order} derivatives.")

        self._analyze_derivative_complexity()
        self._generate_training_data()
        self._train_model()
        self._evaluate_on_slices()
        self.visualize_slice_results(slice_name='zero_slice')

        print("\n" + "="*50 + "\nTutorial Summary\n" + "="*50)
        print(f"Total execution time: {time.time() - start_time:.2f}s")
        print(
            f"Final NRMSE on zero slice: {self.results['zero_slice']['nrmse']:.6f}")
        print("Key Takeaways:")
        print("  - DEGP can effectively model high-dimensional functions from sparse data.")
        print("  - A selective derivative strategy is crucial for managing computational cost.")
        print("  - 2D slicing is a powerful technique for analyzing and visualizing high-dimensional model performance.")


def true_function(X, alg=oti):
    """
    Styblinski–Tang function in 4D.

    Function: f(x₁,x₂,x₃,x₄) = 0.5 * sum_{i=1}^4 (x_i^4 - 16x_i^2 + 5x_i)

    Parameters:
    -----------
    X : array-like, shape (n_samples, 4)
        Input points with columns [x1, x2, x3, x4]
    alg : module
        Numerical library (numpy or pyoti)

    Returns:
    --------
    y : array-like
        Function values
    """
    x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    return 0.5 * (x1**4 - 16*x1**2 + 5*x1 +
                  x2**4 - 16*x2**2 + 5*x2 +
                  x3**4 - 16*x3**2 + 5*x3 +
                  x4**4 - 16*x4**2 + 5*x4)


def main():
    """Main execution block."""
    config = HighDimConfig()
    tutorial = HighDimDEGPTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
