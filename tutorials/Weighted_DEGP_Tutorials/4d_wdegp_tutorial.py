"""
4D Derivative-Enhanced GP with KMeans Submodel Clustering
===========================================================

This module demonstrates a derivative-enhanced GP for a high-dimensional (4D)
function. It introduces several advanced concepts:

1.  **Automatic Submodel Generation**: Instead of manually defining point groups,
    this script uses the KMeans clustering algorithm to automatically partition the
    training data into spatially coherent submodels. This is a data-driven
    approach that scales well to higher dimensions.

2.  **Sobol Sequence Sampling**: For effective coverage of the 4D input space,
    low-discrepancy Sobol sequences are used to generate training and test points
    instead of a standard grid.

3.  **Specific Derivative Structure**: The model is configured to use only "main"
    derivatives (e.g., ∂f/∂x₁, ∂²f/∂x₁²) and exclude cross-derivatives,
    reducing the total number of derivative inputs.

4.  **2D Slice Visualization**: To visualize the performance of the 4D model,
    a 2D slice of the input space is taken (e.g., by setting x₃=0 and x₄=0),
    and the model's predictions are plotted as a contour map.
"""

import numpy as np
import pyoti.sparse as oti
import itertools
from typing import List, Tuple, Callable, Optional, Dict, Any
from wdegp.wdegp import wdegp
import utils
import sobol as sb
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


@dataclass
class HighDimGPConfig:
    """Configuration for a high-dimensional grouped submodel GP."""
    n_bases: int = 4
    n_order: int = 2
    num_points_train: int = 36
    num_points_test: int = 5000
    lower_bounds: List[float] = field(default_factory=lambda: [-5] * 4)
    upper_bounds: List[float] = field(default_factory=lambda: [5] * 4)
    n_clusters: int = 3  # Number of submodels to create via KMeans
    kernel: str = "RQ"
    kernel_type: str = "isotropic"
    normalize: bool = True
    n_restart_optimizer: int = 15
    swarm_size: int = 200
    random_seed: Optional[int] = 1354


class HighDimGP:
    """
    Manages a high-dimensional GP experiment with KMeans clustering.
    """

    def __init__(self, config: HighDimGPConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.gp_model = None
        self.training_data = None
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

    def generate_data_points(self, num_points: int) -> np.ndarray:
        """Generates data points using a Sobol sequence for good space coverage."""
        quasi_samples = sb.create_sobol_samples(
            num_points, self.config.n_bases, 1).T
        return utils.scale_samples(quasi_samples, self.config.lower_bounds, self.config.upper_bounds)

    def create_submodels_and_reorder_data(
        self,
        X_train_initial: np.ndarray
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Uses KMeans to cluster points into submodels and then reorders the data.
        """
        print(
            f"Running KMeans to cluster {X_train_initial.shape[0]} points into {self.config.n_clusters} submodels...")
        kmeans = KMeans(n_clusters=self.config.n_clusters,
                        random_state=self.config.random_seed, n_init='auto').fit(X_train_initial)

        # Create initial groups based on KMeans labels
        unsorted_groups = [[] for _ in range(self.config.n_clusters)]
        for i, label in enumerate(kmeans.labels_):
            unsorted_groups[label].append(i)

        # The new data order is determined by sorting all points by their cluster index
        reorder_map = list(itertools.chain.from_iterable(unsorted_groups))
        X_train_reordered = X_train_initial[reorder_map]

        # Create the new sequential index required by the GP framework
        sequential_indices = []
        current_pos = 0
        for group in unsorted_groups:
            group_len = len(group)
            sequential_indices.append(
                list(range(current_pos, current_pos + group_len)))
            current_pos += group_len

        print("Clustering and reordering complete.")
        return X_train_reordered, sequential_indices

    def prepare_submodel_data(
        self,
        X_train: np.ndarray,
        submodel_indices: List[List[int]]
    ) -> Tuple[List[List[np.ndarray]], List]:
        """
        Prepare training data using only main derivatives for all submodels.
        """
        print("Preparing submodel data using main derivatives...")

        # Define a specific derivative structure: only main derivatives, no cross-terms
        main_derivatives = [
            [[[i + 1, order + 1]] for i in range(self.config.n_bases)]
            for order in range(self.config.n_order)
        ]
        derivative_specs = [main_derivatives for _ in submodel_indices]

        y_function_values = self.true_function(X_train, alg=np)
        submodel_data = []

        for k, point_indices in enumerate(submodel_indices):
            X_sub_oti = oti.array(X_train[point_indices])
            for i in range(self.config.n_bases):
                for j in range(X_sub_oti.shape[0]):
                    X_sub_oti[j, i] += oti.e(i + 1, order=self.config.n_order)

            y_with_derivatives = oti.array(
                [self.true_function(x, alg=oti) for x in X_sub_oti])

            current_submodel_data = [y_function_values]
            for i in range(len(main_derivatives)):
                for j in range(len(main_derivatives[i])):
                    deriv = y_with_derivatives.get_deriv(
                        main_derivatives[i][j]).reshape(-1, 1)
                    current_submodel_data.append(deriv)
            submodel_data.append(current_submodel_data)

        return submodel_data, derivative_specs

    def build_and_optimize_gp(self, X_train: np.ndarray, submodel_data: List,
                              submodel_indices: List[List[int]],
                              derivative_specs: List) -> Dict[str, Any]:
        """Construct the GP model and optimize its hyperparameters."""
        print("\nBuilding and optimizing the weighted GP model...")
        self.gp_model = wdegp(
            X_train, submodel_data, self.config.n_order, self.config.n_bases,
            submodel_indices, derivative_specs, normalize=self.config.normalize,
            kernel=self.config.kernel, kernel_type=self.config.kernel_type,
        )

        params = self.gp_model.optimize_hyperparameters(
            n_restart_optimizer=self.config.n_restart_optimizer,
            swarm_size=self.config.swarm_size
        )
        print("Hyperparameter optimization complete.")
        return params

    def evaluate_model(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate model on the full test set and on a 2D slice."""
        if self.gp_model is None:
            raise ValueError("GP model has not been built.")

        # 1. Evaluate on the full 4D test set
        print(
            f"\nMaking predictions on {self.config.num_points_test} high-dimensional test points...")
        X_test = self.generate_data_points(self.config.num_points_test)
        y_pred = self.gp_model.predict(
            X_test, params, calc_cov=False, return_submodels=False)
        y_true = self.true_function(X_test, alg=np)
        nrmse_full = utils.nrmse(y_true, y_pred, norm_type="minmax")
        print(f"Full 4D NRMSE: {nrmse_full:.6f}")

        # 2. Evaluate on a 2D slice for visualization (setting x3=0, x4=0)
        print("Making predictions on a 2D slice (x3=0, x4=0)...")
        grid_points = 50
        x1x2_lin = np.linspace(
            self.config.lower_bounds[0], self.config.upper_bounds[0], grid_points)
        X1_grid, X2_grid = np.meshgrid(x1x2_lin, x1x2_lin)
        X_slice = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
        X_slice = np.hstack([X_slice, np.zeros((X_slice.shape[0], 2))])

        y_slice_pred = self.gp_model.predict(
            X_slice, params, calc_cov=False, return_submodels=False)
        y_slice_true = self.true_function(X_slice, alg=np)
        nrmse_slice = utils.nrmse(
            y_slice_true, y_slice_pred, norm_type="minmax")
        print(f"2D Slice NRMSE: {nrmse_slice:.6f}")

        return {
            'nrmse_full': nrmse_full, 'nrmse_slice': nrmse_slice,
            'X1_grid': X1_grid, 'X2_grid': X2_grid,
            'y_true_grid': y_slice_true.reshape(X1_grid.shape),
            'y_pred_grid': y_slice_pred.reshape(X1_grid.shape),
        }

    def visualize_slice(self, results: Dict[str, Any]):
        """Generate contour plots for the 2D slice."""
        print("Creating 2D contour visualizations for the slice...")
        fig, axes = plt.subplots(1, 3, figsize=(
            18, 5), sharex=True, sharey=True)

        # For the scatter plot, we show the projection of the training points onto the slice plane
        X_train_proj = self.training_data['X_train'][:, :2]

        c1 = axes[0].contourf(results['X1_grid'], results['X2_grid'],
                              results['y_true_grid'], levels=50, cmap="viridis")
        fig.colorbar(c1, ax=axes[0])
        axes[0].set_title("True Function (Slice at x₃=0, x₄=0)")
        axes[0].scatter(X_train_proj[:, 0], X_train_proj[:, 1], c="red",
                        edgecolor="k", s=50, label="Training Points (projection)")

        c2 = axes[1].contourf(results['X1_grid'], results['X2_grid'],
                              results['y_pred_grid'], levels=50, cmap="viridis")
        fig.colorbar(c2, ax=axes[1])
        axes[1].set_title("GP Prediction (Slice)")
        axes[1].scatter(X_train_proj[:, 0], X_train_proj[:, 1],
                        c="red", edgecolor="k", s=50)

        error_grid = np.abs(results['y_true_grid'] - results['y_pred_grid'])
        c3 = axes[2].contourf(
            results['X1_grid'], results['X2_grid'], error_grid, levels=50, cmap="magma")
        fig.colorbar(c3, ax=axes[2])
        axes[2].set_title("Absolute Prediction Error (Slice)")
        axes[2].scatter(X_train_proj[:, 0], X_train_proj[:, 1],
                        c="red", edgecolor="k", s=50)

        for ax in axes:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
        axes[0].legend()
        plt.tight_layout()
        plt.show()

    def run_complete_experiment(self, visualize: bool = True):
        """Execute the complete high-dimensional GP experiment."""
        print("=== 4D GP Experiment with KMeans Submodel Clustering ===")

        X_train_initial = self.generate_data_points(
            self.config.num_points_train)
        X_train_reordered, sequential_indices = self.create_submodels_and_reorder_data(
            X_train_initial)
        submodel_data, derivative_specs = self.prepare_submodel_data(
            X_train_reordered, sequential_indices)

        self.training_data = {'X_train': X_train_reordered,
                              'submodel_indices': sequential_indices}

        params = self.build_and_optimize_gp(
            X_train_reordered, submodel_data, sequential_indices, derivative_specs)
        results = self.evaluate_model(params)

        if visualize:
            self.visualize_slice(results)

        print("\n=== Experiment Summary ===")
        print(f"Input Dimensions: {self.config.n_bases}")
        print(f"Training Points: {self.config.num_points_train}")
        print(f"Submodels Created (KMeans): {self.config.n_clusters}")
        print("-" * 20)
        print(f"Final Full 4D NRMSE: {results['nrmse_full']:.6f}")
        print(f"Final 2D Slice NRMSE: {results['nrmse_slice']:.6f}")
        print("\nSubmodel Point Distribution:")
        for i, group in enumerate(self.training_data['submodel_indices']):
            print(f"  Submodel {i+1} contains {len(group)} points.")
        return results


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
    """Main execution block to demonstrate the high-dimensional GP."""
    config = HighDimGPConfig()
    experiment = HighDimGP(config, true_function)
    experiment.run_complete_experiment(visualize=True)


if __name__ == "__main__":
    main()
