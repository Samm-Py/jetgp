"""
2D Grouped Submodel Derivative-Enhanced Gaussian Process
=========================================================

This module demonstrates an advanced derivative-enhanced GP for a 2D function.
It showcases a highly flexible approach where training points are grouped into
submodels, and each submodel can be assigned a different derivative order.

The model uses function values (y) from all training points to inform the
global fit. However, it only computes and uses derivative information for a
specified subset of these points, which are defined in the `submodel_point_groups`
list. This allows for a tailored modeling strategy where computational effort
and model complexity (i.e., high-order derivatives) can be focused on specific
regions of interest.

Key Features:
- Approximation of a 2D function.
- Submodel creation from arbitrary groups of training points.
- **Selective Derivative Utilization**: Only points included in submodel groups contribute derivative information.
- **Heterogeneous Derivative Orders**: Each submodel can be assigned a different maximum derivative order (e.g., no derivatives, 1st order only, or full high-order derivatives).
- **Automatic Data Reordering**: The script automatically reorders the training data to satisfy the sequential indexing requirements of the underlying GP framework.
"""

import numpy as np
import pyoti.sparse as oti
import itertools
from typing import List, Tuple, Callable, Optional, Dict, Any
from wdegp.wdegp import wdegp
import utils
import plotting_helper
import matplotlib.pyplot as plt
from dataclasses import dataclass, field


@dataclass
class TwoDimGPConfig:
    """Configuration for a 2D grouped submodel derivative-enhanced GP."""
    n_order: int = 3                     # Max derivative order used in any submodel
    # Number of input dimensions (must be 2)
    n_bases: int = 2
    lb_x: float = -1.0                   # Lower bound of domain (x-axis)
    ub_x: float = 1.0                    # Upper bound of domain (x-axis)
    lb_y: float = -1.0                   # Lower bound of domain (y-axis)
    ub_y: float = 1.0                    # Upper bound of domain (y-axis)
    points_per_axis: int = 4             # Number of points along each axis
    kernel: str = "RQ"                   # Kernel type
    kernel_type: str = "isotropic"     # Kernel parameterization
    normalize: bool = True               # Whether to normalize training data
    n_restart_optimizer: int = 15        # Hyperparameter optimization restarts
    swarm_size: int = 100                # Particle swarm optimization size
    test_points_per_axis: int = 35       # Grid size for test evaluation
    random_seed: Optional[int] = 0       # Random seed for reproducibility
    # User-defined groupings of point indices FOR DERIVATIVE CALCULATION.
    # Points not in this list will contribute function values but not derivatives.
    submodel_point_groups: List[List[int]] = field(default_factory=list)


class TwoDimGroupedGP:
    """
    Manages a 2D derivative-enhanced GP experiment with data reordering
    and heterogeneous derivative structures.
    """

    def __init__(self, config: TwoDimGPConfig, true_function: Callable):
        """
        Initialize the 2D grouped submodel GP experiment.

        Args:
            config: Configuration object with experiment parameters.
            true_function: The 2D target function to approximate.
        """
        self.config = config
        self.true_function = true_function
        self.gp_model = None
        self.training_data = None
        if config.n_bases != 2:
            raise ValueError(
                "This class is configured for 2D problems (n_bases=2).")
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

    def generate_training_points(self) -> np.ndarray:
        """Generate a 2D grid of training points."""
        x_vals = np.linspace(
            self.config.lb_x, self.config.ub_x, self.config.points_per_axis)
        y_vals = np.linspace(
            self.config.lb_y, self.config.ub_y, self.config.points_per_axis)
        return np.array(list(itertools.product(x_vals, y_vals)))

    def reorder_training_data(
        self,
        X_train_initial: np.ndarray
    ) -> Tuple[np.ndarray, List[List[int]]]:
        """
        Reorders training data based on the subset of points used for derivatives.

        This method ensures that the points with derivative information are placed
        at the beginning of the `X_train` array, which is a requirement for the
        `wdegp` model's indexing scheme.

        Returns:
            A tuple containing:
            - X_train_reordered: The training points sorted correctly.
            - sequential_indices: The new, sequential indices for the submodels
                                  that use derivatives.
        """
        print("Reordering training data based on points selected for derivatives...")

        arbitrary_groups = self.config.submodel_point_groups
        arbitrary_flat = list(itertools.chain.from_iterable(arbitrary_groups))

        # Check for duplicates. Points can be omitted, but not used twice.
        if len(arbitrary_flat) != len(set(arbitrary_flat)):
            raise ValueError(
                "Invalid `submodel_point_groups`: a point index cannot be in more than one group.")

        # Identify points NOT used for derivatives
        all_indices = set(range(X_train_initial.shape[0]))
        used_indices = set(arbitrary_flat)
        unused_indices = sorted(list(all_indices - used_indices))

        # The new order is the specified derivative points first, then the rest
        reorder_map = arbitrary_flat + unused_indices
        X_train_reordered = X_train_initial[reorder_map]

        # The sequential indices only refer to the reordered points that have derivatives
        sequential_indices = []
        current_pos = 0
        for group in arbitrary_groups:
            group_len = len(group)
            sequential_indices.append(
                list(range(current_pos, current_pos + group_len)))
            current_pos += group_len

        print("Reordering complete.")
        return X_train_reordered, sequential_indices

    def prepare_submodel_data(
        self,
        X_train: np.ndarray,
        submodel_indices: List[List[int]]
    ) -> Tuple[List[List[np.ndarray]], List]:
        """
        Prepare training data, assigning different derivative orders to submodels.
        """
        print("Preparing submodel data with heterogeneous derivatives...")

        # --- Define Derivative Structures for each Submodel ---
        # This example creates three distinct derivative structures.
        # The order here must correspond to the order of the point groups
        # defined in the `main` function.
        base_der_indices = utils.gen_OTI_indices(
            self.config.n_bases, self.config.n_order)
        derivative_specs = [
            [],  # Submodel 1 (Corners): No derivatives
            # Submodel 2 (Exterior): 1st order derivatives
            utils.gen_OTI_indices(self.config.n_bases, 1),
            base_der_indices  # Submodel 3 (Center): Full 3rd order derivatives
        ]

        # Function values from ALL training points are always used
        y_function_values = self.true_function(X_train, alg=np)
        submodel_data = []

        for k, point_indices in enumerate(submodel_indices):
            # Derivatives are computed only for points in the current submodel group
            X_sub_oti = oti.array(X_train[point_indices])
            for i in range(self.config.n_bases):
                for j in range(X_sub_oti.shape[0]):
                    X_sub_oti[j, i] += oti.e(i + 1, order=self.config.n_order)

            y_with_derivatives = oti.array(
                [self.true_function(x, alg=oti) for x in X_sub_oti])

            # Assemble data: all function values + derivatives specific to this submodel
            current_submodel_data = [y_function_values]
            current_derivative_spec = derivative_specs[k]
            for i in range(len(current_derivative_spec)):
                for j in range(len(current_derivative_spec[i])):
                    deriv = y_with_derivatives.get_deriv(
                        current_derivative_spec[i][j]).reshape(-1, 1)
                    current_submodel_data.append(deriv)
            submodel_data.append(current_submodel_data)

        print(f"Processed data for {len(submodel_indices)} submodels.")
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
        """Evaluate model performance on a dense 2D test grid."""
        if self.gp_model is None:
            raise ValueError("GP model has not been built.")

        print(
            f"\nMaking predictions on a {self.config.test_points_per_axis}x{self.config.test_points_per_axis} grid...")
        x_lin = np.linspace(self.config.lb_x, self.config.ub_x,
                            self.config.test_points_per_axis)
        y_lin = np.linspace(self.config.lb_y, self.config.ub_y,
                            self.config.test_points_per_axis)
        X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
        X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

        y_pred, submodel_vals = self.gp_model.predict(
            X_test, params, calc_cov=False, return_submodels=True
        )

        y_true = self.true_function(X_test, alg=np)
        nrmse = utils.nrmse(y_true, y_pred)
        abs_error = np.abs(y_true - y_pred)
        grid_shape = (self.config.test_points_per_axis,
                      self.config.test_points_per_axis)

        print(f"Model evaluation complete. NRMSE: {nrmse:.6f}")

        return {
            'X_test': X_test, 'y_pred': y_pred, 'y_true': y_true,
            'submodel_vals': submodel_vals, 'nrmse': nrmse,
            'X1_grid': X1_grid, 'X2_grid': X2_grid,
            'y_true_grid': y_true.reshape(grid_shape),
            'y_pred_grid': y_pred.reshape(grid_shape),
            'abs_error_grid': abs_error.reshape(grid_shape),
        }

    def create_contour_visualization(self, results: Dict[str, Any]):
        """Generate contour plots comparing true function, predictions, and errors."""
        print("Creating 2D contour visualizations...")
        fig, axes = plt.subplots(1, 3, figsize=(
            18, 5), sharex=True, sharey=True)
        # We plot all original points to show the full training set
        X_train = self.training_data['X_train_initial']

        c1 = axes[0].contourf(results['X1_grid'], results['X2_grid'],
                              results['y_true_grid'], levels=50, cmap="viridis")
        fig.colorbar(c1, ax=axes[0])
        axes[0].set_title("True Function")
        axes[0].scatter(X_train[:, 0], X_train[:, 1],
                        c="red", edgecolor="k", s=50, label="All Training Points")

        c2 = axes[1].contourf(results['X1_grid'], results['X2_grid'],
                              results['y_pred_grid'], levels=50, cmap="viridis")
        fig.colorbar(c2, ax=axes[1])
        axes[1].set_title("GP Prediction")
        axes[1].scatter(X_train[:, 0], X_train[:, 1],
                        c="red", edgecolor="k", s=50)

        c3 = axes[2].contourf(results['X1_grid'], results['X2_grid'],
                              results['abs_error_grid'], levels=50, cmap="magma")
        fig.colorbar(c3, ax=axes[2])
        axes[2].set_title("Absolute Prediction Error")
        axes[2].scatter(X_train[:, 0], X_train[:, 1],
                        c="red", edgecolor="k", s=50)

        for ax in axes:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")

        axes[0].legend()
        plt.tight_layout()
        plt.show()

    def run_complete_experiment(self, contour_plot: bool = True) -> Dict[str, Any]:
        """Execute the complete 2D GP experiment."""
        print("=== 2D Grouped Submodel GP Experiment with Reordering ===")

        X_train_initial = self.generate_training_points()
        X_train_reordered, sequential_indices = self.reorder_training_data(
            X_train_initial)
        submodel_data, derivative_specs = self.prepare_submodel_data(
            X_train_reordered, sequential_indices
        )

        self.training_data = {
            'X_train_initial': X_train_initial,
            'X_train': X_train_reordered,
            'submodel_data': submodel_data,
            'submodel_indices': sequential_indices,
            'derivative_specs': derivative_specs
        }

        params = self.build_and_optimize_gp(
            X_train_reordered, submodel_data, sequential_indices, derivative_specs
        )
        results = self.evaluate_model(params)

        if contour_plot:
            self.create_contour_visualization(results)

        print("\n=== Experiment Summary ===")
        print(f"Total training points: {self.config.points_per_axis**2}")
        print(
            f"Points used for derivatives: {len(list(itertools.chain.from_iterable(self.config.submodel_point_groups)))}")
        print(f"Number of submodels: {len(self.config.submodel_point_groups)}")
        print(f"Final NRMSE: {results['nrmse']:.6f}")
        print("\nSubmodel Groups (for Derivatives):")
        for i, (group, spec) in enumerate(zip(self.config.submodel_point_groups, self.training_data['derivative_specs'])):
            order = "None" if not spec else str(len(spec))
            print(
                f"  Submodel {i+1}: Original Indices {group} -> Max Derivative Order: {order}")

        return results


def six_hump_camel_function(X, alg=np):
    """Six-hump camel back function, a common 2D benchmark."""
    x1 = X[:, 0]
    x2 = X[:, 1]
    return ((4 - 2.1 * x1**2 + (x1**4) / 3.0) * x1**2 +
            x1 * x2 + (-4 + 4 * x2**2) * x2**2)


def main():
    """Main execution block to demonstrate the 2D GP."""
    # This example demonstrates assigning different derivative orders to
    # different regions of the input space.
    arbitrary_point_groups = [
        # Group 1: The four corner points.
        # These will be assigned NO derivative information.
        [0, 3, 12, 15],

        # Group 2: The exterior points (excluding corners).
        # These will be assigned 1st ORDER derivatives.
        [1, 2, 4, 8, 7, 11, 13, 14],

        # Group 3: The central 2x2 block.
        # These will be assigned the FULL high-order derivatives.
        [5, 6, 9, 10]
    ]

    config = TwoDimGPConfig(
        submodel_point_groups=arbitrary_point_groups,
        n_order=3  # Set the max order for the most complex group
    )

    experiment = TwoDimGroupedGP(config, six_hump_camel_function)
    experiment.run_complete_experiment(contour_plot=True)


if __name__ == "__main__":
    main()
