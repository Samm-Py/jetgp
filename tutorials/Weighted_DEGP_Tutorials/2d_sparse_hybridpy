"""
2D Sparse Derivative-Enhanced Gaussian Process Implementation
===========================================================

This module implements a 2D derivative-enhanced Gaussian Process model using 
pyOTI-based automatic differentiation. The implementation demonstrates:

- 2D grid-based training data generation
- Strategic point reordering to optimize derivative placement
- Flexible derivative specifications for different regions
- Comprehensive visualization and analysis tools
- Modular object-oriented design

Key Features:
- Supports first and second-order derivatives in 2D
- Point swapping for strategic derivative placement
- Contour plot visualization
- Detailed experiment tracking and analysis
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
import pyoti.sparse as oti
import itertools
from wdegp.wdegp import wdegp
import utils
import plotting_helper
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class GPConfig2D:
    """Configuration parameters for 2D GP experiments."""
    n_order: int = 2                    # Maximum derivative order
    n_bases: int = 2                    # Number of input dimensions
    lb_x: float = -1.0                  # Lower bound for x dimension
    ub_x: float = 1.0                   # Upper bound for x dimension
    lb_y: float = -1.0                  # Lower bound for y dimension
    ub_y: float = 1.0                   # Upper bound for y dimension
    # Points per dimension (creates 4x4 = 16 total)
    num_points: int = 4
    kernel: str = "SE"                  # Kernel type (Squared Exponential)
    kernel_type: str = "anisotropic"    # Kernel parameterization
    normalize: bool = True              # Whether to normalize data
    n_restart_optimizer: int = 15       # Hyperparameter optimization restarts
    swarm_size: int = 50               # Particle swarm size
    N_grid: int = 25                   # Test grid resolution per dimension
    random_seed: int = 0               # Random seed for reproducibility


class SparseDerivative2DGP:
    """
    2D derivative-enhanced Gaussian Process with sparse derivative placement.

    This class implements a GP model where derivatives are computed at strategic
    locations within the 2D domain, typically interior points that provide
    maximum information about local function behavior.
    """

    def __init__(self, config: GPConfig2D, true_function: Callable):
        """
        Initialize the 2D sparse derivative GP experiment.

        Args:
            config: Configuration object with experiment parameters
            true_function: The target function to approximate
        """
        self.config = config
        self.true_function = true_function
        self.gp_model = None
        self.X_train = None
        self.y_train_data = None
        self.results = None

        # Set random seed for reproducible results
        np.random.seed(config.random_seed)

    def create_2d_training_grid(self) -> Tuple[np.ndarray, List[List[int]], List]:
        """
        Create a 2D training grid with strategic derivative placement.

        This method:
        1. Generates a uniform 2D grid
        2. Identifies interior points suitable for derivative computation
        3. Reorders points to place derivatives at strategic locations
        4. Defines derivative specifications for the submodel

        Returns:
            Tuple of (training_points, submodel_indices, derivative_specs)
        """
        # Generate uniform grid over the 2D domain
        x_vals = np.linspace(
            self.config.lb_x, self.config.ub_x, self.config.num_points)
        y_vals = np.linspace(
            self.config.lb_y, self.config.ub_y, self.config.num_points)
        X_train = np.array(list(itertools.product(x_vals, y_vals)))

        # Identify strategic points for derivative computation
        # Points [5,6,9,10] correspond to interior grid locations
        interior_points = [5, 6, 9, 10]
        # Target contiguous indices for derivatives
        target_indices = [12, 13, 14, 15]

        # Reorder points to place interior points at contiguous indices
        # This satisfies the requirement that submodel indices be contiguous
        X_reordered = X_train.copy()
        X_reordered[interior_points], X_reordered[target_indices] = \
            X_train[target_indices], X_train[interior_points]

        # Define submodel structure: one submodel with derivatives at target indices
        submodel_indices = [target_indices]

        # Define derivative structure: first and second-order partial derivatives
        derivative_specs = [
            [[[1, 1]], [[2, 1]]],  # First-order: ∂f/∂x₁, ∂f/∂x₂
            [[[1, 2]], [[2, 2]]],  # Second-order: ∂²f/∂x₁², ∂²f/∂x₂²
        ]

        # Replicate derivative specification for each submodel
        derivative_specs = [
            derivative_specs for _ in range(len(submodel_indices))]

        return X_reordered, submodel_indices, derivative_specs

    def prepare_submodel_data(self, X_train: np.ndarray, submodel_indices: List[List[int]],
                              derivative_specs: List) -> List:
        """
        Prepare training data with function values and derivatives for each submodel.

        Args:
            X_train: Training point coordinates
            submodel_indices: Point indices for each submodel  
            derivative_specs: Derivative specifications for each submodel

        Returns:
            List of training data arrays for each submodel
        """
        y_train_data = []
        # Function values at all points
        y_real = self.true_function(X_train, alg=np)

        for k, point_indices in enumerate(submodel_indices):
            # Extract points for this submodel
            X_sub = oti.array(X_train[point_indices])

            # Add OTI perturbations for automatic differentiation
            for i in range(self.config.n_bases):
                for j in range(X_sub.shape[0]):
                    X_sub[j, i] += oti.e(i + 1, order=self.config.n_order)

            # Evaluate function with OTI to compute derivatives
            y_hc = oti.array([self.true_function(x, alg=oti) for x in X_sub])

            # Collect function values and derivatives
            y_sub = [y_real]  # Start with function values at all points

            # Extract specified derivatives for submodel points
            for i in range(len(derivative_specs[k])):
                for j in range(len(derivative_specs[k][i])):
                    deriv = y_hc.get_deriv(
                        derivative_specs[k][i][j]).reshape(-1, 1)
                    y_sub.append(deriv)

            y_train_data.append(y_sub)

        return y_train_data

    def build_gp_model(self, X_train: np.ndarray, y_train_data: List,
                       submodel_indices: List[List[int]], derivative_specs: List) -> Dict[str, Any]:
        """
        Construct and optimize the weighted derivative-enhanced GP model.

        Args:
            X_train: Training point coordinates
            y_train_data: Function and derivative data for each submodel
            submodel_indices: Point indices for each submodel
            derivative_specs: Derivative specifications for each submodel

        Returns:
            Optimized hyperparameters
        """
        print("Constructing weighted derivative-enhanced GP model...")
        self.gp_model = wdegp(
            X_train,
            y_train_data,
            self.config.n_order,
            self.config.n_bases,
            submodel_indices,
            derivative_specs,
            normalize=self.config.normalize,
            kernel=self.config.kernel,
            kernel_type=self.config.kernel_type,
        )

        print("Optimizing hyperparameters using particle swarm optimization...")
        params = self.gp_model.optimize_hyperparameters(
            n_restart_optimizer=self.config.n_restart_optimizer,
            swarm_size=self.config.swarm_size
        )

        return params

    def evaluate_model_performance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate model performance on a dense test grid.

        Args:
            params: Optimized hyperparameters

        Returns:
            Dictionary containing predictions, true values, and error metrics
        """
        # Generate dense test grid for evaluation
        x_lin = np.linspace(
            self.config.lb_x, self.config.ub_x, self.config.N_grid)
        y_lin = np.linspace(
            self.config.lb_y, self.config.ub_y, self.config.N_grid)
        X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
        X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

        # Generate predictions
        y_pred, submodel_vals = self.gp_model.predict(
            X_test, params, calc_cov=False, return_submodels=True
        )

        # Compute true function values and error metrics
        y_true = self.true_function(X_test, alg=np)
        nrmse = utils.nrmse(y_true, y_pred)

        # Reshape results for 2D visualization
        y_true_grid = y_true.reshape(self.config.N_grid, self.config.N_grid)
        y_pred_grid = y_pred.reshape(self.config.N_grid, self.config.N_grid)
        abs_error_grid = np.abs(y_true_grid - y_pred_grid)

        results = {
            'X_test': X_test,
            'X1_grid': X1_grid,
            'X2_grid': X2_grid,
            'y_pred': y_pred,
            'y_true': y_true,
            'y_true_grid': y_true_grid,
            'y_pred_grid': y_pred_grid,
            'abs_error_grid': abs_error_grid,
            'submodel_vals': submodel_vals,
            'nrmse': nrmse
        }

        print("NRMSE between model and true function: {:.6f}".format(nrmse))
        return results

    def create_contour_visualization(self, results: Dict[str, Any]):
        """
        Generate contour plots comparing true function, predictions, and errors.

        Args:
            results: Dictionary containing evaluation results
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # True function contour plot
        c1 = axes[0].contourf(results['X1_grid'], results['X2_grid'],
                              results['y_true_grid'], levels=50, cmap="viridis")
        fig.colorbar(c1, ax=axes[0])
        axes[0].set_title("True Function")
        axes[0].scatter(self.X_train[:, 0], self.X_train[:, 1],
                        c="red", edgecolor="k", s=50, label="Training Points")

        # GP prediction contour plot
        c2 = axes[1].contourf(results['X1_grid'], results['X2_grid'],
                              results['y_pred_grid'], levels=50, cmap="viridis")
        fig.colorbar(c2, ax=axes[1])
        axes[1].set_title("GP Prediction")
        axes[1].scatter(self.X_train[:, 0], self.X_train[:, 1],
                        c="red", edgecolor="k", s=50)

        # Absolute error contour plot
        c3 = axes[2].contourf(results['X1_grid'], results['X2_grid'],
                              results['abs_error_grid'], levels=50, cmap="magma")
        fig.colorbar(c3, ax=axes[2])
        axes[2].set_title("Absolute Prediction Error")
        axes[2].scatter(self.X_train[:, 0], self.X_train[:, 1],
                        c="red", edgecolor="k", s=50)

        # Set axis labels for all subplots
        for ax in axes:
            ax.set_xlabel("x₁")
            ax.set_ylabel("x₂")

        axes[0].legend()
        plt.tight_layout()
        plt.show()

    def visualize_point_reordering(self):
        """
        Visualize the strategic point reordering process.

        Shows before and after configurations to illustrate how interior points
        are moved to contiguous indices for derivative computation.
        """
        # Generate the standard grid ordering
        x_vals = np.linspace(
            self.config.lb_x, self.config.ub_x, self.config.num_points)
        y_vals = np.linspace(
            self.config.lb_y, self.config.ub_y, self.config.num_points)
        X_standard = np.array(list(itertools.product(x_vals, y_vals)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Standard grid ordering
        ax1.scatter(X_standard[:, 0], X_standard[:, 1],
                    c='lightblue', s=100, edgecolor='black', alpha=0.7)
        ax1.scatter(X_standard[[5, 6, 9, 10], 0], X_standard[[5, 6, 9, 10], 1],
                    c='red', s=150, edgecolor='black',
                    label='Interior Points [5,6,9,10]')
        ax1.scatter(X_standard[[12, 13, 14, 15], 0], X_standard[[12, 13, 14, 15], 1],
                    c='green', s=150, edgecolor='black',
                    label='Target Indices [12,13,14,15]')

        # Add point index annotations
        for i, point in enumerate(X_standard):
            ax1.annotate(str(i), (point[0], point[1]), xytext=(5, 5),
                         textcoords='offset points', fontsize=8, weight='bold')

        ax1.set_title("Standard Grid Ordering")
        ax1.set_xlabel("x₁")
        ax1.set_ylabel("x₂")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # After strategic reordering
        ax2.scatter(self.X_train[:, 0], self.X_train[:, 1],
                    c='lightblue', s=100, edgecolor='black', alpha=0.7)
        ax2.scatter(self.X_train[[12, 13, 14, 15], 0], self.X_train[[12, 13, 14, 15], 1],
                    c='red', s=150, edgecolor='black',
                    label='Derivative Computation Points [12,13,14,15]')

        # Add point index annotations
        for i, point in enumerate(self.X_train):
            ax2.annotate(str(i), (point[0], point[1]), xytext=(5, 5),
                         textcoords='offset points', fontsize=8, weight='bold')

        ax2.set_title(
            "Strategic Reordering\n(Derivatives at Contiguous Indices)")
        ax2.set_xlabel("x₁")
        ax2.set_ylabel("x₂")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run_complete_experiment(self, show_reordering: bool = True) -> Dict[str, Any]:
        """
        Execute the complete 2D sparse derivative-enhanced GP experiment.

        Args:
            show_reordering: Whether to display point reordering visualization

        Returns:
            Dictionary containing all experimental results
        """
        print("=== 2D Sparse Derivative-Enhanced GP Experiment ===")

        # Step 1: Create training grid with strategic derivative placement
        print("Creating 2D training grid with strategic derivative placement...")
        X_train, submodel_indices, derivative_specs = self.create_2d_training_grid()
        self.X_train = X_train

        # Visualize point reordering if requested
        if show_reordering:
            self.visualize_point_reordering()

        # Step 2: Prepare training data with derivatives
        print("Preparing training data with function values and derivatives...")
        y_train_data = self.prepare_submodel_data(
            X_train, submodel_indices, derivative_specs)
        self.y_train_data = y_train_data

        # Display data structure information
        print("Training data structure:")
        print(f"  Number of submodels: {len(y_train_data)}")
        for i, submodel_data in enumerate(y_train_data):
            print(f"  Submodel {i}: {len(submodel_data)} data arrays")
            for j, data_array in enumerate(submodel_data):
                if j == 0:
                    print(
                        f"    Array {j}: Function values, shape {data_array.shape}")
                else:
                    print(
                        f"    Array {j}: Derivative data, shape {data_array.shape}")

        # Step 3: Build and optimize GP model
        params = self.build_gp_model(
            X_train, y_train_data, submodel_indices, derivative_specs)

        # Step 4: Evaluate model performance
        print("Evaluating model performance on test grid...")
        results = self.evaluate_model_performance(params)
        self.results = results

        # Step 5: Generate visualizations
        print("Creating contour plot visualizations...")
        self.create_contour_visualization(results)

        print("Experiment completed successfully!")
        return results


def six_hump_camelback_function(X, alg=oti):
    """
    Six-hump camelback function - a standard optimization test function.

    This function has six local minima and is commonly used for testing
    optimization and modeling algorithms in 2D.

    Args:
        X: Input points with shape (n_points, 2)
        alg: Algorithm library (numpy or pyoti)

    Returns:
        Function values at input points
    """
    x1 = X[:, 0]
    x2 = X[:, 1]
    return (
        (4 - 2.1 * x1**2 + (x1**4) / 3.0) * x1**2
        + x1 * x2
        + (-4 + 4 * x2**2) * x2**2
    )


def main():
    """
    Demonstrate the 2D sparse derivative-enhanced GP implementation.

    This example shows how to:
    1. Configure a 2D GP experiment
    2. Apply strategic derivative placement
    3. Train and evaluate the model
    4. Visualize results with contour plots
    """
    print("2D Sparse Derivative-Enhanced Gaussian Process Demo")
    print("=" * 55)

    # Configure the experiment parameters
    config = GPConfig2D()

    # Create and run the experiment
    experiment = SparseDerivative2DGP(config, six_hump_camelback_function)
    results = experiment.run_complete_experiment(show_reordering=True)

    # Display comprehensive results summary
    print(f"\n=== Experiment Results Summary ===")
    print(
        f"Domain: [{config.lb_x}, {config.ub_x}] × [{config.lb_y}, {config.ub_y}]")
    print(
        f"Training grid: {config.num_points}×{config.num_points} = {config.num_points**2} points")
    print(
        f"Test grid: {config.N_grid}×{config.N_grid} = {config.N_grid**2} points")
    print(f"Maximum derivative order: {config.n_order}")
    print(f"Input dimensions: {config.n_bases}")
    print(f"Final NRMSE: {results['nrmse']:.6f}")

    # Analyze the strategic point placement
    print(f"\n=== Strategic Derivative Placement Analysis ===")
    standard_grid = np.array(list(itertools.product(
        np.linspace(config.lb_x, config.ub_x, config.num_points),
        np.linspace(config.lb_y, config.ub_y, config.num_points)
    )))

    print(f"Interior points in standard ordering [5,6,9,10] located at:")
    for idx in [5, 6, 9, 10]:
        x, y = standard_grid[idx]
        print(f"  Point {idx}: ({x:.2f}, {y:.2f})")

    print(f"After reordering, derivatives computed at indices [12,13,14,15]:")
    for idx in [12, 13, 14, 15]:
        x, y = experiment.X_train[idx]
        print(f"  Point {idx}: ({x:.2f}, {y:.2f})")

    print(f"\nDerivative information enhances model accuracy in the interior region")
    print(f"where function behavior is most complex and informative.")


if __name__ == "__main__":
    main()
