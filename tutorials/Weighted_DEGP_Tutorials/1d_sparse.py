"""
Improved Sparse Derivative-Enhanced Gaussian Process Implementation
================================================================

This module provides a clean, modular implementation of weighted, derivative-enhanced 
Gaussian Process (GP) models using pyOTI-based automatic differentiation.

Key Features:
- Modular design with separate classes for configuration and data preparation
- Improved error handling and validation
- Better documentation and type hints
- Configurable experimental setup
- Enhanced visualization capabilities

Author: [Your Name]
Date: [Current Date]
"""

import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Any
import pyoti.sparse as oti
from wdegp.wdegp import wdegp
import utils
import plotting_helper
from dataclasses import dataclass


@dataclass
class GPConfig:
    """Configuration class for GP model parameters."""
    n_order: int = 2
    n_bases: int = 1
    lb_x: float = 0.5
    ub_x: float = 2.5
    num_points: int = 10
    kernel: str = "SE"
    kernel_type: str = "isotropic"
    normalize: bool = True
    n_restart_optimizer: int = 15
    swarm_size: int = 200
    test_points: int = 250


class DerivativeGPExperiment:
    """
    Main class for running derivative-enhanced GP experiments.

    This class handles the complete workflow from data generation to model
    evaluation and visualization.
    """

    def __init__(self, config: GPConfig, true_function: Callable):
        """
        Initialize the experiment.

        Args:
            config: GPConfig object containing all model parameters
            true_function: The true function to approximate
        """
        self.config = config
        self.true_function = true_function
        self.gp_model = None
        self.training_data = None

    def generate_training_data(self,
                               submodel_indices: List[List[int]],
                               point_swaps: Optional[Dict[str, List[int]]] = None) -> Tuple[np.ndarray, List]:
        """
        Generate training data with derivative information.

        Args:
            submodel_indices: List of point indices for each submodel
            point_swaps: Optional dictionary with 'from' and 'to' keys for point swapping

        Returns:
            Tuple of (X_train, y_train_data)
        """
        # Create training inputs
        X_train = np.linspace(self.config.lb_x, self.config.ub_x,
                              self.config.num_points).reshape(-1, 1)

        # Apply point swaps if specified
        if point_swaps:
            X_train = self._apply_point_swaps(X_train, point_swaps)

        # Validate submodel indices
        self._validate_submodel_indices(submodel_indices, len(X_train))

        # Generate derivative indices for all submodels
        base_der_indices = utils.gen_OTI_indices(
            self.config.n_bases, self.config.n_order)
        der_indices = [base_der_indices for _ in submodel_indices]

        # Prepare training data for each submodel
        y_train_data = []
        y_real = self.true_function(X_train, alg=np)

        for k, point_indices in enumerate(submodel_indices):
            y_sub_data = self._prepare_submodel_data(
                X_train, y_real, point_indices, base_der_indices
            )
            y_train_data.append(y_sub_data)

        self.training_data = {
            'X_train': X_train,
            'y_train_data': y_train_data,
            'submodel_indices': submodel_indices,
            'der_indices': der_indices
        }

        return X_train, y_train_data

    def _apply_point_swaps(self, X_train: np.ndarray, point_swaps: Dict[str, List[int]]) -> np.ndarray:
        """Apply point swapping to training data."""
        X_swapped = X_train.copy()
        from_indices = point_swaps.get('from', [])
        to_indices = point_swaps.get('to', [])

        if len(from_indices) != len(to_indices):
            raise ValueError("Number of 'from' and 'to' indices must match")

        X_swapped[from_indices], X_swapped[to_indices] = X_train[to_indices], X_train[from_indices]
        return X_swapped

    def _validate_submodel_indices(self, submodel_indices: List[List[int]], n_points: int):
        """Validate that submodel indices are valid and non-overlapping."""
        all_indices = []
        for indices in submodel_indices:
            all_indices.extend(indices)

        # Check for valid range
        if any(idx < 0 or idx >= n_points for idx in all_indices):
            raise ValueError(f"All indices must be in range [0, {n_points-1}]")

        # Check for duplicates (current implementation doesn't support overlaps)
        if len(all_indices) != len(set(all_indices)):
            raise ValueError(
                "Current implementation doesn't support overlapping submodels")

    def _prepare_submodel_data(self,
                               X_train: np.ndarray,
                               y_real: np.ndarray,
                               point_indices: List[int],
                               base_der_indices: List) -> List:
        """Prepare training data for a single submodel including derivatives."""
        X_sub = oti.array(X_train[point_indices])

        # Add OTI perturbations for derivative calculation
        for i in range(self.config.n_bases):
            for j in range(X_sub.shape[0]):
                X_sub[j, i] += oti.e(i + 1, order=self.config.n_order)

        # Evaluate function with OTI
        y_hc = oti.array([self.true_function(x, alg=oti) for x in X_sub])

        # Collect function values and derivatives
        y_sub = [y_real.reshape(-1, 1)]
        for i in range(len(base_der_indices)):
            for j in range(len(base_der_indices[i])):
                deriv = y_hc.get_deriv(base_der_indices[i][j]).reshape(-1, 1)
                y_sub.append(deriv)

        return y_sub

    def build_model(self) -> wdegp:
        """Build the weighted derivative-enhanced GP model."""
        if self.training_data is None:
            raise ValueError(
                "Training data not generated. Call generate_training_data() first.")

        self.gp_model = wdegp(
            self.training_data['X_train'],
            self.training_data['y_train_data'],
            self.config.n_order,
            self.config.n_bases,
            self.training_data['submodel_indices'],
            self.training_data['der_indices'],
            normalize=self.config.normalize,
            kernel=self.config.kernel,
            kernel_type=self.config.kernel_type,
        )
        return self.gp_model

    def optimize_hyperparameters(self) -> Dict[str, Any]:
        """Optimize model hyperparameters."""
        if self.gp_model is None:
            raise ValueError("Model not built. Call build_model() first.")

        print("Optimizing hyperparameters...")
        params = self.gp_model.optimize_hyperparameters(
            optimizer = 'pso',
            max_iter=self.config.n_restart_optimizer,
            pop_size=self.config.swarm_size
        )
        print("Hyperparameter optimization complete.")
        return params

    def predict_and_evaluate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make predictions and evaluate model performance."""
        if self.gp_model is None:
            raise ValueError("Model not built. Call build_model() first.")

        # Generate test points
        X_test = np.linspace(self.config.lb_x, self.config.ub_x,
                             self.config.test_points).reshape(-1, 1)

        # Make predictions
        y_pred, y_cov, submodel_vals, submodel_cov = self.gp_model.predict(
            X_test, params, calc_cov=True, return_submodels=True
        )

        # Calculate true values and error metrics
        y_true = self.true_function(X_test, alg=np)
        nrmse = utils.nrmse(y_true, y_pred)

        results = {
            'X_test': X_test,
            'y_pred': y_pred,
            'y_true': y_true,
            'y_cov': y_cov,
            'submodel_vals': submodel_vals,
            'submodel_cov': submodel_cov,
            'nrmse': nrmse
        }

        print(f"NRMSE between model and true function: {nrmse:.6f}")
        return results

    def visualize_results(self, results: Dict[str, Any], plot_submodels: bool = True):
        """Generate visualization plots."""
        plotting_helper.make_submodel_plots(
            self.training_data['X_train'],
            self.training_data['y_train_data'],
            results['X_test'],
            results['y_pred'],
            self.true_function,
            cov=results['y_cov'],
            n_order=self.config.n_order,
            n_bases=self.config.n_bases,
            plot_submodels=plot_submodels,
        )

    def run_complete_experiment(self,
                                submodel_indices: List[List[int]],
                                point_swaps: Optional[Dict[str,
                                                           List[int]]] = None,
                                plot_submodels: bool = True) -> Dict[str, Any]:
        """
        Run the complete experiment workflow.

        Args:
            submodel_indices: List of point indices for each submodel
            point_swaps: Optional point swapping configuration
            plot_submodels: Whether to plot individual submodels

        Returns:
            Dictionary containing all results
        """
        print("Starting derivative-enhanced GP experiment...")

        # Generate training data
        print("Generating training data...")
        self.generate_training_data(submodel_indices, point_swaps)

        # Build model
        print("Building GP model...")
        self.build_model()

        # Optimize hyperparameters
        params = self.optimize_hyperparameters()

        # Make predictions and evaluate
        print("Making predictions...")
        results = self.predict_and_evaluate(params)

        # Visualize results
        print("Generating plots...")
        self.visualize_results(results, plot_submodels)

        print("Experiment complete!")
        return results


def example_true_function(X, alg=oti):
    """Example true function for demonstration."""
    x1 = X[:, 0]
    return alg.sin(10 * np.pi * x1) / (2 * x1) + (x1 - 1) ** 4


def main():
    """Main function demonstrating the improved implementation."""
    # Configure the experiment
    config = GPConfig(
        n_order=2,
        n_bases=1,
        lb_x=0.5,
        ub_x=2.5,
        num_points=10,
        kernel="SE",
        kernel_type="anisotropic",
        normalize=True,
        n_restart_optimizer=15,
        swarm_size=200,
        test_points=250
    )

    # Define the experiment
    experiment = DerivativeGPExperiment(config, example_true_function)

    # Define submodel structure
    # Points to include derivative information
    submodel_indices = [[2, 3, 4, 5]]

    # Optional point swapping
    point_swaps = {
        'from': [2, 3, 4],
        'to': [5, 6, 7]
    }

    # Run the complete experiment
    results = experiment.run_complete_experiment(
        submodel_indices=submodel_indices,
        point_swaps=point_swaps,
        plot_submodels=False
    )

    # Print summary
    print(f"\nExperiment Summary:")
    print(f"- Number of training points: {config.num_points}")
    print(f"- Number of test points: {config.test_points}")
    print(f"- Derivative order: {config.n_order}")
    print(f"- NRMSE: {results['nrmse']:.6f}")


if __name__ == "__main__":
    main()
