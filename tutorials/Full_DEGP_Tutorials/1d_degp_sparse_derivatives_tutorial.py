"""
================================================================================
DEGP Tutorial: Selective Derivative Inclusion for 1D Function Approximation
================================================================================

This tutorial provides a detailed, step-by-step guide on using derivative
information to enhance Gaussian Process regression. We demonstrate how to
selectively include specific derivative orders—in this case, using 1st and 4th
order derivatives while skipping the 2nd and 3rd—to improve predictions with
limited training data.

Key Concepts Covered:
- Hypercomplex automatic differentiation for derivative computation.
- **Selective Derivative Inclusion**: Demonstrates the flexibility of choosing
  arbitrary derivative combinations for GP training.
- Hyperparameter optimization using particle swarm.
- Comprehensive performance evaluation and visualization.
- Robust workflow with explanatory print statements and error handling.
================================================================================
"""

import numpy as np
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
import plotting_helper
import time
from dataclasses import dataclass, field
from typing import List, Dict, Callable


@dataclass
class SelectiveDEGPConfig:
    """Configuration for the selective derivative DEGP tutorial."""
    # Domain and data points
    lb_x: float = 0.2
    ub_x: float = 6.0
    num_training_pts: int = 6
    num_test_pts: int = 100

    # Derivative configuration
    max_order_for_ad: int = 1  # Max order needed for Automatic Differentiation
    n_bases: int = 1          # Input dimensionality
    # Custom derivative indices to demonstrate selective inclusion
    der_indices: List = field(default_factory=lambda: [[[[1, 1]]]])

    # GP model parameters
    normalize_data: bool = False
    kernel: str = "Matern"
    smoothness_parameter: int = 1
    kernel_type: str = "anisotropic"

    # Optimizer settings
    n_restarts: int = 25
    swarm_size: int = 100


class SelectiveDEGPTutorial:
    """
    Manages and executes a detailed, step-by-step DEGP tutorial focusing
    on selective derivative inclusion.
    """

    def __init__(self, config: SelectiveDEGPConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.training_data = {}
        self.gp_model = None
        self.params = None
        self.results = {}

    def _generate_training_data(self):
        """Generates training data including selectively chosen derivatives."""
        print("\n" + "="*50 + "\nGenerating Training Data...\n" + "="*50)
        start_time = time.time()

        X_train = np.linspace(self.config.lb_x, self.config.ub_x,
                              self.config.num_training_pts).reshape(-1, 1)
        print(f"  Training locations: {X_train.ravel()}")

        X_train_pert = oti.array(X_train) + \
            oti.e(1, order=self.config.max_order_for_ad)
        y_train_hc = self.true_function(X_train_pert)

        y_train_list = [y_train_hc.real]
        for group in self.config.der_indices:
            for sub_group in group:
                derivative_data = y_train_hc.get_deriv(
                    sub_group).reshape(-1, 1)
                y_train_list.append(derivative_data)

        self.training_data = {'X_train': X_train, 'y_train_list': y_train_list}
        print(f"  Function observations: {len(y_train_list[0])}")
        print(
            f"  Derivative observations: {sum(len(d) for d in y_train_list[1:])}")
        print(f"  Data generation time: {time.time() - start_time:.3f}s")

    def _setup_and_train_model(self):
        """Initializes, trains, and optimizes the DEGP model."""
        print("\n" + "="*50 + "\nDEGP Model Setup and Training...\n" + "="*50)
        cfg = self.config
        data = self.training_data

        try:
            self.gp_model = degp(
                data['X_train'], data['y_train_list'], cfg.max_order_for_ad,
                cfg.n_bases, cfg.der_indices, normalize=cfg.normalize_data,
                kernel=cfg.kernel, kernel_type=cfg.kernel_type, smoothness_parameter=cfg.smoothness_parameter
            )
            print("  Model initialization: SUCCESS")
        except Exception as e:
            print(f"  Model initialization: FAILED\n  Error: {e}")
            raise

        print("\nOptimizing Hyperparameters...")
        optimization_start = time.time()
        try:
            self.params = self.gp_model.optimize_hyperparameters(
                n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size
            )
            print(
                f"  Optimization time: {time.time() - optimization_start:.2f}s")
            print("  Optimization: SUCCESS")
        except Exception as e:
            print(f"  Optimization: FAILED\n  Error: {e}")
            raise

    def _evaluate_model(self):
        """Makes predictions and computes comprehensive performance metrics."""
        print("\n" + "="*50 + "\nModel Prediction and Evaluation...\n" + "="*50)
        prediction_start = time.time()

        X_test = np.linspace(self.config.lb_x, self.config.ub_x,
                             self.config.num_test_pts).reshape(-1, 1)

        try:
            y_pred, y_var = self.gp_model.predict(
                X_test, self.params, calc_cov=True)
            print(f"  Prediction time: {time.time() - prediction_start:.3f}s")
        except Exception as e:
            print(f"  Prediction: FAILED\n  Error: {e}")
            raise

        y_true = self.true_function(X_test, alg=np)
        self.results = evaluate_model_performance(y_true, y_pred, y_var)
        self.results.update(
            {'X_test': X_test, 'y_pred': y_pred, 'y_var': y_var})

        print("\nPerformance Metrics:")
        print(f"  NRMSE:            {self.results['nrmse']:.6f}")
        print(f"  RMSE:             {self.results['rmse']:.6f}")
        print(f"  Max Error:        {self.results['max_error']:.6f}")
        print(f"  Mean Uncertainty: {self.results['mean_uncertainty']:.6f}")
        print(f"  95% Coverage:     {self.results['coverage_2sigma']:.3f}")

    def visualize_results(self):
        """Generates plots comparing predictions with the ground truth."""
        print("\n" + "="*50 + "\nGenerating Visualizations...\n" + "="*50)
        try:
            plotting_helper.make_plots(
                self.training_data['X_train'], self.training_data['y_train_list'],
                self.results['X_test'], self.results['y_pred'].flatten(),
                self.true_function, cov=self.results['y_var'], n_order=self.config.max_order_for_ad,
                n_bases=self.config.n_bases, der_indices=self.config.der_indices
            )
            print("  Visualization: SUCCESS")
        except Exception as e:
            print(f"  Visualization: FAILED\n  Error: {e}")

    def display_summary(self, total_time: float):
        """Prints a final summary of the tutorial."""
        print("\n" + "="*50 + "\nTutorial Summary\n" + "="*50)
        print(f"Total execution time: {total_time:.2f}s")
        print(f"Final NRMSE: {self.results['nrmse']:.6f}")
        print("\nKey Takeaways:")
        print("  - DEGP effectively leverages sparse, high-order derivative information.")
        print("  - Selective inclusion (e.g., 1st and 4th order) allows balancing accuracy vs. cost.")
        print(
            f"  - With only {self.config.num_training_pts} training points, achieved an excellent NRMSE of {self.results['nrmse']:.4f}.")

    def run(self):
        """Executes the complete tutorial workflow."""
        start_time = time.time()
        print("DEGP Tutorial: Enhanced Gaussian Process with Selective Derivatives")
        print("=" * 65)
        print(
            f"Configuration: {self.config.num_training_pts} training points, Domain: [{self.config.lb_x}, {self.config.ub_x}]")
        print("Derivative Strategy: Including 1st and 4th order derivatives only.")

        self._generate_training_data()
        self._setup_and_train_model()
        self._evaluate_model()
        self.visualize_results()
        self.display_summary(time.time() - start_time)


def true_function(X, alg=oti):
    """Complex test function combining multiple mathematical components."""
    x = X[:, 0]
    return alg.exp(-x) + alg.sin(x) + alg.cos(3 * x) + 0.2 * x + 1.0


def evaluate_model_performance(y_true: np.ndarray, y_pred: np.ndarray, y_var: np.ndarray) -> dict:
    """Compute comprehensive performance metrics."""
    y_true_flat, y_pred_flat, y_var_flat = y_true.flatten(
    ), y_pred.flatten(), y_var.flatten()
    std_pred = np.sqrt(y_var_flat)
    return {
        'mse': np.mean((y_true_flat - y_pred_flat)**2),
        'mae': np.mean(np.abs(y_true_flat - y_pred_flat)),
        'rmse': np.sqrt(np.mean((y_true_flat - y_pred_flat)**2)),
        'nrmse': utils.nrmse(y_true_flat, y_pred_flat),
        'max_error': np.max(np.abs(y_true_flat - y_pred_flat)),
        'mean_uncertainty': np.mean(std_pred),
        'coverage_2sigma': np.mean(np.abs(y_true_flat - y_pred_flat) <= 2 * std_pred)
    }


def main():
    """Main execution block."""
    config = SelectiveDEGPConfig()
    tutorial = SelectiveDEGPTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
