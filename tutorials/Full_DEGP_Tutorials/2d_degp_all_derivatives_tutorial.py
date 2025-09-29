"""
================================================================================
DEGP Tutorial: 2D Derivative-Enhanced Gaussian Process Regression
================================================================================

This tutorial demonstrates how to apply derivative-enhanced Gaussian Process
regression to two-dimensional functions. We'll explore how partial derivatives
in multiple dimensions can dramatically improve function approximation with
limited training data.

Key concepts covered:
- 2D hypercomplex automatic differentiation
- Full partial derivative inclusion (∂f/∂x₁, ∂f/∂x₂, ∂²f/∂x₁∂x₂, etc.)
- Multi-dimensional GP regression with derivative constraints
- Spatial performance analysis and 2D visualization of results
================================================================================
"""

import numpy as np
import pyoti.sparse as oti
import itertools
from full_degp.degp import degp
import utils
import plotting_helper
import time
from dataclasses import dataclass
from typing import List, Dict, Callable


@dataclass
class TwoDimConfig:
    """Configuration for the 2D DEGP tutorial."""
    n_order: int = 2
    n_bases: int = 2
    lb_x: float = -1.0
    ub_x: float = 1.0
    lb_y: float = -1.0
    ub_y: float = 1.0
    num_pts_per_axis: int = 5
    sampling_strategy: str = 'uniform'  # 'uniform', 'chebyshev', or 'random'
    test_grid_resolution: int = 25
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"
    n_restarts: int = 15
    swarm_size: int = 100


class TwoDimDEGPTutorial:
    """
    Manages and executes a detailed, step-by-step 2D DEGP tutorial.
    """

    def __init__(self, config: TwoDimConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.training_data: Dict = {}
        self.gp_model = None
        self.params = None
        self.results: Dict = {}

    def _analyze_derivative_structure(self):
        """Generates and analyzes the full 2D derivative structure."""
        print("\n" + "="*50 + "\nDerivative Structure Analysis\n" + "="*50)
        cfg = self.config

        der_indices = utils.gen_OTI_indices(cfg.n_bases, cfg.n_order)
        self.training_data['der_indices'] = der_indices

        total_derivatives = sum(len(group) for group in der_indices)
        print(f"  Including all derivatives up to order {cfg.n_order}.")
        print(
            f"  Total derivative types per point: {total_derivatives} (including mixed partials like ∂²f/∂x₁∂x₂)")

    def _generate_training_data(self):
        """Generates 2D training data using the specified sampling strategy."""
        print("\n" + "="*50 + "\nGenerating Training Data\n" + "="*50)
        start_time = time.time()
        cfg = self.config

        X_train = self._create_training_grid()
        print(
            f"  Training points generated ({cfg.sampling_strategy} sampling): {X_train.shape}")

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

    def _create_training_grid(self) -> np.ndarray:
        """Helper to create 2D training points based on the configured strategy."""
        cfg = self.config
        n = cfg.num_pts_per_axis
        if cfg.sampling_strategy == 'uniform':
            x_vals = np.linspace(cfg.lb_x, cfg.ub_x, n)
            y_vals = np.linspace(cfg.lb_y, cfg.ub_y, n)
            return np.array(list(itertools.product(x_vals, y_vals)))
        elif cfg.sampling_strategy == 'random':
            np.random.seed(42)
            return np.random.uniform([cfg.lb_x, cfg.lb_y], [cfg.ub_x, cfg.ub_y], (n**2, 2))
        else:  # chebyshev
            k = np.arange(1, n + 1)
            x_cheb = 0.5 * (cfg.lb_x + cfg.ub_x) + 0.5 * (cfg.ub_x -
                                                          cfg.lb_x) * np.cos((2*k - 1) * np.pi / (2*n))
            y_cheb = 0.5 * (cfg.lb_y + cfg.ub_y) + 0.5 * (cfg.ub_y -
                                                          cfg.lb_y) * np.cos((2*k - 1) * np.pi / (2*n))
            return np.array(list(itertools.product(x_cheb, y_cheb)))

    def _train_model(self):
        """Initializes, trains, and optimizes the 2D DEGP model."""
        print("\n" + "="*50 + "\nDEGP Model Setup and Training\n" + "="*50)
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

    def _evaluate_model(self):
        """Makes predictions and computes comprehensive 2D performance metrics."""
        print("\n" + "="*50 + "\nModel Prediction and Evaluation\n" + "="*50)
        cfg = self.config

        x_lin = np.linspace(cfg.lb_x, cfg.ub_x, cfg.test_grid_resolution)
        y_lin = np.linspace(cfg.lb_y, cfg.ub_y, cfg.test_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
        X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

        y_pred = self.gp_model.predict(X_test, self.params, calc_cov=False)
        y_true = self.true_function(X_test, alg=np)

        self.results = self._evaluate_2d_performance(
            y_true, y_pred, (cfg.test_grid_resolution, cfg.test_grid_resolution))
        self.results.update(
            {'X_test': X_test, 'X1_grid': X1_grid, 'X2_grid': X2_grid, 'y_pred': y_pred})

        print("Performance Metrics:")
        print(f"  NRMSE:            {self.results['nrmse']:.6f}")
        print(f"  RMSE:             {self.results['rmse']:.6f}")
        print(f"  Max Error:        {self.results['max_error']:.6f}")

        print("\nSpatial Error Analysis (Mean Absolute Error):")
        print(
            f"  Corner regions:   {self.results['spatial_errors']['corners']:.6f}")
        print(
            f"  Edge regions:     {self.results['spatial_errors']['edges']:.6f}")
        print(
            f"  Center region:    {self.results['spatial_errors']['center']:.6f}")

    def _evaluate_2d_performance(self, y_true, y_pred, grid_shape) -> Dict:
        """Helper to compute 2D-specific metrics."""
        y_true_flat, y_pred_flat = y_true.flatten(), y_pred.flatten()
        errors = np.abs(y_true_flat - y_pred_flat).reshape(grid_shape)
        return {
            'rmse': np.sqrt(np.mean((y_true_flat - y_pred_flat)**2)),
            'nrmse': utils.nrmse(y_true_flat, y_pred_flat),
            'max_error': np.max(np.abs(y_true_flat - y_pred_flat)),
            'spatial_errors': {
                'corners': np.mean([errors[0, 0], errors[0, -1], errors[-1, 0], errors[-1, -1]]),
                'edges': np.mean([np.mean(errors[0, :]), np.mean(errors[-1, :]), np.mean(errors[:, 0]), np.mean(errors[:, -1])]),
                'center': np.mean(errors[grid_shape[0]//4:-grid_shape[0]//4, grid_shape[1]//4:-grid_shape[1]//4])
            }
        }

    def visualize_results(self):
        """Generates 2D surface plots comparing predictions with the ground truth."""
        print("\n" + "="*50 + "\nGenerating 2D Visualizations\n" + "="*50)
        try:
            plotting_helper.make_plots(
                self.training_data['X_train'], self.training_data['y_train_list'],
                self.results['X_test'], self.results['y_pred'], self.true_function,
                X1_grid=self.results['X1_grid'], X2_grid=self.results['X2_grid'],
                n_order=self.config.n_order, n_bases=self.config.n_bases,
                der_indices=self.training_data['der_indices']
            )
            print("  Visualization: SUCCESS")
        except Exception as e:
            print(f"  Visualization: FAILED\n  Error: {e}")

    def run(self):
        """Executes the complete tutorial workflow."""
        start_time = time.time()
        cfg = self.config
        print("2D DEGP Tutorial: Multi-Dimensional Function Approximation")
        print("=" * 70)
        print(
            f"Configuration: {cfg.num_pts_per_axis**2} training points on a {cfg.num_pts_per_axis}x{cfg.num_pts_per_axis} grid ({cfg.sampling_strategy})")

        self._analyze_derivative_structure()
        self._generate_training_data()
        self._train_model()
        self._evaluate_model()
        self.visualize_results()

        print("\n" + "="*60 + "\n2D DEGP Tutorial Summary\n" + "="*60)
        print(f"Total execution time: {time.time() - start_time:.2f}s")
        print(f"Final NRMSE: {self.results['nrmse']:.6f}")
        obs_per_pt = 1 + sum(len(group)
                             for group in self.training_data['der_indices'])
        print(
            f"Training efficiency: {obs_per_pt * cfg.num_pts_per_axis**2} observations from {cfg.num_pts_per_axis**2} points ({obs_per_pt}x multiplier).")


def true_function(X, alg=oti):
    """Complex 2D test function with polynomial and oscillatory components."""
    x1, x2 = X[:, 0], X[:, 1]
    return x1**2 * x2 + alg.cos(10 * x1) + alg.cos(10 * x2)


def main():
    """Main execution block."""
    config = TwoDimConfig(sampling_strategy='uniform')
    tutorial = TwoDimDEGPTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
