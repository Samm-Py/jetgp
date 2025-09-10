"""
================================================================================
DEGP Tutorial: Selective Derivative Strategy in 2D Function Approximation
================================================================================

This tutorial demonstrates an advanced DEGP technique: selective derivative
inclusion. Instead of using ALL possible derivatives up to a given order,
we strategically select only the most informative derivatives to balance
accuracy and computational efficiency.

Key concepts covered:
- Strategic derivative selection (e.g., 'gradient_only', 'main_derivatives', 'complete').
- Computational trade-offs in multi-dimensional derivative spaces.
- Performance comparison between different derivative strategies.
- 2D visualization of a DEGP model with a selective derivative set.
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
class StrategyConfig:
    """Configuration for the 2D DEGP derivative strategy tutorial."""
    n_order: int = 2
    n_bases: int = 2
    lb_x: float = -1.0
    ub_x: float = 1.0
    lb_y: float = -1.0
    ub_y: float = 1.0
    num_pts_per_axis: int = 5
    test_grid_resolution: int = 25
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"
    n_restarts: int = 15
    swarm_size: int = 150


class DerivativeStrategyTutorial:
    """
    Manages and executes a comparative study of different derivative
    selection strategies for a 2D DEGP model.
    """

    def __init__(self, config: StrategyConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.strategies: Dict = {}
        self.results: Dict = {}
        self.X_train: np.ndarray
        self.X_test: np.ndarray
        self.grids: Dict = {}

    def _define_and_analyze_strategies(self):
        """Creates and analyzes different derivative selection strategies."""
        print("\n" + "="*50 + "\nDerivative Strategy Analysis\n" + "="*50)
        cfg = self.config

        # Strategy 1: Gradients only
        self.strategies['gradient_only'] = [[[[1, 1]], [[2, 1]]]]
        # Strategy 2: Main derivatives (no mixed terms)
        self.strategies['main_derivatives'] = [
            [[[1, 1]], [[2, 1]]],
            [[[1, 2]], [[2, 2]]]
        ]
        # Strategy 3: Complete set of derivatives
        self.strategies['complete'] = utils.gen_OTI_indices(
            cfg.n_bases, cfg.n_order)

        for name, der_indices in self.strategies.items():
            count = sum(len(group) for group in der_indices)
            print(f"Strategy: {name.upper()}")
            print(f"  Derivatives per point: {count}")
            print(
                f"  Computational trade-off: {'Low' if count <= 2 else 'Medium' if count <= 4 else 'High'}")

    def _run_single_experiment(self, strategy_name: str, der_indices: List) -> Dict:
        """Runs a complete DEGP experiment for a single derivative strategy."""
        print(f"\n--- Running Experiment for '{strategy_name}' Strategy ---")
        start_time = time.time()
        cfg = self.config

        try:
            # Generate training data for this specific strategy
            X_train_pert = oti.array(self.X_train)
            for i in range(cfg.n_bases):
                X_train_pert[:, i] += oti.e(i + 1, order=cfg.n_order)
            y_train_hc = self.true_function(X_train_pert)
            y_train = [y_train_hc.real]
            for group in der_indices:
                for sub_group in group:
                    y_train.append(y_train_hc.get_deriv(sub_group))

            # Initialize and train model
            gp = degp(self.X_train, y_train, cfg.n_order, cfg.n_bases, der_indices,
                      normalize=cfg.normalize_data, kernel=cfg.kernel, kernel_type=cfg.kernel_type)
            params = gp.optimize_hyperparameters(
                n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size)

            # Predict and evaluate
            y_pred = gp.predict(self.X_test, params, calc_cov=False)
            y_true = self.true_function(self.X_test, alg=np)
            nrmse = utils.nrmse(y_true, y_pred)

            return {
                'success': True, 'nrmse': nrmse, 'y_pred': y_pred,
                'total_time': time.time() - start_time,
                'total_obs': sum(len(d) for d in y_train)
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def run_comparison(self):
        """Executes the DEGP experiment for all defined strategies."""
        cfg = self.config
        self._define_and_analyze_strategies()

        # Generate common training and test data
        x_vals = np.linspace(cfg.lb_x, cfg.ub_x, cfg.num_pts_per_axis)
        y_vals = np.linspace(cfg.lb_y, cfg.ub_y, cfg.num_pts_per_axis)
        self.X_train = np.array(list(itertools.product(x_vals, y_vals)))

        x_lin = np.linspace(cfg.lb_x, cfg.ub_x, cfg.test_grid_resolution)
        y_lin = np.linspace(cfg.lb_y, cfg.ub_y, cfg.test_grid_resolution)
        self.grids['X1_grid'], self.grids['X2_grid'] = np.meshgrid(
            x_lin, y_lin)
        self.X_test = np.column_stack(
            [self.grids['X1_grid'].ravel(), self.grids['X2_grid'].ravel()])

        print("\n" + "="*55 + "\nRunning Strategy Comparison Experiments\n" + "="*55)
        for name, der_indices in self.strategies.items():
            result = self._run_single_experiment(name, der_indices)
            self.results[name] = result
            if result['success']:
                print(
                    f"  Result: NRMSE = {result['nrmse']:.6f}, Time = {result['total_time']:.2f}s")
            else:
                print(f"  Result: FAILED - {result['error']}")

    def display_summary(self):
        """Prints a formatted summary table comparing the strategies."""
        print("\n" + "="*60 + "\nStrategy Comparison Results\n" + "="*60)
        successful_results = {
            k: v for k, v in self.results.items() if v.get('success', False)}
        if not successful_results:
            print("No experiments completed successfully.")
            return

        print(f"{'Strategy':<20} {'NRMSE':<12} {'Time(s)':<10} {'Total Obs':<12} {'Efficiency (NRMSE*Time)':<25}")
        print("-" * 80)

        for name in ['gradient_only', 'main_derivatives', 'complete']:
            if name in successful_results:
                r = successful_results[name]
                efficiency = r['nrmse'] * r['total_time']
                print(
                    f"{name:<20} {r['nrmse']:<12.6f} {r['total_time']:<10.2f} {r['total_obs']:<12} {efficiency:<25.4f}")

        print("\nStrategy Selection Guidelines:")
        print(
            "  - Use 'gradient_only' for simple functions or a tight computational budget.")
        print("  - Use 'main_derivatives' for a strong balance between accuracy and efficiency.")
        print("  - Use 'complete' when maximum accuracy is required and cross-derivative effects are strong.")

    def visualize_strategy(self, strategy_name: str = 'main_derivatives'):
        """Generates 2D surface plots for a specified strategy."""
        print(f"\nGenerating Visualization for '{strategy_name}' strategy...")
        if strategy_name not in self.results or not self.results[strategy_name]['success']:
            print(
                f"  Cannot visualize '{strategy_name}' as it did not run successfully.")
            return

        try:
            result = self.results[strategy_name]
            der_indices = self.strategies[strategy_name]
            plotting_helper.make_plots(
                self.X_train, [result['y_pred']
                               ], self.X_test, result['y_pred'],
                self.true_function, X1_grid=self.grids['X1_grid'], X2_grid=self.grids['X2_grid'],
                n_order=self.config.n_order, n_bases=self.config.n_bases, der_indices=der_indices
            )
            print("  Visualization: SUCCESS")
        except Exception as e:
            print(f"  Visualization: FAILED\n  Error: {e}")

    def run(self):
        """Executes the complete tutorial workflow."""
        cfg = self.config
        print("Selective Derivative DEGP Tutorial: Strategic Choice of Derivatives")
        print("=" * 75)
        print(
            f"Configuration: {cfg.num_pts_per_axis**2} training points, up to order {cfg.n_order} derivatives.")

        self.run_comparison()
        self.display_summary()
        self.visualize_strategy(strategy_name='main_derivatives')


def true_function(X, alg=oti):
    """Complex 2D test function with polynomial and oscillatory components."""
    x1, x2 = X[:, 0], X[:, 1]
    return x1**2 * x2 + alg.cos(10 * x1) + alg.cos(10 * x2)


def main():
    """Main execution block."""
    config = StrategyConfig(
        n_order=2)  # Set max order for the 'complete' strategy
    tutorial = DerivativeStrategyTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
