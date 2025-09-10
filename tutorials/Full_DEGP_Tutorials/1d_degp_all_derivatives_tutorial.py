"""
================================================================================
DEGP Tutorial: Introduction to Derivative Enhanced Gaussian Processes
================================================================================

This tutorial demonstrates the fundamental benefit of using derivative information
to improve Gaussian Process regression. We will train several GP models on the
same small set of training points, systematically including more derivative
information (from order 0 up to order 4) to show how the quality and reliability
of predictions improve.

Key Concepts Demonstrated:
- Training a standard GP (Order 0) vs. a DEGP (Order > 0).
- Using automatic differentiation to generate derivative training data.
- The direct impact of higher-order derivative information on model accuracy.
- The trade-off between model accuracy and computational cost.
- The significant advantage of DEGP when training data is sparse.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
import time
from dataclasses import dataclass, field
from typing import List, Dict, Callable

# Set plotting parameters for better readability
plt.rcParams.update({'font.size': 12})


@dataclass
class DEGPConfig:
    """Configuration for the DEGP comparison tutorial."""
    # Domain and data points
    lb_x: float = 0.2
    ub_x: float = 5.0
    num_training_pts: int = 3
    num_test_pts: int = 100

    # List of derivative orders to test and compare
    orders_to_test: List[int] = field(default_factory=lambda: [0, 1, 2, 4])

    # GP model parameters
    normalize_data: bool = False
    kernel: str = "SE"
    kernel_type: str = "anisotropic"

    # Optimizer settings
    n_restarts: int = 20
    swarm_size: int = 100


class DEGPComparisonTutorial:
    """
    Manages the experiment comparing DEGP models with varying derivative orders.
    """

    def __init__(self, config: DEGPConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.results: Dict[int, Dict] = {}
        self.X_train = np.linspace(
            config.lb_x, config.ub_x, config.num_training_pts).reshape(-1, 1)
        self.X_test = np.linspace(
            config.lb_x, config.ub_x, config.num_test_pts).reshape(-1, 1)

    def _train_and_evaluate_single_model(self, n_order: int):
        """
        Trains and evaluates a single DEGP model for a specific derivative order.
        """
        print(f"Processing Order {n_order}...")
        start_time = time.time()

        # 1. Generate Training Data with Derivatives
        der_indices = utils.gen_OTI_indices(1, n_order)
        X_train_pert = oti.array(self.X_train) + oti.e(1, order=n_order)
        y_train_hc = self.true_function(X_train_pert)

        y_train_list = [y_train_hc.real]
        for i in range(len(der_indices)):
            for j in range(len(der_indices[i])):
                derivative = y_train_hc.get_deriv(
                    der_indices[i][j]).reshape(-1, 1)
                y_train_list.append(derivative)

        # 2. Initialize and Train DEGP Model
        gp = degp(
            self.X_train, y_train_list, n_order, n_bases=1, der_indices=der_indices,
            normalize=self.config.normalize_data, kernel=self.config.kernel,
            kernel_type=self.config.kernel_type
        )
        params = gp.optimize_hyperparameters(
            n_restart_optimizer=self.config.n_restarts, swarm_size=self.config.swarm_size
        )

        # 3. Make Predictions
        y_pred, y_var = gp.predict(self.X_test, params, calc_cov=True)

        # 4. Calculate and store metrics
        y_true_flat = self.true_function(self.X_test, alg=np).ravel()
        mse = np.mean((y_pred.ravel() - y_true_flat)**2)

        self.results[n_order] = {
            'y_pred': y_pred, 'y_var': y_var, 'mse': mse,
            'time': time.time() - start_time,
            'n_observations': sum(len(y) for y in y_train_list)
        }
        print(f"  MSE: {mse:.6f}, Time: {self.results[n_order]['time']:.2f}s")

    def run_comparison(self):
        """Runs the full comparison by training a model for each specified order."""
        print("Training DEGP models with different derivative orders...")
        print("-" * 50)
        for n_order in self.config.orders_to_test:
            self._train_and_evaluate_single_model(n_order)
        print()

    def display_summary(self):
        """Prints a formatted summary table of the results."""
        print("RESULTS SUMMARY")
        print("=" * 60)
        print(f"{'Order':<8}{'MSE':<12}{'Time (s)':<10}{'Observations'}")
        print("-" * 60)
        for order, r in self.results.items():
            print(
                f"{order:<8}{r['mse']:<12.6f}{r['time']:<10.2f}{r['n_observations']}")
        print()

    def visualize_results(self):
        """Creates a 2x2 plot comparing the predictions for each derivative order."""
        fig, axs = plt.subplots(2, 2, figsize=(
            14, 10), sharex=True, sharey=True)
        axs = axs.flatten()
        y_true = self.true_function(self.X_test, alg=np)
        y_train_func = self.true_function(self.X_train, alg=np)

        titles = [
            r"Order 0: $f(x)$ only", r"Order 1: $f(x)$, $f'(x)$",
            r"Order 2: $f(x)$, $f'(x)$, $f''(x)$", r"Order 4: $f(x)$, ..., $f^{(4)}(x)$"
        ]

        for i, (order, r) in enumerate(self.results.items()):
            ax = axs[i]
            y_pred, y_var = r['y_pred'], r['y_var']

            ax.plot(self.X_test, y_true, 'k-', lw=2.5, label="True $f(x)$")
            ax.plot(self.X_test, y_pred, 'b--', lw=2, label="GP mean")
            ax.fill_between(
                self.X_test.ravel(),
                y_pred.ravel() - 2 * np.sqrt(y_var.ravel()),
                y_pred.ravel() + 2 * np.sqrt(y_var.ravel()),
                color='blue', alpha=0.15, label='GP 95% CI'
            )
            ax.scatter(self.X_train, y_train_func, c='red', s=60, zorder=5,
                       edgecolors='black', linewidth=1, label="Training points")

            ax.set_title(f"{titles[i]}\nMSE: {r['mse']:.4f}", fontsize=11)
            ax.set(xlabel="$x$", ylabel="$f(x)$")
            ax.grid(True, alpha=0.3)

        fig.suptitle('Derivative Enhanced Gaussian Process Comparison',
                     fontsize=16, fontweight='bold', y=0.98)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(
            0.5, 0.02), ncol=len(handles), frameon=True, fancybox=True, shadow=True)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

    def display_educational_content(self):
        """Prints a final section with key takeaways and concepts."""
        print("\n" + "=" * 60 + "\nKEY TAKEAWAYS\n" + "=" * 60)
        print("This tutorial demonstrates:")
        print("  - Accuracy dramatically improves as derivative information is added,")
        print("    even with very few training points.")
        print("  - Higher-order derivatives provide strong constraints on the function's shape,")
        print("    leading to much lower prediction error and more reliable uncertainty.")
        print(
            "  - The computational cost increases with more observations, but the gain in")
        print(
            "    accuracy can be substantial, especially when data is expensive to acquire.")
        print("\nDEGP is a powerful tool for modeling complex functions from sparse data.")
        print("Tutorial complete!")

    def run(self):
        """Executes the full tutorial workflow."""
        print("DEGP Tutorial: Derivative Enhanced Gaussian Processes")
        print("=" * 60)
        print(
            f"Domain: [{self.config.lb_x}, {self.config.ub_x}] | Training points: {self.config.num_training_pts}")
        print()

        self.run_comparison()
        self.display_summary()
        self.visualize_results()
        self.display_educational_content()


def true_function(X, alg=oti):
    """Test function combining exponential decay, oscillations, and linear trend."""
    x = X[:, 0]
    return alg.exp(-x) + alg.sin(2 * x) + alg.cos(3 * x) + 0.2 * x + 1.0


def main():
    """Main execution block."""
    config = DEGPConfig()
    tutorial = DEGPComparisonTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
