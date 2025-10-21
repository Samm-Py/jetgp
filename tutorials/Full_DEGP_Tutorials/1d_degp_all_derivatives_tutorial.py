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
    lb_x: float = 0.0
    ub_x: float = 6.0
    num_training_pts: int = 4
    num_test_pts: int = 100

    # List of derivative orders to test and compare
    orders_to_test: List[int] = field(default_factory=lambda: [0, 1, 2, 4])

    # GP model parameters
    normalize_data: bool = False
    kernel: str = "SE"
    kernel_type: str = "anisotropic"

    # Optimizer settings
    n_restarts: int = 15
    swarm_size: int = 200


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
        optimizer='jade',
        pop_size = 100,
        n_generations = 15,
        local_opt_every = None,
        debug = True
        )

        # 3. Make Predictions (including derivatives)
        y_pred_full, y_var_full = gp.predict(
            self.X_test, params, calc_cov=True, return_deriv=True
        )

        # 4. Calculate and store metrics
        # MSE is calculated only on the function prediction (0th derivative)
        y_pred_func = y_pred_full[:self.config.num_test_pts]
        y_true_flat = self.true_function(self.X_test, alg=np).ravel()
        mse = np.mean((y_pred_func.ravel() - y_true_flat)**2)

        self.results[n_order] = {
            'y_pred_full': y_pred_full, 'y_var_full': y_var_full, 'mse': mse,
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
            # Extract only the function prediction and variance for this plot
            y_pred = r['y_pred_full'][:self.config.num_test_pts]
            y_var = r['y_var_full'][:self.config.num_test_pts]

            ax.plot(self.X_test, y_true, 'k-', lw=2.5, label="True $f(x)$")
            ax.plot(self.X_test, y_pred.flatten(), 'b--', lw=2, label="GP mean")
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

        fig.suptitle('Function Prediction Comparison',
                     fontsize=16, fontweight='bold', y=0.98)
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(
            0.5, 0.02), ncol=len(handles), frameon=True, fancybox=True, shadow=True)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

    def visualize_derivative_results(self):
        """
        Creates plots comparing the predicted derivatives from each model,
        including uncertainty bounds and training data points, with a single global legend.
        """
        # Determine the maximum derivative order we need to plot
        max_order_to_plot = max(self.config.orders_to_test)

        # 1. Calculate true derivatives for the TEST points (for the true line)
        X_test_pert = oti.array(self.X_test) + \
            oti.e(1, order=max_order_to_plot)
        y_test_hc = self.true_function(X_test_pert)
        true_derivs = {0: y_test_hc.real}
        der_indices_true = utils.gen_OTI_indices(1, max_order_to_plot)
        idx = 1
        for i in range(len(der_indices_true)):
            for j in range(len(der_indices_true[i])):
                true_derivs[idx] = y_test_hc.get_deriv(der_indices_true[i][j])
                idx += 1

        # 2. Calculate true derivatives for the TRAINING points (for the scatter plot)
        X_train_pert = oti.array(self.X_train) + \
            oti.e(1, order=max_order_to_plot)
        y_train_hc = self.true_function(X_train_pert)
        true_train_derivs = {0: y_train_hc.real}
        idx = 1
        for i in range(len(der_indices_true)):
            for j in range(len(der_indices_true[i])):
                true_train_derivs[idx] = y_train_hc.get_deriv(
                    der_indices_true[i][j])
                idx += 1

        # Define which derivative orders to display and their titles
        deriv_orders_to_show = [1, 2, 4]
        plot_titles = [
            "1st Derivative Prediction ($f'(x)$)",
            "2nd Derivative Prediction ($f''(x)$)",
            "4th Derivative Prediction ($f^{(4)}(x)$)"
        ]

        fig, axs = plt.subplots(len(deriv_orders_to_show),
                                1, figsize=(12, 12), sharex=True)
        if len(deriv_orders_to_show) == 1:
            axs = [axs]
        else:
            axs = axs.flatten()

        model_colors = plt.cm.viridis(np.linspace(0, 1, len(self.results)))

        for i, deriv_idx in enumerate(deriv_orders_to_show):
            ax = axs[i]
            # Plot the true derivative line
            if deriv_idx in true_derivs:
                ax.plot(self.X_test, true_derivs[deriv_idx], 'k-',
                        lw=3, label="True $f^{{({n})}}(x)$", zorder=10)

            # Plot the derivative training data as scatter points
            if deriv_idx <= max(self.config.orders_to_test) and deriv_idx in true_train_derivs:
                ax.scatter(self.X_train, true_train_derivs[deriv_idx],
                           c='red', s=60, zorder=11,
                           edgecolors='black', linewidth=1, label="Training Data")

            # Plot predictions from each trained model
            for j, (trained_order, r) in enumerate(self.results.items()):
                if deriv_idx <= trained_order:
                    start_idx = deriv_idx * self.config.num_test_pts
                    end_idx = (deriv_idx + 1) * self.config.num_test_pts

                    y_pred_deriv = r['y_pred_full'][start_idx:end_idx]
                    y_var_deriv = r['y_var_full'][start_idx:end_idx]
                    std_dev = np.sqrt(y_var_deriv.ravel())

                    ax.plot(self.X_test, y_pred_deriv, '--', lw=2, color=model_colors[j],
                            label=f"Prediction from Order {trained_order} DEGP", zorder=10)

                    ax.fill_between(
                        self.X_test.ravel(),
                        y_pred_deriv.ravel() - 2 * std_dev,
                        y_pred_deriv.ravel() + 2 * std_dev,
                        color=model_colors[j],
                        alpha=0.15
                    )

            ax.set_title(plot_titles[i], fontsize=14)
            ax.set_ylabel("Value")
            ax.grid(True, linestyle=':', alpha=0.6)
            # ⭐ The individual legend call is now removed from the loop
            # ax.legend()

        ax.set_xlabel("$x$")
        fig.suptitle('Derivative Prediction Comparison with Uncertainty',
                     fontsize=18, fontweight='bold', y=0.99)

        # ⭐ 1. Collect unique handles and labels from all subplots
        handles, labels = axs[0].get_legend_handles_labels()

        # ⭐ 2. Create a single figure-level legend
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(
            0.5, 0.02), ncol=2, frameon=True, fancybox=True, shadow=True)

        # ⭐ 3. Adjust the layout to make room for the legend
        plt.tight_layout(rect=[0, .1, 0.85, 0.96])
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
        self.visualize_derivative_results()  # <-- Added the new plot call
        self.display_educational_content()


def true_function(X, alg=oti):
    """Complex test function combining multiple mathematical components."""
    x = X[:, 0]
    return alg.exp(-x) + alg.sin(x) + alg.cos(3 * x) + 0.2 * x + 1.0


def main():
    """Main execution block."""
    config = DEGPConfig()
    tutorial = DEGPComparisonTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
