"""
================================================================================
DEGP Tutorial: Grouped Submodel Comparison vs. Standard GP
================================================================================

This tutorial demonstrates a weighted, derivative-enhanced Gaussian Process (GP)
where training points are grouped into distinct submodels. It directly
compares the performance of this grouped Derivative-Enhanced GP (DEGP) against a
standard GP that uses the exact same function data but no derivative information.

The script showcases how to incorporate measurement noise and uses a detailed
4-panel plot to visualize the performance difference, highlighting the value
added by derivative observations in a multi-submodel context.

Key Concepts Demonstrated:
-   Grouping training points into multiple submodels.
-   Correct data structuring for the wdegp library with multiple submodels.
-   Direct performance comparison between a grouped DEGP and a standard GP.
-   Injecting proportional noise into function and derivative data.
-   Advanced 4-panel visualization for model comparison.
"""

import numpy as np
import pyoti.sparse as oti
from wdegp.wdegp import wdegp
import utils
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Callable


@dataclass
class GroupedSubmodelConfig:
    """Configuration for the grouped submodel DEGP tutorial."""
    lb_x: float = 0.0
    ub_x: float = 10.0
    num_training_pts: int = 7
    num_test_pts: int = 500
    n_order: int = 1
    n_bases: int = 1
    # Define how points are grouped into submodels
    submodel_groups: List[List[int]] = field(
        default_factory=lambda: [[0, 1, 2], [3, 4, 5, 6]])
    function_noise_ratio: float = 0.00
    derivative_noise_ratio: float = 1.00
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"
    n_restarts: int = 15
    swarm_size: int = 200
    random_seed: int = 1


class GroupedSubmodelGPTutorial:
    """
    Manages a DEGP experiment comparing a grouped submodel DEGP
    against a standard GP.
    """

    def __init__(self, config: GroupedSubmodelConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.rng = np.random.RandomState(config.random_seed)
        self.training_data: Dict = {}
        self.degp_model = None
        self.std_gp_model = None
        self.results: Dict = {}

    def _generate_training_data(self):
        """
        Generates training data with multiple submodels and injects noise,
        following the correct structure required by wdegp.
        """
        print("\n" + "="*50 + "\nGenerating Training Data with Noise\n" + "="*50)
        cfg = self.config

        # 1. Generate Training Points
        X_candidates = np.linspace(cfg.lb_x, cfg.ub_x, 1000).reshape(-1, 1)
        training_indices = self.rng.choice(
            np.arange(X_candidates.shape[0]), size=cfg.num_training_pts, replace=False)
        X_train = np.sort(X_candidates[training_indices], axis=0)

        # 2. Define Submodel Structure
        submodel_indices = cfg.submodel_groups
        base_der_indices = utils.gen_OTI_indices(cfg.n_bases, cfg.n_order)
        der_indices = [base_der_indices for _ in submodel_indices]

        # 3. Get clean function values for ALL points and add noise
        y_func_clean = self.true_function(X_train, alg=np)
        func_noise_std = np.abs(y_func_clean.flatten()) * \
            cfg.function_noise_ratio
        y_func_noisy = y_func_clean + \
            self.rng.normal(loc=0.0, scale=func_noise_std)

        # 4. Construct the y_train_data list of lists
        y_train_data_for_all_submodels = []
        for group_idx in submodel_indices:
            # Each submodel's data list starts with ALL noisy function values
            current_submodel_data = [y_func_noisy]

            # Compute derivatives ONLY for the points in the current group
            X_group_pert = oti.array(
                X_train[group_idx]) + oti.e(1, order=cfg.n_order)
            y_group_hc = self.true_function(X_group_pert)

            # Append noisy derivatives for the current group
            flat_der_indices = [
                item for sublist in base_der_indices for item in sublist]
            for idx_spec in flat_der_indices:
                deriv_clean = y_group_hc.get_deriv(idx_spec)
                deriv_noise_std = np.abs(
                    deriv_clean.flatten()) * cfg.derivative_noise_ratio
                deriv_noisy = deriv_clean + \
                    self.rng.normal(
                        loc=0.0, scale=deriv_noise_std).reshape(-1, 1)
                current_submodel_data.append(deriv_noisy)

            y_train_data_for_all_submodels.append(current_submodel_data)

        # 5. Construct the single noise vector for the model
        # This vector must correspond to the full potential observation matrix.
        noise_std_vector = [func_noise_std]
        full_y_train_hc = self.true_function(
            oti.array(X_train) + oti.e(1, order=cfg.n_order))
        for idx_spec in flat_der_indices:
            deriv_clean_full = full_y_train_hc.get_deriv(idx_spec)
            deriv_noise_std_full = np.abs(
                deriv_clean_full.flatten()) * cfg.derivative_noise_ratio
            noise_std_vector.append(deriv_noise_std_full)

        self.training_data = {
            'X_train': X_train,
            'y_train_data': y_train_data_for_all_submodels,
            'submodel_indices': submodel_indices,
            'der_indices': der_indices,
            'noise_std_vector': np.concatenate(noise_std_vector)
        }
        print(f"  Data generated for {len(submodel_indices)} submodels.")

    def _train_models(self):
        """Initializes and trains both the DEGP and a standard GP."""
        print("\n" + "="*50 + "\nTraining DEGP and Standard GP Models\n" + "="*50)
        cfg = self.config
        data = self.training_data

        # --- Train the DEGP Model ---
        print("  Training DEGP (with derivatives)...")
        self.degp_model = wdegp(
            data['X_train'], data['y_train_data'], cfg.n_order, cfg.n_bases,
            data['submodel_indices'], data['der_indices'],
            normalize=cfg.normalize_data, sigma_data=data['noise_std_vector'],
            kernel=cfg.kernel, kernel_type=cfg.kernel_type
        )
        self.degp_params = self.degp_model.optimize_hyperparameters(
            n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size
        )

        # --- Train the Standard GP Model (No Derivatives) ---
        print("  Training Standard GP (function data only)...")
        std_gp_y_data = [[data['y_train_data'][0][0]]
                         for _ in data['submodel_indices']]
        std_gp_der_indices = [[] for _ in data['submodel_indices']]

        self.std_gp_model = wdegp(
            data['X_train'], std_gp_y_data, 0, cfg.n_bases,
            data['submodel_indices'], std_gp_der_indices, normalize=cfg.normalize_data,
            sigma_data=data['noise_std_vector'][:cfg.num_training_pts],
            kernel=cfg.kernel, kernel_type=cfg.kernel_type
        )
        self.std_gp_params = self.std_gp_model.optimize_hyperparameters(
            n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size
        )

    def _evaluate_models(self):
        """Makes predictions from both models and computes comparative metrics."""
        print("\n" + "="*50 + "\nModel Prediction and Evaluation\n" + "="*50)
        X_test = np.linspace(self.config.lb_x, self.config.ub_x,
                             self.config.num_test_pts).reshape(-1, 1)
        y_true = self.true_function(X_test, alg=np)

        y_pred_degp, y_cov_degp, _, _ = self.degp_model.predict(
            X_test, self.degp_params, calc_cov=True, return_submodels=True)
        y_pred_std, y_cov_std, _, _ = self.std_gp_model.predict(
            X_test, self.std_gp_params, calc_cov=True, return_submodels=True)

        nrmse_degp = utils.nrmse(y_true, y_pred_degp)
        nrmse_std = utils.nrmse(y_true, y_pred_std)

        self.results = {
            'X_test': X_test, 'y_true': y_true,
            'degp_nrmse': nrmse_degp, 'std_nrmse': nrmse_std,
            'improvement_percent': (nrmse_std - nrmse_degp) / nrmse_std * 100,
            'degp_pred': y_pred_degp, 'std_pred': y_pred_std,
            'degp_cov': y_cov_degp, 'std_cov': y_cov_std
        }
        print(
            f"  DEGP NRMSE: {nrmse_degp:.6f} | Standard GP NRMSE: {nrmse_std:.6f}")
        print(
            f"  Improvement with Derivatives: {self.results['improvement_percent']:.2f}%")

    def visualize_comparison(self):
        """Generates the 4-panel plot comparing DEGP and Standard GP."""
        print("\n" + "="*50 + "\nGenerating Comparison Visualization\n" + "="*50)
        cfg, res = self.config, self.results
        X_train, X_test = self.training_data['X_train'], res['X_test']
        y_train_func = self.training_data['y_train_data'][0][0]
        y_true = res['y_true'].flatten()

        plt.figure(figsize=(16, 8))

        ax1 = plt.subplot(2, 2, 1)
        degp_pred, degp_std = res['degp_pred'].flatten(), np.sqrt(
            res['degp_cov'].flatten())
        ax1.plot(X_test, y_true, 'k-', label='True function', linewidth=2)
        ax1.plot(X_test, degp_pred, 'b-', label='DEGP prediction', linewidth=2)
        ax1.fill_between(X_test.flatten(), degp_pred - 2*degp_std, degp_pred +
                         2*degp_std, alpha=0.3, color='blue', label='±2σ uncertainty')
        ax1.scatter(X_train, y_train_func, c='red', s=80, zorder=5,
                    label='Training data', edgecolors='darkred')
        ax1.set_title(
            f'Grouped DEGP (with {cfg.derivative_noise_ratio*100:.0f}% derivative noise)\nNRMSE: {res["degp_nrmse"]:.4f}')
        ax1.set(xlabel='x', ylabel='y')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        ax2 = plt.subplot(2, 2, 2)
        std_pred, std_std = res['std_pred'].flatten(), np.sqrt(
            res['std_cov'].flatten())
        ax2.plot(X_test, y_true, 'k-', label='True function', linewidth=2)
        ax2.plot(X_test, std_pred, 'g-',
                 label='Standard GP prediction', linewidth=2)
        ax2.fill_between(X_test.flatten(), std_pred - 2*std_std, std_pred +
                         2*std_std, alpha=0.3, color='green', label='±2σ uncertainty')
        ax2.scatter(X_train, y_train_func, c='red', s=80, zorder=5,
                    label='Training data', edgecolors='darkred')
        ax2.set_title(
            f'Standard GP (function only)\nNRMSE: {res["std_nrmse"]:.4f}')
        ax2.set(xlabel='x', ylabel='y')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(X_test, np.abs(y_true - degp_pred),
                 'b-', label='DEGP error', linewidth=2)
        ax3.plot(X_test, np.abs(y_true - std_pred), 'g-',
                 label='Standard GP error', linewidth=2)
        ax3.set_title('Prediction Error Comparison')
        ax3.set(xlabel='x', ylabel='Absolute Error', yscale='log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        ax4 = plt.subplot(2, 2, 4)
        ax4.plot(X_test, degp_std, 'b-',
                 label='DEGP uncertainty (σ)', linewidth=2)
        ax4.plot(X_test, std_std, 'g-',
                 label='Standard GP uncertainty (σ)', linewidth=2)
        ax4.set_title('Uncertainty Quantification Comparison')
        ax4.set(xlabel='x', ylabel='Prediction Uncertainty')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def run(self):
        """Executes the complete tutorial workflow."""
        print("DEGP Tutorial: Grouped Submodel vs. Standard GP")
        print("=" * 65)

        self._generate_training_data()
        self._train_models()
        self._evaluate_models()
        self.visualize_comparison()

        print(
            f"\nTutorial Complete. Grouped DEGP showed a {self.results['improvement_percent']:.2f}% improvement over the standard GP.")


def true_function(X, alg=oti):
    """Test function f(x) = x * sin(x)."""
    x = X[:, 0]
    return x * alg.sin(x)


def main():
    """Main execution block."""
    config = GroupedSubmodelConfig(
        num_training_pts=7, derivative_noise_ratio=.01)
    tutorial = GroupedSubmodelGPTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
