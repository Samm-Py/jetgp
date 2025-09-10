"""
================================================================================
DEGP Tutorial: Robustness of Derivative Information Under Noisy Conditions
================================================================================

This tutorial demonstrates the robustness of derivative-enhanced Gaussian Processes
(DEGPs) when derivative observations are corrupted by measurement noise. We compare
DEGP performance against standard GPs to quantify the value of derivative information
even under challenging measurement conditions.

Key concepts demonstrated:
- Impact of derivative measurement noise on GP performance
- Fair comparison methodology: DEGP vs Standard GP with identical function data
- Noise modeling through sigma_data parameter configuration
- Performance quantification under realistic measurement scenarios
- Visualization of prediction uncertainty under noisy derivative conditions
================================================================================
"""

import numpy as np
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Callable
from dataclasses import dataclass, field


@dataclass
class NoiseStudyConfig:
    """Configuration for the DEGP robustness study."""
    # Domain and discretization
    lb_x: float = 0.0
    ub_x: float = 10.0
    num_training_pts: int = 6
    num_test_pts: int = 100

    # DEGP configuration
    n_order: int = 1  # Include first derivatives
    n_bases: int = 1  # 1D function
    kernel: str = "SE"
    kernel_type: str = "anisotropic"

    # Noise configuration
    function_noise_ratio: float = 0.00  # 0% noise = Perfect function observations
    derivative_noise_ratio: float = 0.25  # 25% noise on derivatives

    # Optimization settings
    n_restarts: int = 15
    swarm_size: int = 150

    # Reproducibility
    random_seed: int = 1


class NoiseRobustnessStudy:
    """
    Manages the experiment comparing DEGP and standard GP performance
    under noisy derivative conditions.
    """

    def __init__(self, config: NoiseStudyConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.rng = np.random.RandomState(config.random_seed)
        self.training_data = {}
        self.degp_model = None
        self.std_gp_model = None
        self.results = {}

    def _generate_training_data(self):
        """
        Generates training data with controlled, proportional noise injection.
        """
        print("Generating training data with controlled noise injection...")
        cfg = self.config

        # Select training points
        X_candidate = np.linspace(cfg.lb_x, cfg.ub_x, 1000).reshape(-1, 1)
        training_indices = self.rng.choice(
            np.arange(X_candidate.shape[0]), size=cfg.num_training_pts, replace=False)
        X_train = np.sort(X_candidate[training_indices], axis=0)
        X_train[0] = 1.0  # Ensure coverage near boundary

        # Use OTI for automatic differentiation
        X_train_pert = oti.array(X_train)
        X_train_pert[:, 0] += oti.e(1, order=cfg.n_order)
        y_hc = self.true_function(X_train_pert)
        der_indices = utils.gen_OTI_indices(cfg.n_bases, cfg.n_order)

        # Generate clean function values and add noise
        y_func_clean = y_hc.real
        func_noise_std = np.abs(y_func_clean.flatten()) * \
            cfg.function_noise_ratio
        y_func_noisy = y_func_clean + \
            self.rng.normal(loc=0.0, scale=func_noise_std).reshape(-1, 1)

        y_train_list = [y_func_noisy]
        full_noise_std_list = [func_noise_std]

        # Generate clean derivative values and add noise
        for i in range(len(der_indices)):
            for j in range(len(der_indices[i])):
                deriv_clean = y_hc.get_deriv(der_indices[i][j]).reshape(-1, 1)
                deriv_noise_std = np.abs(
                    deriv_clean.flatten()) * cfg.derivative_noise_ratio
                deriv_noisy = deriv_clean + \
                    self.rng.normal(
                        loc=0.0, scale=deriv_noise_std).reshape(-1, 1)
                y_train_list.append(deriv_noisy)
                full_noise_std_list.append(deriv_noise_std)

        self.training_data = {
            'X_train': X_train,
            'y_train_list': y_train_list,
            'der_indices': der_indices,
            'noise_std_vector': np.concatenate(full_noise_std_list)
        }

    def _train_models(self):
        """
        Trains both the DEGP and the standard GP for a fair comparison.
        """
        cfg = self.config
        data = self.training_data

        # --- Train DEGP with noisy derivatives ---
        print("Training DEGP with noisy derivatives...")
        self.degp_model = degp(
            data['X_train'], data['y_train_list'], cfg.n_order, cfg.n_bases,
            data['der_indices'], normalize=True, sigma_data=data['noise_std_vector'],
            kernel=cfg.kernel, kernel_type=cfg.kernel_type
        )
        self.degp_params = self.degp_model.optimize_hyperparameters(
            n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size
        )

        # --- Train Standard GP (function values only) ---
        print("Training Standard GP for comparison...")
        self.std_gp_model = degp(
            data['X_train'], [data['y_train_list'][0]], 0, cfg.n_bases, [],
            normalize=True, sigma_data=data['noise_std_vector'][:cfg.num_training_pts],
            kernel=cfg.kernel, kernel_type=cfg.kernel_type
        )
        self.std_gp_params = self.std_gp_model.optimize_hyperparameters(
            n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size
        )

    def _evaluate_models(self):
        """Evaluates both models and computes comparative performance metrics."""
        print("Evaluating model performance...")
        X_test = np.linspace(self.config.lb_x, self.config.ub_x,
                             self.config.num_test_pts).reshape(-1, 1)
        y_true = self.true_function(X_test, alg=np)

        y_pred_degp, y_var_degp = self.degp_model.predict(
            X_test, self.degp_params, calc_cov=True)
        y_pred_std, y_var_std = self.std_gp_model.predict(
            X_test, self.std_gp_params, calc_cov=True)

        degp_nrmse = utils.nrmse(y_true, y_pred_degp)
        std_nrmse = utils.nrmse(y_true, y_pred_std)

        self.results = {
            'X_test': X_test, 'y_true': y_true,
            'degp_nrmse': degp_nrmse, 'std_nrmse': std_nrmse,
            'improvement_percent': (std_nrmse - degp_nrmse) / std_nrmse * 100,
            'degp_predictions': y_pred_degp, 'std_predictions': y_pred_std,
            'degp_variance': y_var_degp, 'std_variance': y_var_std,
            'degp_mean_uncertainty': np.mean(np.sqrt(y_var_degp.flatten())),
            'std_mean_uncertainty': np.mean(np.sqrt(y_var_std.flatten()))
        }

    def display_summary(self):
        """Prints a comprehensive summary of the experimental results."""
        cfg = self.config
        res = self.results
        X_train = self.training_data['X_train']

        print("\n" + "="*70 + "\nEXPERIMENTAL RESULTS\n" + "="*70)
        print(f"Performance Comparison:")
        print(f"  DEGP NRMSE:        {res['degp_nrmse']:.6f}")
        print(f"  Standard GP NRMSE: {res['std_nrmse']:.6f}")
        print(f"  DEGP Improvement:  {res['improvement_percent']:.1f}%")

        print(f"\nUncertainty Analysis:")
        print(f"  DEGP Mean σ:        {res['degp_mean_uncertainty']:.4f}")
        print(f"  Standard GP Mean σ: {res['std_mean_uncertainty']:.4f}")

        print(f"\nNoise Robustness Assessment:")
        print(f"  Function noise level:   {cfg.function_noise_ratio*100:.1f}%")
        print(
            f"  Derivative noise level: {cfg.derivative_noise_ratio*100:.1f}%")
        print(
            f"  DEGP still outperforms: {'YES' if res['improvement_percent'] > 0 else 'NO'}")

    def visualize_results(self):
        """Generates a comprehensive 4-panel comparison visualization."""
        print("\nGenerating comparison visualization...")
        res = self.results
        cfg = self.config
        X_train = self.training_data['X_train']
        X_test = res['X_test']
        y_train_func = self.training_data['y_train_list'][0]
        y_true = res['y_true'].flatten()

        plt.figure(figsize=(16, 8))

        # Subplot 1: DEGP
        ax1 = plt.subplot(2, 2, 1)
        degp_pred, degp_std = res['degp_predictions'].flatten(), np.sqrt(
            res['degp_variance'].flatten())
        ax1.plot(X_test, y_true, 'k-', label='True function', linewidth=2)
        ax1.plot(X_test, degp_pred, 'b-', label='DEGP prediction', linewidth=2)
        ax1.fill_between(X_test.flatten(), degp_pred - 2*degp_std, degp_pred +
                         2*degp_std, alpha=0.3, color='blue', label='±2σ uncertainty')
        ax1.scatter(X_train, y_train_func, c='red', s=80, zorder=5,
                    label='Training data', edgecolors='darkred')
        ax1.set_title(
            f'DEGP (with {cfg.derivative_noise_ratio*100:.0f}% derivative noise)\nNRMSE: {res["degp_nrmse"]:.4f}')
        ax1.set(xlabel='x', ylabel='y')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Standard GP
        ax2 = plt.subplot(2, 2, 2)
        std_pred, std_std = res['std_predictions'].flatten(), np.sqrt(
            res['std_variance'].flatten())
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

        # Subplot 3: Prediction Error
        ax3 = plt.subplot(2, 2, 3)
        ax3.plot(X_test, np.abs(y_true - degp_pred),
                 'b-', label='DEGP error', linewidth=2)
        ax3.plot(X_test, np.abs(y_true - std_pred), 'g-',
                 label='Standard GP error', linewidth=2)
        ax3.set_title('Prediction Error Comparison')
        ax3.set(xlabel='x', ylabel='Absolute Error', yscale='log')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Subplot 4: Uncertainty
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
        """Executes the complete experimental workflow."""
        cfg = self.config
        print("DEGP Tutorial: Derivative Information Robustness Analysis")
        print("="*65)
        print(f"Scientific Question: How much derivative noise can a DEGP tolerate\n"
              f"while still outperforming a standard GP?")
        print(f"\nConfiguration:")
        print(
            f"  Function noise: {cfg.function_noise_ratio*100:.1f}% | Derivative noise: {cfg.derivative_noise_ratio*100:.1f}%")

        self._generate_training_data()
        self._train_models()
        self._evaluate_models()
        self.display_summary()
        self.visualize_results()

        print("\n" + "="*70 + "\nTUTORIAL CONCLUSIONS\n" + "="*70)
        if self.results['improvement_percent'] > 0:
            print(f"✓ DEGP demonstrates robustness to derivative noise.")
            print(
                f"✓ Achieved a {self.results['improvement_percent']:.1f}% performance improvement despite {cfg.derivative_noise_ratio*100:.0f}% derivative noise.")
            print(f"✓ Derivative information remains valuable even when imperfect.")
        else:
            print(
                f"✗ High derivative noise ({cfg.derivative_noise_ratio*100:.0f}%) overwhelmed the information content.")
            print(f"✗ Standard GP performed better in this high-noise regime.")
        print(f"\nThis tutorial demonstrates the trade-off between the value of derivative\n"
              f"information and the impact of measurement noise in practical DEGP applications.")


def true_function(X, alg=oti):
    """Test function f(x) = x * sin(x) with challenging derivative structure."""
    x = X[:, 0]
    return x * alg.sin(x)


def main():
    """Main execution block."""
    config = NoiseStudyConfig(
        derivative_noise_ratio=0.20  # Set the desired noise level here
    )
    study = NoiseRobustnessStudy(config, true_function)
    study.run()


if __name__ == "__main__":
    main()
