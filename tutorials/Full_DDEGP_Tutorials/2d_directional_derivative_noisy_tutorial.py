"""
================================================================================
DEGP Tutorial: Global Directional GP with Noisy Observations
================================================================================

This tutorial demonstrates how to apply a Directional-Derivative Enhanced
Gaussian Process (D-DEGP) when the training observations, including the
derivatives, are corrupted by noise.

This model uses a common set of directional derivatives ("rays") applied
uniformly to all training points. We will inject a specified level of noise
into both the function and derivative values and then inform the GP about this
noise, allowing it to learn robustly from imperfect data.

Key concepts covered:
-   Using a global basis of directional derivatives for all training points.
-   Injecting synthetic noise into both function and derivative observations.
-   Passing observation noise levels to the GP model via the `sigma_data` parameter.
-   Training the `ddegp` model on noisy, high-dimensional data.
-   Visualizing the resulting 2D function approximation.
"""

import numpy as np
import pyoti.sparse as oti
import itertools
from full_ddegp.ddegp import ddegp
import utils
import plotting_helper
from dataclasses import dataclass, field
from typing import Dict, Callable


@dataclass
class NoisyDirectionalConfig:
    """Configuration for the Noisy Global Directional DEGP tutorial."""
    n_order: int = 3
    n_bases: int = 2
    num_pts_per_axis: int = 5
    domain_bounds: tuple = ((-1, 1), (-1, 1))
    test_grid_resolution: int = 50

    # Noise configuration
    function_noise_std: float = 0.10  # Standard deviation of noise on function values
    derivative_noise_std: float = 0.10  # Standard deviation of noise on derivatives

    # Define the global set of directional derivatives (rays)
    rays: np.ndarray = field(default_factory=lambda: np.array([
        [np.cos(2*np.pi/1), np.cos(2*np.pi/2), np.cos(2*np.pi/3)],
        [np.sin(2*np.pi/1), np.sin(2*np.pi/2), np.sin(2*np.pi/3)]
    ]))

    # Model & Optimizer
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"
    n_restarts: int = 15
    swarm_size: int = 100
    random_seed: int = 1


class NoisyDirectionalDEGPTutorial:
    """
    Manages and executes a tutorial for a D-DEGP with noisy observations.
    """

    def __init__(self, config: NoisyDirectionalConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.rng = np.random.RandomState(config.random_seed)
        self.training_data: Dict = {}
        self.gp_model = None
        self.params = None
        self.results: Dict = {}

    def _generate_training_data(self):
        """
        Generates training data, applies global directional perturbations,
        and injects a specified level of noise.
        """
        print("\n" + "="*50 + "\nGenerating Noisy Training Data\n" + "="*50)
        cfg = self.config

        # 1. Create Training Grid
        x_vals = np.linspace(
            cfg.domain_bounds[0][0], cfg.domain_bounds[0][1], cfg.num_pts_per_axis)
        y_vals = np.linspace(
            cfg.domain_bounds[1][0], cfg.domain_bounds[1][1], cfg.num_pts_per_axis)
        X_train = np.array(list(itertools.product(x_vals, y_vals)))

        # 2. Apply Global Directional Perturbations
        e_bases = [oti.e(i + 1, order=cfg.n_order)
                   for i in range(cfg.rays.shape[1])]
        perturbations = np.dot(cfg.rays, e_bases)

        X_pert = oti.array(X_train)
        for j in range(cfg.n_bases):
            X_pert[:, j] += perturbations[j]

        # 3. Evaluate and Extract Clean Data
        f_hc = self.true_function(X_pert, alg=oti)
        for combo in itertools.combinations(range(1, cfg.rays.shape[1] + 1), 2):
            f_hc = f_hc.truncate(combo)

        y_train_list_noisy = []
        noise_std_vector = []

        # 4. Add noise to function values
        y_func_clean = f_hc.real
        y_func_noisy = y_func_clean + \
            self.rng.normal(loc=0.0, scale=cfg.function_noise_std,
                            size=y_func_clean.shape)
        y_train_list_noisy.append(y_func_noisy)
        noise_std_vector.append(
            np.full(y_func_clean.shape[0], cfg.function_noise_std))

        # 5. Add noise to derivative values
        der_indices = [[
            [[1, 1]], [[1, 2]], [[1, 3]],  # Derivatives along ray 1
            [[2, 1]], [[2, 2]], [[2, 3]],  # Derivatives along ray 2
            [[3, 1]], [[3, 2]], [[3, 3]],  # Derivatives along ray 3
        ]]

        for group in der_indices:
            for sub_group in group:
                deriv_clean = f_hc.get_deriv(sub_group)
                deriv_noisy = deriv_clean + \
                    self.rng.normal(
                        loc=0.0, scale=cfg.derivative_noise_std, size=deriv_clean.shape)
                y_train_list_noisy.append(deriv_noisy)
                noise_std_vector.append(
                    np.full(deriv_clean.shape[0], cfg.derivative_noise_std))

        self.training_data = {
            'X_train': X_train,
            'y_train_list': y_train_list_noisy,
            'der_indices': der_indices,
            'noise_std_vector': np.concatenate(noise_std_vector)
        }
        print(
            f"  Injected noise with std={cfg.function_noise_std} (function) and std={cfg.derivative_noise_std} (derivatives).")

    def _train_model(self):
        """Initializes and trains the ddegp model with noisy data."""
        print("\n" + "="*50 + "\nTraining D-DEGP Model\n" + "="*50)
        cfg, data = self.config, self.training_data

        self.gp_model = ddegp(
            data['X_train'], data['y_train_list'], n_order=cfg.n_order,
            der_indices=data['der_indices'], rays=cfg.rays, normalize=cfg.normalize_data,
            # Inform the model of the noise
            sigma_data=data['noise_std_vector'],
            kernel=cfg.kernel, kernel_type=cfg.kernel_type
        )
        print("  Model initialization: SUCCESS")

        print("  Optimizing hyperparameters...")
        self.params = self.gp_model.optimize_hyperparameters(
            n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size
        )
        print("  Hyperparameter optimization: SUCCESS")

    def _evaluate_model(self):
        """Creates a test grid and evaluates the model's performance."""
        print("\n" + "="*50 + "\nModel Prediction and Evaluation\n" + "="*50)
        cfg = self.config

        x_lin = np.linspace(
            cfg.domain_bounds[0][0], cfg.domain_bounds[0][1], cfg.test_grid_resolution)
        y_lin = np.linspace(
            cfg.domain_bounds[1][0], cfg.domain_bounds[1][1], cfg.test_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
        X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

        y_pred = self.gp_model.predict(X_test, self.params, calc_cov=False)
        y_true = self.true_function(X_test, alg=np)

        self.results = {
            'X_test': X_test, 'y_pred': y_pred, 'y_true': y_true,
            'X1_grid': X1_grid, 'X2_grid': X2_grid,
            'nrmse': utils.nrmse(y_true, y_pred)
        }
        print(
            f"  Evaluation complete. Final NRMSE: {self.results['nrmse']:.6f}")

    def visualize_results(self):
        """Generates 2D surface plots comparing the prediction with the true function."""
        print("\n" + "="*50 + "\nGenerating Visualizations\n" + "="*50)
        plotting_helper.make_plots(
            self.training_data['X_train'], self.training_data['y_train_list'],
            self.results['X_test'], self.results['y_pred'], self.true_function,
            X1_grid=self.results['X1_grid'], X2_grid=self.results['X2_grid'],
            n_order=self.config.n_order, plot_derivative_surrogates=False,
            der_indices=self.training_data['der_indices']
        )
        print("  Visualization created.")

    def run(self):
        """Executes the complete tutorial workflow."""
        print("DEGP Tutorial: Global Directional GP with Noisy Observations")
        print("=" * 75)

        self._generate_training_data()
        self._train_model()
        self._evaluate_model()
        self.visualize_results()

        print(f"\nTutorial Complete. Final NRMSE: {self.results['nrmse']:.6f}")

# --- Helper Function ---


def true_function(X, alg=np):
    """A simple 2D quadratic function."""
    x1, x2 = X[:, 0], X[:, 1]
    return x1**2 + x2**2


def main():
    """Main execution block."""
    config = NoisyDirectionalConfig()
    tutorial = NoisyDirectionalDEGPTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
