"""
================================================================================
DEGP Tutorial: 3D Ishigami Function with Directional Derivatives
================================================================================

This tutorial demonstrates how to apply a Directional-Derivative Enhanced
Gaussian Process (DD-GP) to a 3-dimensional function. We visualize the results
of the 3D model by taking a 2D slice, a common and crucial technique for
analyzing high-dimensional surrogate models.

Key concepts covered:
-   Applying DD-GP to a 3D function (the Ishigami function).
-   Using Sobol sequence sampling for efficient coverage of the 3D space.
-   Employing **random directional derivatives** (rays) at each training point.
-   The specific pointwise hypercomplex AD workflow required by the `gddegp` model.
-   Visualizing the 3D model's predictions on a 2D slice (fixing x₃=π).
-   Plotting techniques for comparing the prediction, true function, and error.
"""

import numpy as np
import pyoti.sparse as oti
import itertools
from full_gddegp.gddegp import gddegp
from scipy.stats import qmc
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from dataclasses import dataclass
from typing import Dict, Callable
import utils

plt.rcParams.update({'font.size': 12})


@dataclass
class IshigamiConfig:
    """Configuration for the 3D Ishigami DD-GP tutorial."""
    n_bases: int = 3
    n_order: int = 2
    num_training_pts: int = 150
    domain_bounds: tuple = ((-np.pi, np.pi),) * 3
    test_grid_resolution: int = 100
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"
    n_restarts: int = 15
    swarm_size: int = 150
    random_seed: int = 1


class Ishigami3DTutorial:
    """
    Manages and executes the DD-GP tutorial on the 3D Ishigami function.
    """

    def __init__(self, config: IshigamiConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.training_data = {}
        self.gp_model = None
        self.params = None
        self.results = {}
        np.random.seed(config.random_seed)

    def _generate_training_data(self):
        """
        Generates 3D training data with random directional derivatives.
        """
        print("\n" + "="*50 + "\nGenerating Training Data with Random Rays\n" + "="*50)
        cfg = self.config

        # 1. Generate Points with Sobol sampling
        sampler = qmc.Sobol(d=cfg.n_bases, scramble=True, seed=cfg.random_seed)
        unit_samples = sampler.random(n=cfg.num_training_pts)
        X_train = qmc.scale(unit_samples, [b[0] for b in cfg.domain_bounds], [
                            b[1] for b in cfg.domain_bounds])

        # 2. Generate a random unit vector (ray) for each point
        rays_list, tag_map = [], []
        for i in range(cfg.num_training_pts):
            g = self.rng.randn(cfg.n_bases)
            ray = g / np.linalg.norm(g)
            rays_list.append(ray[:, np.newaxis])
            tag_map.append(i + 1)

        print(
            f"  Generated {len(X_train)} training points and {len(rays_list)} random directional rays.")

        # 3. Apply Pointwise Perturbation
        X_pert = oti.array(X_train)
        for i, ray in enumerate(rays_list):
            e_tag = oti.e(1, order=cfg.n_order)
            perturbation = oti.array(ray) * e_tag
            X_pert[i, :] += perturbation.T

        # 4. Evaluate, Truncate, and Extract Derivatives
        f_hc = self.true_function(X_pert, alg=oti)
        for combo in itertools.combinations(tag_map, 2):
            f_hc = f_hc.truncate(combo)

        y_train_list = [f_hc.real.reshape(-1, 1)]
        der_indices_to_extract = [[[1, i+1]] for i in range(cfg.n_order)]
        for idx in der_indices_to_extract:
            y_train_list.append(f_hc.get_deriv(idx).reshape(-1, 1))

        self.training_data = {
            'X_train': X_train,
            'y_train_list': y_train_list,
            'rays_list': rays_list
        }
        print(
            f"  Extracted function values and {cfg.n_order} directional derivatives per point.")

    def _train_model(self):
        """Initializes and trains the GD-DEGP model."""
        print("\n" + "="*50 + "\nTraining GD-DEGP Model\n" + "="*50)
        cfg, data = self.config, self.training_data

        rays_array = np.hstack(data['rays_list'])
        self.gp_model = gddegp(
            data['X_train'], data['y_train_list'], n_order=cfg.n_order, rays_array=rays_array,
            normalize=cfg.normalize_data, kernel=cfg.kernel, kernel_type=cfg.kernel_type
        )
        print("  Model initialization: SUCCESS")

        print("  Optimizing hyperparameters...")
        self.params = self.gp_model.optimize_hyperparameters(
            n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size
        )
        print("  Hyperparameter optimization: SUCCESS")

    def _evaluate_on_slice(self):
        """Creates a 2D test slice (x₃=π) and evaluates the model's performance."""
        print("\n" + "="*50 + "\nModel Prediction and Evaluation on 2D Slice\n" + "="*50)
        cfg = self.config

        # Create a 2D slice grid at x₃ = π
        gx = np.linspace(
            cfg.domain_bounds[0][0], cfg.domain_bounds[0][1], cfg.test_grid_resolution)
        gy = np.linspace(
            cfg.domain_bounds[1][0], cfg.domain_bounds[1][1], cfg.test_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(gx, gy)
        X_test = np.column_stack(
            [X1_grid.ravel(), X2_grid.ravel(), np.pi * np.ones_like(X1_grid.ravel())])

        # Provide a dummy ray for prediction, as we only need the function value
        dummy_ray = np.array([1.0, 0.0, 0.0])[:, np.newaxis]
        rays_pred = np.hstack([dummy_ray] * X_test.shape[0])

        y_pred = self.gp_model.predict(
            X_test, rays_pred, self.params, calc_cov=False, return_deriv=False)
        y_true = self.true_function(X_test, alg=np)

        self.results = {
            'X_test': X_test, 'y_pred': y_pred, 'y_true': y_true,
            'X1_grid': X1_grid, 'X2_grid': X2_grid,
            'nrmse': utils.nrmse(y_true, y_pred)
        }
        print(
            f"  Evaluation on slice (x₃=π) complete. Final NRMSE: {self.results['nrmse']:.6f}")

    def visualize_slice_results(self):
        """Generates the 3-panel contour plot for the 2D slice."""
        print("\n" + "="*50 + "\nGenerating Visualizations\n" + "="*50)
        res, cfg = self.results, self.config
        X_train = self.training_data['X_train']
        X1, X2 = res['X1_grid'], res['X2_grid']

        fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

        # Panel 1: GP Prediction
        gp_map = res['y_pred'].reshape(X1.shape)
        cf1 = axs[0].contourf(X1, X2, gp_map, levels=30, cmap='viridis')
        fig.colorbar(cf1, ax=axs[0])
        axs[0].set_title("GP Prediction (Slice at x₃=π)")

        # Show only training points near the slice
        tol = 0.5
        mask = np.abs(X_train[:, 2] - np.pi) < tol
        axs[0].scatter(X_train[mask, 0], X_train[mask, 1], c='red',
                       s=40, edgecolors='k', label=f'Train pts (|x₃-π|<{tol})')
        axs[0].legend()

        # Panel 2: True Function
        true_map = res['y_true'].reshape(X1.shape)
        cf2 = axs[1].contourf(X1, X2, true_map, levels=30, cmap='viridis')
        fig.colorbar(cf2, ax=axs[1])
        axs[1].set_title("True Function (Slice at x₃=π)")

        # Panel 3: Absolute Error
        abs_error = np.abs(gp_map - true_map)
        abs_error_clipped = np.clip(abs_error, 1e-8, None)
        # For more control, define the exact level boundaries
        log_levels = np.logspace(
            np.log10(abs_error_clipped.min()),
            np.log10(abs_error_clipped.max()),
            num=50  # The number of levels you want
        )

        cf3 = axs[2].contourf(X1, X2, abs_error_clipped,
                              levels=log_levels,
                              norm=LogNorm(), cmap="magma_r")
        fig.colorbar(cf3, ax=axs[2], format="%.1e")
        axs[2].set_title("Absolute Error (log scale)")

        for ax in axs:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_aspect("equal")

        plt.show()

    def run(self):
        """Executes the complete tutorial workflow."""
        print("DEGP Tutorial: 3D Ishigami Function with Directional Derivatives")
        print("=" * 75)

        self.rng = np.random.RandomState(self.config.random_seed)
        self._generate_training_data()
        self._train_model()
        self._evaluate_on_slice()
        self.visualize_slice_results()

        print("\nTutorial Complete.")

# --- Helper Function ---


def true_function(X, alg=np):
    """3D Ishigami function."""
    a_ish, b_ish = 7.0, 0.1
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return alg.sin(x1) + a_ish * alg.sin(x2)**2 + b_ish * x3**4 * alg.sin(x1)


def main():
    """Main execution block."""
    config = IshigamiConfig(n_order=2)
    tutorial = Ishigami3DTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
