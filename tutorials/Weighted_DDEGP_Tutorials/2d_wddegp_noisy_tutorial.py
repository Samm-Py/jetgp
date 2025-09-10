"""
================================================================================
DEGP Tutorial: Noisy Heterogeneous Submodels with Directional Derivatives
================================================================================

This tutorial demonstrates an advanced Weighted Directional-Derivative GP (WDD-GP)
that combines several powerful features:
-   **Heterogeneous Submodels**: The input space is partitioned into subregions,
    and each region is modeled by a submodel with its own unique set of training
    points and its own basis of directional derivatives (rays).
-   **Noisy Observations**: The model is designed to handle noise in the function
    and derivative data, which is crucial for real-world applications.

This is a powerful technique for modeling complex functions where behavior varies
across the domain and where measurements are imperfect.

Key concepts covered:
-   Data reordering for complex, non-sequential submodel definitions.
-   Per-submodel automatic differentiation with unique directional rays.
-   Correctly structuring training data and noise vectors for a `wddegp` model.
-   Visualizing the combined prediction from all noisy, heterogeneous submodels.
"""

import numpy as np
import pyoti.sparse as oti
import itertools
from wddegp.wddegp import wddegp
import utils
from dataclasses import dataclass, field
from typing import List, Dict, Callable
from matplotlib import pyplot as plt


@dataclass
class NoisyHeteroConfig:
    """Configuration for the Noisy Heterogeneous Submodel DEGP tutorial."""
    n_order: int = 2
    n_bases: int = 2
    num_pts_per_axis: int = 5
    domain_bounds: tuple = ((-.5, .5), (-.5, .5))
    test_grid_resolution: int = 25

    # Define the initial, non-sequential grouping of training points
    submodel_groups_initial: List[List[int]] = field(default_factory=lambda: [
        [1, 2, 3], [5, 10, 15], [9, 14, 19], [
            21, 22, 23], [0], [4], [20], [24],
        [6, 7, 8, 11, 12, 13, 16, 17, 18]
    ])

    # Define the ray angles (thetas) for EACH submodel group
    submodel_ray_thetas: List[List[float]] = field(default_factory=lambda: [
        [-np.pi/2, 0, np.pi/2], [-np.pi, -np.pi/2, 0], [-np.pi, np.pi/2, 0],
        [-np.pi/2, -np.pi, np.pi/2], [-np.pi/2,
                                      0, -np.pi/4], [np.pi/2, 0, np.pi/4],
        [np.pi/2, 0, np.pi/4], [-np.pi/2, -np.pi, -np.pi/4 - np.pi/2],
        [np.pi/2, -np.pi, np.pi/4 + np.pi/2]
    ])

    # Define the derivative indices to extract for EACH submodel group
    submodel_der_indices: List[List] = field(default_factory=lambda: [
        [[[[1, 1]], [[2, 1]], [[3, 1]]]],
        [[[[1, 1]], [[2, 1]], [[3, 1]]]],
        [[[[1, 1]], [[2, 1]], [[3, 2]]]],
        [[[[1, 1]], [[2, 1]], [[3, 1]]]],
        [[[[1, 1]], [[2, 1]], [[3, 1]]]],
        [[[[1, 1]], [[2, 1]], [[3, 1]]]],
        [[[[1, 1]], [[2, 1]], [[3, 1]]]],
        [[[[1, 1]], [[2, 1]], [[3, 1]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]]
    ])

    # Noise configuration
    noise_std: float = 0.1

    # Model & Optimizer
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"
    n_restarts: int = 15
    swarm_size: int = 100
    random_seed: int = 0


class NoisyHeterogeneousTutorial:
    """
    Manages and executes a tutorial on noisy, heterogeneous WDD-GP models.
    """

    def __init__(self, config: NoisyHeteroConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.rng = np.random.RandomState(config.random_seed)
        self.training_data: Dict = {}
        self.gp_model = None
        self.params = None
        self.results: Dict = {}

    def _reorder_training_data(self, X_initial: np.ndarray):
        """Reorders training data to create a sequential index for the GP."""
        cfg = self.config
        old_groups = cfg.submodel_groups_initial

        sequential_groups = []
        current_pos = 0
        for group in old_groups:
            group_len = len(group)
            sequential_groups.append(
                list(range(current_pos, current_pos + group_len)))
            current_pos += group_len

        old_flat = list(itertools.chain.from_iterable(old_groups))
        new_flat = list(itertools.chain.from_iterable(sequential_groups))

        reorder_map = np.zeros(len(old_flat), dtype=int)
        for i in range(len(old_flat)):
            reorder_map[new_flat[i]] = old_flat[i]

        self.training_data['X_train'] = X_initial[reorder_map]
        self.training_data['submodel_indices'] = sequential_groups

    def _generate_training_data(self):
        """Generates the complex, per-submodel training data with noise."""
        print("\n" + "="*50 +
              "\nGenerating Heterogeneous and Noisy Training Data\n" + "="*50)
        cfg = self.config

        x_vals = np.linspace(
            cfg.domain_bounds[0][0], cfg.domain_bounds[0][1], cfg.num_pts_per_axis)
        y_vals = np.linspace(
            cfg.domain_bounds[1][0], cfg.domain_bounds[1][1], cfg.num_pts_per_axis)
        X_initial = np.array(list(itertools.product(x_vals, y_vals)))
        self._reorder_training_data(X_initial)

        X_train = self.training_data['X_train']
        submodel_indices = self.training_data['submodel_indices']

        y_func_values_clean = self.true_function(X_train, alg=np)
        y_func_values_noisy = y_func_values_clean + \
            self.rng.normal(loc=0.0, scale=0.0,
                            size=y_func_values_clean.shape)

        y_train_data_all, rays_data_all = [], []

        for k, group_indices in enumerate(submodel_indices):
            X_sub = X_train[group_indices]
            X_pert = oti.array(X_sub)

            thetas = cfg.submodel_ray_thetas[k]
            rays = np.column_stack([[np.cos(t), np.sin(t)] for t in thetas])
            rays_data_all.append(rays)

            e_bases = [oti.e(i + 1, order=cfg.n_order)
                       for i in range(len(thetas))]
            perts = np.dot(rays, e_bases)
            for j in range(cfg.n_bases):
                X_pert[:, j] += perts[j]

            y_hc = self.true_function(X_pert, alg=oti)
            for comb in itertools.combinations(range(1, len(thetas) + 1), 2):
                y_hc = y_hc.truncate(comb)

            y_train_submodel = [y_func_values_noisy]
            der_indices_submodel = cfg.submodel_der_indices[k]
            for group in der_indices_submodel:
                for sub_group in group:
                    deriv_clean = y_hc.get_deriv(sub_group)
                    deriv_noisy = deriv_clean + \
                        self.rng.normal(
                            loc=0.0, scale=cfg.noise_std, size=deriv_clean.shape)
                    y_train_submodel.append(deriv_noisy)

            y_train_data_all.append(y_train_submodel)

        # The noise vector's structure must match the observation structure for a single submodel
        # We base it on the first submodel for consistency
        noise_std_vector = np.zeros((len(submodel_indices)+1)*X_train.shape[0])
        noise_std_vector[X_train.shape[0]::] = 0.1

        self.training_data.update({
            'y_train_data': y_train_data_all,
            'rays_data': rays_data_all,
            'der_indices': cfg.submodel_der_indices,
            'noise_std_vector': noise_std_vector
        })
        print(
            f"  Generated specific noisy derivative data for {len(submodel_indices)} submodels.")

    def _train_model(self):
        """Initializes and trains the wddegp model."""
        print("\n" + "="*50 + "\nTraining WDD-GP Model\n" + "="*50)
        cfg, data = self.config, self.training_data

        self.gp_model = wddegp(
            data['X_train'], data['y_train_data'], cfg.n_order, cfg.n_bases,
            data['submodel_indices'], data['der_indices'], data['rays_data'],
            normalize=cfg.normalize_data, kernel=cfg.kernel, kernel_type=cfg.kernel_type,
            sigma_data=data['noise_std_vector']
        )
        print("  Model initialization: SUCCESS")

        print("  Optimizing hyperparameters...")
        self.params = self.gp_model.optimize_hyperparameters(
            n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size)
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

        y_pred, _ = self.gp_model.predict(
            X_test, self.params, calc_cov=False, return_submodels=True)
        y_true = self.true_function(X_test, alg=np)

        self.results = {
            'X_test': X_test, 'y_pred': y_pred, 'y_true': y_true,
            'X1_grid': X1_grid, 'X2_grid': X2_grid, 'nrmse': utils.nrmse(y_true, y_pred)
        }
        print(
            f"  Evaluation complete. Final NRMSE: {self.results['nrmse']:.6f}")

    def visualize_results(self):
        """Generates a 3-panel plot of the prediction, truth, and error."""
        print("\n" + "="*50 + "\nGenerating Visualizations\n" + "="*50)
        res = self.results
        X_train = self.training_data['X_train']

        y_true_grid = res['y_true'].reshape(res['X1_grid'].shape)
        y_pred_grid = res['y_pred'].reshape(res['X1_grid'].shape)
        abs_error_grid = np.abs(y_true_grid - y_pred_grid)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        c1 = axes[0].contourf(res['X1_grid'], res['X2_grid'],
                              y_pred_grid, levels=50, cmap="viridis")
        fig.colorbar(c1, ax=axes[0])
        axes[0].set_title("GP Prediction")
        axes[0].scatter(X_train[:, 0], X_train[:, 1], c="red",
                        edgecolor="k", s=50, label="Training Data", zorder=5)

        c2 = axes[1].contourf(res['X1_grid'], res['X2_grid'],
                              y_true_grid, levels=50, cmap="viridis")
        fig.colorbar(c2, ax=axes[1])
        axes[1].set_title("True Function")
        axes[1].scatter(X_train[:, 0], X_train[:, 1], c="red",
                        edgecolor="k", s=50, zorder=5)

        c3 = axes[2].contourf(res['X1_grid'], res['X2_grid'],
                              abs_error_grid, levels=50, cmap="magma")
        fig.colorbar(c3, ax=axes[2])
        axes[2].set_title("Absolute Error")
        axes[2].scatter(X_train[:, 0], X_train[:, 1], c="red",
                        edgecolor="k", s=50, zorder=5)

        for ax in axes:
            ax.set(xlabel="x₁", ylabel="x₂", aspect="equal")
        axes[0].legend()
        plt.tight_layout()
        plt.show()

    def run(self):
        """Executes the complete tutorial workflow."""
        print("DEGP Tutorial: Noisy Heterogeneous Submodels")
        print("=" * 75)

        self._generate_training_data()
        self._train_model()
        self._evaluate_model()
        self.visualize_results()

        print(f"\nTutorial Complete. Final NRMSE: {self.results['nrmse']:.6f}")

# --- Helper Function ---


def true_function(X, alg=np):
    """2D test function with polynomial and oscillatory components."""
    x1, x2 = X[:, 0], X[:, 1]
    return x1**2 * x2 + alg.cos(10 * x1) + alg.cos(10 * x2)


def main():
    """Main execution block."""
    config = NoisyHeteroConfig()
    tutorial = NoisyHeterogeneousTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
