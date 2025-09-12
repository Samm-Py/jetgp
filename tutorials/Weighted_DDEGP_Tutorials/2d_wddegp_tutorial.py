"""
================================================================================
DEGP Tutorial: Heterogeneous Submodels with Directional Derivatives
================================================================================

This tutorial demonstrates an advanced Weighted Directional-Derivative GP (WDD-GP)
with a heterogeneous structure. In this model, the input space is partitioned
into subregions, and each region is modeled by a submodel with its own unique
set of training points and its own basis of directional derivatives (rays).

This is a powerful technique for modeling complex functions where the behavior
(e.g., anisotropy, dominant directions of change) varies across the domain.

Key concepts covered:
-   **Heterogeneous Submodels**: Each submodel has its own points and rays.
-   Data reordering to create a sequential index for the GP framework.
-   Per-submodel automatic differentiation and data assembly.
-   Training the `wddegp` model with complex, structured inputs.
-   Visualizing the combined prediction from all submodels.
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
class HeterogeneousConfig:
    """Configuration for the Heterogeneous Submodel DEGP tutorial."""
    n_order: int = 2
    n_bases: int = 2
    num_pts_per_axis: int = 5
    domain_bounds: tuple = ((-1, 1), (-1, 1))
    test_grid_resolution: int = 50

    # Define the initial, non-sequential grouping of training points
    submodel_groups_initial: List[List[int]] = field(default_factory=lambda: [
        [1, 2, 3], [5, 10, 15], [9, 14, 19], [
            21, 22, 23], [0], [4], [20], [24],
        [6, 7, 8, 11, 12, 13, 16, 17, 18]
    ])

    # Define the ray angles (thetas) for EACH submodel group
    submodel_ray_thetas: List[List[float]] = field(default_factory=lambda: [
        [-np.pi/4, 0, np.pi/4], [-np.pi/4, 0, np.pi/4], [-np.pi/4, 0, np.pi/4],
        [-np.pi/4, 0, np.pi/4], [-np.pi/2, 0, -np.pi/4], [np.pi/2, 0, np.pi/4],
        [np.pi/2, 0, np.pi/4], [-np.pi/2, 0, -np.pi /
                                4], [np.pi/2, np.pi/4, np.pi/4 + np.pi/2]
    ])

    # Define the derivative indices to extract for EACH submodel group
    # Note: In this example, they are all the same, but they could be different.
    submodel_der_indices = [
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
    ]

    # Model & Optimizer
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"
    n_restarts: int = 15
    swarm_size: int = 100
    random_seed: int = 0


class HeterogeneousSubmodelTutorial:
    """
    Manages and executes a tutorial on heterogeneous WDD-GP models.
    """

    def __init__(self, config: HeterogeneousConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.training_data: Dict = {}
        self.gp_model = None
        self.params = None
        self.results: Dict = {}
        np.random.seed(config.random_seed)

    def _reorder_training_data(self, X_initial: np.ndarray):
        """Reorders training data to create a sequential index for the GP."""
        cfg = self.config
        old_groups = cfg.submodel_groups_initial

        # Create the required sequential index structure
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
        """Generates the complex, per-submodel training data required by wddgp."""
        print("\n" + "="*50 + "\nGenerating Heterogeneous Training Data\n" + "="*50)
        cfg = self.config

        # 1. Create and Reorder Initial Training Points
        x_vals = np.linspace(
            cfg.domain_bounds[0][0], cfg.domain_bounds[0][1], cfg.num_pts_per_axis)
        y_vals = np.linspace(
            cfg.domain_bounds[1][0], cfg.domain_bounds[1][1], cfg.num_pts_per_axis)
        X_initial = np.array(list(itertools.product(x_vals, y_vals)))
        self._reorder_training_data(X_initial)

        X_train = self.training_data['X_train']
        submodel_indices = self.training_data['submodel_indices']

        # 2. Loop through submodels to create their specific data
        y_train_data_all = []
        rays_data_all = []
        y_func_values = self.true_function(X_train, alg=np)

        for k, group_indices in enumerate(submodel_indices):
            X_sub = X_train[group_indices]
            X_pert = oti.array(X_sub)

            # Create rays from the angles for this specific submodel
            thetas = cfg.submodel_ray_thetas[k]
            rays = np.column_stack([[np.cos(t), np.sin(t)] for t in thetas])
            rays_data_all.append(rays)

            # Apply perturbations using this submodel's rays
            e_bases = [oti.e(i + 1, order=cfg.n_order)
                       for i in range(len(thetas))]
            perts = np.dot(rays, e_bases)
            for j in range(cfg.n_bases):
                X_pert[:, j] += perts[j]

            # Evaluate, truncate, and extract derivatives for this submodel
            y_hc = self.true_function(X_pert, alg=oti)
            for comb in itertools.combinations(range(1, len(thetas) + 1), 2):
                y_hc = y_hc.truncate(comb)

            # Each submodel's data list starts with ALL function values
            y_train_submodel = [y_func_values]
            der_indices_submodel = cfg.submodel_der_indices[k]
            for group in der_indices_submodel:
                for sub_group in group:
                    y_train_submodel.append(
                        y_hc.get_deriv(sub_group).reshape(-1, 1))

            y_train_data_all.append(y_train_submodel)

        self.training_data['y_train_data'] = y_train_data_all
        self.training_data['rays_data'] = rays_data_all
        self.training_data['der_indices'] = cfg.submodel_der_indices
        print(
            f"  Generated specific derivative data for {len(submodel_indices)} submodels.")

    def _train_model(self):
        """Initializes and trains the wddegp model."""
        print("\n" + "="*50 + "\nTraining WDD-GP Model\n" + "="*50)
        cfg, data = self.config, self.training_data

        self.gp_model = wddegp(
            data['X_train'], data['y_train_data'], cfg.n_order, cfg.n_bases,
            data['submodel_indices'], data['der_indices'], data['rays_data'],
            normalize=cfg.normalize_data, kernel=cfg.kernel, kernel_type=cfg.kernel_type
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
                        edgecolor="k", s=50, label="Training Data")

        c2 = axes[1].contourf(res['X1_grid'], res['X2_grid'],
                              y_true_grid, levels=50, cmap="viridis")
        fig.colorbar(c2, ax=axes[1])
        axes[1].set_title("True Function")
        axes[1].scatter(X_train[:, 0], X_train[:, 1],
                        c="red", edgecolor="k", s=50)

        c3 = axes[2].contourf(res['X1_grid'], res['X2_grid'],
                              abs_error_grid, levels=50, cmap="magma")
        fig.colorbar(c3, ax=axes[2])
        axes[2].set_title("Absolute Error")
        axes[2].scatter(X_train[:, 0], X_train[:, 1],
                        c="red", edgecolor="k", s=50)

        for ax in axes:
            ax.set(xlabel="x₁", ylabel="x₂")
        axes[0].legend()
        plt.tight_layout()
        plt.show()

    def run(self):
        """Executes the complete tutorial workflow."""
        print("DEGP Tutorial: Heterogeneous Submodels with Directional Derivatives")
        print("=" * 75)

        self._generate_training_data()
        self._train_model()
        self._evaluate_model()
        self.visualize_results()

        print(f"\nTutorial Complete. Final NRMSE: {self.results['nrmse']:.6f}")

# --- Helper Function ---


def true_function(X, alg=np):
    """2D Six-Hump Camel function."""
    x1, x2 = X[:, 0], X[:, 1]
    return ((4 - 2.1 * x1**2 + (x1**4) / 3.0) * x1**2 + x1 * x2 + (-4 + 4 * x2**2) * x2**2)


def main():
    """Main execution block."""
    config = HeterogeneousConfig()
    tutorial = HeterogeneousSubmodelTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
