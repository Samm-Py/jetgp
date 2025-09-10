"""
================================================================================
DEGP Tutorial: Active Learning for Reliability Analysis
================================================================================

This tutorial demonstrates an advanced application of Directional-Derivative GPs
(DD-GP) for active learning in the context of reliability analysis. The goal is
to efficiently find a "failure" region, defined by a threshold on a function,
by intelligently selecting the next sample point.

The model uses its own uncertainty to decide where to sample next. This is a
core concept in active learning or "smart" Design of Experiments (DoE).

Key concepts covered:
-   Active learning with a DD-GP to guide data acquisition.
-   Using **Expected Contour Level (ECL) entropy** as an acquisition function
    to find the most informative next sample point.
-   Calculating and visualizing the **Probability of Failure (PoF)**, a key
    metric in reliability engineering.
-   Applying these advanced techniques to a 2D engineering-like function.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from scipy.stats import norm, qmc
from full_gddegp.gddegp import gddegp
import pyoti.sparse as oti
from dataclasses import dataclass
from typing import Dict, Callable

plt.rcParams.update({'font.size': 12})


@dataclass
class ReliabilityStudyConfig:
    """Configuration for the Active Learning DEGP tutorial."""
    n_order: int = 1
    n_bases: int = 2
    num_training_pts: int = 15
    domain_box: tuple = ((-2, 2), (-2, 2))
    failure_threshold: float = 5.0
    test_grid_resolution: int = 50
    normalize_data: bool = True
    kernel: str = "SE"
    kernel_type: str = "anisotropic"
    n_restarts: int = 20
    swarm_size: int = 100
    random_seed: int = 42


class ActiveLearningDEGPTutorial:
    """
    Manages and executes a DEGP-based active learning and reliability study.
    """

    def __init__(self, config: ReliabilityStudyConfig, true_function: Callable):
        self.config = config
        self.true_function = true_function
        self.rng = np.random.RandomState(config.random_seed)
        self.training_data: Dict = {}
        self.gp_model = None
        self.params = None
        self.results: Dict = {}

    def _generate_training_data(self):
        """Generates initial training data using LHS and random directional rays."""
        print("\n" + "="*50 + "\nGenerating Initial Training Data\n" + "="*50)
        cfg = self.config

        sampler = qmc.LatinHypercube(d=cfg.n_bases, seed=cfg.random_seed)
        unit_samples = sampler.random(n=cfg.num_training_pts)
        X_train = qmc.scale(unit_samples, [b[0] for b in cfg.domain_box], [
                            b[1] for b in cfg.domain_box])

        rays_list, tag_map = [], []
        for i in range(cfg.num_training_pts):
            theta = self.rng.uniform(0, 2 * np.pi)
            ray = np.array([[np.cos(theta)], [np.sin(theta)]])
            rays_list.append(ray)
            tag_map.append(i + 1)

        X_pert = oti.array(X_train)
        for i, ray in enumerate(rays_list):
            e_tag = oti.e(1, order=cfg.n_order)
            perturbation = oti.array(ray) * e_tag
            X_pert[i, :] += perturbation.T

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
            'rays_array': np.hstack(rays_list)
        }
        print(f"  Generated {len(X_train)} initial training points.")

    def _train_model(self):
        """Initializes and trains the GD-DEGP model."""
        print("\n" + "="*50 + "\nTraining GD-DEGP Model\n" + "="*50)
        cfg, data = self.config, self.training_data

        self.gp_model = gddegp(data['X_train'], data['y_train_list'], n_order=cfg.n_order, rays_array=data['rays_array'],
                               normalize=cfg.normalize_data, kernel=cfg.kernel, kernel_type=cfg.kernel_type)
        print("  Model initialization: SUCCESS")

        print("  Optimizing hyperparameters...")
        self.params = self.gp_model.optimize_hyperparameters(
            n_restart_optimizer=cfg.n_restarts, swarm_size=cfg.swarm_size)
        print("  Hyperparameter optimization: SUCCESS")

    def _perform_active_learning_step(self):
        """
        Uses the trained model's uncertainty to find the next best sample point.
        """
        print("\n" + "="*50 + "\nPerforming Active Learning Step\n" + "="*50)
        cfg = self.config

        gx = np.linspace(
            cfg.domain_box[0][0], cfg.domain_box[0][1], cfg.test_grid_resolution)
        gy = np.linspace(
            cfg.domain_box[1][0], cfg.domain_box[1][1], cfg.test_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(gx, gy)
        X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

        dummy_ray = np.array([[1.0], [0.0]])
        rays_pred = np.hstack([dummy_ray] * X_test.shape[0])

        mu, var = self.gp_model.predict(
            X_test, rays_pred, self.params, calc_cov=True)

        # Use ECL Entropy as the acquisition function to find the most informative point
        entropy_flat = ecl_entropy(mu, var, cfg.failure_threshold)
        idx_next = np.argmax(entropy_flat)
        next_point = X_test[idx_next]

        print(f"  Acquisition Function: Expected Contour Level (ECL) Entropy.")
        print(
            f"  Next best point to sample identified at: [{next_point[0]:.3f}, {next_point[1]:.3f}]")

        self.results = {
            'X1_grid': X1_grid, 'X2_grid': X2_grid,
            'mu_grid': mu.reshape(X1_grid.shape),
            'var_grid': var.reshape(X1_grid.shape),
            'entropy_grid': entropy_flat.reshape(X1_grid.shape),
            'y_true_grid': self.true_function(X_test, alg=np).reshape(X1_grid.shape),
            'next_point': next_point
        }

    def _visualize_results(self):
        """Generates the GP/ECL plot and the Probability of Failure plot."""
        print("\n" + "="*50 + "\nGenerating Visualizations\n" + "="*50)
        self._visualize_gp_and_entropy()
        self._visualize_pof()

    def _visualize_gp_and_entropy(self):
        """Plots the GP mean and the ECL entropy acquisition function."""
        cfg, res = self.config, self.results
        X_train = self.training_data['X_train']
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

        # Panel 1: GP Mean
        levels = np.linspace(res['mu_grid'].min(), res['mu_grid'].max(), 40)
        cf = ax1.contourf(res['X1_grid'], res['X2_grid'],
                          res['mu_grid'], levels=levels, cmap="viridis")
        fig.colorbar(cf, ax=ax1, label="GP mean")
        ax1.contour(res['X1_grid'], res['X2_grid'], res['mu_grid'], levels=[
                    cfg.failure_threshold], colors='red', linewidths=2)
        ax1.contour(res['X1_grid'], res['X2_grid'], res['y_true_grid'], levels=[
                    cfg.failure_threshold], colors='k', linewidths=2, linestyles='--')
        ax1.scatter(X_train[:, 0], X_train[:, 1], c="red",
                    edgecolor="k", s=40, zorder=5)
        ax1.scatter(res['next_point'][0], res['next_point'][1], marker='*',
                    color='blue', s=200, edgecolor='white', linewidth=1.7, zorder=10)
        ax1.set(xlabel="x₁", ylabel="x₂",
                title="GP Mean and Next Sample Point")

        # Panel 2: ECL Entropy
        ent_levels = np.linspace(
            res['entropy_grid'].min(), res['entropy_grid'].max(), 40)
        cf2 = ax2.contourf(res['X1_grid'], res['X2_grid'],
                           res['entropy_grid'], levels=ent_levels, cmap="inferno")
        fig.colorbar(cf2, ax=ax2, label="ECL entropy")
        ax2.contour(res['X1_grid'], res['X2_grid'], res['mu_grid'], levels=[
                    cfg.failure_threshold], colors='red', linewidths=2)
        ax2.scatter(res['next_point'][0], res['next_point'][1], marker='*',
                    color='blue', s=200, edgecolor='white', linewidth=1.7, zorder=10)
        ax2.set(xlabel="x₁", ylabel="x₂",
                title="Acquisition Function (ECL Entropy)")

        plt.tight_layout()
        plt.show()

    def _visualize_pof(self):
        """Calculates and plots the Probability of Failure."""
        cfg, res = self.config, self.results
        X_train = self.training_data['X_train']

        std_grid = np.sqrt(np.maximum(res['var_grid'], 1e-12))
        z_grid = (cfg.failure_threshold - res['mu_grid']) / std_grid
        pof_grid = 1 - norm.cdf(z_grid)

        fig, ax = plt.subplots(figsize=(7, 6))
        pf = ax.contourf(res['X1_grid'], res['X2_grid'],
                         pof_grid, levels=30, cmap="Reds")
        fig.colorbar(pf, ax=ax, label="Probability of Failure (PoF)")

        ax.contour(res['X1_grid'], res['X2_grid'], res['mu_grid'], levels=[
                   cfg.failure_threshold], colors='red', linewidths=2, linestyles='-')
        ax.contour(res['X1_grid'], res['X2_grid'], res['y_true_grid'], levels=[
                   cfg.failure_threshold], colors='k', linewidths=2, linestyles='--')
        ax.scatter(X_train[:, 0], X_train[:, 1], c="red",
                   edgecolor="k", s=40, zorder=5)
        ax.scatter(res['next_point'][0], res['next_point'][1], marker='*',
                   color='blue', s=200, edgecolor='white', linewidth=1.7, zorder=10)
        ax.set(xlabel="x₁", ylabel="x₂",
               title=f"Probability of f(x) > {cfg.failure_threshold}")

        plt.tight_layout()
        plt.show()

    def run(self):
        """Executes the complete tutorial workflow."""
        print("DEGP Tutorial: Active Learning for Reliability Analysis")
        print("=" * 75)

        self._generate_training_data()
        self._train_model()
        self._perform_active_learning_step()
        self._visualize_results()

        print("\nTutorial Complete.")

# --- Helper Functions ---


def true_function(X, alg=np):
    """Quadratic-plus-linear toy function with oscillations."""
    x, y = X[:, 0], X[:, 1]
    return 3*x**2 + 2*y**2 + x + 2*alg.sin(2*x)*alg.cos(1.5*y)


def ecl_entropy(mu, var, T):
    """Calculates the Expected Contour Level (ECL) entropy."""
    std = np.sqrt(np.maximum(var, 1e-12))
    z = (mu.flatten() - T) / std.flatten()
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    # Clamp cdf to avoid log(0) issues
    eps = 1e-12
    cdf_z = np.clip(cdf_z, eps, 1 - eps)
    # The entropy formula is H = -p*log(p) - (1-p)*log(1-p)
    return -cdf_z * np.log(cdf_z) - (1 - cdf_z) * np.log(1 - cdf_z)


def main():
    """Main execution block."""
    config = ReliabilityStudyConfig()
    tutorial = ActiveLearningDEGPTutorial(config, true_function)
    tutorial.run()


if __name__ == "__main__":
    main()
