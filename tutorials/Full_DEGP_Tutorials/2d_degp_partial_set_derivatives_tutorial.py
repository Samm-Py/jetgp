"""
--------------------------------------------------------------------------------
This script demonstrates a derivative-enhanced Gaussian Process (GP) model
for a two-dimensional function using pyOTI-based automatic differentiation.
We generate a structured training set in 2D, compute function values and only
the main derivatives (∂/∂x₁, ∂/∂x₂, etc.) up to a specified order,
and train a GP using this reduced derivative information set.
Finally, we visualize the model’s predictions and compare them to the true function.
--------------------------------------------------------------------------------
"""

import numpy as np
import pyoti.sparse as oti
import itertools
from full_degp.degp import degp
import utils
import plotting_helper

if __name__ == "__main__":
    # ----- Problem Configuration -----
    n_order = 2      # Maximum derivative order to include
    n_bases = 2      # Number of input dimensions
    lb_x, ub_x = -1, 1
    lb_y, ub_y = -1, 1

    # Use only main directional derivatives for training
    der_indices = [
        [[[1, 1]], [[2, 1]]],  # First-order derivatives: ∂f/∂x1, ∂f/∂x2
        [[[1, 2]], [[2, 2]]],  # Second-order: ∂²f/∂x1², ∂²f/∂x2²
    ]

    # ----- Generate Training Inputs -----
    num_points = 5
    x_vals = np.linspace(lb_x, ub_x, num_points)
    y_vals = np.linspace(lb_y, ub_y, num_points)
    X_train = np.array(list(itertools.product(x_vals, y_vals)))

    # Perturb inputs to embed derivative tracking
    X_train_pert = oti.array(X_train)
    for i in range(n_bases):
        X_train_pert[:, i] += oti.e(i + 1, order=n_order)

    # ----- Define True Function -----
    def true_function(X, alg=oti):
        x1 = X[:, 0]
        x2 = X[:, 1]
        return x1**2 * x2 + alg.cos(10 * x1) + alg.cos(10 * x2)

    # Evaluate function and compute selected derivatives
    y_train_hc = true_function(X_train_pert)
    y_train_real = y_train_hc.real

    y_train = [y_train_real]
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            y_train.append(y_train_hc.get_deriv(der_indices[i][j]))

    # ----- GP Model Setup -----
    gp = degp(
        X_train,
        y_train,
        n_order,
        n_bases,
        der_indices,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic",
    )

    # ----- Hyperparameter Optimization -----
    params = gp.optimize_hyperparameters(
        n_restart_optimizer=15,
        swarm_size=200
    )

    # ----- Generate Test Data -----
    N_grid = 25
    x_lin = np.linspace(lb_x, ub_x, N_grid)
    y_lin = np.linspace(lb_y, ub_y, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    # ----- GP Prediction -----
    y_pred = gp.predict(
        X_test,
        params,
        calc_cov=False,
        return_deriv=False
    )

    # ----- Plot Results -----
    plotting_helper.make_plots(
        X_train,
        y_train,
        X_test,
        y_pred,
        true_function,
        X1_grid=X1_grid,
        X2_grid=X2_grid,
        n_order=n_order,
        n_bases=n_bases,
        plot_derivative_surrogates=False,
        der_indices=der_indices,
    )

    # ----- Compute and Print NRMSE -----
    y_true = true_function(X_test, alg=np).flatten()
    nrmse = utils.nrmse(y_true, y_pred)
    print("NRMSE between model and true function: {:.4f}".format(nrmse))
