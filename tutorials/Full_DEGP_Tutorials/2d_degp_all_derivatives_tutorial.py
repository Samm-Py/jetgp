"""
--------------------------------------------------------------------------------
This script demonstrates a derivative-enhanced Gaussian Process (GP) model
for a two-dimensional function using pyOTI-based automatic differentiation.
We generate a structured training set in 2D, compute function values and
derivatives up to a specified order, and train a GP on this enriched dataset.
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
    n_order = 3      # Max derivative order to include
    n_bases = 2      # Number of input dimensions
    lb_x, ub_x = -1, 1
    lb_y, ub_y = -1, 1

    # Derivative indices: include all derivatives up to n_order
    der_indices = utils.gen_OTI_indices(n_bases, n_order)

    # ----- Generate Training Inputs -----
    num_points = 4
    x_vals = np.linspace(lb_x, ub_x, num_points)
    y_vals = np.linspace(lb_y, ub_y, num_points)
    X_train = np.array(list(itertools.product(x_vals, y_vals)))

    # Create perturbed inputs for hypercomplex AD
    X_train_pert = oti.array(X_train)
    for i in range(n_bases):
        X_train_pert[:, i] += oti.e(i + 1, order=n_order)

    # ----- Define True Function -----
    def true_function(X, alg=oti):
        x1 = X[:, 0]
        x2 = X[:, 1]
        return x1**2 * x2 + alg.cos(10 * x1) + alg.cos(10 * x2)

    # Evaluate function and derivatives at training points
    y_train_hc = true_function(X_train_pert)
    y_train_real = y_train_hc.real

    y_train = [y_train_real]
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            deriv = y_train_hc.get_deriv(der_indices[i][j])
            y_train.append(deriv)

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
        n_restart_optimizer=50,
        swarm_size=50
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

    y_true = true_function(X_test, alg=np).flatten()
    nrmse = utils.nrmse(y_true, y_pred)

    print("NRMSE between model and true function: {}".format(nrmse))
