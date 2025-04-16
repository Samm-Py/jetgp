"""
--------------------------------------------------------------------------------
This script demonstrates a derivative-enhanced Gaussian Process (GP) model
for a 4D function using pyOTI-based automatic differentiation and Sobol
sampling. We train the GP using function values and selected directional
derivatives (∂/∂x₁, ∂/∂x₂, ∂/∂x₃ and up to ∂²/∂x₁², ∂²/∂x₂², ∂²/∂x₃²), and
make predictions on a 2D slice of the input space (x₁-x₂ plane with x₃=x₄=0).
--------------------------------------------------------------------------------
"""

import numpy as np
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
import modules.sobol as sb
import plotting_helper

if __name__ == "__main__":
    # ----- Configuration -----
    np.random.seed(1354)
    n_bases = 4      # Input dimensionality
    n_order = 2      # Max derivative order used
    num_points_train = 26
    lower_bounds = [-2.048] * n_bases
    upper_bounds = [2.048] * n_bases

    # ----- Define Subset of Derivative Indices -----
    # Only include main directional derivatives up to second order for x₁, x₂, x₃
    der_indices = [
        [[[1, 1]], [[2, 1]], [[3, 1]], [[4, 1]]],   # ∂f/∂x₁, ∂f/∂x₂, ∂f/∂x₃
        # ∂²f/∂x₁², ∂²f/∂x₂², ∂²f/∂x₃²
        [[[1, 2]], [[2, 2]], [[3, 2]], [[4, 2]]],
    ]

    # ----- Define True Function -----
    def true_function(X, alg=oti):
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        return 1 * x1**2 + 2 * x2**2 + 3 * x3**2 + 4 * x4**2

    # ----- Generate Training Data -----
    sobol_train = sb.create_sobol_samples(num_points_train, n_bases, 1).T
    X_train = utils.scale_samples(sobol_train, lower_bounds, upper_bounds)

    # Perturb with hypercomplex structure for derivative tracking
    X_train_pert = oti.array(X_train)
    for i in range(n_bases):
        X_train_pert[:, i] += oti.e(i + 1, order=n_order)

    # Evaluate true function and extract function values + selected derivatives
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
        n_restart_optimizer=10,
        swarm_size=25,
        verbose=True
    )

    # ----- Generate 2D Slice Test Data (x₁-x₂ plane) -----
    N_grid = 25
    x_lin = np.linspace(-2.048, 2.048, N_grid)
    y_lin = np.linspace(-2.048, 2.048, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)

    X_test = np.zeros((N_grid**2, n_bases))
    X_test[:, 0:2] = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    X_test[:, 2:] = 0.0  # Fix x₃ and x₄ at 0 for 2D visualization

    # ----- GP Prediction -----
    y_pred = gp.predict(
        X_test,
        params,
        calc_cov=False,
        return_deriv=False
    )

    # ----- Visualization -----
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
    nrmse_val = utils.nrmse(y_true, y_pred)
    print("NRMSE between model and true function: {:.8f}".format(nrmse_val))
