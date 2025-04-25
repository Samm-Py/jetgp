"""
--------------------------------------------------------------------------------
This script demonstrates a derivative-enhanced Gaussian Process (GP) model for a
one-dimensional function, including derivatives up to a user-specified order.
We generate a small training set of points in the domain [lb_x, ub_x], compute
function values and derivative information using OTI-based hypercomplex automatic
differentiation, and train a GP on both function values and derivatives.

We additionally inject Gaussian noise into both the function observations and
their derivatives to simulate realistic measurement noise. The GP is trained
with this noisy data, and the model accounts for this through Tikhonov
regularization (sigma_n). We visualize the predictions and compare them to
the true function.
--------------------------------------------------------------------------------
"""

import numpy as np
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
import plotting_helper

if __name__ == "__main__":
    rng = np.random.RandomState(1)
    # GP and function configuration
    n_order = 0       # Max derivative order included in training
    n_bases = 1       # Input dimension (1D)
    lb_x = 0          # Lower bound of input domain
    ub_x = 10         # Upper bound of input domain

    # Generate training input points from a dense candidate set
    X = np.linspace(lb_x, ub_x, 1000).reshape(-1, 1)
    rng = np.random.RandomState(1)
    training_indices = rng.choice(np.arange(X.shape[0]), size=6, replace=False)
    X_train = np.sort(X[training_indices], axis=0)
    n_train = len(X_train)
    # Convert to OTI array and perturb to track derivatives
    X_train_pert = oti.array(X_train)
    for i in range(n_bases):
        X_train_pert[:, i] += oti.e(i + 1, order=n_order)

    # Define true function
    def true_function(X, alg=oti):
        x = X[:, 0]
        return x * alg.sin(x)

    # Evaluate function with OTI to get derivatives
    y_train_hc = true_function(X_train_pert)
    y_train_real = y_train_hc.real

    # Add Gaussian noise to function and derivative observations

    # Derivative indices to include in training data
    der_indices = utils.gen_OTI_indices(n_bases, n_order)

    arr = np.zeros((len(der_indices) + 1)*n_train)
    arr[0] = 0.75
    arr[1] = 0.75
    arr[2] = 0.75
    arr[3] = 0.75
    arr[4] = 0.75
    arr[5] = 0.75
    noise_std = np.diag(arr)
    for i in range(0, len(y_train_real)):
        y_train_real_noisy = y_train_real + \
            rng.normal(loc=0.0, scale=arr[i], size=1)

    # Build y_train list: function values and noisy derivatives
    y_train = [y_train_real_noisy]
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            deriv = y_train_hc.get_deriv(der_indices[i][j]).reshape(-1, 1)
            deriv_noisy = deriv
            y_train.append(deriv_noisy)

    # Instantiate and configure the GP model
    gp = degp(
        X_train,
        y_train,
        n_order,
        n_bases,
        der_indices,
        normalize=True,
        sigma_data=noise_std,      # Informs the model about expected noise level
        kernel="SE",
        kernel_type="anisotropic",
    )

    # Optimize GP hyperparameters using particle swarm
    params = gp.optimize_hyperparameters(
        n_restart_optimizer=25,
        swarm_size=100
    )

    # Create test points and make predictions
    X_test = np.linspace(lb_x, ub_x, 100).reshape(-1, 1)
    y_pred, y_var = gp.predict(
        X_test, params, calc_cov=True, return_deriv=False
    )

    # Plot results
    plotting_helper.make_plots(
        X_train,
        y_train,
        X_test,
        y_pred.flatten(),
        true_function,
        cov=y_var,
        n_order=n_order,
        n_bases=n_bases,
        plot_derivative_surrogates=False,
        der_indices=der_indices,
    )

    # Compute and report normalized RMSE
    y_true = true_function(X_test, alg=np)
    nrmse = utils.nrmse(y_true, y_pred)
    print("NRMSE between model and true function: {:.4f}".format(nrmse))
