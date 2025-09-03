"""
--------------------------------------------------------------------------------
This script demonstrates a derivative-enhanced Gaussian Process (GP) model for a
one-dimensional function, including derivatives up to a user-specified order.
We generate a small training set of points in the domain [lb_x, ub_x], compute
function values and a subset of derivative information using OTI-based hypercomplex
automatic differentiation, and train a GP on both function values and selected
derivatives. The GP can then make more informed predictions due to the added
derivative constraints. Finally, we visualize and compare the GP’s predictions
with the true function.
--------------------------------------------------------------------------------
"""

import numpy as np
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
import plotting_helper

if __name__ == "__main__":
    # n_order: the maximum derivative order used for perturbation
    # (note: not all orders are included in training)
    n_order = 2

    # n_bases: the dimensionality of the input space (1D in this example)
    n_bases = 1

    # lb_x, ub_x: domain bounds for sampling training data
    lb_x, ub_x = -2*np.pi, 2*np.pi
    num_points = 4

    # der_indices: uniform subset of derivatives to include at each point.
    # Here we include only first-order and fourth-order derivatives with respect to x.
    # This format assumes the same structure is applied at each training point.
    der_indices = utils.gen_OTI_indices(n_bases, n_order)

    print(der_indices)
    # X_train: evenly spaced points over the domain
    X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)

    # X_train_pert: convert to OTI array and apply perturbations up to order n_order
    # so we can later extract required derivatives (even if we only use some of them)
    X_train_pert = oti.array(X_train)
    for i in range(1, n_bases + 1):
        X_train_pert[:, i - 1] = X_train_pert[:,
                                              i - 1] + oti.e(i, order=n_order)

    # Define the true underlying function (combines exponential, sinusoidal, and linear terms)
    def true_function(X, alg=oti):
        x1 = X[:, 0]
        return alg.sin(x1)

    # Evaluate function at hypercomplex (perturbed) inputs
    y_train_hc = true_function(X_train_pert)
    y_train_real = y_train_hc.real

    # y_train: list of outputs and selected derivative components
    y_train = [y_train_real]
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            y_train.append(
                y_train_hc.get_deriv(der_indices[i][j]).reshape(-1, 1)
            )

    # Instantiate the derivative-enhanced GP model
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
    print(X_train)
    print(y_train)
    print(n_order)
    print(n_bases)
    print(der_indices)
    input('press enter to continue...')
    # Optimize hyperparameters using particle swarm optimization
    params = gp.optimize_hyperparameters(
        n_restart_optimizer=15,
        swarm_size=200
    )

    # Create test grid for prediction
    N_grid = 100
    X_test = np.linspace(lb_x, ub_x, N_grid).reshape(-1, 1)

    # Predict GP posterior mean and variance
    y_pred, y_var = gp.predict(
        X_test, params, calc_cov=True, return_deriv=False
    )

# Plot results comparing model predictions vs. ground truth
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

# Compute NRMSE as a quality metric
y_true = true_function(X_test, alg=np)
nrmse = utils.nrmse(y_true, y_pred)

print("NRMSE between model and true function: {}".format(nrmse))
