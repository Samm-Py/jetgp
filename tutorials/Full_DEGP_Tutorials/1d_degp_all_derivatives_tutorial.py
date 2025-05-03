"""
--------------------------------------------------------------------------------
This script demonstrates a derivative-enhanced Gaussian Process (GP) model for a
one-dimensional function, including derivatives up to a user-specified order.
We generate a small training set of points in the domain [lb_x, ub_x], compute
function values and derivative information using OTI-based hypercomplex automatic
differentiation, and train a GP on both function values
and derivatives. The GP can then make more informed predictions due to the
additional derivative constraints. Finally, we visualize and compare the GP’s
predictions with the true function.
--------------------------------------------------------------------------------
"""

import numpy as np
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
import plotting_helper

if __name__ == "__main__":
    # n_order: the maximum derivative order to include in the GP model
    n_order = 3

    # n_bases: the dimensionality of the input space (here 1D)
    n_bases = 1

    # lb_x, ub_x: lower and upper bounds in the 1D domain where we sample points
    lb_x = 0.2
    ub_x = 6

    # der_indices: a list describing which derivatives (up to n_order)
    # will be included in the training data.
    der_indices = utils.gen_OTI_indices(n_bases, n_order)

    # num_points: number of training points in the 1D domain
    num_points = 3

    # X_train: an array of size (num_points, 1) with equally spaced points
    # between lb_x and ub_x
    X_train = np.array([1.65, 3.1, 4.55]).reshape(-1, 1)

    # X_train_pert: convert X_train into an OTI array so we can track
    # derivative information. Then we “perturb” it to embed derivative
    # information up to order n_order for each dimension.
    X_train_pert = oti.array(X_train)
    for i in range(1, n_bases + 1):
        X_train_pert[:, i - 1] = X_train_pert[:,
                                              i - 1] + oti.e(i, order=n_order)

    # Define a 1D function that combines exponential, sine, cosine,
    # and a small linear term
    def true_function(X, alg=oti):
        x = X[:, 0]
        f = alg.exp(-x) + alg.sin(x) + alg.cos(3 * x) + 0.2 * x + 1.0
        return f

    # Evaluate the function at hypercomplex inputs
    y_train_hc = true_function(X_train_pert)
    y_train_real = y_train_hc.real

    # y_train: list of arrays holding function values and its derivatives
    y_train = [y_train_real]
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            y_train.append(
                y_train_hc.get_deriv(der_indices[i][j]).reshape(-1, 1)
            )

    # Instantiate the derivative-enhanced Gaussian process model
    gp = degp(
        # The input locations of the training data
        X_train,

        # A list of function values and derivatives corresponding to X_train
        y_train,

        # The maximum derivative order included in the training set
        n_order,

        # The dimensionality of the input space (1D in this example)
        n_bases,

        # A list specifying which derivatives are used (e.g., first order, second order, etc.)
        der_indices,

        # If True, automatically normalizes/scales the inputs and outputs for numerical stability
        normalize=False,
        # Kernel choice; "SE" means the Squared Exponential (RBF) kernel
        kernel="SE",

        # How the kernel handles different input dimensions; "anisotropic" allows dimension-specific length scales
        kernel_type="anisotropic",
    )

    # Optimize hyperparameters for the GP model using particle swarm optimization.
    # - n_restart_optimizer=25: allows up to 25 iterations of the swarm to refine candidate solutions.
    # - swarm_size=500: uses 500 particles (candidate points) in each iteration, providing a broad search.
    # The returned 'params' contains the final best hyperparameter values found.
    params = gp.optimize_hyperparameters(
        n_restart_optimizer=25,
        swarm_size=100
    )

    # Create a grid of test points for prediction
    N_grid = 100
    X_test = np.linspace(lb_x, ub_x, N_grid).reshape(-1, 1)

    # Make predictions at the test points (mean and variance)
    y_pred, y_var = gp.predict(
        X_test, params, calc_cov=True, return_deriv=False
    )


# Generate and display plots of the GP training data and predictions.
# - X_train, y_train: the training inputs and observed function/derivative values.
# - X_test, y_pred: the test inputs and predicted function values from the GP.
# - true_function: the actual function used for data generation (for reference in the plot).
# - cov=y_var: the predictive variance for plotting confidence intervals.
# - n_order, n_bases, and der_indices inform the plotting routine about the derivative orders.
# - plot_derivative_surrogates controls whether the GP’s derivative predictions are also plotted.
plotting_helper.make_plots(
    X_train,
    y_train,
    X_test,
    y_pred.flatten(),
    true_function,
    cov=y_var,
    n_order=n_order,
    n_bases=n_bases,
    # Set to True to visualize derivative predictions
    plot_derivative_surrogates=False,
    der_indices=der_indices,
)

y_true = true_function(X_test, alg=np)
nrmse = utils.nrmse(y_true, y_pred)

print("NRMSE between model and true function: {}".format(nrmse))
