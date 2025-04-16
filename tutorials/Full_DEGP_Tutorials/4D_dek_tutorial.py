import numpy as np
# Library for automatic differentiation using hyper-complex numbers
import pyoti.sparse as oti
from oti_gp import oti_gp  # Derivative-enhanced Gaussian Process class
import utils  # Utility functions, including one to generate derivative indices
import modules.sobol as sb


if __name__ == "__main__":
    # Set the random seed for reproducibility
    np.random.seed(1354)
    n_bases = 4
    n_order = 3

    num_points_test = 5000
    num_points_train = 26
    quasi = sb.create_sobol_samples(num_points_test, n_bases, 1).T

    lower_bounds = [-2.048 for i in range(4)]
    upper_bounds = [2.048 for i in range(4)]

    X_test = utils.scale_samples(quasi, lower_bounds, upper_bounds)

    def true_function(X, alg=oti):
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        f = (
            100 * (x2 - x1**2) ** 2
            + (1 - x1) ** 2
            + 100 * (x3 - x2**2) ** 2
            + (1 - x2) ** 2
            + 100 * (x4 - x3**2) ** 2
            + (1 - x3) ** 2
        )

        return f

    rmse_data = []

    # Generate indices for all derivatives up to the specified order
    # in a function with n_bases input dimensions.
    der_indices = utils.gen_OTI_indices(n_bases, n_order)

    # If the use wants only to use for example main derivatives in the
    # training process set der_indices as:
    # der_indices = [
    #     [[[1, 1]], [[2, 1]]],
    #     [[[1, 2]], [[2, 2]]],
    #     [[[1, 3]], [[2, 3]]],
    #     [[[1, 4]], [[2, 4]]],
    # ]
    # If the use wants only to use for example first and highest order derivatives in the
    # training process set der_indices as:
    # der_indices = [
    #     [[[1, 1]], [[2, 1]]],
    #     [[[1, 2]], [[2, 2]]],
    # ]
    # We use 5 points for this simple example. In a real case, choose
    # more or fewer points depending on the function's complexity.

    # ----- Generate Training Data -----
    quasi = sb.create_sobol_samples(num_points_train, n_bases, 1).T

    lower_bounds = [-2.048 for i in range(4)]
    upper_bounds = [2.048 for i in range(4)]

    X_train = utils.scale_samples(quasi, lower_bounds, upper_bounds)

    # Convert training data to an OTI array that supports derivative tracking
    X_train_pert = oti.array(X_train)

    # Perturb the training inputs along each coordinate direction to enable derivative computation.
    # For each input dimension, add the elementary perturbation defined by oti.e
    for i in range(1, n_bases + 1):
        X_train_pert[:, i - 1] = X_train_pert[:, i - 1] + oti.e(
            i, order=n_order
        )

    # Evaluate the true function on the perturbed training data.
    # The output is hyper-complex, containing both function value and derivative information.
    y_train_hc = true_function(X_train_pert)
    # Extract the real part (actual function values) from the hyper-complex output.
    y_train_real = y_train_hc.real

    # ----- Assemble Training Data with Derivative Information -----
    # Start with the real function values
    y_train = y_train_real
    # For each derivative index generated, extract the corresponding derivative
    # from the hyper-complex output and vertically stack it with the function values.
    for i in range(0, len(der_indices)):
        for j in range(0, len(der_indices[i])):
            y_train = np.vstack(
                (y_train, y_train_hc.get_deriv(der_indices[i][j]))
            )

    # Flatten the training output into a 1D array (required format for many GP implementations)
    y_train = y_train.flatten()

    # ----- Gaussian Process Model Setup -----
    # Create the derivative-enhanced Gaussian Process model.
    # We pass the original training inputs (X_train) along with the training outputs (y_train)
    # that include both function values and derivative information.
    gp = oti_gp(
        X_train,  # Unperturbed training inputs
        y_train,  # Training outputs (function values and derivatives)
        n_order,  # Order of derivative information used
        n_bases,  # Dimensionality of the input space
        der_indices,  # List of which derivatives to include
        kernel="SE",  # Kernel choice: Rational Quadratic (RQ) kernel
        # Anisotropic kernel to allow different length-scales per dimension
        kernel_type="anisotropic",
    )

    # Optimize the GP hyperparameters (e.g., length-scales, kernel variance) by maximizing the likelihood
    params = gp.optimize_hyperparameters(n_restart_optimizer=5, swarm_size=100)

    # ----- Generate Test Data for Prediction -----
    # Create a grid of test points over the same ranges as the training data.
    X_test = np.zeros((25 * 25, 4))
    N_grid = 25  # Number of grid points per axis for test data
    x_lin = np.linspace(-2.048, 2.048, N_grid)
    y_lin = np.linspace(-2.048, 2.048, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    # Combine the grid coordinates into a 2D array of test points.
    X_test[:, 0:2] = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    # ----- GP Prediction -----
    # Predict the function values on the test data using the optimized GP model.
    y_pred = gp.predict(X_test, params, calc_cov=False, return_deriv=False)
    y_true = true_function(X_test, alg=np)
    nrmse = nrmse(y_true, y_pred, norm_type="minmax")
    print("nrmse: {}".format(nrmse))

    utils.make_plots(
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
