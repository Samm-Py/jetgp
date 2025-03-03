import numpy as np
import matplotlib.pyplot as plt
import pyoti.sparse as oti  # Library for automatic differentiation using hyper-complex numbers
import itertools
from oti_gp import oti_gp  # Derivative-enhanced Gaussian Process class
import utils  # Utility functions, including one to generate derivative indices

if __name__ == "__main__":
    # Set the random seed for reproducibility
    np.random.seed(0)

    # ----- Parameter Setup -----
    n_order = 2  # Use 4th-order derivative information in the model
    n_bases = 2  # The function is two-dimensional (two input variables)
    lb_x = -1  # Lower bound for the first input variable (x1)
    ub_x = 1  # Upper bound for the first input variable (x1)
    lb_y = -1  # Lower bound for the second input variable (x2)
    ub_y = 1  # Upper bound for the second input variable (x2)

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
    num_points = 5  # Number of points per axis for training data
    x_vals = np.linspace(lb_x, ub_x, num_points)  # Uniform grid for x1 values
    y_vals = np.linspace(lb_y, ub_y, num_points)  # Uniform grid for x2 values

    # Create the Cartesian product of x_vals and y_vals to get a grid of training points
    X_train = np.array(list(itertools.product(x_vals, y_vals)))

    # Convert training data to an OTI array that supports derivative tracking
    X_train_pert = oti.array(X_train)

    # Perturb the training inputs along each coordinate direction to enable derivative computation.
    # For each input dimension, add the elementary perturbation defined by oti.e
    for i in range(1, n_bases + 1):
        X_train_pert[:, i - 1] = X_train_pert[:, i - 1] + oti.e(
            i, order=n_order
        )

    # ----- Define the True Function -----
    # This is an arbitrarily chosen polynomial function in two variables.
    # It has nonlinear behavior and is used here for demonstration.
    def true_function(X, alg=oti):
        x1 = X[:, 0]
        x2 = X[:, 1]
        f = x1**2 * x2 + alg.cos(10 * x1) + alg.cos(10 * x2)
        return f

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

    # ----- Noise Handling -----
    # sigma_n_true represents the known noise variance in the training outputs.
    # Here, it is set to zero (i.e., no noise is added) for simplicity.
    sigma_n_true = 0.0000
    noise = sigma_n_true * np.random.randn(len(y_train))
    y_train_noisy = (
        y_train + noise
    )  # Although no noise is added in this example

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
        sigma_n=1e-6,  # Noise variance (set to 0.0 here)
        nugget=1e-6,  # Small regularization term for numerical stability
        kernel="RQ",  # Kernel choice: Rational Quadratic (RQ) kernel
        kernel_type="anisotropic",  # Anisotropic kernel to allow different length-scales per dimension
    )

    # Optimize the GP hyperparameters (e.g., length-scales, kernel variance) by maximizing the likelihood
    params = gp.optimize_hyperparameters()

    # ----- Generate Test Data for Prediction -----
    # Create a grid of test points over the same ranges as the training data.
    N_grid = 25  # Number of grid points per axis for test data
    x_lin = np.linspace(lb_x, ub_x, N_grid)
    y_lin = np.linspace(lb_y, ub_y, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    # Combine the grid coordinates into a 2D array of test points.
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    # ----- GP Prediction -----
    # Predict the function values on the test data using the optimized GP model.
    y_pred = gp.predict(X_test, params, calc_cov=False, return_deriv=True)
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
        plot_derivative_surrogates=True,
        der_indices=der_indices,
    )
