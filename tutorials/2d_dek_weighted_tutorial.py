import numpy as np
import pyoti.sparse as oti  # For automatic differentiation using hyper-complex numbers
import itertools
from oti_gp import (
    oti_gp_weighted,
)  # Weighted derivative-enhanced Gaussian Process class
import utils  # Utility functions (e.g., generating derivative indices, plotting submodels)

# ---------------------------------------------------------------------
# DEMO: Multi-dimensional Example with Separate Length Scales (2D)
# Using a weighted Gaussian Process model with submodel construction.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(0)

    # ----- Parameter Setup -----
    n_order = 2  # Use first-order derivative information
    n_bases = 2  # The function is two-dimensional (x1 and x2)
    lb_x = -1  # Lower bound for x1
    ub_x = 1  # Upper bound for x1
    lb_y = -1  # Lower bound for x2
    ub_y = 1  # Upper bound for x2

    num_points = 3  # Number of points along each axis (total training points = num_points^2)

    # Generate a grid of training points over the square region.
    x_vals = np.linspace(lb_x, ub_x, num_points)
    y_vals = np.linspace(lb_y, ub_y, num_points)
    X_train = np.array(list(itertools.product(x_vals, y_vals)))

    # Generate indices for all derivatives up to n_order for a function with n_bases dimensions.
    # Note that the derivatives used for each submodel can be different. In this particular case
    # we assume that the derivative information used to construct each submodel is the same.
    der_indices = [
        utils.gen_OTI_indices(n_bases, n_order) for _ in range(len(X_train))
    ]

    # To use different derivative information fo each submodel one would supply derivative information as:
    # Below each row corresponds to the derivative information that will be used to construct the submodel corresponding
    # to a particular training point
    # der_indices = [
    #     [[[[1, 1]], [[2, 1]]]],
    #     [[[[1, 1]], [[2, 1]]]],
    #     [[[[1, 1]], [[2, 1]], [[1, 2]], [[1, 1], [2, 1]], [[2, 2]]]],
    #     [[[[1, 1]], [[2, 1]], [[1, 2]], [[2, 2]]]],
    #     [[[[1, 1]], [[2, 1]], [[1, 2]], [[2, 2]]]],
    #     [[[[1, 1]], [[2, 1]], [[1, 2]], [[2, 2]]]],
    #     [[[[1, 1]], [[2, 1]], [[1, 2]], [[1, 1], [2, 1]], [[2, 2]]]],
    #     [[[[1, 1]], [[2, 1]]]],
    #     [[[[1, 1]], [[2, 1]]]],
    # ]

    # If one wishes to use different derivative information for each points, the indices must
    # be supplied such that each training point is supplied with a list of relavant derivative information:

    # Define the true function outside of the loop (common for all submodels)
    # Here, f(x1,x2) = sin(pi*x1) + cos(pi*x2)
    def true_function(X, alg=oti):
        x1 = X[:, 0]
        x2 = X[:, 1]
        return alg.sin(np.pi * x1) + alg.cos(np.pi * x2)

    # ----- Assemble Training Data for Submodels -----
    # The 'index' variable specifies which training points form each submodel.
    # For a global model, we require one submodel per training point.
    index = [[i] for i in range(len(X_train))]
    y_train_data = (
        []
    )  # List to hold training outputs (function values + derivatives) for each submodel
    sigma_n_true = 1e-8  # Known noise variance (set very small)

    # Loop over each group of indices in 'index'
    for k, val in enumerate(index):
        # Extract the training points corresponding to the current index group.
        X_train_subset = X_train[val]

        # Convert these training points to an OTI array to enable automatic derivative tracking.
        X_train_pert = oti.array(X_train_subset)

        # Perturb each training point along each coordinate direction.
        for i in range(1, n_bases + 1):
            for j in range(X_train_pert.shape[0]):
                X_train_pert[j, i - 1] = X_train_pert[j, i - 1] + oti.e(
                    i, order=n_order
                )

        # Evaluate the true function on the perturbed inputs using OTI to capture derivative information.
        y_train_hc = oti.array([true_function(x) for x in X_train_pert])
        # Also evaluate the true function on the original inputs (using numpy) for real function values.
        y_train_real = true_function(X_train, alg=np)

        # Begin with the real function values.
        y_train = y_train_real.reshape(-1, 1)
        # Append derivative information extracted from the hyper-complex outputs.
        for i in range(len(der_indices[k])):
            for j in range(len(der_indices[k][i])):
                y_train = np.vstack(
                    (y_train, y_train_hc.get_deriv(der_indices[k][i][j]))
                )

        # Flatten the assembled training data into a 1D array.
        y_train = y_train.flatten()

        # Optionally add noise (here, sigma_n_true is very small).
        noise = sigma_n_true * np.random.randn(len(y_train))
        y_train_noisy = y_train + noise

        # Append the processed training output for this submodel.
        y_train_data.append(y_train)

    # ----- Weighted Gaussian Process Model Setup -----
    # Create a weighted GP model that handles submodel data.
    gp = oti_gp_weighted(
        X_train,  # Original training inputs.
        y_train_data,  # List of training outputs (function values and derivatives) for each submodel.
        n_order,  # Order of derivative information.
        n_bases,  # Dimensionality of the input space.
        index,  # Grouping of training points for submodel construction.
        der_indices,
        sigma_n=1e-6,  # Noise variance (set very low).
        nugget=1e-6,  # Nugget term for numerical stability.
        kernel="SE",  # Use Squared Exponential (SE) kernel.
        kernel_type="anisotropic",  # Anisotropic kernel (separate length scales per dimension).
    )

    # Optimize the GP hyperparameters (e.g., length scales, kernel variance).
    params = gp.optimize_hyperparameters()

    # ----- Generate Test Data for Prediction -----
    N_grid = 25  # Number of test points per axis.
    x_lin = np.linspace(lb_x, ub_x, N_grid)
    y_lin = np.linspace(lb_y, ub_y, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    # ----- GP Prediction -----
    # Predict the function on the test data using the optimized GP model.
    # Return both the overall prediction and the submodel-specific predictions.
    y_pred, submodel_vals = gp.predict(
        X_test, params, calc_cov=False, return_submodels=True
    )

    # ----- Plotting via Utility Function -----
    # Instead of manually plotting, call the utility function to generate submodel plots.
    # The plotting function compares the GP prediction (and submodel contributions) with the true function.
    utils.make_submodel_plots(
        X_train,  # Training inputs.
        y_train_data,  # Training outputs for each submodel.
        X_test,  # Test inputs.
        y_pred,  # GP mean predictions.
        true_function,  # Function handle for the true function.X1_grid=0,
        X1_grid=X1_grid,
        X2_grid=X2_grid,
        n_order=n_order,  # Order of derivative information.
        n_bases=n_bases,  # Dimensionality of the input space.
        plot_submodels=True,  # Flag to plot individual submodel predictions.
        submodel_vals=submodel_vals,  # Predicted values from each submodel.
    )
