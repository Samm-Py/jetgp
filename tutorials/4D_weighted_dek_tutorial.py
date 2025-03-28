import numpy as np
import pyoti.sparse as oti  # For automatic differentiation using hyper-complex numbers
import itertools
from oti_gp import (
    oti_gp_weighted,
)  # Weighted derivative-enhanced Gaussian Process class
import utils  # Utility functions (e.g., generating derivative indices, plotting submodels)
import modules.sobol as sb

# ---------------------------------------------------------------------
# DEMO: Multi-dimensional Example with Separate Length Scales (2D)
# Using a weighted Gaussian Process model with submodel construction.
# ---------------------------------------------------------------------


def nrmse(y_true, y_pred, norm_type="minmax"):
    """
    Compute the Normalized Root Mean Squared Error (NRMSE).

    Parameters:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.
        norm_type (str): Normalization type:
                         - 'minmax': divide by (max - min) of y_true
                         - 'mean': divide by mean of y_true
                         - 'std': divide by standard deviation of y_true

    Returns:
        float: NRMSE value.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    if norm_type == "minmax":
        norm = np.max(y_true) - np.min(y_true)
    elif norm_type == "mean":
        norm = np.mean(y_true)
    elif norm_type == "std":
        norm = np.std(y_true)
    else:
        raise ValueError("norm_type must be 'minmax', 'mean', or 'std'")

    return rmse / norm if norm != 0 else np.inf


def scale_samples(samples, lower_bounds, upper_bounds):
    """
    Scale each column of samples from [0, 1] to [lb_j, ub_j].

    Parameters:
        samples (ndarray): A (d, n) array of samples in [0, 1]^n.
        lower_bounds (array-like): Length-n array of lower bounds.
        upper_bounds (array-like): Length-n array of upper bounds.

    Returns:
        ndarray: A (d, n) array with each column scaled to its corresponding bounds.
    """
    samples = np.asarray(samples)
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)

    # Ensure correct shapes
    assert (
        samples.shape[1] == len(lower_bounds) == len(upper_bounds)
    ), "Dimension mismatch between samples and bounds"

    # Reshape bounds to broadcast across rows
    lb = lower_bounds[np.newaxis, :]
    ub = upper_bounds[np.newaxis, :]

    return lb + samples * (ub - lb)


if __name__ == "__main__":
    # Set the random seed for reproducibility
    np.random.seed(1354)
    n_bases = 4
    n_order = 1

    num_points_test = 5000
    num_points_train = 40
    quasi = sb.create_sobol_samples(num_points_test, n_bases, 1).T

    lower_bounds = [-2.048 for i in range(4)]
    upper_bounds = [2.048 for i in range(4)]

    X_test = scale_samples(quasi, lower_bounds, upper_bounds)

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

    quasi = sb.create_sobol_samples(num_points_train, n_bases, 1).T

    lower_bounds = [-2.048 for i in range(4)]
    upper_bounds = [2.048 for i in range(4)]

    # Number of points per axis (keep small for manageability)
    num_points_per_axis = 3  # adjust as needed

    # Create linspace for each dimension
    axes = [
        np.linspace(low, high, num_points_per_axis)
        for low, high in zip(lower_bounds, upper_bounds)
    ]

    # Create grid and reshape into list of 4D points
    mesh = np.meshgrid(*axes, indexing="ij")
    X_train = np.stack(mesh, axis=-1).reshape(-1, 4)

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

    # ----- Assemble Training Data for Submodels -----
    # The 'index' variable specifies which training points form each submodel.
    # For a global model, we require one submodel per training point.
    index = [[i] for i in range(len(X_train))]
    y_train_data = (
        []
    )  # List to hold training outputs (function values + derivatives) for each submodel

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

        y_train_noisy = y_train

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
        kernel="SE",  # Use Squared Exponential (SE) kernel.
        kernel_type="anisotropic",  # Anisotropic kernel (separate length scales per dimension).
    )

    # Optimize the GP hyperparameters (e.g., length scales, kernel variance).
    params = gp.optimize_hyperparameters(n_restart_optimizer=25, swarm_size=25)

    # ----- Generate Test Data for Prediction -----
    # Create a grid of test points over the same ranges as the training data.

    N_grid = 15  # Number of grid points per axis for test data
    X_test = np.zeros((N_grid * N_grid, 4))
    x_lin = np.linspace(-2.048, 2.048, N_grid)
    y_lin = np.linspace(-2.048, 2.048, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    # Combine the grid coordinates into a 2D array of test points.
    X_test[:, 0:2] = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    # ----- GP Prediction -----
    # Predict the function values on the test data using the optimized GP model.
    y_pred = gp.predict(X_test, params, calc_cov=False, return_submodels=False)
    y_true = true_function(X_test, alg=np)
    nrmse = nrmse(y_true, y_pred, norm_type="minmax")
    print("nrmse: {}".format(nrmse))
    # true_values = true_function(X_train, alg=np)
    # rmse = np.sqrt(np.mean((y_pred - true_values) ** 2))
    # print("RMSE between model and true function: {}".format(rmse))
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
        plot_submodels=False,  # Flag to plot individual submodel predictions.,  # Predicted values from each submodel.
    )
