import numpy as np
# Library for automatic differentiation using hyper-complex numbers
import pyoti.sparse as oti
from wdegp.wdegp import wdegp
import utils

# ---------------------------------------------------------------------
# DEMO: Multi-dimensional Example with Separate Length Scales
# Using a weighted Gaussian Process model with submodel construction based
# on user-specified indices. Each submodel is constructed from a group of
# training points defined by the 'index' variable.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Define the true function:
    def true_function(X, alg=oti):
        x1 = X[:, 0]
        return alg.sin(10 * np.pi * x1) / (2 * x1) + (x1 - 1) ** 4

    # ----- Parameter Setup -----
    n_order = 3  # Use second-order derivative information in the model.
    n_bases = 1  # The function is one-dimensional (single input variable).
    lb_x = 0.5  # Lower bound for the training input values.
    ub_x = 2.5  # Upper bound for the training input values.

    # 'index' specifies the grouping of training points for submodel construction.
    # For example, with index = [[0], [1], [2], [3], [4]], each training point forms its own submodel.
    # This flexible grouping allows the user to define arbitrary subsets for submodels.
    # Note: For a global GP, the index must be a list of lists with length equal to the number of training points.
    num_points = 10  # Number of training points along the x-axis.
    index = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    # Use provided training values (non-uniform in this case).
    X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)

    # Generate indices for all derivatives up to n_order for a function with n_bases dimensions.
    # Note that the derivatives used for each submodel can be different. In this particular case
    # we assume that the derivative information used to construct each submodel is the same.
    der_indices = [
        utils.gen_OTI_indices(n_bases, n_order) for _ in range(len(index))
    ]

    # To use different derivative information fo each submodel one would supply derivative information as:
    # Below each row corresponds to the derivative information that will be used to construct the submodel corresponding
    # to a particular training point
    # der_indices = [
    #     [[[[1, 1]], [[1, 2]]]],
    #     [[[[1, 1]], [[1, 2]], [[1, 3]], [[1, 4]]]],
    #     [[[[1, 1]], [[1, 2]], [[1, 3]], [[1, 4]]]],
    #     [[[[1, 1]], [[1, 2]], [[1, 3]], [[1, 4]]]],
    #     [[[[1, 1]], [[1, 2]]]],
    # ]

    # ----- Assemble Training Data for Submodels -----
    # y_train_data will hold the training outputs (function values + derivatives)
    # for each submodel as defined by the groups in 'index'.
    y_train_data = []
    # Known noise variance in the training data (set to 0 for simplicity)
    sigma_n_true = 0.0

    # Loop over each group in 'index' to construct submodel training data.
    for k, val in enumerate(index):
        # For the current submodel, extract the training points based on indices in 'val'.
        # Convert the selected training points to an OTI array to enable automatic derivative tracking.
        X_train_pert = oti.array(X_train[val])

        # Perturb each training point along the coordinate directions.
        # This is necessary for computing derivatives via the OTI library.
        for i in range(1, n_bases + 1):
            for j in range(X_train_pert.shape[0]):
                X_train_pert[j, i - 1] = X_train_pert[j, i - 1] + oti.e(
                    i, order=n_order
                )

        # Evaluate the true function on the perturbed inputs to obtain hyper-complex outputs
        # that include both function values and derivative information.
        y_train_hc = oti.array(
            [true_function(x, alg=oti) for x in X_train_pert]
        )
        # Also evaluate the true function on the original inputs (using numpy) to get the real values.
        y_train_real = true_function(X_train, alg=np)

        # Start building the training output for the submodel with the real function values.
        y_train = [y_train_real]
        # Append derivative information extracted from the hyper-complex outputs.
        for i in range(0, len(der_indices[k])):
            for j in range(0, len(der_indices[k][i])):
                y_train.append(y_train_hc.get_deriv(
                    der_indices[k][i][j]).reshape(-1, 1))

        # Append the processed training output for this submodel.
        y_train_data.append(y_train)

    sigma_f = 1.0  # (Defined but not used; could be removed)
    sigma_n = sigma_n_true  # (Same as above)

    # ----- Weighted Gaussian Process Model Setup -----
    # Create a weighted Gaussian Process model designed for submodel data.
    # The 'index' parameter defines the grouping of training points for submodel construction.
    gp = wdegp(
        X_train,
        y_train_data,
        n_order,  # Order of derivative information included.
        n_bases,  # Dimensionality of the input space.
        index,  # Grouping indices for submodel construction.
        der_indices,
        normalize=True,
        kernel="SE",  # Use Squared Exponential (SE) kernel.
        # Anisotropic kernel allowing separate length scales per dimension.
        kernel_type="anisotropic",
    )

    # Optimize the GP hyperparameters (e.g., length scales, kernel variance) via likelihood maximization.
    params = gp.optimize_hyperparameters(
        n_restart_optimizer=40, swarm_size=200)

    # ----- Generate Test Data for Prediction -----
    n_test_points = 250  # Number of test points for evaluation.
    X_test = np.linspace(lb_x, ub_x, n_test_points).reshape(-1, 1)

    # ----- GP Prediction -----
    # Predict using the weighted GP model on the test inputs.
    # The predict function returns:
    #   - y_pred: The overall mean predictions for the test data.
    #   - y_cov: The covariance matrix of the predictions.
    #   - submodel_vals: Predicted values from each individual submodel.
    #   - submodel_cov: Covariance for the individual submodel predictions.
    y_pred, y_cov, submodel_vals, submodel_cov = gp.predict(
        X_test, params, calc_cov=True, return_submodels=True
    )
    # true_values = true_function(X_train, alg=np)
    # rmse = np.sqrt(np.mean((y_pred - true_values) ** 2))
    # print("RMSE between model and true function: {}".format(rmse))
    # ----- Plotting -----
    # Visualize the GP predictions and submodel contributions.
    # 'utils.make_submodel_plots' creates plots showing:
    #   - The overall GP prediction and confidence intervals.
    #   - The true function for comparison.
    #   - The individual submodel predictions and their covariance.
    utils.make_submodel_plots(
        X_train,
        y_train,
        X_test,
        y_pred,
        true_function,
        cov=y_cov,
        n_order=n_order,
        n_bases=n_bases,
        plot_submodels=True,
        submodel_vals=submodel_vals,
        submodel_cov=submodel_cov,
    )
