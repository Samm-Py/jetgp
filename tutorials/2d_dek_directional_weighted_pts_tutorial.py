import numpy as np
import pyoti.sparse as oti  # For automatic differentiation using hyper-complex numbers
import itertools
from wddegp.wddegp import wddegp
# Utility functions (e.g., generating derivative indices, plotting submodels)
import utils

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
    lb_x = (
        -1
    )  # Lower bound for x₁ (using -π to π covers a full period)
    ub_x = 1  # Upper bound for x₁
    lb_y = -1  # Lower bound for x₂
    ub_y = 1  # Upper bound for x₂

    # Number of points along each axis (total training points = num_points^2)
    num_points = 5

    # Generate a grid of training points over the square region.
    x_vals = np.linspace(lb_x, ub_x, num_points)
    y_vals = np.linspace(lb_y, ub_y, num_points)
    X_train = np.array(list(itertools.product(x_vals, y_vals)))

    old_index = [
        [1, 2, 3],
        [5, 10, 15],
        [9, 14, 19],
        [21, 22, 23],
        [0],
        [4],
        [20],
        [24],
        [6, 7, 8, 11, 12, 13, 16, 17, 18]
    ]

    index = [
        [0, 1, 2],
        [3, 4, 5],
        [6, 7, 8],
        [9, 10, 11],
        [12],
        [13],
        [14],
        [15],
        [16, 17, 18, 19, 20, 21, 22, 23, 24]
    ]

    # 2) Flatten both
    old_flat = list(itertools.chain.from_iterable(old_index))
    new_flat = list(itertools.chain.from_iterable(index))

    # 3) Build reorder array
    reorder = np.zeros(25, dtype=int)
    for i in range(25):
        reorder[new_flat[i]] = old_flat[i]

    # 4) Shuffle X_train
    X_train = X_train[reorder]

# Now, X_train_shuffled[new_index[g][j]] == X_train[old_index[g][j]]

    # Generate indices for all derivatives up to n_order for a function with n_bases dimensions.
    # Note that the derivatives used for each submodel can be different. In this particular case
    # we assume that the derivative information used to construct each submodel is the same.
    der_indices = [
        [
            [
                [[1, 1]],
                [[2, 1]],
                [[3, 1]],
            ],
            [
                [[1, 1]],
                [[2, 1]],
                [[3, 1]],
            ],
            [
                [[1, 1]],
                [[2, 1]],
                [[3, 1]],
            ],
            [
                [[1, 1]],
                [[2, 1]],
                [[3, 1]],
            ],
            [
                [[1, 1]],
                [[2, 1]],
                [[3, 1]],
            ],
            [
                [[1, 1]],
                [[2, 1]],
                [[3, 1]],
            ],
            [
                [[1, 1]],
                [[2, 1]],
                [[3, 1]],
            ],
            [
                [[1, 1]],
                [[2, 1]],
                [[3, 1]],
            ],
            [
                [[1, 1]],
                [[1, 2]],
                [[2, 1]],
                [[2, 2]],
                [[3, 1]],
                [[3, 2]],
            ]
        ]
        for _ in range(len(index))
    ]
    lb_x = -1  # Lower bound for x₁ (using -π to π covers a full period)
    ub_x = 1  # Upper bound for x₁
    lb_y = -1  # Lower bound for x₂
    ub_y = 1  # Upper bound for x₂

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
        f = x1**2 * x2 + alg.cos(10 * x1) + alg.cos(10 * x2)
        return f

    # ----- Assemble Training Data for Submodels -----
    # The 'index' variable specifies which training points form each submodel.
    # For a global model, we require one submodel per training point.

    y_train_data = (
        []
    )  # List to hold training outputs (function values + derivatives) for each submodel
    rays_data = []
    thetas = [
        [-np.pi/2, 0, np.pi/2],
        [-np.pi, -np.pi/2, 0],
        [-np.pi, np.pi/2, 0],
        [-np.pi/2, -np.pi, np.pi/2],
        [-np.pi/2, 0, -np.pi/4],
        [np.pi/2, 0, np.pi/4],
        [np.pi/2, 0, np.pi/4],
        [-np.pi/2, -np.pi, -np.pi/4 - np.pi/2],
        [np.pi/2, -np.pi, np.pi/4 + np.pi/2],
        [np.pi/4, np.pi/2 + np.pi/4, np.pi/2],
    ]

    # Loop over each group of indices in 'index'
    for k, val in enumerate(index):
        # Extract the training points corresponding to the current index group.
        X_train_subset = X_train[val]

        # Convert these training points to an OTI array to enable automatic derivative tracking.
        X_train_pert = oti.array(X_train_subset)

        rays = np.zeros((n_bases, len(thetas[k])))

        # For each training point (indexed by thetas_list), compute the directional perturbations.

        for i, theta in enumerate(thetas[k]):
            rays[:, i] = [np.cos(theta), np.sin(theta)]
        rays_data.append(rays)
        nrays = rays.shape[1]
        # Create elementary perturbations for each ray using OTI.
        e = [oti.e(i + 1, order=n_order) for i in range(nrays)]
        # Compute the perturbation components by taking the dot product of rays and e.
        x_p, y_p = np.dot(rays, e)
        perts = [x_p, y_p]
        # Add the computed perturbations to the corresponding training point.
        for j in range(X_train.shape[1]):
            X_train_pert[:, j] = X_train_pert[:, j] + perts[j]

        # ----- Compute Hyper-complex Function Evaluations -----
        # Evaluate the true function on the perturbed training data to obtain a hyper-complex output
        # that includes both function values and directional derivative information.
        y_train_hc = true_function(X_train_pert, alg=oti)

        # ----- Truncation of Directional Combinations -----
        # Remove cross-terms in the hyper-complex representation corresponding to mixed directional derivatives,
        # which are either redundant or zero.
        for comb in itertools.combinations(range(1, nrays + 1), 2):
            y_train_hc = y_train_hc.truncate(comb)

        # ----- Assemble the Training Output -----
        # Extract the real part (function values) from the hyper-complex output.
        y_train_real = true_function(X_train, alg=np)
        y_train = y_train_real.reshape(-1, 1)
        # Append the derivative information as specified by der_indices.
        for i in range(len(der_indices[k])):
            for j in range(len(der_indices[k][i])):
                y_train = np.vstack(
                    (y_train, y_train_hc.get_deriv(der_indices[k][i][j]))
                )
        # Flatten the assembled data into a 1D array for training the GP.
        y_train = y_train.flatten()

        # Append the processed training output for this submodel.
        y_train_data.append(y_train)

    # ----- Weighted Gaussian Process Model Setup -----
    # Create a weighted GP model that handles submodel data.
    gp = wddegp(
        X_train,  # Original training inputs.
        # List of training outputs (function values and derivatives) for each submodel.
        y_train_data,
        n_order,  # Order of derivative information.
        n_bases,  # Dimensionality of the input space.
        index,  # Grouping of training points for submodel construction.
        der_indices,
        rays_data,
        normalize=False,
        kernel="SE",  # Use Squared Exponential (SE) kernel.
        # Anisotropic kernel (separate length scales per dimension).
        kernel_type="anisotropic",
    )

    # Optimize the GP hyperparameters (e.g., length scales, kernel variance).
    params = gp.optimize_hyperparameters(
        n_restart_optimizer=20, swarm_size=50)

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
