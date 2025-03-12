import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pyoti.sparse as oti  # For automatic differentiation using hyper-complex numbers
import itertools
import utils
from oti_gp import (
    oti_gp_directional_weighted,
)  # Weighted GP model incorporating directional derivatives

# (This model uses both function and directional derivative information with weighting.)

# ---------------------------------------------------------------------
# DEMO: Multi-dimensional Example with Directional Derivatives (2D)
# Using a directional derivative enhanced weighted Gaussian Process model.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # ----- Define True Function -----
    # Define a synthetic function of two variables.
    # f(x_p, y_p) = cos(y_p) + cos(x_p) + 2*sin(2*y_p) + 2*sin(2*x_p)
    #             + sin(3*y_p) + sin(3*x_p) + 2*sin(4*y_p) + 2*sin(4*x_p)
    #             + sin(5*y_p) + sin(5*x_p)
    # This function has multiple frequency components and amplitudes.
    def true_function(X, alg=oti):
        f = (
            alg.cos(X[:, 0])
            + alg.cos(X[:, 1])
            + alg.cos(2 * X[:, 0])
            + alg.cos(2 * X[:, 1])
        )
        return f

    # Set random seed for reproducibility
    np.random.seed(0)

    # ----- Parameter Setup -----
    n_order = 4  # High-order directional derivatives will be computed (up to 15th order)
    n_bases = 2  # Problem dimension is 2 (x and y)
    n_dim = 2  # Number of dimensions for rays (should match n_bases)
    lb_x = -1  # Lower bound for x
    ub_x = 1  # Upper bound for x
    lb_y = -1  # Lower bound for y
    ub_y = 1  # Upper bound for y

    # Manually specify derivative indices.
    # Here, only derivatives in direction 1 are used, from 1st to 15th order.
    der_indices = [
        [
            [[1, 1]],
            [[1, 2]],
            [[1, 3]],
            [[1, 4]],
        ]
    ]

    # ----- Generate Training Data -----
    num_points = 1  # Single training point; we will use directional perturbations to obtain derivative info.
    # For this example, the training point is set at the origin (0,0).
    X_train = np.array([0, 0]).reshape(1, -1)
    # Convert training point to an OTI array for derivative tracking.
    X_train_pert = oti.array(X_train)

    # Define a list of angles (in radians) along which to compute directional derivatives.
    thetas = [
        0.0,
        np.pi / 2,
        np.pi,
        np.pi + np.pi / 2,
    ]
    n_rays = len(
        thetas
    )  # Number of rays equals the number of angles provided.
    # Initialize an array to store the directional unit vectors (rays).
    rays = np.zeros((n_dim, n_rays))
    y_train_data = (
        []
    )  # List to store training outputs for each directional ray.
    # 'index' is set so that each ray (angle) is treated as a separate submodel.
    index = [[i] for i in range(len(thetas))]

    # Loop over each index (i.e. each directional ray)
    for val in index:
        # Initialize a ray (column vector) for the current angle.
        ray = np.zeros((n_dim, 1))
        theta = thetas[val[0]]
        # Compute the unit vector for the given angle.
        ray[:, 0] = [np.cos(theta), np.sin(theta)]
        # Store the computed ray in the corresponding column of 'rays'.
        rays[:, val[0]] = ray[:, 0]
        nrays = rays.shape[1]
        # Create elementary perturbation for a single base (since we are perturbing one training point)
        e = [oti.e(i + 1, order=n_order) for i in range(1)]
        # Apply the perturbation along the computed ray.
        x_p, y_p = np.dot(ray, e)
        # Evaluate the true function on the perturbed point.
        X = oti.array([x_p, y_p]).T
        y_train_hc = true_function(X, alg=oti)
        # Truncate cross-terms in the hyper-complex representation.
        for comb in itertools.combinations(range(1, nrays + 1), 2):
            # This removes mixed derivative combinations that are redundant or zero.
            y_train_hc = y_train_hc.truncate(comb)
        # Extract the real part (function value) from the hyper-complex output.
        y_train_real = y_train_hc.real
        # Start with the function value.
        y_train = y_train_real
        # Append the derivative information as specified by der_indices.
        for i in range(0, len(der_indices)):
            for j in range(0, len(der_indices[i])):
                y_train = np.vstack(
                    (y_train, y_train_hc.get_deriv(der_indices[i][j]))
                )
        # Flatten the combined data into a 1D array.
        y_train = y_train.flatten()
        # (Optional) Add noise (here noise level is 0.0)
        sigma_n_true = 0.0
        noise = sigma_n_true * np.random.randn(len(y_train))
        y_train_noisy = y_train + noise
        # Append the training output for the current ray to the list.
        y_train_data.append(y_train)

    sigma_f = 1.0
    sigma_n = sigma_n_true  # Match the noise level

    # ----- GP Model Setup -----
    # Create the weighted GP model that incorporates directional derivative information.
    # The model uses:
    #  - X_train: The original training point.
    #  - y_train_data: A list of training outputs (function value + derivatives) for each ray.
    #  - index, der_indices, and rays: to specify how directional derivatives are computed and weighted.
    gp = oti_gp_directional_weighted(
        X_train,  # Training inputs (1 point)
        y_train_data,  # List of training outputs for each directional submodel
        n_order,  # Order of directional derivatives used
        n_bases,  # Problem dimension (2D)
        index,  # Grouping index for each directional ray
        der_indices,  # Specification of which directional derivatives to include
        rays,  # Array of directional perturbation vectors (rays)
        sigma_n=1e-6,  # Noise variance (set very low)
        nugget=1e-6,  # Nugget term for numerical stability
        kernel="SE",  # Squared Exponential kernel
        kernel_type="anisotropic",  # Allow separate length scales per dimension
    )

    # Optimize the GP hyperparameters (e.g., length scales, kernel variance)
    params = gp.optimize_hyperparameters()

    # ----- Generate Test Data for Prediction -----
    N_grid = 25  # Number of test points per axis
    x_lin = np.linspace(lb_x, ub_x, N_grid)
    y_lin = np.linspace(lb_y, ub_y, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    # ----- GP Prediction -----
    # Compute the GP mean prediction on the test grid.
    y_pred, submodel_vals = gp.predict(
        X_test, params, calc_cov=False, return_submodels=True
    )

    utils.make_weighted_directional_plots(
        X_train,
        y_train,
        X_test,
        y_pred,
        true_function,
        X1_grid=X1_grid,
        X2_grid=X2_grid,
        n_order=n_order,
        n_bases=n_bases,
        plot_directional_ders=True,
        submodel_vals=submodel_vals,
        thetas_list=thetas,
    )
