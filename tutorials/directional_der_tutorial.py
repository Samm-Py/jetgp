import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pyoti.sparse as oti  # For automatic differentiation using hyper-complex numbers
import itertools
from oti_gp import (
    oti_gp_directional,
)  # Directional derivative enhanced GP model
import utils  # Utility functions (e.g., to generate derivative indices, plotting submodels)

# ---------------------------------------------------------------------
# DEMO: Multi-dimensional Example with Directional Derivatives (2D)
# Using a directional derivative enhanced Gaussian Process model.
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(0)

    # ----- Parameter Setup -----
    n_order = 4  # Order of directional derivatives to be calculated (up to 4th order)
    n_bases = 2  # Dimension of the problem (2D: x₁ and x₂)
    lb_x = -np.pi  # Lower bound for x₁ (using -π to π covers a full period)
    ub_x = np.pi  # Upper bound for x₁
    lb_y = -np.pi  # Lower bound for x₂
    ub_y = np.pi  # Upper bound for x₂

    # der_indices describes the directional derivative information:
    # For example:
    #   [[1,1]] → first-order derivative in direction 1,
    #   [[1,2]] → second-order derivative in direction 1,
    #   [[2,1]] → first-order derivative in direction 2, etc.
    der_indices = [
        [
            [[1, 1]],
            [[1, 2]],
            [[1, 3]],
            [[1, 4]],
            [[2, 1]],
            [[2, 2]],
            [[2, 3]],
            [[2, 4]],
            [[3, 1]],
            [[3, 2]],
            [[3, 3]],
            [[3, 4]],
        ]
    ]

    # ----- Generate Training Data -----
    num_points = (
        3  # Number of points along each axis (total training points = 3x3 = 9)
    )
    x_vals = np.linspace(-np.pi, np.pi, num_points)
    y_vals = np.linspace(-np.pi, np.pi, num_points)
    # Create a grid of 2D training points using the Cartesian product
    X_train = np.array(list(itertools.product(x_vals, y_vals)))
    # Convert the training data to an OTI array to enable automatic differentiation
    X_train_pert = oti.array(X_train)

    # ----- Define True Function -----
    # Define the synthetic true function f(x₁,x₂) as:
    # f(x₁,x₂) = cos(x₁) + cos(x₂) + cos(2*x₁) + cos(2*x₂)
    # This function is chosen due to its periodic behavior and multiple frequency components.
    def true_function(X, alg=oti):
        f = (
            alg.cos(X[:, 0])
            + alg.cos(X[:, 1])
            + alg.cos(2 * X[:, 0])
            + alg.cos(2 * X[:, 1])
        )
        return f

    # ----- Directional Derivative Setup -----
    nrays = 3  # Number of directional perturbations (rays) to be used per training point.
    ndim = 2  # Dimensionality of the problem (2D: x₁ and x₂)
    order = n_order  # Set the derivative order for directional perturbations

    # Initialize an array to store the directional rays (unit vectors)

    # thetas_list provides a list of angles (in radians) for each training point.
    thetas = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    rays = np.zeros((ndim, len(thetas)))

    # For each training point (indexed by thetas_list), compute the directional perturbations.

    for i, theta in enumerate(thetas):
        rays[:, i] = [np.cos(theta), np.sin(theta)]
    nrays = rays.shape[1]
    # Create elementary perturbations for each ray using OTI.
    e = [oti.e(i + 1, order=order) for i in range(nrays)]
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
    y_train_real = y_train_hc.real
    y_train = y_train_real
    # Append the derivative information as specified by der_indices.
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            y_train = np.vstack(
                (y_train, y_train_hc.get_deriv(der_indices[i][j]))
            )
    # Flatten the assembled data into a 1D array for training the GP.
    y_train = y_train.flatten()

    # Optionally add noise (here sigma_n_true is set to 0.0)
    sigma_n_true = 0.0000
    noise = sigma_n_true * np.random.randn(len(y_train))
    y_train_noisy = y_train + noise

    # ----- GP Model Setup -----
    # Set fixed hyperparameters.
    sigma_f = 1.0
    sigma_n = sigma_n_true  # Matching the noise level used above

    # Create the directional derivative enhanced GP model.
    gp = oti_gp_directional(
        X_train,  # Training inputs
        y_train,  # Flattened training outputs (function values + directional derivatives)
        n_order,  # Order of directional derivatives used
        n_bases,  # Input dimension (2D)
        der_indices,  # Specification of directional derivatives to include
        rays,  # Array of directional perturbation vectors used to compute derivatives
        sigma_n=1e-6,  # Noise variance (set very low)
        nugget=1e-6,  # Nugget for numerical stability
        kernel="SE",  # Squared Exponential kernel
        kernel_type="anisotropic",  # Allow separate length scales per dimension
    )

    # Optimize the GP hyperparameters (e.g., length scales, kernel variance)
    params = gp.optimize_hyperparameters()

    # ----- Generate Test Data for Prediction -----
    N_grid = 40  # Number of test points per axis
    x_lin = np.linspace(lb_x - 1, ub_x + 1, N_grid)
    y_lin = np.linspace(lb_y - 1, ub_y + 1, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    # ----- GP Prediction -----
    # Predict the function on the test grid using the optimized GP model.
    # Here, we only compute the mean prediction (calc_cov=False).
    y_pred = gp.predict(X_test, params, calc_cov=False, return_deriv=True)

    # ----- Plotting with Utility Function -----
    # Instead of manually plotting, call the utility function to generate directional derivative plots.
    utils.make_directional_plots(
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
        der_indices=der_indices,
        thetas_list=thetas,
    )
