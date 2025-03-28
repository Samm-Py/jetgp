import numpy as np
import pyoti.sparse as oti  # Library for automatic differentiation using hyper-complex numbers
from oti_gp import oti_gp  # Derivative-enhanced Gaussian Process class
import utils  # Utility functions, including one to generate derivative indices
import modules.sobol as sb
import modules.lhs as lhs
import pickle
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Not always needed but helps IDEs


# Zhis function will generate random samples for parameters.
# Zhese samples will be used for MC simulation
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


if __name__ == "__main__":
    # Set the random seed for reproducibility
    np.random.seed(1354)
    n_bases = 2
    lb_x = -5.12
    ub_x = 5.12
    lb_y = -5.12
    ub_y = 5.12

    num_points_test = 5000
    quasi = sb.create_sobol_samples(num_points_test, n_bases, 1).T

    lower_bounds = [lb_x, lb_y]
    upper_bounds = [ub_x, ub_y]

    X_test = scale_samples(quasi, lower_bounds, upper_bounds)
    rmse_data = []
    for order in range(2, 3):
        n_order = order

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
        #     [[[2, 1]]],
        #     [[[2, 2]]],
        #     [[[2, 3]]],
        # ]
        # We use 5 points for this simple example. In a real case, choose
        # more or fewer points depending on the function's complexity.

        # ----- Generate Training Data -----
        pts = [80]
        rmse_data_i = []
        for pt in pts:
            num_points = pt  # Number of points per axis for training data
            lb_x = -5.12
            ub_x = 5.12
            lb_y = -5.12
            ub_y = 5.12

            quasi = sb.create_sobol_samples(num_points, n_bases, 1).T

            lower_bounds = [lb_x, lb_y]
            upper_bounds = [ub_x, ub_y]

            X_train = scale_samples(quasi, lower_bounds, upper_bounds)

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

                f = (
                    10 * 2
                    + x1**2
                    - 10 * alg.cos(2 * np.pi * x1)
                    + x2**2
                    - 10 * alg.cos(2 * np.pi * x2)
                )

                return f

            # Evaluate the true function on the perturbed training data.
            # The output is hyper-complex, containing both function value and derivative information.
            y_train_hc = true_function(X_train_pert)
            y_train_real = y_train_hc.real

            y_train = y_train_real
            for i in range(0, len(der_indices)):
                for j in range(0, len(der_indices[i])):
                    y_train = np.vstack(
                        (y_train, y_train_hc.get_deriv(der_indices[i][j]))
                    )
            # ----- Noise Handling -----
            # sigma_n_true represents the known noise variance in the training outputs.
            # Here, it is set to zero (i.e., no noise is added) for simplicity.
            sigma_n_true = 0.0
            y_train = y_train.flatten()
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
                kernel="SineExp",  # Kernel choice: Rational Quadratic (RQ) kernel
                kernel_type="anisotropic",  # Anisotropic kernel to allow different length-scales per dimension
            )

            # Optimize the GP hyperparameters (e.g., length-scales, kernel variance) by maximizing the likelihood
            params = gp.optimize_hyperparameters(n_restart_optimizer=20)
            print(params)

            # ----- Generate Test Data for Prediction -----
            # Create a grid of test points over the same ranges as the training data.
            N_grid = 100  # Number of grid points per axis for test data
            # ----- Generate Test Data for Prediction -----
            # Create a grid of test points over the same ranges as the training data.
            x_lin = np.linspace(-5.12, 5.12, N_grid)
            y_lin = np.linspace(-5.12, 5.12, N_grid)
            X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
            # Combine the grid coordinates into a 2D array of test points.
            X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

            y_pred = gp.predict(
                X_test,
                params,
                calc_cov=False,
                return_deriv=False,
                n_restart_optimizer=50,
            )
            # ----- Plotting the Results -----
            # Create a figure with two subplots: one for the GP prediction and one for the true function.
            plt.figure(figsize=(6, 6))
            plt.rcParams.update({"font.size": 12})

            # Subplot (a): GP Prediction
            # plt.subplot(1, 2, 1)
            # plt.title(
            #     "Order {} Enhanced Gaussian Process\n True Function Prediction".format(
            #         n_order
            #     )
            # )
            # # Contour plot of the GP predicted mean
            plt.figure(123412)
            plt.contourf(
                X1_grid,
                X2_grid,
                y_pred.reshape(N_grid, N_grid),
                levels=25,
                cmap="viridis",
            )
            plt.colorbar()
            # Compute the min and max values of the grids
            x1_min, x1_max = np.min(X1_grid), np.max(X1_grid)
            x2_min, x2_max = np.min(X2_grid), np.max(X2_grid)

            # Create a mask for points within the grid bounds
            mask = (
                (X_train[:, 0] > x1_min)
                & (X_train[:, 0] < x1_max)
                & (X_train[:, 1] > x2_min)
                & (X_train[:, 1] < x2_max)
            )

            # Filtered points
            X_filtered = X_train[mask]
            # # Overlay the training points on the prediction plot
            # plt.scatter(
            #     X_filtered[:, 0],
            #     X_filtered[:, 1],
            #     c="white",
            #     edgecolors="k",
            #     label="Train pts",
            # )
            # # plt.xlabel("X1")
            # # plt.ylabel("X2")
            # plt.xticks([])  # no x ticks
            # plt.yticks([])  # no y ticks
            # # plt.legend()
            # plt.tight_layout(pad=2.0)

            # # ----- GP Prediction -----
            # # Predict the function values on the test data using the optimized GP model.
            # utils.make_plots(
            #     X_train,
            #     y_train,
            #     X_test,
            #     y_pred,
            #     true_function,
            #     X1_grid=X1_grid,
            #     X2_grid=X2_grid,
            #     n_order=n_order,
            #     n_bases=n_bases,
            #     plot_derivative_surrogates=False,
            #     der_indices=der_indices,
            # )
            true_values = true_function(X_test, alg=np)
            plt.figure(123414)
            plt.contourf(
                X1_grid,
                X2_grid,
                true_values.reshape(N_grid, N_grid),
                levels=25,
                cmap="viridis",
            )
            plt.colorbar()
            # # Overlay the training points on the true function plot
            plt.scatter(
                X_filtered[:, 0], X_filtered[:, 1], c="white", edgecolors="k"
            )
            plt.xticks([])  # no x ticks
            plt.yticks([])  # no y ticks
            # plt.legend()
            plt.tight_layout(pad=2.0)

            nrmse_vals = nrmse(y_pred.flatten(), true_values.flatten())
            print(
                "NRMSE between model and true function: {}".format(nrmse_vals)
            )
            rmse_data_i.append(nrmse_vals)
        rmse_data.append(rmse_data_i)
        print(rmse_data)
    # plt.semilogy(rmse_data)
    # plt.xlabel("Order of Derivative")
    # plt.ylabel("NRMSE (log scale)")
    # plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    # plt.minorticks_on()  # Ensure minor ticks are activated
    # plt.tight_layout()
    # plt.show()
    # with open("my_list.pkl", "wb") as f:
    #     pickle.dump(rmse_data, f)
