import numpy as np
import pyoti.sparse as oti  # Library for automatic differentiation using hyper-complex numbers
from oti_gp import oti_gp  # Derivative-enhanced Gaussian Process class
import utils  # Utility functions, including one to generate derivative indices
import modules.sobol as sb
import modules.lhs as lhs
import pickle


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
    n_bases = 4

    num_points_test = 5000
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

    rmse_data = []
    for order in range(1, 2):
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
        # ----- Generate Training Data -----
        if order == 0:
            pts = [10 * i + 2 for i in range(0, 10)]
        elif order == 1 or order == 2:
            pts = [5 * i + 2 for i in range(0, 7)]
        elif order == 2:
            pts = [4 * i + 2 for i in range(0, 7)]
        else:
            pts = [i + 1 for i in range(0, 9)]
        nrmse_data_i = []
        for pt in pts:
            num_points = pt  # Number of points per axis for training data
            lb_x = -2
            ub_x = 2
            lb_y = -1
            ub_y = 1
            quasi = sb.create_sobol_samples(num_points, n_bases, 1).T

            lower_bounds = [-2.048 for i in range(4)]
            upper_bounds = [2.048 for i in range(4)]

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
                kernel="SE",  # Kernel choice: Rational Quadratic (RQ) kernel
                kernel_type="anisotropic",  # Anisotropic kernel to allow different length-scales per dimension
            )

            # Optimize the GP hyperparameters (e.g., length-scales, kernel variance) by maximizing the likelihood
            params = gp.optimize_hyperparameters(
                n_restart_optimizer=25, swarm_size=25
            )
            print(params)

            num_points_test = 5000
            quasi = sb.create_sobol_samples(num_points_test, n_bases, 1).T

            lower_bounds = [-2.048 for i in range(4)]
            upper_bounds = [2.048 for i in range(4)]

            X_test = scale_samples(quasi, lower_bounds, upper_bounds)

            y_pred = gp.predict(
                X_test,
                params,
                calc_cov=False,
                return_deriv=False,
                n_restart_optimizer=50,
            )
            true_values = true_function(X_test, alg=np)
            nrmse_vals = nrmse(y_pred.flatten(), true_values.flatten())
            print(
                "NRMSE between model and true function: {}".format(nrmse_vals)
            )
            nrmse_data_i.append(nrmse_vals)
        filename = "NRMSE_camel_order_{}.pkl".format(order)
        with open(filename, "wb") as f:
            pickle.dump(nrmse_data_i, f)

order = 1  # or whichever order you saved
filename = "NRMSE_camel_order_{}.pkl".format(order)

with open(filename, "rb") as f:
    nrmse_vals = pickle.load(f)

print(nrmse_vals)
