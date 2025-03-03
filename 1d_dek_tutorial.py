import numpy as np
import matplotlib.pyplot as plt
import pyoti.sparse as oti
from oti_gp import oti_gp
import utils


if __name__ == "__main__":
    # This code is a tutorial demonstrating how to use a derivative-enhanced
    # Gaussian process for a 1D function. We generate sample points, compute
    # the function (and its derivatives) at those points, and then use them
    # to train our model.

    n_order = 4  # The order of derivative information used by the GP model
    n_bases = 1  # The dimensionality of the function (1D in this demo)
    lb_x = 0.2  # Lower bound for sampling
    ub_x = 6  # Upper bound for sampling

    # generates all derivative indices up to n_order
    # for a function of dimension n_bases
    der_indices = utils.gen_OTI_indices(n_bases, n_order)

    # If the use wants only to use for example 4th order derivatives in the
    # training process set der_indices = [[[[1, 4]]]] i.e
    # der_indices = [[[[1, 4]]]]
    # We use 5 points for this simple example. In a real case, choose
    # more or fewer points depending on the function's complexity.
    num_points = 5

    # Create a uniform mesh from lb_x to ub_x
    X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)

    # Convert to an OTI array so we can track derivatives automatically.
    # We then perturb the input in the directions needed to compute derivative info.
    X_train_pert = oti.array(X_train)
    for i in range(1, n_bases + 1):
        X_train_pert[:, i - 1] = X_train_pert[:, i - 1] + oti.e(
            i, order=n_order
        )

    # This function is arbitrary—any 1D function can be used. Here we mix
    # exponential, sine, and cosine terms to give the GP something interesting
    # to learn. We also add a linear term 0.2*x + 1.0 for a shift.
    def true_function(X, alg=oti):
        x = X[:, 0]
        f = alg.exp(-x) + alg.sin(x) + alg.cos(3 * x) + 0.2 * x + 1.0
        return f

    # Evaluate the function on the perturbed X
    # y_train_hc is hyper-complex (Q7) to track derivatives
    y_train_hc = true_function(X_train_pert)
    y_train_real = y_train_hc.real  # Extract just the real part

    # We stack the function values and derivatives into y_train
    # so the GP can learn from both.
    y_train = y_train_real
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            y_train = np.vstack(
                (y_train, y_train_hc.get_deriv(der_indices[i][j]))
            )

    # Flatten the data to a 1D array (Q9).
    y_train = y_train.flatten()

    # sigma_n_true is the true noise in the model outputs. Here it's zero
    # so there's no added noise (Q9).
    sigma_n_true = 0.0
    noise = sigma_n_true * np.random.randn(len(y_train))
    y_train_noisy = y_train + noise

    # ----- Create and Configure the Gaussian Process (GP) Model -----
    gp = oti_gp(
        X_train,  # Training inputs
        y_train,  # Training targets (function values + derivatives)
        n_order,  # Maximum derivative order used
        n_bases,  # Dimensionality of the input space
        der_indices,  # List of which derivatives to include
        sigma_n=sigma_n_true,  # Noise level in the data (assumed known/estimated)
        nugget=0.001,  # A small diagonal term added for numerical stability
        kernel="SE",  # Kernel choice: "SE" (squared exponential)
        kernel_type="anisotropic",  # Kernel is isotropic in the input space
    )

    # ----- Hyperparameter Optimization -----
    # The GP will adjust its kernel parameters (length-scale, etc.)
    # to best fit the data, typically via log-likelihood maximization.
    params = gp.optimize_hyperparameters()

    # ----- Generate Test Grid -----
    # We create a finer grid of points for visualizing the GP's predictions
    N_grid = 100
    X_test = np.linspace(lb_x, ub_x, N_grid).reshape(-1, 1)

    # ----- Predict with the GP Model -----
    # This returns both the mean prediction (y_pred) and the covariance (cov)
    y_pred, y_cov = gp.predict(
        X_test, params, calc_cov=True, return_deriv=True
    )
    utils.make_plots(
        X_train,
        y_train,
        X_test,
        y_pred,
        true_function,
        cov=y_cov,
        n_order=n_order,
        n_bases=n_bases,
        plot_derivative_surrogates=True,
        der_indices=der_indices,
    )
