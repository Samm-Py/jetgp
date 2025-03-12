import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pyoti.sparse as oti
import random
import itertools
import pyoti.core as coti
from oti_gp import oti_gp
import utils

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

if __name__ == "__main__":
    np.random.seed(0)

    n_order = 0
    n_bases = 1
    lb_x = 0.2
    ub_x = 6

    der_indices = utils.gen_OTI_indices(n_bases, n_order)

    num_points = 9

    X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)
    X_train_pert = oti.array(X_train)

    for i in range(1, n_bases + 1):
        X_train_pert[:, i - 1] = X_train_pert[:, i - 1] + oti.e(
            i, order=n_order
        )

    def true_function(X, alg=oti):
        x = X[:, 0]

        f = alg.exp(-x) + alg.sin(x) + alg.cos(3 * x) + 0.2 * x + 1.0

        return f

    y_train_hc = true_function(X_train_pert)
    y_train_real = y_train_hc.real

    y_train = y_train_real
    for i in range(0, len(der_indices)):
        for j in range(0, len(der_indices[i])):
            y_train = np.vstack(
                (y_train, y_train_hc.get_deriv(der_indices[i][j]))
            )

    y_train = y_train.flatten()
    sigma_n_true = 0.0
    noise = sigma_n_true * np.random.randn(len(y_train))
    y_train_noisy = y_train + noise
    sigma_f = 1.0
    sigma_n = sigma_n_true

    gp = oti_gp(
        X_train,
        y_train,
        n_order,
        n_bases,
        der_indices,
        sigma_n=0.0,
        nugget=0.001,
        kernel="SE",
        kernel_type="isotropic",
    )

    params = gp.optimize_hyperparameters()

    N_grid = 100
    X_test = np.linspace(lb_x, ub_x, N_grid).reshape(-1, 1)

    y_pred, cov = gp.predict(X_test, params, calc_cov=True)

    sigma = np.sqrt(np.diag(cov))

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(
        X_test, true_function(X_test, alg=np), "r--", label="True function"
    )
    plt.scatter(X_train, y_train, color="k", label="Training points")
    plt.plot(X_test, y_pred, "b", label="GP Prediction")
    plt.fill_between(
        X_test.ravel(),
        y_pred - 1.96 * sigma,
        y_pred + 1.96 * sigma,
        color="b",
        alpha=0.2,
        label="95% Confidence Interval",
    )
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Gaussian Process Regression Fit")
    plt.show()

    # Define the Gaussian Process kernel
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0)

    # Fit the Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gp.fit(X_train, y_train)

    # Generate test data
    y_pred, sigma = gp.predict(X_test, return_std=True)

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(
        X_test, true_function(X_test, alg=np), "r--", label="True function"
    )
    plt.scatter(X_train, y_train, color="k", label="Training points")
    plt.plot(X_test, y_pred, "b", label="GP Prediction")
    plt.fill_between(
        X_test.ravel(),
        y_pred - 1.96 * sigma,
        y_pred + 1.96 * sigma,
        color="b",
        alpha=0.2,
        label="95% Confidence Interval",
    )
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Gaussian Process Regression Fit")
    plt.show()

    # rmse = np.sqrt(np.mean((f_mean - true_values) ** 2))
