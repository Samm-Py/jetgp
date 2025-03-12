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

    n_order = 2
    n_bases = 5
    lb_x = -1
    ub_x = 1

    der_indices = utils.gen_OTI_indices(n_bases, n_order)

    # Define the limits of the 5D box
    bounds = np.array(
        [
            [-1, 1],  # x1 range
            [-1, 1],  # x2 range
            [-1, 1],  # x3 range
            [-1, 1],  # x4 range
            [-1, 1],
        ]
    )  # x5 range

    # Define the number of points per dimension
    num_points_per_dim = [
        2,
        2,
        2,
        2,
        2,
    ]  # Different resolution for each dimension

    # Create grid points for each dimension
    grid_axes = [
        np.linspace(bounds[i, 0], bounds[i, 1], num_points_per_dim[i])
        for i in range(5)
    ]

    # Create the full grid using itertools.product

    X_train = np.array(list(itertools.product(*grid_axes)))

    X_train_pert = oti.array(X_train)

    for i in range(1, n_bases + 1):
        X_train_pert[:, i - 1] = X_train_pert[:, i - 1] + oti.e(
            i, order=n_order
        )

    def true_function(X, alg=oti):
        x1, x2, x3, x4, x5 = X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4]

        f = (
            alg.sin(0.25 * np.pi * x1)
            + 2 * alg.cos(0.25 * np.pi * x2)
            + 3 * alg.sin(0.25 * np.pi * x3)
            + 2 * alg.cos(0.25 * np.pi * x4)
            + alg.sin(0.25 * np.pi * x5)
        )

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
        sigma_n=0.001,
        nugget=0.001,
        kernel="SE",
        kernel_type="anisotropic",
    )

    params = gp.optimize_hyperparameters()

    N_grid = 100

    X_test = np.random.uniform(-1, 1, size=(5000, 5))  # 50 points in 5D

    f_mean = gp.predict(X_test, params, calc_cov=False)

    # Define the Gaussian Process kernel
    kernel = ConstantKernel(1.0) * RBF(length_scale=[1.0, 1.0, 1.0, 1.0, 1.0])

    # Fit the Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=50)
    gp.fit(X_train, y_train[: X_train.shape[0]])

    # Generate test data
    y_pred = gp.predict(X_test, return_std=False)

    true_values = true_function(X_test, alg=np)

    rmse = np.sqrt(np.mean((f_mean - true_values) ** 2))
    print("RMSE Custom GP {}".format(rmse))

    rmse = np.sqrt(np.mean((y_pred - true_values) ** 2))
    print("RMSE Sklearn GP {}".format(rmse))
