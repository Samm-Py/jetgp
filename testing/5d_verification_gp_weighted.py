import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pyoti.sparse as oti
import random
import itertools
import pyoti.core as coti
from oti_gp import oti_gp_weighted
import utils

# ---------------------------------------------------------------------
# DEMO: multi-dimensional example with separate length scales
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # np.random.seed(422)

    n_order = 3
    n_bases = 5
    lb_x = -1
    ub_x = 1
    sigma_n_true = 1e-8
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

    index = [[i] for i in range(0, X_train.shape[0])]
    y_train_data = []

    for val in index:
        X_train_pert = oti.array(X_train[val])

        for i in range(1, n_bases + 1):
            for j in range(X_train_pert.shape[0]):
                X_train_pert[j, i - 1] = X_train_pert[j, i - 1] + oti.e(
                    i, order=n_order
                )

        y_train_hc = oti.array([true_function(x) for x in X_train_pert])
        y_train_real = true_function(X_train, alg=np)

        y_train = y_train_real.reshape(-1, 1)
        for i in range(0, len(der_indices)):
            for j in range(0, len(der_indices[i])):
                y_train = np.vstack(
                    (y_train, y_train_hc.get_deriv(der_indices[i][j]))
                )

        y_train = y_train.flatten()

        noise = sigma_n_true * np.random.randn(len(y_train))
        y_train_noisy = y_train + noise
        y_train_data.append(y_train)

    sigma_f = 1.0
    sigma_n = sigma_n_true

    gp = oti_gp_weighted(
        X_train,
        y_train_data,
        n_order,
        n_bases,
        index,
        sigma_n=sigma_n,
        nugget=1e-6,
        kernel="SE",
        kernel_type="anisotropic",
    )

    params = gp.optimize_hyperparameters()

    X_test = np.random.uniform(-1, 1, size=(250, 5))  # 50 points in 5D

    f_mean = gp.predict(X_test, params, calc_cov=False)

    true_values = true_function(X_test, alg=np)

    rmse = np.sqrt(np.mean((f_mean - true_values) ** 2))
    print("RMSE Custom Weighted GP {}".format(rmse))
