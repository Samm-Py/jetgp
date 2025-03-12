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

    # Let's make a synthetic problem in D=2.
    # We'll define f(x1, x2) = sin(x1) * cos(x2) or something like that.
    # We'll sample some points from a square region and add noise.
    np.random.seed(0)
    # 1) Generate training data

    n_order = 1
    n_bases = 2
    lb_x = -1
    ub_x = 1
    lb_y = -1
    ub_y = 1
    der_indices = utils.gen_OTI_indices(n_bases, n_order)

    num_points = 3  # 4x4 grid to get 16 points
    x_vals = np.linspace(lb_x, ub_x, num_points)
    y_vals = np.linspace(lb_y, ub_y, num_points)
    X_train = np.array(list(itertools.product(x_vals, y_vals)))
    y_train_data = []
    sigma_n_true = 1e-8

    index = [[i] for i in range(len(X_train))]

    def true_function(X, alg=oti):
        x1 = X[:, 0]
        x2 = X[:, 1]

        f = alg.sin(np.pi * x1) + alg.cos(np.pi * x2)

        return f

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

    gp = oti_gp_weighted(
        X_train,
        y_train_data,
        n_order,
        n_bases,
        index,
        sigma_n=1e-6,
        nugget=1e-6,
        kernel="SE",
        kernel_type="anisotropic",
    )

    params = gp.optimize_hyperparameters()

    N_grid = 25
    x_lin = np.linspace(lb_x, ub_x, N_grid)
    y_lin = np.linspace(lb_y, ub_y, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    f_mean = gp.predict(X_test, params, calc_cov=False)


plt.figure(52, figsize=(12, 8))


f_mean_2d = f_mean.reshape(N_grid, N_grid)  # for plotting


# 5) Plot results
#    We'll do a contour plot of the predicted mean vs. the true function.
true_values = true_function(X_test, alg=np).reshape((N_grid, N_grid))

# # (a) Predicted mean
plt.subplot(1, 2, 1)
plt.title("Order {} Enhanced Weighted Gaussian Process".format(n_order))
plt.contourf(X1_grid, X2_grid, f_mean_2d, levels=50, cmap="viridis")
plt.colorbar()
plt.scatter(
    X_train[:, 0],
    X_train[:, 1],
    c="white",
    edgecolors="k",
    label="Train pts",
)
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.tight_layout()

# (b) True function
plt.subplot(1, 2, 2)
title_str = r"$f(x_1, x_2) = \sin(x_1) + cos(x_2)$"
plt.title(title_str, fontsize=12)
plt.contourf(X1_grid, X2_grid, true_values, levels=50, cmap="viridis")
plt.colorbar()
plt.scatter(X_train[:, 0], X_train[:, 1], c="white", edgecolors="k")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()

plt.tight_layout()
