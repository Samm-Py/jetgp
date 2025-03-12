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

if __name__ == "__main__":
    np.random.seed(0)

    n_order = 1
    n_bases = 2
    lb_x = -2
    ub_x = 2
    lb_y = -1
    ub_y = 1
    der_indices = utils.gen_OTI_indices(n_bases, n_order)

    num_points = 5

    x_vals = np.linspace(lb_x, ub_x, num_points)
    y_vals = np.linspace(lb_y, ub_y, num_points)

    X_train = np.array(list(itertools.product(x_vals, y_vals)))
    X_train_pert = oti.array(X_train)

    for i in range(1, n_bases + 1):
        X_train_pert[:, i - 1] = X_train_pert[:, i - 1] + oti.e(
            i, order=n_order
        )

    def true_function(X, alg=oti):
        x1 = X[:, 0]
        x2 = X[:, 1]

        f = (
            (4 - 2.1 * x1**2 + (x1**4) / 3.0) * x1**2
            + x1 * x2
            + (-4 + 4 * x2**2) * x2**2
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
    sigma_n_true = 0.0000
    noise = sigma_n_true * np.random.randn(len(y_train))
    y_train_noisy = y_train + noise
    sigma_f = 1.0
    sigma_n = sigma_n_true

    gp = oti_gp(
        X_train,
        y_train,
        n_order,
        n_bases,
        sigma_n=0.0,
        nugget=0.001,
        kernel="RQ",
        kernel_type="anisotropic",
    )

    params = gp.optimize_hyperparameters()

    N_grid = 25
    x_lin = np.linspace(lb_x, ub_x, N_grid)
    y_lin = np.linspace(lb_y, ub_y, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    f_mean = gp.predict(X_test, params)
    f_mean_2d = f_mean.reshape(N_grid, N_grid)  # for plotting
    true_values = true_function(X_test, alg=np).reshape((N_grid, N_grid))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title(
        "Order {} Enhanced Gaussian Process Regression Prediction".format(
            n_order
        )
    )
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
    title_str = (
        r"$f(x_1, x_2) = \left(4 - 2.1\,x_1^2 + \frac{x_1^4}{3}\right)x_1^2 "
        r"+ x_1\,x_2 + \left(-4 + 4\,x_2^2\right)x_2^2$"
    )
    plt.title(title_str, fontsize=12)
    plt.contourf(X1_grid, X2_grid, true_values, levels=50, cmap="viridis")
    plt.colorbar()
    plt.scatter(X_train[:, 0], X_train[:, 1], c="white", edgecolors="k")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

    plt.tight_layout()

    rmse = np.sqrt(np.mean((f_mean_2d - true_values) ** 2))
