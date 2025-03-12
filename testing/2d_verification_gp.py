import numpy as np
import matplotlib.pyplot as plt
import pyoti.sparse as oti
import itertools
from oti_gp import oti_gp
import utils
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel


if __name__ == "__main__":
    np.random.seed(0)

    n_order = 0
    n_bases = 2
    lb_x = -np.pi
    ub_x = np.pi
    lb_y = -np.pi
    ub_y = np.pi
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

        f = alg.sin(x1) + alg.cos(x2)

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
        nugget=0.000,
        kernel="SE",
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
    plt.contourf(X1_grid, X2_grid, f_mean_2d, levels=10, cmap="viridis")
    # plt.colorbar()
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
    plt.contourf(X1_grid, X2_grid, true_values, levels=10, cmap="viridis")
    plt.colorbar()
    plt.scatter(X_train[:, 0], X_train[:, 1], c="white", edgecolors="k")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

    plt.tight_layout()

    # Define the Gaussian Process kernel (anisotropic RBF + noise)
    kernel = RBF(length_scale=[1.0, 1.0], length_scale_bounds=(1e-05, 1000.0))

    # Fit the Gaussian Process model
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20)
    gp.fit(X_train, y_train)

    # Predict using the GP model
    y_pred, sigma = gp.predict(X_test, return_std=True)
    y_pred = y_pred.reshape(X1_grid.shape)
    sigma = sigma.reshape(X1_grid.shape)

    # Plot the true function
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    # True function
    ax[1].contourf(
        X1_grid,
        X2_grid,
        true_function(X_test, alg=np).reshape(X1_grid.shape),
        cmap="viridis",
    )
    ax[1].scatter(
        X_train[:, 0],
        X_train[:, 1],
        color="white",
        edgecolors="k",
        label="Training points",
    )
    ax[1].set_title("$f(x_1, x_2) = \sin(x_1) + cos(x_2)$")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("y")
    ax[1].legend()

    # GP Prediction
    contour = ax[0].contourf(X1_grid, X2_grid, y_pred, cmap="viridis")
    ax[0].scatter(
        X_train[:, 0],
        X_train[:, 1],
        color="white",
        edgecolors="k",
        label="Training points",
    )
    ax[0].set_title(r"Gaussian Process SkLearn")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    fig.colorbar(contour, ax=ax[1])
    plt.tight_layout()

    plt.show()
