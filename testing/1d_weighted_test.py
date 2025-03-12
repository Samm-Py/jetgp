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
    np.random.seed(0)

    n_order = 2
    n_bases = 1
    lb_x = 0.2
    ub_x = 6.0
    der_indices = utils.gen_OTI_indices(n_bases, n_order)
    num_points = 5

    x_vals = np.array([0.3, 1.65, 3.1, 4.55, 5.5]).reshape(-1, 1)
    index = [[0], [1], [2], [3], [4]]
    y_train_data = []
    sigma_n_true = 1e-6

    for val in index:
        X_train = x_vals
        X_train_pert = oti.array(X_train[val])

        for i in range(1, n_bases + 1):
            for j in range(X_train_pert.shape[0]):
                X_train_pert[j, i - 1] = X_train_pert[j, i - 1] + oti.e(
                    i, order=n_order
                )

        def true_function(X):
            x = X[:, 0]
            return np.exp(-x) + np.sin(x) + np.cos(3 * x) + 0.2 * x + 1.0

        def true_function_hc(x):
            return (
                oti.exp(-x[0, 0])
                + oti.sin(x[0, 0])
                + oti.cos(3 * x[0, 0])
                + 0.2 * x
                + 1.0
            )

        y_train_hc = oti.array([true_function_hc(x) for x in X_train_pert])
        y_train_real = true_function(X_train)

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
        sigma_n=1e-6,
        nugget=1e-6,
        kernel="SE",
        kernel_type="anisotropic",
    )

    params = gp.optimize_hyperparameters()

    n_test_points = 250
    X_test = np.linspace(lb_x, ub_x, n_test_points).reshape(-1, 1)

    f_mean, f_cov = gp.predict(X_test, params, calc_cov=True)


plt.figure(52, figsize=(12, 8))
# plt.plot(
#     X_test,
#     true_values,
#     ls="--",
#     color="tab:red",
#     label="True Function",
# )
plt.plot(X_test, f_mean, label="WDEK \n order {} derivative".format(n_order))
plt.fill_between(
    X_test.ravel(),
    f_mean - 2 * np.sqrt(f_cov),
    f_mean + 2 * np.sqrt(f_cov),
    alpha=0.2,
    label="95% confidence interval",
)

# plt.plot(X_test, f_mean, label="Mean Predictions")
plt.scatter(
    X_train[:, 0],
    true_function(X_train),
    c="white",
    edgecolors="k",
    label="Train pts",
)

true_values = true_function(X_test)
plt.plot(
    X_test,
    true_values,
    ls="--",
    color="tab:red",
    label="True Function",
)
plt.scatter(
    X_train[:, 0],
    true_function(X_train),
    c="white",
    edgecolors="k",
)
plt.legend(bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
