'''
--------------------------------------------------------------------------------
This script demonstrates a weighted, derivative-enhanced Gaussian Process (GP)
model using pyOTI-based automatic differentiation in 2D. This example shows how
submodels can be constructed from **multiple training points**. Each submodel may
use a different set of  derivatives depending on local behavior. Since
the weighted GP framework requires numerically ordered training indices, we remap
the desired point groupings to an ordered index before model construction.

Note:
- Submodels near the domain boundary use only first-order derivatives.
- Interior submodels use full derivatives up to the specified order.
- **Repetition of points across submodels is not currently supported**.
--------------------------------------------------------------------------------
'''

import numpy as np
import pyoti.sparse as oti
import itertools
from wdegp.wdegp import wdegp
import utils
import plotting_helper

if __name__ == "__main__":
    np.random.seed(0)

    n_order = 2
    n_bases = 2
    lb_x, ub_x = -1, 1
    lb_y, ub_y = -1, 1
    num_points = 4


    x_vals = np.linspace(lb_x, ub_x, num_points)
    y_vals = np.linspace(lb_y, ub_y, num_points)
    X_train = np.array(list(itertools.product(x_vals, y_vals)))

    old_index = [5, 6, 9, 10]   # Interior submodel
    new_index = [12, 13, 14, 15]S

    index = [[12,13,14,15]]
    index = [new_index]
    # Make a copy to swap
    X_new = X_train.copy()

    # Swap rows between old_index and new_index
    X_new[old_index], X_new[new_index] = X_train[new_index], X_train[old_index]

    X_train = X_new.copy()
    der_indices = [
        [[[1, 1]], [[2, 1]]],  # First-order derivatives: ∂f/∂x1, ∂f/∂x2
        [[[1, 2]], [[2, 2]]],  # Second-order: ∂²f/∂x1², ∂²f/∂x2²
    ]

    der_indices = [
        der_indices for _ in range(len(index))
    ]
    print(der_indices)

    def true_function(X, alg=oti):
        x1 = X[:, 0]
        x2 = X[:, 1]
        return (
            (4 - 2.1 * x1**2 + (x1**4) / 3.0) * x1**2
            + x1 * x2
            + (-4 + 4 * x2**2) * x2**2
        )

    y_train_data = []
    y_real = true_function(X_train, alg=np)
    for k, val in enumerate(index):
        X_sub = oti.array(X_train[val])
        for i in range(n_bases):
            for j in range(X_sub.shape[0]):
                X_sub[j, i] += oti.e(i + 1, order=n_order)

        y_hc = oti.array([true_function(x, alg=oti) for x in X_sub])
        y_sub = [y_real]
        for i in range(len(der_indices[k])):
            for j in range(len(der_indices[k][i])):
                y_sub.append(y_hc.get_deriv(
                    der_indices[k][i][j]).reshape(-1, 1))

        y_train_data.append(y_sub)
    print(y_train_data)
    gp = wdegp(
        X_train,
        y_train_data,
        n_order,
        n_bases,
        index,
        der_indices,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic",
    )

    params = gp.optimize_hyperparameters(n_restart_optimizer=15, swarm_size=50)

    N_grid = 25
    x_lin = np.linspace(lb_x, ub_x, N_grid)
    y_lin = np.linspace(lb_y, ub_y, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    y_pred, submodel_vals = gp.predict(
        X_test, params, calc_cov=False, return_submodels=True
    )

    # plotting_helper.make_submodel_plots(
    #     X_train,
    #     y_train_data,
    #     X_test,
    #     y_pred,
    #     true_function,
    #     X1_grid=X1_grid,
    #     X2_grid=X2_grid,
    #     n_order=n_order,
    #     n_bases=n_bases,
    #     plot_submodels=True,
    #     submodel_vals=submodel_vals,
    # )

    y_true = true_function(X_test, alg=np)
    nrmse = utils.nrmse(y_true, y_pred)
    print("NRMSE between model and true function: {}".format(nrmse))
import matplotlib.pyplot as plt

# Reshape predictions and ground truth back to grid
y_true_grid = y_true.reshape(N_grid, N_grid)
y_pred_grid = y_pred.reshape(N_grid, N_grid)
abs_error_grid = np.abs(y_true_grid - y_pred_grid)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# True function
c1 = axes[0].contourf(X1_grid, X2_grid, y_true_grid, levels=50, cmap="viridis")
fig.colorbar(c1, ax=axes[0])
axes[0].set_title("True Function")
axes[0].scatter(X_train[:, 0], X_train[:, 1], c="red", edgecolor="k", s=50, label="Training Data")

# GP Prediction
c2 = axes[1].contourf(X1_grid, X2_grid, y_pred_grid, levels=50, cmap="viridis")
fig.colorbar(c2, ax=axes[1])
axes[1].set_title("GP Prediction")
axes[1].scatter(X_train[:, 0], X_train[:, 1], c="red", edgecolor="k", s=50)

# Absolute Error
c3 = axes[2].contourf(X1_grid, X2_grid, abs_error_grid, levels=50, cmap="magma")
fig.colorbar(c3, ax=axes[2])
axes[2].set_title("Absolute Error")
axes[2].scatter(X_train[:, 0], X_train[:, 1], c="red", edgecolor="k", s=50)

# Axis labels
for ax in axes:
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

# Add legend only once (first axis already has it)
axes[0].legend()

plt.tight_layout()
plt.show()