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

    n_order = 3
    n_bases = 2
    lb_x, ub_x = -1, 1
    lb_y, ub_y = -1, 1
    num_points = 4

    x_vals = np.linspace(lb_x, ub_x, num_points)
    y_vals = np.linspace(lb_y, ub_y, num_points)
    X_train = np.array(list(itertools.product(x_vals, y_vals)))

    old_index = [
        [1, 2],
        [4, 8],
        [7, 11],
        [13, 14],
        [0],
        [3],
        [12],
        [15],
        [5, 6, 9, 10]  # Interior submodel
    ]

    index = [
        [0, 1],
        [2, 3],
        [4, 5],
        [6, 7],
        [8],
        [9],
        [10],
        [11],
        [12, 13, 14, 15]
    ]

    old_flat = list(itertools.chain.from_iterable(old_index))
    new_flat = list(itertools.chain.from_iterable(index))
    reorder = np.zeros(num_points**2, dtype=int)
    for i in range(num_points**2):
        reorder[new_flat[i]] = old_flat[i]

    X_train = X_train[reorder]

    der_indices = [
        utils.gen_OTI_indices(n_bases, 1) for _ in range(len(index) - 1)
    ]

    der_indices.append(utils.gen_OTI_indices(n_bases, n_order))

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

    params = gp.optimize_hyperparameters(n_restart_optimizer=10, swarm_size=25)

    N_grid = 25
    x_lin = np.linspace(lb_x, ub_x, N_grid)
    y_lin = np.linspace(lb_y, ub_y, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    y_pred, submodel_vals = gp.predict(
        X_test, params, calc_cov=False, return_submodels=True
    )

    plotting_helper.make_submodel_plots(
        X_train,
        y_train_data,
        X_test,
        y_pred,
        true_function,
        X1_grid=X1_grid,
        X2_grid=X2_grid,
        n_order=n_order,
        n_bases=n_bases,
        plot_submodels=True,
        submodel_vals=submodel_vals,
    )

    y_true = true_function(X_test, alg=np)
    nrmse = utils.nrmse(y_true, y_pred)
    print("NRMSE between model and true function: {}".format(nrmse))
