"""
--------------------------------------------------------------------------------
This script demonstrates a weighted, derivative-enhanced Gaussian Process (GP)
model using pyOTI-based automatic differentiation. Here, m submodels are built
from n training points (with m < n). Each submodel aggregates multiple points,
and each submodel can use a different subset of derivatives depending on local
function behavior. Submodels are combined using a weighted GP framework.

Note:
- In the current implementation, **repetition of points across submodels is not supported**.
  Each training point should be assigned to one and only one submodel.
--------------------------------------------------------------------------------
"""

import numpy as np
import pyoti.sparse as oti
from wdegp.wdegp import wdegp
import utils
import plotting_helper

if __name__ == "__main__":
    def true_function(X, alg=oti):
        x1 = X[:, 0]
        return alg.sin(5 * np.pi * x1) * alg.exp(-3 * x1) + x1

    n_order = 3
    n_bases = 1
    lb_x, ub_x = 0.0, 2.0
    num_points = 8

    X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)
    index = [[0, 1, 2, 3], [4, 5, 6, 7]]

    # Different derivative structures for each submodel
    der_indices = [
        utils.gen_OTI_indices(n_bases, n_order),      # More nonlinear region
        utils.gen_OTI_indices(n_bases, 2),            # More linear region
    ]

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
                deriv = y_hc.get_deriv(der_indices[k][i][j]).reshape(-1, 1)
                y_sub.append(deriv)

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

    params = gp.optimize_hyperparameters(
        n_restart_optimizer=25,
        swarm_size=200
    )

    X_test = np.linspace(lb_x, ub_x, 250).reshape(-1, 1)
    y_pred, y_cov, submodel_vals, submodel_cov = gp.predict(
        X_test,
        params,
        calc_cov=True,
        return_submodels=True
    )

    plotting_helper.make_submodel_plots(
        X_train,
        y_train_data,
        X_test,
        y_pred,
        true_function,
        cov=y_cov,
        n_order=n_order,
        n_bases=n_bases,
        plot_submodels=True,
        submodel_vals=submodel_vals,
        submodel_cov=submodel_cov,
    )

    y_true = true_function(X_test, alg=np)
    nrmse = utils.nrmse(y_true, y_pred)

    print("NRMSE between model and true function: {}".format(nrmse))
