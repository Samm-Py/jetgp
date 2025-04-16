
"""
--------------------------------------------------------------------------------
This script demonstrates a weighted, derivative-enhanced Gaussian Process (GP)
model using pyOTI-based automatic differentiation. Here, m submodels are built
from n training points (with m < n). Each submodel aggregates multiple points,
and all submodels use the same full set of derivatives up to a 
specified order. Submodels are combined using a weighted GP framework.

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
    # Define the true function
    def true_function(X, alg=oti):
        x1 = X[:, 0]
        return alg.sin(10 * np.pi * x1) / (2 * x1) + (x1 - 1) ** 4

    # GP configuration
    n_order = 3
    n_bases = 1
    lb_x, ub_x = 0.5, 2.5
    num_points = 10

    # Create training inputs
    X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)

    # Define 2 submodels using grouped training points
    index = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]

    # All submodels use the same full derivative structure
    base_der_indices = utils.gen_OTI_indices(n_bases, n_order)
    der_indices = [base_der_indices for _ in index]

    # Assemble training data for each submodel
    y_train_data = []
    y_real = true_function(X_train, alg=np)
    for k, val in enumerate(index):
        X_sub = oti.array(X_train[val])
        for i in range(n_bases):
            for j in range(X_sub.shape[0]):
                X_sub[j, i] += oti.e(i + 1, order=n_order)

        y_hc = oti.array([true_function(x, alg=oti) for x in X_sub])

        y_sub = [y_real]
        for i in range(len(base_der_indices)):
            for j in range(len(base_der_indices[i])):
                deriv = y_hc.get_deriv(base_der_indices[i][j]).reshape(-1, 1)
                y_sub.append(deriv)

        y_train_data.append(y_sub)

    # Build the weighted GP model
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

    # Optimize hyperparameters
    params = gp.optimize_hyperparameters(
        n_restart_optimizer=25,
        swarm_size=200
    )

    # Generate test inputs and predict
    X_test = np.linspace(lb_x, ub_x, 250).reshape(-1, 1)
    y_pred, y_cov, submodel_vals, submodel_cov = gp.predict(
        X_test,
        params,
        calc_cov=True,
        return_submodels=True
    )

    # Plot results
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
