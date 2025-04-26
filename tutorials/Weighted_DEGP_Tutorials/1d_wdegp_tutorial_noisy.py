"""
--------------------------------------------------------------------------------
This script demonstrates a weighted, derivative-enhanced Gaussian Process (GP)
model using pyOTI-based automatic differentiation. Each training point forms its
own submodel, and all submodels use the same full set of directional derivatives
up to a specified order. The final prediction is obtained by combining these
submodels through a weighted GP framework.
--------------------------------------------------------------------------------
"""

import numpy as np
import pyoti.sparse as oti
from wdegp.wdegp import wdegp
import utils
import plotting_helper

if __name__ == "__main__":
    # Define the true function
    # Define true function
    def true_function(X, alg=oti):
        x = X[:, 0]
        return x * alg.sin(x)

    lb_x = 0          # Lower bound of input domain
    ub_x = 10         # Upper bound of input domain
    # GP configuration
    n_order = 1
    n_bases = 1
    # Generate training input points from a dense candidate set
    num_points = 6
    X = np.linspace(lb_x, ub_x, 1000).reshape(-1, 1)
    rng = np.random.RandomState(1)
    training_indices = rng.choice(
        np.arange(X.shape[0]), size=num_points, replace=False)
    # X_train = np.sort(X[training_indices], axis=0)
    X_train = X[training_indices]
    # X_train[0] = 1
    # index = [[i] for i in range(num_points)]
    index = [[0, 1, 2], [3, 4, 5]]

    # Each submodel uses the same full derivative index structure
    base_der_indices = utils.gen_OTI_indices(n_bases, n_order)
    der_indices = [base_der_indices for _ in index]

    # Construct training data for each submodel
    y_train_data = []
    y_train_real = true_function(X_train, alg=np)
    arr = np.zeros((len(base_der_indices)+1)*num_points)
    arr[:] = .5
    noise_std = np.diag(arr)
    y_train_real_noisy = y_train_real.copy()
    for i in range(0, len(y_train_real)):
        y_train_real_noisy[i] = y_train_real_noisy[i] + \
            rng.normal(loc=0.0, scale=arr[i], size=1)
    for k, val in enumerate(index):
        X_train_pert = oti.array(X_train[val])
        for i in range(n_bases):
            for j in range(X_train_pert.shape[0]):
                X_train_pert[j, i] += oti.e(i + 1, order=n_order)

        y_train_hc = oti.array([true_function(x, alg=oti)
                               for x in X_train_pert])
        y_train = [y_train_real_noisy]
        for i in range(len(base_der_indices)):
            for j in range(len(base_der_indices[i])):
                deriv = y_train_hc.get_deriv(
                    base_der_indices[i][j]).reshape(-1, 1)
                deriv_noisy = deriv.copy()
                for k in range(0, len(deriv_noisy)):
                    deriv_noisy[k] = deriv_noisy[k] + \
                        rng.normal(loc=0.0, scale=0.1, size=1)
                y_train.append(deriv_noisy)

        y_train_data.append(y_train)

    # Create weighted GP model
    gp = wdegp(
        X_train,
        y_train_data,
        n_order,
        n_bases,
        index,
        der_indices,
        normalize=True,
        sigma_data=noise_std,      # Informs the model about expected noise level
        kernel="SE",
        kernel_type="anisotropic",
    )

    # Hyperparameter optimization
    params = gp.optimize_hyperparameters(
        n_restart_optimizer=25,
        swarm_size=100
    )

    # Generate test inputs and make predictions
    X_test = np.linspace(lb_x, ub_x, 1000).reshape(-1, 1)
    y_pred, y_cov,  submodel_vals, submodel_cov = gp.predict(
        X_test,
        params,
        calc_cov=True,
        return_submodels=True
    )

    # Plot predictions and submodel contributions
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
    )

    y_true = true_function(X_test, alg=np)
    nrmse = utils.nrmse(y_true, y_pred)

    print("NRMSE between model and true function: {}".format(nrmse))
