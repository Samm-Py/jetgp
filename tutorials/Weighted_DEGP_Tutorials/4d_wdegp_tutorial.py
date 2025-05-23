'''
--------------------------------------------------------------------------------
This script demonstrates a weighted, derivative-enhanced Gaussian Process (GP)
model using pyOTI-based automatic differentiation in 4D. Five submodels are
constructed, each containing training points that are spatially close to one
another in the input space. Each submodel uses only main derivatives.
Submodels are determined via KMeans clustering and combined using a weighted GP
framework for final prediction.
--------------------------------------------------------------------------------
'''

import numpy as np
import pyoti.sparse as oti
from wdegp.wdegp import wdegp as oti_gp_weighted
import utils
import sys
sys.path.append("../../modules/")
import sobol as sb
from sklearn.cluster import KMeans
import itertools
import plotting_helper

if __name__ == "__main__":
    np.random.seed(1354)

    n_bases = 4
    n_order = 2
    num_points_train = 26
    num_points_test = 5000

    lower_bounds = [-2.048] * n_bases
    upper_bounds = [2.048] * n_bases

    # Generate training and testing samples using Sobol sequences
    quasi = sb.create_sobol_samples(num_points_train, n_bases, 1).T
    X_train = utils.scale_samples(quasi, lower_bounds, upper_bounds)

    quasi_test = sb.create_sobol_samples(num_points_test, n_bases, 1).T
    X_test = utils.scale_samples(quasi_test, lower_bounds, upper_bounds)

    # Cluster training points into 5 submodels based on spatial proximity
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X_train)
    labels = kmeans.labels_
    index_unsorted = [[] for _ in range(n_clusters)]
    for i, label in enumerate(labels):
        index_unsorted[label].append(i)

    # Remap the training data to be numerically ordered by submodel
    flat_sorted_indices = sorted(itertools.chain.from_iterable(index_unsorted))
    X_train = X_train[flat_sorted_indices]
    old_to_new = {old_idx: new_idx for new_idx,
                  old_idx in enumerate(flat_sorted_indices)}
    index = [[old_to_new[i] for i in group] for group in index_unsorted]

    # Rebuild numerically ordered index
    index_flat = list(itertools.chain.from_iterable(index))
    index = []
    start = 0
    for group in index_unsorted:
        end = start + len(group)
        index.append(list(range(start, end)))
        start = end

    # Use only main derivative terms for all submodels
    der_indices = [
        [[[[1, 1]], [[2, 1]], [[3, 1]], [[4, 1]]],   # ∂f/∂x₁, ∂f/∂x₂, ∂f/∂x₃
         # ∂²f/∂x₁², ∂²f/∂x₂², ∂²f/∂x₃²
         [[[1, 2]], [[2, 2]], [[3, 2]], [[4, 2]]]]
        for _ in range(len(index))]

    # ----- Define True Function -----
    def true_function(X, alg=oti):
        x1, x2, x3, x4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        return 1 * x1**2 + 2 * x2**2 + 3 * x3**2 + 4 * x4**2

    # Assemble training data with derivative information
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
                y_sub.append(
                    y_hc.get_deriv(der_indices[k][i][j]).reshape(-1, 1)
                )

        y_train_data.append(y_sub)

    # Build weighted GP model
    gp = oti_gp_weighted(
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
        n_restart_optimizer=15, swarm_size=25
    )

    # Predict on full test set
    y_pred = gp.predict(
        X_test, params, calc_cov=False, return_submodels=False)
    y_true = true_function(X_test, alg=np)
    nrmse_val = utils.nrmse(y_true, y_pred, norm_type="minmax")
    print("NRMSE between model and true function: {}".format(nrmse_val))

    # ----- Slice Prediction at x3 = x4 = 0 -----
    x1x2_lin = np.linspace(-2.048, 2.048, 50)
    X1_grid, X2_grid = np.meshgrid(x1x2_lin, x1x2_lin)
    X_slice = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    X_slice = np.hstack([X_slice, np.zeros((X_slice.shape[0], 2))])

    y_slice_pred = gp.predict(
        X_slice, params, calc_cov=False, return_submodels=False)
    y_slice_true = true_function(X_slice, alg=np)
    nrmse_slice = utils.nrmse(y_slice_true, y_slice_pred, norm_type="minmax")
    print("NRMSE for x3 = x4 = 0 slice: {}".format(nrmse_slice))

    # Plot 2D slice prediction
    plotting_helper.make_submodel_plots(
        X_train,
        y_train_data,
        X_slice,
        y_slice_pred,
        true_function,
        X1_grid=X1_grid,
        X2_grid=X2_grid,
        n_order=n_order,
        n_bases=n_bases,
        plot_submodels=False,
        submodel_vals=None,
    )
