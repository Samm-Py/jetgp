import numpy as np
import pyoti.sparse as oti  # Hyper-complex AD
import itertools
from full_ddegp.ddegp import ddegp
import utils  # Plotting and helper utilities
import plotting_helper
# -----------------------------
# Directional Derivative Enhanced GP (Full Model)
# -----------------------------


def true_function(X, alg=oti):
    """True function with directional structure."""
    x1, x2 = X[:, 0], X[:, 1]
    return x1**2 * x2 + alg.cos(10 * x1) + alg.cos(10 * x2)


def generate_rays(order, ndim=2):
    """Generate unit vectors (rays) and their hypercomplex perturbations."""
    thetas = [2 * np.pi / i for i in range(1, 4)]
    rays = np.column_stack([[np.cos(t), np.sin(t)] for t in thetas])
    e = [oti.e(i + 1, order=order) for i in range(rays.shape[1])]
    perts = np.dot(rays, e)
    return rays, perts


def generate_training_data(n_order, num_points=5):
    x_vals = np.linspace(-1, 1, num_points)
    y_vals = np.linspace(-1, 1, num_points)

    # Cartesian product for 3D grid
    X_train = np.array(list(itertools.product(x_vals, y_vals)))

    # Convert to OTI array for hypercomplex perturbation
    X_train_pert = oti.array(X_train)

    # Apply directional perturbations
    rays, perts = generate_rays(n_order)
    for j in range(rays.shape[0]):
        X_train_pert[:, j] += perts[j]

    y_train_hc = true_function(X_train_pert, alg=oti)
    for comb in itertools.combinations(range(1, rays.shape[1] + 1), 2):
        y_train_hc = y_train_hc.truncate(comb)

    y_train_real = y_train_hc.real
    y_train = [y_train_real]

    # Derivative index structure must be consistent across all training points
    der_indices = [[
        [[1, 1]], [[1, 2]], [[1, 3]], [[1, 4]],
        [[2, 1]], [[2, 2]], [[2, 3]], [[2, 4]],
        [[3, 1]], [[3, 2]], [[3, 3]], [[3, 4]],
    ]]

    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            y_train.append(y_train_hc.get_deriv(
                der_indices[i][j]).reshape(-1, 1))

    return X_train, y_train, der_indices, rays


def main():
    np.random.seed(0)
    n_order = 4

    X_train, y_train, der_indices, rays = generate_training_data(
        n_order)

    gp = ddegp(
        X_train,
        y_train,
        n_order=n_order,
        der_indices=der_indices,
        rays=rays,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic",
    )

    params = gp.optimize_hyperparameters(
        n_restart_optimizer=15, swarm_size=50, verbose=True)

    # Test data grid
    N_grid = 20
    x_lin = np.linspace(-1, 1, N_grid)
    y_lin = np.linspace(-1, 1, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    # Predict with directional derivatives
    y_pred = gp.predict(X_test, params, calc_cov=False, return_deriv=True)

    plotting_helper.make_plots(
        X_train,
        y_train,
        X_test,
        y_pred,
        true_function,
        X1_grid=X1_grid,
        X2_grid=X2_grid,
        n_order=n_order,
        plot_derivative_surrogates=False,
        der_indices=der_indices,
    )

    y_true = true_function(X_test, alg=np).flatten()
    nrmse = utils.nrmse(y_true, y_pred.flatten())

    print("NRMSE between model and true function: {}".format(nrmse))


if __name__ == "__main__":
    main()
