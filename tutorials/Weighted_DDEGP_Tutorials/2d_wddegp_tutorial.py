import numpy as np
import pyoti.sparse as oti
import itertools
from wddegp.wddegp import wddegp
import utils
import plotting_helper


def true_function(X, alg=oti):
    x1, x2 = X[:, 0], X[:, 1]
    return x1**2 * x2 + alg.cos(10 * x1) + alg.cos(10 * x2)


def generate_training_data(num_points=5):
    x_vals = np.linspace(-1, 1, num_points)
    y_vals = np.linspace(-1, 1, num_points)
    X = np.array(list(itertools.product(x_vals, y_vals)))

    old_index = [[1, 2, 3], [5, 10, 15], [9, 14, 19], [21, 22, 23], [
        0], [4], [20], [24], [6, 7, 8, 11, 12, 13, 16, 17, 18]]
    index = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [12],
             [13], [14], [15], [16, 17, 18, 19, 20, 21, 22, 23, 24]]

    old_flat = list(itertools.chain.from_iterable(old_index))
    new_flat = list(itertools.chain.from_iterable(index))
    reorder = np.zeros(25, dtype=int)
    for i in range(25):
        reorder[new_flat[i]] = old_flat[i]

    return X[reorder], index


def build_gp_model(X_train, index, n_order=2, n_bases=2):
    y_train_real = true_function(X_train, alg=np)
    der_indices = [
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
        [[[[1, 1]], [[1, 2]], [[2, 1]], [[2, 2]], [[3, 1]], [[3, 2]]]],
    ]

    thetas = [
        [-np.pi/2, 0, np.pi/2],
        [-np.pi, -np.pi/2, 0],
        [-np.pi, np.pi/2, 0],
        [-np.pi/2, -np.pi, np.pi/2],
        [-np.pi/2, 0, -np.pi/4],
        [np.pi/2, 0, np.pi/4],
        [np.pi/2, 0, np.pi/4],
        [-np.pi/2, -np.pi, -np.pi/4 - np.pi/2],
        [np.pi/2, np.pi/4, np.pi/4 + np.pi/2]
    ]

    y_train_data = []
    rays_data = []

    for k, val in enumerate(index):
        X_sub = X_train[val]
        X_pert = oti.array(X_sub)

        rays = np.zeros((n_bases, len(thetas[k])))
        for i, theta in enumerate(thetas[k]):
            rays[:, i] = [np.cos(theta), np.sin(theta)]
        rays_data.append(rays)

        nrays = rays.shape[1]
        e = [oti.e(i + 1, order=n_order) for i in range(nrays)]
        x_p, y_p = np.dot(rays, e)
        perts = [x_p, y_p]
        for j in range(X_train.shape[1]):
            X_pert[:, j] = X_pert[:, j] + perts[j]

        y_hc = true_function(X_pert, alg=oti)

        for comb in itertools.combinations(range(1, nrays + 1), 2):
            y_hc = y_hc.truncate(comb)

        y_train = [y_train_real]
        for i in range(len(der_indices[k])):
            for j in range(len(der_indices[k][i])):
                y_train.append(y_hc.get_deriv(
                    der_indices[k][i][j]).reshape(-1, 1))

        y_train_data.append(y_train)

    gp = wddegp(
        X_train,
        y_train_data,
        n_order,
        n_bases,
        index,
        der_indices,
        rays_data,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic",
    )

    return gp


def main():
    np.random.seed(0)
    n_order, n_bases, num_points = 2, 2, 5
    X_train, index = generate_training_data(num_points)
    gp = build_gp_model(X_train, index, n_order, n_bases)

    params = gp.optimize_hyperparameters(n_restart_optimizer=20, swarm_size=50)

    N_grid = 25
    x_lin = np.linspace(-1, 1, N_grid)
    y_lin = np.linspace(-1, 1, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    y_pred, submodel_vals = gp.predict(
        X_test, params, calc_cov=False, return_submodels=True)

    plotting_helper.make_submodel_plots(
        X_train,
        gp.y_train,
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

    y_true = true_function(X_test, alg=np).flatten()
    nrmse = utils.nrmse(y_true, y_pred)

    print("NRMSE between model and true function: {}".format(nrmse))


if __name__ == "__main__":
    main()
