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
    thetas = [0, np.pi/2]
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
        [[1, 1]], [[1, 2]],
        [[2, 1]], [[2, 2]],
        [[3, 1]], [[3, 2]],
    ]]

    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            y_train.append(y_train_hc.get_deriv(
                der_indices[i][j]).reshape(-1, 1))

    return X_train, y_train, der_indices, rays


def main():
    np.random.seed(0)
    n_order = 3

    X_train, y_train, der_indices, rays = generate_training_data(
        n_order)


if __name__ == "__main__":
    main()
