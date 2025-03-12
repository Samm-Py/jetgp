import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pyoti.sparse as oti
import itertools
from oti_gp import oti_gp_directional


# ---------------------------------------------------------------------
# DEMO: multi-dimensional example with separate length scales
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # np.random.seed(422)

    # Let's make a synthetic problem in D=2.
    # We'll define f(x1, x2) = sin(x1) * cos(x2) or something like that.
    # We'll sample some points from a square region and add noise.
    np.random.seed(0)
    # 1) Generate training data

    n_order = 4
    n_bases = 2
    lb_x = -np.pi
    ub_x = np.pi
    lb_y = -np.pi
    ub_y = np.pi
    # der_indices = gen_OTI_indices(n_bases, n_order)
    der_indices = [
        [
            [[1, 1]],
            [[1, 2]],
            [[1, 3]],
            [[1, 4]],
            [[2, 1]],
            [[2, 2]],
            [[2, 3]],
            [[2, 4]],
            [[3, 1]],
            [[3, 2]],
            [[3, 3]],
            [[3, 4]],
        ]
    ]

    num_points = 3  # 4x4 grid to get 16 points
    x_vals = np.linspace(-np.pi, np.pi, num_points)
    y_vals = np.linspace(-np.pi, np.pi, num_points)
    X_train = np.array(list(itertools.product(x_vals, y_vals)))
    X_train_pert = oti.array(X_train)
    # X_norm = (X_train - np.mean(X_train)) / (np.std(X_train, ddof=1))

    def true_function(X, alg=oti):
        f = (
            alg.cos(X[:, 0])
            + alg.cos(X[:, 1])
            + alg.cos(2 * X[:, 0])
            + alg.cos(2 * X[:, 1])
            + alg.cos(3 * X[:, 0])
            + alg.cos(3 * X[:, 1])
        )

        return f

    nrays = 3
    ndim = 2
    order = n_order

    # Vector with the dimensions of the rays.
    rays = np.zeros((ndim, nrays))

    # Subdivide half space

    thetas_list = [
        [np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        [np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        [np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        [np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        [np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        [np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        [np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        [np.pi / 4, np.pi / 2, 3 * np.pi / 4],
        [np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    ]
    for index_point, thetas in enumerate(thetas_list):
        for i, theta in enumerate(thetas):
            rays[:, i] = [np.cos(theta), np.sin(theta)]

        nrays = rays.shape[1]

        e = [oti.e(i + 1, order=order) for i in range(nrays)]
        x_p, y_p = np.dot(rays, e)
        perts = [x_p, y_p]

        for j in range(X_train.shape[1]):
            X_train_pert[index_point, j] = (
                X_train_pert[index_point, j] + perts[j]
            )

    y_train_hc = true_function(X_train_pert, alg=oti)

    for comb in itertools.combinations(range(1, nrays + 1), 2):
        # This removes any combination between directions e_i * e_j = 0
        # Truncate all possible combinations between imdirections.
        y_train_hc = y_train_hc.truncate(comb)

    y_train_real = y_train_hc.real
    y_train = y_train_real

    for i in range(0, len(der_indices)):
        for j in range(0, len(der_indices[i])):
            y_train = np.vstack(
                (y_train, y_train_hc.get_deriv(der_indices[i][j]))
            )

    y_train = y_train.flatten()
    sigma_n_true = 0.0000
    noise = sigma_n_true * np.random.randn(len(y_train))
    y_train_noisy = y_train + noise

    # 2) We'll keep sigma_f=1.0 and sigma_n=0.1 fixed,
    #    only optimize D=2 length scales.
    sigma_f = 1.0
    sigma_n = sigma_n_true  # matching the noise used above

    gp = oti_gp_directional(
        X_train,
        y_train,
        n_order,
        n_bases,
        der_indices,
        rays,
        sigma_n=1e-6,
        nugget=1e-6,
        kernel="SE",
        kernel_type="anisotropic",
    )

    params = gp.optimize_hyperparameters()

    N_grid = 25
    x_lin = np.linspace(lb_x, ub_x, N_grid)
    y_lin = np.linspace(lb_y, ub_y, N_grid)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    f_mean = gp.predict(X_test, params, calc_cov=False)
    f_mean_2d = f_mean.reshape(N_grid, N_grid)  # for plotting

    # 5) Plot results
    #    We'll do a contour plot of the predicted mean vs. the true function.
    true_values = true_function(X_test, alg=np).reshape((N_grid, N_grid))

    plt.figure(figsize=(12, 5))

    # # (a) Predicted mean
    plt.subplot(1, 2, 1)
    plt.title(
        "Order {} Directional Derivative Enhanced \n Gaussian Process Regression Prediction".format(
            n_order
        )
    )
    plt.contourf(X1_grid, X2_grid, f_mean_2d, levels=50, cmap="viridis")
    plt.colorbar()
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c="white",
        edgecolors="k",
        label="Train pts",
    )
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.tight_layout()

    # (b) True function
    plt.subplot(1, 2, 2)
    title_str = r"$f(x_1, x_2) = cos(x_1) + \cos(x_2)$"
    plt.title(title_str, fontsize=12)
    plt.contourf(X1_grid, X2_grid, true_values, levels=50, cmap="viridis")
    plt.colorbar()
    plt.scatter(X_train[:, 0], X_train[:, 1], c="white", edgecolors="k")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

    plt.tight_layout()

    rmse = np.sqrt(np.mean((f_mean_2d - true_values) ** 2))

    print(
        "RMSE between Gaussin Proccess Prediction and True Function: {}".format(
            rmse
        )
    )

    # Create figure
    fig = plt.figure(figsize=(12, 5))
    ax = fig.add_subplot(projection="3d")

    # Plot surface
    surf = ax.plot_surface(
        X1_grid, X2_grid, f_mean_2d, linewidth=0, zorder=1, alpha=0.7
    )

    # For plotting directional derivative rays
    t = np.linspace(-np.pi / 2, np.pi / 2, 1000)
    x_plot = []
    y_plot = []
    f_plot = []

    X = oti.array([x_p, y_p]).T
    f_p = true_function(X, alg=oti)[0, 0]
    for i in range(nrays):
        x_plot.append(x_p.rom_eval_object([i + 1], [t]) * np.ones(t.size))
        y_plot.append(y_p.rom_eval_object([i + 1], [t]) * np.ones(t.size))
        f_plot.append(f_p.rom_eval_object([i + 1], [t]) * np.ones(t.size))

    # Plot directional derivative lines with labels
    lines = []
    labels = ["Gaussian Process\nResponse Surface"]
    proxy = plt.Line2D(
        [0],
        [0],
        linestyle="none",
        marker="s",
        markersize=10,
        color="tab:blue",
        alpha=0.5,
    )
    lines.append(proxy)  # Proxy for surface

    for i in range(len(x_plot)):
        (line,) = ax.plot3D(
            x_plot[i],
            y_plot[i],
            f_plot[i],
            zorder=100,
            label=f"Directional Derivative $v_{i+1}$",
        )
        lines.append(line)
        labels.append(f"Directional Derivative $v_{i+1}$")

    # Show legend in the margin
    ax.legend(
        lines,
        labels,
        loc="center left",
        bbox_to_anchor=(1.2, 0.5),
        frameon=False,
    )

    # Labels and formatting
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    plt.tight_layout()
    plt.show()
