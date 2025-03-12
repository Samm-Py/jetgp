import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import pyoti.sparse as oti
import itertools
from oti_gp import oti_gp_directional_weighted


# ---------------------------------------------------------------------
# DEMO: multi-dimensional example with separate length scales
# ---------------------------------------------------------------------
if __name__ == "__main__":

    def true_function(x_p, y_p, alg=oti):
        f = (
            alg.cos(y_p)
            + alg.cos(x_p)
            + 2 * alg.sin(2 * y_p)
            + 2 * alg.sin(2 * x_p)
            + alg.sin(3 * y_p)
            + alg.sin(3 * x_p)
            + 2 * alg.sin(4 * y_p)
            + 2 * alg.sin(4 * x_p)
            + alg.sin(5 * y_p)
            + alg.sin(5 * x_p)
        )

        return f

    # np.random.seed(422)

    # Let's make a synthetic problem in D=2.
    # We'll define f(x1, x2) = sin(x1) * cos(x2) or something like that.
    # We'll sample some points from a square region and add noise.
    np.random.seed(0)
    # 1) Generate training data

    n_order = 15
    n_bases = 2
    n_dim = 2
    lb_x = -1
    ub_x = 1
    lb_y = 0.0
    ub_y = 1
    # der_indices = gen_OTI_indices(n_bases, n_order)
    der_indices = [
        [
            [[1, 1]],
            [[1, 2]],
            [[1, 3]],
            [[1, 4]],
            [[1, 5]],
            [[1, 6]],
            [[1, 7]],
            [[1, 8]],
            [[1, 9]],
            [[1, 10]],
            [[1, 11]],
            [[1, 12]],
            [[1, 13]],
            [[1, 14]],
            [[1, 15]],
        ]
    ]

    num_points = 1  # 4x4 grid to get 16 points

    # Generate linearly spaced values in each dimension
    # x_vals = np.linspace(lb_x, ub_x, num_points)
    # y_vals = np.linspace(lb_y, ub_y, num_points)

    # Create a grid of points
    # X_train = x_vals
    X_train = np.array([0, 0]).reshape(1, -1)
    X_train_pert = oti.array(X_train)
    thetas = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    n_rays = len(thetas)
    rays = np.zeros((n_dim, n_rays))
    y_train_data = []
    index = [[i] for i in range(len(thetas))]
    for val in index:
        ray = np.zeros((n_dim, 1))
        theta = thetas[val[0]]

        ray[:, 0] = [np.cos(theta), np.sin(theta)]
        rays[:, val[0]] = ray[:, 0]

        nrays = rays.shape[1]

        e = [oti.e(i + 1, order=n_order) for i in range(1)]
        x_p, y_p = np.dot(ray, e)

        y_train_hc = true_function(x_p, y_p)

        for comb in itertools.combinations(range(1, nrays + 1), 2):
            # This removes any combination between directions e_i * e_j = 0
            # Truncate all possible combinations between imdirections.
            y_train_hc = y_train_hc.truncate(comb)
        # end for

        y_train_real = y_train_hc.real
        # y_train_real_norm = (y_train_real - np.mean(y_train_real)) / (
        #     np.std(y_train_real, ddof=1)
        # )

        y_train = y_train_real
        for i in range(0, len(der_indices)):
            for j in range(0, len(der_indices[i])):
                y_train = np.vstack(
                    (y_train, y_train_hc.get_deriv(der_indices[i][j]))
                )

        y_train = y_train.flatten()
        sigma_n_true = 0.0
        noise = sigma_n_true * np.random.randn(len(y_train))
        y_train_noisy = y_train + noise
        y_train_data.append(y_train)

    sigma_f = 1.0
    sigma_n = sigma_n_true  # matching the noise used above

    gp = oti_gp_directional_weighted(
        X_train,
        y_train_data,
        n_order,
        n_bases,
        index,
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
    true_values = true_function(X_test[:, 0], X_test[:, 1], alg=np).reshape(
        (N_grid, N_grid)
    )

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
    t = np.linspace(-0.0, 1.0, 1000)
    x_plot = []
    y_plot = []
    f_plot = []

    thetas = [0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi]
    for i, theta in enumerate(thetas):
        rays[:, i] = [np.cos(theta), np.sin(theta)]

    nrays = rays.shape[1]

    e = [oti.e(i + 1, order=n_order) for i in range(nrays)]
    x_p, y_p = np.dot(rays, e)

    f_p = true_function(x_p, y_p, alg=oti)
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
