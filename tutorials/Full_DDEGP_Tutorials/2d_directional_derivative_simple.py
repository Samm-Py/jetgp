import numpy as np
import pyoti.sparse as oti  # Hyper-complex AD
import itertools
from full_ddegp.ddegp import ddegp
import utils  # Plotting and helper utilities
import plotting_helper
from matplotlib import pyplot as plt
# -----------------------------
# Directional Derivative Enhanced GP (Full Model)
# -----------------------------


def true_function(X, alg=np):
    """Branin-Hoo function with directional structure."""
    x1, x2 = X[:, 0], X[:, 1]
    return 3*x1**2 + 2*x2**2 + x1


def generate_rays(order, ndim=2):
    """Generate unit vectors (rays) and their hypercomplex perturbations."""
    thetas = [np.pi/2]
    rays = np.column_stack([[np.cos(t), np.sin(t)] for t in thetas])
    e = [oti.e(i + 1, order=order) for i in range(rays.shape[1])]
    perts = np.dot(rays, e)
    return rays, perts


def generate_training_data(n_order, num_points=3):
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
        [[1, 1]]

    ]]

    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            y_train.append(y_train_hc.get_deriv(
                der_indices[i][j]).reshape(-1, 1))

    return X_train, y_train, der_indices, rays


def clipped_arrow(ax, origin, direction, length, bounds):
    """Draw an arrow clipped at axis bounds."""
    x0, y0 = origin
    dx, dy = direction * length
    xlim, ylim = bounds

    # Compute t limits to avoid drawing outside bounds
    tx = np.inf if dx == 0 else (
        xlim[1] - x0)/dx if dx > 0 else (xlim[0] - x0)/dx
    ty = np.inf if dy == 0 else (
        ylim[1] - y0)/dy if dy > 0 else (ylim[0] - y0)/dy
    t = min(1.0, tx, ty)

    dx *= t
    dy *= t

    ax.arrow(x0, y0, dx, dy,
             head_width=0.25, head_length=0.35,
             fc='white', ec='white', clip_on=True)


def main():
    np.random.seed(0)
    n_order = 1

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
        n_restart_optimizer=51, swarm_size=50, verbose=True)

    # ==== Prediction Grid ====
    gx = np.linspace(-2, 2, 120)
    gy = np.linspace(-2, 2, 120)
    X1, X2 = np.meshgrid(gx, gy)
    X_pred = np.column_stack([X1.ravel(), X2.ravel()])

    # ==== GP Prediction ====
    y_pred = gp.predict(X_pred, params, calc_cov=False,
                        return_deriv=False).ravel()
    y_pred_train = gp.predict(X_train, params, calc_cov=False,
                              return_deriv=True).ravel()
    y_true = true_function(X_pred, alg=np).ravel()

    # ==== Compute Error ====
    # Mask where y_true is not zero
    nonzero_mask = y_true != 0

    # Initialize with absolute error
    rel_err = np.abs(y_pred - y_true)

    # Where y_true ≠ 0, replace with relative error
    rel_err[nonzero_mask] = np.abs(
        (y_pred[nonzero_mask] - y_true[nonzero_mask]) / y_true[nonzero_mask])
    print(rel_err)
    nrmse = utils.nrmse(y_true, y_pred)
    print(f"NRMSE between model and true function: {nrmse:.4e}")

    # ==== Plotting ====
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    # === A. GP Prediction Contour Plot ===
    cf1 = axs[0].contourf(X1, X2, y_pred.reshape(
        X1.shape), levels=30, cmap='viridis')
    axs[0].scatter(X_train[:, 0], X_train[:, 1],
                   c='red', s=40, label='Train Points')

    # Get plot bounds
    xlim = axs[0].get_xlim()
    ylim = axs[0].get_ylim()

    # Draw clipped arrows
    rays_list = [rays for i in range(X_train.shape[0])]

    for pt, ray in zip(X_train, rays_list):
        clipped_arrow(axs[0], pt, ray.flatten(),
                      length=0.75, bounds=(xlim, ylim))

    axs[0].set_title("GP Prediction")
    axs[0].legend()
    fig.colorbar(cf1, ax=axs[0])
    axs[0].set_xlim(gx[0], gx[-1])
    axs[0].set_ylim(gy[0], gy[-1])
    xlim = (gx[0], gx[-1])
    ylim = (gy[0], gy[-1])
    # === B. True Function Contour Plot ===
    cf2 = axs[1].contourf(X1, X2, y_true.reshape(
        X1.shape), levels=30, cmap='viridis')
    axs[1].set_title("True Function")
    fig.colorbar(cf2, ax=axs[1])

    # === C. Relative Error Contour Plot ===
    cf3 = axs[2].contourf(X1, X2, rel_err.reshape(
        X1.shape), levels=30, cmap='magma')
    axs[2].set_title("Absolute Relative Error")
    fig.colorbar(cf3, ax=axs[2])

    # === Common Axes Settings ===
    for ax in axs:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    main()
