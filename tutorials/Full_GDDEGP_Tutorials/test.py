import numpy as np
import pyoti.sparse as oti  # Hyper-complex AD
import itertools
from full_gddegp.gddegp import gddegp
import utils  # Plotting and helper utilities
import plotting_helper
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
# -----------------------------
# Directional Derivative Enhanced GP (Full Model)
# -----------------------------


# ----------------------------------------------------------------------
#  0.  problem function
# ----------------------------------------------------------------------
def true_function(X, alg=np):
    """f(x1,x2) = x1² + x2²  (analytically friendly)."""
    x1, x2 = X[:, 0], X[:, 1]
    return 3*x1**2 + 2*x2**2 + x1


# ----------------------------------------------------------------------
#  1.  per-point direction (ray) generator
# ----------------------------------------------------------------------
def generate_pointwise_rays(n_order, X_train):
    """
    Returns
    -------
    rays_list : list[ ndarray (2×1) ]
        Unit direction column for each point.
    tag_map   : list[ int ]
        Unique tag ID assigned to each point (1-based).
    """
    n_pts = X_train.shape[0]
    rays_list, tag_map = [], []
    thetas = [np.pi/2, -np.pi/2, np.pi/4, -np.pi/2,
              np.pi/2, -np.pi/6, np.pi/2, np.pi/4, -np.pi/2]
    for idx, theta in enumerate(thetas):
        # rotate 22.5° point-to-point
        ray = np.array([[np.cos(theta)],         # (2,1)
                        [np.sin(theta)]])
        tag = idx + 1                            # unique tag per point
        rays_list.append(ray)
        tag_map.append(tag)
    return rays_list, tag_map


# ----------------------------------------------------------------------
#  2.  apply hyper-complex perturbation (one tag per point)
# ----------------------------------------------------------------------
def apply_pointwise_perturb(X_train, rays_list, tag_map, n_order):
    X_hc = oti.array(X_train)
    for i, (ray, tag) in enumerate(zip(rays_list, tag_map)):
        e_tag = oti.e(1, order=n_order)   # create dual-unit
        perturb = (oti.array(ray) * e_tag)        # (2,)
        X_hc[i, :] += perturb.T
    return X_hc


# ----------------------------------------------------------------------
#  4.  master routine
# ----------------------------------------------------------------------
def generate_training_data(n_order=2, num_points=3, max_order=2):
    """
    Builds 2D grid, attaches individual directions, returns y-blocks.

    Returns
    -------
    X_train      : (n,2)
    y_blocks     : [ f,  D_v f,  D_v² f, ... ] each shape (n,1)
    der_indices  : list of derivative indices compatible with your GP code
    rays_list    : list of (2×1) direction columns in grid order
    """
    # ---- grid on [-1,1]² --------------------------------------------------
    grid = np.linspace(-1, 1, num_points)
    X_train = np.array(list(itertools.product(grid, grid)))   # (n,2)

    # ---- per-point rays & tags -------------------------------------------
    rays_list, tag_map = generate_pointwise_rays(n_order, X_train)

    # ---- perturb with hyper-complex tags ---------------------------------
    X_pert = apply_pointwise_perturb(X_train, rays_list, tag_map, n_order)

    # ---- evaluate f in hyper-complex algebra -----------------------------
    f_hc = true_function(X_pert, alg=oti)

    # remove cross-tag parts beyond first order (keeps only per-point dirs)
    for combo in itertools.combinations(tag_map, 2):
        f_hc = f_hc.truncate(combo)

    # ---- assemble output blocks ------------------------------------------
    y_blocks = [f_hc.real.reshape(-1, 1)]          # function values

    der_indices = [[[1, 1]], [[1, 2]]]

    for idx in der_indices:
        y_blocks.append(
            f_hc.get_deriv(idx).reshape(-1, 1)
        )

    return X_train, y_blocks, rays_list

# ==== Helper function ====


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
    n_order = 2

    X_train, y_train, rays_list = generate_training_data(
        n_order)

    gp = gddegp(
        X_train,
        y_train,
        n_order=n_order,
        rays_list=rays_list,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic",
    )

    params = gp.optimize_hyperparameters(
        n_restart_optimizer=51, swarm_size=50, verbose=True)

    # ==== Prediction Grid ====
    gx = np.linspace(-2, 2, 200)
    gy = np.linspace(-2, 2, 200)
    X1, X2 = np.meshgrid(gx, gy)
    X_pred = np.column_stack([X1.ravel(), X2.ravel()])

    # ==== GP Prediction ====

    theta = np.pi/2
    ray = np.array([[np.cos(theta)],         # (2,1)
                    [np.sin(theta)]])
    rays_pred = [ray for i in range(X_pred.shape[0])]
    y_pred = gp.predict(X_pred, rays_pred, params, calc_cov=False,
                        return_deriv=True).ravel()
    N = X_pred.shape[0]
    # ------------------------------------------------------------------
    # 2)  Pick an arbitrary direction  u = (cos θ, sin θ)
    # ------------------------------------------------------------------
    # <-- change this angle as you like
    u = np.array([np.cos(theta), np.sin(theta)])
    u_col = u[:, None]                   # (2,1)  -- needed by gp.predict
    rays_pred = [u_col] * N               # list of identical direction vectors

    # directional-derivative slice
    gp_dd = y_pred[N: 2 * N].reshape(X1.shape)

    # ------------------------------------------------------------------
    # 4)  Analytic directional derivative of  f(x1,x2)=3x1² + 2x2² + x1
    # ------------------------------------------------------------------
    def analytic_dd(X, direction):
        x1, x2 = X[:, 0], X[:, 1]
        grad = np.column_stack([6.0 * x1 + 1.0, 4.0 * x2])   # ∇f
        return grad @ direction

    true_dd = analytic_dd(X_pred, u).reshape(X1.shape)

    # ------------------------------------------------------------------
    # 5)  Absolute error map
    # ------------------------------------------------------------------
    abs_err = np.abs(gp_dd - true_dd)

    # -- LOG-SCALE absolute error --------------------------------------
    eps = 1e-6                          # avoids log(0)
    abs_err_clip = np.clip(abs_err, eps, None)

    err_min, err_max = abs_err_clip.min(), abs_err_clip.max()
    n_levels = 12                       # how many filled bands
    log_levels = np.logspace(np.log10(err_min),
                             np.log10(err_max),
                             n_levels)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    # --- analytic --------------------------------------------------
    cf0 = axes[0].contourf(X1, X2, true_dd,
                           levels=np.linspace(-15, 15, 13),
                           vmin=-15, vmax=15)
    fig.colorbar(cf0, ax=axes[0])
    axes[0].set_title("Analytic $\\partial_{u}f$")

    # --- GP prediction --------------------------------------------
    cf1 = axes[1].contourf(X1, X2, gp_dd,
                           levels=np.linspace(-15, 15, 13),
                           vmin=-15, vmax=15)
    fig.colorbar(cf1, ax=axes[1])
    axes[1].set_title("GP prediction")

    # --- absolute error (log scale) -------------------------------
    cf2 = axes[2].contourf(X1, X2, abs_err_clip,
                           levels=log_levels,
                           norm=LogNorm(vmin=err_min, vmax=err_max),
                           cmap="magma_r")
    fig.colorbar(cf2, ax=axes[2], format="%.1e")
    axes[2].set_title("Absolute error (log scale)")

    for ax in axes:
        ax.set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")

    fig.suptitle(
        fr"Directional derivative comparison,  $\theta = {theta:.3f}$ rad")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    y_pred = y_pred[0:N]
    y_true = true_function(X_pred, alg=np).ravel()

    # ==== Compute Error ====
    # Mask where y_true is not zero
    nonzero_mask = y_true != 0

    # Initialize with absolute error
    rel_err = np.abs(y_pred[0:N] - y_true)

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
