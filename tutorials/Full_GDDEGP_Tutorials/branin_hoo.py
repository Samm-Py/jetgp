import numpy as np
import pyoti.sparse as oti  # Hyper-complex AD
import itertools
from full_gddegp.gddegp import gddegp
import utils  # Plotting and helper utilities
import plotting_helper
from scipy.stats import qmc
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import utils
# -----------------------------
# Directional Derivative Enhanced GP (Full Model)
# -----------------------------


# ----------------------------------------------------------------------
#  0.  problem function
# ----------------------------------------------------------------------

pi = np.pi
a = 1.0
b = 5.1 / (4*pi**2)
c = 5.0 / pi
r = 6.0
s = 10.0
t = 1.0 / (8*pi)


def true_function(X, alg=np):
    """f(x1,x2) = x1² + x2²  (analytically friendly)."""
    x, y = X[:, 0], X[:, 1]
    return (y - b*x**2 + c*x - r)**2 + s*(1 - t)*alg.cos(x) + s


def gradient_angles_lhs(n_samples=25, seed=None):
    sampler = qmc.LatinHypercube(d=2, seed=1)
    unit = sampler.random(n=n_samples)             # (n,2)

    # 2) affine map to Branin rectangle
    X = np.empty_like(unit)
    X[:, 0] = -5.0 + (10.0 - (-5.0)) * unit[:, 0]     # x ∈ [-5,10]
    X[:, 1] = 0.0 + (15.0 - 0.0) * unit[:, 1]     # y ∈ [0,15]

    # 2) compute gradient angles
    thetas = []
    for x, y in X:
        gx, gy = grad_branin(x, y)
        theta = np.arctan2(gy, gx)      # rad
        thetas.append(theta)

    thetas = np.array(thetas)
    return X, thetas, np.degrees(thetas)

# ----------------------------------------------------------------------
#  1.  per-point direction (ray) generator
# ----------------------------------------------------------------------


def generate_pointwise_rays(n_order):
    """
    Returns
    -------
    rays_list : list[ ndarray (2×1) ]
        Unit direction column for each point.
    tag_map   : list[ int ]
        Unique tag ID assigned to each point (1-based).
    """

    X_train, thetas, _ = gradient_angles_lhs(n_samples=15)
    n_pts = X_train.shape[0]
    rays_list, tag_map = [], []
    for idx, theta in enumerate(thetas):
        # rotate 22.5° point-to-point
        ray = np.array([[np.cos(theta)],         # (2,1)
                        [np.sin(theta)]])
        tag = idx + 1                            # unique tag per point
        rays_list.append(ray)
        tag_map.append(tag)
    return X_train, rays_list, thetas, tag_map


def grad_branin(x, y):
    """Analytic gradient (gx, gy)."""
    gx = 2*(y - b*x**2 + c*x - r)*(-2*b*x + c) - s*(1 - t)*np.sin(x)
    gy = 2*(y - b*x**2 + c*x - r)
    return gx, gy


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

    # ---- per-point rays & tags -------------------------------------------
    X_train, rays_list, thetas, tag_map = generate_pointwise_rays(n_order)

    # ---- perturb with hyper-complex tags ---------------------------------
    X_pert = apply_pointwise_perturb(X_train, rays_list, tag_map, n_order)

    # ---- evaluate f in hyper-complex algebra -----------------------------
    f_hc = true_function(X_pert, alg=oti)

    # remove cross-tag parts beyond first order (keeps only per-point dirs)
    for combo in itertools.combinations(tag_map, 2):
        f_hc = f_hc.truncate(combo)

    # ---- assemble output blocks ------------------------------------------
    y_blocks = [f_hc.real.reshape(-1, 1)]          # function values

    der_indices = [[[1, 1]]]

    for idx in der_indices:
        y_blocks.append(
            f_hc.get_deriv(idx).reshape(-1, 1)
        )

    return X_train, thetas, y_blocks, rays_list


def main():
    np.random.seed(0)
    n_order = 1

    X_train, thetas, y_train, rays_list = generate_training_data(
        n_order)
    rays_array = np.hstack(rays_list)
    gp = gddegp(
        X_train,
        y_train,
        n_order=n_order,
        rays_array=rays_array,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic",
    )

    params = gp.optimize_hyperparameters(
        n_restart_optimizer=25, swarm_size=50, verbose=True)

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

    # ==== Prediction Grid ====
    gx = np.linspace(-5, 10, 20)
    gy = np.linspace(0, 15, 20)
    X1, X2 = np.meshgrid(gx, gy)
    X_pred = np.column_stack([X1.ravel(), X2.ravel()])

    # ==== GP Prediction ====

    theta = np.pi/2
    ray = np.array([[np.cos(theta)],         # (2,1)
                    [np.sin(theta)]])
    rays_pred = [ray for i in range(X_pred.shape[0])]
    rays_pred = np.hstack(rays_pred)
    y_pred = gp.predict(X_pred[-1, :], rays_pred[:, -1].reshape(-1, 1), params, calc_cov=False,
                        return_deriv=True).ravel()
    y_pred = gp.predict(X_pred, rays_pred, params, calc_cov=False,
                        return_deriv=True).ravel()
    # y_actual = true_function(X_pred[:10,:], alg = np)
    # utils.check_gp_gradient(gp, X_pred[1000, :], params)
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
        """
        Directional derivative of the Branin–Hoo function.

        Parameters
        ----------
        X : ndarray, shape (N, 2)
            Query points where each row is (x1, x2).
        direction : array_like, shape (2,)
            Direction vector (will be normalised automatically).

        Returns
        -------
        dd : ndarray, shape (N,)
            Directional derivative f'(X) · direction for every row of X.
        """
        X = np.asarray(X, dtype=float)
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)            # optional: use the unit direction

        x1, x2 = X[:, 0], X[:, 1]

        # ---- Branin–Hoo constants -------------------------------------------------
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        # ---- helper ---------------------------------------------------------------
        g = x2 - b * x1**2 + c * x1 - r          # inner quadratic/cross term

        # gradient components
        df_dx1 = 2 * a * g * (-2 * b * x1 + c) - s * (1 - t) * np.sin(x1)
        df_dx2 = 2 * a * g

        grad = np.column_stack([df_dx1, df_dx2])  # ∇f(X)

        return grad @ d

    true_dd = analytic_dd(X_pred, u).reshape(X1.shape)

    # ------------------------------------------------------------------
    # 5)  Absolute error map
    # ------------------------------------------------------------------
    abs_err = np.abs(gp_dd - true_dd)

    # -- LOG-SCALE absolute error --------------------------------------
    eps = 1e-6                          # avoids log(0)
    abs_err_clip = np.clip(abs_err, eps, None)

    err_min, err_max = abs_err_clip.min(), abs_err_clip.max()
    n_levels = 100                       # how many filled bands
    log_levels = np.logspace(np.log10(err_min),
                             1,
                             n_levels)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)

    # --- analytic --------------------------------------------------
    cf0 = axes[0].contourf(X1, X2, true_dd,
                           levels=np.linspace(-40, 30, 100),
                           vmin=-40, vmax=30)
    fig.colorbar(cf0, ax=axes[0])
    axes[0].set_title("Analytic $\\partial_{u}f$")

    # --- GP prediction --------------------------------------------
    cf1 = axes[1].contourf(X1, X2, gp_dd,
                           levels=np.linspace(-40, 30, 100),
                           vmin=-40, vmax=30)
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
    rel_err = np.abs(y_pred - y_true)

    # Where y_true ≠ 0, replace with relative error
    rel_err[nonzero_mask] = np.abs(
        (y_pred[nonzero_mask] - y_true[nonzero_mask]) / y_true[nonzero_mask])
    print(rel_err)
    nrmse = utils.nrmse(y_true, y_pred)
    print(f"NRMSE between model and true function: {nrmse:.4e}")

    # ==== Get Training Points and Rays ====
    rays_list = [np.array([[np.cos(theta)], [np.sin(theta)]])
                 for theta in thetas]

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

    # --- absolute error (log scale) -------------------------------
    abs_err = np.abs(y_pred.reshape(X1.shape) - y_true.reshape(X1.shape))

    # -- LOG-SCALE absolute error --------------------------------------
    eps = 1e-6                   # avoids log(0)
    abs_err_clip = np.clip(abs_err, eps, None)

    err_min, err_max = abs_err_clip.min(), abs_err_clip.max()
    n_levels = 200                       # how many filled bands
    log_levels = np.logspace(np.log10(err_min),
                             1,
                             n_levels)
    cf3 = axs[2].contourf(X1, X2, abs_err_clip,
                          levels=log_levels,
                          norm=LogNorm(vmin=err_min, vmax=err_max),
                          cmap="magma_r")
    fig.colorbar(cf3, ax=axs[2], format="%.1e")
    axs[2].set_title("Absolute error (log scale)")

    # === Common Axes Settings ===
    for ax in axs:
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal")

    plt.show()


if __name__ == "__main__":
    main()
