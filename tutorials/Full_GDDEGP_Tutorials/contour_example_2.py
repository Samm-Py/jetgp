"""
ddgp_lhs_contour.py
------------------------------------------------------------
• 16 Latin–Hypercube points inside a box  [–2,2] × [–2,2]
• Steepest–ascent (or descent) ray at each point, chosen to point
  TOWARD the quadratic-plus-linear contour  f(x,y)=17
• Hyper-complex tagging  (one tag per point) so a DD-GP can use
  function values + first / second directional derivatives.
• Fit with full_gddegp.gddegp   (anisotropic SE kernel)
• Plot
    1. true contour  f(x,y)=true_function
    2. GP mean contour  (same levels)
    3. training points  + white arrows showing the rays
------------------------------------------------------------
"""

import itertools
import numpy as np
import pyoti.sparse as oti
from scipy.stats import qmc
from matplotlib import pyplot as plt
from full_gddegp.gddegp import gddegp          # <- your DD-GP class
from matplotlib.patches import FancyArrowPatch
from matplotlib.lines import Line2D
from shapely.geometry import LineString, box
# ------------------------------------------------------------------
# 0.  quadratic-plus-linear toy function  f(x,y)=3x²+2y²+x
# ------------------------------------------------------------------


# --- adjustable “shape” parameters ------------------------------------------
A = 3.0      # peak amplitude ( > 1 so we can clear the 1-threshold)
sigma2 = 0.8**2   # width of each bump   – smaller → tighter peaks
cx, cy = 2.0, 2.0  # centres of the two peaks: (+cx,+cy) and (-cx,-cy)
T = 1.0      # threshold you care about
# -----------------------------------------------------------------------------


def true_function(X, alg=np):
    """
    Two Gaussian bumps centred at (±cx, ±cy), shifted down so that the saddle
    lies below the threshold T.

    Parameters
    ----------
    X   : (N,2) array-like
    alg : numpy-like module (defaults to np) so you can pass jax.numpy, etc.

    Returns
    -------
    f   : (N,) array
    """
    x, y = X[:, 0], X[:, 1]

    # Gaussian bumps
    g1 = A * alg.exp(-((x - cx)**2 + (y - cy)**2) / sigma2)
    g2 = A * alg.exp(-((x + cx)**2 + (y + cy)**2) / sigma2)

    # shift down so the baseline sits well below the threshold
    return g1 + g2 - T


def grad_true(x, y):
    """
    Gradient of `true_function` evaluated at a *single* point (x, y).

    Returns
    -------
    ∇f : array([df/dx, df/dy])
    """
    # individual exponents (re-use for efficiency)
    e1 = np.exp(-((x - cx)**2 + (y - cy)**2) / sigma2)
    e2 = np.exp(-((x + cx)**2 + (y + cy)**2) / sigma2)

    # common prefactor
    pref = -2 * A / sigma2

    gx = pref * ((x - cx) * e1 + (x + cx) * e2)
    gy = pref * ((y - cy) * e1 + (y + cy) * e2)
    return np.array([gx, gy])


# ------------------------------------------------------------------
# 1.  helper: Latin-Hypercube points in a 2-D box
# ------------------------------------------------------------------


def lhs_points(n_samples, box, seed=None):
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n_samples)
    X = np.empty_like(unit)
    for j, (lo, hi) in enumerate(box):
        X[:, j] = lo + (hi - lo) * unit[:, j]
    return X

# ------------------------------------------------------------------
# 2.  rays pointing toward the f(x)=threshold contour
# ------------------------------------------------------------------


def rays_to_contour(X, threshold=17.0, eps=1e-12):
    rays, tags, rays_plot = [], [], []
    for idx, (x, y) in enumerate(X):
        f_val = true_function(np.array([[x, y]]))[0]
        g = grad_true(x, y)
        if np.linalg.norm(g) < eps:
            v = np.array([[1.0], [0.0]])            # arbitrary axis
        else:
            direction = g if f_val < threshold else -g
            v = (direction / np.linalg.norm(direction)).reshape(2, 1)
        rays.append(v)
        rays_plot.append(v * .4)
        tags.append(idx + 1)
    return rays, tags, rays_plot

# ------------------------------------------------------------------
# 3.  build training data  (f, D_v f, D_v² f)
# ------------------------------------------------------------------


def generate_training_data_lhs(n_samples=16,
                               box=((-2, 2), (-2, 2)),
                               n_order=2,
                               max_order=2,
                               threshold=17.0,
                               seed=None):
    X_train = lhs_points(n_samples, box, seed)
    rays_list, tag_map, rays_plot = rays_to_contour(X_train, threshold)

    # hyper-complex inputs
    X_hc = oti.array(X_train)
    for i, tag in enumerate(tag_map):
        e_tag = oti.e(1, order=n_order)   # create dual-unit
        perturb = (oti.array(rays_list[i]) * e_tag)        # (2,)
        X_hc[i, :] += perturb.T

    # evaluate in HC algebra
    f_hc = true_function(X_hc, alg=oti)
    for a, b in itertools.combinations(tag_map, 2):
        f_hc = f_hc.truncate((a, b))

    # targets
    y_blocks = [f_hc.real.reshape(-1, 1)]                # f
    der_indices = [[[1, 1]], [[1, 2]], [[1, 3]], [[1, 4]], [
        [1, 5]], [[1, 6]], [[1, 7]], [[1, 8]], [[1, 9]], [[1, 10]]]
    for idx in der_indices:
        y_blocks.append(f_hc.get_deriv(idx).reshape(-1, 1))

    return X_train, y_blocks, rays_list, rays_plot

# ------------------------------------------------------------------
# arrow helper (clipped to axes)
# ------------------------------------------------------------------

# ------------- Clipped Arrow Function ------------------------


def clipped_arrow(ax, start, vec, color="white", **kwargs):
    """
    Draw an arrow clipped to the axes limits.
    """
    x0, y0 = start
    dx, dy = vec
    x1, y1 = x0 + dx, y0 + dy

    # Axes limits
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Create line segment and bounding box
    line = LineString([(x0, y0), (x1, y1)])
    bounds = box(xlim[0], ylim[0], xlim[1], ylim[1])
    clipped = line.intersection(bounds)

    if clipped.is_empty:
        return

    if clipped.geom_type == 'LineString':
        x_start, y_start = clipped.coords[0]
        x_end, y_end = clipped.coords[-1]
        ax.add_patch(FancyArrowPatch((x_start, y_start),
                                     (x_end, y_end),
                                     arrowstyle='->',
                                     color=color,
                                     linewidth=1.5,
                                     mutation_scale=10,
                                     **kwargs))

# ------------------------------------------------------------------
# main demo
# ------------------------------------------------------------------


def main():
    n_order = 10
    X_train, y_blocks, rays_list, rays_plot = generate_training_data_lhs(
        n_samples=16,
        box=((-4, 4), (-4, 4)),
        n_order=n_order,
        max_order=n_order,
        threshold=1.0,
        seed=42)

    # ---- fit DD-GP -------------------------------------------------
    gp = gddegp(X_train, y_blocks,
                n_order=n_order,
                rays_list=rays_list,
                normalize=True,
                kernel="SE",
                kernel_type="anisotropic",)

    params = gp.optimize_hyperparameters(
        n_restart_optimizer=25, swarm_size=30, verbose=True)

    # ---- prediction grid ------------------------------------------
    gx = np.linspace(-4, 4, 100)
    gy = np.linspace(-4, 4, 100)
    X1, X2 = np.meshgrid(gx, gy)
    X_pred = np.column_stack([X1.ravel(), X2.ravel()])

    # same global ray for prediction (not used for f)
    ray0 = np.array([[1.0], [0.0]])
    rays_pred = [ray0 for _ in range(X_pred.shape[0])]

    y_pred = gp.predict(X_pred, rays_pred, params,
                        calc_cov=False, return_deriv=False).ravel()

    y_pred_train = gp.predict(X_train, rays_list, params,
                              calc_cov=False, return_deriv=True).ravel()

    print(y_pred_train)
    print(y_blocks)
    y_true = true_function(X_pred).ravel()

    # ===================== Plotting ===============================
    threshold = 1.0

    fig, (ax_gp, ax_true) = plt.subplots(1, 2, figsize=(13, 5),
                                         sharex=True, sharey=True)

    Zp = y_pred.reshape(X1.shape)      # GP mean grid
    Zt = y_true.reshape(X1.shape)      # true grid

    # common filled-contour levels
    levels = np.linspace(min(Zp.min(), Zt.min()),
                         max(Zp.max(), Zt.max()), 40)

    # ===== GP panel ==================================================
    cf1 = ax_gp.contourf(X1, X2, Zp, levels=levels, cmap="viridis")
    fig.colorbar(cf1, ax=ax_gp, fraction=0.046)
    ax_gp.set_title("GP mean")

    gp_line = ax_gp.contour(X1, X2, Zp, levels=[threshold],
                            colors="red", linewidths=1.8, linestyles="-")
    true_line = ax_gp.contour(X1, X2, Zt, levels=[threshold],
                              colors="black", linewidths=1.8, linestyles="--")

    ax_gp.scatter(X_train[:, 0], X_train[:, 1],
                  c="red", edgecolor="k", s=35, zorder=3)
    for pt, v in zip(X_train, rays_plot):
        clipped_arrow(ax_gp, pt, v.flatten(), color="white")

    handles = [Line2D([], [], color="red", lw=2, label="GP  f=17"),
               Line2D([], [], color="black", lw=2, ls="--", label="True f=17")]
    ax_gp.legend(handles=handles, loc="upper right")

    # ===== True panel =================================================
    cf2 = ax_true.contourf(X1, X2, Zt, levels=levels, cmap="viridis")
    fig.colorbar(cf2, ax=ax_true, fraction=0.046)
    ax_true.set_title("True function")

    ax_true.contour(X1, X2, Zt, levels=[threshold],
                    colors="black", linewidths=1.8, linestyles="--")
    ax_true.contour(X1, X2, Zp, levels=[threshold],
                    colors="red", linewidths=1.8, linestyles="-")

    ax_true.scatter(X_train[:, 0], X_train[:, 1],
                    c="red", edgecolor="k", s=35, zorder=3)
    for pt, v in zip(X_train, rays_plot):
        clipped_arrow(ax_true, pt, v.flatten(), color="white")

    # ===== Axis Formatting ============================================
    for ax in (ax_gp, ax_true):
        ax.set_xlim([-4, 4])
        ax.set_ylim([-4, 4])
        ax.set_aspect("equal")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
