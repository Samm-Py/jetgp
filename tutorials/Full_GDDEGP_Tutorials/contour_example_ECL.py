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
import utils
import os
# ------------------------------------------------------------------
# 0.  quadratic-plus-linear toy function  f(x,y)=3x²+2y²+x
# ------------------------------------------------------------------


def true_function(X, alg=np):
    x, y = X[:, 0], X[:, 1]
    return 3*x**2 + 2*y**2 + x + 2*alg.sin(2*x)*alg.cos(1.5*y)


def grad_true(x, y):
    gx = 6*x + 1 + 4*np.cos(2*x)*np.cos(1.5*y)
    gy = 4*y - 3*np.sin(2*x)*np.sin(1.5*y)
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


def rays_random(X, seed=None):
    """
    For each point in X, assigns a random unit vector direction.
    Returns:
        rays: list of shape (2,1) arrays
        tags: list of tags (e.g., 1...N)
        rays_plot: list of (2,1) arrays (shortened for plotting)
    """
    rng = np.random.default_rng(seed)
    rays, tags, rays_plot = [], [], []
    for idx, _ in enumerate(X):
        theta = rng.uniform(0, 2 * np.pi)  # random angle
        v = np.array([[np.cos(theta)], [np.sin(theta)]])
        rays.append(v)
        rays_plot.append(v * 0.4)
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
    rays_list, tag_map, rays_plot = rays_random(X_train, seed=seed)

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
    der_indices = [[[1, i+1]] for i in range(n_order)]
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


def sobol_points(n_samples, box, seed=None):
    """
    Generate n_samples Sobol points in a given box.

    Parameters
    ----------
    n_samples : int
        Number of points to generate
    box : list of tuple
        Each tuple is (lo, hi) for one dimension
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    X : ndarray, shape (n_samples, d)
        Sobol-sampled points in the box
    """
    d = len(box)
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    # Sobol requires sample size to be a power of 2 for best uniformity
    m = int(np.ceil(np.log2(n_samples)))
    sobol_raw = sampler.random_base2(m=m)[:n_samples]  # get exactly n_samples
    X = np.empty((n_samples, d))
    for j, (lo, hi) in enumerate(box):
        X[:, j] = lo + (hi - lo) * sobol_raw[:, j]
    return X


def plot_gp_state(
    X1, X2, X_pred, gp, params, X_train, rays_plot, true_function,
    threshold=4.0, title_prefix="", plot_entropy=True, plot_variance=True,
    ecl_func=None, highlight_point=None, savepath=None, save_pdf=True, save_png=True
):
    """
    Plots the GP mean, true function, training points+arrows, variance, and entropy (ECL)
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 11), sharex=True, sharey=True)
    axes = axes.ravel()

    # Compute rays (for mean/var: can use a fixed direction, e.g., x1)
    d = X_pred.shape[1]
    ray0 = np.zeros((d, 1))
    ray0[0, 0] = 1.0
    rays_pred = [ray0 for _ in range(X_pred.shape[0])]
    rays_pred = np.hstack(rays_pred)

    y_pred, y_var = gp.predict(X_pred, rays_pred, params,
                               calc_cov=True, return_deriv=False)
    y_true = true_function(X_pred).ravel()
    Zp = y_pred.reshape(X1.shape)
    Zt = y_true.reshape(X1.shape)
    Zv = y_var.reshape(X1.shape)

    # For entropy panel
    if plot_entropy:
        if ecl_func is None:
            from utils import ecl_acquisition
            ecl_func = ecl_acquisition
        ecl_vals = ecl_func(y_pred, y_var, threshold=threshold)
        Zecl = ecl_vals.reshape(X1.shape)

    levels = np.linspace(min(Zp.min(), Zt.min()), max(Zp.max(), Zt.max()), 40)
    var_levels = np.linspace(Zv.min(), Zv.max(), 40)
    if plot_entropy:
        ent_levels = np.linspace(Zecl.min(), Zecl.max(), 40)

    # Panel 1: GP mean
    ax_gp = axes[0]
    cf1 = ax_gp.contourf(X1, X2, Zp, levels=levels, cmap="viridis")
    fig.colorbar(cf1, ax=ax_gp, fraction=0.046)
    ax_gp.set_title(f"{title_prefix}GP mean")
    ax_gp.contour(X1, X2, Zp, levels=[threshold],
                  colors="red", linewidths=1.8, linestyles="-")
    ax_gp.contour(X1, X2, Zt, levels=[threshold],
                  colors="black", linewidths=1.8, linestyles="--")
    ax_gp.scatter(X_train[:, 0], X_train[:, 1],
                  c="red", edgecolor="k", s=35, zorder=3)
    for pt, v in zip(X_train, rays_plot):
        clipped_arrow(ax_gp, pt, v.flatten(), color="white")

    # Panel 2: True function
    ax_true = axes[1]
    cf2 = ax_true.contourf(X1, X2, Zt, levels=levels, cmap="viridis")
    fig.colorbar(cf2, ax=ax_true, fraction=0.046)
    ax_true.set_title(f"{title_prefix}True function")
    ax_true.contour(X1, X2, Zt, levels=[threshold],
                    colors="black", linewidths=1.8, linestyles="--")
    ax_true.contour(X1, X2, Zp, levels=[threshold],
                    colors="red", linewidths=1.8, linestyles="-")
    ax_true.scatter(X_train[:, 0], X_train[:, 1],
                    c="red", edgecolor="k", s=35, zorder=3)
    for pt, v in zip(X_train, rays_plot):
        clipped_arrow(ax_true, pt, v.flatten(), color="white")

    # Panel 3: GP variance
    ax_var = axes[2]
    cf3 = ax_var.contourf(X1, X2, Zv, levels=var_levels, cmap="plasma")
    fig.colorbar(cf3, ax=ax_var, fraction=0.046)
    ax_var.set_title(f"{title_prefix}GP variance")
    ax_var.scatter(X_train[:, 0], X_train[:, 1],
                   c="white", edgecolor="k", s=35, zorder=3)
    for pt, v in zip(X_train, rays_plot):
        clipped_arrow(ax_var, pt, v.flatten(), color="black")

    # Panel 4: ECL/entropy
    if plot_entropy:
        ax_ecl = axes[3]
        cf4 = ax_ecl.contourf(X1, X2, Zecl, levels=ent_levels, cmap="inferno")
        fig.colorbar(cf4, ax=ax_ecl, fraction=0.046)
        ax_ecl.set_title(f"{title_prefix}ECL (entropy)")
        ax_ecl.scatter(X_train[:, 0], X_train[:, 1],
                       c="red", edgecolor="k", s=35, zorder=3)
        for pt, v in zip(X_train, rays_plot):
            clipped_arrow(ax_ecl, pt, v.flatten(), color="black")
    else:
        axes[3].axis('off')

    for ax in axes:
        ax.set_xlim([X1.min(), X1.max()])
        ax.set_ylim([X2.min(), X2.max()])
        ax.set_aspect("equal")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")
        # ---- Highlight new point if provided ----
        if highlight_point is not None:
            pt = np.asarray(highlight_point).reshape(-1)
            ax.plot(pt[0], pt[1], marker="*", color="blue",
                    markersize=19, markeredgecolor="white", zorder=5)

    plt.tight_layout()

    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        if save_pdf:
            fig.savefig(savepath + ".pdf", bbox_inches="tight")
        if save_png:
            fig.savefig(savepath + ".png", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
# ------------------------------------------------------------------
# main demo
# ------------------------------------------------------------------


def main():
    plot_dir = "contour_example_ECL_plots"
    n_order = 1
    box = ((-2, 2), (-2, 2))  # variable bounds for optimizer
    bounds = np.array(box)
    lb, ub = bounds[:, 0], bounds[:, 1]
    threshold = 20

    # ----- Initial training set -----
    X_train, y_blocks, rays_list, rays_plot = generate_training_data_lhs(
        n_samples=2,   # start with a small number!
        box=box,
        n_order=n_order,
        max_order=n_order,
        threshold=threshold,
        seed=42)
    rays_array = np.hstack(rays_list)

    # ---- fit DD-GP -----
    gp = gddegp(X_train, y_blocks,
                n_order=n_order,
                rays_array=rays_array,
                normalize=True,
                kernel="SE",
                kernel_type="anisotropic",)
    params = gp.optimize_hyperparameters(
        n_restart_optimizer=10, swarm_size=50, verbose=True)

    # ---- Active Learning Loop ----
    n_active = 16  # number of active samples to add
    previous_params = params
    gx = np.linspace(-2.0, 2.0, 20)
    gy = np.linspace(-2.0, 2.0, 20)
    X1, X2 = np.meshgrid(gx, gy)
    X_pred = np.column_stack([X1.ravel(), X2.ravel()])
    x_next = None
    for al_iter in range(n_active):
        print(f"\nActive Learning Iteration {al_iter+1}/{n_active}")

        # Define maximization of ECL (negative because PSO minimizes)
        def neg_ecl(x, threshold):
            x = np.atleast_2d(x)
            n_points = x.shape[0]
            d = x.shape[1]  # input dimension

            # For [1, 0, ..., 0] in every column:
            ray0 = np.zeros((d, 1))
            ray0[0, 0] = 1.0  # First axis

            # Repeat this ray for each point:
            rays = np.tile(ray0, n_points)  # shape (d, n_points)
            mu, var = gp.predict(
                x, rays, previous_params, calc_cov=True, return_deriv=False)
            # maximize ECL
            return -utils.ecl_acquisition(mu, var, threshold=threshold)

        def neg_ecl_with_thresh(x): return neg_ecl(x, threshold=threshold)
        # Use PSO to maximize ECL over current model

        candidate_points = sobol_points(1000, box, seed=al_iter)
        entropy = -1 * neg_ecl_with_thresh(candidate_points)
        # Indices of 10 largest values (ascending order)
        idx = np.argsort(entropy)[-40:]
        top20_points = candidate_points[idx]     # Select those points

        x_next, fg = utils.pso(
            neg_ecl_with_thresh, lb, ub, swarmsize=40, maxiter=40, debug=True, seed=al_iter*111 + 42, initial_positions=top20_points)
        print(neg_ecl(x_next, threshold=threshold))
        print(fg)
        # Check if x_next is already in X_train (avoid duplicates)
        if np.any(np.all(np.isclose(X_train, x_next, atol=1e-4), axis=1)):
            print("PSO returned a duplicate point. Skipping this iteration.")
            continue

        print(f"Next sample: {x_next}")

        # Evaluate function and directional derivatives at new point
        # ray_next = utils.get_surrogate_gradient_ray(
        #     gp, x_next, previous_params, fallback_axis=0, normalize=True, threshold=threshold)
        ray_next = utils.get_entropy_ridge_direction_nd_2(
            gp, x_next, previous_params, threshold=threshold)
        # ray_next = utils.maximize_ier_direction(
        #     gp, x_next.reshape(1, -1), X_train, y_blocks, rays_array, previous_params, box, threshold=threshold,
        #     n_integration=200, seed=al_iter+42
        # )
        # utils.check_gp_gradient(gp, x_next, previous_params)
        rays_list_next = [ray_next]
        rays_plot_next = [.4*ray_next]
        # Hypercomplex construction for new point
        X_hc_next = oti.array(x_next.reshape(1, -1))
        e_tag = oti.e(1, order=n_order)
        perturb = oti.array(ray_next) * e_tag
        X_hc_next[0, :] += perturb.T
        f_hc_next = true_function(X_hc_next, alg=oti)
        for a, b in itertools.combinations(range(1, ray_next.shape[1] + 1), 2):
            f_hc_next = f_hc_next.truncate((a, b))
        y_blocks_next = [f_hc_next.real.reshape(-1, 1)]
        der_indices = [[[1, i+1]] for i in range(n_order)]
        for idx in der_indices:
            y_blocks_next.append(f_hc_next.get_deriv(idx).reshape(-1, 1))

        # Augment training set
        X_train = np.vstack([X_train, x_next])
        for k in range(len(y_blocks)):
            y_blocks[k] = np.vstack([y_blocks[k], y_blocks_next[k]])
        rays_list.append(rays_list_next[0])
        rays_plot.append(rays_plot_next[0])
        rays_array = np.hstack(rays_list)

        plot_gp_state(
            X1, X2, X_pred, gp, previous_params, X_train, rays_plot, true_function,
            threshold=threshold,
            title_prefix=f"Before iter {al_iter+1}: ",
            plot_entropy=True, plot_variance=True,
            ecl_func=utils.ecl_acquisition,
            highlight_point=x_next,
            savepath=f"{plot_dir}/before_iter_{al_iter+1:02d}"
        )
        # Refit GP (can warm-start with previous params if your optimizer supports it)
        gp = gddegp(X_train, y_blocks,
                    n_order=n_order,
                    rays_array=rays_array,
                    normalize=True,
                    kernel="SE",
                    kernel_type="anisotropic",)
        previous_params = gp.optimize_hyperparameters(
            n_restart_optimizer=20, swarm_size=50, verbose=True, x0=previous_params)
        plot_gp_state(
            X1, X2, X_pred, gp, previous_params, X_train, rays_plot, true_function,
            threshold=threshold,
            title_prefix=f"After iter {al_iter+1}: ",
            plot_entropy=True, plot_variance=True,
            ecl_func=utils.ecl_acquisition,
            highlight_point=x_next,
            savepath=f"{plot_dir}/after_iter_{al_iter+1:02d}"
        )

    # ---- Final Prediction Grid (for plot) -----
    gx = np.linspace(-2.0, 2.0, 30)
    gy = np.linspace(-2.0, 2.0, 30)
    X1, X2 = np.meshgrid(gx, gy)
    X_pred = np.column_stack([X1.ravel(), X2.ravel()])

    ray0 = np.array([[1.0], [0.0]])
    rays_pred = [ray0 for _ in range(X_pred.shape[0])]
    rays_pred = np.hstack(rays_pred)
    y_pred, y_var = gp.predict(X_pred, rays_pred, previous_params,
                               calc_cov=True, return_deriv=False)
    y_true = true_function(X_pred).ravel()

    # ======= Plotting (unchanged) ===========
    threshold = threshold
    fig, (ax_gp, ax_true, ax_var) = plt.subplots(1, 3, figsize=(19, 5),
                                                 sharex=True, sharey=True)
    Zp = y_pred.reshape(X1.shape)
    Zt = y_true.reshape(X1.shape)
    Zv = y_var.reshape(X1.shape)
    levels = np.linspace(min(Zp.min(), Zt.min()), max(Zp.max(), Zt.max()), 40)
    var_levels = np.linspace(Zv.min(), Zv.max(), 40)

    # GP panel
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
    handles = [Line2D([], [], color="red", lw=2, label="GP  f={}".format(threshold)),
               Line2D([], [], color="black", lw=2, ls="--", label="True f=4")]
    ax_gp.legend(handles=handles, loc="upper right")

    # True panel
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

    # Variance panel
    cf3 = ax_var.contourf(X1, X2, Zv, levels=var_levels, cmap="plasma")
    fig.colorbar(cf3, ax=ax_var, fraction=0.046)
    ax_var.set_title("GP variance")
    ax_var.scatter(X_train[:, 0], X_train[:, 1],
                   c="white", edgecolor="k", s=35, zorder=3)
    for pt, v in zip(X_train, rays_plot):
        clipped_arrow(ax_var, pt, v.flatten(), color="black")

    for ax in (ax_gp, ax_true, ax_var):
        ax.set_xlim([-2.0, 2.0])
        ax.set_ylim([-2.0, 2.0])
        ax.set_aspect("equal")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
