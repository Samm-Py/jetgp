import numpy as np
import itertools
import pyoti.sparse as oti
from scipy.stats import qmc
import matplotlib.pyplot as plt
from full_gddegp.gddegp import gddegp
import utils
from matplotlib.patches import FancyArrowPatch
from shapely.geometry import LineString, box
import os
# ---- Ishigami function ----


def ishigami(X, alg=np, a=7, b=0.1):
    x1, x2, x3 = X[:, 0], X[:, 1], X[:, 2]
    return alg.sin(x1) + a * alg.sin(x2)**2 + b * (x3**4) * alg.sin(x1)

# ---- LHS points in 3D ----


def lhs_points(n_samples, box, seed=None):
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    unit = sampler.random(n_samples)
    X = np.empty_like(unit)
    for j, (lo, hi) in enumerate(box):
        X[:, j] = lo + (hi - lo) * unit[:, j]
    return X

# ---- Random rays in 3D ----


def rays_random(X, seed=None):
    rng = np.random.default_rng(seed)
    rays, tags, rays_plot = [], [], []
    for idx, _ in enumerate(X):
        v = rng.normal(size=(3, 1))
        v = v / np.linalg.norm(v)
        rays.append(v)
        rays_plot.append(v * 0.6)
        tags.append(idx + 1)
    return rays, tags, rays_plot

# ---- Build training data (f, D_v f, D_v² f...) ----


def generate_training_data_lhs(n_samples=10, box=None, n_order=2, seed=None):
    if box is None:
        box = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    X_train = lhs_points(n_samples, box, seed)
    rays_list, tag_map, rays_plot = rays_random(X_train, seed=seed)
    X_hc = oti.array(X_train)
    for i, tag in enumerate(tag_map):
        e_tag = oti.e(1, order=n_order)
        perturb = oti.array(rays_list[i]) * e_tag
        X_hc[i, :] += perturb.T
    f_hc = ishigami(X_hc, alg=oti)
    for a, b in itertools.combinations(tag_map, 2):
        f_hc = f_hc.truncate((a, b))
    y_blocks = [f_hc.real.reshape(-1, 1)]
    der_indices = [[[1, i+1]] for i in range(n_order)]
    for idx in der_indices:
        y_blocks.append(f_hc.get_deriv(idx).reshape(-1, 1))
    return X_train, y_blocks, rays_list, rays_plot

# ---- Sobol points in 3D ----


def sobol_points(n_samples, box, seed=None):
    d = len(box)
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(n_samples)))
    pts = sampler.random_base2(m=m)[:n_samples]
    for j, (lo, hi) in enumerate(box):
        pts[:, j] = lo + (hi - lo) * pts[:, j]
    return pts

# ---- Arrow plotting helper ----


def clipped_arrow(ax, start, vec, color="white", **kwargs):
    x0, y0 = start
    dx, dy = vec
    x1, y1 = x0 + dx, y0 + dy
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
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

# ---- Plot GP on a 2D slice x3=slice_value ----


def plot_gp_slice(
        gp, params, X_train, rays_plot, x3_slice=np.pi, threshold=3.0,
        title_prefix='', savepath=None, show_train=True, ecl_func=None):
    gx = np.linspace(-np.pi, np.pi, 40)
    gy = np.linspace(-np.pi, np.pi, 40)
    X1, X2 = np.meshgrid(gx, gy)
    X_pred = np.column_stack([X1.ravel(), X2.ravel()])
    X_pred_full = np.c_[X_pred, np.full((X_pred.shape[0], 1), x3_slice)]
    # Predict GP
    ray0 = np.zeros((3, 1))
    ray0[0, 0] = 1.0
    rays_pred = np.hstack([ray0] * X_pred.shape[0])
    y_pred, y_var = gp.predict(
        X_pred_full, rays_pred, params, calc_cov=True, return_deriv=False)
    # True function
    y_true = ishigami(X_pred_full).reshape(X1.shape)
    Zm = y_pred.reshape(X1.shape)
    Zv = y_var.reshape(X1.shape)
    # Entropy/ECL
    if ecl_func is None:
        from utils import ecl_acquisition
        ecl_func = ecl_acquisition
    entropy_vals = ecl_func(y_pred, y_var, threshold=threshold)
    Zecl = entropy_vals.reshape(X1.shape)
    # Levels
    levels = np.linspace(np.min([Zm, y_true]), np.max([Zm, y_true]), 40)
    var_levels = np.linspace(Zv.min(), Zv.max(), 40)
    ent_levels = np.linspace(Zecl.min(), Zecl.max(), 40)
    if not np.all(np.diff(ent_levels) > 0):
        ent_levels = np.linspace(ent_levels[0], ent_levels[0] + 1e-6, 40)
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), sharex=True, sharey=True)
    ax_gp, ax_true, ax_var, ax_ecl = axes.ravel()
    # GP mean
    cf0 = ax_gp.contourf(X1, X2, Zm, levels=levels, cmap="viridis")
    plt.colorbar(cf0, ax=ax_gp, fraction=0.046)
    ax_gp.set_title(f"{title_prefix}GP mean (x3={x3_slice:.2f})")
    # GP mean threshold contour
    ax_gp.contour(X1, X2, Zm, levels=[threshold], colors="red", linewidths=2.2)
    # True function threshold contour
    ax_gp.contour(X1, X2, y_true, levels=[
                  threshold], colors="black", linewidths=2.2, linestyles="--")
    # True function
    cf1 = ax_true.contourf(X1, X2, y_true, levels=levels, cmap="viridis")
    plt.colorbar(cf1, ax=ax_true, fraction=0.046)
    ax_true.set_title(f"Ishigami True (x3={x3_slice:.2f})")
    # True threshold contour
    ax_true.contour(X1, X2, y_true, levels=[
                    threshold], colors="black", linewidths=2.2, linestyles="--")
    # GP threshold contour
    ax_true.contour(X1, X2, Zm, levels=[
                    threshold], colors="red", linewidths=2.2)
    # GP variance
    cf2 = ax_var.contourf(X1, X2, Zv, levels=var_levels, cmap="plasma")
    plt.colorbar(cf2, ax=ax_var, fraction=0.046)
    ax_var.set_title("GP Variance")
    # ECL/entropy
    cf3 = ax_ecl.contourf(X1, X2, Zecl, levels=ent_levels, cmap="inferno")
    plt.colorbar(cf3, ax=ax_ecl, fraction=0.046)
    ax_ecl.set_title("ECL (entropy)")
    for ax in axes.ravel():
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_aspect('equal')
        # Plot all train points
        if show_train and X_train is not None:
            mask = np.abs(X_train[:, 2] - x3_slice) < 1e-2
            ax.scatter(X_train[mask, 0], X_train[mask, 1], c='r',
                       edgecolor='k', s=40, label='Train @ slice', zorder=5)
            ax.scatter(X_train[:, 0], X_train[:, 1], c='w', alpha=0.13, s=14)
    plt.tight_layout()
    if savepath is not None:
        os.makedirs(os.path.dirname(savepath), exist_ok=True)
        fig.savefig(savepath + ".pdf", bbox_inches="tight")
        fig.savefig(savepath + ".png", bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# ---- Main active learning loop ----


def main():
    box = [(-np.pi, np.pi), (-np.pi, np.pi), (-np.pi, np.pi)]
    n_order = 3
    threshold = 16.226
    plot_dir = "ishigami_gp_al_slices"
    x3_slice = np.pi   # Value of x3 for plotting

    # Initial training set
    X_train, y_blocks, rays_list, rays_plot = generate_training_data_lhs(
        n_samples=10, box=box, n_order=n_order, seed=42)
    rays_array = np.hstack(rays_list)

    gp = gddegp(X_train, y_blocks,
                n_order=n_order,
                rays_array=rays_array,
                normalize=True,
                kernel="SE",
                kernel_type="anisotropic")
    params = gp.optimize_hyperparameters(
        n_restart_optimizer=50, swarm_size=100, verbose=True)

    n_active = 20
    previous_params = params

    for al_iter in range(n_active):
        print(f"\nActive Learning Iteration {al_iter+1}/{n_active}")
        # Maximize ECL over entire domain!

        def neg_ecl(x, threshold):
            x = np.atleast_2d(x)
            n_points = x.shape[0]
            d = x.shape[1]
            # For [1,0,0] in each column:
            ray0 = np.zeros((3, 1))
            ray0[0, 0] = 1.0
            rays = np.tile(ray0, n_points)
            mu, var = gp.predict(x, rays, previous_params,
                                 calc_cov=True, return_deriv=False)
            return -utils.ecl_acquisition(mu, var, threshold=threshold)

        def neg_ecl_with_thresh(x): return neg_ecl(x, threshold=threshold)
        candidate_points = sobol_points(2000, box, seed=al_iter)
        entropy = -1 * neg_ecl_with_thresh(candidate_points)
        idx = np.argsort(entropy)[-20:]
        top20_points = candidate_points[idx]
        lb = np.array([b[0] for b in box])
        ub = np.array([b[1] for b in box])
        x_next, fg = utils.pso(
            neg_ecl_with_thresh, lb, ub, swarmsize=20, maxiter=10, minstep=1e-8, minfunc=1e-15, debug=True,
            seed=al_iter*111+42, initial_positions=top20_points)
        print(f"ECL(x_next): {-neg_ecl(x_next, threshold=threshold)}")
        print("x_next:", x_next)
        if np.any(np.all(np.isclose(X_train, x_next, atol=1e-4), axis=1)):
            print("Duplicate point; skipping.")
            continue
        # Choose direction (e.g., x1)
        ray_next = utils.get_entropy_ridge_direction_nd_2(
            gp, x_next, previous_params, threshold=threshold)
        rays_list_next = [ray_next]
        rays_plot_next = [0.6*ray_next]
        # Hypercomplex construction for new point
        X_hc_next = oti.array(x_next.reshape(1, -1))
        e_tag = oti.e(1, order=n_order)
        perturb = oti.array(ray_next) * e_tag
        X_hc_next[0, :] += perturb.T
        f_hc_next = ishigami(X_hc_next, alg=oti)
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
        # Plotting after each iteration on the slice
        plot_gp_slice(
            gp, previous_params, X_train, rays_plot, x3_slice=x3_slice, threshold=threshold,
            title_prefix=f"AL iter {al_iter+1}: ", savepath=f"{plot_dir}/before_iter_{al_iter+1:02d}")
        # Refit GP
        gp = gddegp(X_train, y_blocks,
                    n_order=n_order,
                    rays_array=rays_array,
                    normalize=True,
                    kernel="SE",
                    kernel_type="anisotropic")
        previous_params = gp.optimize_hyperparameters(
            n_restart_optimizer=10, swarm_size=60, verbose=True, x0=previous_params)
        plot_gp_slice(
            gp, previous_params, X_train, rays_plot, x3_slice=x3_slice, threshold=threshold,
            title_prefix=f"AL iter {al_iter+1}: ", savepath=f"{plot_dir}/after_iter_{al_iter+1:02d}")

    # Final plot on the slice
    plot_gp_slice(
        gp, previous_params, X_train, rays_plot, x3_slice=x3_slice, threshold=threshold,
        title_prefix=f"Final model: ", savepath=f"{plot_dir}/before_iter_{al_iter+1:02d}")


if __name__ == "__main__":
    main()
