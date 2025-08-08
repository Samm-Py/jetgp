import numpy as np
import matplotlib.pyplot as plt
import os
import itertools
from scipy.stats import norm
from full_gddegp.gddegp import gddegp
import pyoti.sparse as oti


from scipy.stats import norm

def plot_pof_panel(
    X1, X2, mu_grid, var_grid, X_train, threshold, y_true_grid, next_point=None, savepath=None, show=True
):
    # Probability of failure grid
    std_grid = np.sqrt(np.maximum(var_grid, 1e-12))
    z_grid = (threshold - mu_grid) / std_grid
    pof_grid = 1 - norm.cdf(z_grid)

    plt.rcParams.update({'font.size': 12})
    fig, ax = plt.subplots(figsize=(7, 5))

    # PoF heatmap
    pf = ax.contourf(X1, X2, pof_grid, levels=30, cmap="Reds", alpha=0.93)
    plt.colorbar(pf, ax=ax, shrink=0.8, label="Probability of Failure")
    # GP threshold contour (where mean = T)
    ax.contour(X1, X2, mu_grid, levels=[threshold], colors='red', linewidths=2, linestyles='-')
    # True threshold contour
    ax.contour(X1, X2, y_true_grid, levels=[threshold], colors='k', linewidths=2, linestyles='--')
    # Samples and next point
    ax.scatter(X_train[:, 0], X_train[:, 1], c="red", edgecolor="k", s=40, zorder=5)
    if next_point is not None:
        ax.scatter([next_point[0]], [next_point[1]], marker='*', color='blue', s=200, edgecolor='white', linewidth=1.7, zorder=10)
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title("Probability of Failure (PoF)\nRed: GP threshold, Black dashed: True threshold")

    # Legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='red', lw=2, label="GP contour"),
        Line2D([0], [0], color='k', lw=2, ls="--", label="True contour"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='k', markersize=10, label='Samples'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markeredgecolor='white', markersize=17, label='Next sample')
    ]
    fig.legend(
        handles=custom_lines,
        labels=["GP contour", "True contour", "Samples", "Next sample"],
        loc='lower center', bbox_to_anchor=(0.5, -0.12), ncol=4, frameon=False, fontsize=14
    )
    plt.subplots_adjust(bottom=0.18)
    plt.tight_layout()

    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=400)
        plt.close(fig)
    elif show:
        plt.show()

def true_function(X, alg=np):
    x, y = X[:, 0], X[:, 1]
    return 3*x**2 + 2*y**2 + x + 2*alg.sin(2*x)*alg.cos(1.5*y)

def generate_training_data_lhs(
    n_samples=16, box=((-2, 2), (-2, 2)), n_order=1, max_order=1, threshold=17.0, seed=None
):
    from scipy.stats import qmc
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    unit = sampler.random(n_samples)
    X_train = np.empty_like(unit)
    for j, (lo, hi) in enumerate(box):
        X_train[:, j] = lo + (hi - lo) * unit[:, j]

    rng = np.random.default_rng(seed)
    rays_list = []
    tag_map = []
    for idx in range(n_samples):
        theta = rng.uniform(0, 2 * np.pi)
        v = np.array([[np.cos(theta)], [np.sin(theta)]])
        rays_list.append(v)
        tag_map.append(idx + 1)

    X_hc = oti.array(X_train)
    for i, tag in enumerate(tag_map):
        e_tag = oti.e(1, order=n_order)
        perturb = (oti.array(rays_list[i]) * e_tag)
        X_hc[i, :] += perturb.T

    f_hc = true_function(X_hc, alg=oti)
    for a, b in itertools.combinations(tag_map, 2):
        f_hc = f_hc.truncate((a, b))

    y_blocks = [f_hc.real.reshape(-1, 1)]
    der_indices = [[[1, i+1]] for i in range(n_order)]
    for idx in der_indices:
        y_blocks.append(f_hc.get_deriv(idx).reshape(-1, 1))

    rays_array = np.hstack(rays_list)
    return X_train, y_blocks, rays_array

def ecl_entropy(mu, var, T):
    std = np.sqrt(np.maximum(var, 1e-10))
    z = (mu - T) / std
    Phi = norm.cdf(z)
    one_minus_Phi = 1 - Phi
    eps = 1e-12
    Phi = np.clip(Phi, eps, 1 - eps)
    one_minus_Phi = np.clip(one_minus_Phi, eps, 1 - eps)
    entropy = - one_minus_Phi * np.log(one_minus_Phi) - Phi * np.log(Phi)
    return entropy

def plot_gp_and_ecl_panels(
    X1, X2, gp, params, X_train, y_blocks, rays_array, threshold, true_function,
    next_point=None, savepath=None, show=True
):
    X_pred = np.column_stack([X1.ravel(), X2.ravel()])
    d = X_pred.shape[1]
    ray0 = np.zeros((d, 1))
    ray0[0, 0] = 1.0
    rays_pred = [ray0 for _ in range(X_pred.shape[0])]
    rays_pred = np.hstack(rays_pred)

    mu, var = gp.predict(X_pred, rays_pred, params, calc_cov=True, return_deriv=False)
    mu_grid = mu.reshape(X1.shape)
    var_grid = var.reshape(X1.shape)
    entropy_grid = ecl_entropy(mu, var, threshold).reshape(X1.shape)
    y_true = true_function(X_pred, alg=np).reshape(X1.shape)

    plt.rcParams.update({'font.size': 12})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # GP mean + contours + points + next point
    levels = np.linspace(mu_grid.min(), mu_grid.max(), 40)
    cf = ax1.contourf(X1, X2, mu_grid, levels=levels, cmap="viridis", alpha=0.93)
    plt.colorbar(cf, ax=ax1, shrink=0.8, label="GP mean")
    ax1.contour(X1, X2, mu_grid, levels=[threshold], colors='red', linewidths=2, linestyles='-')
    ax1.contour(X1, X2, y_true, levels=[threshold], colors='k', linewidths=2, linestyles='--')
    ax1.scatter(X_train[:, 0], X_train[:, 1], c="red", edgecolor="k", s=40, zorder=5)
    if next_point is not None:
        ax1.scatter([next_point[0]], [next_point[1]], marker='*', color='blue', s=200, edgecolor='white', linewidth=1.7, zorder=10)

    ax1.set_xlabel("x₁")
    ax1.set_ylabel("x₂")
    ax1.set_title("GP mean\nRed: GP threshold, Black dashed: True threshold")

    # Entropy (ECL) + both contours + next point
    ent_levels = np.linspace(entropy_grid.min(), entropy_grid.max(), 40)
    cf2 = ax2.contourf(X1, X2, entropy_grid, levels=ent_levels, cmap="inferno", alpha=0.95)
    plt.colorbar(cf2, ax=ax2, shrink=0.8, label="ECL entropy")
    ax2.contour(X1, X2, mu_grid, levels=[threshold], colors='red', linewidths=2, linestyles='-')
    ax2.contour(X1, X2, y_true, levels=[threshold], colors='k', linewidths=2, linestyles='--')
    if next_point is not None:
        ax2.scatter([next_point[0]], [next_point[1]], marker='*', color='blue', s=200, edgecolor='white', linewidth=1.7, zorder=10)
    ax2.set_xlabel("x₁")
    ax2.set_ylabel("x₂")
    ax2.set_title("ECL entropy\nRed: GP threshold, Black dashed: True threshold")

    # Custom legend, positioned in the margin below both plots
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], color='red', lw=2, label="GP contour"),
        Line2D([0], [0], color='k', lw=2, ls="--", label="True contour"),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='k', markersize=10, label='Samples'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='blue', markeredgecolor='white', markersize=17, label='Next sample')
    ]
    fig.legend(handles=custom_lines, labels=["GP contour", "True contour", "Samples", "Next sample"],
               loc='lower center', bbox_to_anchor=(0.5, -0.1), ncol=4, frameon=False, fontsize=14)
    plt.subplots_adjust(bottom=0.15)

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath, bbox_inches="tight", dpi=400)
        plt.close(fig)
    elif show:
        plt.show()

if __name__ == "__main__":
    np.random.seed(123)
    os.makedirs("ECL_summary_plots", exist_ok=True)
    xg = np.linspace(-2, 2, 40)
    yg = np.linspace(-2, 2, 40)
    X1, X2 = np.meshgrid(xg, yg)
    threshold = 5.0

    # --- Training data ---
    X_train, y_blocks, rays_array = generate_training_data_lhs(
        n_samples=15, box=((-2, 2), (-2, 2)), n_order=1, max_order=1, threshold=threshold, seed=42
    )

    # --- Fit DD-GP ---
    gp = gddegp(X_train, y_blocks,
                n_order=1,
                rays_array=rays_array,
                normalize=True,
                kernel="SE",
                kernel_type="anisotropic",)
    params = gp.optimize_hyperparameters(
        n_restart_optimizer=20, swarm_size=50, verbose=True)

    # --- Next point (example) ---
    # Let's just pick the highest ECL entropy as the next point (for demonstration)
    X_pred = np.column_stack([X1.ravel(), X2.ravel()])
    d = X_pred.shape[1]
    ray0 = np.zeros((d, 1))
    ray0[0, 0] = 1.0
    rays_pred = [ray0 for _ in range(X_pred.shape[0])]
    rays_pred = np.hstack(rays_pred)
    mu, var = gp.predict(X_pred, rays_pred, params, calc_cov=True, return_deriv=False)
    entropy_flat = ecl_entropy(mu, var, threshold)
    idx_next = np.argmax(entropy_flat)
    next_point = X_pred[idx_next]

    # --- Plot with blue star ---
    plot_gp_and_ecl_panels(
        X1, X2, gp, params, X_train, y_blocks, rays_array, threshold, true_function,
        next_point=next_point, savepath="ECL_summary_plots/final_ecl_summary.png"
    )

# Build prediction grid
xg = np.linspace(-2, 2, 40)
yg = np.linspace(-2, 2, 40)
X1, X2 = np.meshgrid(xg, yg)
X_pred = np.column_stack([X1.ravel(), X2.ravel()])
d = X_pred.shape[1]
ray0 = np.zeros((d, 1))
ray0[0, 0] = 1.0
rays_pred = [ray0 for _ in range(X_pred.shape[0])]
rays_pred = np.hstack(rays_pred)
mu, var = gp.predict(X_pred, rays_pred, params, calc_cov=True, return_deriv=False)
mu_grid = mu.reshape(X1.shape)
var_grid = var.reshape(X1.shape)
y_true_grid = true_function(X_pred, alg=np).reshape(X1.shape)

# (Optional: select next_point as you already do)
# Plot PoF panel
plot_pof_panel(
    X1, X2, mu_grid, var_grid, X_train, threshold, y_true_grid,
    next_point=next_point,
    savepath="ECL_summary_plots/final_probability_of_failure.png"
)