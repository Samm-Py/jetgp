import numpy as np
import pyoti.sparse as oti  # Hyper-complex AD
import itertools
from full_gddegp.gddegp import gddegp
from scipy.stats import qmc
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

plt.rcParams.update({'font.size': 12})

# Branin function constants
pi = np.pi
a = 1.0
b = 5.1 / (4*pi**2)
c = 5.0 / pi
r = 6.0
s = 10.0
t = 1.0 / (8*pi)

def true_function(X, alg=np):
    x, y = X[:, 0], X[:, 1]
    return (y - b*x**2 + c*x - r)**2 + s*(1 - t)*alg.cos(x) + s

def grad_branin(x, y):
    gx = 2*(y - b*x**2 + c*x - r)*(-2*b*x + c) - s*(1 - t)*np.sin(x)
    gy = 2*(y - b*x**2 + c*x - r)
    return gx, gy

def gradient_angles_lhs(n_samples=25, seed=None):
    sampler = qmc.LatinHypercube(d=2, seed=1)
    unit = sampler.random(n=n_samples)
    X = np.empty_like(unit)
    X[:, 0] = -5.0 + (10.0 - (-5.0)) * unit[:, 0]
    X[:, 1] = 0.0 + (15.0 - 0.0) * unit[:, 1]
    thetas = []
    for x, y in X:
        gx, gy = grad_branin(x, y)
        theta = np.arctan2(gy, gx)
        thetas.append(theta)
    thetas = np.array(thetas)
    return X, thetas, np.degrees(thetas)

def generate_pointwise_rays(n_order):
    X_train, thetas, _ = gradient_angles_lhs(n_samples=15)
    rays_list, tag_map = [], []
    for idx, theta in enumerate(thetas):
        ray = np.array([[np.cos(theta)], [np.sin(theta)]])
        tag = idx + 1
        rays_list.append(ray)
        tag_map.append(tag)
    return X_train, rays_list, thetas, tag_map

def apply_pointwise_perturb(X_train, rays_list, tag_map, n_order):
    X_hc = oti.array(X_train)
    for i, (ray, tag) in enumerate(zip(rays_list, tag_map)):
        e_tag = oti.e(1, order=n_order)
        perturb = (oti.array(ray) * e_tag)
        X_hc[i, :] += perturb.T
    return X_hc

def generate_training_data(n_order=2, num_points=3, max_order=2):
    X_train, rays_list, thetas, tag_map = generate_pointwise_rays(n_order)
    X_pert = apply_pointwise_perturb(X_train, rays_list, tag_map, n_order)
    f_hc = true_function(X_pert, alg=oti)
    for combo in itertools.combinations(tag_map, 2):
        f_hc = f_hc.truncate(combo)
    y_blocks = [f_hc.real.reshape(-1, 1)]
    der_indices = [[[1, 1]], [[1,2]]]
    for idx in der_indices:
        y_blocks.append(f_hc.get_deriv(idx).reshape(-1, 1))
    return X_train, thetas, y_blocks, rays_list

def clipped_arrow(ax, origin, direction, length, bounds, color="black"):
    x0, y0 = origin
    dx, dy = direction * length
    xlim, ylim = bounds
    tx = np.inf if dx == 0 else (
        xlim[1] - x0)/dx if dx > 0 else (xlim[0] - x0)/dx
    ty = np.inf if dy == 0 else (
        ylim[1] - y0)/dy if dy > 0 else (ylim[0] - y0)/dy
    t = min(1.0, tx, ty)
    dx *= t
    dy *= t
    ax.arrow(x0, y0, dx, dy,
             head_width=0.25, head_length=0.35,
             fc=color, ec=color, clip_on=True)

def main():
    np.random.seed(0)
    n_order = 2
    X_train, thetas, y_train, rays_list = generate_training_data(n_order)
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
        n_restart_optimizer=15, swarm_size=200, verbose=True)
    gx = np.linspace(-5, 10, 200)
    gy = np.linspace(0, 15, 200)
    X1, X2 = np.meshgrid(gx, gy)
    X_pred = np.column_stack([X1.ravel(), X2.ravel()])

    theta = np.pi/2
    ray = np.array([[np.cos(theta)], [np.sin(theta)]])
    N = X_pred.shape[0]
    rays_pred = [ray for i in range(N)]
    rays_pred = np.hstack(rays_pred)
    y_pred = gp.predict(X_pred, rays_pred, params, calc_cov=False, return_deriv=True).ravel()
    y_pred_func = y_pred[:N]
    y_true_func = true_function(X_pred, alg=np).ravel()
    abs_err_func = np.abs(y_pred_func - y_true_func).reshape(X1.shape)
    abs_err_func_clip = np.clip(abs_err_func, 1e-6, None)
    log_levels_func = np.logspace(np.log10(abs_err_func_clip.min()), 1, 200)

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))

    # A. GP Prediction Contour Plot
    cf1 = axs[0].contourf(X1, X2, y_pred_func.reshape(X1.shape), levels=30, cmap='viridis')
    axs[0].scatter(X_train[:, 0], X_train[:, 1], c='red', s=40, label='Train Points')
    xlim = axs[0].get_xlim()
    ylim = axs[0].get_ylim()
    for pt, ray_ in zip(X_train, rays_list):
        clipped_arrow(axs[0], pt, ray_.flatten(), length=0.75, bounds=(xlim, ylim), color="black")
    axs[0].set_title("GP Prediction")
    fig.colorbar(cf1, ax=axs[0])
    axs[0].set_xlim(gx[0], gx[-1])
    axs[0].set_ylim(gy[0], gy[-1])

    # B. True Function Contour Plot
    cf2 = axs[1].contourf(X1, X2, y_true_func.reshape(X1.shape), levels=30, cmap='viridis')
    axs[1].set_title("True Function")
    fig.colorbar(cf2, ax=axs[1])

    # C. Absolute error (log scale)
    cf3 = axs[2].contourf(X1, X2, abs_err_func_clip, levels=log_levels_func,
                          norm=LogNorm(vmin=abs_err_func_clip.min(), vmax=abs_err_func_clip.max()), cmap="magma_r")
    fig.colorbar(cf3, ax=axs[2], format="%.1e")
    axs[2].set_title("Absolute error (log scale)")

    for ax in axs:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")

    # Legend: black line for "ray", red circle for train points
    # Legend: black line for "ray", red circle for train points
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markeredgecolor='k', markersize=10, label='Train Points'),
        Line2D([0], [0], color='black', lw=2, label='Ray (direction)'),
    ]

    # Add legend *before* adjusting layout
    fig.legend(
        handles=custom_lines,
        loc='lower center',
        ncol=2,
        frameon=False,
        fontsize=12,
    )
    plt.savefig("branin_gp_prediction.png", dpi=600, bbox_inches='tight')

    plt.subplots_adjust(bottom=0.18)  # Enough space for legend
    # Do NOT call plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
