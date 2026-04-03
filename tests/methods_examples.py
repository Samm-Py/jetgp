"""Methods Section Numerical Examples
=====================================
Generates figures for Section 3 (Methods), starting with the DEGP (Section 3.1).

Demonstrates the benefit of arbitrary-order derivatives by comparing
p=0 (GP), p=1 (GEGP), p=2 (HEGP), and p=3 (DEGP) on the shared test function
f(x1, x2) = sin(x1) * cos(x2) on [0, 2pi]^2.
"""

import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from jetgp.full_degp.degp import degp

# ---------------------------------------------------------------------------
# LaTeX-compatible plot styling
# ---------------------------------------------------------------------------
try:
    matplotlib.rcParams.update({"text.usetex": True})
    _fig, _ax = plt.subplots()
    _ax.set_title(r"$x$")
    _fig.savefig("/dev/null", format="png")
    plt.close(_fig)
    _usetex = True
except Exception:
    _usetex = False
    matplotlib.rcParams.update({"text.usetex": False})

matplotlib.rcParams.update({
    "text.usetex": _usetex,
    "font.family": "serif",
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
    "mathtext.fontset": "cm",
})

np.random.seed(42)

# ---------------------------------------------------------------------------
# True function and ALL derivatives up to order 3
# ---------------------------------------------------------------------------
# f = sin(x1) * cos(x2)
def f(X):       return  np.sin(X[:,0]) * np.cos(X[:,1])
# Order 1
def df_1(X):    return  np.cos(X[:,0]) * np.cos(X[:,1])
def df_2(X):    return -np.sin(X[:,0]) * np.sin(X[:,1])
# Order 2
def d2f_11(X):  return -np.sin(X[:,0]) * np.cos(X[:,1])
def d2f_12(X):  return -np.cos(X[:,0]) * np.sin(X[:,1])
def d2f_22(X):  return -np.sin(X[:,0]) * np.cos(X[:,1])
# Order 3
def d3f_111(X): return -np.cos(X[:,0]) * np.cos(X[:,1])
def d3f_112(X): return  np.sin(X[:,0]) * np.sin(X[:,1])
def d3f_122(X): return -np.cos(X[:,0]) * np.cos(X[:,1])
def d3f_222(X): return  np.sin(X[:,0]) * np.sin(X[:,1])

# ---------------------------------------------------------------------------
# Training and test grids
# ---------------------------------------------------------------------------
n_train_per_dim = 5
n_test_per_dim = 30

x1_tr = np.linspace(0.3, 2 * np.pi - 0.3, n_train_per_dim)
x2_tr = np.linspace(0.3, 2 * np.pi - 0.3, n_train_per_dim)
g1, g2 = np.meshgrid(x1_tr, x2_tr)
X_train = np.column_stack([g1.ravel(), g2.ravel()])
n_train = len(X_train)
all_idx = list(range(n_train))

x1_te = np.linspace(0, 2 * np.pi, n_test_per_dim)
x2_te = np.linspace(0, 2 * np.pi, n_test_per_dim)
g1t, g2t = np.meshgrid(x1_te, x2_te)
X_test = np.column_stack([g1t.ravel(), g2t.ravel()])
shape2d = (n_test_per_dim, n_test_per_dim)
extent = [0, 2 * np.pi, 0, 2 * np.pi]

# Pre-compute training data
y_func   = f(X_train).reshape(-1, 1)
y_d1     = df_1(X_train).reshape(-1, 1)
y_d2     = df_2(X_train).reshape(-1, 1)
y_d11    = d2f_11(X_train).reshape(-1, 1)
y_d12    = d2f_12(X_train).reshape(-1, 1)
y_d22    = d2f_22(X_train).reshape(-1, 1)
y_d111   = d3f_111(X_train).reshape(-1, 1)
y_d112   = d3f_112(X_train).reshape(-1, 1)
y_d122   = d3f_122(X_train).reshape(-1, 1)
y_d222   = d3f_222(X_train).reshape(-1, 1)

# True test values
f_true    = f(X_test).reshape(shape2d)
d1_true   = df_1(X_test).reshape(shape2d)
d2_true   = df_2(X_test).reshape(shape2d)
d11_true  = d2f_11(X_test).reshape(shape2d)
d12_true  = d2f_12(X_test).reshape(shape2d)
d22_true  = d2f_22(X_test).reshape(shape2d)
d111_true = d3f_111(X_test).reshape(shape2d)
d112_true = d3f_112(X_test).reshape(shape2d)
d122_true = d3f_122(X_test).reshape(shape2d)
d222_true = d3f_222(X_test).reshape(shape2d)

# Derivatives to predict (all up to order 3)
derivs_order1 = [[[1,1]], [[2,1]]]
derivs_order2 = [[[1,2]], [[1,1],[2,1]], [[2,2]]]
derivs_order3 = [[[1,3]], [[1,2],[2,1]], [[1,1],[2,2]], [[2,3]]]
all_derivs = derivs_order1 + derivs_order2 + derivs_order3

# All true grids in prediction order: f, d1, d2, d11, d12, d22, d111, d112, d122, d222
all_true_grids = [f_true, d1_true, d2_true, d11_true, d12_true, d22_true,
                  d111_true, d112_true, d122_true, d222_true]

# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def plot_example(title, fname, trues, means, stds, row_labels, train_pts=None):
    """Plot n_rows x 3 grid: True | GP Mean | GP Std."""
    n_rows = len(trues)
    fig, axes = plt.subplots(n_rows, 3, figsize=(14, 3.8 * n_rows + 0.8))
    if n_rows == 1:
        axes = axes[None, :]

    col_headers = [r"True", r"GP Mean", r"GP Std"]

    for row in range(n_rows):
        true = trues[row]
        gp_mean = means[row]
        gp_std = stds[row]
        vmin, vmax = true.min(), true.max()

        im0 = axes[row, 0].imshow(
            true, origin="lower", extent=extent, aspect="auto",
            vmin=vmin, vmax=vmax, cmap="RdBu_r")
        plt.colorbar(im0, ax=axes[row, 0]).ax.tick_params(labelsize=11)

        im1 = axes[row, 1].imshow(
            gp_mean, origin="lower", extent=extent, aspect="auto",
            vmin=vmin, vmax=vmax, cmap="RdBu_r")
        plt.colorbar(im1, ax=axes[row, 1]).ax.tick_params(labelsize=11)

        im2 = axes[row, 2].imshow(
            gp_std, origin="lower", extent=extent, aspect="auto",
            cmap="viridis")
        cb2 = plt.colorbar(im2, ax=axes[row, 2])
        cb2.ax.tick_params(labelsize=11)
        cb2.ax.yaxis.get_offset_text().set_visible(False)
        std_max = gp_std.max()
        if std_max < 1e-4:
            axes[row, 2].text(
                0.97, 0.97, f"max: {std_max:.1e}",
                transform=axes[row, 2].transAxes, fontsize=9,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        axes[row, 0].set_ylabel(row_labels[row], fontsize=14)
        for col in range(3):
            axes[row, col].set_xlabel(r"$x_1$")

        if row == 0:
            for col in range(3):
                axes[row, col].set_title(col_headers[col])

    if train_pts is not None:
        for col in range(2):
            axes[0, col].scatter(
                train_pts[:, 0], train_pts[:, 1],
                c="k", s=25, zorder=5, marker="o")

    plt.suptitle(title, y=1.01)
    plt.tight_layout()
    plt.savefig(f"./{fname}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def compute_rmse(means, trues):
    """Compute RMSE for each row."""
    return [np.sqrt(np.mean((m - t) ** 2)) for m, t in zip(means, trues)]


# ===================================================================
# Train models at each order p = 0, 1, 2, 3
# ===================================================================
results = {}

# --- p = 0: Standard GP ---
print("=" * 60)
print("DEGP p=0 (Standard GP)")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_p0 = degp(
        X_train, [y_func],
        n_order=0, n_bases=2,
        der_indices=None,
        normalize=True, kernel="SE", kernel_type="anisotropic")

params_p0 = model_p0.optimize_hyperparameters(
    optimizer="powell", n_restart_optimizer=20, debug=False)
# p=0 can predict any order via the inference-time perturbation
mean_p0, var_p0 = model_p0.predict(
    X_test, params_p0, calc_cov=True,
    return_deriv=True, derivs_to_predict=all_derivs)
print(f"  Params: {params_p0}")

# --- p = 1: GEGP ---
print("=" * 60)
print("DEGP p=1 (GEGP)")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_p1 = degp(
        X_train, [y_func, y_d1, y_d2],
        n_order=1, n_bases=2,
        der_indices=[[[[1,1]], [[2,1]]]],
        derivative_locations=[all_idx, all_idx],
        normalize=True, kernel="SE", kernel_type="anisotropic")

params_p1 = model_p1.optimize_hyperparameters(
    optimizer="pso", pop_size=100, n_generations=15,
    local_opt_every=15, debug=False)
# p=1 model can only predict up to order 1 derivatives
mean_p1, var_p1 = model_p1.predict(
    X_test, params_p1, calc_cov=True,
    return_deriv=True, derivs_to_predict=derivs_order1)
print(f"  Params: {params_p1}")

# --- p = 2: HEGP ---
print("=" * 60)
print("DEGP p=2 (HEGP)")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_p2 = degp(
        X_train,
        [y_func, y_d1, y_d2, y_d11, y_d12, y_d22],
        n_order=2, n_bases=2,
        der_indices=[
            [[[1,1]], [[2,1]]],
            [[[1,2]], [[1,1],[2,1]], [[2,2]]]
        ],
        derivative_locations=[all_idx]*5,
        normalize=True, kernel="SE", kernel_type="anisotropic")

params_p2 = model_p2.optimize_hyperparameters(
    optimizer="pso", pop_size=100, n_generations=15,
    local_opt_every=15, debug=False)
# p=2 model can predict up to order 2 derivatives
mean_p2, var_p2 = model_p2.predict(
    X_test, params_p2, calc_cov=True,
    return_deriv=True, derivs_to_predict=derivs_order1 + derivs_order2)
print(f"  Params: {params_p2}")

# --- p = 3: DEGP ---
print("=" * 60)
print("DEGP p=3")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_p3 = degp(
        X_train,
        [y_func, y_d1, y_d2, y_d11, y_d12, y_d22,
         y_d111, y_d112, y_d122, y_d222],
        n_order=3, n_bases=2,
        der_indices=[
            [[[1,1]], [[2,1]]],                                    # order 1
            [[[1,2]], [[1,1],[2,1]], [[2,2]]],                     # order 2
            [[[1,3]], [[1,2],[2,1]], [[1,1],[2,2]], [[2,3]]]       # order 3
        ],
        derivative_locations=[all_idx]*9,
        normalize=True, kernel="SE", kernel_type="anisotropic")

params_p3 = model_p3.optimize_hyperparameters(
    optimizer="pso", pop_size=100, n_generations=15,
    local_opt_every=15, debug=False)
mean_p3, var_p3 = model_p3.predict(
    X_test, params_p3, calc_cov=True,
    return_deriv=True, derivs_to_predict=all_derivs)
print(f"  Params: {params_p3}")


# ===================================================================
# Extract grids and compute RMSE for each order
# ===================================================================
def extract_all_grids(mean, var):
    n_out = mean.shape[0]
    means_out = [mean[i, :].reshape(shape2d) for i in range(n_out)]
    stds_out = [np.sqrt(np.abs(var[i, :])).reshape(shape2d) for i in range(n_out)]
    return means_out, stds_out

# p=0 returns: f + all 9 derivs = 10 rows
# p=1 returns: f + 2 order-1 derivs = 3 rows
# p=2 returns: f + 2 order-1 + 3 order-2 = 6 rows
# p=3 returns: f + 2 + 3 + 4 = 10 rows
means_p0, stds_p0 = extract_all_grids(mean_p0, var_p0)
means_p1, stds_p1 = extract_all_grids(mean_p1, var_p1)
means_p2, stds_p2 = extract_all_grids(mean_p2, var_p2)
means_p3, stds_p3 = extract_all_grids(mean_p3, var_p3)

# True grids matching each model's prediction output
trues_p0 = all_true_grids          # f, d1, d2, d11, d12, d22, d111, d112, d122, d222
trues_p1 = [f_true, d1_true, d2_true]
trues_p2 = [f_true, d1_true, d2_true, d11_true, d12_true, d22_true]
trues_p3 = all_true_grids

# ===================================================================
# RMSE table — compare f, df/dx1, df/dx2 across all orders
# ===================================================================
print("\n" + "=" * 80)
print("RMSE Comparison: DEGP with increasing derivative order p")
print("=" * 80)
print(f"{'Order':<8} {'f':>12} {'df/dx1':>12} {'df/dx2':>12}")
print("-" * 48)

# All models predict at least f, df/dx1, df/dx2 (first 3 rows)
for name, ms, ts in [("p=0", means_p0, trues_p0),
                      ("p=1", means_p1, trues_p1),
                      ("p=2", means_p2, trues_p2),
                      ("p=3", means_p3, trues_p3)]:
    rmses = compute_rmse(ms[:3], ts[:3])
    print(f"{name:<8} {rmses[0]:>12.2e} {rmses[1]:>12.2e} {rmses[2]:>12.2e}")


# ===================================================================
# Figure: DEGP p=3 predictions (function + all derivatives)
# ===================================================================
row_labels_full = [
    r"$f$",
    r"$\partial f / \partial x_1$",
    r"$\partial f / \partial x_2$",
    r"$\partial^2 f / \partial x_1^2$",
    r"$\partial^2 f / \partial x_1 \partial x_2$",
    r"$\partial^2 f / \partial x_2^2$",
    r"$\partial^3 f / \partial x_1^3$",
    r"$\partial^3 f / \partial x_1^2 \partial x_2$",
    r"$\partial^3 f / \partial x_1 \partial x_2^2$",
    r"$\partial^3 f / \partial x_2^3$",
]

plot_example(
    r"DEGP ($p=3$): $f$ + all derivatives up to third order",
    "degp_p3_example.pdf",
    all_true_grids, means_p3, stds_p3,
    row_labels_full,
    train_pts=X_train)


# ===================================================================
# Figure: RMSE vs derivative order bar chart
# ===================================================================
orders = [0, 1, 2, 3]
all_rmses = {
    "p=0": compute_rmse(means_p0, all_true_grids),
    "p=1": compute_rmse(means_p1, all_true_grids),
    "p=2": compute_rmse(means_p2, all_true_grids),
    "p=3": compute_rmse(means_p3, all_true_grids),
}

# Bar chart: f, df/dx1, df/dx2 RMSE for each order
fig, ax = plt.subplots(figsize=(8, 5))
x_pos = np.arange(len(orders))
width = 0.22

f_rmses = [all_rmses[f"p={p}"][0] for p in orders]
d1_rmses = [all_rmses[f"p={p}"][1] for p in orders]
d2_rmses = [all_rmses[f"p={p}"][2] for p in orders]

bars1 = ax.bar(x_pos - width, f_rmses, width, label=r"$f$")
bars2 = ax.bar(x_pos, d1_rmses, width, label=r"$\partial f / \partial x_1$")
bars3 = ax.bar(x_pos + width, d2_rmses, width, label=r"$\partial f / \partial x_2$")

ax.set_yscale("log")
ax.set_xticks(x_pos)
ax.set_xticklabels([f"$p={p}$" for p in orders])
ax.set_xlabel(r"Derivative order $p$")
ax.set_ylabel(r"RMSE")
ax.set_title(r"DEGP prediction accuracy vs.\ derivative training order")
ax.legend(loc="upper right")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("./degp_order_comparison.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved degp_order_comparison.pdf")

print("\nDone!")
