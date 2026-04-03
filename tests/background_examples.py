"""Background Section Numerical Examples
========================================
Generates one figure per GP variant (Sections 2.1.1–2.5) using the shared
test function f(x1, x2) = sin(x1) * cos(x2) on [0, 2pi]^2.

Each figure: n_rows x 3 cols (True, GP Mean, GP Std).
"""

import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from jetgp.full_degp.degp import degp
from jetgp.full_ddegp.ddegp import ddegp
from jetgp.wdegp.wdegp import wdegp

# ---------------------------------------------------------------------------
# LaTeX-compatible plot styling (mathtext fallback if usetex unavailable)
# ---------------------------------------------------------------------------
try:
    matplotlib.rcParams.update({"text.usetex": True})
    # Quick test: create a throwaway figure to check LaTeX works
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
    "mathtext.fontset": "cm",  # Computer Modern when usetex=False
})

np.random.seed(42)

# ---------------------------------------------------------------------------
# True function and derivatives
# ---------------------------------------------------------------------------
def f(X):
    return np.sin(X[:, 0]) * np.cos(X[:, 1])

def df_dx1(X):
    return np.cos(X[:, 0]) * np.cos(X[:, 1])

def df_dx2(X):
    return -np.sin(X[:, 0]) * np.sin(X[:, 1])

def d2f_dx1dx1(X):
    return -np.sin(X[:, 0]) * np.cos(X[:, 1])

def d2f_dx1dx2(X):
    return -np.cos(X[:, 0]) * np.sin(X[:, 1])

def d2f_dx2dx2(X):
    return -np.sin(X[:, 0]) * np.cos(X[:, 1])


# ---------------------------------------------------------------------------
# Training and test grids
# ---------------------------------------------------------------------------
n_train_per_dim = 5
n_test_per_dim = 30

# Offset grid slightly to avoid landing on zero crossings of sin(x1) and sin(x2),
# which causes the standard GP hyperparameter optimizer to learn overly long
# length scales (3 of 5 linspace points fall on zeros of sin).
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
y_func = f(X_train).reshape(-1, 1)
y_dx1 = df_dx1(X_train).reshape(-1, 1)
y_dx2 = df_dx2(X_train).reshape(-1, 1)
y_d2x1x1 = d2f_dx1dx1(X_train).reshape(-1, 1)
y_d2x1x2 = d2f_dx1dx2(X_train).reshape(-1, 1)
y_d2x2x2 = d2f_dx2dx2(X_train).reshape(-1, 1)

# Pre-compute true test values
f_true = f(X_test).reshape(shape2d)
dx1_true = df_dx1(X_test).reshape(shape2d)
dx2_true = df_dx2(X_test).reshape(shape2d)
d2x1x1_true = d2f_dx1dx1(X_test).reshape(shape2d)
d2x1x2_true = d2f_dx1dx2(X_test).reshape(shape2d)
d2x2x2_true = d2f_dx2dx2(X_test).reshape(shape2d)


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def plot_example(title, fname, trues, means, stds, row_labels,
                 train_pts=None, deriv_pts=None):
    """Plot n_rows x 3 grid: True | GP Mean | GP Std.

    Marker logic for the function row (row 0), columns 0–1:
      - If deriv_pts is given and differs from train_pts, show triangles at
        derivative locations and circles only at function-only locations.
      - Otherwise show circles at all training points.
    Legend is placed below the figure.
    """
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
        # If std is near-zero, annotate the max value in the corner
        std_max = gp_std.max()
        if std_max < 1e-4:
            axes[row, 2].text(
                0.97, 0.97, f"max: {std_max:.1e}",
                transform=axes[row, 2].transAxes, fontsize=9,
                va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))

        # Row labels via ylabel on first column
        axes[row, 0].set_ylabel(row_labels[row], fontsize=14)
        for col in range(3):
            axes[row, col].set_xlabel(r"$x_1$")

        # Column headers on the top row only
        if row == 0:
            for col in range(3):
                axes[row, col].set_title(col_headers[col])

    # --- Overlay training / derivative markers on function row ---
    legend_handles = []
    if train_pts is not None:
        if deriv_pts is not None and not np.array_equal(deriv_pts, train_pts):
            # Determine function-only points (those NOT in deriv_pts)
            deriv_set = set(map(tuple, np.round(deriv_pts, 10)))
            fo_mask = np.array([tuple(np.round(p, 10)) not in deriv_set
                                for p in train_pts])
            fo_pts = train_pts[fo_mask]
            for col in range(2):
                h1 = axes[0, col].scatter(
                    fo_pts[:, 0], fo_pts[:, 1],
                    c="k", s=25, zorder=5, marker="o")
                h2 = axes[0, col].scatter(
                    deriv_pts[:, 0], deriv_pts[:, 1],
                    c="red", s=40, zorder=6, marker="^")
            legend_handles = [
                (h1, r"$f$ only"),
                (h2, r"$f + \nabla f$"),
            ]
        else:
            for col in range(2):
                h1 = axes[0, col].scatter(
                    train_pts[:, 0], train_pts[:, 1],
                    c="k", s=25, zorder=5, marker="o")
            legend_handles = [(h1, r"Training pts")]

    plt.suptitle(title, y=1.01)
    plt.tight_layout()

    # Legend below the figure
    if legend_handles:
        fig.legend(
            [h for h, _ in legend_handles],
            [l for _, l in legend_handles],
            loc="lower center", ncol=len(legend_handles),
            bbox_to_anchor=(0.5, -0.02), frameon=True)
        fig.subplots_adjust(bottom=0.06)

    plt.savefig(f"./{fname}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved {fname}")


def train_and_predict(model, derivs_to_predict=None):
    """Optimize hyperparameters and predict with derivatives."""
    params = model.optimize_hyperparameters(
        optimizer="pso", pop_size=100, n_generations=15,
        local_opt_every=15, debug=False)

    if derivs_to_predict is not None:
        mean, var = model.predict(
            X_test, params, calc_cov=True,
            return_deriv=True, derivs_to_predict=derivs_to_predict)
    else:
        mean, var = model.predict(X_test, params, calc_cov=True)
        mean = mean[np.newaxis, :] if mean.ndim == 1 else mean
        var = var[np.newaxis, :] if var.ndim == 1 else var
    return mean, var, params


def extract_grids(mean, var, indices=None):
    """Extract (mean_grid, std_grid) for given row indices."""
    if indices is None:
        indices = list(range(mean.shape[0]))
    means_out, stds_out = [], []
    for i in indices:
        means_out.append(mean[i, :].reshape(shape2d))
        stds_out.append(np.sqrt(np.abs(var[i, :])).reshape(shape2d))
    return means_out, stds_out


# ===================================================================
# Example 0: Standard GP (Section 2) — no derivative information
# ===================================================================
print("=" * 60)
print("Example 0: Standard GP (Section 2)")
print("=" * 60)

# Train on function values only with n_order=0 (standard GP).
# Derivative predictions are obtained at inference time via the kernel
# cross-covariance structure, without any derivative training data.
# Powell with many restarts is more reliable than PSO for this sparse case.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_gp = degp(
        X_train, [y_func],
        n_order=0, n_bases=2,
        der_indices=None,
        normalize=True, kernel="SE", kernel_type="anisotropic")

params_gp = model_gp.optimize_hyperparameters(
    optimizer="powell", n_restart_optimizer=20, debug=False)
print(f"  Params: {params_gp}")

mean_gp, var_gp = model_gp.predict(
    X_test, params_gp, calc_cov=True,
    return_deriv=True, derivs_to_predict=[[[1, 1]], [[2, 1]]])

means_gp, stds_gp = extract_grids(mean_gp, var_gp)
plot_example(
    r"Standard GP: function values only",
    "gp_example.pdf",
    [f_true, dx1_true, dx2_true], means_gp, stds_gp,
    [r"$f(x_1, x_2)$", r"$\partial f / \partial x_1$",
     r"$\partial f / \partial x_2$"],
    train_pts=X_train)


# ===================================================================
# Example 1: GEGP (Section 2.1.1) — full first-order gradients
# ===================================================================
print("=" * 60)
print("Example 1: GEGP (Section 2.1.1)")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_gegp = degp(
        X_train, [y_func, y_dx1, y_dx2],
        n_order=1, n_bases=2,
        der_indices=[[[[1, 1]], [[2, 1]]]],
        derivative_locations=[all_idx, all_idx],
        normalize=True, kernel="SE", kernel_type="anisotropic")

mean, var, params = train_and_predict(
    model_gegp, derivs_to_predict=[[[1, 1]], [[2, 1]]])
print(f"  Params: {params}")

means_g, stds_g = extract_grids(mean, var)
plot_example(
    r"GEGP: $f$ + $\nabla f$ at all training points",
    "gegp_example.pdf",
    [f_true, dx1_true, dx2_true], means_g, stds_g,
    [r"$f(x_1, x_2)$", r"$\partial f / \partial x_1$",
     r"$\partial f / \partial x_2$"],
    train_pts=X_train)


# ===================================================================
# Example 2: HEGP (Section 2.1.2) — up to second-order derivatives
# ===================================================================
print("=" * 60)
print("Example 2: HEGP (Section 2.1.2)")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_hegp = degp(
        X_train,
        [y_func, y_dx1, y_dx2, y_d2x1x1, y_d2x1x2, y_d2x2x2],
        n_order=2, n_bases=2,
        der_indices=[
            [[[1, 1]], [[2, 1]]],                              # first order
            [[[1, 2]], [[1, 1], [2, 1]], [[2, 2]]]             # second order
        ],
        derivative_locations=[
            all_idx, all_idx,       # df/dx1, df/dx2
            all_idx, all_idx, all_idx  # d2f/dx1^2, d2f/dx1dx2, d2f/dx2^2
        ],
        normalize=True, kernel="SE", kernel_type="anisotropic")

mean, var, params = train_and_predict(
    model_hegp, derivs_to_predict=[[[1, 1]], [[2, 1]],
                                   [[1, 2]], [[1, 1], [2, 1]], [[2, 2]]])
print(f"  Params: {params}")

means_h, stds_h = extract_grids(mean, var)
plot_example(
    r"HEGP: $f$ + $\nabla f$ + $\nabla^2 f$ at all training points",
    "hegp_example.pdf",
    [f_true, dx1_true, dx2_true, d2x1x1_true, d2x1x2_true, d2x2x2_true],
    means_h, stds_h,
    [r"$f(x_1, x_2)$", r"$\partial f / \partial x_1$",
     r"$\partial f / \partial x_2$",
     r"$\partial^2 f / \partial x_1^2$",
     r"$\partial^2 f / \partial x_1 \partial x_2$",
     r"$\partial^2 f / \partial x_2^2$"],
    train_pts=X_train)


# ===================================================================
# Example 3: PGEGP (Section 2.2) — partial gradients (dx1 only)
# ===================================================================
print("=" * 60)
print("Example 3: PGEGP (Section 2.2)")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_pgegp = degp(
        X_train, [y_func, y_dx1],
        n_order=1, n_bases=2,
        der_indices=[[[[1, 1]]]],
        derivative_locations=[all_idx],
        normalize=True, kernel="SE", kernel_type="anisotropic")

mean, var, params = train_and_predict(
    model_pgegp, derivs_to_predict=[[[1, 1]], [[2, 1]]])
print(f"  Params: {params}")

means_p, stds_p = extract_grids(mean, var)
plot_example(
    r"PGEGP: $f$ + $\partial f/\partial x_1$ only",
    "pgegp_example.pdf",
    [f_true, dx1_true, dx2_true], means_p, stds_p,
    [r"$f(x_1, x_2)$", r"$\partial f / \partial x_1$",
     r"$\partial f / \partial x_2$"],
    train_pts=X_train)


# ===================================================================
# Example 4: DDGP (Section 2.3) — directional derivatives
# ===================================================================
print("=" * 60)
print("Example 4: DDGP (Section 2.3)")
print("=" * 60)

# Two directions at 45deg and 135deg
angle1, angle2 = np.pi / 4, 3 * np.pi / 4
rays = np.array([
    [np.cos(angle1), np.cos(angle2)],
    [np.sin(angle1), np.sin(angle2)]
])  # shape (2, 2)

# Compute directional derivatives: dv_j f = grad(f) . v_j
grad_x1 = df_dx1(X_train)
grad_x2 = df_dx2(X_train)
y_dir1 = (grad_x1 * rays[0, 0] + grad_x2 * rays[1, 0]).reshape(-1, 1)
y_dir2 = (grad_x1 * rays[0, 1] + grad_x2 * rays[1, 1]).reshape(-1, 1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_ddgp = ddegp(
        X_train, [y_func, y_dir1, y_dir2],
        n_order=1,
        der_indices=[[[[1, 1]], [[2, 1]]]],
        rays=rays,
        derivative_locations=[all_idx, all_idx],
        normalize=True, kernel="SE", kernel_type="anisotropic")

mean, var, params = train_and_predict(
    model_ddgp, derivs_to_predict=[[[1, 1]], [[2, 1]]])
print(f"  Params: {params}")

# True directional derivatives at test points
grad_x1_te = df_dx1(X_test)
grad_x2_te = df_dx2(X_test)
dir1_true = (grad_x1_te * rays[0, 0] + grad_x2_te * rays[1, 0]).reshape(shape2d)
dir2_true = (grad_x1_te * rays[0, 1] + grad_x2_te * rays[1, 1]).reshape(shape2d)

means_d, stds_d = extract_grids(mean, var)
v1_str = rf"$\mathbf{{v}}_1 = ({rays[0,0]:.2f}, {rays[1,0]:.2f})$"
v2_str = rf"$\mathbf{{v}}_2 = ({rays[0,1]:.2f}, {rays[1,1]:.2f})$"
plot_example(
    rf"DDGP: $f$ + directional derivatives at 45° and 135°",
    "ddgp_example.pdf",
    [f_true, dir1_true, dir2_true], means_d, stds_d,
    [r"$f(x_1, x_2)$",
     r"$\partial_{\mathbf{v}_1} f$ (45°)",
     r"$\partial_{\mathbf{v}_2} f$ (135°)"],
    train_pts=X_train)


# ===================================================================
# Example 5: RdGEGP (Section 2.4) — gradients at subset of points
# ===================================================================
print("=" * 60)
print("Example 5: RdGEGP (Section 2.4)")
print("=" * 60)

# Select 10 of 25 points to have gradients (every other + a few extras)
np.random.seed(42)
grad_idx = sorted(np.random.choice(n_train, size=10, replace=False).tolist())
y_dx1_sub = df_dx1(X_train[grad_idx]).reshape(-1, 1)
y_dx2_sub = df_dx2(X_train[grad_idx]).reshape(-1, 1)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_rdgegp = degp(
        X_train, [y_func, y_dx1_sub, y_dx2_sub],
        n_order=1, n_bases=2,
        der_indices=[[[[1, 1]], [[2, 1]]]],
        derivative_locations=[grad_idx, grad_idx],
        normalize=True, kernel="SE", kernel_type="anisotropic")

mean, var, params = train_and_predict(
    model_rdgegp, derivs_to_predict=[[[1, 1]], [[2, 1]]])
print(f"  Params: {params}")

means_r, stds_r = extract_grids(mean, var)
plot_example(
    rf"RdGEGP: $f$ at all {n_train} pts + $\nabla f$ at {len(grad_idx)} pts",
    "rdgegp_example.pdf",
    [f_true, dx1_true, dx2_true], means_r, stds_r,
    [r"$f(x_1, x_2)$", r"$\partial f / \partial x_1$",
     r"$\partial f / \partial x_2$"],
    train_pts=X_train, deriv_pts=X_train[grad_idx])


# ===================================================================
# Example 6: WGEGP (Section 2.5) — weighted submodels
# ===================================================================
print("=" * 60)
print("Example 6: WGEGP (Section 2.5)")
print("=" * 60)

# Split gradient points into 2 disjoint submodels
np.random.seed(42)
perm = np.random.permutation(n_train)
sm1_idx = sorted(perm[:n_train // 2].tolist())
sm2_idx = sorted(perm[n_train // 2:].tolist())

y_dx1_sm1 = df_dx1(X_train[sm1_idx]).reshape(-1, 1)
y_dx2_sm1 = df_dx2(X_train[sm1_idx]).reshape(-1, 1)
y_dx1_sm2 = df_dx1(X_train[sm2_idx]).reshape(-1, 1)
y_dx2_sm2 = df_dx2(X_train[sm2_idx]).reshape(-1, 1)

y_train_w = [
    [y_func, y_dx1_sm1, y_dx2_sm1],
    [y_func, y_dx1_sm2, y_dx2_sm2],
]
der_indices_w = [
    [[[[1, 1]], [[2, 1]]]],
    [[[[1, 1]], [[2, 1]]]],
]
deriv_locs_w = [
    [sm1_idx, sm1_idx],
    [sm2_idx, sm2_idx],
]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_wgegp = wdegp(
        X_train, y_train_w,
        n_order=1, n_bases=2,
        der_indices=der_indices_w,
        derivative_locations=deriv_locs_w,
        submodel_type='degp',
        normalize=True, kernel="SE", kernel_type="anisotropic")

params = model_wgegp.optimize_hyperparameters(
    optimizer="pso", pop_size=100, n_generations=15,
    local_opt_every=15, debug=False)
print(f"  Params: {params}")

mean_w, var_w = model_wgegp.predict(
    X_test, params, calc_cov=True, return_deriv=True,
    derivs_to_predict=[[[1, 1]], [[2, 1]]])

means_w, stds_w = extract_grids(mean_w, var_w)
plot_example(
    rf"WGEGP: $f$ at all pts + $\nabla f$ split into 2 submodels",
    "wgegp_example.pdf",
    [f_true, dx1_true, dx2_true], means_w, stds_w,
    [r"$f(x_1, x_2)$", r"$\partial f / \partial x_1$",
     r"$\partial f / \partial x_2$"],
    train_pts=X_train)


# ===================================================================
# Summary RMSE table and comparison figure
# ===================================================================
print("\n" + "=" * 60)
print("RMSE Summary")
print("=" * 60)
print(f"{'Method':<12} {'f RMSE':>12} {'df/dx1 RMSE':>14} {'df/dx2 RMSE':>14}")
print("-" * 56)

names = ["GP", "GEGP", "HEGP", "PGEGP", "DDGP", "RdGEGP", "WGEGP"]
all_means = [means_gp, means_g, means_h, means_p, means_d, means_r, means_w]
f_rmses, d1_rmses, d2_rmses = [], [], []

for name, ms in zip(names, all_means):
    f_rmse = np.sqrt(np.mean((ms[0] - f_true) ** 2))
    if name == "DDGP":
        d1_rmse = np.sqrt(np.mean((ms[1] - dir1_true) ** 2))
        d2_rmse = np.sqrt(np.mean((ms[2] - dir2_true) ** 2))
    else:
        d1_rmse = np.sqrt(np.mean((ms[1] - dx1_true) ** 2))
        d2_rmse = np.sqrt(np.mean((ms[2] - dx2_true) ** 2))
    f_rmses.append(f_rmse)
    d1_rmses.append(d1_rmse)
    d2_rmses.append(d2_rmse)
    d1_str = f"{d1_rmse:>14.4e}" if not np.isnan(d1_rmse) else f"{'N/A':>14}"
    d2_str = f"{d2_rmse:>14.4e}" if not np.isnan(d2_rmse) else f"{'N/A':>14}"
    print(f"{name:<12} {f_rmse:>12.4e} {d1_str} {d2_str}")

# --- Summary bar chart ---
x_pos = np.arange(len(names))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x_pos - width, f_rmses, width, label=r"$f$")
bars2 = ax.bar(x_pos, d1_rmses, width, label=r"$\partial f / \partial x_1$ (or $\partial_{\mathbf{v}_1} f$)")
bars3 = ax.bar(x_pos + width, d2_rmses, width, label=r"$\partial f / \partial x_2$ (or $\partial_{\mathbf{v}_2} f$)")

ax.set_yscale("log")
ax.set_xticks(x_pos)
ax.set_xticklabels(names)
ax.set_ylabel(r"RMSE")
ax.set_title(r"Prediction accuracy comparison across GP variants")
ax.legend(loc="upper left")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig("./summary_rmse.pdf", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved summary_rmse.pdf")

print("\nDone!")
