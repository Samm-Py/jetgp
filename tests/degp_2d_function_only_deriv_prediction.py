"""DEGP — 2D Function-Only Training with Derivative Predictions
==============================================================
Train the DEGP on function values only (no derivative observations) over a
2D input domain, then predict f, df/dx1, and df/dx2 with uncertainty at
test points.

The model uses n_order=1 and n_bases=2 so the kernel captures first-order
partial derivative structure via OTI arithmetic.  der_indices=[] means no
derivative data enters the training covariance.  At prediction time,
derivs_to_predict=[[[1,1]], [[1,2]]] requests both first-order partials.

True function : f(x1, x2) = sin(x1) * cos(x2)
True partials : df/dx1 = cos(x1)*cos(x2),  df/dx2 = -sin(x1)*sin(x2)
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from jetgp.full_degp.degp import degp


# ---------------------------------------------------------------------------
# True function and partial derivatives
# ---------------------------------------------------------------------------
def f(X):
    """X shape: (n, 2)"""
    return np.sin(X[:, 0]) * np.cos(X[:, 1])

def df_dx1(X):
    return np.cos(X[:, 0]) * np.cos(X[:, 1])

def df_dx2(X):
    return -np.sin(X[:, 0]) * np.sin(X[:, 1])


# ---------------------------------------------------------------------------
# Training data — function values ONLY on a sparse 2D grid
# ---------------------------------------------------------------------------
n_train_per_dim = 6
x1_tr = np.linspace(0, 2 * np.pi, n_train_per_dim)
x2_tr = np.linspace(0, 2 * np.pi, n_train_per_dim)
g1, g2 = np.meshgrid(x1_tr, x2_tr)
X_train = np.column_stack([g1.ravel(), g2.ravel()])   # (25, 2)
y_func  = f(X_train).reshape(-1, 1)
y_train = [y_func]   # no derivative training data


# ---------------------------------------------------------------------------
# Build model
#   n_order=1    : enables first-order OTI arithmetic
#   n_bases=2    : two input dimensions
#   der_indices=[] : no derivatives observed during training
# ---------------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")   # suppress "0 derivatives" notice
    model = degp(
        X_train, y_train,
        n_order=1, n_bases=2,
        der_indices=[],
        normalize=True,
        kernel="SE", kernel_type="anisotropic",
    )


# ---------------------------------------------------------------------------
# Optimise hyperparameters
# ---------------------------------------------------------------------------
params = model.optimize_hyperparameters(
    optimizer="pso",
    pop_size=100,
    n_generations=15,
    local_opt_every=15,
    debug=False,
)
print("Optimised hyperparameters:", params)


# ---------------------------------------------------------------------------
# Predict f, df/dx1, df/dx2 with uncertainty on a dense 2D grid
#   derivs_to_predict=[[[1,1]], [[1,2]]] : df/dx1, df/dx2 (first-order)
#   calc_cov=True                        : also return predictive variance
# ---------------------------------------------------------------------------
n_test_per_dim = 40
x1_te = np.linspace(0, 2 * np.pi, n_test_per_dim)
x2_te = np.linspace(0, 2 * np.pi, n_test_per_dim)
g1t, g2t = np.meshgrid(x1_te, x2_te)
X_test = np.column_stack([g1t.ravel(), g2t.ravel()])   # (1600, 2)

mean, var = model.predict(
    X_test, params,
    calc_cov=True,
    return_deriv=True,
    derivs_to_predict=[[[1, 1]], [[2, 1]]],
)

# mean shape: (3, n_test) — rows: [f, df/dx1, df/dx2]
f_mean    = mean[0, :]
dx1_mean  = mean[1, :]
dx2_mean  = mean[2, :]
f_std     = np.sqrt(np.abs(var[0, :]))
dx1_std   = np.sqrt(np.abs(var[1, :]))
dx2_std   = np.sqrt(np.abs(var[2, :]))

# Reshape for plotting
shape2d = (n_test_per_dim, n_test_per_dim)
f_true_grid    = f(X_test).reshape(shape2d)
dx1_true_grid  = df_dx1(X_test).reshape(shape2d)
dx2_true_grid  = df_dx2(X_test).reshape(shape2d)
f_mean_grid    = f_mean.reshape(shape2d)
dx1_mean_grid  = dx1_mean.reshape(shape2d)
dx2_mean_grid  = dx2_mean.reshape(shape2d)
f_std_grid     = f_std.reshape(shape2d)
dx1_std_grid   = dx1_std.reshape(shape2d)
dx2_std_grid   = dx2_std.reshape(shape2d)


# ---------------------------------------------------------------------------
# Accuracy summary
# ---------------------------------------------------------------------------
f_rmse   = float(np.sqrt(np.mean((f_mean   - f(X_test))    ** 2)))
dx1_rmse = float(np.sqrt(np.mean((dx1_mean - df_dx1(X_test)) ** 2)))
dx2_rmse = float(np.sqrt(np.mean((dx2_mean - df_dx2(X_test)) ** 2)))
dx1_corr = float(np.corrcoef(dx1_mean, df_dx1(X_test))[0, 1])
dx2_corr = float(np.corrcoef(dx2_mean, df_dx2(X_test))[0, 1])

print(f"Function   RMSE : {f_rmse:.4e}")
print(f"df/dx1     RMSE : {dx1_rmse:.4e}   Pearson r: {dx1_corr:.3f}")
print(f"df/dx2     RMSE : {dx2_rmse:.4e}   Pearson r: {dx2_corr:.3f}")


# ---------------------------------------------------------------------------
# Plot — 3 rows (f, df/dx1, df/dx2) x 3 cols (true, GP mean, GP std)
# ---------------------------------------------------------------------------
titles_row = ["f(x1, x2)", "df/dx1", "df/dx2"]
trues      = [f_true_grid, dx1_true_grid, dx2_true_grid]
means      = [f_mean_grid, dx1_mean_grid, dx2_mean_grid]
stds       = [f_std_grid,  dx1_std_grid,  dx2_std_grid]

fig, axes = plt.subplots(3, 3, figsize=(14, 12))
extent = [0, 2 * np.pi, 0, 2 * np.pi]

for row, (label, true, gp_mean, gp_std) in enumerate(
    zip(titles_row, trues, means, stds)
):
    vmin, vmax = true.min(), true.max()

    # True
    im0 = axes[row, 0].imshow(
        true, origin="lower", extent=extent, aspect="auto",
        vmin=vmin, vmax=vmax, cmap="RdBu_r",
    )
    axes[row, 0].set_title(f"True {label}")
    plt.colorbar(im0, ax=axes[row, 0])

    # GP mean
    im1 = axes[row, 1].imshow(
        gp_mean, origin="lower", extent=extent, aspect="auto",
        vmin=vmin, vmax=vmax, cmap="RdBu_r",
    )
    axes[row, 1].set_title(f"GP mean {label}")
    plt.colorbar(im1, ax=axes[row, 1])

    # GP std
    im2 = axes[row, 2].imshow(
        gp_std, origin="lower", extent=extent, aspect="auto",
        cmap="viridis",
    )
    axes[row, 2].set_title(f"GP std {label}")
    plt.colorbar(im2, ax=axes[row, 2])

    for col in range(3):
        axes[row, col].set_xlabel("x1")
        axes[row, col].set_ylabel("x2")

# Overlay training points on the function row
for col in range(2):
    axes[0, col].scatter(
        X_train[:, 0], X_train[:, 1],
        c="k", s=15, zorder=5, label="Training pts",
    )
axes[0, 0].legend(fontsize=7, loc="upper right")

plt.suptitle(
    "DEGP 2D: function-only training, derivative predictions\n"
    "f(x1,x2)=sin(x1)cos(x2)",
    fontsize=13,
)
plt.tight_layout()
plt.savefig("./degp_2d_function_only_deriv_prediction.png", dpi=150)
plt.show()
