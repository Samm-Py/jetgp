"""
Illustrative example for JetGP paper - Section: Illustrative Examples
Demonstrates using derivative-enhanced GPs with adjoint-style derivatives
obtained via automatic differentiation (JAX) through a numerical PDE solver.

Problem: 2D Poisson equation with a Gaussian heat source
  -∇²u = f(x, y; x_s, y_s)   on [0,1]²
  u = 0                        on ∂Ω (Dirichlet BCs)

where f(x, y; x_s, y_s) = A * exp(-((x-x_s)² + (y-y_s)²) / (2σ²))

The quantity of interest (QoI) is the average temperature:
  T_avg(x_s, y_s) = mean(u)

We build a surrogate of T_avg as a function of source location (x_s, y_s).
Derivatives up to second order are obtained via JAX autodiff (jax.grad,
jax.hessian) through the finite difference solver -- mimicking what an
adjoint solver would provide.

Four JetGP models are compared:
  1. Standard GP (no derivatives)
  2. DEGP with first order only  (gradient: dT/dx_s, dT/dy_s)
  3. DEGP with full second order (gradient + full Hessian)
  4. DEGP with selective second order (gradient + diagonal Hessian only,
     no mixed term d²T/dx_s dy_s)
"""

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from jetgp.full_degp.degp import degp

# ── PDE solver ────────────────────────────────────────────────────────────────
def build_laplacian(n):
    """Build 2D Laplacian matrix via finite differences on n x n interior grid."""
    e = jnp.ones(n)
    T = jnp.diag(-2 * e) + jnp.diag(e[:-1], 1) + jnp.diag(e[:-1], -1)
    I = jnp.eye(n)
    return jnp.kron(I, T) + jnp.kron(T, I)

def solve_poisson(source_params, n=30, A=10.0, sigma=0.1):
    """
    Solve -∇²u = f on [0,1]² with homogeneous Dirichlet BCs.

    Parameters
    ----------
    source_params : array [x_s, y_s]
        Location of the Gaussian heat source.
    n : int
        Number of interior grid points per dimension.
    A : float
        Source amplitude.
    sigma : float
        Source width.

    Returns
    -------
    float
        Average temperature mean(u).
    """
    x_s, y_s = source_params[0], source_params[1]
    h = 1.0 / (n + 1)
    x = jnp.linspace(h, 1 - h, n)
    y = jnp.linspace(h, 1 - h, n)
    X, Y = jnp.meshgrid(x, y)
    f = A * jnp.exp(-((X - x_s)**2 + (Y - y_s)**2) / (2 * sigma**2))
    L = build_laplacian(n)
    u = jnp.linalg.solve(-L / h**2, f.flatten())
    return jnp.mean(u)

# JIT compile solver, gradient and Hessian
solve_jit  = jax.jit(solve_poisson)
grad_solve = jax.jit(jax.grad(solve_poisson))
hess_solve = jax.jit(jax.hessian(solve_poisson))

# ── Training data ─────────────────────────────────────────────────────────────
np.random.seed(42)
n_train = 5

X_train = np.random.uniform(0.2, 0.8, (n_train, 2))

print("Computing training function values, gradients and Hessians via JAX autodiff...")
y_vals   = np.array([float(solve_jit(X_train[i]))  for i in range(n_train)])
grads    = np.array([grad_solve(X_train[i])         for i in range(n_train)])
hessians = np.array([hess_solve(X_train[i])         for i in range(n_train)])

# First order
dy_dxs  = grads[:, 0]           # dT_avg/dx_s
dy_dys  = grads[:, 1]           # dT_avg/dy_s

# Second order
d2y_dxs2  = hessians[:, 0, 0]   # d²T_avg/dx_s²
d2y_dys2  = hessians[:, 1, 1]   # d²T_avg/dy_s²
d2y_dxdys = hessians[:, 0, 1]   # d²T_avg/dx_s dy_s (mixed)
print("Done.")

# ── Model 1: Standard GP (no derivatives) ─────────────────────────────────────
y_train_1 = [y_vals.reshape(-1, 1)]
der_indices_1 = []

model_1 = degp(X_train, y_train_1, n_order=0, n_bases=2,
               der_indices=der_indices_1, normalize=True,
               kernel="SE", kernel_type="anisotropic")
params_1 = model_1.optimize_hyperparameters(optimizer='jade',
                                             pop_size=400, n_generations=30,
                                             local_opt_every=15)

# ── Model 2: DEGP first order only ────────────────────────────────────────────
y_train_2 = [
    y_vals.reshape(-1, 1),
    dy_dxs.reshape(-1, 1),
    dy_dys.reshape(-1, 1)
]
der_indices_2 = [
    [[[1, 1]], [[2, 1]]]          # df/dx_s, df/dy_s
]

model_2 = degp(X_train, y_train_2, n_order=1, n_bases=2,
               der_indices=der_indices_2, normalize=True,
               kernel="SE", kernel_type="anisotropic")
params_2 = model_2.optimize_hyperparameters(optimizer='jade',
                                             pop_size=400, n_generations=30,
                                             local_opt_every=15)

# ── Model 3: DEGP full second order (gradient + full Hessian) ─────────────────
y_train_3 = [
    y_vals.reshape(-1, 1),
    dy_dxs.reshape(-1, 1),
    dy_dys.reshape(-1, 1),
    d2y_dxs2.reshape(-1, 1),
    d2y_dxdys.reshape(-1, 1),
    d2y_dys2.reshape(-1, 1)
]
der_indices_3 = [
    [[[1, 1]], [[2, 1]]],                           # df/dx_s, df/dy_s
    [[[1, 2]], [[1, 1], [2, 1]], [[2, 2]]]          # d²f/dx_s², d²f/dx_s dy_s, d²f/dy_s²
]

model_3 = degp(X_train, y_train_3, n_order=2, n_bases=2,
               der_indices=der_indices_3, normalize=True,
               kernel="SE", kernel_type="anisotropic")
params_3 = model_3.optimize_hyperparameters(optimizer='jade',
                                             pop_size=400, n_generations=30,
                                             local_opt_every=15)

# ── Model 4: DEGP selective second order (gradient + diagonal Hessian only) ───
y_train_4 = [
    y_vals.reshape(-1, 1),
    dy_dxs.reshape(-1, 1),
    dy_dys.reshape(-1, 1),
    d2y_dxs2.reshape(-1, 1),
    d2y_dys2.reshape(-1, 1)
]
der_indices_4 = [
    [[[1, 1]], [[2, 1]]],         # df/dx_s, df/dy_s
    [[[1, 2]], [[2, 2]]]          # d²f/dx_s², d²f/dy_s² (no mixed term)
]

model_4 = degp(X_train, y_train_4, n_order=2, n_bases=2,
               der_indices=der_indices_4, normalize=True,
               kernel="SE", kernel_type="anisotropic")
params_4 = model_4.optimize_hyperparameters(optimizer='jade',
                                             pop_size=400, n_generations=30,
                                             local_opt_every=15)

# ── Test data ─────────────────────────────────────────────────────────────────
np.random.seed(7)
n_test = 100
X_test = np.random.uniform(0.2, 0.8, (n_test, 2))

print("Computing test reference values...")
y_test = np.array([float(solve_jit(X_test[i])) for i in range(n_test)])
print("Done.")

y_pred_1, y_var_1 = model_1.predict(X_test, params_1, calc_cov=True)
y_pred_2, y_var_2 = model_2.predict(X_test, params_2, calc_cov=True)
y_pred_3, y_var_3 = model_3.predict(X_test, params_3, calc_cov=True)
y_pred_4, y_var_4 = model_4.predict(X_test, params_4, calc_cov=True)

mu_1 = y_pred_1[0, :];  mu_2 = y_pred_2[0, :]
mu_3 = y_pred_3[0, :];  mu_4 = y_pred_4[0, :]

rmse_1 = np.sqrt(np.mean((mu_1 - y_test)**2))
rmse_2 = np.sqrt(np.mean((mu_2 - y_test)**2))
rmse_3 = np.sqrt(np.mean((mu_3 - y_test)**2))
rmse_4 = np.sqrt(np.mean((mu_4 - y_test)**2))
print(f"RMSE - Standard GP:             {rmse_1:.4f}")
print(f"RMSE - 1st order only:          {rmse_2:.4f}")
print(f"RMSE - Full 2nd order:          {rmse_3:.4f}")
print(f"RMSE - Selective 2nd order:     {rmse_4:.4f}")

# ── Prediction grid ───────────────────────────────────────────────────────────
xs = np.linspace(0.2, 0.8, 40)
ys = np.linspace(0.2, 0.8, 40)
XS, YS = np.meshgrid(xs, ys)
X_grid = np.column_stack([XS.flatten(), YS.flatten()])

print("Computing true T_avg on grid for plotting...")
Z_true = np.array([float(solve_jit(X_grid[i])) for i in range(len(X_grid))]).reshape(40, 40)
Z_1 = model_1.predict(X_grid, params_1, calc_cov=False)[0, :].reshape(40, 40)
Z_2 = model_2.predict(X_grid, params_2, calc_cov=False)[0, :].reshape(40, 40)
Z_3 = model_3.predict(X_grid, params_3, calc_cov=False)[0, :].reshape(40, 40)
Z_4 = model_4.predict(X_grid, params_4, calc_cov=False)[0, :].reshape(40, 40)
Z_err_1 = np.abs(Z_1 - Z_true)
Z_err_2 = np.abs(Z_2 - Z_true)
Z_err_3 = np.abs(Z_3 - Z_true)
Z_err_4 = np.abs(Z_4 - Z_true)
print("Done.")

# ── Plot ──────────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update({'font.size': 12})

def add_colorbar(fig, cp, ax):
    cb = fig.colorbar(cp, ax=ax, shrink=0.8, pad=0.08)
    cb.formatter = ScalarFormatter(useMathText=False)
    cb.formatter.set_powerlimits((-2, 2))
    cb.locator = plt.MaxNLocator(nbins=4)
    cb.update_ticks()
    cb.ax.tick_params(labelsize=10)
    cb.ax.yaxis.get_offset_text().set_fontsize(9)
    return cb

fig, axes = plt.subplots(5, 2, figsize=(10, 13))

vmin, vmax = Z_true.min(), Z_true.max()

# Row 0: Standard GP prediction (left) | Standard GP error (right)
# Row 1: 1st order prediction (left)   | 1st order error (right)
# Row 2: Full 2nd order prediction (left) | Full 2nd order error (right)
# Row 3: Selective 2nd order prediction (left) | Selective 2nd order error (right)
# Row 4: True surface (left) | Predicted vs true scatter (right)

pred_rows = [
    ("Standard GP (no derivatives)",                Z_1, Z_err_1, rmse_1),
    ("1st order: $\\nabla T$",                      Z_2, Z_err_2, rmse_2),
    ("Full 2nd order: $\\nabla T + \\nabla^2 T$",  Z_3, Z_err_3, rmse_3),
    ("Selective 2nd order: $\\nabla T$ + diag($\\nabla^2 T$)", Z_4, Z_err_4, rmse_4),
]

for row, (title, Z_pred, Z_err, rmse_val) in enumerate(pred_rows):
    ax = axes[row, 0]
    cp = ax.contourf(XS, YS, Z_pred, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
    add_colorbar(fig, cp, ax)
    ax.scatter(X_train[:, 0], X_train[:, 1], c='k', s=25, zorder=5)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("$x_s$"); ax.set_ylabel("$y_s$")

    ax = axes[row, 1]
    cp = ax.contourf(XS, YS, Z_err, levels=20, cmap='Reds')
    add_colorbar(fig, cp, ax)
    ax.scatter(X_train[:, 0], X_train[:, 1], c='k', s=25, zorder=5)
    ax.set_title(f"Absolute error  (RMSE = {rmse_val:.4f})", fontsize=12)
    ax.set_xlabel("$x_s$"); ax.set_ylabel("$y_s$")

# Row 4 left: True surface
ax = axes[4, 0]
cp = ax.contourf(XS, YS, Z_true, levels=20, cmap='viridis', vmin=vmin, vmax=vmax)
add_colorbar(fig, cp, ax)
ax.scatter(X_train[:, 0], X_train[:, 1], c='k', s=25, zorder=5)
ax.set_title("True $T_{avg}$", fontsize=12)
ax.set_xlabel("$x_s$"); ax.set_ylabel("$y_s$")

# Row 4 right: Predicted vs true scatter
ax = axes[4, 1]
scatter_handles = []
for mu, c, m, lab in zip(
    [mu_1, mu_2, mu_3, mu_4],
    ['gray', 'steelblue', 'seagreen', 'tomato'],
    ['D', 'o', 's', '^'],
    [f'Standard GP (RMSE={rmse_1:.4f})',
     f'1st order (RMSE={rmse_2:.4f})',
     f'Full 2nd order (RMSE={rmse_3:.4f})',
     f'Selective 2nd order (RMSE={rmse_4:.4f})']
):
    h = ax.scatter(y_test, mu, c=c, s=25, marker=m, label=lab)
    scatter_handles.append(h)

# Add training points to legend
from matplotlib.lines import Line2D
train_handle = Line2D([0], [0], marker='o', color='w', markerfacecolor='k',
                      markersize=6, label=f'Training points ($N={n_train}$)')
scatter_handles.append(train_handle)

lims = [min(y_test.min(), mu_1.min(), mu_2.min(), mu_3.min(), mu_4.min()) * 0.95,
        max(y_test.max(), mu_1.max(), mu_2.max(), mu_3.max(), mu_4.max()) * 1.05]
ax.plot(lims, lims, 'k--', lw=1.0)
ax.set_xlim(lims); ax.set_ylim(lims)
ax.set_xlabel("True $T_{avg}$", fontsize=12)
ax.set_ylabel("Predicted $T_{avg}$", fontsize=12)
ax.set_title("Predicted vs. true", fontsize=12)

# Legend below the figure
fig.legend(handles=scatter_handles,
           loc='lower center', bbox_to_anchor=(0.5, -0.04),
           ncol=3, fontsize=12, frameon=True)

plt.tight_layout()
plt.savefig("adjoint_degp_example.pdf", bbox_inches='tight')
plt.savefig("adjoint_degp_example.png", dpi=150, bbox_inches='tight')
print("Figures saved.")