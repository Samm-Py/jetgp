"""Active Subspace Estimation via DEGP Derivative Predictions
==============================================================
Implementation of the strategy from:
  Wycoff, Binois & Wild — "Sequential Learning of Active Subspaces"

Method:
  1. Build a GP on function values ONLY (no gradient observations)
  2. Predict all partial derivatives at a set of points using the
     kernel cross-covariance (DEGP with der_indices=[])
  3. Estimate the uncentered gradient covariance matrix
       C = (1/M) sum_i  nabla_f(x_i) nabla_f(x_i)^T
  4. Eigendecompose C — the leading eigenvector(s) span the active subspace

Test function (10-D with decaying cross-directions):
  f(x) = z1^3 + 0.5*z1 + sum_j decay_j * z_j^2
  where Z = X @ Q_true (random orthogonal rotation), X ~ N(0, I).
  The dominant active direction is Q_true[:, 0].
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from jetgp.full_degp.degp import degp


# ---------------------------------------------------------------------------
# 1. Define the test function with a dominant 1-D active subspace
# ---------------------------------------------------------------------------
np.random.seed(42)

d = 10
N_train = 250
N_test = 10000

# Random orthogonal basis — the first column is the dominant active direction
Q_true, _ = np.linalg.qr(np.random.randn(d, d))
w_true = Q_true[:, 0]
decay_coefficients = np.array(
    [0.0, 0.08, 0.04, 0.02, 0.01, 0.005, 0.003, 0.002, 0.001, 0.0005]
)


def f_full(X):
    Z = X @ Q_true
    result = Z[:, 0] ** 3 + 0.5 * Z[:, 0]
    for j in range(1, d):
        result += decay_coefficients[j] * Z[:, j] ** 2
    return result


# ---------------------------------------------------------------------------
# 2. Generate training data — N(0,1) inputs, function values ONLY
# ---------------------------------------------------------------------------
X_train = np.random.randn(N_train, d)
y_func  = f_full(X_train).reshape(-1, 1)
y_train = [y_func]  # no derivative data


# ---------------------------------------------------------------------------
# 3. Build DEGP — function values only, but n_order=1 for derivative kernels
# ---------------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = degp(
        X_train, y_train,
        n_order=1, n_bases=d,
        der_indices=[],          # NO derivative observations
        normalize=True,
        kernel="SE", kernel_type="anisotropic",
    )


# ---------------------------------------------------------------------------
# 4. Optimise hyperparameters (JADE optimizer)
# ---------------------------------------------------------------------------
params = model.optimize_hyperparameters(
    optimizer="jade",
    pop_size=40,
    n_generations=15,
    local_opt_every=15,
    debug=True,
)
print("Optimised hyperparameters:", params)


# ---------------------------------------------------------------------------
# 5. Predict gradients at N(0,1) Monte Carlo points
# ---------------------------------------------------------------------------
X_mc = np.random.randn(N_test, d)

# Each entry [[j, 1]] means first-order partial w.r.t. dimension j
derivs_to_predict = [[[j + 1, 1]] for j in range(d)]

mean = model.predict(
    X_mc, params,
    calc_cov=False,
    return_deriv=True,
    derivs_to_predict=derivs_to_predict,
)

# mean shape: (d+1, N_test) — row 0 is f, rows 1..d are df/dx_j
f_pred    = mean[0, :]        # GP function predictions
grad_mean = mean[1:, :]       # (d, N_test) gradient predictions
print(f"\nGradient predictions shape: {grad_mean.shape}")


# ---------------------------------------------------------------------------
# 6. Estimate the active subspace matrix  C = (1/M) Σ ∇f ∇f^T
# ---------------------------------------------------------------------------
C_hat = (grad_mean @ grad_mean.T) / N_test   # (d, d)

eigenvalues, eigenvectors = np.linalg.eigh(C_hat)
# eigh returns ascending order — flip to descending
eigenvalues  = eigenvalues[::-1]
eigenvectors = eigenvectors[:, ::-1]

# Sign-align estimated direction with true direction
w_est = eigenvectors[:, 0]
if np.dot(w_est, w_true) < 0:
    w_est = -w_est

dist = np.linalg.norm(w_est - w_true)
alignment = np.abs(np.dot(w_est, w_true))
print(f"\n--- Active Subspace Estimation ---")
print(f"Eigenvalues of C:  {eigenvalues}")
print(f"Eigenvalue ratio (lambda1 / lambda2): {eigenvalues[0] / eigenvalues[1]:.1f}")
print(f"||w-hat - w_true|| = {dist:.6f}")
print(f"Alignment |cos(theta)| = {alignment:.4f}  (1.0 = perfect)")


# ---------------------------------------------------------------------------
# 7. Compute output PDFs via KDE and KL divergence
# ---------------------------------------------------------------------------
y_true = f_full(X_mc)           # analytic function values
y_gp   = f_pred                 # GP predictions

clip_lo, clip_hi = np.percentile(y_true, [0.5, 99.5])
mask_true = (y_true > clip_lo) & (y_true < clip_hi)
mask_gp   = (y_gp > clip_lo) & (y_gp < clip_hi)

kde_true = gaussian_kde(y_true[mask_true])
kde_gp   = gaussian_kde(y_gp[mask_gp], bw_method=kde_true.factor)

y_grid = np.linspace(clip_lo, clip_hi, 5000)
p = kde_true(y_grid)
q = kde_gp(y_grid)

eps = 1e-12
p_safe = np.clip(p, eps, None)
q_safe = np.clip(q, eps, None)
kl_div = trapezoid(p_safe * np.log(p_safe / q_safe), y_grid)
print(f"KL divergence D_KL(true || GP) = {kl_div:.2e}")


# ---------------------------------------------------------------------------
# 8. Plot: (a) PDF comparison, (b) active direction bar chart
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# (a) Output PDF: analytic vs GP
ax1 = axes[0]
y_grid_plot = np.linspace(clip_lo, clip_hi, 500)

ax1.fill_between(y_grid_plot, kde_true(y_grid_plot), alpha=0.35, color='#0C2340',
                 label='True $f(\\mathbf{x})$  ($d = 10$)')
ax1.plot(y_grid_plot, kde_true(y_grid_plot), color='#0C2340', lw=2.5)
ax1.fill_between(y_grid_plot, kde_gp(y_grid_plot), alpha=0.35, color='#F15A22',
                 label=f'GP surrogate  ($n = {N_train}$)')
ax1.plot(y_grid_plot, kde_gp(y_grid_plot), color='#F15A22', lw=2.5, linestyle='--')

ax1.set_xlabel('Output', fontsize=14)
ax1.set_ylabel('Probability Density', fontsize=14)
ax1.set_title(f'Output PDF: True vs GP Surrogate\n$D_{{KL}} = {kl_div:.2e}$',
              fontsize=16, fontweight='bold')
ax1.legend(fontsize=12, loc='upper right')
ax1.tick_params(labelsize=11)

# (b) Grouped bar chart: true w1 vs estimated w1
ax2 = axes[1]
x_pos = np.arange(1, d + 1)
width = 0.35

ax2.bar(x_pos - width/2, w_true, width, color='#0C2340',
        label='True $\\mathbf{w}_1$', edgecolor='white', linewidth=0.5)
ax2.bar(x_pos + width/2, w_est, width, color='#F15A22',
        label='Estimated $\\hat{\\mathbf{w}}_1$', edgecolor='white', linewidth=0.5)

ax2.set_xlabel('Component Index', fontsize=14)
ax2.set_ylabel('Component Value', fontsize=14)
ax2.set_title(f'Active Direction Recovery\n'
              f'$\\|\\hat{{\\mathbf{{w}}}}_1 - \\mathbf{{w}}_1\\| = {dist:.4f}$',
              fontsize=16, fontweight='bold')
ax2.legend(fontsize=12)
ax2.set_xticks(x_pos)
ax2.tick_params(labelsize=11)
ax2.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')

plt.tight_layout()
plt.savefig('wycoff_demo.png', dpi=200, bbox_inches='tight')
plt.savefig('wycoff_demo.pdf', bbox_inches='tight')
print(f"\nSaved wycoff_demo.png and wycoff_demo.pdf")
plt.show()