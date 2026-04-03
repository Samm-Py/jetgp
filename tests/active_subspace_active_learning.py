"""Active Subspace Estimation — Sequential Active Learning
============================================================
Start with a small initial design, iteratively add the point that
maximises the total gradient predictive variance until
D_KL(true || GP) < 1%.

Acquisition: max Σ_j Var[∂f/∂x_j] over a candidate pool.
Because tr(C_hat) = (1/N) Σ_i ||∇f(x_i)||², reducing gradient
variance directly reduces the variance of the eigenvalue sum,
giving tighter active-subspace estimates per sample.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.integrate import trapezoid
from scipy.spatial.distance import cdist
from jetgp.full_degp.degp import degp


# ---------------------------------------------------------------------------
# 1. Test function (same as wycoff_demo)
# ---------------------------------------------------------------------------
np.random.seed(42)

d = 10
N_test = 10000

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
# 2. Fixed evaluation sets
# ---------------------------------------------------------------------------
X_mc = np.random.randn(N_test, d)          # MC points for gradient / KL
y_true_mc = f_full(X_mc)                   # true output for KL reference

N_candidates = 500
X_candidates = np.random.randn(N_candidates, d)  # acquisition candidate pool
y_candidates = f_full(X_candidates)

derivs_to_predict = [[[j + 1, 1]] for j in range(d)]


# ---------------------------------------------------------------------------
# 3. Helper: compute KL divergence between true and GP output PDFs
# ---------------------------------------------------------------------------
def compute_kl(y_true, y_gp):
    clip_lo, clip_hi = np.percentile(y_true, [0.5, 99.5])
    mask_true = (y_true > clip_lo) & (y_true < clip_hi)
    mask_gp   = (y_gp > clip_lo) & (y_gp < clip_hi)
    if mask_gp.sum() < 10:
        return np.inf
    kde_true = gaussian_kde(y_true[mask_true])
    kde_gp   = gaussian_kde(y_gp[mask_gp], bw_method=kde_true.factor)
    y_grid = np.linspace(clip_lo, clip_hi, 5000)
    p = kde_true(y_grid)
    q = kde_gp(y_grid)
    eps = 1e-12
    return float(trapezoid(
        np.clip(p, eps, None) * np.log(np.clip(p, eps, None) / np.clip(q, eps, None)),
        y_grid
    ))


# ---------------------------------------------------------------------------
# 4. Active learning loop
# ---------------------------------------------------------------------------
N_init       = 15
batch_size   = 5
max_iters    = 80
kl_threshold = 0.01

# Initial design
X_train = np.random.randn(N_init, d)
y_train_vals = f_full(X_train)

# Tracking
history_n   = []
history_kl  = []
history_dist = []
history_align = []

for iteration in range(max_iters):
    n_current = len(X_train)
    y_list = [y_train_vals.reshape(-1, 1)]

    # Build GP
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = degp(
            X_train, y_list,
            n_order=1, n_bases=d,
            der_indices=[],
            normalize=True,
            kernel="SE", kernel_type="anisotropic",
        )

    # Optimise — more budget on first iteration, lighter thereafter
    n_gen = 15 if iteration == 0 else 10
    params = model.optimize_hyperparameters(
        optimizer="jade",
        pop_size=40,
        n_generations=n_gen,
        local_opt_every=n_gen,
        debug=False,
    )

    # Predict on MC points (function + gradients + gradient variances)
    mean_mc, var_mc = model.predict(
        X_mc, params,
        calc_cov=True,
        return_deriv=True,
        derivs_to_predict=derivs_to_predict,
    )
    f_pred_mc   = mean_mc[0, :]
    grad_mean   = mean_mc[1:, :]
    grad_var_mc = var_mc[1:, :]          # (d, N_test)

    # Active subspace estimate
    C_hat = (grad_mean @ grad_mean.T) / N_test
    eigvals, eigvecs = np.linalg.eigh(C_hat)
    eigvals  = eigvals[::-1]
    eigvecs  = eigvecs[:, ::-1]
    w_est = eigvecs[:, 0]
    if np.dot(w_est, w_true) < 0:
        w_est = -w_est
    dist  = np.linalg.norm(w_est - w_true)
    align = np.abs(np.dot(w_est, w_true))

    # KL divergence
    kl = compute_kl(y_true_mc, f_pred_mc)

    history_n.append(n_current)
    history_kl.append(kl)
    history_dist.append(dist)
    history_align.append(align)

    print(f"Iter {iteration:3d} | n={n_current:4d} | "
          f"D_KL={kl:.4e} | ||w-w_true||={dist:.4f} | "
          f"|cos(theta)|={align:.4f}")

    if kl < kl_threshold:
        print(f"\n>>> KL = {kl:.4e} < {kl_threshold} — converged at n = {n_current}")
        break

    # Acquisition: maximise reduction in Var[tr(C_hat)] = Var[Σ λ_j]
    # For g_j(x_i) ~ N(μ, σ²), Var[g²] = 4μ²σ² + 2σ⁴.
    # Score each MC point by its contribution to eigenvalue-sum variance,
    # then pick the candidate nearest the highest-scoring MC region.
    eig_var_contribution = (                       # (d, N_test)
        4 * grad_mean**2 * grad_var_mc + 2 * grad_var_mc**2
    )
    mc_scores = eig_var_contribution.sum(axis=0)   # (N_test,)

    # For each candidate, score = sum of MC scores weighted by proximity
    # Use squared-exponential weighting with median-distance bandwidth
    D2 = cdist(X_candidates, X_mc, 'sqeuclidean')        # (N_cand, N_test)
    bw = np.median(D2)
    weights = np.exp(-D2 / bw)                            # (N_cand, N_test)
    cand_scores = weights @ mc_scores                     # (N_cand,)
    top_idx = np.argsort(cand_scores)[-batch_size:]

    # Add selected points to training set
    X_new = X_candidates[top_idx]
    y_new = f_full(X_new)
    X_train = np.vstack([X_train, X_new])
    y_train_vals = np.concatenate([y_train_vals, y_new])

    # Replenish candidate pool
    X_candidates[top_idx] = np.random.randn(batch_size, d)
    y_candidates[top_idx] = f_full(X_candidates[top_idx])

else:
    print(f"\n>>> Reached max iterations ({max_iters}). "
          f"Final KL = {history_kl[-1]:.4e}")

n_final = history_n[-1]


# ---------------------------------------------------------------------------
# 5. Final KDE for plotting
# ---------------------------------------------------------------------------
clip_lo, clip_hi = np.percentile(y_true_mc, [0.5, 99.5])
mask_true = (y_true_mc > clip_lo) & (y_true_mc < clip_hi)
mask_gp   = (f_pred_mc > clip_lo) & (f_pred_mc < clip_hi)
kde_true  = gaussian_kde(y_true_mc[mask_true])
kde_gp    = gaussian_kde(f_pred_mc[mask_gp], bw_method=kde_true.factor)
y_grid_plot = np.linspace(clip_lo, clip_hi, 500)


# ---------------------------------------------------------------------------
# 6. Plot: 2x2 — (a) PDF, (b) bar chart, (c) KL convergence, (d) alignment
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (a) Output PDF
ax = axes[0, 0]
ax.fill_between(y_grid_plot, kde_true(y_grid_plot), alpha=0.35, color='#0C2340',
                label='True $f(\\mathbf{x})$  ($d = 10$)')
ax.plot(y_grid_plot, kde_true(y_grid_plot), color='#0C2340', lw=2.5)
ax.fill_between(y_grid_plot, kde_gp(y_grid_plot), alpha=0.35, color='#F15A22',
                label=f'GP surrogate  ($n = {n_final}$)')
ax.plot(y_grid_plot, kde_gp(y_grid_plot), color='#F15A22', lw=2.5, ls='--')
ax.set_xlabel('Output', fontsize=14)
ax.set_ylabel('Probability Density', fontsize=14)
ax.set_title(f'Output PDF: True vs GP Surrogate\n$D_{{KL}} = {history_kl[-1]:.2e}$',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper right')
ax.tick_params(labelsize=11)

# (b) Active direction bar chart
ax = axes[0, 1]
x_pos = np.arange(1, d + 1)
width = 0.35
ax.bar(x_pos - width/2, w_true, width, color='#0C2340',
       label='True $\\mathbf{w}_1$', edgecolor='white', linewidth=0.5)
ax.bar(x_pos + width/2, w_est, width, color='#F15A22',
       label='Estimated $\\hat{\\mathbf{w}}_1$', edgecolor='white', linewidth=0.5)
ax.set_xlabel('Component Index', fontsize=14)
ax.set_ylabel('Component Value', fontsize=14)
ax.set_title(f'Active Direction Recovery\n'
             f'$\\|\\hat{{\\mathbf{{w}}}}_1 - \\mathbf{{w}}_1\\| = {dist:.4f}$',
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.set_xticks(x_pos)
ax.tick_params(labelsize=11)
ax.axhline(y=0, color='gray', linewidth=0.5, linestyle='-')

# (c) KL divergence convergence
ax = axes[1, 0]
ax.semilogy(history_n, history_kl, 'o-', color='#0C2340', lw=2, markersize=5)
ax.axhline(kl_threshold, color='#F15A22', ls='--', lw=2,
           label=f'Threshold = {kl_threshold}')
ax.set_xlabel('Training set size $n$', fontsize=14)
ax.set_ylabel('$D_{KL}$(true $\\|$ GP)', fontsize=14)
ax.set_title('KL Divergence Convergence', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.tick_params(labelsize=11)
ax.grid(True, alpha=0.3)

# (d) Direction alignment convergence
ax = axes[1, 1]
ax.plot(history_n, history_align, 's-', color='#0C2340', lw=2, markersize=5,
        label='$|\\cos\\theta|$')
ax.plot(history_n, history_dist, 'd-', color='#F15A22', lw=2, markersize=5,
        label='$\\|\\hat{\\mathbf{w}}_1 - \\mathbf{w}_1\\|$')
ax.set_xlabel('Training set size $n$', fontsize=14)
ax.set_ylabel('Metric', fontsize=14)
ax.set_title('Active Direction Convergence', fontsize=16, fontweight='bold')
ax.legend(fontsize=12)
ax.tick_params(labelsize=11)
ax.grid(True, alpha=0.3)

plt.suptitle(
    'Sequential Active Learning of Active Subspaces via DEGP',
    fontsize=15, fontweight='bold', y=1.01,
)
plt.tight_layout()
plt.savefig('wycoff_active_learning.png', dpi=200, bbox_inches='tight')
plt.savefig('wycoff_active_learning.pdf', bbox_inches='tight')
print(f"\nSaved wycoff_active_learning.png and wycoff_active_learning.pdf")
plt.show()
