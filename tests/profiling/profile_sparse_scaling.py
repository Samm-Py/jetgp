"""
Profile sparse vs dense DEGP scaling across dimensions.

Tests 2D, 5D, and 10D with 1st-order derivatives.
For each dimension:
  - Measures NLML evaluation time (single call)
  - Measures full optimisation wall time
  - Reports prediction RMSE
  - Reports sparsity statistics and K-matrix size

Uses a sum-of-sines test function:  f(x) = sum_d sin(2*pi*x_d)
which has analytic gradients and works in any dimension.
"""

import os, sys, time
import numpy as np
from scipy.stats.qmc import LatinHypercube
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from jetgp.full_degp.degp import degp as DenseDEGP
from jetgp.full_degp_sparse.degp import degp as SparseDEGP


# =============================================================================
# Test function: sum of sines (works in any dimension)
# =============================================================================

def sum_of_sines(X):
    """f(x) = sum_d sin(2*pi*x_d),  X in [0,1]^d"""
    return np.sum(np.sin(2 * np.pi * X), axis=1)


def sum_of_sines_gradient(X):
    """df/dx_d = 2*pi*cos(2*pi*x_d),  returns (n, d)"""
    return 2 * np.pi * np.cos(2 * np.pi * X)


# =============================================================================
# Helpers
# =============================================================================

JADE_KWARGS = dict(
    n_generations=50,
    pop_size=15,
    local_opt_every=51,
    debug=False,
)


def build_y_train(X, func, grad_func):
    """Build y_train list: [f_vals, df/dx1, df/dx2, ...]"""
    y_vals = func(X).reshape(-1, 1)
    grads = grad_func(X)
    dim = X.shape[1]
    y_list = [y_vals]
    for d in range(dim):
        y_list.append(grads[:, d].reshape(-1, 1))
    return y_list


def make_der_indices(dim):
    """1st-order derivative indices for dim dimensions."""
    return [[[[d + 1, 1]] for d in range(dim)]]


def time_single_nlml(model, x0):
    """Time a single NLML evaluation (warm-up + timed)."""
    model.optimizer.negative_log_marginal_likelihood(x0)  # warm-up
    t0 = time.perf_counter()
    n_calls = 5
    for _ in range(n_calls):
        model.optimizer.negative_log_marginal_likelihood(x0)
    return (time.perf_counter() - t0) / n_calls


def sparsity_info(model):
    """Return sparsity stats from the model."""
    S = model.sparse_S
    N_phys = model.num_points
    nnz_phys = sum(len(v) for v in S.values())
    max_phys = N_phys * (N_phys + 1) // 2
    sp_phys = 1 - nnz_phys / max_phys

    S_full = model.sparse_S_full
    N_total = len(model.mmd_P_full)
    nnz_full = sum(len(v) for v in S_full.values())
    max_full = N_total * (N_total + 1) // 2
    sp_full = 1 - nnz_full / max_full

    n_supernodes = len(model.sparse_supernodes) if model.sparse_supernodes else N_phys
    return {
        'N_phys': N_phys, 'N_total': N_total,
        'K_size': f'{N_total}x{N_total}',
        'sp_phys': sp_phys, 'sp_full': sp_full,
        'n_supernodes': n_supernodes,
    }


# =============================================================================
# Run one configuration
# =============================================================================

def run_config(dim, n_train, rho, n_test=200):
    """Run dense and sparse DEGP for one (dim, n_train, rho) config."""

    der_indices = make_der_indices(dim)

    # Generate data
    sampler = LatinHypercube(d=dim, seed=42)
    X_train = sampler.random(n=n_train)
    y_train = build_y_train(X_train, sum_of_sines, sum_of_sines_gradient)

    sampler_test = LatinHypercube(d=dim, seed=99)
    X_test = sampler_test.random(n=n_test)
    y_test = sum_of_sines(X_test)

    N_total = n_train * (1 + dim)  # function + dim derivatives per point

    # --- Dense ---
    dense = DenseDEGP(
        X_train, y_train, n_order=1, n_bases=dim,
        der_indices=der_indices, normalize=True,
        kernel='SE', kernel_type='anisotropic',
    )

    # Time single NLML
    x0 = np.zeros(dim + 2)  # log10 length scales + log_sf + log_sn
    x0[-1] = -3.0
    t_nlml_dense = time_single_nlml(dense, x0)

    # Full optimisation
    t0 = time.perf_counter()
    dp = dense.optimize_hyperparameters(optimizer='jade', **JADE_KWARGS)
    t_opt_dense = time.perf_counter() - t0

    y_pred_d = dense.predict(X_test, dp).flatten()
    rmse_d = np.sqrt(np.mean((y_pred_d - y_test) ** 2))

    # --- Sparse ---
    sparse = SparseDEGP(
        X_train, y_train, n_order=1, n_bases=dim,
        der_indices=der_indices, normalize=True,
        kernel='SE', kernel_type='anisotropic',
        rho=rho, use_supernodes=False,
    )

    sp_info = sparsity_info(sparse)

    t_nlml_sparse = time_single_nlml(sparse, x0)

    t0 = time.perf_counter()
    sp_params = sparse.optimize_hyperparameters(optimizer='jade', **JADE_KWARGS)
    t_opt_sparse = time.perf_counter() - t0

    y_pred_s = sparse.predict(X_test, sp_params).flatten()
    rmse_s = np.sqrt(np.mean((y_pred_s - y_test) ** 2))

    return {
        'dim': dim, 'n_train': n_train, 'rho': rho,
        'N_total': N_total, 'K_size': sp_info['K_size'],
        'sp_phys': sp_info['sp_phys'], 'sp_full': sp_info['sp_full'],
        'n_supernodes': sp_info['n_supernodes'],
        't_nlml_dense': t_nlml_dense, 't_nlml_sparse': t_nlml_sparse,
        't_opt_dense': t_opt_dense, 't_opt_sparse': t_opt_sparse,
        'rmse_dense': rmse_d, 'rmse_sparse': rmse_s,
        'nlml_dense': float(dense.opt_nll), 'nlml_sparse': float(sparse.opt_nll),
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    configs = [
        # (dim, n_train, rho)
        # K size = n_train * (1 + dim)
        (2,  100, 1),   # K: 90x90
        (5,  100, 1),   # K: 120x120
        (10, 100, 1),   # K: 165x165
    ]

    print("=" * 110)
    print("  Sparse vs Dense DEGP: Scaling with dimension (1st-order derivatives, SE anisotropic)")
    print("  Test function: f(x) = sum_d sin(2*pi*x_d)")
    print("=" * 110)

    header = (
        f"{'dim':>4} {'N':>4} {'rho':>4} │ {'K size':>10} {'sp%':>6} "
        f"│ {'t_nlml_d':>9} {'t_nlml_s':>9} {'speedup':>8} "
        f"│ {'t_opt_d':>8} {'t_opt_s':>8} {'speedup':>8} "
        f"│ {'RMSE_d':>10} {'RMSE_s':>10}"
    )
    print(header)
    print("─" * 110)

    for dim, n_train, rho in configs:
        r = run_config(dim, n_train, rho)

        nlml_speedup = r['t_nlml_dense'] / max(r['t_nlml_sparse'], 1e-9)
        opt_speedup = r['t_opt_dense'] / max(r['t_opt_sparse'], 1e-9)

        print(
            f"{r['dim']:>4} {r['n_train']:>4} {r['rho']:>4} │ "
            f"{r['K_size']:>10} {r['sp_full']:>5.0%} "
            f"│ {r['t_nlml_dense']:>8.4f}s {r['t_nlml_sparse']:>8.4f}s {nlml_speedup:>7.2f}x "
            f"│ {r['t_opt_dense']:>7.2f}s {r['t_opt_sparse']:>7.2f}s {opt_speedup:>7.2f}x "
            f"│ {r['rmse_dense']:>10.2e} {r['rmse_sparse']:>10.2e}"
        )

    print("─" * 110)
    print("  sp% = sparsity of full (derivative-expanded) pattern")
    print("  t_nlml = time for single NLML evaluation; t_opt = full JADE optimisation")
    print("  speedup = dense_time / sparse_time")
