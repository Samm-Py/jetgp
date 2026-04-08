"""
test_sparse_degp.py
===================
Compare dense DEGP (full_degp) vs sparse DEGP (full_degp_sparse) on:

  (a) 1-D Griewank — small problem, easy to inspect sparsity effect
  (b) Borehole 8-D  — higher-dimensional benchmark from the test suite

For the sparse model we sweep rho in [1, 2, 3, 5] and report:
  - Optimised NLML
  - RMSE on held-out test points
  - Hyperparameter optimisation time
  - Number of non-zero entries in U (sparsity level)

Both models use the SE anisotropic kernel, first-order derivatives,
JADE optimiser.
"""

import os
import sys
import time
import numpy as np
from scipy.stats.qmc import LatinHypercube

# Make sure the project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import (
    borehole, borehole_gradient,
    generate_test_data, compute_metrics,
)
from jetgp.full_degp.degp import degp as DenseDEGP
from jetgp.full_degp_sparse.degp import degp as SparseDEGP

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1-D Griewank helpers (no external dependency)
# ---------------------------------------------------------------------------

def griewank_1d(X):
    x = X[:, 0]
    return x**2 / 4000 - np.cos(x) + 1

def griewank_1d_grad(X):
    x = X[:, 0]
    return (x / 2000 + np.sin(x)).reshape(-1, 1)

# ---------------------------------------------------------------------------
# Shared optimiser kwargs (kept light for a test script)
# ---------------------------------------------------------------------------

JADE_KWARGS = dict(
    n_generations=30,
    pop_size=10,
    local_opt_every=30,
    debug=False,
)

RHO_VALUES = [1.0, 2.0, 3.0, 5.0, 1000.0]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sparsity_info(model):
    """Return (nnz_U, total, fill_fraction) for the cached U factor."""
    U = getattr(model, '_cached_U', None)
    if U is None:
        return None, None, None
    nnz = int(np.sum(np.abs(U) > 1e-14))
    total = U.size
    return nnz, total, nnz / total


def run_dense(X_train, y_train_list, X_test, y_test, dim, der_indices,
              n_order=1, kernel="SE", kernel_type="anisotropic"):
    model = DenseDEGP(
        X_train, y_train_list,
        n_order=n_order, n_bases=dim,
        der_indices=der_indices,
        normalize=True,
        kernel=kernel, kernel_type=kernel_type,
    )
    t0 = time.perf_counter()
    params = model.optimize_hyperparameters(optimizer="jade", **JADE_KWARGS)
    t_train = time.perf_counter() - t0

    y_pred = model.predict(X_test, params, calc_cov=False, return_deriv=False)
    y_pred = y_pred.flatten()
    metrics = compute_metrics(y_test, y_pred)
    metrics['train_time'] = t_train
    metrics['nlml'] = float(model.opt_nll)
    metrics['label'] = 'dense'
    return metrics


def run_sparse(X_train, y_train_list, X_test, y_test, dim, der_indices,
               rho, n_order=1, kernel="SE", kernel_type="anisotropic"):
    model = SparseDEGP(
        X_train, y_train_list,
        n_order=n_order, n_bases=dim,
        der_indices=der_indices,
        normalize=True,
        kernel=kernel, kernel_type=kernel_type,
        rho=rho,
        use_supernodes=True,
    )
    t0 = time.perf_counter()
    params = model.optimize_hyperparameters(optimizer="jade", **JADE_KWARGS)
    t_train = time.perf_counter() - t0

    y_pred = model.predict(X_test, params, calc_cov=False, return_deriv=False)
    y_pred = y_pred.flatten()
    metrics = compute_metrics(y_test, y_pred)
    metrics['train_time'] = t_train
    metrics['nlml'] = float(model.opt_nll)

    nnz, total, fill = _sparsity_info(model)
    metrics['U_nnz'] = nnz
    metrics['U_total'] = total
    metrics['U_fill'] = fill
    metrics['label'] = f'sparse(rho={rho})'
    return metrics


def print_row(r):
    fill = f"{r['U_fill']*100:.1f}%" if r.get('U_fill') is not None else "  —  "
    nnz  = f"{r['U_nnz']}" if r.get('U_nnz') is not None else "—"
    print(
        f"  {r['label']:<22}  "
        f"NLML={r['nlml']:>10.3f}  "
        f"RMSE={r['rmse']:.4e}  "
        f"NRMSE={r['nrmse']:.4e}  "
        f"t={r['train_time']:.2f}s  "
        f"U_fill={fill} (nnz={nnz})"
    )


# ---------------------------------------------------------------------------
# Pointwise recovery check: evaluate NLL at the same params for dense vs sparse
# ---------------------------------------------------------------------------

def check_nlml_recovery(X_train, y_train_list, dim, der_indices, rho_values,
                        n_order=1, kernel="SE", kernel_type="anisotropic"):
    """
    Build dense and sparse models, optimise the dense model, then evaluate
    both NLML and gradient at the dense-optimal params for each rho.
    This is the definitive check that large rho recovers the dense NLL.
    """
    print("\n" + "=" * 75)
    print("  Recovery check: NLL(params_dense*) for dense vs sparse")
    print("=" * 75)

    dense_model = DenseDEGP(
        X_train, y_train_list,
        n_order=n_order, n_bases=dim,
        der_indices=der_indices,
        normalize=True,
        kernel=kernel, kernel_type=kernel_type,
    )
    dense_params = dense_model.optimize_hyperparameters(optimizer="jade", **JADE_KWARGS)
    dense_nlml_at_opt = dense_model.optimizer.negative_log_marginal_likelihood(dense_params)
    print(f"\n  Dense optimal params : {10**dense_params}")
    print(f"  Dense NLML(params*)  : {dense_nlml_at_opt:.6f}")

    # Dense gradient at params*
    dense_grad = dense_model.optimizer.nll_grad(dense_params)
    print(f"  Dense grad norm      : {np.linalg.norm(dense_grad):.4e}")

    for rho in rho_values:
        sparse_model = SparseDEGP(
            X_train, y_train_list,
            n_order=n_order, n_bases=dim,
            der_indices=der_indices,
            normalize=True,
            kernel=kernel, kernel_type=kernel_type,
            rho=rho,
            use_supernodes=True,
        )
        sparse_nlml = sparse_model.optimizer.negative_log_marginal_likelihood(dense_params)
        sparse_grad = sparse_model.optimizer.nll_grad(dense_params)
        delta_nlml = sparse_nlml - dense_nlml_at_opt
        delta_grad = np.linalg.norm(sparse_grad - dense_grad)
        U = sparse_model._cached_U
        nnz = int(np.sum(np.abs(U) > 1e-14)) if U is not None else 0
        fill = nnz / U.size if U is not None else 0
        print(f"  rho={rho:<8}  sparse NLML={sparse_nlml:.6f}  "
              f"ΔNLML={delta_nlml:+.4e}  Δgrad={delta_grad:.4e}  "
              f"U_fill={fill*100:.1f}%")


# ---------------------------------------------------------------------------
# Problem (a): 1-D Griewank
# ---------------------------------------------------------------------------

def test_griewank(n_train=20, n_test=500):
    print("\n" + "=" * 75)
    print(f"  (a) 1-D Griewank   n_train={n_train}  n_test={n_test}")
    print("=" * 75)

    dim = 1
    der_indices = [[[[1, 1]]]]  # df/dx1, first order

    rng = np.random.default_rng(42)
    X_train = rng.uniform(-np.pi, np.pi, size=(n_train, 1))
    y_vals = griewank_1d(X_train)
    grads  = griewank_1d_grad(X_train)

    y_train_list = [y_vals.reshape(-1, 1), grads]

    rng_test = np.random.default_rng(99)
    X_test = rng_test.uniform(-np.pi, np.pi, size=(n_test, 1))
    y_test = griewank_1d(X_test)

    results = []

    r = run_dense(X_train, y_train_list, X_test, y_test, dim, der_indices)
    results.append(r)
    print_row(r)

    for rho in RHO_VALUES:
        r = run_sparse(X_train, y_train_list, X_test, y_test, dim, der_indices, rho=rho)
        results.append(r)
        print_row(r)

    return results


# ---------------------------------------------------------------------------
# Problem (b): 8-D Borehole
# ---------------------------------------------------------------------------

def test_borehole(n_train=50, n_test=500):
    print("\n" + "=" * 75)
    print(f"  (b) 8-D Borehole   n_train={n_train}  n_test={n_test}")
    print("=" * 75)

    dim = 8
    der_indices = [[[[i, 1]] for i in range(1, dim + 1)]]

    sampler = LatinHypercube(d=dim, seed=42)
    X_train = sampler.random(n=n_train)
    y_vals  = borehole(X_train)
    grads   = borehole_gradient(X_train)

    y_train_list = [y_vals.reshape(-1, 1)]
    for j in range(dim):
        y_train_list.append(grads[:, j].reshape(-1, 1))

    X_test, y_test = generate_test_data(borehole, n_test, dim, seed=99)

    results = []

    r = run_dense(X_train, y_train_list, X_test, y_test, dim, der_indices)
    results.append(r)
    print_row(r)

    for rho in RHO_VALUES:
        r = run_sparse(X_train, y_train_list, X_test, y_test, dim, der_indices, rho=rho)
        results.append(r)
        print_row(r)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Recovery check first
    dim = 1
    der_indices = [[[[1, 1]]]]
    rng = np.random.default_rng(42)
    X_tr = rng.uniform(-np.pi, np.pi, size=(20, 1))
    y_tr = [griewank_1d(X_tr).reshape(-1, 1), griewank_1d_grad(X_tr)]
    check_nlml_recovery(X_tr, y_tr, dim, der_indices,
                        rho_values=[1.0, 3.0, 5.0, 100.0, 1000.0])

    all_results = {}
    all_results['griewank'] = test_griewank(n_train=20, n_test=500)

    print("\n" + "=" * 75)
    print("  Done.")
    print("=" * 75)
