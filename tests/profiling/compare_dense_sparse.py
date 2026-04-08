"""
Compare dense DEGP vs sparse DEGP NLML evaluation time.
Run with: python tests/profiling/compare_dense_sparse.py
"""

import os, sys, time
import numpy as np
from scipy.stats.qmc import LatinHypercube
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_functions import morris, morris_gradient
from jetgp.full_degp.degp import degp as DenseDEGP
from jetgp.full_degp_sparse.degp import degp as SparseDEGP

DIM = 20
DER_INDICES = [[[[i, 1]] for i in range(1, DIM + 1)]]

def run_comparison(N_TRAIN, rho=1.0, n_evals=50):
    sampler = LatinHypercube(d=DIM, seed=42)
    X_train = sampler.random(n=N_TRAIN)
    y_vals = morris(X_train)
    grads = morris_gradient(X_train)

    y_train_list = [y_vals.reshape(-1, 1)]
    for j in range(DIM):
        y_train_list.append(grads[:, j].reshape(-1, 1))

    # Dense model
    dense_model = DenseDEGP(
        X_train, y_train_list,
        n_order=1, n_bases=DIM,
        der_indices=DER_INDICES,
        normalize=True, kernel='SE', kernel_type='anisotropic',
    )
    dense_opt = dense_model.optimizer
    x0 = np.array([0.1] * (len(dense_model.bounds) - 2) + [0.5, -3.0])

    # Sparse model
    sparse_model = SparseDEGP(
        X_train, y_train_list,
        n_order=1, n_bases=DIM,
        der_indices=DER_INDICES,
        normalize=True, kernel='SE', kernel_type='anisotropic',
        rho=rho, use_supernodes=False,
    )
    sparse_opt = sparse_model.optimizer

    N_total = len(sparse_model.mmd_P_full)
    S = sparse_model.sparse_S_full
    nnz = sum(len(v) for v in S.values())
    max_nnz = N_total * (N_total + 1) // 2
    sparsity = 1 - nnz / max_nnz

    # Warmup
    dense_opt.negative_log_marginal_likelihood(x0)
    sparse_opt.negative_log_marginal_likelihood(x0)

    # Time dense
    t0 = time.perf_counter()
    for _ in range(n_evals):
        dense_opt.negative_log_marginal_likelihood(x0)
    t_dense = (time.perf_counter() - t0) / n_evals

    # Time sparse
    t0 = time.perf_counter()
    for _ in range(n_evals):
        sparse_opt.negative_log_marginal_likelihood(x0)
    t_sparse = (time.perf_counter() - t0) / n_evals

    speedup = t_dense / t_sparse
    print(f"  rho={rho:4.1f}  N={N_TRAIN:3d}  K={N_total:5d}x{N_total:<5d}  "
          f"sparsity={sparsity:5.1%}  "
          f"dense={t_dense*1000:7.1f}ms  "
          f"sparse={t_sparse*1000:7.1f}ms  "
          f"speedup={speedup:.2f}x")

print(f"DIM={DIM}, comparing dense vs sparse NLML eval times")
print(f"{'='*90}")

for rho in [0.5, 1.0, 3.0, 5.0, 10.0]:
    print(f"\n--- rho={rho} ---")
    for N in [20, 50, 100, 150]:
        run_comparison(N, rho=rho)
