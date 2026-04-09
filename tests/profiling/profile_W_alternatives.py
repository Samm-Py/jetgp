"""
Profile alternatives for _sparse_W_and_alpha.

The current bottleneck is:
  K_inv_ord = U @ U.T                          (dense matmul ~85ms)
  K_inv[np.ix_(P,P)] = K_inv_ord               (permute ~22ms)
  W = K_inv - np.outer(alpha, alpha)            (outer + sub ~15ms)
  _project_W_to_phi_space(W, ...)               (projection ~17ms)
  trace(W)                                      (used for noise grad)

Total: ~140ms per call for the W computation path.

Alternatives tested:
  1. scipy.sparse U @ U.T
  2. Direct trace from U: ||U||_F^2 - ||alpha||^2
  3. Project W_proj from U directly (avoid dense K_inv entirely)
"""

import os, sys
import time
import numpy as np
import scipy.sparse as sp
from scipy.stats.qmc import LatinHypercube
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_functions import morris, morris_gradient
from jetgp.full_degp_sparse.degp import degp as SparseDEGP
from jetgp.full_degp_sparse.sparse_cholesky import build_U, nlml_from_U, alpha_from_U

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
DIM = 20
N_TRAIN = 100
DER_INDICES = [[[[i, 1]] for i in range(1, DIM + 1)]]

sampler = LatinHypercube(d=DIM, seed=42)
X_train = sampler.random(n=N_TRAIN)
y_vals = morris(X_train)
grads = morris_gradient(X_train)

y_train_list = [y_vals.reshape(-1, 1)]
for j in range(DIM):
    y_train_list.append(grads[:, j].reshape(-1, 1))

model = SparseDEGP(
    X_train, y_train_list,
    n_order=1, n_bases=DIM,
    der_indices=DER_INDICES,
    normalize=True, kernel='SE', kernel_type='anisotropic',
    rho=1.0, use_supernodes=False,
)

opt = model.optimizer
x0 = np.array([0.1] * (len(model.bounds) - 2) + [0.5, -3.0])
P_full = model.mmd_P_full
N_total = len(P_full)
P_ix = np.ix_(P_full, P_full)

# Build K and U
K, phi, n_bases, oti, diffs = opt._build_K_and_phi(x0)
K_ord = K[P_ix]
y_ord = model.y_train[P_full]

U_buf = np.zeros((N_total, N_total))
U = build_U(K_ord, model.sparse_S_full_arr, N_total,
            block_size=model.n_bases + 1, out=U_buf).copy()

alpha_ord = alpha_from_U(U, y_ord)
alpha_v = np.empty_like(alpha_ord)
alpha_v[P_full] = alpha_ord

# Check U sparsity
nnz_U = np.count_nonzero(U)
print(f"U shape: {U.shape}, nnz: {nnz_U} ({nnz_U/U.size:.1%}), "
      f"sparsity: {1 - nnz_U/U.size:.1%}")

N_ITER = 20

def bench(label, func, n=N_ITER):
    func()  # warm up
    t0 = time.perf_counter()
    for _ in range(n):
        func()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"  {label:55s} {dt:8.2f} ms")
    return dt

# ---------------------------------------------------------------------------
# 1. Dense U @ U.T vs sparse alternatives
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("1. K_inv = U @ U.T alternatives")
print(f"{'='*70}")

bench("Dense: U @ U.T",
      lambda: U @ U.T)

# scipy.sparse CSC
U_csc = sp.csc_matrix(U)
bench("Sparse CSC: U_csc @ U_csc.T → dense",
      lambda: (U_csc @ U_csc.T).toarray())

# scipy.sparse CSR
U_csr = sp.csr_matrix(U)
bench("Sparse CSR: U_csr @ U_csr.T → dense",
      lambda: (U_csr @ U_csr.T).toarray())

# Just the sparse matmul (keep sparse result)
bench("Sparse CSC: U_csc @ U_csc.T (sparse result)",
      lambda: U_csc @ U_csc.T)

# Verify correctness
K_inv_dense = U @ U.T
K_inv_sparse = (U_csc @ U_csc.T).toarray()
print(f"  Max error (sparse vs dense): {np.max(np.abs(K_inv_dense - K_inv_sparse)):.2e}")

# ---------------------------------------------------------------------------
# 2. trace(W) alternatives
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("2. trace(W) = trace(K_inv) - ||alpha||^2")
print(f"{'='*70}")

K_inv_orig = np.empty_like(K_inv_dense)
K_inv_orig[P_ix] = K_inv_dense
W = K_inv_orig - np.outer(alpha_v, alpha_v)
trace_ref = np.trace(W)

bench("Current: trace(W) from precomputed dense W",
      lambda: np.trace(W))

def full_trace_current():
    K_inv_ord = U @ U.T
    K_inv = np.empty_like(K_inv_ord)
    K_inv[P_ix] = K_inv_ord
    W = K_inv - np.outer(alpha_v, alpha_v)
    return np.trace(W)

bench("Full current path: U@U.T → permute → outer → sub → trace",
      full_trace_current)

def trace_from_U_frobenius():
    """trace(K_inv) = ||U||_F^2, then trace(W) = ||U||_F^2 - ||alpha||^2"""
    return np.sum(U * U) - np.dot(alpha_v, alpha_v)

bench("Direct: ||U||_F^2 - ||alpha||^2",
      trace_from_U_frobenius)

# even faster: use the Frobenius norm squared
def trace_from_U_vdot():
    return np.vdot(U, U) - np.dot(alpha_v, alpha_v)

bench("Direct: vdot(U,U) - dot(alpha,alpha)",
      trace_from_U_vdot)

# Only sum nonzero entries
U_flat_nz = U[U != 0]
def trace_from_U_nz():
    return np.dot(U_flat_nz, U_flat_nz) - np.dot(alpha_v, alpha_v)

bench("Direct: dot(U_nz, U_nz) - dot(alpha,alpha)",
      trace_from_U_nz)

print(f"  trace(W) reference:           {trace_ref:.10f}")
print(f"  trace from ||U||_F^2:         {trace_from_U_frobenius():.10f}")
print(f"  Max error:                     {abs(trace_ref - trace_from_U_frobenius()):.2e}")

# ---------------------------------------------------------------------------
# 3. Permutation cost: can we avoid K_inv permutation?
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("3. Permutation alternatives")
print(f"{'='*70}")

bench("K_inv[np.ix_(P,P)] = K_inv_ord (current)",
      lambda: K_inv_orig.__setitem__(P_ix, K_inv_dense))

inv_P = np.empty_like(P_full)
inv_P[P_full] = np.arange(N_total)

bench("K_inv_orig = K_inv_ord[np.ix_(inv_P, inv_P)]",
      lambda: K_inv_dense[np.ix_(inv_P, inv_P)])

# ---------------------------------------------------------------------------
# 4. W_proj directly from U (the big prize)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("4. _project_W_to_phi_space: current vs from-U alternatives")
print(f"{'='*70}")

# Current path: compute full W, then project
from math import comb
from jetgp.full_degp_sparse import degp_utils as utils

deriv_order = 2 * model.n_order
opt._ensure_kernel_plan(n_bases)
plan = opt._kernel_plan
base_shape = (N_total - plan['n_pts_with_derivs'],) * 2
ndir = comb(n_bases + deriv_order, deriv_order)
proj_shape = (ndir, base_shape[0], base_shape[1])
W_proj_ref = np.empty(proj_shape)

row_off = plan.get('row_offsets_abs', plan['row_offsets'] + base_shape[0])
col_off = plan.get('col_offsets_abs', plan['col_offsets'] + base_shape[1])

# warm up numba
utils._project_W_to_phi_space(
    W, W_proj_ref, base_shape[0], base_shape[1],
    plan['fd_flat_indices'], plan['df_flat_indices'],
    plan['dd_flat_indices'],
    plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
    plan['signs'], plan['n_deriv_types'], row_off, col_off,
)

bench("_project_W_to_phi_space (from dense W)",
      lambda: utils._project_W_to_phi_space(
          W, W_proj_ref, base_shape[0], base_shape[1],
          plan['fd_flat_indices'], plan['df_flat_indices'],
          plan['dd_flat_indices'],
          plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
          plan['signs'], plan['n_deriv_types'], row_off, col_off,
      ))

# Full current pipeline for W_proj
W_proj_cur = np.empty(proj_shape)
def current_full_W_proj_pipeline():
    K_inv_ord = U @ U.T
    K_inv = np.empty_like(K_inv_ord)
    K_inv[P_ix] = K_inv_ord
    W_loc = K_inv - np.outer(alpha_v, alpha_v)
    utils._project_W_to_phi_space(
        W_loc, W_proj_cur, base_shape[0], base_shape[1],
        plan['fd_flat_indices'], plan['df_flat_indices'],
        plan['dd_flat_indices'],
        plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
        plan['signs'], plan['n_deriv_types'], row_off, col_off,
    )
    return W_proj_cur

bench("FULL CURRENT: U@U.T → permute → outer → sub → project",
      current_full_W_proj_pipeline)

# Alternative: compute K_inv with sparse matmul, then project
def sparse_W_proj_pipeline():
    K_inv_ord = (U_csc @ U_csc.T).toarray()
    K_inv = np.empty_like(K_inv_ord)
    K_inv[P_ix] = K_inv_ord
    W_loc = K_inv - np.outer(alpha_v, alpha_v)
    utils._project_W_to_phi_space(
        W_loc, W_proj_cur, base_shape[0], base_shape[1],
        plan['fd_flat_indices'], plan['df_flat_indices'],
        plan['dd_flat_indices'],
        plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
        plan['signs'], plan['n_deriv_types'], row_off, col_off,
    )
    return W_proj_cur

bench("ALT sparse: U_csc@U_csc.T → permute → outer → sub → project",
      sparse_W_proj_pipeline)

# Alternative: compute W_proj in MMD order, skip permutation
# We need W_ord = K_inv_ord - outer(alpha_ord, alpha_ord)
# Then project W_ord with a re-mapped plan
# For now, just time the pieces to see if it's worth it.

def mmd_order_W_proj():
    """Compute W in MMD order and project (would need remapped plan)."""
    K_inv_ord = U @ U.T
    W_ord = K_inv_ord - np.outer(alpha_ord, alpha_ord)
    # TODO: would need _project_W_to_phi_space with remapped indices
    return W_ord

bench("Partial: U@U.T → outer → sub (skip permutation, no project)",
      mmd_order_W_proj)

# ---------------------------------------------------------------------------
# 5. Summary of best options
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print("5. Component-level comparison")
print(f"{'='*70}")

bench("U @ U.T (dense BLAS)",           lambda: U @ U.T)
bench("U_csc @ U_csc.T (scipy sparse)", lambda: (U_csc @ U_csc.T).toarray())
bench("K_inv permutation",              lambda: K_inv_dense[np.ix_(inv_P, inv_P)])
bench("np.outer(alpha, alpha)",          lambda: np.outer(alpha_v, alpha_v))
bench("K_inv - outer (subtraction)",     lambda: K_inv_orig - np.outer(alpha_v, alpha_v))
bench("trace from U: vdot(U,U)-dot(a,a)", trace_from_U_vdot)
bench("_project_W_to_phi_space",
      lambda: utils._project_W_to_phi_space(
          W, W_proj_ref, base_shape[0], base_shape[1],
          plan['fd_flat_indices'], plan['df_flat_indices'],
          plan['dd_flat_indices'],
          plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
          plan['signs'], plan['n_deriv_types'], row_off, col_off,
      ))

print("\nDone.")
