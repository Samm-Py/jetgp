"""
Benchmark _permute_and_subtract_outer variants to identify bottleneck.
python tests/profiling/bench_permute_outer.py
"""
import os, sys, time
import numpy as np
import numba

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# ── build realistic data ──────────────────────────────────────────────
from scipy.stats.qmc import LatinHypercube
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from benchmark_functions import morris, morris_gradient
from jetgp.full_degp_sparse.degp import degp as SparseDEGP

DIM = 20; N_TRAIN = 100
X_train = LatinHypercube(d=DIM, seed=42).random(n=N_TRAIN)
y_vals = morris(X_train)
grads = morris_gradient(X_train)
y_train_list = [y_vals.reshape(-1, 1)] + [grads[:, j].reshape(-1, 1) for j in range(DIM)]
DER_INDICES = [[[[i, 1]] for i in range(1, DIM + 1)]]

model = SparseDEGP(
    X_train, y_train_list, n_order=1, n_bases=DIM,
    der_indices=DER_INDICES, normalize=True,
    kernel='SE', kernel_type='anisotropic', rho=1.0, use_supernodes=False,
)
opt = model.optimizer
x0 = np.array([0.1] * (len(model.bounds) - 2) + [0.5, -3.0])
opt.negative_log_marginal_likelihood(x0)  # warm up

# Get U and alpha_v from a real call
from jetgp.full_degp_sparse.optimizer import _permute_and_subtract_outer
alpha_v, U, nll, *_ = opt._sparse_nlml_direct(x0)
P_full = model.mmd_P_full
N_total = len(P_full)

# Compute K_inv_ord via dsyrk (lower triangle)
from scipy.linalg import blas
K_inv_buf = np.empty((N_total, N_total), order='F')
K_inv_ord = blas.dsyrk(1.0, U, lower=1, c=K_inv_buf, overwrite_c=1)

W = np.empty((N_total, N_total))

# ── Warm up numba ────────────────────────────────────────────────────
_permute_and_subtract_outer(K_inv_ord, alpha_v, P_full, W)

# ── Variant 1: current fused kernel ──────────────────────────────────
N_ITER = 50
t0 = time.perf_counter()
for _ in range(N_ITER):
    _permute_and_subtract_outer(K_inv_ord, alpha_v, P_full, W)
t1 = time.perf_counter()
print(f"Current fused kernel:        {(t1-t0)/N_ITER*1e3:8.2f} ms")

# ── Variant 2: read-only (no writes to W) ────────────────────────────
@numba.jit(nopython=True, parallel=True)
def _read_only(K_inv_ord, alpha_v, P_full, W):
    """Same reads, but accumulate into a dummy scalar to prevent dead-code elim."""
    N = len(P_full)
    for i in numba.prange(N):
        pi = P_full[i]
        ai = alpha_v[pi]
        s = K_inv_ord[i, i] - ai * ai
        for j in range(i + 1, N):
            pj = P_full[j]
            s += K_inv_ord[j, i] - ai * alpha_v[pj]
        W[pi, pi] = s  # single write per row

_read_only(K_inv_ord, alpha_v, P_full, W)  # warm up
t0 = time.perf_counter()
for _ in range(N_ITER):
    _read_only(K_inv_ord, alpha_v, P_full, W)
t1 = time.perf_counter()
print(f"Read-only (no scatter):      {(t1-t0)/N_ITER*1e3:8.2f} ms")

# ── Variant 3: write-only (no K_inv reads) ───────────────────────────
@numba.jit(nopython=True, parallel=True)
def _write_only(K_inv_ord, alpha_v, P_full, W):
    """Only scattered writes, no K_inv reads."""
    N = len(P_full)
    for i in numba.prange(N):
        pi = P_full[i]
        ai = alpha_v[pi]
        W[pi, pi] = -ai * ai
        for j in range(i + 1, N):
            pj = P_full[j]
            val = -ai * alpha_v[pj]
            W[pi, pj] = val
            W[pj, pi] = val

_write_only(K_inv_ord, alpha_v, P_full, W)  # warm up
t0 = time.perf_counter()
for _ in range(N_ITER):
    _write_only(K_inv_ord, alpha_v, P_full, W)
t1 = time.perf_counter()
print(f"Write-only (no K_inv read):  {(t1-t0)/N_ITER*1e3:8.2f} ms")

# ── Variant 4: sequential (no prange) ────────────────────────────────
@numba.jit(nopython=True, parallel=False)
def _sequential(K_inv_ord, alpha_v, P_full, W):
    N = len(P_full)
    for i in range(N):
        pi = P_full[i]
        ai = alpha_v[pi]
        W[pi, pi] = K_inv_ord[i, i] - ai * ai
        for j in range(i + 1, N):
            pj = P_full[j]
            val = K_inv_ord[j, i] - ai * alpha_v[pj]
            W[pi, pj] = val
            W[pj, pi] = val

_sequential(K_inv_ord, alpha_v, P_full, W)  # warm up
t0 = time.perf_counter()
for _ in range(N_ITER):
    _sequential(K_inv_ord, alpha_v, P_full, W)
t1 = time.perf_counter()
print(f"Sequential (no prange):      {(t1-t0)/N_ITER*1e3:8.2f} ms")

# ── Variant 5: identity permutation (contiguous writes) ──────────────
P_identity = np.arange(N_total, dtype=P_full.dtype)

_permute_and_subtract_outer(K_inv_ord, alpha_v, P_identity, W)  # warm up
t0 = time.perf_counter()
for _ in range(N_ITER):
    _permute_and_subtract_outer(K_inv_ord, alpha_v, P_identity, W)
t1 = time.perf_counter()
print(f"Identity perm (contiguous):  {(t1-t0)/N_ITER*1e3:8.2f} ms")

print(f"\nN_total = {N_total}")
