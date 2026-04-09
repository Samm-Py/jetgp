"""
Profile _sparse_W_from_U — the W = K^{-1} - αα^T computation from sparse U.
"""

import os, sys, time, cProfile, pstats
import numpy as np
from scipy.stats.qmc import LatinHypercube
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_functions import morris, morris_gradient
from jetgp.full_degp_sparse.degp import degp as SparseDEGP

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

# Warm up — runs _sparse_nlml_direct which builds the plan, index maps, etc.
opt.negative_log_marginal_likelihood(x0)
# Run nll_and_grad once to init the sparse structure cache in _sparse_W_from_U
opt.nll_and_grad(x0)

# Get a fresh U and alpha_v for isolated benchmarking
alpha_v, U, nll, phi, n_bases, oti, diffs = opt._sparse_nlml_direct(x0)

P_full = model.mmd_P_full
N_total = len(P_full)
print(f"K size: {N_total}x{N_total}")
print(f"U nnz: {np.count_nonzero(U)} ({np.count_nonzero(U)/U.size:.1%})")
print()

# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
N_ITER = 50

def bench(label, func, n=N_ITER):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000
    print(f"  {label:55s}  mean={times.mean():7.2f}  median={np.median(times):7.2f}  min={times.min():7.2f} ms")
    return np.median(times)

print(f"{'='*75}")
print(f"Overall: _sparse_W_from_U ({N_ITER} calls)")
print(f"{'='*75}")

bench("_sparse_W_from_U(U, alpha_v)",
      lambda: opt._sparse_W_from_U(U, alpha_v))

# ---------------------------------------------------------------------------
# cProfile
# ---------------------------------------------------------------------------
N_PROF = 30
print(f"\n{'='*75}")
print(f"cProfile top-down ({N_PROF} calls)")
print(f"{'='*75}")

pr = cProfile.Profile()
pr.enable()
for _ in range(N_PROF):
    opt._sparse_W_from_U(U, alpha_v)
pr.disable()
pstats.Stats(pr).strip_dirs().sort_stats('tottime').print_stats(25)

# ---------------------------------------------------------------------------
# Component isolation
# ---------------------------------------------------------------------------
import scipy.sparse as sp

rows, cols = opt._U_sparse_structure
coo_order = opt._coo_to_csc_order
U_csc = opt._U_csc_template

print(f"{'='*75}")
print(f"Component isolation ({N_ITER} calls)")
print(f"{'='*75}")

# 1. vals extraction
bench("U[rows, cols] (extract vals from dense U)",
      lambda: U[rows, cols])

# 2. trace computation
vals = U[rows, cols]
bench("dot(vals,vals) - dot(alpha,alpha) (trace_W)",
      lambda: np.dot(vals, vals) - np.dot(alpha_v, alpha_v))

# 3. CSC data update
bench("vals[coo_order] (reorder for CSC)",
      lambda: vals[coo_order])

bench("U_csc.data[:] = ... (CSC data update)",
      lambda: U_csc.data.__setitem__(slice(None), vals[coo_order]))

# 4. Sparse matmul
bench("U_csc @ U_csc.T (sparse matmul)",
      lambda: U_csc @ U_csc.T)

# 5. toarray
sparse_result = U_csc @ U_csc.T
bench(".toarray() (sparse → dense)",
      lambda: sparse_result.toarray())

# 6. Combined matmul + toarray
bench("(U_csc @ U_csc.T).toarray()",
      lambda: (U_csc @ U_csc.T).toarray())

# 7. Permutation
K_inv_ord = (U_csc @ U_csc.T).toarray()
P_ix = np.ix_(P_full, P_full)
bench("K_inv[ix_(P,P)] = K_inv_ord (permute back)",
      lambda: np.empty_like(K_inv_ord).__setitem__(P_ix, K_inv_ord))

# 8. outer + subtract
bench("np.outer(alpha_v, alpha_v)",
      lambda: np.outer(alpha_v, alpha_v))

K_inv = np.empty_like(K_inv_ord)
K_inv[P_ix] = K_inv_ord
bench("K_inv - outer (W computation)",
      lambda: K_inv - np.outer(alpha_v, alpha_v))

print("\nDone.")
