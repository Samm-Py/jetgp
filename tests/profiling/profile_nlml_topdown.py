"""
Top-down cProfile of sparse DEGP NLML evaluation.

Profiles both the direct-phi path and the K-based path, and prints
cumulative timings sorted by tottime so we can see where to optimise.

Run with:  python tests/profiling/profile_nlml_topdown.py
"""

import os, sys
import cProfile
import pstats
import numpy as np
from scipy.stats.qmc import LatinHypercube
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_functions import morris, morris_gradient
from jetgp.full_degp_sparse.degp import degp as SparseDEGP

# ---------------------------------------------------------------------------
# Setup: 20D Morris function, N=100 → K size ~ 2100 x 2100
# ---------------------------------------------------------------------------
DIM = 20
N_TRAIN = 20
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

N_total = len(model.mmd_P_full)
S_full = model.sparse_S_full
nnz = sum(len(v) for v in S_full.values())
max_nnz = N_total * (N_total + 1) // 2
print(f"K size: {N_total}x{N_total}, sparsity: {1 - nnz/max_nnz:.0%}")

opt = model.optimizer
x0 = np.array([0.1] * (len(model.bounds) - 2) + [0.5, -3.0])

N_ITER = 20



# ---------------------------------------------------------------------------
# Warm-up (JIT / lazy init / kernel plan)
# ---------------------------------------------------------------------------
opt.negative_log_marginal_likelihood(x0)
# Ensure _kernel_plan is initialized so subsequent NLML calls use build_U_from_phi
diffs = model.differences_by_dim
phi_tmp = model.kernel_func(diffs, x0[:-1])
n_bases_tmp = phi_tmp.get_active_bases()[-1]
opt._ensure_kernel_plan(n_bases_tmp)

# ---------------------------------------------------------------------------
# Profile: negative_log_marginal_likelihood (uses direct path when possible)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"Profiling negative_log_marginal_likelihood  ({N_ITER} calls)")
print(f"{'='*70}")

pr = cProfile.Profile()
pr.enable()
for _ in range(N_ITER):
    opt.negative_log_marginal_likelihood(x0)
pr.disable()

stats = pstats.Stats(pr)
stats.strip_dirs()
stats.sort_stats('tottime')
stats.print_stats(40)

# ---------------------------------------------------------------------------
# Profile: nll_and_grad (K-based path + gradient)
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"Profiling nll_and_grad  ({N_ITER} calls)")
print(f"{'='*70}")

pr2 = cProfile.Profile()
pr2.enable()
for _ in range(N_ITER):
    opt.nll_and_grad(x0)
pr2.disable()

stats2 = pstats.Stats(pr2)
stats2.strip_dirs()
stats2.sort_stats('tottime')
stats2.print_stats(40)

# ---------------------------------------------------------------------------
# Profile: just _build_K_and_phi to see kernel construction cost
# ---------------------------------------------------------------------------
print(f"\n{'='*70}")
print(f"Profiling _build_K_and_phi  ({N_ITER} calls)")
print(f"{'='*70}")

pr3 = cProfile.Profile()
pr3.enable()
for _ in range(N_ITER):
    opt._build_K_and_phi(x0)
pr3.disable()

stats3 = pstats.Stats(pr3)
stats3.strip_dirs()
stats3.sort_stats('tottime')
stats3.print_stats(40)

# ---------------------------------------------------------------------------
# Micro-benchmarks: individual operations
# ---------------------------------------------------------------------------
import time

def bench(label, func, n=N_ITER):
    # warm up
    func()
    t0 = time.perf_counter()
    for _ in range(n):
        func()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"  {label:45s} {dt:8.2f} ms")

print(f"\n{'='*70}")
print(f"Micro-benchmarks (avg over {N_ITER} calls)")
print(f"{'='*70}")

# Full NLML
bench("negative_log_marginal_likelihood",
      lambda: opt.negative_log_marginal_likelihood(x0))

# nll_and_grad
bench("nll_and_grad",
      lambda: opt.nll_and_grad(x0))

# K construction
bench("_build_K_and_phi",
      lambda: opt._build_K_and_phi(x0))

# K permutation cost
K, _, _, _, _ = opt._build_K_and_phi(x0)
P_full = model.mmd_P_full
P_ix = np.ix_(P_full, P_full)
bench("K permutation K[ix_(P,P)]",
      lambda: K[P_ix])

# y permutation cost
bench("y permutation y[P_full]",
      lambda: model.y_train[P_full])

# build_U from K_ord
from jetgp.full_degp_sparse.sparse_cholesky import build_U, nlml_from_U, alpha_from_U
K_ord = K[P_ix]
y_ord = model.y_train[P_full]
U_buf = np.zeros((N_total, N_total))
bench("build_U (sparse Cholesky)",
      lambda: build_U(K_ord, model.sparse_S_full_arr, N_total,
                       block_size=model.n_bases + 1, out=U_buf))

U = build_U(K_ord, model.sparse_S_full_arr, N_total,
            block_size=model.n_bases + 1, out=U_buf)
bench("nlml_from_U",
      lambda: nlml_from_U(U, y_ord))

bench("alpha_from_U",
      lambda: alpha_from_U(U, y_ord))

# alpha un-permutation
alpha_ord = alpha_from_U(U, y_ord)
def _unperm(alpha_ord, P_full):
    alpha_v = np.empty_like(alpha_ord)
    alpha_v[P_full] = alpha_ord
    return alpha_v

bench("alpha un-permute",
      lambda: _unperm(alpha_ord, P_full))

print("\nDone.")
