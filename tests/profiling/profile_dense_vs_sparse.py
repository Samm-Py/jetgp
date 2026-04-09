"""
Head-to-head comparison: Dense DEGP vs Sparse DEGP.

Profiles negative_log_marginal_likelihood and nll_and_grad for both,
plus cProfile top-down for each nll_and_grad path.
"""

import os, sys, time, cProfile, pstats
import numpy as np
from scipy.stats.qmc import LatinHypercube
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_functions import morris, morris_gradient
from jetgp.full_degp.degp import degp as DenseDEGP
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

print(f"Setup: {DIM}D Morris, N={N_TRAIN}, K size={N_TRAIN*(DIM+1)}x{N_TRAIN*(DIM+1)}")
print()

# ---------------------------------------------------------------------------
# Dense DEGP
# ---------------------------------------------------------------------------
print("Building Dense DEGP...", flush=True)
dense_model = DenseDEGP(
    X_train, y_train_list,
    n_order=1, n_bases=DIM,
    der_indices=DER_INDICES,
    normalize=True, kernel='SE', kernel_type='anisotropic',
)
dense_opt = dense_model.optimizer
x0_dense = np.array([0.1] * (len(dense_model.bounds) - 2) + [0.5, -3.0])

# ---------------------------------------------------------------------------
# Sparse DEGP
# ---------------------------------------------------------------------------
print("Building Sparse DEGP...", flush=True)
sparse_model = SparseDEGP(
    X_train, y_train_list,
    n_order=1, n_bases=DIM,
    der_indices=DER_INDICES,
    normalize=True, kernel='SE', kernel_type='anisotropic',
    rho=1.0, use_supernodes=False,
)
sparse_opt = sparse_model.optimizer
x0_sparse = np.array([0.1] * (len(sparse_model.bounds) - 2) + [0.5, -3.0])

N_total = len(sparse_model.mmd_P_full)
S_full = sparse_model.sparse_S_full
nnz = sum(len(v) for v in S_full.values())
max_nnz = N_total * (N_total + 1) // 2
print(f"Sparse: sparsity={1 - nnz/max_nnz:.0%}")
print()

# ---------------------------------------------------------------------------
# Warm up
# ---------------------------------------------------------------------------
print("Warming up...", flush=True)
dense_opt.negative_log_marginal_likelihood(x0_dense)
dense_opt.nll_and_grad(x0_dense)
sparse_opt.negative_log_marginal_likelihood(x0_sparse)
sparse_opt.nll_and_grad(x0_sparse)

# ---------------------------------------------------------------------------
# Micro-benchmarks
# ---------------------------------------------------------------------------
N_ITER = 30

def bench(label, func, n=N_ITER):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    times = np.array(times) * 1000
    print(f"  {label:50s}  mean={times.mean():7.1f}  median={np.median(times):7.1f}  min={times.min():7.1f} ms")
    return np.median(times)

print(f"{'='*80}")
print(f"Micro-benchmarks ({N_ITER} calls each)")
print(f"{'='*80}")

print("\nnegative_log_marginal_likelihood:")
t_dense_nlml = bench("Dense DEGP",
      lambda: dense_opt.negative_log_marginal_likelihood(x0_dense))
t_sparse_nlml = bench("Sparse DEGP",
      lambda: sparse_opt.negative_log_marginal_likelihood(x0_sparse))
print(f"  {'Speedup:':<50s}  {t_dense_nlml/t_sparse_nlml:.2f}x")

print("\nnll_and_grad:")
t_dense_grad = bench("Dense DEGP",
      lambda: dense_opt.nll_and_grad(x0_dense))
t_sparse_grad = bench("Sparse DEGP",
      lambda: sparse_opt.nll_and_grad(x0_sparse))
print(f"  {'Speedup:':<50s}  {t_dense_grad/t_sparse_grad:.2f}x")

# ---------------------------------------------------------------------------
# Verify they agree
# ---------------------------------------------------------------------------
nll_d, grad_d = dense_opt.nll_and_grad(x0_dense)
nll_s, grad_s = sparse_opt.nll_and_grad(x0_sparse)
print(f"\nNLL agreement: dense={nll_d:.6f}, sparse={nll_s:.6f}, diff={abs(nll_d-nll_s):.2e}")
print(f"Grad agreement: max |diff|={np.max(np.abs(grad_d - grad_s)):.2e}")

# ---------------------------------------------------------------------------
# cProfile: Dense nll_and_grad
# ---------------------------------------------------------------------------
N_PROF = 20
print(f"\n{'='*80}")
print(f"cProfile: Dense DEGP nll_and_grad ({N_PROF} calls)")
print(f"{'='*80}")

pr = cProfile.Profile()
pr.enable()
for _ in range(N_PROF):
    dense_opt.nll_and_grad(x0_dense)
pr.disable()
pstats.Stats(pr).strip_dirs().sort_stats('tottime').print_stats(30)

# ---------------------------------------------------------------------------
# cProfile: Sparse nll_and_grad
# ---------------------------------------------------------------------------
print(f"\n{'='*80}")
print(f"cProfile: Sparse DEGP nll_and_grad ({N_PROF} calls)")
print(f"{'='*80}")

pr2 = cProfile.Profile()
pr2.enable()
for _ in range(N_PROF):
    sparse_opt.nll_and_grad(x0_sparse)
pr2.disable()
pstats.Stats(pr2).strip_dirs().sort_stats('tottime').print_stats(30)

print("\nDone.")
