"""
Profile sparse DEGP optimisation loop on Morris 20D, N=50.
K size: 50 * 21 = 1050 x 1050.
Run with: kernprof -lv tests/profiling/profile_sparse_m20n5.py
"""

import os, sys
import numpy as np
from scipy.stats.qmc import LatinHypercube
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_functions import morris, morris_gradient
from jetgp.full_degp_sparse.degp import degp as SparseDEGP

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

N_total = len(model.mmd_P_full)
S_full = model.sparse_S_full
nnz = sum(len(v) for v in S_full.values())
max_nnz = N_total * (N_total + 1) // 2
print(f"K size: {N_total}x{N_total}, sparsity: {1 - nnz/max_nnz:.0%}")
print(f"Supernodes: {len(model.sparse_supernodes) if model.sparse_supernodes else 'None (column-wise)'}")

# Warm up
opt = model.optimizer
x0 = np.array([0.1] * (len(model.bounds) - 2) + [0.5, -3.0])
opt.negative_log_marginal_likelihood(x0)

# Profile
for _ in range(10):
    opt.negative_log_marginal_likelihood(x0)

print("Done.")
