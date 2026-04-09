"""
kernprof -lv tests/profiling/kernprof_build_U_from_phi.py
"""
import os, sys
import numpy as np
from scipy.stats.qmc import LatinHypercube
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from benchmark_functions import morris, morris_gradient
from jetgp.full_degp_sparse.degp import degp as SparseDEGP
from jetgp.full_degp_sparse.sparse_cholesky import build_U_from_phi

DIM = 20
N_TRAIN = 100
DER_INDICES = [[[[i, 1]] for i in range(1, DIM + 1)]]

X_train = LatinHypercube(d=DIM, seed=42).random(n=N_TRAIN)
y_vals = morris(X_train)
grads = morris_gradient(X_train)
y_train_list = [y_vals.reshape(-1, 1)] + [grads[:, j].reshape(-1, 1) for j in range(DIM)]

model = SparseDEGP(
    X_train, y_train_list,
    n_order=1, n_bases=DIM,
    der_indices=DER_INDICES,
    normalize=True, kernel='SE', kernel_type='anisotropic',
    rho=1.0, use_supernodes=False,
)

opt = model.optimizer
x0 = np.array([0.1] * (len(model.bounds) - 2) + [0.5, -3.0])

# Warm up + init kernel plan
opt.negative_log_marginal_likelihood(x0)
diffs = model.differences_by_dim
phi = model.kernel_func(diffs, x0[:-1])
n_bases = phi.get_active_bases()[-1]
opt._ensure_kernel_plan(n_bases)
opt._ensure_phi_index_maps(N_TRAIN)

sigma_n_sq = (10.0 ** x0[-1]) ** 2
deriv_order = 2 * model.n_order
phi_exp = opt._expand_derivs(phi, n_bases, deriv_order)
phi_3d = phi_exp.reshape(phi_exp.shape[0], phi.shape[0], phi.shape[1])

k_type, k_phys, deriv_lookup, sign_lookup = opt._k_index_map
N_total = len(model.mmd_P_full)
U_buf = np.zeros((N_total, N_total))

for _ in range(20):
    build_U_from_phi(
        phi_3d, model.sparse_S_full_arr, N_total,
        block_size=model.n_bases + 1,
        k_type=k_type, k_phys=k_phys,
        deriv_lookup=deriv_lookup, sign_lookup=sign_lookup,
        P_full=model.mmd_P_full,
        sigma_n_sq=sigma_n_sq,
        sigma_data_diag=opt._sigma_data_diag_mmd,
        out=U_buf,
    )

print("Done.")
