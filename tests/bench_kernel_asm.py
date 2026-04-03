"""Benchmark: numba element-by-element vs numpy block copy for kernel assembly."""
import sys; sys.path.insert(0, '.')
import numpy as np
import numba
import time
from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.full_degp.degp import degp
from jetgp.full_degp.degp_utils import _assemble_kernel_numba

DIM = 20
N = 100

sampler = LatinHypercube(d=DIM, seed=42)
X = sampler.random(n=N)
y = morris(X); g = morris_gradient(X)
di = [[[[i+1,1]]] for i in range(DIM)]
dl = [list(range(N)) for _ in range(DIM)]
yt = [y.reshape(-1,1)] + [g[:,j:j+1] for j in range(DIM)]
m = degp(X, yt, n_order=1, n_bases=DIM, der_indices=di, derivative_locations=dl,
         normalize=True, kernel="SE", kernel_type="anisotropic")
x0 = np.zeros(len(m.bounds))
for i,b in enumerate(m.bounds): x0[i]=0.5*(b[0]+b[1])

# Get phi_exp_3d
oti = m.oti
diffs = m.differences_by_dim
phi = m.kernel_func(diffs, x0[:-1])
n_bases = phi.get_active_bases()[-1]
deriv_order = 2 * m.n_order
factors = m.optimizer._get_deriv_factors(n_bases, deriv_order)
buf = m.optimizer._get_deriv_buf(phi, n_bases, deriv_order)
phi_exp = phi.get_all_derivs_fast(factors, buf)
phi_exp_3d = phi_exp.reshape(phi_exp.shape[0], N, N)

# Get kernel plan
m.optimizer._ensure_kernel_plan(n_bases)
plan = m.optimizer._kernel_plan

n_rows_func = N
n_cols_func = N
total = N + plan['n_pts_with_derivs']
row_off = plan['row_offsets'] + n_rows_func
col_off = plan['col_offsets'] + n_cols_func

# Warm up numba
K1 = np.empty((total, total))
_assemble_kernel_numba(phi_exp_3d, K1, n_rows_func, n_cols_func,
    plan['fd_flat_indices'], plan['df_flat_indices'], plan['dd_flat_indices'],
    plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
    plan['signs'], plan['n_deriv_types'], row_off, col_off)

# Benchmark current numba approach
times = []
for _ in range(10):
    K1 = np.empty((total, total))
    t0 = time.perf_counter()
    _assemble_kernel_numba(phi_exp_3d, K1, n_rows_func, n_cols_func,
        plan['fd_flat_indices'], plan['df_flat_indices'], plan['dd_flat_indices'],
        plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
        plan['signs'], plan['n_deriv_types'], row_off, col_off)
    t1 = time.perf_counter()
    times.append(t1 - t0)
print(f"Current numba:     {np.median(times)*1000:8.2f}ms")

# Benchmark numpy block-copy approach (for trivial indices case)
def assemble_kernel_numpy_blocks(phi_exp_3d, plan, n_rows_func, n_cols_func):
    total = n_rows_func + plan['n_pts_with_derivs']
    K = np.empty((total, total))
    signs = plan['signs']
    n_dt = plan['n_deriv_types']
    fd_fi = plan['fd_flat_indices']
    df_fi = plan['df_flat_indices']
    dd_fi = plan['dd_flat_indices']
    idx_sizes = plan['index_sizes']

    # ff block
    K[:n_rows_func, :n_cols_func] = phi_exp_3d[0, :, :] * signs[0]

    # fd blocks
    ro_base = n_rows_func
    for j in range(n_dt):
        sz = idx_sizes[j]
        co = plan['col_offsets'][j] + n_cols_func
        K[:n_rows_func, co:co+sz] = phi_exp_3d[fd_fi[j], :, :sz] * signs[j+1]

    # df blocks
    for i in range(n_dt):
        sz = idx_sizes[i]
        ro = plan['row_offsets'][i] + n_rows_func
        K[ro:ro+sz, :n_cols_func] = phi_exp_3d[df_fi[i], :sz, :] * signs[0]

    # dd blocks
    for i in range(n_dt):
        sz_i = idx_sizes[i]
        ro = plan['row_offsets'][i] + n_rows_func
        for j in range(n_dt):
            sz_j = idx_sizes[j]
            co = plan['col_offsets'][j] + n_cols_func
            K[ro:ro+sz_i, co:co+sz_j] = phi_exp_3d[dd_fi[i,j], :sz_i, :sz_j] * signs[j+1]

    return K

# Warm up
K2 = assemble_kernel_numpy_blocks(phi_exp_3d, plan, n_rows_func, n_cols_func)

# Verify correctness
print(f"Max diff: {np.max(np.abs(K1 - K2)):.2e}")

times = []
for _ in range(10):
    t0 = time.perf_counter()
    K2 = assemble_kernel_numpy_blocks(phi_exp_3d, plan, n_rows_func, n_cols_func)
    t1 = time.perf_counter()
    times.append(t1 - t0)
print(f"Numpy block-copy:  {np.median(times)*1000:8.2f}ms")

# Even faster: batch the dd blocks where sign is the same
# Group dd blocks by sign, then use a single vectorized copy
def assemble_kernel_numpy_batched(phi_exp_3d, plan, n_rows_func, n_cols_func):
    total = n_rows_func + plan['n_pts_with_derivs']
    K = np.empty((total, total))
    signs = plan['signs']
    n_dt = plan['n_deriv_types']
    fd_fi = plan['fd_flat_indices']
    df_fi = plan['df_flat_indices']
    dd_fi = plan['dd_flat_indices']
    idx_sizes = plan['index_sizes']

    # ff block
    if signs[0] == 1.0:
        K[:n_rows_func, :n_cols_func] = phi_exp_3d[0, :, :]
    else:
        np.multiply(phi_exp_3d[0, :, :], signs[0], out=K[:n_rows_func, :n_cols_func])

    # fd blocks
    for j in range(n_dt):
        sz = idx_sizes[j]
        co = plan['col_offsets'][j] + n_cols_func
        sj = signs[j+1]
        if sj == 1.0:
            K[:n_rows_func, co:co+sz] = phi_exp_3d[fd_fi[j], :, :sz]
        else:
            np.multiply(phi_exp_3d[fd_fi[j], :, :sz], sj, out=K[:n_rows_func, co:co+sz])

    # df blocks
    s0 = signs[0]
    for i in range(n_dt):
        sz = idx_sizes[i]
        ro = plan['row_offsets'][i] + n_rows_func
        if s0 == 1.0:
            K[ro:ro+sz, :n_cols_func] = phi_exp_3d[df_fi[i], :sz, :]
        else:
            np.multiply(phi_exp_3d[df_fi[i], :sz, :], s0, out=K[ro:ro+sz, :n_cols_func])

    # dd blocks
    for i in range(n_dt):
        sz_i = idx_sizes[i]
        ro = plan['row_offsets'][i] + n_rows_func
        for j in range(n_dt):
            sz_j = idx_sizes[j]
            co = plan['col_offsets'][j] + n_cols_func
            sj = signs[j+1]
            if sj == 1.0:
                K[ro:ro+sz_i, co:co+sz_j] = phi_exp_3d[dd_fi[i,j], :sz_i, :sz_j]
            else:
                np.multiply(phi_exp_3d[dd_fi[i,j], :sz_i, :sz_j], sj,
                           out=K[ro:ro+sz_i, co:co+sz_j])

    return K

K3 = assemble_kernel_numpy_batched(phi_exp_3d, plan, n_rows_func, n_cols_func)
print(f"Max diff (batched): {np.max(np.abs(K1 - K3)):.2e}")

times = []
for _ in range(10):
    t0 = time.perf_counter()
    K3 = assemble_kernel_numpy_batched(phi_exp_3d, plan, n_rows_func, n_cols_func)
    t1 = time.perf_counter()
    times.append(t1 - t0)
print(f"Numpy batched:     {np.median(times)*1000:8.2f}ms")

# Check: what fraction of signs are 1.0?
print(f"\nSigns: {signs}")
n_pos = np.sum(signs == 1.0)
n_neg = np.sum(signs == -1.0)
print(f"Positive: {n_pos}/{len(signs)}, Negative: {n_neg}/{len(signs)}")
