"""
Standalone benchmark for _assemble_kernel_numba.

Isolates the kernel-assembly cost from everything else in nll_and_grad
so we can see whether 567 us/call is real work, Numba overhead, or
non-vectorized scalar codegen in the df/fd/dd blocks.

Run with:
    python bench_assemble_kernel.py
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.wdegp.wdegp import wdegp
from jetgp.wdegp.optimizer import Optimizer
from jetgp.wdegp.wdegp_utils import _assemble_kernel_numba
import jetgp.utils as utils

np.random.seed(42)
DIM = 20
N = 200

sampler = LatinHypercube(d=DIM, seed=1000)
X_train = sampler.random(n=N)
y_vals = morris(X_train)
grads = morris_gradient(X_train)

y_all_col = y_vals.reshape(-1, 1)
der_specs = utils.gen_OTI_indices(DIM, 1)

submodel_data = []
derivative_specs_list = []
derivative_locations_list = []
for i in range(N):
    data_i = [y_all_col] + [grads[i:i+1, j:j+1] for j in range(DIM)]
    submodel_data.append(data_i)
    derivative_specs_list.append(der_specs)
    derivative_locations_list.append([[i] for _ in range(DIM)])

model = wdegp(
    X_train, submodel_data,
    1, DIM,
    derivative_specs_list,
    derivative_locations=derivative_locations_list,
    normalize=True,
    kernel="SE", kernel_type="anisotropic",
)
opt = Optimizer(model)
x0 = np.array([0.1] * (len(model.bounds) - 2) + [0.5, -3.0])

# Run one nll_and_grad to populate buffers, then grab the first submodel's
# phi_exp_3d and plan so we can replay the kernel call in a tight loop.
opt.nll_and_grad(x0)

diffs = model.differences_by_dim
phi = model.kernel_func(diffs, x0[:-1])
n_bases = phi.get_active_bases()[-1]
deriv_order = 2 * model.n_order
phi_exp = opt._expand_derivs(phi, n_bases, deriv_order)
base_shape = phi.shape
phi_exp_3d = phi_exp.reshape(phi_exp.shape[0], base_shape[0], base_shape[1]).copy()

plan = model.kernel_plans[0]
n_rows_func = base_shape[0]
n_cols_func = base_shape[1]
total = n_rows_func + plan['n_pts_with_derivs']
K_buf = np.empty((total, total))

row_off = plan['row_offsets'] + n_rows_func
col_off = plan['col_offsets'] + n_rows_func

print(f"N={N}  n_rows_func={n_rows_func}  total={total}  "
      f"n_pts_with_derivs={plan['n_pts_with_derivs']}  "
      f"n_deriv_types={plan['n_deriv_types']}")
print(f"phi_exp_3d shape: {phi_exp_3d.shape}  dtype={phi_exp_3d.dtype}")

ITERS = 2000

# Warmup (numba compile + cache prime)
for _ in range(20):
    _assemble_kernel_numba(
        phi_exp_3d, K_buf, n_rows_func, n_cols_func,
        plan['fd_flat_indices'], plan['df_flat_indices'], plan['dd_flat_indices'],
        plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
        plan['signs'], plan['n_deriv_types'], row_off, col_off,
    )

t0 = time.perf_counter_ns()
for _ in range(ITERS):
    _assemble_kernel_numba(
        phi_exp_3d, K_buf, n_rows_func, n_cols_func,
        plan['fd_flat_indices'], plan['df_flat_indices'], plan['dd_flat_indices'],
        plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
        plan['signs'], plan['n_deriv_types'], row_off, col_off,
    )
full_ns = (time.perf_counter_ns() - t0) / ITERS

t0 = time.perf_counter_ns()
for _ in range(ITERS):
    for i in range(len(model.kernel_plans)):
        p = model.kernel_plans[i]
        _assemble_kernel_numba(
            phi_exp_3d, K_buf, n_rows_func, n_cols_func,
            p['fd_flat_indices'], p['df_flat_indices'], p['dd_flat_indices'],
            p['idx_flat'], p['idx_offsets'], p['index_sizes'],
            p['signs'], p['n_deriv_types'],
            p['row_offsets'] + n_rows_func, p['col_offsets'] + n_rows_func,
        )
cycle_ns = (time.perf_counter_ns() - t0) / (ITERS * len(model.kernel_plans))
print(f"  cycling all {len(model.kernel_plans)} plans: {cycle_ns:8.1f} ns/call")


# FF-only baseline: pure memcpy of the 200x200 block
for _ in range(20):
    K_buf[:n_rows_func, :n_rows_func] = phi_exp_3d[0]
t0 = time.perf_counter_ns()
for _ in range(ITERS):
    K_buf[:n_rows_func, :n_rows_func] = phi_exp_3d[0]
ff_ns = (time.perf_counter_ns() - t0) / ITERS

# Null-call overhead baseline: a numba function that does nothing but
# accepts the same argument shapes. Tells us how much is marshaling.
# We'll just measure an empty python-side loop as a floor.
t0 = time.perf_counter_ns()
for _ in range(ITERS):
    pass
empty_ns = (time.perf_counter_ns() - t0) / ITERS

print()
print(f"  full _assemble_kernel_numba: {full_ns:8.1f} ns/call")
print(f"  FF-block memcpy baseline:    {ff_ns:8.1f} ns/call")
print(f"  empty python loop:           {empty_ns:8.1f} ns/call")
print()
print(f"  full - FF = df/fd/dd blocks: {full_ns - ff_ns:8.1f} ns/call "
      f"({(full_ns - ff_ns) / full_ns * 100:.1f}% of full)")
