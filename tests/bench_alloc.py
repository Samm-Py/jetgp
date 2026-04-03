"""Benchmark: np.empty allocation vs buffer reuse for kernel assembly."""
import sys; sys.path.insert(0, '.')
import numpy as np
import time
from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.full_degp.degp import degp
from jetgp.full_degp import degp_utils as utils

DIM = 20; N = 100

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

oti = m.oti
diffs = m.differences_by_dim
phi = m.kernel_func(diffs, x0[:-1])
n_bases = phi.get_active_bases()[-1]
deriv_order = 2 * m.n_order
factors = m.optimizer._get_deriv_factors(n_bases, deriv_order)
buf = m.optimizer._get_deriv_buf(phi, n_bases, deriv_order)
phi_exp = phi.get_all_derivs_fast(factors, buf)
phi_exp_3d = phi_exp.reshape(phi_exp.shape[0], N, N)

m.optimizer._ensure_kernel_plan(n_bases)
plan = m.optimizer._kernel_plan
# Pre-cache absolute offsets
plan['row_offsets_abs'] = plan['row_offsets'] + N
plan['col_offsets_abs'] = plan['col_offsets'] + N

total = N + plan['n_pts_with_derivs']
print(f"K size: {total}x{total}, {total*total*8/1e6:.1f} MB")

# Warm up
K1 = utils.rbf_kernel_fast(phi_exp_3d, plan)
K_buf = np.empty((total, total))
K2 = utils.rbf_kernel_fast(phi_exp_3d, plan, out=K_buf)

N_ITER = 200  # simulate ~one JADE evaluation's worth of calls (22 per eval * ~10)

# Benchmark: fresh allocation each time
times = []
for _ in range(5):
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        K = utils.rbf_kernel_fast(phi_exp_3d, plan)
    t1 = time.perf_counter()
    times.append(t1 - t0)
alloc_time = np.median(times)
print(f"Fresh alloc ({N_ITER} calls): {alloc_time*1000:.1f}ms  ({alloc_time/N_ITER*1000:.3f}ms/call)")

# Benchmark: buffer reuse
times = []
for _ in range(5):
    t0 = time.perf_counter()
    for _ in range(N_ITER):
        K = utils.rbf_kernel_fast(phi_exp_3d, plan, out=K_buf)
    t1 = time.perf_counter()
    times.append(t1 - t0)
reuse_time = np.median(times)
print(f"Buffer reuse ({N_ITER} calls): {reuse_time*1000:.1f}ms  ({reuse_time/N_ITER*1000:.3f}ms/call)")

savings = (alloc_time - reuse_time) / alloc_time * 100
print(f"Savings: {savings:.1f}%  ({(alloc_time-reuse_time)/N_ITER*1000:.3f}ms/call)")
print(f"Over 4400 JADE calls: {(alloc_time-reuse_time)/N_ITER*4400:.1f}s saved")
