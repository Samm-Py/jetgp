"""
Profile _gc internals for WDEGP Morris (N=200, N_SUB=40).
Break down: get_all_derivs_fast vs vdot.
"""
import os
import numpy as np
import time
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.wdegp.wdegp import wdegp
import jetgp.utils as utils

np.random.seed(42)
DIM = 20
N = 200
N_SUB = 40
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"

sampler = LatinHypercube(d=DIM, seed=1000)
X_train = sampler.random(n=N)
y_vals = morris(X_train)
grads = morris_gradient(X_train)

y_all_col = y_vals.reshape(-1, 1)
der_specs = utils.gen_OTI_indices(DIM, 1)
pts_per_sub = N // N_SUB

submodel_data = []
derivative_specs_list = []
derivative_locations_list = []

for s in range(N_SUB):
    start = s * pts_per_sub
    end = start + pts_per_sub
    data_s = [y_all_col]
    for j in range(DIM):
        data_s.append(grads[start:end, j:j+1])
    submodel_data.append(data_s)
    derivative_specs_list.append(der_specs)
    derivative_locations_list.append([list(range(start, end)) for _ in range(DIM)])

model = wdegp(
    X_train, submodel_data,
    1, DIM,
    derivative_specs_list,
    derivative_locations=derivative_locations_list,
    normalize=True,
    kernel=KERNEL, kernel_type=KERNEL_TYPE
)

from jetgp.wdegp.optimizer import Optimizer
opt = Optimizer(model)
x0 = np.array([0.1] * (len(model.bounds) - 2) + [0.5, -3.0])

# Warm up
opt.nll_and_grad(x0)

# Now manually time the pieces
# Get phi by running the kernel computation
import pyoti.static.onumm20n2 as oti
from math import comb

n_bases = model.n_bases
deriv_order = 2 * model.n_order

# Run nll_and_grad to populate internal state, then time _gc components
# We need a dphi to test with - use the sigma_f one: 2*ln10*phi
ln10 = np.log(10.0)

# Build phi
D = DIM
ell = 10.0 ** x0[:D]
sigma_f_sq = 10.0 ** (2.0 * x0[-2])
sigma_n_sq = 10.0 ** (2.0 * x0[-1])

diffs = model.differences_by_dim

phi = model.kernel_func(diffs, ell)
phi = oti.mul(sigma_f_sq, phi)

# Now time get_all_derivs_fast
ndir = comb(n_bases + deriv_order, deriv_order)
buf = np.zeros((ndir, phi.shape[0], phi.shape[1]), dtype=np.float64)
factors = opt._get_deriv_factors(n_bases, deriv_order)

# Also get W_proj shape
proj_shape = (ndir, N, N)
W_proj = np.random.randn(*proj_shape)  # dummy for timing

dphi = oti.mul(2.0 * ln10, phi)

N_REPS = 100

# Time get_all_derivs_fast
t0 = time.perf_counter()
for _ in range(N_REPS):
    dphi.get_all_derivs_fast(factors, buf)
t_expand = time.perf_counter() - t0

# Time vdot
dphi_3d = buf.reshape(proj_shape)
t0 = time.perf_counter()
for _ in range(N_REPS):
    np.vdot(W_proj, dphi_3d)
t_vdot = time.perf_counter() - t0

# Time the fused_scale_sq_mul_sparse (creating dphi)
dphi_buf = oti.zeros(phi.shape)
t0 = time.perf_counter()
for _ in range(N_REPS):
    dphi_buf.fused_scale_sq_mul_sparse(diffs[0], phi, -ln10 * ell[0] ** 2, 0)
t_fused = time.perf_counter() - t0

# Time oti.mul (scalar * phi)
t0 = time.perf_counter()
for _ in range(N_REPS):
    oti.mul(2.0 * ln10, phi)
t_mul = time.perf_counter() - t0

print(f"Matrix shape: {phi.shape}, OTI type: m{n_bases}n{deriv_order}")
print(f"ndir (expansion dirs): {ndir}")
print(f"Expanded size: {ndir} x {N} x {N} = {ndir*N*N:,} elements")
print(f"\nPer-call timings ({N_REPS} reps):")
print(f"  get_all_derivs_fast:      {t_expand/N_REPS*1000:.3f} ms")
print(f"  np.vdot:                  {t_vdot/N_REPS*1000:.3f} ms")
print(f"  fused_scale_sq_mul_sparse:{t_fused/N_REPS*1000:.3f} ms")
print(f"  oti.mul (scalar):         {t_mul/N_REPS*1000:.3f} ms")
print(f"\nPer iteration (21 _gc calls):")
print(f"  get_all_derivs_fast:      {21*t_expand/N_REPS*1000:.1f} ms")
print(f"  np.vdot:                  {21*t_vdot/N_REPS*1000:.1f} ms")
print(f"  fused_scale_sq_mul_sparse:{20*t_fused/N_REPS*1000:.1f} ms  (20 length scales)")
print(f"  oti.mul (scalar):         {1*t_mul/N_REPS*1000:.1f} ms   (1 sigma_f)")
