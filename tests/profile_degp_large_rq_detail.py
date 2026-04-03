"""
Detailed RQ gradient breakdown for large DEGP.
Instruments _compute_grad to show where time is spent.
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')

from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.full_degp.degp import degp
import jetgp.full_degp.degp_utils as utils

DIM = 20
N_TRAIN = 200
KERNEL = "RQ"
KERNEL_TYPE = "anisotropic"

sampler = LatinHypercube(d=DIM, seed=42)
X_train = sampler.random(n=N_TRAIN)
y_vals = morris(X_train)
grads = morris_gradient(X_train)

der_indices = [[[[i + 1, 1]]] for i in range(DIM)]
derivative_locations = [list(range(N_TRAIN)) for _ in range(DIM)]
y_train = [y_vals.reshape(-1, 1)] + [grads[:, j:j+1] for j in range(DIM)]

print(f"Building DEGP model: n_train={N_TRAIN}, DIM={DIM}, kernel={KERNEL}")

model = degp(
    X_train, y_train,
    n_order=1, n_bases=DIM,
    der_indices=der_indices,
    derivative_locations=derivative_locations,
    normalize=True,
    kernel=KERNEL, kernel_type=KERNEL_TYPE
)

opt = model.optimizer
x0 = np.zeros(len(model.bounds))
for i, b in enumerate(model.bounds):
    x0[i] = 0.5 * (b[0] + b[1])

# Warm up
print("Warming up...")
opt.nll_and_grad(x0)

# Now manually replicate _compute_grad with timing
print("\nManual instrumented _compute_grad:")

# First get W by running the nll portion
oti = model.oti
diffs = model.differences_by_dim

# Build K and get W by replicating nll_and_grad logic
t_start = time.perf_counter()
ell = x0[:-1]
phi = model.kernel_func(diffs, ell)
n_bases = phi.get_active_bases()[-1]
deriv_order = 2 * model.n_order
buf = opt._get_deriv_buf(phi, n_bases, deriv_order)
factors = opt._get_deriv_factors(n_bases, deriv_order)
phi_exp = phi.get_all_derivs_fast(factors, buf)
opt._ensure_kernel_plan(n_bases)
base_shape = phi.shape
phi_3d = phi_exp.reshape(phi_exp.shape[0], base_shape[0], base_shape[1])
K = utils.rbf_kernel_fast(phi_3d, opt._kernel_plan)
sigma_n_sq = (10.0 ** x0[-1]) ** 2
K.flat[::K.shape[0] + 1] += sigma_n_sq
K += model.sigma_data**2
from scipy.linalg import cho_factor, cho_solve
L = cho_factor(K)
alpha = cho_solve(L, model.y_train)
W = cho_solve(L, np.eye(K.shape[0])) - np.outer(alpha, alpha)
t_setup = time.perf_counter()
print(f"Setup (kernel + cholesky): {t_setup - t_start:.4f}s")

ln10 = np.log(10.0)
D = len(diffs)
use_fast = opt._kernel_plan is not None
base_shape = (W.shape[0] - opt._kernel_plan['n_pts_with_derivs'],) * 2 if use_fast else None
deriv_order = 2 * model.n_order
deriv_factors = opt._get_deriv_factors(n_bases, deriv_order) if model.n_order != 0 else None
grad = np.zeros(len(x0))

def _gc(dphi):
    if model.n_order == 0:
        dphi_exp = dphi.real[np.newaxis, :, :]
    else:
        buf = opt._get_deriv_buf(dphi, n_bases, deriv_order)
        dphi_exp = dphi.get_all_derivs_fast(deriv_factors, buf)
    if use_fast:
        dphi_3d = dphi_exp.reshape(dphi_exp.shape[0], base_shape[0], base_shape[1])
        dK = utils.rbf_kernel_fast(dphi_3d, opt._kernel_plan)
    else:
        raise NotImplementedError
    return 0.5 * np.vdot(W, dK)

# Signal variance
t0 = time.perf_counter()
grad[-2] = _gc(oti.mul(2.0 * ln10, phi))
t1 = time.perf_counter()
print(f"  signal var _gc:           {t1-t0:.4f}s")

# RQ preamble: r², base, inv_base, phi_over_base
ell = 10.0 ** x0[:D]
alpha_rq = 10.0 ** float(x0[D])

t2 = time.perf_counter()
r2 = oti.zeros(phi.shape)
ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
r2.fused_sqdist(diffs, ell_sq)
t3 = time.perf_counter()
print(f"  fused_sqdist (r²):        {t3-t2:.4f}s")

t4 = time.perf_counter()
r2_scaled = oti.mul(r2, 1.0 / (2.0 * alpha_rq))
t4a = time.perf_counter()
base = oti.sum(1.0, r2_scaled)
t4b = time.perf_counter()
inv_base = oti.pow(base, -1)
t4c = time.perf_counter()
phi_over_base = oti.mul(phi, inv_base)
t5 = time.perf_counter()
print(f"  mul(r2, scale):           {t4a-t4:.4f}s")
print(f"  sum(1, r2_scaled):        {t4b-t4a:.4f}s")
print(f"  pow(base, -1):            {t4c-t4b:.4f}s")
print(f"  mul(phi, inv_base):       {t5-t4c:.4f}s")
print(f"  RQ preamble total:        {t5-t4:.4f}s")

# Ell gradient loop
t6 = time.perf_counter()
dphi_buf = oti.zeros(phi.shape)
fused_total = 0.0
gc_total = 0.0
for d in range(D):
    ta = time.perf_counter()
    dphi_buf.fused_scale_sq_mul(diffs[d], phi_over_base, -ln10 * ell[d] ** 2)
    tb = time.perf_counter()
    grad[d] = _gc(dphi_buf)
    tc = time.perf_counter()
    fused_total += tb - ta
    gc_total += tc - tb
t7 = time.perf_counter()
print(f"  ell loop ({D} dims):       {t7-t6:.4f}s")
print(f"    fused_scale_sq_mul:     {fused_total:.4f}s  ({fused_total/D*1000:.1f}ms avg)")
print(f"    _gc calls:              {gc_total:.4f}s  ({gc_total/D*1000:.1f}ms avg)")

# Break down a single _gc call
print(f"\n  Single _gc breakdown (dim 0):")
dphi_buf.fused_scale_sq_mul(diffs[0], phi_over_base, -ln10 * ell[0] ** 2)
ta = time.perf_counter()
buf = opt._get_deriv_buf(dphi_buf, n_bases, deriv_order)
tb = time.perf_counter()
dphi_exp = dphi_buf.get_all_derivs_fast(deriv_factors, buf)
tc = time.perf_counter()
dphi_3d = dphi_exp.reshape(dphi_exp.shape[0], base_shape[0], base_shape[1])
td = time.perf_counter()
dK = utils.rbf_kernel_fast(dphi_3d, opt._kernel_plan)
te = time.perf_counter()
val = 0.5 * np.vdot(W, dK)
tf = time.perf_counter()
print(f"    _get_deriv_buf:         {tb-ta:.4f}s")
print(f"    get_all_derivs_fast:    {tc-tb:.4f}s")
print(f"    reshape:                {td-tc:.4f}s")
print(f"    rbf_kernel_fast:        {te-td:.4f}s")
print(f"    np.vdot(W, dK):         {tf-te:.4f}s")
print(f"    total:                  {tf-ta:.4f}s")

# Alpha gradient
t8 = time.perf_counter()
log_base = oti.log(base)
t8a = time.perf_counter()
term = oti.sum(oti.mul(-1.0, log_base),
               oti.sum(1.0, oti.mul(-1.0, inv_base)))
t8b = time.perf_counter()
alpha_factor = ln10 * alpha_rq
grad[D] = _gc(oti.mul(alpha_factor, oti.mul(phi, term)))
t9 = time.perf_counter()
print(f"\n  alpha log:                {t8a-t8:.4f}s")
print(f"  alpha term build:         {t8b-t8a:.4f}s")
print(f"  alpha _gc:                {t9-t8b:.4f}s")

# Noise
sigma_n_sq_val = (10.0 ** x0[-1]) ** 2
grad[-1] = ln10 * sigma_n_sq_val * np.trace(W)

t_end = time.perf_counter()
print(f"\n  TOTAL _compute_grad:      {t_end-t0:.4f}s")
