"""Quick test: vdot_expand_fast vs get_all_derivs_fast + vdot."""
import numpy as np
import sys, time
sys.path.insert(0, '.')

from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.wdegp.wdegp import wdegp
import jetgp.utils as utils

np.random.seed(42)
DIM = 20; N = 200; N_SUB = 40
sampler = LatinHypercube(d=DIM, seed=1000)
X_train = sampler.random(n=N)
y_vals = morris(X_train)
grads = morris_gradient(X_train)

y_all_col = y_vals.reshape(-1, 1)
der_specs = utils.gen_OTI_indices(DIM, 1)
pts_per_sub = N // N_SUB

submodel_data, derivative_specs_list, derivative_locations_list = [], [], []
for s in range(N_SUB):
    start = s * pts_per_sub
    end = start + pts_per_sub
    data_s = [y_all_col] + [grads[start:end, j:j+1] for j in range(DIM)]
    submodel_data.append(data_s)
    derivative_specs_list.append(der_specs)
    derivative_locations_list.append([list(range(start, end)) for _ in range(DIM)])

model = wdegp(X_train, submodel_data, 1, DIM, derivative_specs_list,
              derivative_locations=derivative_locations_list,
              normalize=True, kernel='SE', kernel_type='anisotropic')

from jetgp.wdegp.optimizer import Optimizer
import pyoti.static.onumm20n2 as oti
from math import comb

opt = Optimizer(model)
x0 = np.array([0.1] * (len(model.bounds) - 2) + [0.5, -3.0])

D = DIM
ln10 = np.log(10.0)
ell = 10.0 ** x0[:D]
sigma_f_sq = 10.0 ** (2.0 * x0[-2])

diffs = model.differences_by_dim
phi = model.kernel_func(diffs, ell)
phi = oti.mul(sigma_f_sq, phi)

n_bases = model.n_bases
deriv_order = 2 * model.n_order
ndir = comb(n_bases + deriv_order, deriv_order)
factors = opt._get_deriv_factors(n_bases, deriv_order)
buf = np.zeros((ndir, N, N), dtype=np.float64)
W_proj = np.random.randn(ndir, N, N)

# Test with a few different dphi values
dphi_list = [
    oti.mul(2.0 * ln10, phi),
]
dphi_buf = oti.zeros(phi.shape)
for d in range(min(5, D)):
    dphi_buf.fused_scale_sq_mul_sparse(diffs[d], phi, -ln10 * ell[d] ** 2, d)
    dphi_list.append(oti.mul(1.0, dphi_buf))  # copy

print(f"Testing vdot_expand_fast vs get_all_derivs_fast + vdot...")
print(f"Matrix: {N}x{N}, ndir={ndir}\n")

has_vdot = hasattr(dphi_list[0], 'vdot_expand_fast')
print(f"vdot_expand_fast available: {has_vdot}")
if not has_vdot:
    print("ERROR: vdot_expand_fast not found on OTI matrix!")
    sys.exit(1)

for i, dphi in enumerate(dphi_list):
    # Reference: expand + vdot
    dphi_exp = dphi.get_all_derivs_fast(factors, buf)
    ref = np.vdot(W_proj, dphi_exp)

    # New: fused
    fused = dphi.vdot_expand_fast(factors, W_proj)

    err = abs(ref - fused) / (abs(ref) + 1e-15)
    status = "PASS" if err < 1e-12 else "FAIL"
    print(f"  dphi[{i}]: ref={ref:.10e}, fused={fused:.10e}, rel_err={err:.2e} [{status}]")

# Timing comparison
print(f"\nTiming ({100} reps each):")
dphi = dphi_list[0]

t0 = time.perf_counter()
for _ in range(100):
    dphi.get_all_derivs_fast(factors, buf)
    np.vdot(W_proj, buf)
t_old = time.perf_counter() - t0

t0 = time.perf_counter()
for _ in range(100):
    dphi.vdot_expand_fast(factors, W_proj)
t_new = time.perf_counter() - t0

print(f"  expand + vdot:    {t_old/100*1000:.3f} ms")
print(f"  vdot_expand_fast: {t_new/100*1000:.3f} ms")
print(f"  Speedup:          {t_old/t_new:.2f}x")
