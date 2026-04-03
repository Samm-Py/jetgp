"""
Detailed breakdown of every oti.mul/sum/pow/log in RQ preamble + alpha gradient.
"""
import numpy as np
import time
import sys
sys.path.insert(0, '.')

from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.full_degp.degp import degp

DIM = 20
N_TRAIN = 200

sampler = LatinHypercube(d=DIM, seed=42)
X_train = sampler.random(n=N_TRAIN)
y_vals = morris(X_train)
grads = morris_gradient(X_train)

der_indices = [[[[i + 1, 1]]] for i in range(DIM)]
derivative_locations = [list(range(N_TRAIN)) for _ in range(DIM)]
y_train = [y_vals.reshape(-1, 1)] + [grads[:, j:j+1] for j in range(DIM)]

model = degp(
    X_train, y_train,
    n_order=1, n_bases=DIM,
    der_indices=der_indices,
    derivative_locations=derivative_locations,
    normalize=True,
    kernel="RQ", kernel_type="anisotropic"
)

opt = model.optimizer
x0 = np.zeros(len(model.bounds))
for i, b in enumerate(model.bounds):
    x0[i] = 0.5 * (b[0] + b[1])

# Warm up
opt.nll_and_grad(x0)

oti = model.oti
diffs = model.differences_by_dim
ln10 = np.log(10.0)
D = len(diffs)

# Get phi from kernel
ell_hp = x0[:-1]
phi = model.kernel_func(diffs, ell_hp)

ell = 10.0 ** x0[:D]
alpha_rq = 10.0 ** float(x0[D])

print(f"OTI array shape: {phi.shape}")
print(f"OTI array size: {phi.shape[0] * phi.shape[1]} elements")
print()

# Time each operation individually, average over 5 runs
def bench(name, fn, n=5):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    med = np.median(times) * 1000
    print(f"  {name:40s} {med:8.2f}ms")
    return result

print("=== RQ preamble operations ===")

# fused_sqdist
r2 = oti.zeros(phi.shape)
ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)
bench("fused_sqdist(r2)", lambda: r2.fused_sqdist(diffs, ell_sq))

# mul(r2, scalar)
bench("mul(r2, 1/(2*alpha))  [mul_ro]", lambda: oti.mul(r2, 1.0 / (2.0 * alpha_rq)))

r2_scaled = oti.mul(r2, 1.0 / (2.0 * alpha_rq))

# sum(1.0, r2_scaled)
bench("sum(1.0, r2_scaled)   [sum_ro]", lambda: oti.sum(1.0, r2_scaled))

base = oti.sum(1.0, r2_scaled)

# pow(base, -1)
bench("pow(base, -1)", lambda: oti.pow(base, -1))

inv_base = oti.pow(base, -1)

# mul(phi, inv_base)
bench("mul(phi, inv_base)    [mul_oo]", lambda: oti.mul(phi, inv_base))

phi_over_base = oti.mul(phi, inv_base)

print()
print("=== Signal variance ===")
bench("mul(2*ln10, phi)      [mul_ro]", lambda: oti.mul(2.0 * ln10, phi))

print()
print("=== Alpha gradient term build ===")

# log(base)
bench("log(base)", lambda: oti.log(base))

log_base = oti.log(base)

# mul(-1, log_base) - negate
bench("mul(-1, log_base)     [mul_ro]", lambda: oti.mul(-1.0, log_base))

neg_log_base = oti.mul(-1.0, log_base)

# mul(-1, inv_base) - negate
bench("mul(-1, inv_base)     [mul_ro]", lambda: oti.mul(-1.0, inv_base))

neg_inv_base = oti.mul(-1.0, inv_base)

# sum(1.0, neg_inv_base)
bench("sum(1.0, neg_inv_base) [sum_ro]", lambda: oti.sum(1.0, neg_inv_base))

one_minus_inv = oti.sum(1.0, neg_inv_base)

# sum(neg_log, one_minus_inv)
bench("sum(neg_log, 1-1/base) [sum_oo]", lambda: oti.sum(neg_log_base, one_minus_inv))

term = oti.sum(neg_log_base, one_minus_inv)

# mul(phi, term)
bench("mul(phi, term)        [mul_oo]", lambda: oti.mul(phi, term))

phi_term = oti.mul(phi, term)

# mul(alpha_factor, phi_term)
alpha_factor = ln10 * alpha_rq
bench("mul(alpha_factor, ..) [mul_ro]", lambda: oti.mul(alpha_factor, phi_term))

print()
print("=== Iso ell gradient (if it were iso) ===")
# mul(sum_sq, phi_over_base)
# For iso, we'd also need sum_sq * phi_over_base then scale
bench("mul(sum_sq_proxy, phi_over_base) [mul_oo]", lambda: oti.mul(r2, phi_over_base))
bench("mul(scalar, ...)       [mul_ro]", lambda: oti.mul(-ln10 * ell[0]**2, oti.mul(r2, phi_over_base)))

print()
print("=== Summary: potential fusions ===")
print("The negate operations (mul by -1) could be replaced by oti.neg() if available,")
print("or the entire alpha term could be fused into one C-level pass.")
print()

# Check if neg exists
print(f"Has oti.neg: {hasattr(oti, 'neg')}")
print(f"Has oti.sub: {hasattr(oti, 'sub')}")
