"""
Profile DDEGP optimization to identify bottlenecks.
Uses Morris 20D: n_train=20, 2 random directional derivatives.
"""

import numpy as np
import cProfile
import pstats
import time
import sys
sys.path.insert(0, '.')

from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from scipy.stats import ortho_group
from jetgp.full_ddegp.ddegp import ddegp

DIM = 20
N_TRAIN = 20
N_RAYS = 2
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"

# Generate data
sampler = LatinHypercube(d=DIM, seed=42)
X_train = sampler.random(n=N_TRAIN)
y_vals = morris(X_train)
grads = morris_gradient(X_train)

# Random orthonormal directions
rng = np.random.default_rng(42)
Q = ortho_group.rvs(DIM, random_state=rng)
rays = Q[:, :N_RAYS]  # (DIM, N_RAYS)

# Directional derivatives
dir_derivs = grads @ rays  # (N_TRAIN, N_RAYS)

# y_train: [func_values, dir_deriv_1, ..., dir_deriv_k]
y_train = [y_vals.reshape(-1, 1)] + [dir_derivs[:, r:r+1] for r in range(N_RAYS)]

# der_indices: one per ray direction
der_indices = [[[[r + 1, 1]]] for r in range(N_RAYS)]
derivative_locations = [list(range(N_TRAIN)) for _ in range(N_RAYS)]

print(f"Building DDEGP model: n_train={N_TRAIN}, DIM={DIM}, n_rays={N_RAYS}")
model = ddegp(
    X_train, y_train,
    n_order=1,
    der_indices=der_indices,
    rays=rays,
    derivative_locations=derivative_locations,
    normalize=True,
    kernel=KERNEL, kernel_type=KERNEL_TYPE
)

opt = model.optimizer
x0 = np.zeros(len(model.bounds))
for i, b in enumerate(model.bounds):
    x0[i] = 0.5 * (b[0] + b[1])

print(f"\nTiming single nll_wrapper call...")
t0 = time.perf_counter()
nll_val = opt.nll_wrapper(x0)
t1 = time.perf_counter()
print(f"  nll_wrapper: {t1-t0:.4f}s  (NLL={nll_val:.4f})")

print(f"\nTiming single nll_and_grad call...")
t0 = time.perf_counter()
nll_val, grad = opt.nll_and_grad(x0)
t1 = time.perf_counter()
print(f"  nll_and_grad: {t1-t0:.4f}s  (NLL={nll_val:.4f}, |grad|={np.linalg.norm(grad):.4f})")

print(f"\n{'='*60}")
print(f"  cProfile: 10 calls to nll_and_grad")
print(f"{'='*60}")
profiler = cProfile.Profile()
profiler.enable()
for _ in range(10):
    opt.nll_and_grad(x0)
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(30)

print(f"\n{'='*60}")
print(f"  cProfile: 10 calls to nll_and_grad (sorted by tottime)")
print(f"{'='*60}")
stats.sort_stats('tottime')
stats.print_stats(30)

print(f"\n{'='*60}")
print(f"  cProfile: full optimize_hyperparameters (jade, 5 gen)")
print(f"{'='*60}")
profiler2 = cProfile.Profile()
profiler2.enable()
params = model.optimize_hyperparameters(
    optimizer="jade",
    n_generations=5,
    local_opt_every=5,
    pop_size=10,
    debug=True
)
profiler2.disable()

stats2 = pstats.Stats(profiler2)
stats2.sort_stats('tottime')
stats2.print_stats(30)
