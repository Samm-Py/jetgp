"""
Profile WDEGP optimization to identify bottlenecks.
Uses a small Morris 20D case: n_train=20 (=DIM), all points with gradients.
"""

import numpy as np
import cProfile
import pstats
import time
import sys
sys.path.insert(0, '.')

from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.wdegp.wdegp import wdegp
import jetgp.utils as utils

DIM = 20
N_TRAIN = 20
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"

# Generate data
sampler = LatinHypercube(d=DIM, seed=42)
X_train = sampler.random(n=N_TRAIN)
y_vals = morris(X_train)
grads = morris_gradient(X_train)

y_all_col = y_vals.reshape(-1, 1)
der_specs = utils.gen_OTI_indices(DIM, 1)

submodel_data = []
derivative_specs_list = []
derivative_locations_list = []

for i in range(N_TRAIN):
    data_i = [y_all_col] + [grads[i:i+1, j:j+1] for j in range(DIM)]
    submodel_data.append(data_i)
    derivative_specs_list.append(der_specs)
    derivative_locations_list.append([[i] for _ in range(DIM)])

print(f"Building WDEGP model: n_train={N_TRAIN}, DIM={DIM}")
model = wdegp(
    X_train, submodel_data,
    1, DIM,
    derivative_specs_list,
    derivative_locations=derivative_locations_list,
    normalize=True,
    kernel=KERNEL, kernel_type=KERNEL_TYPE
)

# Time a single NLL + grad call
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

# cProfile the nll_and_grad call
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

# Profile full optimization
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
