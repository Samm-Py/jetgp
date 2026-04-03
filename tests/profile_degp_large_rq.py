"""
Quick large-scale DEGP profile with RQ kernel: n_train=200, DIM=20 → 4200×4200 kernel.
Just profiles nll_and_grad calls (no full optimization).
"""
import numpy as np
import cProfile
import pstats
import time
import sys
sys.path.insert(0, '.')

from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.full_degp.degp import degp

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
print(f"Kernel matrix size: {N_TRAIN + DIM*N_TRAIN} x {N_TRAIN + DIM*N_TRAIN}")

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

print(f"\nTiming single nll_and_grad call...")
t0 = time.perf_counter()
nll_val, grad = opt.nll_and_grad(x0)
t1 = time.perf_counter()
print(f"  nll_and_grad: {t1-t0:.4f}s  (NLL={nll_val:.4f}, |grad|={np.linalg.norm(grad):.4f})")

print(f"\n{'='*60}")
print(f"  cProfile: 3 calls to nll_and_grad")
print(f"{'='*60}")
profiler = cProfile.Profile()
profiler.enable()
for _ in range(3):
    opt.nll_and_grad(x0)
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('tottime')
stats.print_stats(30)
