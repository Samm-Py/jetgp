import numpy as np
import cProfile
import pstats
from jetgp.full_degp.degp import degp
from jetgp.full_degp.optimizer import Optimizer
from jetgp.utils import gen_OTI_indices

np.random.seed(42)
D = 10
N = 30
n_order = 2

X = np.random.rand(N, D)

# y_train = [function values, grad_dim1, ..., grad_dimD, hess_11, hess_12, ...]
y_func = np.random.rand(N, 1)
y_train = [y_func]

# 1st order derivatives
der_indices = gen_OTI_indices(D, n_order)
n_first = D
n_second = D * (D + 1) // 2  # unique 2nd order partials
n_derivs = n_first + n_second

for d in range(n_derivs):
    y_train.append(np.random.rand(N, 1))

# derivative_locations: all derivatives at all points
deriv_locs = [[i for i in range(N)]] * n_derivs

model = degp(X, y_train, n_order=n_order, n_bases=D,
             der_indices=der_indices,
             derivative_locations=deriv_locs,
             normalize=True, kernel='SE', kernel_type='anisotropic')

opt = Optimizer(model)
x0 = np.array([0.1] * (len(model.bounds) - 2) + [0.5, -3.0])

# Warm up
print(f"m{D}n{2*n_order}, D={D}, N={N}, n_derivs={n_derivs}")
print(f"K size: {len(model.y_train)} x {len(model.y_train)}")
print("Warming up...")
opt.nll_and_grad(x0)

# Profile
print("Profiling (10 iterations)...")
pr = cProfile.Profile()
pr.enable()
for _ in range(10):
    opt.nll_and_grad(x0)
pr.disable()

stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(30)

print("\n--- Sorted by tottime ---")
stats.sort_stats('tottime')
stats.print_stats(30)
