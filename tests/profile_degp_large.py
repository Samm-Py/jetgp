import numpy as np
import cProfile
import pstats
from jetgp.full_degp.degp import degp
from jetgp.full_degp.optimizer import Optimizer
from jetgp.utils import gen_OTI_indices

np.random.seed(42)
D = 20
N = 100

X = np.random.rand(N, D)

# y_train = [function values, grad_dim1, grad_dim2, ..., grad_dimD]
y_func = np.random.rand(N, 1)
y_train = [y_func]
for d in range(D):
    y_train.append(np.random.rand(N, 1))

# der_indices for n_order=1: one list of [basis, order] pairs
der_indices = gen_OTI_indices(D, 1)

# derivative_locations: all derivatives at all points
deriv_locs = [[i for i in range(N)]] * D

model = degp(X, y_train, n_order=1, n_bases=D,
             der_indices=der_indices,
             derivative_locations=deriv_locs,
             normalize=True, kernel='SE', kernel_type='anisotropic')

opt = Optimizer(model)
x0 = np.array([0.1] * (len(model.bounds) - 2) + [0.5, -3.0])

# Warm up
opt.nll_and_grad(x0)

# Profile
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