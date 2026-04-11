"""
Profile: DEGP on random 20D data with n_train=100.

Run with:
    kernprof -l -v profile_degp_large.py
"""

import numpy as np
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

# Register negative_log_marginal_likelihood for line-level profiling
try:
    profile.add_function(opt.negative_log_marginal_likelihood)
except NameError:
    pass  # not running under kernprof

# Warm up
opt.nll_and_grad(x0)
opt.nll_wrapper(x0)


@profile
def run_nll_and_grad():
    for _ in range(10):
        opt.nll_and_grad(x0)


@profile
def run_nlml():
    for _ in range(10):
        opt.nll_wrapper(x0)


print("Profiling nll_and_grad (10 iterations)...")
run_nll_and_grad()

print("Profiling negative_log_marginal_likelihood (10 iterations)...")
run_nlml()
