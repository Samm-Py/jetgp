"""
Profile: WDEGP on Morris 20D with n_train=200.
One submodel per training point; each gets all function values + its own gradient.

Run with:
    kernprof -l -v profile_wdegp_morris.py
"""

import os
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.wdegp.wdegp import wdegp
import jetgp.utils as utils

np.random.seed(42)
DIM = 20
N = 200
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"

# Generate training data
sampler = LatinHypercube(d=DIM, seed=1000)
X_train = sampler.random(n=N)
y_vals = morris(X_train)
grads = morris_gradient(X_train)

# Build WDEGP submodel structure
y_all_col = y_vals.reshape(-1, 1)
der_specs = utils.gen_OTI_indices(DIM, 1)

submodel_data = []
derivative_specs_list = []
derivative_locations_list = []

for i in range(N):
    data_i = [y_all_col] + [grads[i:i+1, j:j+1] for j in range(DIM)]
    submodel_data.append(data_i)
    derivative_specs_list.append(der_specs)
    derivative_locations_list.append([[i] for _ in range(DIM)])

print(f"WDEGP Morris: D={DIM}, N={N}, submodels={N}")
print(f"Each submodel K size: {N+DIM} x {N+DIM} = {(N+DIM)**2} elements")
print("Building model...")

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
print("Warming up...")
opt.nll_and_grad(x0)

@profile
def run_nll_and_grad():
    for _ in range(3):
        opt.nll_and_grad(x0)

@profile
def run_nlml():
    for _ in range(3):
        opt.nll_wrapper(x0)

print("Profiling nll_and_grad (3 iterations)...")
run_nll_and_grad()

print("Profiling negative_log_marginal_likelihood (3 iterations)...")
run_nlml()
