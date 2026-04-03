"""
Profile WDEGP prediction to identify bottlenecks.
Uses Morris 20D: n_train=20, 3 submodels, predict at 100 test points.
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
N_TEST = 100
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"

# Generate training data
sampler = LatinHypercube(d=DIM, seed=42)
X_train = sampler.random(n=N_TRAIN)
y_vals = morris(X_train)
grads = morris_gradient(X_train)

y_all_col = y_vals.reshape(-1, 1)
der_specs = utils.gen_OTI_indices(DIM, 1)

# Build 3 submodels with different derivative coverage
submodel_data = []
derivative_specs_list = []
derivative_locations_list = []

# Submodel 1: first 7 points, function only
sub1_idx = list(range(7))
submodel_data.append([y_all_col])
derivative_specs_list.append([])
derivative_locations_list.append([[]])

# Submodel 2: next 7 points, function + gradients
sub2_idx = list(range(7, 14))
data_2 = [y_all_col] + [grads[sub2_idx, j:j+1] for j in range(DIM)]
submodel_data.append(data_2)
derivative_specs_list.append(der_specs)
derivative_locations_list.append([sub2_idx for _ in range(DIM)])

# Submodel 3: last 6 points, function + gradients
sub3_idx = list(range(14, 20))
data_3 = [y_all_col] + [grads[sub3_idx, j:j+1] for j in range(DIM)]
submodel_data.append(data_3)
derivative_specs_list.append(der_specs)
derivative_locations_list.append([sub3_idx for _ in range(DIM)])

print(f"Building WDEGP model: n_train={N_TRAIN}, DIM={DIM}, 3 submodels")
model = wdegp(
    X_train, submodel_data,
    1, DIM,
    derivative_specs_list,
    derivative_locations=derivative_locations_list,
    normalize=True,
    kernel=KERNEL, kernel_type=KERNEL_TYPE
)

# Quick optimization (few generations, just to get valid params)
print("Optimizing hyperparameters (quick)...")
params = model.optimize_hyperparameters(
    optimizer="jade",
    n_generations=5,
    local_opt_every=5,
    pop_size=20,
    debug=False
)
print(f"Params: {params}")

# Generate test data
np.random.seed(99)
X_test = np.random.uniform(0, 1, (N_TEST, DIM))

# --- Time single predict calls ---
print(f"\n{'='*60}")
print(f"Timing predict (no cov, no deriv) at {N_TEST} points")
print(f"{'='*60}")
t0 = time.perf_counter()
y_pred = model.predict(X_test, params, calc_cov=False, return_deriv=False)
t1 = time.perf_counter()
print(f"  predict (no cov): {t1-t0:.4f}s")

print(f"\nTiming predict (with cov, no deriv) at {N_TEST} points")
t0 = time.perf_counter()
y_pred, y_var = model.predict(X_test, params, calc_cov=True, return_deriv=False)
t1 = time.perf_counter()
print(f"  predict (with cov): {t1-t0:.4f}s")

# --- cProfile the predict call ---
print(f"\n{'='*60}")
print(f"cProfile: 5 predict calls (no cov)")
print(f"{'='*60}")
profiler = cProfile.Profile()
profiler.enable()
for _ in range(5):
    model.predict(X_test, params, calc_cov=False, return_deriv=False)
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('tottime')
stats.print_stats(30)

print(f"\n{'='*60}")
print(f"cProfile: 5 predict calls (with cov)")
print(f"{'='*60}")
profiler2 = cProfile.Profile()
profiler2.enable()
for _ in range(5):
    model.predict(X_test, params, calc_cov=True, return_deriv=False)
profiler2.disable()

stats2 = pstats.Stats(profiler2)
stats2.sort_stats('tottime')
stats2.print_stats(30)
