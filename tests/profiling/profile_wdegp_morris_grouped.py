"""
Profile: WDEGP on Morris 20D with n_train=200, grouped submodels.
N_SUB submodels, each gets all function values + gradients at (n_train/N_SUB) points.
"""

import os
import numpy as np
import cProfile
import pstats
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from jetgp.wdegp.wdegp import wdegp
import jetgp.utils as utils

np.random.seed(42)
DIM = 20
N = 200
N_SUB = 40  # 5 gradients per submodel
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"

# Generate training data
sampler = LatinHypercube(d=DIM, seed=1000)
X_train = sampler.random(n=N)
y_vals = morris(X_train)
grads = morris_gradient(X_train)

# Build WDEGP submodel structure: N_SUB submodels, each with all function
# values + gradients at (N // N_SUB) points
y_all_col = y_vals.reshape(-1, 1)
der_specs = utils.gen_OTI_indices(DIM, 1)
pts_per_sub = N // N_SUB

# Balanced k-means grouping
kmeans = KMeans(n_clusters=N_SUB, random_state=42, n_init=10)
kmeans.fit(X_train)
dists = cdist(X_train, kmeans.cluster_centers_)
cluster_ranks = np.argsort(dists, axis=1)
groups = [[] for _ in range(N_SUB)]
point_order = np.argsort(dists.min(axis=1))
for pt in point_order:
    for c in cluster_ranks[pt]:
        if len(groups[c]) < pts_per_sub:
            groups[c].append(pt)
            break

submodel_data = []
derivative_specs_list = []
derivative_locations_list = []

for s in range(N_SUB):
    idx = sorted(groups[s])
    data_s = [y_all_col]
    for j in range(DIM):
        data_s.append(grads[idx, j:j+1])
    submodel_data.append(data_s)
    derivative_specs_list.append(der_specs)
    derivative_locations_list.append([idx for _ in range(DIM)])

print(f"WDEGP Morris grouped: D={DIM}, N={N}, submodels={N_SUB}, pts_per_sub={pts_per_sub}")
print(f"Each submodel K size: {N + DIM * pts_per_sub} x {N + DIM * pts_per_sub}")
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

# Profile
print("Profiling (3 iterations)...")
pr = cProfile.Profile()
pr.enable()
for _ in range(3):
    opt.nll_and_grad(x0)
pr.disable()

stats = pstats.Stats(pr)
stats.sort_stats('cumulative')
stats.print_stats(30)

print("\n--- Sorted by tottime ---")
stats.sort_stats('tottime')
stats.print_stats(30)
