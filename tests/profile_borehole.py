"""
Profile a single borehole DEGP run to find where OMP overhead is.
"""
import cProfile
import pstats
import io
import sys
sys.path.insert(0, '.')

import numpy as np
from benchmark_functions import borehole, borehole_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.full_degp.degp import degp

DIM = 8
N_TRAIN = 80
SEED = 1000

sampler = LatinHypercube(d=DIM, seed=SEED)
X_train = sampler.random(n=N_TRAIN)
y_vals = borehole(X_train)
grads = borehole_gradient(X_train)

y_train_list = [y_vals.reshape(-1, 1)]
for j in range(DIM):
    y_train_list.append(grads[:, j].reshape(-1, 1))

DER_INDICES = [[[[i, 1]] for i in range(1, DIM + 1)]]

model = degp(
    X_train, y_train_list,
    n_order=1, n_bases=DIM,
    der_indices=DER_INDICES,
    kernel="SE", kernel_type="anisotropic",
)

pr = cProfile.Profile()
pr.enable()
params = model.optimize_hyperparameters(
    optimizer="jade",
    n_generations=10,
    local_opt_every=10,
    pop_size=20,
    debug=False
)
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
ps.print_stats(40)
print(s.getvalue())

print("\n\n=== Sorted by tottime ===\n")
s2 = io.StringIO()
ps2 = pstats.Stats(pr, stream=s2).sort_stats('tottime')
ps2.print_stats(40)
print(s2.getvalue())
