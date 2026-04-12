import time, numpy as np
import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.wdegp.wdegp import wdegp
from jetgp.wdegp.optimizer import Optimizer
from jetgp.wdegp.wdegp_utils import _assemble_kernel_numba
import jetgp.utils as utils

np.random.seed(42)
DIM = 20
N = 200

sampler = LatinHypercube(d=DIM, seed=1000)
X_train = sampler.random(n=N)
y_vals = morris(X_train)
grads = morris_gradient(X_train)

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

model = wdegp(
    X_train, submodel_data,
    1, DIM,
    derivative_specs_list,
    derivative_locations=derivative_locations_list,
    normalize=True,
    kernel="SE", kernel_type="anisotropic",
)
opt = Optimizer(model)
x0 = np.array([0.1] * (len(model.bounds) - 2) + [0.5, -3.0])


# Warmup
for _ in range(3):
    opt.nll_and_grad(x0)

ITERS = 10
t0 = time.perf_counter_ns()
for _ in range(ITERS):
    opt.nll_and_grad(x0)
total_ns = time.perf_counter_ns() - t0
print(f"nll_and_grad: {total_ns / ITERS / 1e6:.2f} ms/call")