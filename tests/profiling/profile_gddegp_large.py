import numpy as np
import cProfile
import pstats
from jetgp.full_gddegp.gddegp import gddegp
from jetgp.full_gddegp.optimizer import Optimizer

np.random.seed(42)
D = 20
N = 100

X = np.random.rand(N, D)

# y_train = [function values, directional_derivatives]
y_func = np.random.rand(N, 1)
y_dir = np.random.rand(N, 1)
y_train = [y_func, y_dir]

# Generate per-point random rays (one direction type, N points)
# rays_list[i] has shape (D, n_points_with_direction_i)
rays_array = np.random.randn(D, N)
# Normalize each column to unit length
norms = np.linalg.norm(rays_array, axis=0, keepdims=True)
rays_array = rays_array / norms
rays_list = [rays_array]  # one direction type

# der_indices: one group, one direction
der_indices = [
    [[[1, 1]]]
]

# derivative_locations: all points have directional derivative
deriv_locs = [list(range(N))]

model = gddegp(X, y_train, n_order=1,
               rays_list=rays_list,
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
