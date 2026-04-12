"""
Debug script: investigate SDEGP optimization convergence on 2D Branin-Hoo.

Compares SDEGP and DEGP optimization with aggressive settings (large population,
many generations, local optimization) to check whether both converge to similar
hyperparameters and NLL values.
"""

import numpy as np
import sympy as sp
from scipy.stats import qmc

from jetgp.full_degp.degp import degp
from jetgp.sdegp.sdegp import sdegp
import jetgp.utils as utils

# ---------------------------------------------------------------------------
# Branin-Hoo setup
# ---------------------------------------------------------------------------
x1_sym, x2_sym = sp.symbols("x1 x2", real=True)
a, b, c, r, s, t = (
    1.0, 5.1 / (4 * sp.pi**2), 5.0 / sp.pi,
    6.0, 10.0, 1.0 / (8 * sp.pi),
)
f_sym = a * (x2_sym - b * x1_sym**2 + c * x1_sym - r)**2 + \
        s * (1 - t) * sp.cos(x1_sym) + s

f_np = sp.lambdify([x1_sym, x2_sym], f_sym, "numpy")
g1_np = sp.lambdify([x1_sym, x2_sym], sp.diff(f_sym, x1_sym), "numpy")
g2_np = sp.lambdify([x1_sym, x2_sym], sp.diff(f_sym, x2_sym), "numpy")

# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------
N_TRAIN = 15
DIM = 2
BOUNDS_LO = np.array([-5.0, 0.0])
BOUNDS_HI = np.array([10.0, 15.0])

sampler = qmc.LatinHypercube(d=DIM, seed=42)
X_train = qmc.scale(sampler.random(n=N_TRAIN), BOUNDS_LO, BOUNDS_HI)

y_train = np.atleast_1d(f_np(X_train[:, 0], X_train[:, 1])).reshape(-1, 1)
g1 = np.atleast_1d(g1_np(X_train[:, 0], X_train[:, 1])).reshape(-1, 1)
g2 = np.atleast_1d(g2_np(X_train[:, 0], X_train[:, 1])).reshape(-1, 1)
grads = np.hstack([g1, g2])

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
N_TEST = 500
sampler_test = qmc.LatinHypercube(d=DIM, seed=99)
X_test = qmc.scale(sampler_test.random(n=N_TEST), BOUNDS_LO, BOUNDS_HI)
y_test = np.atleast_1d(f_np(X_test[:, 0], X_test[:, 1]))

# ---------------------------------------------------------------------------
# Build models
# ---------------------------------------------------------------------------
M = 5

print("Building SDEGP model...")
sdegp_model = sdegp(
    X_train, y_train, grads,
    n_order=1, m=M,
    kernel="SE", kernel_type="anisotropic",
)
print(f"  {sdegp_model.num_submodels} submodels, slice_dim={sdegp_model.slice_dim}")
print(f"  Bounds: {sdegp_model.bounds}")

print("\nBuilding DEGP model...")
der_specs = utils.gen_OTI_indices(DIM, 1)
all_pts = list(range(N_TRAIN))
degp_model = degp(
    X_train,
    [y_train, g1, g2],
    n_order=1, n_bases=DIM,
    der_indices=der_specs,
    derivative_locations=[all_pts, all_pts],
    normalize=True,
    kernel="SE", kernel_type="anisotropic",
)
print(f"  Bounds: {degp_model.bounds}")

# ---------------------------------------------------------------------------
# 1) Evaluate NLL at a grid of points to compare landscapes
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  NLL comparison at fixed points")
print("=" * 60)

test_points = [
    np.array([0.0, 0.0, 0.0, -4.0]),
    np.array([0.5, 0.5, 0.0, -4.0]),
    np.array([-0.5, -0.5, 0.0, -4.0]),
    np.array([0.3, 0.3, 0.5, -3.0]),
    np.array([1.0, 1.0, 0.0, -5.0]),
]

for x0 in test_points:
    nll_degp = degp_model.optimizer.nll_wrapper(x0)
    nll_sdegp = sdegp_model.optimizer.nll_wrapper(x0)
    print(f"  x0={x0}  DEGP={nll_degp:10.4f}  SDEGP={nll_sdegp:10.4f}  "
          f"diff={nll_sdegp - nll_degp:+.4f}")

# ---------------------------------------------------------------------------
# 2) Gradient comparison at a single point
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Gradient comparison")
print("=" * 60)

x0 = np.array([0.3, 0.3, 0.0, -4.0])
nll_d, grad_d = degp_model.optimizer.nll_and_grad(x0)
nll_s, grad_s = sdegp_model.optimizer.nll_and_grad(x0)

print(f"  DEGP  NLL={nll_d:.6f}  grad={grad_d}")
print(f"  SDEGP NLL={nll_s:.6f}  grad={grad_s}")

# FD check for SDEGP
eps = 1e-5
grad_fd = np.zeros_like(x0)
for i in range(len(x0)):
    xp = x0.copy(); xp[i] += eps
    xm = x0.copy(); xm[i] -= eps
    grad_fd[i] = (sdegp_model.optimizer.nll_wrapper(xp) -
                  sdegp_model.optimizer.nll_wrapper(xm)) / (2 * eps)
print(f"  SDEGP FD  grad={grad_fd}")

rel_err = np.abs(grad_s - grad_fd) / np.maximum(np.abs(grad_fd), 1e-12)
print(f"  SDEGP analytic vs FD rel error: {rel_err}")

# ---------------------------------------------------------------------------
# 3) Optimize with aggressive settings
# ---------------------------------------------------------------------------
import time

opt_settings = dict(
    optimizer="jade",
    pop_size=80,
    n_generations=50,
    local_opt_every=10,
    debug=True,
)

print("\n" + "=" * 60)
print(f"  Optimizing DEGP (jade, pop=80, gen=50, local every 10)")
print("=" * 60)
t0 = time.perf_counter()
params_degp = degp_model.optimize_hyperparameters(**opt_settings)
t_degp = time.perf_counter() - t0
nll_degp_opt = degp_model.optimizer.nll_wrapper(params_degp)
print(f"\n  DEGP params:  {params_degp}")
print(f"  DEGP NLL:     {nll_degp_opt:.6f}")
print(f"  DEGP time:    {t_degp:.2f}s")

print("\n" + "=" * 60)
print(f"  Optimizing SDEGP (jade, pop=80, gen=50, local every 10)")
print("=" * 60)
t0 = time.perf_counter()
params_sdegp = sdegp_model.optimize_hyperparameters(**opt_settings)
t_sdegp = time.perf_counter() - t0
nll_sdegp_opt = sdegp_model.optimizer.nll_wrapper(params_sdegp)
print(f"\n  SDEGP params: {params_sdegp}")
print(f"  SDEGP NLL:    {nll_sdegp_opt:.6f}")
print(f"  SDEGP time:   {t_sdegp:.2f}s")

# ---------------------------------------------------------------------------
# 4) Cross-evaluate: SDEGP params on DEGP NLL and vice versa
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Cross-evaluation")
print("=" * 60)
print(f"  DEGP  NLL at DEGP  params: {degp_model.optimizer.nll_wrapper(params_degp):.6f}")
print(f"  DEGP  NLL at SDEGP params: {degp_model.optimizer.nll_wrapper(params_sdegp):.6f}")
print(f"  SDEGP NLL at DEGP  params: {sdegp_model.optimizer.nll_wrapper(params_degp):.6f}")
print(f"  SDEGP NLL at SDEGP params: {sdegp_model.optimizer.nll_wrapper(params_sdegp):.6f}")

# ---------------------------------------------------------------------------
# 5) Prediction comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Prediction comparison")
print("=" * 60)

y_pred_degp = degp_model.predict(X_test, params_degp, calc_cov=False, return_deriv=False)
if isinstance(y_pred_degp, tuple):
    y_pred_degp = y_pred_degp[0]
y_pred_degp = np.asarray(y_pred_degp).flatten()

y_pred_sdegp = sdegp_model.predict(X_test, params_sdegp, calc_cov=False, return_deriv=False)
if isinstance(y_pred_sdegp, tuple):
    y_pred_sdegp = y_pred_sdegp[0]
y_pred_sdegp = np.asarray(y_pred_sdegp).flatten()

# Also: SDEGP model predicting with DEGP-optimized params
y_pred_sdegp_degp_params = sdegp_model.predict(
    X_test, params_degp, calc_cov=False, return_deriv=False
)
if isinstance(y_pred_sdegp_degp_params, tuple):
    y_pred_sdegp_degp_params = y_pred_sdegp_degp_params[0]
y_pred_sdegp_degp_params = np.asarray(y_pred_sdegp_degp_params).flatten()

def nrmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2)) / np.std(y_true)

print(f"  DEGP  (own params)  NRMSE: {nrmse(y_test, y_pred_degp):.6f}")
print(f"  SDEGP (own params)  NRMSE: {nrmse(y_test, y_pred_sdegp):.6f}")
print(f"  SDEGP (DEGP params) NRMSE: {nrmse(y_test, y_pred_sdegp_degp_params):.6f}")

# ---------------------------------------------------------------------------
# 6) Parameter diff
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("  Parameter comparison")
print("=" * 60)
labels = [f"theta_{i+1}" for i in range(len(params_degp) - 2)] + ["sigma_f", "sigma_n"]
for lbl, pd, ps in zip(labels, params_degp, params_sdegp):
    print(f"  {lbl:10s}  DEGP={pd:+8.4f}  SDEGP={ps:+8.4f}  diff={ps-pd:+8.4f}")
