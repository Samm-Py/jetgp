"""
Test script comparing full_gddegp (2 bases per direction) vs gddegp (1 base per direction).
Both should produce identical predictions and covariances.
"""
import numpy as np
import warnings

# Old (2 bases per direction)
from jetgp.full_gddegp.gddegp import gddegp as gddegp_old

# New (1 base per direction, stationarity sign corrections)
from jetgp.gddegp.gddegp import gddegp as gddegp_new

np.random.seed(42)

# --- Setup ---
d = 2
n_train = 20
n_test = 10

X_train = np.random.uniform(0, 2 * np.pi, (n_train, d))
X_test = np.random.uniform(0, 2 * np.pi, (n_test, d))

def f(X):
    return np.sin(X[:, 0]) * np.cos(X[:, 1])

def df_dx1(X):
    return np.cos(X[:, 0]) * np.cos(X[:, 1])

def df_dx2(X):
    return -np.sin(X[:, 0]) * np.sin(X[:, 1])

y_func = f(X_train).reshape(-1, 1)
y_dx1 = df_dx1(X_train).reshape(-1, 1)
y_dx2 = df_dx2(X_train).reshape(-1, 1)
y_train = [y_func, y_dx1, y_dx2]

# Direction 1 = x1-axis, Direction 2 = x2-axis (at all points)
rays_list = [
    np.tile([[1.0], [0.0]], (1, n_train)),  # shape (2, n_train)
    np.tile([[0.0], [1.0]], (1, n_train)),  # shape (2, n_train)
]
der_indices = [[[[1, 1]], [[2, 1]]]]  # GDDEGP format: extra nesting level
derivative_locations = [list(range(n_train)), list(range(n_train))]

# Prediction rays along coordinate axes
rays_predict = [
    np.tile([[np.sqrt(2)/2], [np.sqrt(2)/2]], (1, n_test)),
    np.tile([[-np.sqrt(2)/2], [np.sqrt(2)/2]], (1, n_test)),
]

derivs_to_predict = [[[1, 1]], [[2, 1]]]

print("=" * 60)
print("Test 1: Training with derivatives, predicting derivatives")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    # Old model: n_bases = 2 * 2 = 4
    model_old = gddegp_old(
        X_train, y_train, n_order=1,
        rays_list=rays_list, der_indices=der_indices,
        derivative_locations=derivative_locations,
        normalize=True, kernel="SE", kernel_type="anisotropic",
    )

    # New model: n_bases = 2
    model_new = gddegp_new(
        X_train, y_train, n_order=1,
        rays_list=rays_list, der_indices=der_indices,
        derivative_locations=derivative_locations,
        normalize=True, kernel="SE", kernel_type="anisotropic",
    )

print(f"Old n_bases: {model_old.n_bases}")
print(f"New n_bases: {model_new.n_bases}")

# Use same hyperparameters for comparison
params_old = model_old.optimize_hyperparameters(optimizer='powell', n_restart_optimizer=3, debug=False)
print(f"\nOptimized params (old): {params_old}")

# Use same params for both
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    pred_old = model_old.predict(
        X_test, params_old, rays_predict=rays_predict,
        calc_cov=True, return_deriv=True, derivs_to_predict=derivs_to_predict
    )
    pred_new = model_new.predict(
        X_test, params_old, rays_predict=rays_predict,
        calc_cov=True, return_deriv=True, derivs_to_predict=derivs_to_predict
    )

mean_old, var_old = pred_old
mean_new, var_new = pred_new

print(f"\nMean shapes: old={mean_old.shape}, new={mean_new.shape}")
print(f"Var shapes:  old={var_old.shape}, new={var_new.shape}")

mean_diff = np.max(np.abs(mean_old - mean_new))
var_diff = np.max(np.abs(var_old - var_new))
print(f"\nMax |mean_old - mean_new|: {mean_diff:.2e}")
print(f"Max |var_old  - var_new|:  {var_diff:.2e}")

# Check accuracy against true values
f_true = f(X_test)
dx1_true = df_dx1(X_test)
dx2_true = df_dx2(X_test)

print(f"\nOld model RMSE:")
print(f"  f:     {np.sqrt(np.mean((mean_old[0] - f_true)**2)):.4e}")
print(f"  df/dx1: {np.sqrt(np.mean((mean_old[1] - dx1_true)**2)):.4e}")
print(f"  df/dx2: {np.sqrt(np.mean((mean_old[2] - dx2_true)**2)):.4e}")

print(f"\nNew model RMSE:")
print(f"  f:     {np.sqrt(np.mean((mean_new[0] - f_true)**2)):.4e}")
print(f"  df/dx1: {np.sqrt(np.mean((mean_new[1] - dx1_true)**2)):.4e}")
print(f"  df/dx2: {np.sqrt(np.mean((mean_new[2] - dx2_true)**2)):.4e}")

if mean_diff < 1e-8 and var_diff < 1e-8:
    print("\nPASS: Old and new implementations agree within 1e-8")
else:
    print(f"\nFAIL: Implementations differ (mean: {mean_diff:.2e}, var: {var_diff:.2e})")

# --- Test 2: Function-only training ---
print("\n" + "=" * 60)
print("Test 2: Function-only training, predicting derivatives")
print("=" * 60)

y_train_func_only = [y_func]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    model_new_fo = gddegp_new(
        X_train, y_train_func_only, n_order=1,
        rays_list=[], der_indices=[], derivative_locations=[],
        n_bases=2,  # new: n_prediction_directions (halved!)
        normalize=True, kernel="SE", kernel_type="anisotropic",
    )

print(f"New n_bases: {model_new_fo.n_bases}")

params_fo = model_new_fo.optimize_hyperparameters(optimizer='powell', n_restart_optimizer=3, debug=False)
print(f"\nOptimized params: {params_fo}")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    pred_new_fo = model_new_fo.predict(
        X_test, params_fo, rays_predict=rays_predict,
        calc_cov=True, return_deriv=True, derivs_to_predict=derivs_to_predict
    )

mean_new_fo, var_new_fo = pred_new_fo

f_true = f(X_test)
dx1_true = df_dx1(X_test)
dx2_true = df_dx2(X_test)

rmse_f = np.sqrt(np.mean((mean_new_fo[0] - f_true)**2))
rmse_dx1 = np.sqrt(np.mean((mean_new_fo[1] - dx1_true)**2))
rmse_dx2 = np.sqrt(np.mean((mean_new_fo[2] - dx2_true)**2))
corr_dx1 = np.corrcoef(mean_new_fo[1], dx1_true)[0, 1]
corr_dx2 = np.corrcoef(mean_new_fo[2], dx2_true)[0, 1]

print(f"\nNew model (function-only, n_bases=2) RMSE:")
print(f"  f:     {rmse_f:.4e}")
print(f"  df/dx1: {rmse_dx1:.4e} (corr: {corr_dx1:.4f})")
print(f"  df/dx2: {rmse_dx2:.4e} (corr: {corr_dx2:.4f})")

if corr_dx1 > 0.8 and corr_dx2 > 0.8:
    print("\nPASS: Function-only derivative predictions have reasonable correlation")
else:
    print(f"\nFAIL: Poor correlation (dx1: {corr_dx1:.4f}, dx2: {corr_dx2:.4f})")
