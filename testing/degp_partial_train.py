"""
Experiment: 4-variable DEGP trained with only 2 partial derivatives.

Function:  f(x1, x2, x3, x4) = sin(x1) + cos(x2) + x3^2 + 0.5*x4^3
Training:  f values  +  df/dx1  +  df/dx2  at all points
           (df/dx3 and df/dx4 are intentionally withheld)

Questions:
  1. How well does the partial DEGP predict function values?
  2. How well does it predict the trained derivatives (dx1, dx2)?
  3. What happens when we try to predict untrained derivatives (dx3, dx4)?
  4. How does a full DEGP (all 4 derivatives) compare?
"""

import warnings
import numpy as np
from jetgp.full_degp.degp import degp


# ---------------------------------------------------------------------------
# True function and its partial derivatives
# ---------------------------------------------------------------------------

def f(X):
    return np.sin(X[:, 0]) + np.cos(X[:, 1]) + X[:, 2] ** 2 + 0.5 * X[:, 3] ** 3

def df_dx1(X): return  np.cos(X[:, 0])
def df_dx2(X): return -np.sin(X[:, 1])
def df_dx3(X): return  2.0 * X[:, 2]
def df_dx4(X): return  1.5 * X[:, 3] ** 2


# ---------------------------------------------------------------------------
# Training data  (random Latin-Hypercube-like sample in [0, 1]^4)
# ---------------------------------------------------------------------------

np.random.seed(42)
n_train = 40
X_train = np.random.uniform(0, 1, (n_train, 4))

y_func = f(X_train).reshape(-1, 1)
y_dx1  = df_dx1(X_train).reshape(-1, 1)
y_dx2  = df_dx2(X_train).reshape(-1, 1)
y_dx3  = df_dx3(X_train).reshape(-1, 1)
y_dx4  = df_dx4(X_train).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

np.random.seed(99)
n_test = 200
X_test = np.random.uniform(0, 1, (n_test, 4))

f_true   = f(X_test)
dx1_true = df_dx1(X_test)
dx2_true = df_dx2(X_test)
dx3_true = df_dx3(X_test)
dx4_true = df_dx4(X_test)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def rmse(a, b):
    return float(np.sqrt(np.mean((a.flatten() - b.flatten()) ** 2)))


# ===========================================================================
# MODEL A: partial DEGP — trained with df/dx1 and df/dx2 only
# ===========================================================================

print("=" * 60)
print("MODEL A: DEGP with df/dx1 and df/dx2 (2 of 4 derivatives)")
print("=" * 60)

y_train_partial = [y_func, y_dx1, y_dx2]

# der_indices format:  [ order_level [ deriv_spec, ... ] ]
# Each deriv_spec is a list of [variable_index (1-based), power] pairs.
# [[1,1]] = df/dx1,  [[2,1]] = df/dx2
der_indices_partial = [[[[1, 1]], [[2, 1]]]]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_A = degp(
        X_train,
        y_train_partial,
        n_order=1,
        n_bases=4,
        der_indices=der_indices_partial,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic",
    )

print("Optimising hyperparameters for Model A …")
params_A = model_A.optimize_hyperparameters(
    optimizer="pso",
    pop_size=80,
    n_generations=20,
    local_opt_every=20,
    debug=False,
)
print(f"  params: {params_A}\n")


# Predict function + trained derivatives (return_deriv=True gives f, dx1, dx2)
pred_A = model_A.predict(X_test, params_A, calc_cov=False, return_deriv=True)
f_A    = pred_A[0, :]   # function values
dx1_A  = pred_A[1, :]   # df/dx1
dx2_A  = pred_A[2, :]   # df/dx2

print("Prediction accuracy (Model A — trained derivatives):")
print(f"  RMSE  f(x)   : {rmse(f_A,   f_true):.4e}")
print(f"  RMSE df/dx1  : {rmse(dx1_A, dx1_true):.4e}")
print(f"  RMSE df/dx2  : {rmse(dx2_A, dx2_true):.4e}")


# --- What happens when we request UNTRAINED derivatives? ---

print("\n--- Requesting df/dx3 and df/dx4 (NOT in training set) ---")
print("  (guard rail removed — constructing K_* from kernel derivatives)")

pred_untrained = model_A.predict(
    X_test,
    params_A,
    calc_cov=False,
    return_deriv=True,
    derivs_to_predict=[[[3, 1]], [[4, 1]]],
)

# pred_untrained shape: (3, n_test) — [f, dx3, dx4]
f_A_unt   = pred_untrained[0, :]
dx3_A_unt = pred_untrained[1, :]
dx4_A_unt = pred_untrained[2, :]

print(f"  RMSE  f(x)   (sanity): {rmse(f_A_unt, f_true):.4e}")
print(f"  RMSE df/dx3  (untrained): {rmse(dx3_A_unt, dx3_true):.4e}")
print(f"  RMSE df/dx4  (untrained): {rmse(dx4_A_unt, dx4_true):.4e}")


print("\n--- Requesting ALL four derivatives at once ---")

pred_all = model_A.predict(
    X_test,
    params_A,
    calc_cov=False,
    return_deriv=True,
    derivs_to_predict=[[[1, 1]], [[2, 1]], [[3, 1]], [[4, 1]]],
)

# pred_all shape: (5, n_test) — [f, dx1, dx2, dx3, dx4]
f_A_all   = pred_all[0, :]
dx1_A_all = pred_all[1, :]
dx2_A_all = pred_all[2, :]
dx3_A_all = pred_all[3, :]
dx4_A_all = pred_all[4, :]

print(f"  RMSE  f(x)              : {rmse(f_A_all,   f_true):.4e}")
print(f"  RMSE df/dx1 (trained)   : {rmse(dx1_A_all, dx1_true):.4e}")
print(f"  RMSE df/dx2 (trained)   : {rmse(dx2_A_all, dx2_true):.4e}")
print(f"  RMSE df/dx3 (untrained) : {rmse(dx3_A_all, dx3_true):.4e}")
print(f"  RMSE df/dx4 (untrained) : {rmse(dx4_A_all, dx4_true):.4e}")


# ===========================================================================
# MODEL B: full DEGP — trained with ALL four first-order derivatives
# ===========================================================================

print("\n" + "=" * 60)
print("MODEL B: DEGP with all four derivatives df/dx1 … df/dx4")
print("=" * 60)

y_train_full = [y_func, y_dx1, y_dx2, y_dx3, y_dx4]

der_indices_full = [[[[1, 1]], [[2, 1]], [[3, 1]], [[4, 1]]]]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_B = degp(
        X_train,
        y_train_full,
        n_order=1,
        n_bases=4,
        der_indices=der_indices_full,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic",
    )

print("Optimising hyperparameters for Model B …")
params_B = model_B.optimize_hyperparameters(
    optimizer="pso",
    pop_size=80,
    n_generations=20,
    local_opt_every=20,
    debug=False,
)
print(f"  params: {params_B}\n")

pred_B = model_B.predict(X_test, params_B, calc_cov=False, return_deriv=True)
f_B    = pred_B[0, :]
dx1_B  = pred_B[1, :]
dx2_B  = pred_B[2, :]
dx3_B  = pred_B[3, :]
dx4_B  = pred_B[4, :]

print("Prediction accuracy (Model B — all derivatives):")
print(f"  RMSE  f(x)   : {rmse(f_B,   f_true):.4e}")
print(f"  RMSE df/dx1  : {rmse(dx1_B, dx1_true):.4e}")
print(f"  RMSE df/dx2  : {rmse(dx2_B, dx2_true):.4e}")
print(f"  RMSE df/dx3  : {rmse(dx3_B, dx3_true):.4e}")
print(f"  RMSE df/dx4  : {rmse(dx4_B, dx4_true):.4e}")


# ===========================================================================
# MODEL C: vanilla GP — no derivative information at all
# ===========================================================================

print("\n" + "=" * 60)
print("MODEL C: vanilla GP (no derivatives)")
print("=" * 60)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model_C = degp(
        X_train,
        [y_func],
        n_order=0,
        n_bases=4,
        der_indices=[],
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic",
    )

print("Optimising hyperparameters for Model C …")
params_C = model_C.optimize_hyperparameters(
    optimizer="pso",
    pop_size=80,
    n_generations=20,
    local_opt_every=20,
    debug=False,
)
print(f"  params: {params_C}\n")

pred_C = model_C.predict(X_test, params_C, calc_cov=False, return_deriv=False)
f_C    = pred_C[0, :]

print("Prediction accuracy (Model C — no derivatives):")
print(f"  RMSE  f(x)   : {rmse(f_C, f_true):.4e}")


# ===========================================================================
# Summary comparison
# ===========================================================================

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("Function value RMSE on 200 test points:")
print(f"  Vanilla GP (no derivs)   : {rmse(f_C, f_true):.4e}")
print(f"  Partial DEGP (dx1, dx2)  : {rmse(f_A, f_true):.4e}")
print(f"  Full DEGP (dx1-dx4)      : {rmse(f_B, f_true):.4e}")
print()
print("Derivative RMSE — Model A (trained on dx1, dx2 only):")
print(f"  df/dx1 (trained)         : {rmse(dx1_A,  dx1_true):.4e}")
print(f"  df/dx2 (trained)         : {rmse(dx2_A,  dx2_true):.4e}")
print(f"  df/dx3 (NOT in training) : {rmse(dx3_A_unt, dx3_true):.4e}")
print(f"  df/dx4 (NOT in training) : {rmse(dx4_A_unt, dx4_true):.4e}")
print()
print("Derivative RMSE — Model B (trained on all 4 derivatives):")
print(f"  df/dx1                   : {rmse(dx1_B, dx1_true):.4e}")
print(f"  df/dx2                   : {rmse(dx2_B, dx2_true):.4e}")
print(f"  df/dx3                   : {rmse(dx3_B, dx3_true):.4e}")
print(f"  df/dx4                   : {rmse(dx4_B, dx4_true):.4e}")
