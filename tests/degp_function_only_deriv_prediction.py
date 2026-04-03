"""DEGP — Function-Only Training with Derivative Predictions
===========================================================
Train the DEGP on function values only (no derivative observations),
then predict f and f' with uncertainty at test points.

The model uses n_order=1 so the kernel captures derivative structure
via OTI arithmetic, but der_indices=[] means no derivative data enters
the training covariance.  At prediction time, derivs_to_predict=[[[1, 1]]]
requests the first-order partial derivative — even though it was never
observed — using the cross-covariance between f and df/dx implied by
the kernel.

True function: f(x) = sin(x),  f'(x) = cos(x)
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt
from jetgp.full_degp.degp import degp


# ---------------------------------------------------------------------------
# True function and derivative
# ---------------------------------------------------------------------------
def f(x):
    return np.exp(-1*x) + np.sin(x) + np.cos(3*x) + .2*x + 1
def df(x):
    return -1 * np.exp(-1 * x) + np.cos(x) + -3 * np.sin(3 * x) + .2


# ---------------------------------------------------------------------------
# Training data — function values ONLY
# ---------------------------------------------------------------------------
X_train = np.linspace(.2, 6, 8).reshape(-1, 1)
y_func  = f(X_train)
y_train = [y_func]   # no derivative training data


# ---------------------------------------------------------------------------
# Build model
#   n_order=1    : enables first-order OTI arithmetic (needed for derivative kernels)
#   n_bases=1    : one input dimension
#   der_indices=[] : no derivatives observed during training
# ---------------------------------------------------------------------------
with warnings.catch_warnings():
    warnings.simplefilter("ignore")   # suppress "0 derivatives" notice
    model = degp(
        X_train, y_train,
        n_order=1, n_bases=1,
        der_indices=[],
        normalize=False,
        kernel="SE", kernel_type="anisotropic",
    )


# ---------------------------------------------------------------------------
# Optimise hyperparameters
# ---------------------------------------------------------------------------
params = model.optimize_hyperparameters(
    optimizer="pso",
    pop_size=100,
    n_generations=15,
    local_opt_every=15,
    debug=False,
)
print("Optimised hyperparameters:", params)


# ---------------------------------------------------------------------------
# Predict f and f' with uncertainty
#   derivs_to_predict=[[[1, 1]]] : df/dx1, first-order
#   calc_cov=True                : also return predictive variance
# ---------------------------------------------------------------------------
X_test = np.linspace(.2, 6, 200).reshape(-1, 1)

mean, var = model.predict(
    X_test, params,
    calc_cov=True,
    return_deriv=True,
    derivs_to_predict=[[[1, 1]]],
)

f_mean  = mean[0, :]        # shape (n_test,)
df_mean = mean[1, :]
f_std   = np.sqrt(np.abs(var[0, :]))
df_std  = np.sqrt(np.abs(var[1, :]))


# ---------------------------------------------------------------------------
# Accuracy summary
# ---------------------------------------------------------------------------
x_flat = X_test.flatten()
f_rmse  = float(np.sqrt(np.mean((f_mean  - f(X_test).flatten()) ** 2)))
df_rmse = float(np.sqrt(np.mean((df_mean - df(X_test).flatten()) ** 2)))
df_corr = float(np.corrcoef(df_mean, df(X_test).flatten())[0, 1])

print(f"Function  RMSE : {f_rmse:.4e}")
print(f"Derivative RMSE: {df_rmse:.4e}")
print(f"Derivative Pearson r: {df_corr:.3f}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Function ---
ax = axes[0]
ax.plot(x_flat, f(X_test).flatten(), "k--", label="True f(x)")
ax.plot(x_flat, f_mean, "r-", label="GP mean")
ax.fill_between(
    x_flat,
    f_mean - 2 * f_std, f_mean + 2 * f_std,
    alpha=0.3, color="r", label=r"$\mu \pm 2\sigma$",
)
ax.scatter(X_train.flatten(), y_func.flatten(), c="b", zorder=5, label="Training points")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.set_title("Function prediction\n(trained on f values)")
ax.legend()

# --- Derivative ---
ax = axes[1]
ax.plot(x_flat, df(X_test).flatten(), "k--", label="True f'(x)")
ax.plot(x_flat, df_mean, "g-", label="GP derivative mean")
ax.fill_between(
    x_flat,
    df_mean - 2 * df_std, df_mean + 2 * df_std,
    alpha=0.3, color="g", label=r"$\partial \mu \pm 2\sigma$"
)
ax.set_xlabel("x")
ax.set_ylabel("f'(x)")
ax.set_title("Derivative prediction\n(NOT trained on f' values)")
ax.legend()

plt.tight_layout()
plt.savefig("./degp_function_only_deriv_prediction.png", dpi=250)
plt.show()
