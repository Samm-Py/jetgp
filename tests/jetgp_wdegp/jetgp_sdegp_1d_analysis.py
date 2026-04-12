"""
Reproduces the 1D analysis from Cheng & Zimmermann (2024), Section 4.1.

Test function (eq. 26):
    g(x) = e^{-x} + sin(5x) + cos(5x) + 0.2x + 4,   x in [0, 6]

Generates three figures matching the paper:

  Figure 1 — Likelihood comparison (cf. Figs 5-6):
      Full GE-Kriging NLL vs 2-appendant SGE-Kriging NLL
      as a function of log10(theta), for m = 5 and m = 10.

  Figure 2 — Prediction comparison (cf. Figs 7-8):
      True function, training samples, GE-Kriging mean/variance,
      and SGE-Kriging mean/variance for m = 10.

  Figure 3 — Prediction error:
      Pointwise absolute error for GE-Kriging and SGE-Kriging.

Usage:
    conda run -n pyoti_2 python tests/jetgp_wdegp/jetgp_sdegp_1d_analysis.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Ensure jetgp is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jetgp.full_degp.degp import degp
from jetgp.sdegp.sdegp import sdegp
import jetgp.utils as utils


# ---------------------------------------------------------------------------
# Test function (eq. 26)
# ---------------------------------------------------------------------------
def g(x):
    x = np.asarray(x).ravel()
    return np.exp(-x) + np.sin(5 * x) + np.cos(5 * x) + 0.2 * x + 4.0


def dg(x):
    x = np.asarray(x).ravel()
    return (-np.exp(-x) + 5.0 * np.cos(5 * x) - 5.0 * np.sin(5 * x) + 0.2).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N = 10                # training samples (paper: 10 via LHS)
DIM = 1
N_TEST = 3000         # test points (paper: 3000)
M_VALUES = [5, 10]    # slice counts to compare
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"

# Theta sweep range for likelihood plots
THETA_LO, THETA_HI, N_THETA = -2.0, 1.5, 200

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "..", "data")
os.makedirs(OUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Build training data
# ---------------------------------------------------------------------------
np.random.seed(42)
X_train = np.linspace(0, 6, N).reshape(-1, 1)
y_train = g(X_train)
grads = dg(X_train)

X_test = np.linspace(0, 6, N_TEST).reshape(-1, 1)
y_test = g(X_test)


# ---------------------------------------------------------------------------
# Build full DEGP (standard GE-Kriging)
# ---------------------------------------------------------------------------
der_specs = utils.gen_OTI_indices(DIM, 1)
all_pts = list(range(N))
degp_model = degp(
    X_train,
    [y_train.reshape(-1, 1), grads],
    n_order=1,
    n_bases=DIM,
    der_indices=der_specs,
    derivative_locations=[all_pts],
    normalize=True,
    kernel=KERNEL,
    kernel_type=KERNEL_TYPE,
)

# ---------------------------------------------------------------------------
# Build sliced SDEGP models for each m
# ---------------------------------------------------------------------------
sdegp_models = {}
for m in M_VALUES:
    if m > N:
        print(f"  Skipping m={m} (m > N={N})")
        continue

    model = sdegp(
        X_train, y_train, grads,
        n_order=1, m=m,
        kernel=KERNEL,
        kernel_type=KERNEL_TYPE,
    )
    sdegp_models[m] = model
    print(f"  Built SDEGP m={m}: {model.num_submodels} submodels, "
          f"slice_dim={model.slice_dim}, sizes={[len(s) for s in model.slices]}")


# ===========================================================================
# Figure 1 — Likelihood comparison
# ===========================================================================
print("\nSweeping theta for likelihood comparison ...")
thetas = np.linspace(THETA_LO, THETA_HI, N_THETA)
nll_full = np.empty(N_THETA)
nll_sliced = {m: np.empty(N_THETA) for m in sdegp_models}

for i, t in enumerate(thetas):
    x0 = np.array([t, 0.0, -6.0])
    nll_full[i] = degp_model.optimizer.nll_wrapper(x0)
    for m, model in sdegp_models.items():
        nll_sliced[m][i] = model.optimizer.nll_wrapper(x0)

# Clip to a reasonable range for plotting (match paper's y-axis ~60-100)
# Find the minimum to set the plot range
nll_min = min(nll_full.min(), min(v.min() for v in nll_sliced.values()))
nll_max_plot = nll_min + 60  # show ~60 units above minimum

# Identify optima
idx_opt_full = np.argmin(nll_full)
theta_opt_full = thetas[idx_opt_full]
print(f"  Full DEGP:  optimal log10(theta) = {theta_opt_full:.4f}, "
      f"NLL = {nll_full[idx_opt_full]:.4f}")
for m in sdegp_models:
    idx_opt = np.argmin(nll_sliced[m])
    print(f"  SDEGP m={m}: optimal log10(theta) = {thetas[idx_opt]:.4f}, "
          f"NLL = {nll_sliced[m][idx_opt]:.4f}")

fig1, axes = plt.subplots(1, len(sdegp_models), figsize=(6 * len(sdegp_models), 5),
                          squeeze=False)
for ax_idx, m in enumerate(sorted(sdegp_models)):
    ax = axes[0, ax_idx]
    ax.plot(thetas, nll_full, 'b-', linewidth=1.5, label='Original likelihood')
    ax.plot(thetas, nll_sliced[m], 'r--', linewidth=1.5,
            label=f'2-appendant likelihood (m={m})')
    ax.set_xlabel(r'$\log_{10}(\theta)$', fontsize=12)
    ax.set_ylabel('Likelihood', fontsize=12)
    ax.set_title(f'GE-Kriging vs SGE-Kriging, m = {m}', fontsize=13)
    ax.set_ylim(nll_min - 5, nll_max_plot)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

fig1.tight_layout()
fig1_path = os.path.join(OUT_DIR, "sdegp_1d_likelihood_comparison.png")
fig1.savefig(fig1_path, dpi=150)
print(f"\nFigure 1 saved to {fig1_path}")


# ===========================================================================
# Figure 2 — Prediction comparison (use the largest m)
# ===========================================================================
m_pred = max(sdegp_models.keys())
model_pred = sdegp_models[m_pred]

# Find optimal theta for the SDEGP model via the sweep
idx_opt_sdegp = np.argmin(nll_sliced[m_pred])
theta_opt_sdegp = thetas[idx_opt_sdegp]
params_sdegp = np.array([theta_opt_sdegp, 0.0, -6.0])
params_degp = np.array([theta_opt_full, 0.0, -6.0])

print(f"\nPredicting with optimal params:")
print(f"  DEGP:  params = {params_degp}")
print(f"  SDEGP: params = {params_sdegp}")

# Full DEGP prediction
y_pred_degp = degp_model.predict(
    X_test, params_degp, calc_cov=True, return_deriv=False
)
if isinstance(y_pred_degp, tuple):
    y_mean_degp = np.asarray(y_pred_degp[0]).flatten()
    y_var_degp = np.asarray(y_pred_degp[1]).flatten()
else:
    y_mean_degp = np.asarray(y_pred_degp).flatten()
    y_var_degp = np.zeros_like(y_mean_degp)

# SDEGP prediction
y_pred_sdegp = model_pred.predict(
    X_test, params_sdegp, calc_cov=True, return_deriv=False
)
if isinstance(y_pred_sdegp, tuple):
    y_mean_sdegp = np.asarray(y_pred_sdegp[0]).flatten()
    y_var_sdegp = np.asarray(y_pred_sdegp[1]).flatten()
else:
    y_mean_sdegp = np.asarray(y_pred_sdegp).flatten()
    y_var_sdegp = np.zeros_like(y_mean_sdegp)

# Compute RMSE (eq. 25)
def rmse_relative(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)
    var_true = np.mean((y_true - np.mean(y_true)) ** 2)
    return np.sqrt(mse / var_true)

rmse_degp = rmse_relative(y_test, y_mean_degp)
rmse_sdegp = rmse_relative(y_test, y_mean_sdegp)
print(f"\n  RMSE (relative):")
print(f"    GE-Kriging:                {rmse_degp:.6f}")
print(f"    2-appendant SGE-Kriging:   {rmse_sdegp:.6f}")

x_plot = X_test.ravel()

fig2, (ax_degp, ax_sdegp) = plt.subplots(1, 2, figsize=(14, 5))

# --- GE-Kriging ---
ax_degp.plot(x_plot, y_test, 'b-', linewidth=1.2, label='True function')
ax_degp.plot(x_plot, y_mean_degp, 'r--', linewidth=1.2, label='GE-Kriging mean')
if np.any(y_var_degp > 0):
    std_degp = np.sqrt(np.abs(y_var_degp))
    ax_degp.fill_between(x_plot, y_mean_degp - 2 * std_degp,
                         y_mean_degp + 2 * std_degp,
                         alpha=0.2, color='red', label='GE-Kriging variance')
ax_degp.plot(X_train.ravel(), y_train, 'ro', markersize=6, label='Samples')
ax_degp.set_xlabel('x', fontsize=12)
ax_degp.set_ylabel('y', fontsize=12)
ax_degp.set_title(f'GE-Kriging (RMSE = {rmse_degp:.4f})', fontsize=13)
ax_degp.legend(fontsize=9)
ax_degp.grid(True, alpha=0.3)

# --- SGE-Kriging ---
ax_sdegp.plot(x_plot, y_test, 'b-', linewidth=1.2, label='True function')
ax_sdegp.plot(x_plot, y_mean_sdegp, 'r--', linewidth=1.2,
              label=f'2-appendant SGE-Kriging mean (m={m_pred})')
if np.any(y_var_sdegp > 0):
    std_sdegp = np.sqrt(np.abs(y_var_sdegp))
    ax_sdegp.fill_between(x_plot, y_mean_sdegp - 2 * std_sdegp,
                          y_mean_sdegp + 2 * std_sdegp,
                          alpha=0.2, color='red',
                          label='SGE-Kriging variance')
ax_sdegp.plot(X_train.ravel(), y_train, 'ro', markersize=6, label='Samples')
ax_sdegp.set_xlabel('x', fontsize=12)
ax_sdegp.set_ylabel('y', fontsize=12)
ax_sdegp.set_title(f'SGE-Kriging m={m_pred} (RMSE = {rmse_sdegp:.4f})',
                    fontsize=13)
ax_sdegp.legend(fontsize=9)
ax_sdegp.grid(True, alpha=0.3)

fig2.tight_layout()
fig2_path = os.path.join(OUT_DIR, "sdegp_1d_prediction_comparison.png")
fig2.savefig(fig2_path, dpi=150)
print(f"Figure 2 saved to {fig2_path}")


# ===========================================================================
# Figure 3 — Pointwise absolute error
# ===========================================================================
fig3, ax = plt.subplots(figsize=(8, 4))
ax.semilogy(x_plot, np.abs(y_test - y_mean_degp), 'b-', linewidth=1.0,
            label='GE-Kriging', alpha=0.8)
ax.semilogy(x_plot, np.abs(y_test - y_mean_sdegp), 'r--', linewidth=1.0,
            label=f'SGE-Kriging (m={m_pred})', alpha=0.8)
ax.plot(X_train.ravel(), np.zeros(N), 'ko', markersize=5, label='Training points')
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('|error|', fontsize=12)
ax.set_title('Pointwise Absolute Prediction Error', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

fig3.tight_layout()
fig3_path = os.path.join(OUT_DIR, "sdegp_1d_error_comparison.png")
fig3.savefig(fig3_path, dpi=150)
print(f"Figure 3 saved to {fig3_path}")

plt.show()
