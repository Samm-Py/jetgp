"""
DEGP Example: f(x1, x2) = sin(x1) * cos(x2)
--------------------------------------------
This example demonstrates the Derivative-Enhanced Gaussian Process (DEGP)
on a 2D function using a 3×3 training grid with first- and second-order
coordinate derivatives.

The script trains the model, predicts over a dense test grid,
and generates a figure showing:

- GP prediction (left)
- True function (center)
- Absolute error (right)

The output image is saved as:
    docs/images/degp_sin_cos.png
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from jetgp.full_degp.degp import degp

# ============================
# 1. Training Data Generation
# ============================

# Generate 3x3 training grid
X1 = np.array([0.0, 0.5, 1.0])
X2 = np.array([0.0, 0.5, 1.0])
X1_grid, X2_grid = np.meshgrid(X1, X2)
X_train = np.column_stack([X1_grid.flatten(), X2_grid.flatten()])

# Define function and derivatives: f(x, y) = sin(x)cos(y)
y_func = np.sin(X_train[:, 0]) * np.cos(X_train[:, 1])
y_deriv_x = np.cos(X_train[:, 0]) * np.cos(X_train[:, 1])
y_deriv_y = -np.sin(X_train[:, 0]) * np.sin(X_train[:, 1])
y_deriv_xx = -np.sin(X_train[:, 0]) * np.cos(X_train[:, 1])
y_deriv_yy = -np.sin(X_train[:, 0]) * np.cos(X_train[:, 1])

# Organize training data
y_train = [
    y_func.reshape(-1, 1),
    y_deriv_x.reshape(-1, 1),
    y_deriv_y.reshape(-1, 1),
    y_deriv_xx.reshape(-1, 1),
    y_deriv_yy.reshape(-1, 1)
]

# Specify derivative structure
der_indices = [
    [[[1, 1]], [[2, 1]]],  # First-order
    [[[1, 2]], [[2, 2]]]   # Second-order
]

print("Initializing DEGP model...")

# ============================
# 2. Model Initialization
# ============================

model = degp(
    X_train,
    y_train,
    n_order=2,
    n_bases=2,
    der_indices=der_indices,
    normalize=True,
    kernel="SE",
    kernel_type="anisotropic"
)

# ============================
# 3. Hyperparameter Optimization
# ============================

print("Optimizing hyperparameters...")
params = model.optimize_hyperparameters(
    optimizer='jade',
    pop_size=100,
    n_generations=15
)
print("Optimization complete!")

# ============================
# 4. Prediction
# ============================

# Create test grid
x_test = np.linspace(0, 1, 50)
X1_test, X2_test = np.meshgrid(x_test, x_test)
X_test = np.column_stack([X1_test.flatten(), X2_test.flatten()])

# Predict and evaluate
y_pred = model.predict(X_test, params, return_deriv=False)
y_true = np.sin(X_test[:, 0]) * np.cos(X_test[:, 1])
abs_error = np.abs(y_true - y_pred.flatten())

print(f"Mean absolute error: {np.mean(abs_error):.6f}")
print(f"Max absolute error: {np.max(abs_error):.6f}")

# ============================
# 5. Visualization
# ============================

plt.rcParams.update({'font.size': 12})
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Reshape for plotting
y_pred_grid = y_pred.reshape(X1_test.shape)
y_true_grid = y_true.reshape(X1_test.shape)
abs_error_grid = abs_error.reshape(X1_test.shape)

# (1) GP Prediction
c1 = axes[0].contourf(X1_test, X2_test, y_pred_grid, levels=50, cmap='viridis')
axes[0].scatter(X_train[:, 0], X_train[:, 1], c='red', s=100, edgecolors='k', linewidths=2, zorder=5)
axes[0].set_xlabel('$x_1$')
axes[0].set_ylabel('$x_2$')
axes[0].set_title('GP Prediction')
fig.colorbar(c1, ax=axes[0])

# (2) True Function
c2 = axes[1].contourf(X1_test, X2_test, y_true_grid, levels=50, cmap='viridis')
axes[1].scatter(X_train[:, 0], X_train[:, 1], c='red', s=100, edgecolors='k', linewidths=2, zorder=5)
axes[1].set_xlabel('$x_1$')
axes[1].set_ylabel('$x_2$')
axes[1].set_title(r'True Function: $f(x_1, x_2) = \sin(x_1)\cos(x_2)$')
fig.colorbar(c2, ax=axes[1])

# (3) Absolute Error
c3 = axes[2].contourf(X1_test, X2_test, abs_error_grid, levels=50, cmap='Reds')
axes[2].scatter(X_train[:, 0], X_train[:, 1], c='red', s=100, edgecolors='k', linewidths=2, zorder=5)
axes[2].set_xlabel('$x_1$')
axes[2].set_ylabel('$x_2$')
axes[2].set_title(r'Absolute Error: $|y_{pred} - y_{true}|$')
fig.colorbar(c3, ax=axes[2])

# Legend
custom_lines = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markeredgecolor='k', markersize=10,
           label=r'Training points with $\frac{\partial}{\partial x_1}$ and $\frac{\partial}{\partial x_2}$')
]
fig.legend(handles=custom_lines, loc='lower center', ncol=1, frameon=False,
           fontsize=12, bbox_to_anchor=(0.5, -0.05))

plt.tight_layout(rect=[0, 0.02, 1, 1])

# ============================
# 6. Save Figure
# ============================

output_path = "./docs/source/_static/degp_sin_cos.png"
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.close(fig)

print(f"Figure saved to: {output_path}")