"""
DEGP Tutorial: Derivative Enhanced Gaussian Processes

This tutorial demonstrates how to use derivative information to improve 
Gaussian Process regression. We'll show how including derivatives of 
different orders affects the quality of predictions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pyoti.sparse as oti
from full_degp.degp import degp
import utils
import time

# Set plotting parameters for better readability
plt.rcParams.update({'font.size': 12})

def true_function(X, alg=oti):
    """
    Test function combining exponential decay, oscillations, and linear trend.
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, 1)
        Input points
    alg : module
        Numerical library (numpy or pyoti)
        
    Returns:
    --------
    y : array-like
        Function values
    """
    x = X[:, 0]
    return alg.exp(-x) + alg.sin(2*x) + alg.cos(3 * x) + 0.2 * x + 1.0

# =============================================================================
# Tutorial Configuration
# =============================================================================

print("DEGP Tutorial: Derivative Enhanced Gaussian Processes")
print("=" * 60)

# Problem setup
lb_x, ub_x = 0.2, 5.0
num_training_points = 3  # Very few points to show DEGP advantage
num_test_points = 100

print(f"Domain: [{lb_x}, {ub_x}]")
print(f"Training points: {num_training_points}")
print(f"Test points: {num_test_points}")

# Generate training and test data
X_train = np.linspace(lb_x, ub_x, num_training_points).reshape(-1, 1)
X_test = np.linspace(lb_x, ub_x, num_test_points).reshape(-1, 1)
y_true = true_function(X_test, alg=np)

print(f"Training points: {X_train.ravel()}")
print()

# =============================================================================
# DEGP with Different Derivative Orders
# =============================================================================

orders = [0, 1, 2, 4]
titles = [
    r"Order 0: $f(x)$ only",
    r"Order 1: $f(x)$, $f'(x)$",
    r"Order 2: $f(x)$, $f'(x)$, $f''(x)$",
    r"Order 4: $f(x)$, ..., $f^{(4)}(x)$"
]

# Store results for comparison
results = {}

fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
axs = axs.flatten()

print("Training DEGP models with different derivative orders...")
print("-" * 50)

for idx, n_order in enumerate(orders):
    print(f"Processing Order {n_order}...")
    start_time = time.time()
    
    # Setup derivative indices structure
    # der_indices are structured as nested lists representing derivative terms
    # For 1D functions:
    #   Order 1: [[[[1, 1]]]] → ∂f/∂x₁ 
    #   Order 2: [[[[1, 1]]], [[[1, 2]]]] → ∂f/∂x₁, ∂²f/∂x₁²
    n_bases = 1  # Number of input dimensions
    der_indices = utils.gen_OTI_indices(n_bases, n_order)
    
    print(f"  Derivative indices: {der_indices}")
    
    # Generate training data with derivatives using automatic differentiation
    X_train_pert = oti.array(X_train)
    
    # Add hypercomplex perturbation for automatic differentiation
    # Note: In practice, derivatives would typically come from the user
    # rather than being computed this way
    for i in range(1, n_bases + 1):
        X_train_pert[:, i - 1] = X_train_pert[:, i - 1] + oti.e(i, order=n_order)

    # Evaluate function and all derivatives at training points
    y_train_hc = true_function(X_train_pert)
    
    # Extract function values and derivatives
    # The derivative information in y_train must match the exact order of der_indices
    y_train = [y_train_hc.real]  # Function values always come first
    total_derivatives = 0
    
    # Add derivatives in the same order as specified by der_indices
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            # Extract the specific derivative according to the index structure
            derivative = y_train_hc.get_deriv(der_indices[i][j]).reshape(-1, 1)
            y_train.append(derivative)  # Order must match der_indices
            total_derivatives += 1
    
    # Calculate total observations for understanding covariance matrix size
    # Each training point contributes: 1 function value + total_derivatives derivative values
    total_observations = len(y_train[0]) + total_derivatives * len(y_train[0])
    print(f"  Total training observations: {total_observations}")
    
    # Initialize and train DEGP
    gp = degp(
        X_train, y_train, n_order, n_bases, der_indices,
        normalize=False, 
        kernel="SE",  # Squared Exponential kernel
        kernel_type="anisotropic"
    )
    
    # Optimize hyperparameters using particle swarm optimization
    # n_restart_optimizer: number of generations for the particle swarm optimizer
    # swarm_size: number of particles in the swarm
    params = gp.optimize_hyperparameters(
        n_restart_optimizer=20, 
        swarm_size=100,
        x0 = None,  # Initial guess for [length_scale, variance, noise]
        local_opt_every = 10
    )
    
    # Make predictions
    y_pred, y_var = gp.predict(X_test, params, calc_cov=True, return_deriv=False)
    
    # Calculate metrics
    mse = np.mean((y_pred.ravel() - y_true.ravel())**2)
    mae = np.mean(np.abs(y_pred.ravel() - y_true.ravel()))
    
    # Store results
    results[n_order] = {
        'mse': mse,
        'mae': mae,
        'time': time.time() - start_time,
        'n_observations': total_observations
    }
    
    print(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, Time: {time.time() - start_time:.2f}s")
    
    # Plot results
    ax = axs[idx]
    
    # True function
    l0, = ax.plot(X_test, y_true, 'k-', lw=2.5, label="True $f(x)$")
    
    # GP prediction
    l1, = ax.plot(X_test, y_pred, 'b--', lw=2, label="GP mean")
    
    # Uncertainty bounds (95% confidence interval)
    l2 = ax.fill_between(
        X_test.ravel(),
        y_pred.ravel() - 2*np.sqrt(y_var.ravel()),
        y_pred.ravel() + 2*np.sqrt(y_var.ravel()),
        color='blue', alpha=0.15, label='GP 95% CI'
    )
    
    # Training points
    l3 = ax.scatter(X_train, y_train[0], c='red', s=60, zorder=5, 
                   edgecolors='black', linewidth=1, label="Training points")
    
    # Formatting
    ax.set_title(f"{titles[idx]}\nMSE: {mse:.4f}", fontsize=11)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$f(x)$")
    ax.grid(True, alpha=0.3)

print()

# =============================================================================
# Results Summary
# =============================================================================

print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Order':<8}{'MSE':<12}{'MAE':<12}{'Time (s)':<10}{'Observations'}")
print("-" * 60)

for order in orders:
    r = results[order]
    print(f"{order:<8}{r['mse']:<12.6f}{r['mae']:<12.6f}{r['time']:<10.2f}{r['n_observations']}")

print()
print("Key Observations:")
print("- Higher derivative orders generally improve accuracy")
print("- More derivative information helps with fewer training points")
print("- Computational cost increases with derivative order")
print("- DEGP is particularly useful when training data is sparse")

# =============================================================================
# Plotting and Layout
# =============================================================================

# Adjust layout
plt.tight_layout(rect=[0, 0.15, 1, 0.95])

# Add a main title
fig.suptitle('Derivative Enhanced Gaussian Process Comparison', 
             fontsize=16, fontweight='bold', y=0.98)

# Create shared legend
handles, labels = [], []
for ax in axs:
    h, l = ax.get_legend_handles_labels()
    for handle, label in zip(h, l):
        if label not in labels:
            handles.append(handle)
            labels.append(label)

fig.legend(
    handles, labels,
    loc='lower center', 
    bbox_to_anchor=(0.5, 0.02),
    ncol=len(handles), 
    frameon=True, 
    fontsize=11,
    fancybox=True,
    shadow=True
)

# Save and display
plt.savefig("degp_tutorial_comparison.png", dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# Additional Educational Content
# =============================================================================

print("\nWHAT IS DEGP?")
print("=" * 60)
print("Derivative Enhanced Gaussian Processes (DEGP) incorporate derivative")
print("information alongside function values to improve predictions.")
print()
print("Benefits:")
print("• Better accuracy with fewer training points")
print("• More informative priors from derivative constraints") 
print("• Useful when derivatives are available (physics, optimization)")
print()
print("When to use DEGP:")
print("• Limited training data available")
print("• Derivatives can be computed (analytically or via AD)")
print("• Smooth functions where derivative information is meaningful")
print()
print("Tutorial complete! Check the generated plot for visual comparison.")