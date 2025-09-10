#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combined Surface Plot for Submodeling Analysis
Created on Tue Sep  9 12:49:24 2025

@author: root
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calculate_cholesky_flops(matrix_size):
    """
    Calculate FLOPs for Cholesky decomposition.
    Complexity: O(n^3)
    """
    return matrix_size**3


def flops_to_time(flops, cpu_gflops=10):
    """
    Convert FLOPs to training time in seconds.
    Default: 10 GFLOPS effective performance
    """
    return flops / (cpu_gflops * 1e9)


def submodeling_matrix_size(n_points, n_dimensions, j_submodels):
    """
    Submodeling GP matrix size with j submodels.

    Formula: j × (n_points + (n_points/j) × n_dimensions)

    Each submodel:
    - Gets n_points function observations (shared)
    - Gets (n_points/j) × n_dimensions gradient observations

    Total effective size: j × (n_points + (n_points/j) × n_dimensions)
    """
    points_per_submodel = n_points / j_submodels
    gradients_per_submodel = points_per_submodel * n_dimensions
    size_per_submodel = n_points + gradients_per_submodel
    return j_submodels**(1/3) * size_per_submodel


def full_gp_matrix_size(n_points, n_dimensions):
    """
    Full GP matrix size for comparison.
    Total: n_points × (1 + n_dimensions)
    """
    return n_points * (1 + n_dimensions)


# Analysis parameters
n_points_range = np.logspace(1, 2, 20)  # 10 to 100 training points
dimensions_range = np.arange(1, 101)     # 1 to 30 dimensions
j_values = [2, 4, 8]             # Different numbers of submodels

# Create single 3D plot with all surfaces
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# Create meshgrid
N_mesh, D_mesh = np.meshgrid(n_points_range, dimensions_range)

# Color map for different j values
colors = ['green', 'blue', 'purple']
cmaps = ['Greens', 'Blues', 'Purples']
alphas = [0.9, 0.8, 0.8, 0.7, 0.3]  # Decreasing alpha for layering

max_time = 1800  # 30 minutes for clipping

# Plot surfaces for each j value
surfaces = []
labels = []

for idx, j in enumerate(j_values):
    # Calculate training times
    training_times = np.zeros_like(N_mesh)

    for i, d in enumerate(dimensions_range):
        for k, n in enumerate(n_points_range):
            # Ensure j doesn't exceed n
            j_actual = min(j, int(n))
            matrix_size = submodeling_matrix_size(n, d, j_actual)
            training_times[i, k] = flops_to_time(
                calculate_cholesky_flops(matrix_size))

    # Clip for visualization
    times_clipped = np.clip(training_times, 0, max_time)

    # Create surface
    surf = ax.plot_surface(N_mesh, D_mesh, times_clipped,
                           cmap=cmaps[idx], alpha=alphas[idx],
                           linewidth=0, antialiased=True, zorder=100-idx)

    surfaces.append(surf)
    labels.append(f'j = {j} submodels')

# Add full GP surface for comparison
full_times = np.zeros_like(N_mesh)
for i, d in enumerate(dimensions_range):
    for k, n in enumerate(n_points_range):
        matrix_size = full_gp_matrix_size(n, d)
        full_times[i, k] = flops_to_time(calculate_cholesky_flops(matrix_size))

full_times_clipped = np.clip(full_times, 0, max_time)
surf_full = ax.plot_surface(N_mesh, D_mesh, full_times_clipped,
                            cmap='Reds', alpha=0.8, linewidth=0,
                            antialiased=True)

surfaces.append(surf_full)
labels.append('Full GP (baseline)')

# Set labels and title
ax.set_xlabel('Training Points', fontsize=18, fontweight='bold')
ax.set_ylabel('Dimensions', fontsize=18, fontweight='bold')
ax.set_zlabel('Training Time (seconds)', fontsize=18, fontweight='bold')
# ax.set_title('Submodeling GP Training Time Comparison\n(All j values overlaid)',
#              fontsize=16, fontweight='bold', pad=20)

# Adjust viewing angle for better visualization
ax.view_init(elev=30, azim=135)

# Create custom legend
legend_elements = []
legend_colors = colors + ['red']
for i, label in enumerate(labels):
    legend_elements.append(plt.Rectangle((0, 0), 1, 1,
                                         facecolor=legend_colors[i],
                                         alpha=0.7, label=label))

# ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0, 1))

# Add text box with key insights
# textstr = '''Key Insights:
# • Lower j values (fewer submodels) = faster training
# • j=1 equivalent to full GP
# • Higher dimensions favor more submodels
# • Sweet spot often around j=2-4'''

# props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
# ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
#           verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()

# Alternative: Side-by-side comparison plot
fig2, axes = plt.subplots(2, 3, figsize=(
    18, 12), subplot_kw={'projection': '3d'})
axes = axes.flatten()

# Individual plots for clearer comparison
for idx, j in enumerate(j_values):
    ax = axes[idx]

    # Calculate training times
    training_times = np.zeros_like(N_mesh)

    for i, d in enumerate(dimensions_range):
        for k, n in enumerate(n_points_range):
            j_actual = min(j, int(n))
            matrix_size = submodeling_matrix_size(n, d, j_actual)
            training_times[i, k] = flops_to_time(
                calculate_cholesky_flops(matrix_size))

    times_clipped = np.clip(training_times, 0, max_time)

    # Create surface
    surf = ax.plot_surface(N_mesh, D_mesh, times_clipped,
                           cmap=cmaps[idx], alpha=0.8,
                           linewidth=0, antialiased=True)

    ax.set_xlabel('Training Points', fontsize=10)
    ax.set_ylabel('Dimensions', fontsize=10)
    ax.set_zlabel('Time (s)', fontsize=10)
    ax.set_title(f'j = {j} submodels', fontsize=12, fontweight='bold')
    ax.view_init(elev=20, azim=45)

# Full GP in the last subplot
ax_full = axes[5]
surf_full = ax_full.plot_surface(N_mesh, D_mesh, full_times_clipped,
                                 cmap='Greys', alpha=0.8, linewidth=0,
                                 antialiased=True)

ax_full.set_xlabel('Training Points', fontsize=10)
ax_full.set_ylabel('Dimensions', fontsize=10)
ax_full.set_zlabel('Time (s)', fontsize=10)
ax_full.set_title('Full GP (baseline)', fontsize=12, fontweight='bold')
ax_full.view_init(elev=30, azim=-45)

plt.suptitle('Submodeling GP Training Time Analysis',
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

print("Plots generated:")
print("1. Combined overlay plot showing all surfaces together")
print("2. Side-by-side subplot comparison (original layout)")
print("\nThe combined plot shows how different j values compare directly,")
print("while the subplot version provides clearer individual visualization.")


def calculate_performance_ratio(n_points, n_dimensions, j_submodels):
    """
    Calculate the performance ratio: Full GP time / Submodel time
    Ratio > 1 means submodel is faster
    """
    # Full GP performance
    full_matrix_size = full_gp_matrix_size(n_points, n_dimensions)
    full_time = flops_to_time(calculate_cholesky_flops(full_matrix_size))

    # Submodel performance
    j_actual = min(j_submodels, int(n_points))
    sub_matrix_size = submodeling_matrix_size(n_points, n_dimensions, j_actual)
    sub_time = flops_to_time(calculate_cholesky_flops(sub_matrix_size))

    # Ratio (speedup factor)
    ratio = full_time / sub_time
    return ratio


# Speedup analysis for fixed training points
fixed_n = 100
dims_range = range(2, 101)
# Create side-by-side speedup comparison
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

time = []
time_2 = []
time_4 = []
time_8 = []
for d in dims_range:
    full_matrix_size = full_gp_matrix_size(fixed_n, d)
    full_time = flops_to_time(
        calculate_cholesky_flops(full_matrix_size))
    sub_matrix_size = submodeling_matrix_size(fixed_n, d, 2)
    time_2_submodels = flops_to_time(
        calculate_cholesky_flops(sub_matrix_size))
    sub_matrix_size = submodeling_matrix_size(fixed_n, d, 4)
    time_4_submodels = flops_to_time(
        calculate_cholesky_flops(sub_matrix_size))
    sub_matrix_size = submodeling_matrix_size(fixed_n, d, 8)
    time_8_submodels = flops_to_time(
        calculate_cholesky_flops(sub_matrix_size))

    time.append(full_time/full_time)
    time_2.append(full_time/time_2_submodels)
    time_4.append(full_time / time_4_submodels)
    time_8.append(full_time / time_8_submodels)

ax2.plot(dims_range, time, color='tab:red', linewidth=3)
ax2.plot(dims_range, time_2, color='tab:green', linewidth=3)
ax2.plot(dims_range, time_4, color='tab:blue', linewidth=3)
ax2.plot(dims_range, time_8, color='tab:purple', linewidth=3,)
ax2.set_xlabel('Dimensions', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
ax2.set_title(
    f'Speedup vs Dimensions\n({fixed_n} training points)', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
# ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Analysis parameters
n_points_range = np.logspace(1, 2, 20)  # 10 to 100 training points
dimensions_range = np.arange(1, 101)     # 1 to 100 dimensions
j_values = [2, 4, 8]             # Different numbers of submodels

# Create meshgrid
N_mesh, D_mesh = np.meshgrid(n_points_range, dimensions_range)

# PERFORMANCE RATIO ANALYSIS
print("="*80)
print("PERFORMANCE RATIO ANALYSIS: FULL DEGP vs SUBMODELS")
print("="*80)
print("Ratio > 1.0 = Submodel is faster than Full GP")
print("Ratio < 1.0 = Full GP is faster than Submodel")
print("="*80)


# DETAILED ANALYSIS FOR SPECIFIC CASES
print("\n" + "="*80)
print("DETAILED RATIO ANALYSIS FOR SPECIFIC PROBLEM SIZES")
print("="*80)

test_cases = [
    (100, 10, "Small problem"),
    (100, 20, "Medium problem"),
    (100, 50, "Large problem"),
    (100, 100, "Very large problem")
]

for n, d, description in test_cases:
    print(f"\n{description.upper()} (n={n}, d={d}):")
    print(f"{'j':<5} {'Ratio':<10} {'Interpretation':<30}")
    print("-" * 50)

    for j in j_values:
        ratio = calculate_performance_ratio(n, d, j)
        if ratio > 1.5:
            interpretation = f"{ratio:.1f}x faster (significant speedup)"
        elif ratio > 1.0:
            interpretation = f"{ratio:.1f}x faster (modest speedup)"
        elif ratio > 0.8:
            interpretation = "Similar performance"
        else:
            interpretation = f"{1/ratio:.1f}x slower (submodel overhead)"

        print(f"{j:<5} {ratio:<10.2f} {interpretation}")
