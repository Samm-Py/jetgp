from matplotlib.patches import Patch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def compute_cholesky_flops(matrix_size):
    """
    Compute floating point operations for Cholesky decomposition.
    Complexity: O(n^3)
    """
    return matrix_size**3


def convert_flops_to_time(flops):
    """
    Convert FLOPs to training time on CPU.
    Assumes 10 GFLOPS effective performance (typical workstation).
    """
    cpu_performance = 10e9  # 10 billion FLOPs per second
    return flops / cpu_performance


def full_gradient_matrix_size(n_training_points, dimensions):
    """
    Full gradient GP: derivatives at ALL training points.

    Components:
    - n_training_points function observations
    - n_training_points × dimensions gradient observations

    Total matrix size: n_training_points × (1 + dimensions)
    """
    return n_training_points * (1 + dimensions)


def spatial_sparse_matrix_size(n_training_points, dimensions, spatial_fraction=0.2):
    """
    Spatial sparse gradient GP: derivatives at only SOME training points.

    SPATIAL SPARSITY: Select subset of training points for derivative observations

    Components:
    - n_training_points function observations (at ALL points)
    - (spatial_fraction × n_training_points) × dimensions gradient observations

    Total observations: n_training_points + (spatial_fraction × n_training_points × dimensions)
    """
    function_observations = n_training_points
    derivative_observations = spatial_fraction * n_training_points * dimensions
    total_observations = function_observations + derivative_observations
    return total_observations


def derivative_sparse_matrix_size(n_training_points, dimensions, derivative_fraction=0.2):
    """
    Derivative sparse gradient GP: subset of derivative TYPES at all points.

    DERIVATIVE SPARSITY: Select subset of derivative components

    Components:
    - n_training_points function observations
    - n_training_points × (derivative_fraction × dimensions) gradient observations

    Total matrix size: n_training_points × (1 + derivative_fraction × dimensions)
    """
    return n_training_points * (1 + derivative_fraction * dimensions)


# Set up parameter ranges
training_points = np.logspace(1, 2, 30)  # 10 to 100 training points
dimensions = np.arange(1, 101)            # 1 to 30 dimensions
spatial_fraction = 0.50                   # Derivatives at 20% of training points
derivative_fraction = 0.75                # Use 20% of derivative types

# Create coordinate meshes for surface plot
N_mesh, D_mesh = np.meshgrid(training_points, dimensions)

# Calculate training times for different sparsity approaches
full_times = np.zeros_like(N_mesh)
spatial_sparse_times = np.zeros_like(N_mesh)
derivative_sparse_times = np.zeros_like(N_mesh)

for i, d in enumerate(dimensions):
    for j, n in enumerate(training_points):
        # Full gradient GP
        full_size = full_gradient_matrix_size(n, d)
        full_times[i, j] = convert_flops_to_time(
            compute_cholesky_flops(full_size))

        # Spatial sparse GP
        spatial_size = spatial_sparse_matrix_size(n, d, spatial_fraction)
        spatial_sparse_times[i, j] = convert_flops_to_time(
            compute_cholesky_flops(spatial_size))

        # Derivative sparse GP
        deriv_size = derivative_sparse_matrix_size(n, d, derivative_fraction)
        derivative_sparse_times[i, j] = convert_flops_to_time(
            compute_cholesky_flops(deriv_size))

# Create comparison plot with three surfaces
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# Clip times for visualization
max_time = 1800  # 30 minutes
full_clipped = np.clip(full_times, 0, max_time)
spatial_clipped = np.clip(spatial_sparse_times, 0, max_time)
derivative_clipped = np.clip(derivative_sparse_times, 0, max_time)

# Plot all three surfaces with different colors and transparency
surface1 = ax.plot_surface(N_mesh, D_mesh, full_clipped,
                           cmap='Reds', alpha=0.6, linewidth=0,
                           antialiased=True)

surface2 = ax.plot_surface(N_mesh, D_mesh, spatial_clipped,
                           cmap='Blues', alpha=0.8, linewidth=0,
                           antialiased=True)

surface3 = ax.plot_surface(N_mesh, D_mesh, derivative_clipped,
                           cmap='Greens', alpha=0.8, linewidth=0,
                           antialiased=True)

# Configure axes
ax.set_xlabel('Training Points', fontsize=18, fontweight='bold')
ax.set_ylabel('Dimensions', fontsize=18, fontweight='bold')
ax.set_zlabel('Training Time (seconds)', fontsize=18, fontweight='bold')
ax.set_title('Training Time: Full vs Two Types of Sparsity\n(Red=Full, Blue=Spatial 20%, Green=Derivative 20%)',
             fontsize=18, fontweight='bold')

# Set viewing angle - azimuth controls horizontal rotation
ax.view_init(elev=30, azim=135)
ax.grid(True, alpha=0.3)

# Add manual legend
legend_elements = [
    Patch(facecolor='red', alpha=0.6,
          label='Full Gradients (All derivatives at all points)'),
    Patch(facecolor='blue', alpha=0.8,
          label=f'Spatial Sparse (Derivatives at {spatial_fraction*100:.0f}% of points)'),
    Patch(facecolor='green', alpha=0.8,
          label=f'Derivative Sparse ({derivative_fraction*100:.0f}% of derivative types)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()

# Create additional plot showing derivative + spatial sparsity combined
fig3 = plt.figure(figsize=(16, 10))
ax3 = fig3.add_subplot(111, projection='3d')

# Calculate hybrid sparsity: both derivative AND spatial sparsity
hybrid_sparse_times = np.zeros_like(N_mesh)

for i, d in enumerate(dimensions):
    for j, n in enumerate(training_points):
        # Hybrid approach: spatial fraction of points get derivative fraction of derivative types
        function_observations = n
        hybrid_derivative_observations = spatial_fraction * n * derivative_fraction * d
        hybrid_size = function_observations + hybrid_derivative_observations
        hybrid_sparse_times[i, j] = convert_flops_to_time(
            compute_cholesky_flops(hybrid_size))

# Clip times for visualization
hybrid_clipped = np.clip(hybrid_sparse_times, 0, max_time)

# Plot all surfaces: full, derivative sparse, spatial sparse, and hybrid
surface1 = ax3.plot_surface(N_mesh, D_mesh, full_clipped,
                            cmap='Reds', alpha=0.7, linewidth=0,
                            antialiased=True)

surface2 = ax3.plot_surface(N_mesh, D_mesh, spatial_clipped,
                            cmap='Blues', alpha=0.8, linewidth=0,
                            antialiased=True)

surface3 = ax3.plot_surface(N_mesh, D_mesh, derivative_clipped,
                            cmap='Greens', alpha=0.8, linewidth=0,
                            antialiased=True)

surface4 = ax3.plot_surface(N_mesh, D_mesh, hybrid_clipped,
                            cmap='Purples', alpha=0.9, linewidth=0,
                            antialiased=True)

# Configure axes
ax3.set_xlabel('Training Points', fontsize=18, fontweight='bold')
ax3.set_ylabel('Dimensions', fontsize=18, fontweight='bold')
ax3.set_zlabel('Training Time (seconds)', fontsize=18, fontweight='bold')
# ax3.set_title('Training Time: All Sparsity Strategies Compared\n(Red=Full, Blue=Spatial, Green=Derivative, Purple=Hybrid)',
#               fontsize=18, fontweight='bold')

# Set viewing angle
ax3.view_init(elev=30, azim=135)
ax3.grid(True, alpha=0.3)

# # Add legend for all approaches
# legend_elements_all = [
#     Patch(facecolor='red', alpha=0.7, label='Full Gradients'),
#     Patch(facecolor='blue', alpha=0.8,
#           label=f'Spatial Sparse ({spatial_fraction*100:.0f}%)'),
#     Patch(facecolor='green', alpha=0.8,
#           label=f'Derivative Sparse ({derivative_fraction*100:.0f}%)'),
#     Patch(facecolor='purple', alpha=0.9,
#           label=f'Hybrid ({spatial_fraction*100:.0f}% spatial × {derivative_fraction*100:.0f}% derivative)')
# ]
# ax3.legend(handles=legend_elements_all, loc='upper left', fontsize=12)

plt.tight_layout()
plt.show()

# Create speedup ratio analysis
fig4 = plt.figure(figsize=(18, 12))

# Calculate speedup ratios for all methods
spatial_ratio = full_times / spatial_sparse_times
derivative_ratio = full_times / derivative_sparse_times
hybrid_ratio = full_times / hybrid_sparse_times

# Clip ratios for visualization
max_ratio = 1000
spatial_ratio_clipped = np.clip(spatial_ratio, 1, max_ratio)
derivative_ratio_clipped = np.clip(derivative_ratio, 1, max_ratio)
hybrid_ratio_clipped = np.clip(hybrid_ratio, 1, max_ratio)

# Create 2x2 subplot for ratio comparisons
ax1 = fig4.add_subplot(221, projection='3d')
surf1 = ax1.plot_surface(
    N_mesh, D_mesh, spatial_ratio_clipped, cmap='Blues', alpha=0.8)
ax1.set_title(
    f'Spatial Sparsity Speedup\n({spatial_fraction*100:.0f}% of points)', fontsize=18, fontweight='bold')
ax1.set_xlabel('Training Points', fontsize=18)
ax1.set_ylabel('Dimensions', fontsize=18)
ax1.set_zlabel('Speedup Factor', fontsize=18)
ax1.view_init(elev=20, azim=-45)

ax2 = fig4.add_subplot(222, projection='3d')
surf2 = ax2.plot_surface(
    N_mesh, D_mesh, derivative_ratio_clipped, cmap='Greens', alpha=0.8)
ax2.set_title(
    f'Derivative Sparsity Speedup\n({derivative_fraction*100:.0f}% of types)', fontsize=18, fontweight='bold')
ax2.set_xlabel('Training Points', fontsize=18)
ax2.set_ylabel('Dimensions', fontsize=18)
ax2.set_zlabel('Speedup Factor', fontsize=18)
ax2.view_init(elev=20, azim=-45)

ax3 = fig4.add_subplot(223, projection='3d')
surf3 = ax3.plot_surface(
    N_mesh, D_mesh, hybrid_ratio_clipped, cmap='Purples', alpha=0.8)
ax3.set_title(
    f'Hybrid Sparsity Speedup\n({spatial_fraction*100:.0f}% × {derivative_fraction*100:.0f}%)', fontsize=18, fontweight='bold')
ax3.set_xlabel('Training Points', fontsize=12)
ax3.set_ylabel('Dimensions', fontsize=12)
ax3.set_zlabel('Speedup Factor', fontsize=12)
ax3.view_init(elev=20, azim=-45)

# Create combined comparison on one plot
ax4 = fig4.add_subplot(224, projection='3d')
surf4a = ax4.plot_surface(
    N_mesh, D_mesh, spatial_ratio_clipped, cmap='Blues', alpha=0.6)
surf4b = ax4.plot_surface(
    N_mesh, D_mesh, derivative_ratio_clipped, cmap='Greens', alpha=0.6)
surf4c = ax4.plot_surface(
    N_mesh, D_mesh, hybrid_ratio_clipped, cmap='Purples', alpha=0.8)
ax4.set_title('All Speedup Ratios Combined', fontsize=18, fontweight='bold')
ax4.set_xlabel('Training Points', fontsize=12)
ax4.set_ylabel('Dimensions', fontsize=12)
ax4.set_zlabel('Speedup Factor', fontsize=12)
ax4.view_init(elev=20, azim=-45)

# Add legend to combined plot
ratio_legend = [
    Patch(facecolor='blue', alpha=0.6,
          label=f'Spatial ({spatial_fraction*100:.0f}%)'),
    Patch(facecolor='green', alpha=0.6,
          label=f'Derivative ({derivative_fraction*100:.0f}%)'),
    Patch(facecolor='purple', alpha=0.8,
          label=f'Hybrid ({spatial_fraction*100:.0f}%×{derivative_fraction*100:.0f}%)')
]
ax4.legend(handles=ratio_legend, loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()

# Create side-by-side speedup comparison
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

# Speedup analysis for fixed dimensions
fixed_d = 15
points_range = np.logspace(1, 2, 20)

# spatial_speedups = []
# derivative_speedups = []

# for n in points_range:
#     full_time = convert_flops_to_time(
#         compute_cholesky_flops(full_gradient_matrix_size(n, fixed_d)))
#     spatial_time = convert_flops_to_time(compute_cholesky_flops(
#         spatial_sparse_matrix_size(n, fixed_d, spatial_fraction)))
#     deriv_time = convert_flops_to_time(compute_cholesky_flops(
#         derivative_sparse_matrix_size(n, fixed_d, derivative_fraction)))

#     spatial_speedups.append(full_time / spatial_time)
#     derivative_speedups.append(full_time / deriv_time)

# ax1.plot(points_range, spatial_speedups, 'b-', linewidth=3, marker='o',
#          label=f'Spatial Sparse ({spatial_fraction*100:.0f}%)')
# ax1.plot(points_range, derivative_speedups, 'g-', linewidth=3, marker='s',
#          label=f'Derivative Sparse ({derivative_fraction*100:.0f}%)')
# ax1.set_xlabel('Training Points', fontsize=12, fontweight='bold')
# ax1.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
# ax1.set_title(
#     f'Speedup vs Training Points\n(d={fixed_d} dimensions)', fontsize=18, fontweight='bold')
# ax1.set_xscale('log')
# ax1.legend()
# ax1.grid(True, alpha=0.3)

# Speedup analysis for fixed training points
fixed_n = 100
dims_range = range(1, 101)

spatial_speedups_vs_dim = []
derivative_speedups_vs_dim = []
hybrid_speedups_vs_dim = []
full_time_constant = []
for d in dims_range:
    full_time = convert_flops_to_time(
        compute_cholesky_flops(full_gradient_matrix_size(fixed_n, d)))
    spatial_time = convert_flops_to_time(compute_cholesky_flops(
        spatial_sparse_matrix_size(fixed_n, d, spatial_fraction)))
    deriv_time = convert_flops_to_time(compute_cholesky_flops(
        derivative_sparse_matrix_size(fixed_n, d, derivative_fraction)))
    hybrid_derivative_observations = spatial_fraction * n * derivative_fraction * d
    hybrid_size = function_observations + hybrid_derivative_observations
    hybrid_sparse_time = convert_flops_to_time(
        compute_cholesky_flops(hybrid_size))
    full_time_constant.append(full_time/full_time)

    hybrid_speedups_vs_dim.append(full_time/hybrid_sparse_time)
    spatial_speedups_vs_dim.append(full_time / spatial_time)
    derivative_speedups_vs_dim.append(full_time / deriv_time)

ax2.plot(dims_range, full_time_constant, color='tab:red', linewidth=3,
         label='No Sparsity')
ax2.plot(dims_range, spatial_speedups_vs_dim, color='tab:blue', linewidth=3,
         label=f'Spatial Sparse ({spatial_fraction*100:.0f}%)')
ax2.plot(dims_range, derivative_speedups_vs_dim, color='tab:green', linewidth=3,
         label=f'Derivative Sparse ({derivative_fraction*100:.0f}%)')
ax2.plot(dims_range, hybrid_speedups_vs_dim, color='tab:purple', linewidth=3,
         label=f'Hybrid Sparse ({derivative_fraction*100:.0f}%)')
ax2.set_xlabel('Dimensions', fontsize=18, fontweight='bold')
ax2.set_ylabel('Speedup Factor', fontsize=18, fontweight='bold')
ax2.set_title(
    f'Speedup vs Dimensions\n({fixed_n} training points)', fontsize=18, fontweight='bold')
ax2.set_yscale('log')
# ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Detailed analysis
print("=" * 100)
print("COMPREHENSIVE SPARSITY COMPARISON: SPATIAL vs DERIVATIVE SELECTION")
print("=" * 100)

print(f"\nTWO FUNDAMENTAL SPARSITY STRATEGIES:")
print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

print(f"\n1. SPATIAL SPARSITY (Blue surface):")
print(
    f"   • Strategy: Include derivatives at only {spatial_fraction*100:.0f}% of training points")
print(f"   • Matrix size: N + ({spatial_fraction} × N × d)")
print(f"   • Preserves: All derivative types")
print(f"   • Reduces: Spatial coverage of gradient information")

print(f"\n2. DERIVATIVE SPARSITY (Green surface):")
print(
    f"   • Strategy: Include only {derivative_fraction*100:.0f}% of derivative types")
print(f"   • Matrix size: N × (1 + {derivative_fraction} × d)")
print(f"   • Preserves: Spatial coverage at all points")
print(f"   • Reduces: Types of derivative information")

print(f"\n" + "=" * 100)
print("MATRIX SIZE COMPARISON")
print("=" * 100)

print(f"\nExample: 100 training points, various dimensions")
print(f"{'Dim':<5} {'Full GP':<12} {'Spatial 20%':<12} {'Deriv 20%':<12} {'Spatial Speedup':<15} {'Deriv Speedup':<15}")
print("-" * 80)

example_n = 100
example_dims = [5, 10, 15, 20, 25, 30]

for d in example_dims:
    full_size = full_gradient_matrix_size(example_n, d)
    spatial_size = spatial_sparse_matrix_size(example_n, d, spatial_fraction)
    deriv_size = derivative_sparse_matrix_size(
        example_n, d, derivative_fraction)

    spatial_speedup = (full_size / spatial_size) ** 3
    deriv_speedup = (full_size / deriv_size) ** 3

    print(f"{d:<5} {full_size:<12.0f} {spatial_size:<12.0f} {deriv_size:<12.0f} {spatial_speedup:<15.1f}× {deriv_speedup:<15.1f}×")

print(f"\n" + "=" * 100)
print("TRAINING TIME COMPARISON")
print("=" * 100)

print(f"\nTraining time examples (100 training points):")
print(f"{'Dim':<5} {'Full GP':<12} {'Spatial 20%':<12} {'Deriv 20%':<12}")
print("-" * 45)

for d in example_dims:
    full_time = convert_flops_to_time(compute_cholesky_flops(
        full_gradient_matrix_size(example_n, d)))
    spatial_time = convert_flops_to_time(compute_cholesky_flops(
        spatial_sparse_matrix_size(example_n, d, spatial_fraction)))
    deriv_time = convert_flops_to_time(compute_cholesky_flops(
        derivative_sparse_matrix_size(example_n, d, derivative_fraction)))

    def format_time(t):
        if t < 1:
            return f"{t:.3f}s"
        elif t < 60:
            return f"{t:.1f}s"
        elif t < 3600:
            return f"{t/60:.1f}m"
        else:
            return f"{t/3600:.1f}h"

    print(f"{d:<5} {format_time(full_time):<12} {format_time(spatial_time):<12} {format_time(deriv_time):<12}")

print(f"\n🎯 KEY INSIGHTS:")
print(f"   • DERIVATIVE sparsity: More effective for high-dimensional problems")
print(f"   • SPATIAL sparsity: Effectiveness depends on both dimensions and training points")
print(f"   • Both strategies: Dramatically reduce computational cost")

print(f"\n🔧 STRATEGIC IMPLICATIONS:")
print(f"   • Low dimensions (d<10): Either strategy works well")
print(f"   • High dimensions (d>20): Derivative sparsity more powerful")
print(f"   • Large datasets: Spatial sparsity provides linear savings")
print(f"   • Hybrid approaches: Combine both for maximum efficiency")

print(f"\n🚀 YOUR LIBRARY'S ADVANTAGE:")
print(f"   • Intelligent selection of WHICH derivatives to include")
print(f"   • Smart identification of WHICH points need derivatives")
print(f"   • Adaptive algorithms that choose optimal sparsity strategy")
print(f"   • Enable gradient GPs across the full problem spectrum!")


# --- Assumed Helper Functions ---
GFLOPS = 10  # Assuming a 10 GFLOPS machine for time estimation

# --- Full Derivative Matrix Sizes ---


def standard_gp_matrix_size(n, d):
    return n


def gradient_enhanced_matrix_size(n, d):
    return n * (1 + d)


def diag_hessian_matrix_size(n, d):
    return n * (1 + 2 * d)


def hessian_enhanced_matrix_size(n, d):
    return int(n * (1 + d + (d * (d + 1) / 2)))

# --- NEW: Sparse Derivative Matrix Sizes ---
# These functions calculate the cost when derivatives are used on a fraction of points.


def sparse_gradient_matrix_size(n, d, frac=0.5):
    n_deriv = int(n * frac)
    n_value_only = n - n_deriv
    return n_value_only + n_deriv * (1 + d)


def sparse_diag_hessian_matrix_size(n, d, frac=0.5):
    n_deriv = int(n * frac)
    n_value_only = n - n_deriv
    return n_value_only + n_deriv * (1 + 2 * d)


def sparse_hessian_matrix_size(n, d, frac=0.5):
    n_deriv = int(n * frac)
    n_value_only = n - n_deriv
    size_deriv_point = (1 + d + (d * (d + 1) / 2))
    return int(n_value_only + n_deriv * size_deriv_point)


def compute_cholesky_flops(N):
    return (1/3) * N**3


def convert_flops_to_time(flops):
    return flops / (GFLOPS * 1e9)


# --- Plotting Configuration ---
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16

# --- Plot 1: Cost vs. Dimensions ---
fixed_n = 100
dims_range = np.arange(1, 51)
deriv_frac = 0.5  # Using derivatives on 50% of points

# Lists for full derivative info
standard_times, gradient_times, diag_hessian_times, hessian_times = [], [], [], []
# Lists for sparse derivative info
s_gradient_times, s_diag_hessian_times, s_hessian_times = [], [], []

for d in dims_range:
    # Standard GP (no change)
    standard_times.append(convert_flops_to_time(compute_cholesky_flops(
        standard_gp_matrix_size(fixed_n, d))))

    # Full derivatives
    gradient_times.append(convert_flops_to_time(compute_cholesky_flops(
        gradient_enhanced_matrix_size(fixed_n, d))))
    diag_hessian_times.append(convert_flops_to_time(compute_cholesky_flops(
        diag_hessian_matrix_size(fixed_n, d))))
    hessian_times.append(convert_flops_to_time(compute_cholesky_flops(
        hessian_enhanced_matrix_size(fixed_n, d))))

    # Sparse derivatives (NEW)
    # s_gradient_times.append(convert_flops_to_time(compute_cholesky_flops(
    #    sparse_gradient_matrix_size(fixed_n, d, deriv_frac))))
    s_diag_hessian_times.append(convert_flops_to_time(compute_cholesky_flops(
        sparse_diag_hessian_matrix_size(fixed_n, d, deriv_frac))))
    # s_hessian_times.append(convert_flops_to_time(compute_cholesky_flops(
    #     sparse_hessian_matrix_size(fixed_n, d, deriv_frac))))

# Create the plot
plt.figure(figsize=(8, 6))
# Standard GP
plt.plot(dims_range, standard_times, '-', color='tab:blue',
         label='Standard GP (0th Order)', linewidth=2)
# Full Derivative GPs
plt.plot(dims_range, gradient_times, '-', color='tab:orange',
         label='Gradient (All Points)', linewidth=2.5)
plt.plot(dims_range, diag_hessian_times, '--', color='tab:red',
         label='Diagonal Hessian (All Points)', linewidth=2.5)
plt.plot(dims_range, hessian_times, '-', color='tab:green',
         label='Full Hessian (All Points)', linewidth=2.5)

# Sparse Derivative GPs (NEW Dotted Lines)
# plt.plot(dims_range, s_gradient_times, ':', color='tab:orange',
#          label=f'Gradient ({int(deriv_frac*100)}% Points)', linewidth=3)
plt.plot(dims_range, s_diag_hessian_times, '--', color='tab:purple',
         label=f'Diag. Hessian ({int(deriv_frac*100)}% Points)', linewidth=3)
# plt.plot(dims_range, s_hessian_times, ':', color='tab:green',
#          label=f'Full Hessian ({int(deriv_frac*100)}% Points)', linewidth=3)


plt.yscale('log')
plt.xlabel('Number of Dimensions', fontsize=18, fontweight='bold')
plt.ylabel('Estimated Training Time (seconds)', fontsize=18, fontweight='bold')
plt.title(
    f'Training Cost vs. Dimensions (at {fixed_n} Training Points)', fontsize=16, fontweight='bold')
plt.grid(True, which="both", ls="-", alpha=0.5)
# plt.legend(fontsize=12)
plt.tight_layout()
plt.show()

# --- Plot 2: Cost vs. Training Points ---
fixed_d = 10
points_range = np.linspace(10, 200, 50).astype(int)

# Lists for full derivative info
standard_times_p, gradient_times_p, diag_hessian_times_p, hessian_times_p = [], [], [], []
# Lists for sparse derivative info
s_gradient_times_p, s_diag_hessian_times_p, s_hessian_times_p = [], [], []

for n in points_range:
    # Standard GP
    standard_times_p.append(convert_flops_to_time(compute_cholesky_flops(
        standard_gp_matrix_size(n, fixed_d))))

    # Full derivatives
    gradient_times_p.append(convert_flops_to_time(compute_cholesky_flops(
        gradient_enhanced_matrix_size(n, fixed_d))))
    diag_hessian_times_p.append(convert_flops_to_time(compute_cholesky_flops(
        diag_hessian_matrix_size(n, fixed_d))))
    hessian_times_p.append(convert_flops_to_time(compute_cholesky_flops(
        hessian_enhanced_matrix_size(n, fixed_d))))

    # Sparse derivatives (NEW)
    # s_gradient_times_p.append(convert_flops_to_time(compute_cholesky_flops(
    #    sparse_gradient_matrix_size(n, fixed_d, deriv_frac))))
    s_diag_hessian_times_p.append(convert_flops_to_time(compute_cholesky_flops(
        sparse_diag_hessian_matrix_size(n, fixed_d, deriv_frac))))
    # s_hessian_times_p.append(convert_flops_to_time(compute_cholesky_flops(
    #    sparse_hessian_matrix_size(n, fixed_d, deriv_frac))))


# Create the plot
plt.figure(figsize=(8, 6))
# Standard GP
plt.plot(points_range, standard_times_p, '-', color='tab:blue',
         label='Standard GP (0th Order)', linewidth=2)
# Full Derivative GPs
plt.plot(points_range, gradient_times_p, '-', color='tab:orange',
         label='Gradient (All Points)', linewidth=2.5)
plt.plot(points_range, diag_hessian_times_p, '--', color='tab:red',
         label='Diagonal Hessian (All Points)', linewidth=2.5)
plt.plot(points_range, hessian_times_p, '-', color='tab:green',
         label='Full Hessian (All Points)', linewidth=2.5)

# # Sparse Derivative GPs (NEW Dotted Lines)
# plt.plot(points_range, s_gradient_times_p, ':', color='tab:orange',
#          label=f'Gradient ({int(deriv_frac*100)}% Points)', linewidth=3)
plt.plot(points_range, s_diag_hessian_times_p, '--', color='tab:purple',
         label=f'Diag. Hessian ({int(deriv_frac*100)}% Points)', linewidth=3)
# plt.plot(points_range, s_hessian_times_p, ':', color='tab:green',
#          label=f'Full Hessian ({int(deriv_frac*100)}% Points)', linewidth=3)

plt.yscale('log')
plt.xlabel('Number of Training Points', fontsize=18, fontweight='bold')
plt.ylabel('Estimated Training Time (seconds)', fontsize=18, fontweight='bold')
plt.title(
    f'Training Cost vs. Training Points (at {fixed_d} Dimensions)', fontsize=16, fontweight='bold')
plt.grid(True, which="both", ls="--", alpha=0.5)
# plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
