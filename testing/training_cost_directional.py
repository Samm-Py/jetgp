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


def function_only_matrix_size(n_training_points):
    """
    Function-only GP: no derivative information.

    Components:
    - n_training_points function observations only

    Total matrix size: n_training_points
    """
    return n_training_points


def directional_derivatives_matrix_size(n_training_points, num_directions):
    """
    Directional derivatives GP: function values + directional derivatives.

    Components:
    - n_training_points function observations
    - n_training_points × num_directions directional derivative observations

    Total matrix size: n_training_points × (1 + num_directions)
    """
    return n_training_points * (1 + num_directions)


def full_gradient_matrix_size(n_training_points, dimensions):
    """
    Full gradient GP: function values + all partial derivatives.

    Components:
    - n_training_points function observations
    - n_training_points × dimensions gradient observations

    Total matrix size: n_training_points × (1 + dimensions)
    """
    return n_training_points * (1 + dimensions)


# Set up parameter ranges
training_points = np.logspace(1, 2, 30)  # 10 to 100 training points
dimensions = np.arange(10, 101)           # 1 to 100 dimensions
# Different numbers of directional derivatives
direction_counts = [1, 2, 3]

# Create coordinate meshes for surface plot
N_mesh, D_mesh = np.meshgrid(training_points, dimensions)

# Calculate training times for different approaches
function_only_times = np.zeros_like(N_mesh)
directional_times = {}
full_gradient_times = np.zeros_like(N_mesh)

# Function-only baseline
for i, d in enumerate(dimensions):
    for j, n in enumerate(training_points):
        func_size = function_only_matrix_size(n)
        function_only_times[i, j] = convert_flops_to_time(
            compute_cholesky_flops(func_size))

# Different directional derivative counts
for num_dirs in direction_counts:
    directional_times[num_dirs] = np.zeros_like(N_mesh)
    for i, d in enumerate(dimensions):
        for j, n in enumerate(training_points):
            dir_size = directional_derivatives_matrix_size(n, num_dirs)
            directional_times[num_dirs][i, j] = convert_flops_to_time(
                compute_cholesky_flops(dir_size))

# Full gradient for comparison
for i, d in enumerate(dimensions):
    for j, n in enumerate(training_points):
        full_size = full_gradient_matrix_size(n, d)
        full_gradient_times[i, j] = convert_flops_to_time(
            compute_cholesky_flops(full_size))

# Create main comparison plot
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

# For log scale, we need to avoid zero values and work with log of the data
min_time = 1e-6  # Small positive value to avoid log(0)
max_time = 1800  # 30 minutes

# Apply log transformation and clipping
function_log = np.log10(np.clip(function_only_times, min_time, max_time))
full_log = np.log10(np.clip(full_gradient_times, min_time, max_time))

# Plot directional derivatives surfaces
colors = ['Greens', 'Blues', 'Purples']
alphas = [0.8, 0.8, 0.8, 0.8]

for idx, num_dirs in enumerate(direction_counts):
    dir_log = np.log10(
        np.clip(directional_times[num_dirs], min_time, max_time))
    ax.plot_surface(N_mesh, D_mesh, dir_log,
                    cmap=colors[idx], alpha=alphas[idx], linewidth=0,
                    antialiased=True)

# Plot full gradient surface (commented out in your code)
surface_full = ax.plot_surface(N_mesh, D_mesh, full_log,
                               cmap='Reds', alpha=0.7, linewidth=0,
                               antialiased=True)

# Configure axes
ax.set_xlabel('Training Points', fontsize=18, fontweight='bold')
ax.set_ylabel('Dimensions', fontsize=18, fontweight='bold')
ax.set_zlabel('Training Time (seconds)', fontsize=18, fontweight='bold')
# Set z-axis limits to start at 1e-6 (in log scale)
ax.set_zlim(bottom=np.log10(1e-6))
# Set custom z-tick labels to show actual time values
z_ticks = ax.get_zticks()
z_labels = [f'{10**tick:.1e}' if tick <
            0 else f'{10**tick:.1f}' for tick in z_ticks]
ax.set_zticklabels(z_labels)

# Set viewing angle
ax.view_init(elev=30, azim=40)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


# Create speedup ratio analysis
fig2 = plt.figure(figsize=(18, 12))

# Calculate speedup ratios vs full gradient
directional_ratios = {}
for num_dirs in direction_counts:
    directional_ratios[num_dirs] = full_gradient_times / \
        directional_times[num_dirs]

# Clip ratios for visualization
max_ratio = 1000
for num_dirs in direction_counts:
    directional_ratios[num_dirs] = np.clip(
        directional_ratios[num_dirs], 0.1, max_ratio)

# Create 2x2 subplot for ratio comparisons
for idx, num_dirs in enumerate(direction_counts):
    ax = fig2.add_subplot(2, 2, idx+1, projection='3d')
    surf = ax.plot_surface(N_mesh, D_mesh, directional_ratios[num_dirs],
                           cmap=colors[idx], alpha=0.8, linewidth=0, antialiased=True)

    ax.set_title(f'{num_dirs} Directional Derivatives\nSpeedup vs Full Gradient',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Training Points', fontsize=12)
    ax.set_ylabel('Dimensions', fontsize=12)
    ax.set_zlabel('Speedup Factor', fontsize=12)
    ax.view_init(elev=30, azim=40)

plt.tight_layout()
plt.show()

# Create side-by-side speedup comparison
fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Speedup analysis for fixed dimensions
fixed_d = 50
points_range = np.logspace(1, 2, 20)

speedups_vs_points = {}
for num_dirs in direction_counts:
    speedups_vs_points[num_dirs] = []

    for n in points_range:
        full_time = convert_flops_to_time(
            compute_cholesky_flops(full_gradient_matrix_size(n, fixed_d)))
        dir_time = convert_flops_to_time(
            compute_cholesky_flops(directional_derivatives_matrix_size(n, num_dirs)))

        speedups_vs_points[num_dirs].append(full_time / dir_time)

# Plot speedup vs training points
for idx, num_dirs in enumerate(direction_counts):
    ax1.plot(points_range, speedups_vs_points[num_dirs],
             linewidth=3, marker='o', label=f'{num_dirs} directions')

ax1.set_xlabel('Training Points', fontsize=12, fontweight='bold')
ax1.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
ax1.set_title(f'Speedup vs Training Points\n(d={fixed_d} dimensions)',
              fontsize=14, fontweight='bold')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Speedup analysis for fixed training points
fixed_n = 50
dims_range = range(5, 101, 5)

speedups_vs_dims = {}
for num_dirs in direction_counts:
    speedups_vs_dims[num_dirs] = []

    for d in dims_range:
        full_time = convert_flops_to_time(
            compute_cholesky_flops(full_gradient_matrix_size(fixed_n, d)))
        dir_time = convert_flops_to_time(
            compute_cholesky_flops(directional_derivatives_matrix_size(fixed_n, num_dirs)))

        speedups_vs_dims[num_dirs].append(full_time / dir_time)

# Plot speedup vs dimensions
for idx, num_dirs in enumerate(direction_counts):
    ax2.plot(dims_range, speedups_vs_dims[num_dirs],
             linewidth=3, marker='s', label=f'{num_dirs} directions')

ax2.set_xlabel('Dimensions', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
ax2.set_title(f'Speedup vs Dimensions\n({fixed_n} training points)',
              fontsize=14, fontweight='bold')
ax2.set_yscale('log')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Create efficiency frontier plot
fig4, ax = plt.subplots(1, 1, figsize=(12, 8))

# For different problem sizes, show the efficiency frontier
problem_sizes = [(25, 20), (50, 50), (100, 100)]
colors_frontier = ['blue', 'green', 'red']

for idx, (n, d) in enumerate(problem_sizes):
    # Calculate cost and speedup for different numbers of directions
    directions_range = range(1, min(d, 51))
    costs = []
    speedups = []

    full_time = convert_flops_to_time(
        compute_cholesky_flops(full_gradient_matrix_size(n, d)))

    for num_dirs in directions_range:
        dir_time = convert_flops_to_time(
            compute_cholesky_flops(directional_derivatives_matrix_size(n, num_dirs)))

        # Cost as fraction of full gradient cost
        cost_fraction = dir_time / full_time
        speedup = full_time / dir_time

        costs.append(cost_fraction)
        speedups.append(speedup)

    ax.plot(costs, speedups, 'o-', linewidth=2, markersize=6,
            color=colors_frontier[idx], label=f'n={n}, d={d}')

ax.set_xlabel('Computational Cost (fraction of full gradient)',
              fontsize=12, fontweight='bold')
ax.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
ax.set_title('Efficiency Frontier: Directional Derivatives',
             fontsize=14, fontweight='bold')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Detailed analysis
print("=" * 100)
print("DIRECTIONAL DERIVATIVES GP ANALYSIS")
print("=" * 100)

print(f"\nMATRIX SIZE SCALING:")
print(f"━━━━━━━━━━━━━━━━━━━━━━")
print(f"• Function only:        n")
print(f"• Directional (k dirs): n × (1 + k)")
print(f"• Full gradient:        n × (1 + d)")

print(f"\nKEY ADVANTAGE: Directional derivatives scale independently of problem dimension!")

print(f"\n" + "=" * 100)
print("MATRIX SIZE COMPARISON")
print("=" * 100)

print(f"\nExample: 100 training points, various dimensions")
print(f"{'Dim':<5} {'Function':<10} {'5 dirs':<10} {'10 dirs':<10} {'20 dirs':<10} {'Full grad':<12}")
print("-" * 65)

example_n = 100
example_dims = [5, 10, 20, 50, 100]

for d in example_dims:
    func_size = function_only_matrix_size(example_n)
    dir5_size = directional_derivatives_matrix_size(example_n, 5)
    dir10_size = directional_derivatives_matrix_size(example_n, 10)
    dir20_size = directional_derivatives_matrix_size(example_n, 20)
    full_size = full_gradient_matrix_size(example_n, d)

    print(f"{d:<5} {func_size:<10.0f} {dir5_size:<10.0f} {dir10_size:<10.0f} {dir20_size:<10.0f} {full_size:<12.0f}")

print(f"\n" + "=" * 100)
print("SPEEDUP ANALYSIS")
print("=" * 100)

print(f"\nSpeedup vs Full Gradient (100 training points):")
print(f"{'Dim':<5} {'5 dirs':<10} {'10 dirs':<10} {'20 dirs':<10} {'50 dirs':<10}")
print("-" * 55)

for d in example_dims:
    speedups = []
    for num_dirs in [5, 10, 20, 50]:
        full_time = convert_flops_to_time(
            compute_cholesky_flops(full_gradient_matrix_size(example_n, d)))
        dir_time = convert_flops_to_time(
            compute_cholesky_flops(directional_derivatives_matrix_size(example_n, num_dirs)))
        speedup = full_time / dir_time
        speedups.append(speedup)

    print(
        f"{d:<5} {speedups[0]:<10.1f} {speedups[1]:<10.1f} {speedups[2]:<10.1f} {speedups[3]:<10.1f}")

print(f"\n" + "=" * 100)
print("TRAINING TIME COMPARISON")
print("=" * 100)

print(f"\nTraining time examples (100 training points, 50 dimensions):")
print(f"{'Method':<20} {'Matrix Size':<12} {'Time':<12}")
print("-" * 50)


def format_time(t):
    if t < 1:
        return f"{t:.3f}s"
    elif t < 60:
        return f"{t:.1f}s"
    elif t < 3600:
        return f"{t/60:.1f}m"
    else:
        return f"{t/3600:.1f}h"


example_n, example_d = 100, 50

methods = [
    ("Function only", function_only_matrix_size(example_n)),
    ("5 directions", directional_derivatives_matrix_size(example_n, 5)),
    ("10 directions", directional_derivatives_matrix_size(example_n, 10)),
    ("20 directions", directional_derivatives_matrix_size(example_n, 20)),
    ("50 directions", directional_derivatives_matrix_size(example_n, 50)),
    ("Full gradient", full_gradient_matrix_size(example_n, example_d))
]

for method_name, size in methods:
    time = convert_flops_to_time(compute_cholesky_flops(size))
    print(f"{method_name:<20} {size:<12.0f} {format_time(time):<12}")

print(f"\n" + "=" * 100)
print("KEY INSIGHTS")
print("=" * 100)

print(f"\n🎯 COMPUTATIONAL ADVANTAGES:")
print(f"   • DIMENSION-INDEPENDENT scaling: Cost doesn't grow with problem dimension")
print(f"   • TUNABLE gradient information: Choose exactly how much derivative info to include")
print(f"   • MASSIVE speedups in high dimensions: 10x-1000x faster than full gradients")
print(f"   • PREDICTABLE performance: Linear scaling with number of directions")

print(f"\n🔧 STRATEGIC IMPLICATIONS:")
print(f"   • Low dimensions (d<10): Modest advantage over full gradients")
print(f"   • High dimensions (d>20): Dramatic computational savings")
print(f"   • Very high dimensions (d>50): Often the ONLY feasible gradient approach")
print(f"   • Adaptive selection: Can dynamically choose number of directions")

print(f"\n🚀 OPTIMAL USAGE:")
print(f"   • 5-10 directions: Good balance for most problems")
print(f"   • 10-20 directions: Rich gradient information with modest cost")
print(f"   • >20 directions: Diminishing returns unless very high precision needed")
print(f"   • Direction selection: Smart algorithms can choose most informative directions")

print(f"\n📊 EFFICIENCY FRONTIER:")
print(f"   • Each additional direction provides decreasing marginal information")
print(f"   • Sweet spot: Usually 5-15 directions for practical problems")
print(f"   • Adaptive strategies: Start with few directions, add more if needed")


def cholesky_flops(matrix_size):
    """Calculate FLOPs for Cholesky decomposition: O(n^3)"""
    return matrix_size**3


def flops_to_seconds(flops, gflops=10):
    """Convert FLOPs to time assuming 10 GFLOPS performance"""
    return flops / (gflops * 1e9)


def directional_matrix_size(n_points, num_directions):
    """Matrix size for directional derivatives: n * (1 + num_directions)"""
    return n_points * (1 + num_directions)


def full_gradient_matrix_size(n_points, dimensions):
    """Matrix size for full gradient: n * (1 + dimensions)"""
    return n_points * (1 + dimensions)


def calculate_ratio(n_points, dimensions, num_directions):
    """Calculate speedup ratio: full_gradient_time / directional_time"""
    # Full gradient approach
    full_size = full_gradient_matrix_size(n_points, dimensions)
    full_time = flops_to_seconds(cholesky_flops(full_size))

    # Directional derivatives approach
    dir_size = directional_matrix_size(n_points, num_directions)
    dir_time = flops_to_seconds(cholesky_flops(dir_size))

    return full_time / dir_time


# Parameter ranges
n_points = np.logspace(1, 2, 25)    # 10 to 100 training points
dimensions = np.arange(5, 101)      # 5 to 100 dimensions
direction_counts = [1, 2, 5, 10]        # Number of directional derivatives

# Create meshgrid
N, D = np.meshgrid(n_points, dimensions)

# Calculate ratios for each directional derivative count
ratios = {}
for num_dirs in direction_counts:
    ratios[num_dirs] = np.zeros_like(N)

    for i, d in enumerate(dimensions):
        for j, n in enumerate(n_points):
            ratios[num_dirs][i, j] = calculate_ratio(n, d, num_dirs)

# Create 3D surface plots
fig, axes = plt.subplots(1, 4, figsize=(
    20, 6), subplot_kw={'projection': '3d'})
colors = ['Greens', 'Blues', 'Purples', 'Reds']

for idx, num_dirs in enumerate(direction_counts):
    ax = axes[idx]

    # Clip extreme values for better visualization
    ratio_data = np.clip(ratios[num_dirs], 1, 500)

    surf = ax.plot_surface(N, D, ratio_data, cmap=colors[idx],
                           alpha=0.8, linewidth=0, antialiased=True)

    ax.set_xlabel('Training Points', fontweight='bold')
    ax.set_ylabel('Dimensions', fontweight='bold')
    ax.set_zlabel('Speedup Factor', fontweight='bold')
    ax.set_title(f'{num_dirs} Directional Derivative{"s" if num_dirs > 1 else ""}\nvs Full Gradient',
                 fontweight='bold')
    ax.view_init(elev=25, azim=-45)

plt.tight_layout()
plt.show()

# Create combined surface plot
fig2 = plt.figure(figsize=(14, 8))
ax = fig2.add_subplot(111, projection='3d')

alphas = [0.9, 0.7, 0.5, .5]
for idx, num_dirs in enumerate(direction_counts):
    ratio_data = np.clip(ratios[num_dirs], 1, 500)
    ax.plot_surface(N, D, ratio_data, cmap=colors[idx],
                    alpha=alphas[idx], linewidth=0, antialiased=True)

ax.set_xlabel('Training Points', fontsize=14, fontweight='bold')
ax.set_ylabel('Dimensions', fontsize=14, fontweight='bold')
ax.set_zlabel('Speedup Factor', fontsize=14, fontweight='bold')
ax.set_title('Directional Derivatives Speedup Analysis',
             fontsize=16, fontweight='bold')
ax.view_init(elev=25, azim=-45)

plt.tight_layout()
plt.show()

# Create 2D heatmaps
fig3, axes = plt.subplots(1, 4, figsize=(18, 5))

for idx, num_dirs in enumerate(direction_counts):
    ax = axes[idx]

    ratio_data = np.clip(ratios[num_dirs], 1, 100)
    im = ax.imshow(ratio_data, aspect='auto', origin='lower',
                   cmap=colors[idx], vmin=1, vmax=100,
                   extent=[n_points.min(), n_points.max(),
                           dimensions.min(), dimensions.max()])

    ax.set_xlabel('Training Points', fontweight='bold')
    ax.set_ylabel('Dimensions', fontweight='bold')
    ax.set_title(f'{num_dirs} Direction{"s" if num_dirs > 1 else ""} - Speedup Ratio',
                 fontweight='bold')

    plt.colorbar(im, ax=ax, label='Speedup Factor')

plt.tight_layout()
plt.show()


# Speedup analysis for fixed training points
fixed_n = 100
dims_range = range(3, 101)
# Create side-by-side speedup comparison
fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

time = []
time_1 = []
time_2 = []
time_3 = []
for d in dims_range:
    full_matrix_size = full_gradient_matrix_size(fixed_n, d)
    full_time = convert_flops_to_time(
        compute_cholesky_flops(full_matrix_size))
    sub_matrix_size = directional_derivatives_matrix_size(fixed_n, 1)
    time_1_submodels = convert_flops_to_time(
        compute_cholesky_flops(sub_matrix_size))
    sub_matrix_size = directional_derivatives_matrix_size(fixed_n, 2)
    time_2_submodels = convert_flops_to_time(
        compute_cholesky_flops(sub_matrix_size))
    sub_matrix_size = directional_derivatives_matrix_size(fixed_n, 3)
    time_3_submodels = convert_flops_to_time(
        compute_cholesky_flops(sub_matrix_size))

    time.append(full_time/full_time)
    time_1.append(full_time/time_1_submodels)
    time_2.append(full_time / time_2_submodels)
    time_3.append(full_time / time_3_submodels)

ax2.plot(dims_range, time, color='tab:red', linewidth=3)
ax2.plot(dims_range, time_1, color='tab:purple', linewidth=3)
ax2.plot(dims_range, time_2, color='tab:blue', linewidth=3)
ax2.plot(dims_range, time_3, color='tab:green', linewidth=3,)
ax2.set_xlabel('Dimensions', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speedup Factor', fontsize=12, fontweight='bold')
ax2.set_title(
    f'Speedup vs Dimensions\n({fixed_n} training points)', fontsize=14, fontweight='bold')
ax2.set_yscale('log')
# ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Numerical analysis
print("=" * 70)
print("DIRECTIONAL DERIVATIVES SPEEDUP ANALYSIS")
print("=" * 70)

test_cases = [
    (50, 20),   # Medium problem
    (50, 50),   # Square problem
    (100, 50),  # More points
    (100, 100)  # Large problem
]

print(f"\nSpeedup ratios for different problem sizes:")
print(f"{'Problem':<12} {'1 dir':<8} {'2 dirs':<8} {'5 dirs':<8}")
print("-" * 40)

for n, d in test_cases:
    speedups = []
    for num_dirs in direction_counts:
        ratio = calculate_ratio(n, d, num_dirs)
        speedups.append(ratio)

    print(
        f"n={n:2d}, d={d:2d}   {speedups[0]:6.1f}x  {speedups[1]:6.1f}x  {speedups[2]:6.1f}x")

print(f"\n" + "=" * 70)
print("STATISTICAL SUMMARY")
print("=" * 70)

for num_dirs in direction_counts:
    data = ratios[num_dirs]
    print(f"\n{num_dirs} Directional Derivative{'s' if num_dirs > 1 else ''}:")
    print(f"  Mean speedup:   {np.mean(data):6.1f}x")
    print(f"  Median speedup: {np.median(data):6.1f}x")
    print(f"  Max speedup:    {np.max(data):6.1f}x")
    print(f"  Min speedup:    {np.min(data):6.1f}x")

print(f"\n" + "=" * 70)
print("DIMENSIONALITY ANALYSIS")
print("=" * 70)

# Fixed training points, varying dimensions
fixed_n = 75
dim_range = [5, 25, 50, 75, 100]

print(f"\nSpeedup vs dimensions (n={fixed_n} training points):")
print(f"{'Dimensions':<12} {'1 dir':<8} {'2 dirs':<8} {'5 dirs':<8}")
print("-" * 40)

for d in dim_range:
    speedups = []
    for num_dirs in direction_counts:
        ratio = calculate_ratio(fixed_n, d, num_dirs)
        speedups.append(ratio)

    print(
        f"d={d:2d}         {speedups[0]:6.1f}x  {speedups[1]:6.1f}x  {speedups[2]:6.1f}x")

print(f"\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)
print(f"• Speedup increases dramatically with problem dimension")
print(f"• 1 directional derivative: {np.mean(ratios[1]):.0f}x average speedup")
print(
    f"• 2 directional derivatives: {np.mean(ratios[2]):.0f}x average speedup")
print(
    f"• 5 directional derivatives: {np.mean(ratios[5]):.0f}x average speedup")
print(f"• Maximum benefits occur in high-dimensional problems")
print(f"• Diminishing returns beyond 2-5 directional derivatives")
