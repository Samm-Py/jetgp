import numpy as np
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.stats import qmc
import pyoti.sparse as oti
# Assuming these are defined elsewhere in your codebase
# You'll need to import or define these functions:
# - generate_pointwise_rays
# - apply_pointwise_perturb
# - oti (hypercomplex library)

# Placeholder constants - adjust these to match your actual values
a_ish = 1.0
b_ish = 0.5


def true_function(X, alg=np):
    """Your original function"""
    x1, x2, x3 = X[0, 0], X[0, 1], X[0, 2]
    return (alg.sin(x1)
            + a_ish * alg.sin(x2)**2
            + b_ish * x3**4 * alg.sin(x1))


def time_function_evaluation(func, X, alg, n_runs=100):
    """Time a function evaluation over multiple runs"""
    times = []
    for _ in range(n_runs):
        start_time = time.perf_counter()
        result = func(X, alg=alg)
        end_time = time.perf_counter()
        times.append(end_time - start_time)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'times': times
    }


def sobol_points(n_samples, seed=None, scramble=True):
    """
    Sobol sampling on [-π, π]^3.

    Parameters
    ----------
    n_samples : int
        Number of points you want.  
        • If `scramble=False` you must supply a power-of-two (2, 4, 8, …).  
        • With `scramble=True` any positive integer works.
    seed : int or None
        Random seed for reproducible scrambling.
    scramble : bool
        Scramble the sequence (Owen scrambling). Recommended unless you
        truly need the deterministic Sobol order.

    Returns
    -------
    X : ndarray, shape (n_samples, 3)
        Points mapped to [-π, π] in each coordinate.
    """
    sampler = qmc.Sobol(d=3, scramble=scramble, seed=seed)

    if scramble:
        # Sobol.random works for any n when scrambled
        unit = sampler.random(n_samples)                # (n,3) in [0,1)
    else:
        # must draw a power-of-two with random_base2
        m = int(np.ceil(np.log2(n_samples)))
        unit = sampler.random_base2(m)[:n_samples]

    X = -np.pi + 2 * np.pi * unit        # affine map to [-π, π]^3
    return X


def generate_pointwise_rays(n_samples=24, n_order=1, seed=1):
    """Generate random rays instead of gradient-based ones"""
    X = sobol_points(n_samples, seed)
    rays, tags = [], []

    # Set random seed for reproducible random rays
    np.random.seed(seed)

    for idx, (x1, x2, x3) in enumerate(X):
        # Generate random unit vector instead of gradient
        g = np.random.randn(3)  # Random normal vector
        g /= np.linalg.norm(g)  # Normalize to unit vector
        rays.append(g[:, None])  # (3,1)
        tags.append(idx + 1)     # unique tag
    return X, rays, tags


def apply_pointwise_perturb(X, rays, tags, n_order):
    X_hc = oti.array(X)
    for i, (ray, tag) in enumerate(zip(rays, tags)):
        e_tag = oti.e(1, order=n_order)            # dual unit
        X_hc[i, :] += (oti.array(ray)*e_tag).T
    return X_hc


def benchmark_hypercomplex_vs_numpy(orders=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_runs=1000):
    """
    Compare timing of numpy vs hypercomplex function evaluation across different orders
    Single point evaluation to avoid numpy parallelization issues
    """
    results = {
        'numpy': [],
        'hypercomplex': defaultdict(list),
        'orders': orders
    }

    print("Starting single-point benchmark...")
    print(f"Number of runs per test: {n_runs}")
    print("=" * 50)

    for order in orders:
        print(f"\nTesting order {order}...")

        # Generate single point data for this order
        X, rays, tags = generate_pointwise_rays(1, order)  # Single point only
        X_hc = apply_pointwise_perturb(X, rays, tags, order)

        # Time numpy evaluation
        print("  Timing numpy evaluation...")
        numpy_stats = time_function_evaluation(true_function, X, np, n_runs)
        results['numpy'].append(numpy_stats)

        # Time hypercomplex evaluation
        print("  Timing hypercomplex evaluation...")
        hc_stats = time_function_evaluation(true_function, X_hc, oti, n_runs)
        results['hypercomplex'][order] = hc_stats

        # Print results for this order
        print(
            f"  NumPy:        {numpy_stats['mean']*1e6:.1f} ± {numpy_stats['std']*1e6:.1f} μs")
        print(
            f"  Hypercomplex: {hc_stats['mean']*1e6:.1f} ± {hc_stats['std']*1e6:.1f} μs")
        print(f"  Slowdown:     {hc_stats['mean']/numpy_stats['mean']:.2f}x")

    return results


def plot_timing_results(results):
    """Plot the timing comparison results"""
    orders = results['orders']
    numpy_means = [stats['mean'] *
                   1e6 for stats in results['numpy']]  # Convert to μs
    numpy_stds = [stats['std'] * 1e6 for stats in results['numpy']]

    hc_means = [results['hypercomplex'][order]
                ['mean'] * 1e6 for order in orders]
    hc_stds = [results['hypercomplex'][order]['std'] * 1e6 for order in orders]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Absolute timing
    ax1.errorbar(orders, numpy_means, yerr=numpy_stds,
                 label='NumPy', marker='o', capsize=5)
    ax1.errorbar(orders, hc_means, yerr=hc_stds,
                 label='Hypercomplex', marker='s', capsize=5)
    ax1.set_xlabel('Order')
    ax1.set_ylabel('Time (μs)')
    ax1.set_title('Single Point Function Evaluation Time')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Slowdown factor
    slowdowns = [hc_means[i] / numpy_means[i] for i in range(len(orders))]
    ax2.plot(orders, slowdowns, marker='o', linewidth=2)
    ax2.set_xlabel('Order')
    ax2.set_ylabel('Slowdown Factor (Hypercomplex / NumPy)')
    ax2.set_title('Hypercomplex Overhead')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def detailed_single_point_comparison(order=2, n_runs=1000):
    """
    Detailed comparison for a single order with more runs
    """
    print(f"\nDetailed comparison for order {order}")
    print("=" * 40)

    # Generate single point data
    X, rays, tags = generate_pointwise_rays(1, order)  # Single point
    X_hc = apply_pointwise_perturb(X, rays, tags, order)

    # Time both approaches
    numpy_stats = time_function_evaluation(true_function, X, np, n_runs)
    hc_stats = time_function_evaluation(true_function, X_hc, oti, n_runs)

    print(f"NumPy evaluation:")
    print(f"  Mean: {numpy_stats['mean']*1e6:.1f} μs")
    print(f"  Std:  {numpy_stats['std']*1e6:.1f} μs")
    print(f"  Min:  {numpy_stats['min']*1e6:.1f} μs")
    print(f"  Max:  {numpy_stats['max']*1e6:.1f} μs")

    print(f"\nHypercomplex evaluation:")
    print(f"  Mean: {hc_stats['mean']*1e6:.1f} μs")
    print(f"  Std:  {hc_stats['std']*1e6:.1f} μs")
    print(f"  Min:  {hc_stats['min']*1e6:.1f} μs")
    print(f"  Max:  {hc_stats['max']*1e6:.1f} μs")

    print(f"\nSlowdown factor: {hc_stats['mean']/numpy_stats['mean']:.2f}x")

    return numpy_stats, hc_stats


if __name__ == "__main__":
    # Make sure to import/define the required functions and libraries before running:
    # from your_module import generate_pointwise_rays, apply_pointwise_perturb, oti

    try:
        # Run the main benchmark (single point only)
        results = benchmark_hypercomplex_vs_numpy(
            orders=[1, 2, 3, 4, 5],
            n_runs=10000000  # More runs since we're doing single points
        )

        # Plot results
        plot_timing_results(results)

        # Detailed single-point comparison
        detailed_single_point_comparison(order=2, n_runs=1000)

    except NameError as e:
        print(f"Error: {e}")
        print("Please make sure to import the required functions:")
        print("- generate_pointwise_rays")
        print("- apply_pointwise_perturb")
        print("- oti (hypercomplex library)")
        print("- Set the correct values for a_ish and b_ish")
