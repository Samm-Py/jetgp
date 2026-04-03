"""
Benchmark: JetGP DEGP on the Dette-Pepelyshev function (8D)
First-order gradient-enhanced GP with SE anisotropic kernel.

Follows the methodology of Erickson et al. (2018) with sample sizes
n = 5d = 40 and n = 10d = 80, using 5 macroreplicates.
"""

import numpy as np
import time
import json
import sys
sys.path.insert(0, '.')

from benchmark_functions import (
    dette_pepelyshev, dette_pepelyshev_gradient,
    generate_test_data, compute_metrics
)
from scipy.stats.qmc import LatinHypercube

from jetgp.full_degp.degp import degp

# ============================================================================
# Configuration
# ============================================================================
FUNCTION_NAME = "dette_pepelyshev"
DIM = 8
SAMPLE_SIZES = [5 * DIM, 10 * DIM]  # 40, 80
N_MACROREPLICATES = 5
N_TEST = 2000
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"
OPTIMIZER = "lbfgs"
N_RESTARTS = 10

# First-order derivatives: df/dx_i for i = 1..8
DER_INDICES = [[[[i, 1]] for i in range(1, DIM + 1)]]


def run_single(n_train, seed):
    """
    Run a single JetGP DEGP benchmark.

    Parameters
    ----------
    n_train : int
        Number of training points.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    result : dict
        Dictionary with metrics and timing.
    """
    # Generate training data
    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = dette_pepelyshev(X_train)
    grads = dette_pepelyshev_gradient(X_train)

    # Generate test data (fixed seed for consistency across runs)
    X_test, y_test = generate_test_data(dette_pepelyshev, N_TEST, DIM, seed=99)

    # Package training data for JetGP
    y_train_list = [y_vals.reshape(-1, 1)]
    for j in range(DIM):
        y_train_list.append(grads[:, j].reshape(-1, 1))

    # Initialize model
    model = degp(
        X_train, y_train_list,
        n_order=1, n_bases=DIM,
        der_indices=DER_INDICES,
        normalize=True,
        kernel=KERNEL, kernel_type=KERNEL_TYPE
    )

    # Optimize hyperparameters
    t_start = time.perf_counter()
    params = model.optimize_hyperparameters(
        optimizer=OPTIMIZER,
        n_restart_optimizer=N_RESTARTS,
        debug=True
    )

    t_train = time.perf_counter() - t_start

    # Predict
    t_pred_start = time.perf_counter()
    y_pred = model.predict(X_test, params, calc_cov=False, return_deriv=False)
    t_pred = time.perf_counter() - t_pred_start

    # Handle output shape
    if y_pred.ndim == 2:
        y_pred = y_pred[0, :]
    y_pred = y_pred.flatten()

    # Compute metrics
    metrics = compute_metrics(y_test, y_pred)
    metrics['train_time'] = t_train
    metrics['pred_time'] = t_pred
    metrics['n_train'] = n_train
    metrics['seed'] = seed

    return metrics


def main():
    results = []

    for n_train in SAMPLE_SIZES:
        print(f"\n{'='*60}")
        print(f"  JetGP DEGP — Dette-Pepelyshev — n_train = {n_train}")
        print(f"{'='*60}")

        for rep in range(N_MACROREPLICATES):
            seed = 1000 + rep
            print(f"\n  Macroreplicate {rep + 1}/{N_MACROREPLICATES} (seed={seed})")

            result = run_single(n_train, seed)
            result['macroreplicate'] = rep + 1
            results.append(result)

            print(f"    RMSE:       {result['rmse']:.6e}")
            print(f"    NRMSE:      {result['nrmse']:.6e}")
            print(f"    Train time: {result['train_time']:.2f}s")
            print(f"    Pred time:  {result['pred_time']:.4f}s")

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for n_train in SAMPLE_SIZES:
        subset = [r for r in results if r['n_train'] == n_train]
        rmses = [r['rmse'] for r in subset]
        times = [r['train_time'] for r in subset]
        print(f"\n  n = {n_train}:")
        print(f"    RMSE:  mean={np.mean(rmses):.6e}, std={np.std(rmses):.6e}")
        print(f"    Time:  mean={np.mean(times):.2f}s, std={np.std(times):.2f}s")

    # Save results
    output_file = f"results_jetgp_{FUNCTION_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()