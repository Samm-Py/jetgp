"""
Benchmark: GEKPLS (SMT) on the OTL Circuit function (6D)
Gradient-enhanced kriging with partial least squares.
CPU only, single thread for fair comparison.

Follows the methodology of Erickson et al. (2018) with sample sizes
n = d, 5d, 10d, using 5 macroreplicates.
"""

import os
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import time
import json
import sys
sys.path.insert(0, '.')

from benchmark_functions import (
    otl_circuit, otl_circuit_gradient,
    generate_test_data, compute_metrics
)
from scipy.stats.qmc import LatinHypercube
from smt.surrogate_models import GEKPLS, DesignSpace

# ============================================================================
# Configuration
# ============================================================================
FUNCTION_NAME = "otl_circuit"
DIM = 6
SAMPLE_SIZES = [DIM, 5 * DIM, 10 * DIM]  # 6, 30, 60
N_MACROREPLICATES = 5
N_TEST = 2000
N_COMP = DIM


def run_single(n_train, seed):
    """
    Run a single GEKPLS benchmark.
    """
    # Generate training data
    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = otl_circuit(X_train)
    grads = otl_circuit_gradient(X_train)

    # Generate test data (fixed seed for consistency)
    X_test, y_test = generate_test_data(otl_circuit, N_TEST, DIM, seed=99)

    # GEKPLS setup — inputs in [0, 1]^d
    xlimits = np.array([[0.0, 1.0]] * DIM)
    design_space = DesignSpace(xlimits)

    t_start = time.perf_counter()

    sm = GEKPLS(
        design_space=design_space,
        theta0=[1e-2] * N_COMP,
        extra_points=1,
        print_prediction=False,
        print_global=False,
        n_comp=N_COMP,
    )

    # Set training values
    sm.set_training_values(X_train, y_vals)

    # Set training derivatives for each dimension
    for j in range(DIM):
        sm.set_training_derivatives(
            X_train, grads[:, j].reshape(-1, 1), j
        )

    # Train
    sm.train()

    t_train = time.perf_counter() - t_start

    # Predict
    t_pred_start = time.perf_counter()
    y_pred = sm.predict_values(X_test).flatten()
    t_pred = time.perf_counter() - t_pred_start

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
        print(f"  GEKPLS — OTL Circuit — n_train = {n_train}")
        print(f"{'='*60}")

        for rep in range(N_MACROREPLICATES):
            seed = 1000 + rep
            print(f"\n  Macroreplicate {rep + 1}/{N_MACROREPLICATES} (seed={seed})")

            try:
                result = run_single(n_train, seed)
                result['macroreplicate'] = rep + 1
                results.append(result)

                print(f"    RMSE:       {result['rmse']:.6e}")
                print(f"    NRMSE:      {result['nrmse']:.6e}")
                print(f"    Train time: {result['train_time']:.2f}s")
                print(f"    Pred time:  {result['pred_time']:.4f}s")
            except Exception as e:
                print(f"    FAILED: {e}")
                results.append({
                    'rmse': float('nan'),
                    'nrmse': float('nan'),
                    'train_time': 0.0,
                    'pred_time': 0.0,
                    'n_train': n_train,
                    'seed': seed,
                    'macroreplicate': rep + 1,
                })

    # Summary
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for n_train in SAMPLE_SIZES:
        subset = [r for r in results if r['n_train'] == n_train]
        rmses = [r['rmse'] for r in subset if not np.isnan(r['rmse'])]
        times = [r['train_time'] for r in subset if r['train_time'] > 0]
        if rmses:
            print(f"\n  n = {n_train}:")
            print(f"    RMSE:  mean={np.mean(rmses):.6e}, std={np.std(rmses):.6e}")
            if times:
                print(f"    Time:  mean={np.mean(times):.2f}s, std={np.std(times):.2f}s")
        else:
            print(f"\n  n = {n_train}: All runs failed")

    # Save results
    output_file = f"results_gekpls_{FUNCTION_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
