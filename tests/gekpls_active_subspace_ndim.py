"""
Benchmark: GEKPLS (SMT) on the 10D active subspace function.
Gradient-enhanced kriging with partial least squares, all 10 partial derivatives.

Same as gekpls_active_subspace.py but with extra_points=DIM instead of 1.
This adds DIM extra points per training point for the PLS regression,
giving GEKPLS more gradient information to work with.
"""

import numpy as np
import time
import json
import sys
sys.path.insert(0, '.')

from benchmark_functions import (
    active_subspace_10d, active_subspace_10d_gradient,
    generate_test_data, compute_metrics
)
from scipy.stats.qmc import LatinHypercube
from smt.surrogate_models import GEKPLS, DesignSpace

FUNCTION_NAME = "active_subspace_10d"
DIM = 10
SAMPLE_SIZES = [DIM, 5 * DIM, 10 * DIM]
N_MACROREPLICATES = 5
N_TEST = 2000
N_COMP = DIM
EXTRA_POINTS = DIM  # N_DIM extra points per training point


def run_single(n_train, seed):
    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = active_subspace_10d(X_train)
    grads = active_subspace_10d_gradient(X_train)

    X_test, y_test = generate_test_data(active_subspace_10d, N_TEST, DIM, seed=99)

    xlimits = np.array([[0.0, 1.0]] * DIM)
    design_space = DesignSpace(xlimits)

    t_start = time.perf_counter()

    sm = GEKPLS(
        design_space=design_space,
        theta0=[1e-2] * N_COMP,
        extra_points=EXTRA_POINTS,
        print_prediction=False,
        print_global=False,
        n_comp=N_COMP,
    )

    sm.set_training_values(X_train, y_vals)
    for j in range(DIM):
        sm.set_training_derivatives(X_train, grads[:, j].reshape(-1, 1), j)

    sm.train()
    t_train = time.perf_counter() - t_start

    t_pred_start = time.perf_counter()
    y_pred = sm.predict_values(X_test).flatten()
    t_pred = time.perf_counter() - t_pred_start

    metrics = compute_metrics(y_test, y_pred)
    metrics['train_time'] = t_train
    metrics['pred_time'] = t_pred
    metrics['n_train'] = n_train
    metrics['seed'] = seed
    metrics['extra_points'] = EXTRA_POINTS
    return metrics


def main():
    results = []
    for n_train in SAMPLE_SIZES:
        print(f"\n{'='*60}")
        print(f"  GEKPLS (extra_points={EXTRA_POINTS}) — Active Subspace 10D — n_train = {n_train}")
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
                    'rmse': float('nan'), 'nrmse': float('nan'),
                    'train_time': 0.0, 'pred_time': 0.0,
                    'n_train': n_train, 'seed': seed,
                    'macroreplicate': rep + 1,
                    'extra_points': EXTRA_POINTS,
                })

    print(f"\n{'='*60}")
    print(f"  Summary (extra_points={EXTRA_POINTS})")
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

    output_file = f"results_gekpls_ndim_{FUNCTION_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
