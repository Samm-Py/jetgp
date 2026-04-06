"""
Benchmark: JetGP DDEGP on the 10D active subspace function.
Directional-derivative GP using the two known active subspace directions.
SE anisotropic kernel, JADE optimizer with local L-BFGS-B refinement.
"""

import os
import numpy as np
import time
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import (
    active_subspace_10d, active_subspace_10d_directional,
    get_active_subspace_directions,
    generate_test_data, compute_metrics
)
from scipy.stats.qmc import LatinHypercube
from jetgp.full_ddegp.ddegp import ddegp

FUNCTION_NAME = "active_subspace_10d"
DIM = 10
N_RAYS = 2
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TESTS_DIR, 'data')
SAMPLE_SIZES = [DIM, 5 * DIM, 10 * DIM]
N_MACROREPLICATES = 5
N_TEST = 2000
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"

# Active subspace directions as (DIM, N_RAYS) matrix
_RAYS = get_active_subspace_directions()

# der_indices: one group, N_RAYS directional derivatives
# OTI ray indices are 1-based
DER_INDICES = [[[[r + 1, 1]] for r in range(N_RAYS)]]


def run_single(n_train, seed):
    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)

    y_vals = active_subspace_10d(X_train)
    dir_derivs = active_subspace_10d_directional(X_train)  # (n_train, 2)

    X_test, y_test = generate_test_data(active_subspace_10d, N_TEST, DIM, seed=99)

    # y_train_list: function values + one array per directional derivative
    y_train_list = [y_vals.reshape(-1, 1)]
    for r in range(N_RAYS):
        y_train_list.append(dir_derivs[:, r].reshape(-1, 1))

    # All training points have both directional derivatives
    derivative_locations = [list(range(n_train)) for _ in range(N_RAYS)]

    model = ddegp(
        X_train, y_train_list,
        n_order=1,
        der_indices=DER_INDICES,
        rays=_RAYS,
        derivative_locations=derivative_locations,
        normalize=True,
        kernel=KERNEL, kernel_type=KERNEL_TYPE
    )

    t_start = time.perf_counter()
    params = model.optimize_hyperparameters(
        optimizer="jade",
        n_generations=10,
        local_opt_every=10,
        pop_size=10,
        debug=True
    )

    t_train = time.perf_counter() - t_start

    t_pred_start = time.perf_counter()
    y_pred = model.predict(X_test, params, calc_cov=False, return_deriv=False)
    t_pred = time.perf_counter() - t_pred_start

    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    y_pred = np.array(y_pred).flatten()

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
        print(f"  JetGP DDEGP — Active Subspace 10D — n_train = {n_train}")
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

    output_file = os.path.join(DATA_DIR, f"results_jetgp_ddegp_{FUNCTION_NAME}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def single():
    n_train = int(sys.argv[2])
    seed = int(sys.argv[3])
    rep = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    outfile = os.path.join(DATA_DIR, f"results_jetgp_ddegp_{FUNCTION_NAME}.json")
    print(f"  JetGP DDEGP — {FUNCTION_NAME} — n_train={n_train}, seed={seed}")
    result = run_single(n_train, seed)
    result['macroreplicate'] = rep
    print(f"    NRMSE:      {result['nrmse']:.6e}")
    print(f"    Train time: {result['train_time']:.2f}s")
    if os.path.exists(outfile):
        with open(outfile) as f:
            results = json.load(f)
    else:
        results = []
    results.append(result)
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {outfile} ({len(results)} total)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--single':
        single()
    else:
        main()
