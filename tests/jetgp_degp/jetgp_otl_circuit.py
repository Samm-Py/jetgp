"""
Benchmark: JetGP DEGP on the OTL Circuit function (6D)
First-order gradient-enhanced GP with SE anisotropic kernel.
PSO optimizer with local refinement (exploration-exploitation strategy).
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
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import (
    otl_circuit, otl_circuit_gradient,
    generate_test_data, compute_metrics
)
from scipy.stats.qmc import LatinHypercube
from jetgp.full_degp.degp import degp

FUNCTION_NAME = "otl_circuit"
DIM = 6
SAMPLE_SIZES = [DIM, 5 * DIM, 10 * DIM]
N_MACROREPLICATES = 5
N_TEST = 2000
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"
DER_INDICES = [[[[i, 1]] for i in range(1, DIM + 1)]]
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TESTS_DIR, 'data')


def run_single(n_train, seed):
    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = otl_circuit(X_train)
    grads = otl_circuit_gradient(X_train)
    X_test, y_test = generate_test_data(otl_circuit, N_TEST, DIM, seed=99)

    y_train_list = [y_vals.reshape(-1, 1)]
    for j in range(DIM):
        y_train_list.append(grads[:, j].reshape(-1, 1))

    model = degp(
        X_train, y_train_list,
        n_order=1, n_bases=DIM,
        der_indices=DER_INDICES,
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

    if y_pred.ndim == 2:
        y_pred = y_pred[0, :]
    y_pred = y_pred.flatten()

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
        print(f"  JetGP DEGP — OTL Circuit — n_train = {n_train}")
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

    outfile = os.path.join(DATA_DIR, f"results_jetgp_{FUNCTION_NAME}.json")
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")


def single():
    n_train = int(sys.argv[2])
    seed = int(sys.argv[3])
    rep = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    outfile = os.path.join(DATA_DIR, f"results_jetgp_{FUNCTION_NAME}.json")
    print(f"  JetGP DEGP — {FUNCTION_NAME} — n_train={n_train}, seed={seed}")
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
