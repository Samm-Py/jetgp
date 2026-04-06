"""
JetGP DEGP Morris benchmark — resume script.
Runs n_train=20 and n_train=100 fresh (all 5 seeds),
embeds completed n_train=200 results (seeds 1000-1003),
runs only the missing n_train=200 seed=1004,
and saves everything to one JSON file.
"""

import os
import numpy as np
import time
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import (
    morris, morris_gradient,
    generate_test_data, compute_metrics
)
from scipy.stats.qmc import LatinHypercube
from jetgp.full_degp.degp import degp

FUNCTION_NAME = "morris"
DIM = 20
N_TEST = 2000
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"
DER_INDICES = [[[[i, 1]] for i in range(1, DIM + 1)]]
N_MACROREPLICATES = 5


def run_single(n_train, seed):
    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = morris(X_train)
    grads = morris_gradient(X_train)
    X_test, y_test = generate_test_data(morris, N_TEST, DIM, seed=99)

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


# Previously completed n_train=200 results (seeds 1000-1003)
COMPLETED_200 = [
    {
        "rmse": 2.245389e+00, "nrmse": 9.647981e-03,
        "train_time": 236.58, "pred_time": 21.8771,
        "n_train": 200, "seed": 1000, "macroreplicate": 1
    },
    {
        "rmse": 2.194696e+00, "nrmse": 9.430164e-03,
        "train_time": 255.42, "pred_time": 21.2560,
        "n_train": 200, "seed": 1001, "macroreplicate": 2
    },
    {
        "rmse": 1.893942e+00, "nrmse": 8.137882e-03,
        "train_time": 241.91, "pred_time": 23.0547,
        "n_train": 200, "seed": 1002, "macroreplicate": 3
    },
    {
        "rmse": 2.628547e+00, "nrmse": 1.129433e-02,
        "train_time": 271.08, "pred_time": 23.3720,
        "n_train": 200, "seed": 1003, "macroreplicate": 4
    },
]


def main():
    results = []

    # ---- Run n_train=20 and n_train=100 fresh ----
    for n_train in [20, 100]:
        print(f"\n{'='*60}")
        print(f"  JetGP DEGP -- Morris -- n_train = {n_train}")
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

    # ---- n_train=200: add completed results ----
    print(f"\n{'='*60}")
    print(f"  JetGP DEGP -- Morris -- n_train = 200")
    print(f"{'='*60}")
    for r in COMPLETED_200:
        print(f"\n  Macroreplicate {r['macroreplicate']}/5 (seed={r['seed']}) [cached]")
        print(f"    RMSE:       {r['rmse']:.6e}")
        print(f"    NRMSE:      {r['nrmse']:.6e}")
        print(f"    Train time: {r['train_time']:.2f}s")
        print(f"    Pred time:  {r['pred_time']:.4f}s")
    results.extend(COMPLETED_200)

    # ---- n_train=200: run missing seed=1004 ----
    print(f"\n  Macroreplicate 5/5 (seed=1004) [running]")
    result = run_single(200, 1004)
    result['macroreplicate'] = 5
    results.append(result)
    print(f"    RMSE:       {result['rmse']:.6e}")
    print(f"    NRMSE:      {result['nrmse']:.6e}")
    print(f"    Train time: {result['train_time']:.2f}s")
    print(f"    Pred time:  {result['pred_time']:.4f}s")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for n_train in [20, 100, 200]:
        subset = [r for r in results if r['n_train'] == n_train]
        rmses = [r['rmse'] for r in subset]
        times = [r['train_time'] for r in subset]
        print(f"\n  n = {n_train}:")
        print(f"    RMSE:  mean={np.mean(rmses):.6e}, std={np.std(rmses):.6e}")
        print(f"    Time:  mean={np.mean(times):.2f}s, std={np.std(times):.2f}s")

    # ---- Save ----
    outfile = f"results_jetgp_{FUNCTION_NAME}.json"
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
