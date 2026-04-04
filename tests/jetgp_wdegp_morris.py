"""
Benchmark: JetGP WDEGP on the Morris function (20D).

WDEGP: one submodel per training point; each submodel gets ALL function
values and the full gradient at its own point only.
"""

import numpy as np
import time
import json
import sys
sys.path.insert(0, '.')

from benchmark_functions import (
    morris, morris_gradient,
    generate_test_data, compute_metrics
)
from scipy.stats.qmc import LatinHypercube
from jetgp.wdegp.wdegp import wdegp
import jetgp.utils as utils

FUNCTION_NAME = "morris"
DIM = 20
SAMPLE_SIZES = [DIM, 5 * DIM, 10 * DIM]
N_MACROREPLICATES = 5
N_TEST = 2000
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"


def run_wdegp(X_train, y_vals, grads, X_test):
    """
    WDEGP: one submodel per training point.
    Every submodel gets ALL function values + the full gradient at its own point.
    """
    n_train = len(X_train)
    y_all_col = y_vals.reshape(-1, 1)

    # First-order derivative specs for all DIM dimensions (shared by all submodels)
    der_specs = utils.gen_OTI_indices(DIM, 1)

    submodel_data = []
    derivative_specs_list = []
    derivative_locations_list = []

    for i in range(n_train):
        # All function values + DIM partial derivatives at point i
        data_i = [y_all_col] + [grads[i:i+1, j:j+1] for j in range(DIM)]
        submodel_data.append(data_i)
        derivative_specs_list.append(der_specs)
        # Each derivative type maps to index i (this submodel's own point)
        derivative_locations_list.append([[i] for _ in range(DIM)])

    model = wdegp(
        X_train, submodel_data,
        1, DIM,
        derivative_specs_list,
        derivative_locations=derivative_locations_list,
        normalize=True,
        kernel=KERNEL, kernel_type=KERNEL_TYPE
    )
    t0 = time.perf_counter()
    params = model.optimize_hyperparameters(
        optimizer="jade",
        n_generations=10,
        local_opt_every=10,
        pop_size=10,
        debug=True
    )
    t_train = time.perf_counter() - t0

    t_pred_start = time.perf_counter()
    y_pred = model.predict(X_test, params, calc_cov=False, return_deriv=False)
    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    t_pred = time.perf_counter() - t_pred_start

    return np.array(y_pred).flatten(), t_train, t_pred


def run_single(n_train, seed):
    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = morris(X_train)
    grads  = morris_gradient(X_train)
    X_test, y_test = generate_test_data(morris, N_TEST, DIM, seed=99)

    y_pred_wdegp, t_train, t_pred = run_wdegp(X_train, y_vals, grads, X_test)
    m = compute_metrics(y_test, y_pred_wdegp)

    result = {
        'n_train': n_train,
        'seed': seed,
        'rmse': m['rmse'],
        'nrmse': m['nrmse'],
        'train_time': t_train,
        'pred_time': t_pred,
    }
    return result


def main():
    results = []
    for n_train in SAMPLE_SIZES:
        print(f"\n{'='*60}")
        print(f"  WDEGP — Morris 20D — n_train={n_train}")
        print(f"{'='*60}")
        for rep in range(N_MACROREPLICATES):
            seed = 1000 + rep
            print(f"\n  Macroreplicate {rep + 1}/{N_MACROREPLICATES} (seed={seed})")
            result = run_single(n_train, seed)
            result['macroreplicate'] = rep + 1
            results.append(result)
            print(f"    NRMSE:      {result['nrmse']:.6e}")
            print(f"    Train time: {result['train_time']:.2f}s")
            print(f"    Pred time:  {result['pred_time']:.4f}s")

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for n_train in SAMPLE_SIZES:
        subset = [r for r in results if r['n_train'] == n_train]
        nrmses = [r['nrmse'] for r in subset]
        times  = [r['train_time'] for r in subset]
        print(f"\n  n={n_train}  WDEGP:")
        print(f"    NRMSE: mean={np.mean(nrmses):.6e}, std={np.std(nrmses):.6e}")
        print(f"    Time:  mean={np.mean(times):.2f}s, std={np.std(times):.2f}s")

    output_file = f"results_wdegp_{FUNCTION_NAME}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
