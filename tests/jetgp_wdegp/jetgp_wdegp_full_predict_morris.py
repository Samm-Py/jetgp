"""
Experiment: train WDEGP on Morris 20D, then use its optimized hyperparameters
to do a FULL DEGP prediction (no submatrix approximation at inference time).

Compares NRMSE of:
  - WDEGP predict (weighted sum of submodels)
  - full DEGP predict using WDEGP-optimized hyperparameters
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
from jetgp.wdegp.wdegp import wdegp
import jetgp.utils as utils

FUNCTION_NAME = "morris"
DIM = 20
SAMPLE_SIZES = [DIM, 5 * DIM, 10 * DIM]
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TESTS_DIR, 'data')
N_MACROREPLICATES = 5
N_TEST = 2000
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"


def build_wdegp(X_train, y_vals, grads):
    """One submodel per point: all function values + gradient at own point."""
    n_train = len(X_train)
    y_all_col = y_vals.reshape(-1, 1)
    der_specs = utils.gen_OTI_indices(DIM, 1)

    submodel_data = []
    derivative_specs_list = []
    derivative_locations_list = []
    for i in range(n_train):
        data_i = [y_all_col] + [grads[i:i+1, j:j+1] for j in range(DIM)]
        submodel_data.append(data_i)
        derivative_specs_list.append(der_specs)
        derivative_locations_list.append([[i] for _ in range(DIM)])

    return wdegp(
        X_train, submodel_data,
        1, DIM,
        derivative_specs_list,
        derivative_locations=derivative_locations_list,
        normalize=True,
        kernel=KERNEL, kernel_type=KERNEL_TYPE,
    )


def run_single(n_train, seed):
    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = morris(X_train)
    grads  = morris_gradient(X_train)
    X_test, y_test = generate_test_data(morris, N_TEST, DIM, seed=99)

    # --- Train WDEGP ---
    wmodel = build_wdegp(X_train, y_vals, grads)
    t0 = time.perf_counter()
    params = wmodel.optimize_hyperparameters(
        optimizer="jade",
        n_generations=10,
        local_opt_every=10,
        pop_size=10,
        debug=False,
    )
    t_train = time.perf_counter() - t0

    # --- WDEGP prediction (baseline) ---
    t0 = time.perf_counter()
    y_pred_w = wmodel.predict(X_test, params, calc_cov=False, return_deriv=False)
    if isinstance(y_pred_w, tuple):
        y_pred_w = y_pred_w[0]
    y_pred_w = np.asarray(y_pred_w).flatten()
    t_pred_w = time.perf_counter() - t0

    # --- Full prediction via wdegp.predict(mode='full') ---
    t0 = time.perf_counter()
    y_pred_f = wmodel.predict(
        X_test, params, calc_cov=False, return_deriv=False, mode="full"
    )
    if isinstance(y_pred_f, tuple):
        y_pred_f = y_pred_f[0]
    y_pred_f = np.asarray(y_pred_f)
    if y_pred_f.ndim == 2:
        y_pred_f = y_pred_f[0, :]
    y_pred_f = y_pred_f.flatten()
    t_pred_f = time.perf_counter() - t0

    m_w = compute_metrics(y_test, y_pred_w)
    m_f = compute_metrics(y_test, y_pred_f)

    return {
        'n_train': n_train,
        'seed': seed,
        'train_time': t_train,
        'wdegp_nrmse': m_w['nrmse'],
        'wdegp_rmse':  m_w['rmse'],
        'wdegp_pred_time': t_pred_w,
        'full_nrmse':  m_f['nrmse'],
        'full_rmse':   m_f['rmse'],
        'full_pred_time': t_pred_f,
        'nrmse_ratio_full_over_wdegp': m_f['nrmse'] / m_w['nrmse'],
    }


def main():
    results = []
    for n_train in SAMPLE_SIZES:
        print(f"\n{'='*66}")
        print(f"  WDEGP train -> {{WDEGP, full-DEGP}} predict — Morris 20D — n={n_train}")
        print(f"{'='*66}")
        for rep in range(N_MACROREPLICATES):
            seed = 1000 + rep
            print(f"\n  Macroreplicate {rep + 1}/{N_MACROREPLICATES} (seed={seed})")
            r = run_single(n_train, seed)
            r['macroreplicate'] = rep + 1
            results.append(r)
            print(f"    WDEGP NRMSE: {r['wdegp_nrmse']:.6e}")
            print(f"    Full  NRMSE: {r['full_nrmse']:.6e}")
            print(f"    ratio (full/wdegp): {r['nrmse_ratio_full_over_wdegp']:.4f}")
            print(f"    Train time: {r['train_time']:.2f}s")

    print(f"\n{'='*66}")
    print(f"  Summary")
    print(f"{'='*66}")
    for n_train in SAMPLE_SIZES:
        subset = [r for r in results if r['n_train'] == n_train]
        w = [r['wdegp_nrmse'] for r in subset]
        f = [r['full_nrmse']  for r in subset]
        print(f"\n  n={n_train}:")
        print(f"    WDEGP NRMSE: mean={np.mean(w):.6e}, std={np.std(w):.6e}")
        print(f"    Full  NRMSE: mean={np.mean(f):.6e}, std={np.std(f):.6e}")

    outfile = os.path.join(DATA_DIR, f"results_wdegp_full_predict_{FUNCTION_NAME}.json")
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
