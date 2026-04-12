"""
Benchmark: sliced SDEGP (option C — shared ff block, slice-restricted
gradients, signed weights from eq. 17) vs standard WDEGP (one-submodel-
per-point) on Morris 20D.

Reports NRMSE for:
  - WDEGP  (weighted sum of per-point submodels)
  - SDEGP  (sliced, option C)
"""
import os
import sys
import time
import json
import numpy as np
from scipy.stats.qmc import LatinHypercube

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from benchmark_functions import morris, morris_gradient, generate_test_data, compute_metrics

import jetgp.utils as utils
from jetgp.wdegp.wdegp import wdegp
from jetgp.sdegp.sdegp import sdegp

FUNCTION_NAME = "morris"
DIM = 20
SAMPLE_SIZES = [5 * DIM]
N_MACROREPLICATES = 1
N_TEST = 2000
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"
# Number of slices for the sliced likelihood. With m slices we get
# (m-1) pair blocks + (m-2) single-slice correction blocks = 2m - 3 submodels.
M_SLICES = 5

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TESTS_DIR, 'data')


def build_wdegp_point_per_submodel(X, y, grads):
    n = len(X)
    y_col = y.reshape(-1, 1)
    der_specs = utils.gen_OTI_indices(DIM, 1)
    submodel_data, der_spec_list, der_loc_list = [], [], []
    for i in range(n):
        d = [y_col] + [grads[i:i + 1, j:j + 1] for j in range(DIM)]
        submodel_data.append(d)
        der_spec_list.append(der_specs)
        der_loc_list.append([[i] for _ in range(DIM)])
    return wdegp(
        X, submodel_data, 1, DIM, der_spec_list,
        derivative_locations=der_loc_list,
        normalize=True, kernel=KERNEL, kernel_type=KERNEL_TYPE,
    )


def build_sdegp_sliced(X, y, grads, m):
    model = sdegp(
        X, y, grads,
        n_order=1, m=m,
        kernel=KERNEL, kernel_type=KERNEL_TYPE,
    )
    return model, model.slice_dim, [len(s) for s in model.slices]


def run_single(n_train, seed):
    sampler = LatinHypercube(d=DIM, seed=seed)
    X = sampler.random(n=n_train)
    y = morris(X)
    g = morris_gradient(X)
    X_test, y_test = generate_test_data(morris, N_TEST, DIM, seed=99)

    # --- WDEGP baseline ---
    wmodel = build_wdegp_point_per_submodel(X, y, g)
    t0 = time.perf_counter()
    w_params = wmodel.optimize_hyperparameters(
        optimizer="jade", n_generations=10, local_opt_every=10,
        pop_size=10, debug=True,
    )
    t_w_train = time.perf_counter() - t0
    y_pred_w = wmodel.predict(X_test, w_params, calc_cov=False, return_deriv=False, mode = 'full')
    if isinstance(y_pred_w, tuple):
        y_pred_w = y_pred_w[0]
    y_pred_w = np.asarray(y_pred_w).flatten()
    m_w = compute_metrics(y_test, y_pred_w)

    # --- Sliced SDEGP (option C) ---
    smodel, slice_dim, slice_sizes = build_sdegp_sliced(X, y, g, M_SLICES)
    t0 = time.perf_counter()
    s_params = smodel.optimize_hyperparameters(
        optimizer="jade", n_generations=10, local_opt_every=10,
        pop_size=10, debug=True,
    )
    t_s_train = time.perf_counter() - t0
    y_pred_s = smodel.predict(X_test, s_params, calc_cov=False, return_deriv=False)
    if isinstance(y_pred_s, tuple):
        y_pred_s = y_pred_s[0]
    y_pred_s = np.asarray(y_pred_s).flatten()
    m_s = compute_metrics(y_test, y_pred_s)

    return {
        'n_train': n_train, 'seed': seed, 'm_slices': M_SLICES,
        'slice_dim': slice_dim, 'slice_sizes': slice_sizes,
        'wdegp_nrmse': m_w['nrmse'], 'wdegp_rmse': m_w['rmse'],
        'wdegp_train_time': t_w_train,
        'sdegp_nrmse': m_s['nrmse'], 'sdegp_rmse': m_s['rmse'],
        'sdegp_train_time': t_s_train,
        'nrmse_ratio_sdegp_over_wdegp': m_s['nrmse'] / m_w['nrmse'],
    }


def main():
    results = []
    for n_train in SAMPLE_SIZES:
        print(f"\n{'=' * 66}")
        print(f"  WDEGP vs sliced SDEGP (m={M_SLICES}) — Morris 20D — n={n_train}")
        print(f"{'=' * 66}")
        for rep in range(N_MACROREPLICATES):
            seed = 1000 + rep
            print(f"\n  Macroreplicate {rep + 1}/{N_MACROREPLICATES} (seed={seed})")
            r = run_single(n_train, seed)
            r['macroreplicate'] = rep + 1
            results.append(r)
            print(f"    WDEGP NRMSE: {r['wdegp_nrmse']:.6e}   (train {r['wdegp_train_time']:.1f}s)")
            print(f"    SDEGP NRMSE: {r['sdegp_nrmse']:.6e}   (train {r['sdegp_train_time']:.1f}s)")
            print(f"    slice_dim={r['slice_dim']}  sizes={r['slice_sizes']}")
            print(f"    ratio (sdegp/wdegp): {r['nrmse_ratio_sdegp_over_wdegp']:.4f}")

    print(f"\n{'=' * 66}")
    print(f"  Summary")
    print(f"{'=' * 66}")
    for n_train in SAMPLE_SIZES:
        subset = [r for r in results if r['n_train'] == n_train]
        w = [r['wdegp_nrmse'] for r in subset]
        s = [r['sdegp_nrmse'] for r in subset]
        print(f"\n  n={n_train}:")
        print(f"    WDEGP NRMSE: mean={np.mean(w):.6e}, std={np.std(w):.6e}")
        print(f"    SDEGP NRMSE: mean={np.mean(s):.6e}, std={np.std(s):.6e}")

    outfile = os.path.join(DATA_DIR, f"results_sdegp_sliced_{FUNCTION_NAME}.json")
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
