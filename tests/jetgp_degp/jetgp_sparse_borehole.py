"""
Benchmark: JetGP Sparse DEGP on the Borehole function (8D)
First-order gradient-enhanced GP with SE anisotropic kernel.
Sparse DEGP at various rho values.  Dense DEGP results are produced
by jetgp_borehole.py and combined in a plotting script.
"""

import os
import numpy as np
import time
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import (
    borehole, borehole_gradient,
    generate_test_data, compute_metrics
)
from scipy.stats.qmc import LatinHypercube
from jetgp.full_degp_sparse.degp import degp as SparseDEGP

FUNCTION_NAME = "borehole"
DIM = 8
SAMPLE_SIZES = [DIM, 5 * DIM, 10 * DIM]
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TESTS_DIR, 'data')
N_MACROREPLICATES = 5
N_TEST = 2000
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"
DER_INDICES = [[[[i, 1]] for i in range(1, DIM + 1)]]

RHO_VALUES = [.5,1.0,1.25]
USE_SUPERNODES = False

JADE_KWARGS = dict(
    optimizer="jade",
    n_generations=10,
    local_opt_every=10,
    pop_size=10,
    debug=True,
)


def _sparsity_info(model):
    U = getattr(model, '_cached_U', None)
    if U is None:
        return None, None, None
    nnz = int(np.sum(np.abs(U) > 1e-14))
    total = U.size
    return nnz, total, nnz / total


def run_sparse(n_train, seed, rho):
    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = borehole(X_train)
    grads = borehole_gradient(X_train)
    X_test, y_test = generate_test_data(borehole, N_TEST, DIM, seed=99)

    y_train_list = [y_vals.reshape(-1, 1)]
    for j in range(DIM):
        y_train_list.append(grads[:, j].reshape(-1, 1))

    model = SparseDEGP(
        X_train, y_train_list,
        n_order=1, n_bases=DIM,
        der_indices=DER_INDICES,
        normalize=True,
        kernel=KERNEL, kernel_type=KERNEL_TYPE,
        rho=rho,
        use_supernodes=USE_SUPERNODES,
    )

    t_start = time.perf_counter()
    params = model.optimize_hyperparameters(**JADE_KWARGS)
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
    metrics['method'] = f'sparse(rho={rho})'
    metrics['rho'] = rho

    nnz, total, fill = _sparsity_info(model)
    metrics['U_nnz'] = nnz
    metrics['U_total'] = total
    metrics['U_fill'] = fill
    return metrics


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    results = []

    for n_train in SAMPLE_SIZES:
        print(f"\n{'='*70}")
        print(f"  Borehole 8D — n_train = {n_train}")
        print(f"{'='*70}")

        for rep in range(N_MACROREPLICATES):
            seed = 1000 + rep
            print(f"\n  Macroreplicate {rep + 1}/{N_MACROREPLICATES} (seed={seed})")

            for rho in RHO_VALUES:
                print(f"    Running sparse DEGP (rho={rho}) ...")
                r = run_sparse(n_train, seed, rho)
                r['macroreplicate'] = rep + 1
                results.append(r)
                fill_str = f"{r['U_fill']*100:.1f}%" if r.get('U_fill') else "?"
                print(f"    [sparse rho={rho:<4}]  NRMSE={r['nrmse']:.4e}  "
                      f"t_train={r['train_time']:.2f}s  t_pred={r['pred_time']:.4f}s  "
                      f"U_fill={fill_str}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary (mean +/- std over {N_MACROREPLICATES} replicates)")
    print(f"{'='*70}")

    methods = [f'sparse(rho={rho})' for rho in RHO_VALUES]
    for n_train in SAMPLE_SIZES:
        print(f"\n  n_train = {n_train}")
        print(f"  {'Method':<22} {'NRMSE':>14} {'Train time':>14} {'Pred time':>12}")
        print(f"  {'-'*22} {'-'*14} {'-'*14} {'-'*12}")
        for method in methods:
            subset = [r for r in results
                      if r['n_train'] == n_train and r['method'] == method]
            if not subset:
                continue
            nrmses = [r['nrmse'] for r in subset]
            times = [r['train_time'] for r in subset]
            ptimes = [r['pred_time'] for r in subset]
            print(f"  {method:<22} "
                  f"{np.mean(nrmses):.4e}+/-{np.std(nrmses):.1e}  "
                  f"{np.mean(times):>6.2f}+/-{np.std(times):>5.2f}s  "
                  f"{np.mean(ptimes):>6.4f}s")

    outfile = os.path.join(DATA_DIR, f"results_sparse_degp_{FUNCTION_NAME}.json")
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {outfile}")


def single():
    rho = float(sys.argv[2])
    n_train = int(sys.argv[3])
    seed = int(sys.argv[4])
    rep = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    os.makedirs(DATA_DIR, exist_ok=True)
    outfile = os.path.join(DATA_DIR, f"results_sparse_degp_{FUNCTION_NAME}.json")

    print(f"  Sparse DEGP (rho={rho}) — Borehole — n_train={n_train}, seed={seed}")
    result = run_sparse(n_train, seed, rho)

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
