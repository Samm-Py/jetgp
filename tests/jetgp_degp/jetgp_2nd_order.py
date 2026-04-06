"""
Benchmark: JetGP DEGP 2nd-order on Borehole (8D), OTL Circuit (6D), Morris (20D).

Uses n_order=2 with first derivatives and diagonal second derivatives.
SE anisotropic kernel, JADE optimizer.
Sample sizes: DIM, 3*DIM, 5*DIM per function.
"""

import os
import numpy as np
import time
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import (
    borehole, borehole_gradient, borehole_hessian_diag,
    otl_circuit, otl_circuit_gradient, otl_circuit_hessian_diag,
    morris, morris_gradient, morris_hessian_diag,
    generate_test_data, compute_metrics
)
from scipy.stats.qmc import LatinHypercube
from jetgp.full_degp.degp import degp

N_MACROREPLICATES = 5
N_TEST = 2000
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"
TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(TESTS_DIR, 'data')

BENCHMARKS = {
    'borehole': {
        'func': borehole,
        'grad_func': borehole_gradient,
        'hess_func': borehole_hessian_diag,
        'dim': 8,
    },
    'otl_circuit': {
        'func': otl_circuit,
        'grad_func': otl_circuit_gradient,
        'hess_func': otl_circuit_hessian_diag,
        'dim': 6,
    },
    'morris': {
        'func': morris,
        'grad_func': morris_gradient,
        'hess_func': morris_hessian_diag,
        'dim': 20,
    },
}


def make_der_indices(dim):
    """
    der_indices for 1st + diagonal 2nd order derivatives.
    Group 1: [[1,1]], [[2,1]], ..., [[D,1]]   (first derivatives)
    Group 2: [[1,2]], [[2,2]], ..., [[D,2]]   (diagonal second derivatives)
    """
    first_order = [[[i, 1]] for i in range(1, dim + 1)]
    second_order = [[[i, 2]] for i in range(1, dim + 1)]
    return [first_order, second_order]


def run_single(func_name, n_train, seed):
    cfg = BENCHMARKS[func_name]
    dim = cfg['dim']
    func = cfg['func']
    grad_func = cfg['grad_func']
    hess_func = cfg['hess_func']

    sampler = LatinHypercube(d=dim, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = func(X_train)
    grads = grad_func(X_train)
    hess_diag = hess_func(X_train)
    X_test, y_test = generate_test_data(func, N_TEST, dim, seed=99)

    # Training data: [f, df/dx1,...,df/dxD, d2f/dx1^2,...,d2f/dxD^2]
    y_train_list = [y_vals.reshape(-1, 1)]
    for j in range(dim):
        y_train_list.append(grads[:, j].reshape(-1, 1))
    for j in range(dim):
        y_train_list.append(hess_diag[:, j].reshape(-1, 1))

    der_indices = make_der_indices(dim)

    model = degp(
        X_train, y_train_list,
        n_order=2, n_bases=dim,
        der_indices=der_indices,
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
    metrics['function'] = func_name
    metrics['dim'] = dim
    return metrics


def main():
    for func_name, cfg in BENCHMARKS.items():
        dim = cfg['dim']
        sample_sizes = [dim, 3 * dim, 5 * dim]
        results = []

        for n_train in sample_sizes:
            print(f"\n{'='*60}")
            print(f"  JetGP DEGP 2nd Order — {func_name} {dim}D — n_train={n_train}")
            print(f"  K matrix size: {n_train * (1 + 2*dim)}")
            print(f"{'='*60}")
            for rep in range(N_MACROREPLICATES):
                seed = 1000 + rep
                print(f"\n  Macroreplicate {rep + 1}/{N_MACROREPLICATES} (seed={seed})")
                result = run_single(func_name, n_train, seed)
                result['macroreplicate'] = rep + 1
                results.append(result)
                print(f"    RMSE:       {result['rmse']:.6e}")
                print(f"    NRMSE:      {result['nrmse']:.6e}")
                print(f"    Train time: {result['train_time']:.2f}s")
                print(f"    Pred time:  {result['pred_time']:.4f}s")

        print(f"\n{'='*60}")
        print(f"  Summary — {func_name}")
        print(f"{'='*60}")
        for n_train in sample_sizes:
            subset = [r for r in results if r['n_train'] == n_train]
            nrmses = [r['nrmse'] for r in subset]
            times = [r['train_time'] for r in subset]
            print(f"\n  n={n_train}:")
            print(f"    NRMSE: mean={np.mean(nrmses):.6e}, std={np.std(nrmses):.6e}")
            print(f"    Time:  mean={np.mean(times):.2f}s, std={np.std(times):.2f}s")

        outfile = os.path.join(DATA_DIR, f"results_jetgp_2nd_order_{func_name}.json")
        with open(outfile, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {outfile}")


def single():
    """Run a single (func_name, n_train, seed) and append to output JSON."""
    func_name = sys.argv[2]
    n_train = int(sys.argv[3])
    seed = int(sys.argv[4])
    rep = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    outfile = os.path.join(DATA_DIR, f"results_jetgp_2nd_order_{func_name}.json")

    print(f"  JetGP DEGP 2nd Order — {func_name} — n_train={n_train}, seed={seed}")
    result = run_single(func_name, n_train, seed)
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
