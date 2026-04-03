"""
Benchmark: JetGP GDDEGP on the Morris function (20D)
Point-wise directional derivatives with 1, 2, and 3 mutually orthogonal
random directions per training point. SE anisotropic kernel.
JADE optimizer with local L-BFGS-B refinement.
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
from scipy.stats import ortho_group
from jetgp.full_gddegp.gddegp import gddegp

FUNCTION_NAME = "morris"
DIM = 20
SAMPLE_SIZES = [DIM, 5 * DIM, 10 * DIM]
N_MACROREPLICATES = 5
N_TEST = 2000
KERNEL = "SE"
KERNEL_TYPE = "anisotropic"
N_DIRS_LIST = [1, 2, 3]


def generate_orthonormal_directions(n_train, dim, n_dirs, seed):
    """
    For each training point draw n_dirs mutually orthonormal directions in R^dim.

    Returns
    -------
    rays_list : list of n_dirs arrays, each shape (dim, n_train)
        rays_list[k][:, i] is the k-th direction for training point i.
    """
    rng = np.random.RandomState(seed)
    # One (dim x dim) orthogonal matrix per training point; take first n_dirs cols
    rays = np.zeros((n_dirs, dim, n_train))
    for i in range(n_train):
        Q = ortho_group.rvs(dim, random_state=rng)   # (dim, dim)
        rays[:, :, i] = Q[:n_dirs, :]                 # (n_dirs, dim)
    # rays_list[k] has shape (dim, n_train)
    return [rays[k] for k in range(n_dirs)]


def run_single(n_train, seed, n_dirs):
    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = morris(X_train)
    grads = morris_gradient(X_train)          # (n_train, DIM)

    X_test, y_test = generate_test_data(morris, N_TEST, DIM, seed=99)

    # Random orthonormal directions — seed offset by n_dirs for independence
    rays_list = generate_orthonormal_directions(n_train, DIM, n_dirs, seed=seed + n_dirs)

    # Directional derivatives: ∇f(x_i) · ray_k[:, i]
    y_train_list = [y_vals.reshape(-1, 1)]
    for k in range(n_dirs):
        # rays_list[k] shape (DIM, n_train); grads shape (n_train, DIM)
        d_deriv = np.einsum('di,id->i', rays_list[k], grads)   # (n_train,)
        y_train_list.append(d_deriv.reshape(-1, 1))

    # der_indices: one group per direction type, each with its own OTI basis index
    der_indices = [[[[k + 1, 1]]] for k in range(n_dirs)]

    model = gddegp(
        X_train, y_train_list,
        n_order=1,
        rays_list=rays_list,
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

    if isinstance(y_pred, tuple):
        y_pred = y_pred[0]
    y_pred = np.array(y_pred).flatten()

    metrics = compute_metrics(y_test, y_pred)
    metrics['train_time'] = t_train
    metrics['pred_time'] = t_pred
    metrics['n_train'] = n_train
    metrics['seed'] = seed
    metrics['n_dirs'] = n_dirs
    return metrics


def main():
    all_results = {k: [] for k in N_DIRS_LIST}

    for n_dirs in N_DIRS_LIST:
        results = []
        for n_train in SAMPLE_SIZES:
            print(f"\n{'='*60}")
            print(f"  JetGP GDDEGP ({n_dirs} dir/pt) — Morris — n_train = {n_train}")
            print(f"{'='*60}")
            for rep in range(N_MACROREPLICATES):
                seed = 1000 + rep
                print(f"\n  Macroreplicate {rep + 1}/{N_MACROREPLICATES} (seed={seed})")
                result = run_single(n_train, seed, n_dirs)
                result['macroreplicate'] = rep + 1
                results.append(result)
                print(f"    RMSE:       {result['rmse']:.6e}")
                print(f"    NRMSE:      {result['nrmse']:.6e}")
                print(f"    Train time: {result['train_time']:.2f}s")
                print(f"    Pred time:  {result['pred_time']:.4f}s")

        print(f"\n{'='*60}")
        print(f"  Summary — GDDEGP {n_dirs} dir/pt")
        print(f"{'='*60}")
        for n_train in SAMPLE_SIZES:
            subset = [r for r in results if r['n_train'] == n_train]
            rmses = [r['rmse'] for r in subset]
            times = [r['train_time'] for r in subset]
            print(f"\n  n = {n_train}:")
            print(f"    RMSE:  mean={np.mean(rmses):.6e}, std={np.std(rmses):.6e}")
            print(f"    Time:  mean={np.mean(times):.2f}s, std={np.std(times):.2f}s")

        output_file = f"results_jetgp_gddegp_{n_dirs}dirs_{FUNCTION_NAME}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_file}")
        all_results[n_dirs] = results


if __name__ == "__main__":
    main()
