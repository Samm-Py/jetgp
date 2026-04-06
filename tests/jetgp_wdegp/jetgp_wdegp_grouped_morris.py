"""
Benchmark: JetGP WDEGP (grouped) on the Morris function (20D).

WDEGP with optimal submodel grouping: each submodel gets ALL function
values and gradients at (n_train/N_SUB) points, where N_SUB minimizes
N_SUB * (n_train + D * n_train/N_SUB)^3.
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
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
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


def optimal_n_sub(n_train, D):
    """Find N_SUB that minimizes N_SUB * (n_train + D * n_train/N_SUB)^3."""
    best_cost = float('inf')
    best_n = n_train
    for n in range(1, n_train + 1):
        if n_train % n != 0:
            continue
        m = n_train + D * (n_train // n)
        cost = n * m ** 3
        if cost < best_cost:
            best_cost = cost
            best_n = n
    return best_n


def balanced_kmeans_groups(X, n_sub, pts_per_sub, seed=42):
    """
    Assign points to n_sub equal-sized groups using k-means centers.
    Greedily assigns each point to its nearest cluster that isn't full.
    """
    kmeans = KMeans(n_clusters=n_sub, random_state=seed, n_init=10)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_

    # Distance from each point to each center
    dists = cdist(X, centers)
    # Sort points by their distance to their nearest center (greedy assignment)
    # For each point, rank the clusters by distance
    cluster_ranks = np.argsort(dists, axis=1)

    groups = [[] for _ in range(n_sub)]
    assigned = np.full(len(X), -1, dtype=int)

    # Sort points by how close they are to their nearest center (assign closest first)
    min_dists = dists.min(axis=1)
    point_order = np.argsort(min_dists)

    for pt in point_order:
        for c in cluster_ranks[pt]:
            if len(groups[c]) < pts_per_sub:
                groups[c].append(pt)
                assigned[pt] = c
                break

    return groups


def run_wdegp_grouped(X_train, y_vals, grads, X_test, seed=42):
    """
    WDEGP with optimal grouping using spatial clustering.
    """
    n_train = len(X_train)
    n_sub = optimal_n_sub(n_train, DIM)
    pts_per_sub = n_train // n_sub

    groups = balanced_kmeans_groups(X_train, n_sub, pts_per_sub, seed=seed)

    y_all_col = y_vals.reshape(-1, 1)
    der_specs = utils.gen_OTI_indices(DIM, 1)

    submodel_data = []
    derivative_specs_list = []
    derivative_locations_list = []

    for s in range(n_sub):
        idx = sorted(groups[s])
        data_s = [y_all_col]
        for j in range(DIM):
            data_s.append(grads[idx, j:j+1])
        submodel_data.append(data_s)
        derivative_specs_list.append(der_specs)
        derivative_locations_list.append([idx for _ in range(DIM)])

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

    return np.array(y_pred).flatten(), t_train, t_pred, n_sub


def run_single(n_train, seed):
    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = morris(X_train)
    grads  = morris_gradient(X_train)
    X_test, y_test = generate_test_data(morris, N_TEST, DIM, seed=99)

    y_pred, t_train, t_pred, n_sub = run_wdegp_grouped(X_train, y_vals, grads, X_test, seed=seed)
    m = compute_metrics(y_test, y_pred)

    result = {
        'n_train': n_train,
        'seed': seed,
        'rmse': m['rmse'],
        'nrmse': m['nrmse'],
        'train_time': t_train,
        'pred_time': t_pred,
        'n_sub': n_sub,
    }
    return result


def main():
    results = []
    for n_train in SAMPLE_SIZES:
        n_sub = optimal_n_sub(n_train, DIM)
        pts_per_sub = n_train // n_sub
        print(f"\n{'='*60}")
        print(f"  WDEGP Grouped — Morris 20D — n_train={n_train}")
        print(f"  N_SUB={n_sub}, pts_per_sub={pts_per_sub}")
        print(f"  Submodel K size: {n_train + DIM * pts_per_sub}")
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
            print(f"    N_SUB:      {result['n_sub']}")

    print(f"\n{'='*60}")
    print(f"  Summary")
    print(f"{'='*60}")
    for n_train in SAMPLE_SIZES:
        subset = [r for r in results if r['n_train'] == n_train]
        nrmses = [r['nrmse'] for r in subset]
        times  = [r['train_time'] for r in subset]
        print(f"\n  n={n_train}  WDEGP Grouped (N_SUB={subset[0]['n_sub']}):")
        print(f"    NRMSE: mean={np.mean(nrmses):.6e}, std={np.std(nrmses):.6e}")
        print(f"    Time:  mean={np.mean(times):.2f}s, std={np.std(times):.2f}s")

    output_file = os.path.join(DATA_DIR, f"results_wdegp_grouped_{FUNCTION_NAME}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_file}")


def single():
    """Run a single (n_train, seed) and append result to the output JSON."""
    n_train = int(sys.argv[2])
    seed = int(sys.argv[3])
    rep = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    outfile = os.path.join(DATA_DIR, f"results_wdegp_grouped_{FUNCTION_NAME}.json")

    print(f"  WDEGP Grouped — Morris — n_train={n_train}, seed={seed}")
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
