"""
Sweep over submodel count (n_sub) for WDEGP on the Morris function (20D).

For each training size N and each valid divisor n_sub of N, runs N_MACROREPLICATES
experiments and records NRMSE and training time. Configurations where the submodel
K matrix would exceed MAX_K_SIZE are skipped to avoid excessive runtimes.

The theoretical cost-optimal n_sub minimises n_sub * (N + D * N/n_sub)^3.
This script asks whether the NRMSE-optimal n_sub coincides with the cost-optimal one,
or whether larger submodels (smaller n_sub, more pts_per_sub) give meaningfully
better accuracy.

Usage:
    python jetgp_wdegp_nsub_sweep_morris.py            # full sweep
    python jetgp_wdegp_nsub_sweep_morris.py --single <n_train> <n_sub> <seed> [rep]
"""

import os
import numpy as np
import time
import json
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark_functions import (
    morris, morris_gradient,
)
from scipy.stats.qmc import LatinHypercube
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from jetgp.wdegp.wdegp import wdegp
import jetgp.utils as utils

# ── Config ────────────────────────────────────────────────────────────────────
FUNCTION_NAME    = "morris"
DIM              = 20
SAMPLE_SIZES     = [DIM, 5 * DIM, 10 * DIM]   # 20, 100, 200
N_MACROREPLICATES = 3
KERNEL           = "SE"
KERNEL_TYPE      = "anisotropic"

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR  = os.path.join(TESTS_DIR, 'data')


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_divisors(n):
    return sorted([d for d in range(1, n + 1) if n % d == 0])


def theoretical_optimal_n_sub(n_train, D):
    """n_sub that minimises n_sub * (n_train + D * pts_per_sub)^3."""
    best_cost, best_n = float('inf'), n_train
    for n in range(1, n_train + 1):
        if n_train % n != 0:
            continue
        m = n_train + D * (n_train // n)
        cost = n * m ** 3
        if cost < best_cost:
            best_cost, best_n = cost, n
    return best_n


def balanced_kmeans_groups(X, n_sub, pts_per_sub, seed=42):
    """Assign n_train points to n_sub equal-sized spatially compact groups."""
    kmeans = KMeans(n_clusters=n_sub, random_state=seed, n_init=10)
    kmeans.fit(X)
    centers = kmeans.cluster_centers_
    dists = cdist(X, centers)
    cluster_ranks = np.argsort(dists, axis=1)

    groups    = [[] for _ in range(n_sub)]
    min_dists = dists.min(axis=1)
    point_order = np.argsort(min_dists)

    for pt in point_order:
        for c in cluster_ranks[pt]:
            if len(groups[c]) < pts_per_sub:
                groups[c].append(pt)
                break
    return groups


# ── Core run ──────────────────────────────────────────────────────────────────
def run_single(n_train, n_sub, seed):
    pts_per_sub = n_train // n_sub

    sampler = LatinHypercube(d=DIM, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals  = morris(X_train)
    grads   = morris_gradient(X_train)           # (n_train, DIM)

    groups = balanced_kmeans_groups(X_train, n_sub, pts_per_sub, seed=seed)

    y_all_col   = y_vals.reshape(-1, 1)
    der_specs   = utils.gen_OTI_indices(DIM, 1)

    submodel_data             = []
    derivative_specs_list     = []
    derivative_locations_list = []

    for s in range(n_sub):
        idx    = sorted(groups[s])
        data_s = [y_all_col]
        for j in range(DIM):
            data_s.append(grads[idx, j:j + 1])
        submodel_data.append(data_s)
        derivative_specs_list.append(der_specs)
        derivative_locations_list.append([idx for _ in range(DIM)])

    model = wdegp(
        X_train, submodel_data,
        1, DIM,
        derivative_specs_list,
        derivative_locations=derivative_locations_list,
        normalize=True,
        kernel=KERNEL, kernel_type=KERNEL_TYPE,
    )

    t0 = time.perf_counter()
    model.optimize_hyperparameters(
        optimizer="jade",
        n_generations=1,
        local_opt_every=None,
        pop_size=20,
        debug=True,
    )
    t_train = time.perf_counter() - t0

    return {
        'n_train':     n_train,
        'n_sub':       n_sub,
        'pts_per_sub': pts_per_sub,
        'k_size':      n_train + DIM * pts_per_sub,
        'seed':        seed,
        'train_time':  t_train,
    }


# ── Main sweep ────────────────────────────────────────────────────────────────
def main():
    output_file = os.path.join(DATA_DIR, f"results_wdegp_nsub_sweep_{FUNCTION_NAME}.json")

    # Load existing results so the sweep can be resumed
    if os.path.exists(output_file):
        with open(output_file) as f:
            all_results = json.load(f)
    else:
        all_results = []

    already_done = {(r['n_train'], r['n_sub'], r['seed']) for r in all_results}

    for n_train in SAMPLE_SIZES:
        divisors    = get_divisors(n_train)
        opt_n_sub   = theoretical_optimal_n_sub(n_train, DIM)

        print(f"\n{'='*65}")
        print(f"  n_train={n_train}  |  theoretical-optimal n_sub={opt_n_sub} "
              f"(pts/sub={n_train//opt_n_sub})")
        print(f"{'='*65}")

        for n_sub in divisors:
            pts_per_sub = n_train // n_sub
            k_size      = n_train + DIM * pts_per_sub

            if n_sub == 1:
                print(f"  n_sub=   1  pts/sub={pts_per_sub:4d}  "
                      f"K={k_size:5d}  → SKIPPED (single submodel)")
                continue

            print(f"\n  n_sub={n_sub:4d}  pts/sub={pts_per_sub:4d}  K={k_size:5d}"
                  f"{'  [COST-OPTIMAL]' if n_sub == opt_n_sub else ''}")

            for rep in range(N_MACROREPLICATES):
                seed = 1000 + rep
                if (n_train, n_sub, seed) in already_done:
                    print(f"    rep {rep+1}: already done, skipping")
                    continue

                print(f"    rep {rep+1}/{N_MACROREPLICATES} (seed={seed})")
                result = run_single(n_train, n_sub, seed)
                result['macroreplicate'] = rep + 1
                all_results.append(result)
                already_done.add((n_train, n_sub, seed))

                print(f"      Train time: {result['train_time']:.4f}s")

                # Save after every rep so a crash doesn't lose data
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  SUMMARY  (mean ± std over macroreplicates)")
    print(f"{'='*65}")
    import collections
    grouped = collections.defaultdict(list)
    for r in all_results:
        grouped[(r['n_train'], r['n_sub'])].append(r)

    for n_train in SAMPLE_SIZES:
        opt_n_sub = theoretical_optimal_n_sub(n_train, DIM)
        print(f"\n  n_train={n_train}:")
        for n_sub in get_divisors(n_train):
            recs = grouped[(n_train, n_sub)]
            if not recs:
                continue
            times  = [r['train_time'] for r in recs]
            marker = '  ← cost-optimal' if n_sub == opt_n_sub else ''
            print(f"    n_sub={n_sub:4d}  pts/sub={n_train//n_sub:4d}  "
                  f"time={np.mean(times):.4f}s{marker}")

    print(f"\nResults saved to {output_file}")


# ── Single-run entry point ────────────────────────────────────────────────────
def single():
    n_train = int(sys.argv[2])
    n_sub   = int(sys.argv[3])
    seed    = int(sys.argv[4])
    rep     = int(sys.argv[5]) if len(sys.argv) > 5 else 1

    outfile = os.path.join(DATA_DIR, f"results_wdegp_nsub_sweep_{FUNCTION_NAME}.json")

    pts_per_sub = n_train // n_sub
    k_size      = n_train + DIM * pts_per_sub
    print(f"  WDEGP n_sub sweep — Morris — n_train={n_train}, n_sub={n_sub}, "
          f"pts/sub={pts_per_sub}, K={k_size}, seed={seed}")

    result = run_single(n_train, n_sub, seed)
    result['macroreplicate'] = rep
    print(f"    Train time: {result['train_time']:.4f}s")

    if os.path.exists(outfile):
        with open(outfile) as f:
            results = json.load(f)
    else:
        results = []
    results.append(result)
    with open(outfile, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Saved to {outfile} ({len(results)} total records)")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--single':
        single()
    else:
        main()
