"""
Profile baseline for borehole (m8n2), OTL circuit (m6n2), and active subspace (m10n2).
Saves cProfile stats to .prof files and a summary JSON for later comparison.

Usage:
    python profile_baseline.py [--tag baseline]
"""
import argparse
import cProfile
import pstats
import io
import json
import sys
import time
sys.path.insert(0, '.')

import numpy as np
from benchmark_functions import (
    borehole, borehole_gradient,
    otl_circuit, otl_circuit_gradient,
    active_subspace_10d, active_subspace_10d_gradient,
    generate_test_data,
)
from scipy.stats.qmc import LatinHypercube
from jetgp.full_degp.degp import degp

parser = argparse.ArgumentParser()
parser.add_argument('--tag', type=str, default='baseline',
                    help='Tag for output filenames (e.g. baseline, prealloc)')
args = parser.parse_args()

SEED = 1000
JADE_KWARGS = dict(optimizer="jade", n_generations=10, local_opt_every=10,
                   pop_size=20, debug=False)

BENCHMARKS = {
    'borehole_m8n2': {
        'func': borehole, 'grad_func': borehole_gradient,
        'dim': 8, 'n_train': 80,
    },
    'otl_circuit_m6n2': {
        'func': otl_circuit, 'grad_func': otl_circuit_gradient,
        'dim': 6, 'n_train': 60,
    },
    'active_subspace_m10n2': {
        'func': active_subspace_10d, 'grad_func': active_subspace_10d_gradient,
        'dim': 10, 'n_train': 50,
    },
}


def extract_profile_summary(pr):
    """Extract key function timings from a cProfile.Profile object."""
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('tottime')
    ps.print_stats(60)
    text = s.getvalue()

    # Parse top functions
    summary = {}
    stats = ps.stats
    for (file, line, func), (cc, nc, tt, ct, callers) in stats.items():
        key = func
        if 'get_all_derivs' in func or 'get_all_derivs_fast' in func:
            key = 'get_all_derivs'
        elif func == 'mul' and 'pyoti' in file:
            key = 'mul'
        elif '_assemble_kernel_numba' in func:
            key = '_assemble_kernel_numba'
        elif 'rbf_kernel_fast' in func:
            key = 'rbf_kernel_fast'
        elif func == 'sum' and 'pyoti' in file:
            key = 'sum'
        elif func == 'exp' and 'pyoti' in file:
            key = 'exp'
        elif '_cholesky' in func:
            key = 'cholesky'
        elif 'cho_solve' in func:
            key = 'cho_solve'
        elif 'nll_and_grad' in func:
            key = 'nll_and_grad'
        elif '_gc' in func and 'optimizer' in file:
            key = '_gc'
        else:
            continue
        if key not in summary or tt > summary[key]['tottime']:
            summary[key] = {
                'ncalls': nc,
                'tottime': round(tt, 4),
                'cumtime': round(ct, 4),
            }
    return summary, text


def run_benchmark(name, cfg):
    dim = cfg['dim']
    n_train = cfg['n_train']
    func = cfg['func']
    grad_func = cfg['grad_func']

    print(f"\n{'='*60}")
    print(f"  Profiling: {name} (DIM={dim}, n_train={n_train})")
    print(f"{'='*60}")

    sampler = LatinHypercube(d=dim, seed=SEED)
    X_train = sampler.random(n=n_train)
    y_vals = func(X_train)
    grads = grad_func(X_train)

    y_train_list = [y_vals.reshape(-1, 1)]
    for j in range(dim):
        y_train_list.append(grads[:, j].reshape(-1, 1))

    der_indices = [[[[i, 1]] for i in range(1, dim + 1)]]

    model = degp(
        X_train, y_train_list,
        n_order=1, n_bases=dim,
        der_indices=der_indices,
        kernel="SE", kernel_type="anisotropic",
    )

    pr = cProfile.Profile()
    t0 = time.perf_counter()
    pr.enable()
    params = model.optimize_hyperparameters(**JADE_KWARGS)
    pr.disable()
    wall_time = time.perf_counter() - t0

    # Save .prof file
    prof_path = f"profile_{name}_{args.tag}.prof"
    pr.dump_stats(prof_path)
    print(f"  Saved profile: {prof_path}")

    # Extract summary
    summary, text = extract_profile_summary(pr)
    summary['_wall_time'] = round(wall_time, 3)
    summary['_dim'] = dim
    summary['_n_train'] = n_train

    # Print top functions
    print(f"\n  Wall time: {wall_time:.2f}s")
    print(f"\n  Top functions by tottime:")
    for k, v in sorted(summary.items(), key=lambda x: -x[1].get('tottime', 0) if isinstance(x[1], dict) else 0):
        if isinstance(v, dict) and 'tottime' in v:
            print(f"    {k:30s} {v['ncalls']:6d} calls  {v['tottime']:8.3f}s tot  {v['cumtime']:8.3f}s cum")

    return summary, text


# ── Run all ──────────────────────────────────────────────────────────────────
results = {}
full_text = {}
for name, cfg in BENCHMARKS.items():
    summary, text = run_benchmark(name, cfg)
    results[name] = summary
    full_text[name] = text

# Save JSON summary
out_path = f"profile_summary_{args.tag}.json"
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved summary: {out_path}")

# Save full text profiles
txt_path = f"profile_full_{args.tag}.txt"
with open(txt_path, 'w') as f:
    for name, text in full_text.items():
        f.write(f"\n{'='*60}\n  {name}\n{'='*60}\n")
        f.write(text)
        f.write('\n')
print(f"Saved full profiles: {txt_path}")
print("\nDone!")
