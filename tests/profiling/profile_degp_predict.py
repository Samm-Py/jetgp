"""
Profile the DEGP predict() method to identify bottlenecks.

Runs 1st-order and 2nd-order models on borehole (8D) and times each
section of predict(): differences, kernel eval, derivative extraction,
K_s assembly, matrix multiply, denormalization, and (optionally)
covariance computation.

Usage:
    python profile_degp_predict.py              # default: borehole 8D
    python profile_degp_predict.py --func otl_circuit --order 1
    python profile_degp_predict.py --func borehole --order 2 --cov
"""

import os
import sys
import time
import argparse
import numpy as np
from collections import OrderedDict

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS_DIR)

from benchmark_functions import (
    borehole, borehole_gradient, borehole_hessian_diag,
    otl_circuit, otl_circuit_gradient, otl_circuit_hessian_diag,
    morris, morris_gradient, morris_hessian_diag,
    generate_test_data,
)
from scipy.stats.qmc import LatinHypercube

from jetgp.full_degp.degp import degp
from jetgp.full_degp import degp_utils
import jetgp.utils as utils
from jetgp.kernel_funcs.kernel_funcs import KernelFactory, get_oti_module
from scipy.linalg import cho_solve, cho_factor, solve_triangular


BENCHMARKS = {
    'borehole':    {'func': borehole,    'grad': borehole_gradient,    'hess': borehole_hessian_diag,    'dim': 8},
    'otl_circuit': {'func': otl_circuit,  'grad': otl_circuit_gradient,  'hess': otl_circuit_hessian_diag,  'dim': 6},
    'morris':      {'func': morris,       'grad': morris_gradient,       'hess': morris_hessian_diag,       'dim': 20},
}


def make_der_indices_1st(dim):
    return [[[[i, 1]] for i in range(1, dim + 1)]]


def make_der_indices_2nd(dim):
    first_order = [[[i, 1]] for i in range(1, dim + 1)]
    second_order = [[[i, 2]] for i in range(1, dim + 1)]
    return [first_order, second_order]


class Timer:
    """Context manager that records elapsed time."""
    def __init__(self, label, results):
        self.label = label
        self.results = results

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.results[self.label] = time.perf_counter() - self.t0


def profile_predict(func_name, n_order, n_train, n_test, calc_cov, seed=42, n_reps=5):
    cfg = BENCHMARKS[func_name]
    dim = cfg['dim']

    # --- Generate data ---
    sampler = LatinHypercube(d=dim, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = cfg['func'](X_train).reshape(-1, 1)
    grads = cfg['grad'](X_train)

    y_train = [y_vals] + [grads[:, j].reshape(-1, 1) for j in range(dim)]

    if n_order == 2:
        hess = cfg['hess'](X_train)
        y_train += [hess[:, j].reshape(-1, 1) for j in range(dim)]
        der_indices = make_der_indices_2nd(dim)
    else:
        der_indices = make_der_indices_1st(dim)

    X_test, _ = generate_test_data(cfg['func'], n_test, dim, seed=99)

    # --- Build model and optimize ---
    print(f"\n{'='*70}")
    print(f"  {func_name} ({dim}D) | order={n_order} | n_train={n_train} | n_test={n_test}")
    n_tasks = 1 + dim if n_order == 1 else 1 + 2 * dim
    print(f"  K_train size: {n_train * n_tasks} x {n_train * n_tasks}")
    print(f"  K_s size:     {n_train * n_tasks} x {n_test}")
    if calc_cov:
        print(f"  K_ss size:    {n_test} x {n_test}")
    print(f"{'='*70}")

    model = degp(
        X_train, y_train, n_order=n_order, n_bases=dim,
        der_indices=der_indices, normalize=True,
        kernel='SE', kernel_type='anisotropic'
    )

    print("\n  Optimizing hyperparameters...")
    t0 = time.perf_counter()
    params = model.optimize_hyperparameters()
    t_opt = time.perf_counter() - t0
    print(f"  Optimization: {t_opt:.3f}s")

    # --- Warm-up predict call ---
    _ = model.predict(X_test, params, calc_cov=calc_cov)

    # --- Instrumented predict (manual step-by-step) ---
    # We replicate predict() logic but time each section

    all_timings = []
    for rep in range(n_reps):
        timings = OrderedDict()
        length_scales = params[:-1]
        sigma_n = params[-1]

        # 1. Cache check / K_train solve
        with Timer('1_cache_check_or_cholesky', timings):
            _cache_hit = (
                hasattr(model, '_cached_params')
                and model._cached_params is not None
                and np.array_equal(model._cached_params, params)
            )
            if _cache_hit:
                L = model._cached_L
                low = model._cached_low
                alpha = model._cached_alpha
            else:
                phi_train = model.kernel_func(model.differences_by_dim, length_scales)
                if model.n_order > 0:
                    phi_exp_train = phi_train.get_all_derivs(model.n_bases, 2 * model.n_order)
                else:
                    phi_exp_train = phi_train.real[np.newaxis, :, :]
                K = degp_utils.rbf_kernel(
                    phi_train, phi_exp_train, model.n_order, model.n_bases,
                    model.flattened_der_indices, model.powers,
                    index=model.derivative_locations
                )
                K += (10 ** sigma_n) ** 2 * np.eye(K.shape[0])
                K += model.sigma_data ** 2
                L, low = cho_factor(K, lower=True)
                alpha = cho_solve((L, low), model.y_train)

        # 2. Normalize test inputs
        with Timer('2_normalize_test', timings):
            if model.normalize:
                X_test_norm = utils.normalize_x_data_test(X_test, model.sigmas_x, model.mus_x)
            else:
                X_test_norm = X_test

        # 3. Compute train-test differences (OTI)
        with Timer('3_differences_train_test', timings):
            diff_x_test_x_train = degp_utils.differences_by_dim_func(
                model.x_train, X_test_norm, model.n_order, model.oti
            )

        # 4. Kernel evaluation on train-test differences
        with Timer('4_kernel_eval_train_test', timings):
            phi_train_test = model.kernel_func(diff_x_test_x_train, length_scales)

        # 5. Extract derivatives from OTI kernel result
        with Timer('5_get_all_derivs_train_test', timings):
            if model.n_order > 0:
                phi_exp_train_test = phi_train_test.get_all_derivs(model.n_bases, model.n_order)
            else:
                phi_exp_train_test = phi_train_test.real[np.newaxis, :, :]

        # 6. Assemble K_s cross-covariance
        with Timer('6_assemble_K_s', timings):
            K_s = degp_utils.rbf_kernel_predictions(
                phi_train_test, phi_exp_train_test, model.n_order, model.n_bases,
                model.flattened_der_indices, model.powers,
                return_deriv=False,
                index=model.derivative_locations,
                common_derivs=[],
                powers_predict=None
            )

        # 7. Matrix multiply: f_mean = K_s.T @ alpha
        with Timer('7_matmul_Ks_alpha', timings):
            f_mean = K_s.T @ alpha

        # 8. Denormalize
        with Timer('8_denormalize', timings):
            if model.normalize:
                f_mean_out = model.mu_y + f_mean * model.sigma_y
            else:
                f_mean_out = f_mean

        if calc_cov:
            # 9. Compute test-test differences
            with Timer('9_differences_test_test', timings):
                diff_x_test_x_test = degp_utils.differences_by_dim_func(
                    X_test_norm, X_test_norm, model.n_order, model.oti
                )

            # 10. Kernel evaluation on test-test differences
            with Timer('10_kernel_eval_test_test', timings):
                phi_test_test = model.kernel_func(diff_x_test_x_test, length_scales)

            # 11. Extract derivatives for test-test
            with Timer('11_get_all_derivs_test_test', timings):
                if model.n_order > 0:
                    phi_exp_test_test = phi_test_test.get_all_derivs(model.n_bases, 2 * model.n_order)
                else:
                    phi_exp_test_test = phi_test_test.real[np.newaxis, :, :]

            # 12. Assemble K_ss
            with Timer('12_assemble_K_ss', timings):
                derivative_locations_test = None
                K_ss = degp_utils.rbf_kernel_predictions(
                    phi_test_test, phi_exp_test_test, model.n_order, model.n_bases,
                    model.flattened_der_indices, model.powers,
                    return_deriv=False,
                    index=derivative_locations_test,
                    common_derivs=[],
                    calc_cov=True,
                    powers_predict=None
                )

            # 13. Solve for covariance: v = L^{-1} K_s, f_cov = K_ss - v.T @ v
            with Timer('13_cov_triangular_solve', timings):
                v = solve_triangular(L, K_s, lower=low)
                f_cov = K_ss - v.T @ v

        all_timings.append(timings)

    # --- Report ---
    print(f"\n  Profiling results (mean of {n_reps} reps):")
    print(f"  {'Step':<40s} {'Mean (ms)':>10s} {'Std (ms)':>10s} {'% Total':>8s}")
    print(f"  {'-'*70}")

    # Compute means
    keys = all_timings[0].keys()
    means = {}
    stds = {}
    for k in keys:
        vals = [t[k] for t in all_timings]
        means[k] = np.mean(vals)
        stds[k] = np.std(vals)

    total = sum(means.values())

    for k in keys:
        pct = 100.0 * means[k] / total if total > 0 else 0
        print(f"  {k:<40s} {means[k]*1000:>10.3f} {stds[k]*1000:>10.3f} {pct:>7.1f}%")

    print(f"  {'-'*70}")
    print(f"  {'TOTAL':<40s} {total*1000:>10.3f} {'':>10s} {'100.0%':>8s}")
    print(f"\n  Cache hit: {_cache_hit}")

    # Also run cProfile on predict() for fine-grained view
    print(f"\n  {'='*70}")
    print(f"  cProfile of predict() ({n_reps} calls):")
    print(f"  {'='*70}")
    import cProfile
    import pstats
    import io

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(n_reps):
        model.predict(X_test, params, calc_cov=calc_cov)
    pr.disable()

    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('cumulative')
    ps.print_stats(25)
    print(s.getvalue())

    print("\n  --- Sorted by tottime ---")
    s2 = io.StringIO()
    ps2 = pstats.Stats(pr, stream=s2)
    ps2.sort_stats('tottime')
    ps2.print_stats(25)
    print(s2.getvalue())


def main():
    parser = argparse.ArgumentParser(description='Profile DEGP predict()')
    parser.add_argument('--func', default='borehole', choices=list(BENCHMARKS.keys()))
    parser.add_argument('--order', type=int, default=1, choices=[1, 2])
    parser.add_argument('--n-train', type=int, default=None,
                        help='Training size (default: 3*dim)')
    parser.add_argument('--n-test', type=int, default=2000)
    parser.add_argument('--cov', action='store_true', help='Also profile covariance computation')
    parser.add_argument('--reps', type=int, default=5, help='Number of repetitions for timing')
    args = parser.parse_args()

    dim = BENCHMARKS[args.func]['dim']
    n_train = args.n_train if args.n_train is not None else 3 * dim

    profile_predict(
        func_name=args.func,
        n_order=args.order,
        n_train=n_train,
        n_test=args.n_test,
        calc_cov=args.cov,
        n_reps=args.reps,
    )


if __name__ == '__main__':
    main()
