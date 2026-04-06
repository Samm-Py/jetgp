"""
Profile the WDEGP predict() method to identify bottlenecks.

Runs a 1-submodel-per-point WDEGP on morris (20D) or borehole (8D) and
times each section of predict(): weight computation, OTI differences,
kernel evaluation, per-submodel K build + solve + K_s, and (optionally)
covariance.

Usage:
    python profile_wdegp_predict.py                          # default: morris 20D
    python profile_wdegp_predict.py --func borehole
    python profile_wdegp_predict.py --func morris --cov
    python profile_wdegp_predict.py --func morris --n-train 40 --reps 3
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
    borehole, borehole_gradient,
    otl_circuit, otl_circuit_gradient,
    morris, morris_gradient,
    generate_test_data,
)
from scipy.stats.qmc import LatinHypercube

from jetgp.wdegp.wdegp import wdegp
import jetgp.utils as utils

BENCHMARKS = {
    'borehole':    {'func': borehole,    'grad': borehole_gradient,    'dim': 8},
    'otl_circuit': {'func': otl_circuit,  'grad': otl_circuit_gradient,  'dim': 6},
    'morris':      {'func': morris,       'grad': morris_gradient,       'dim': 20},
}


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


def build_wdegp_model(func_name, n_train, seed=42):
    """Build a 1-submodel-per-point WDEGP and optimize hyperparameters."""
    cfg = BENCHMARKS[func_name]
    dim = cfg['dim']

    sampler = LatinHypercube(d=dim, seed=seed)
    X_train = sampler.random(n=n_train)
    y_vals = cfg['func'](X_train).reshape(-1, 1)
    grads = cfg['grad'](X_train)

    der_specs = utils.gen_OTI_indices(dim, 1)

    submodel_data = []
    derivative_specs_list = []
    derivative_locations_list = []

    for i in range(n_train):
        data_i = [y_vals] + [grads[i:i+1, j:j+1] for j in range(dim)]
        submodel_data.append(data_i)
        derivative_specs_list.append(der_specs)
        derivative_locations_list.append([[i] for _ in range(dim)])

    model = wdegp(
        X_train, submodel_data,
        1, dim,
        derivative_specs_list,
        derivative_locations=derivative_locations_list,
        normalize=True,
        kernel='SE', kernel_type='anisotropic'
    )

    print("  Optimizing hyperparameters...")
    t0 = time.perf_counter()
    params = model.optimize_hyperparameters(
        optimizer='jade',
        n_generations=10,
        local_opt_every=10,
        pop_size=10,
        debug=False
    )
    t_opt = time.perf_counter() - t0
    print(f"  Optimization: {t_opt:.3f}s")

    return model, params, dim


def profile_predict(func_name, n_train, n_test, calc_cov, seed=42, n_reps=5):
    cfg = BENCHMARKS[func_name]
    dim = cfg['dim']

    model, params, dim = build_wdegp_model(func_name, n_train, seed)
    X_test, _ = generate_test_data(cfg['func'], n_test, dim, seed=99)

    print(f"\n{'='*70}")
    print(f"  WDEGP: {func_name} ({dim}D) | n_train={n_train} | n_test={n_test}")
    print(f"  Submodels: {model.num_submodels}")
    n_tasks = 1 + dim  # function + gradient
    print(f"  Per-submodel K size: varies (submodel-specific derivative locations)")
    print(f"  calc_cov: {calc_cov}")
    print(f"{'='*70}")

    # Warm-up
    _ = model.predict(X_test, params, calc_cov=calc_cov)

    # --- Instrumented predict ---
    # We replicate the predict() logic step-by-step with timers

    gp_utils = model._get_utils_module()
    ell = params[:-1]
    sigma_n = params[-1]
    x_train = model.x_train_normalized if model.normalize else model.x_train

    all_timings = []
    for rep in range(n_reps):
        timings = OrderedDict()

        # 1. Normalize test inputs
        with Timer('01_normalize_test', timings):
            if model.normalize:
                X_test_norm = utils.normalize_x_data_test(X_test, model.sigmas_x, model.mus_x)
            else:
                X_test_norm = X_test

        # 2. Compute weight differences + weights
        with Timer('02_weight_differences', timings):
            if model.num_submodels > 1:
                if model.submodel_type == 'degp':
                    from jetgp.wdegp import wdegp_utils
                    diffs_for_weights = wdegp_utils.differences_by_dim_func(
                        X_test_norm, x_train, 0, model.oti, return_deriv=False
                    )
                else:
                    diffs_for_weights = model._compute_weight_differences(X_test_norm, x_train)

        with Timer('03_compute_weights', timings):
            if model.num_submodels > 1:
                diffs_train_for_weights = model.differences_by_dim
                weights_matrix = gp_utils.determine_weights(
                    diffs_train_for_weights, diffs_for_weights, ell, model.kernel_func, sigma_n
                )

        # 4. Train-test OTI differences
        with Timer('04_differences_train_test', timings):
            diffs_train_test = model._compute_train_test_differences(
                x_train, X_test_norm, False, None,
                predict_order=model.n_order, predict_oti=model.oti
            )

        # 5. Train-train kernel evaluation
        with Timer('05_kernel_train_train', timings):
            diffs_train_train = model.differences_by_dim
            phi_train_train = model.kernel_func(diffs_train_train, ell).copy()

        # 6. Extract train-train derivatives
        with Timer('06_get_derivs_train_train', timings):
            if model.n_order == 0:
                n_bases_rays = 0
                phi_exp_train_train = phi_train_train.real[np.newaxis, :, :]
            else:
                n_bases_rays = phi_train_train.get_active_bases()[-1]
                phi_exp_train_train = phi_train_train.get_all_derivs(n_bases_rays, 2 * model.n_order)

        # 7. Train-test kernel evaluation
        with Timer('07_kernel_train_test', timings):
            phi_train_test = model.kernel_func(diffs_train_test, ell).copy()

        # 8. Extract train-test derivatives
        with Timer('08_get_derivs_train_test', timings):
            if model.n_order > 0:
                phi_exp_train_test = phi_train_test.get_all_derivs(n_bases_rays, model.n_order)
            else:
                phi_exp_train_test = phi_train_test.real[np.newaxis, :, :]

        # 9. Test-test kernel (if covariance)
        if calc_cov:
            with Timer('09_differences_test_test', timings):
                diffs_test_test = model._compute_test_test_differences(
                    X_test_norm, False, None,
                    predict_order=model.n_order, predict_oti=model.oti
                )

            with Timer('10_kernel_test_test', timings):
                phi_test_test = model.kernel_func(diffs_test_test, ell).copy()

            with Timer('11_get_derivs_test_test', timings):
                if model.n_order == 0:
                    phi_exp_test_test = phi_test_test.real[np.newaxis, :, :]
                else:
                    phi_exp_test_test = phi_test_test.get_all_derivs(n_bases_rays, 2 * model.n_order)

        # 10. Submodel loop
        with Timer('12_submodel_loop_total', timings):
            submodel_K_times = []
            submodel_solve_times = []
            submodel_Ks_times = []
            submodel_mean_times = []

            for i in range(model.num_submodels):
                deriv_locs_i = model.derivative_locations[i]

                t0 = time.perf_counter()
                K = gp_utils.rbf_kernel(
                    phi_train_train, phi_exp_train_train, model.n_order, n_bases_rays,
                    model.flattened_der_indices[i], model.powers[i],
                    index=deriv_locs_i
                )
                K += (10 ** sigma_n) ** 2 * np.eye(len(K))
                submodel_K_times.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                from scipy.linalg import cho_solve, cho_factor
                L, low = cho_factor(K, lower=True)
                alpha = cho_solve((L, low), model.y_train_normalized[i])
                submodel_solve_times.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                K_s = gp_utils.rbf_kernel_predictions(
                    phi_train_test, phi_exp_train_test, model.n_order, n_bases_rays,
                    model.flattened_der_indices[i], model.powers[i],
                    return_deriv=False,
                    index=deriv_locs_i,
                    common_derivs=[],
                    powers_predict=None
                )
                submodel_Ks_times.append(time.perf_counter() - t0)

                t0 = time.perf_counter()
                f_mean = K_s.T @ alpha
                submodel_mean_times.append(time.perf_counter() - t0)

        # Store submodel breakdown
        timings['12a_submodel_K_build (total)'] = sum(submodel_K_times)
        timings['12b_submodel_cholesky (total)'] = sum(submodel_solve_times)
        timings['12c_submodel_Ks_build (total)'] = sum(submodel_Ks_times)
        timings['12d_submodel_matmul (total)'] = sum(submodel_mean_times)
        timings['12e_submodel_K_build (mean/sub)'] = np.mean(submodel_K_times)
        timings['12f_submodel_cholesky (mean/sub)'] = np.mean(submodel_solve_times)

        all_timings.append(timings)

    # --- Report ---
    print(f"\n  Profiling results (mean of {n_reps} reps):")
    print(f"  {'Step':<45s} {'Mean (ms)':>10s} {'Std (ms)':>10s} {'% Total':>8s}")
    print(f"  {'-'*75}")

    keys = all_timings[0].keys()
    means = {}
    stds = {}
    for k in keys:
        vals = [t[k] for t in all_timings]
        means[k] = np.mean(vals)
        stds[k] = np.std(vals)

    # Only sum top-level steps for percentage (not 12a-12f breakdowns)
    top_level_keys = [k for k in keys if not k.startswith('12') or k == '12_submodel_loop_total']
    total = sum(means[k] for k in top_level_keys)

    for k in keys:
        if k.startswith('12') and k != '12_submodel_loop_total':
            # Sub-breakdown: indent and no percentage
            print(f"    {k:<43s} {means[k]*1000:>10.3f} {stds[k]*1000:>10.3f}")
        else:
            pct = 100.0 * means[k] / total if total > 0 else 0
            print(f"  {k:<45s} {means[k]*1000:>10.3f} {stds[k]*1000:>10.3f} {pct:>7.1f}%")

    print(f"  {'-'*75}")
    print(f"  {'TOTAL':<45s} {total*1000:>10.3f} {'':>10s} {'100.0%':>8s}")

    # Also run cProfile
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
    parser = argparse.ArgumentParser(description='Profile WDEGP predict()')
    parser.add_argument('--func', default='morris', choices=list(BENCHMARKS.keys()))
    parser.add_argument('--n-train', type=int, default=None,
                        help='Training size (default: dim)')
    parser.add_argument('--n-test', type=int, default=2000)
    parser.add_argument('--cov', action='store_true', help='Also profile covariance computation')
    parser.add_argument('--reps', type=int, default=5, help='Number of repetitions for timing')
    args = parser.parse_args()

    dim = BENCHMARKS[args.func]['dim']
    n_train = args.n_train if args.n_train is not None else dim

    profile_predict(
        func_name=args.func,
        n_train=n_train,
        n_test=args.n_test,
        calc_cov=args.cov,
        n_reps=args.reps,
    )


if __name__ == '__main__':
    main()
