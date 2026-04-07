"""
Verify that the fused_from_real_with_perturbations path in differences_by_dim_func
produces identical results to the original Python-loop path.

Usage:
    python verify_fused_diffs.py
"""

import os
import sys
import time
import argparse
import numpy as np

TESTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TESTS_DIR)

# Default; overridden by --module flag
oti = None

# ─── Original (fallback) implementation ───────────────────────────────────────

def differences_by_dim_func_original(X1, X2, n_order, oti_module, return_deriv=True):
    """Original Python-loop implementation (copy of the fallback path)."""
    X1_np = np.asarray(X1, dtype=np.float64)
    X2_np = np.asarray(X2, dtype=np.float64)
    n1, d = X1_np.shape
    n2 = X2_np.shape[0]

    X1_oti = oti_module.array(X1_np)
    X2_oti = oti_module.array(X2_np)

    differences_by_dim = []

    if n_order == 0:
        for k in range(d):
            diffs_k = oti_module.zeros((n1, n2))
            for i in range(n1):
                diffs_k[i, :] = X1_oti[i, k] - (oti_module.transpose(X2_oti[:, k]))
            differences_by_dim.append(diffs_k)
    elif not return_deriv:
        for k in range(d):
            diffs_k = oti_module.zeros((n1, n2))
            for i in range(n1):
                diffs_k[i, :] = (
                    X1_oti[i, k] + oti_module.e(k + 1, order=n_order)
                ) - (oti_module.transpose(X2_oti[:, k]))
            differences_by_dim.append(diffs_k)
    else:
        for k in range(d):
            diffs_k = oti_module.zeros((n1, n2))
            for i in range(n1):
                diffs_k[i, :] = (
                    X1_oti[i, k] + oti_module.e(k + 1, order=2 * n_order)
                ) - (oti_module.transpose(X2_oti[:, k]))
            differences_by_dim.append(diffs_k)

    return differences_by_dim


# ─── Fused implementation (from degp_utils) ──────────────────────────────────

def differences_by_dim_func_fused(X1, X2, n_order, oti_module, return_deriv=True):
    """Fused C-level implementation."""
    X1_np = np.asarray(X1, dtype=np.float64)
    X2_np = np.asarray(X2, dtype=np.float64)
    n1, d = X1_np.shape
    n2 = X2_np.shape[0]

    differences_by_dim = []
    perturb2 = oti_module.zeros((n2, 1))

    if return_deriv:
        hc_order = 2 * n_order
    else:
        hc_order = n_order

    for k in range(d):
        real_diffs = np.ascontiguousarray(
            X1_np[:, k:k+1] - X2_np[:, k:k+1].T, dtype=np.float64
        )
        perturb1 = oti_module.zeros((n1, 1)) + oti_module.e(k + 1, order=hc_order)
        diffs_k = oti_module.zeros((n1, n2))
        diffs_k.fused_from_real_with_perturbations(real_diffs, perturb1, perturb2)
        differences_by_dim.append(diffs_k)

    return differences_by_dim


# ─── Compare ─────────────────────────────────────────────────────────────────

def compare_results(diffs_orig, diffs_fused, label):
    """Compare two lists of OTI arrays element-by-element."""
    assert len(diffs_orig) == len(diffs_fused), f"{label}: length mismatch"
    max_err = 0.0
    for k, (a, b) in enumerate(zip(diffs_orig, diffs_fused)):
        # Extract the full OTI coefficient arrays for comparison
        a_all = a.get_all_derivs(a.get_active_bases()[-1] if hasattr(a, 'get_active_bases') else 0, 2)
        b_all = b.get_all_derivs(b.get_active_bases()[-1] if hasattr(b, 'get_active_bases') else 0, 2)
        err = np.max(np.abs(a_all - b_all))
        max_err = max(max_err, err)
        if err > 1e-12:
            print(f"  {label} dim {k}: MAX ERROR = {err:.2e}  *** MISMATCH ***")
    return max_err


def main():
    global oti

    parser = argparse.ArgumentParser(description='Verify fused vs original differences_by_dim_func')
    parser.add_argument('--module', default='m8n2',
                        help='OTI module to use, e.g. m8n2, m20n2 (default: m8n2)')
    parser.add_argument('--n1', type=int, default=None, help='Override n1 (rows)')
    parser.add_argument('--n2', type=int, default=None, help='Override n2 (cols)')
    parser.add_argument('--reps', type=int, default=3, help='Timing repetitions')
    args = parser.parse_args()

    # Import the requested OTI module
    mod_name = f"pyoti.static.onum{args.module}"
    import importlib
    oti = importlib.import_module(mod_name)
    print(f"Using OTI module: {mod_name}")

    # Infer dimension from module name
    dim = int(args.module.split('n')[0][1:])  # e.g. m8n2 -> 8, m20n2 -> 20

    np.random.seed(42)

    # Check fused is available
    test_arr = oti.zeros((1, 1))
    has_fused = hasattr(test_arr, 'fused_from_real_with_perturbations')
    print(f"fused_from_real_with_perturbations available: {has_fused}")
    if not has_fused:
        print("ERROR: Fused function not found. Rebuild OTI module with updated cmod_writer.")
        sys.exit(1)

    n1_default = dim
    n2_default = 2000
    n1 = args.n1 if args.n1 is not None else n1_default
    n2 = args.n2 if args.n2 is not None else n2_default

    configs = [
        # (n1, n2, d, n_order, return_deriv, label)
        (5,  10, dim, 1, True,  f"small 5x10 d={dim} order=1 deriv=True"),
        (5,  10, dim, 1, False, f"small 5x10 d={dim} order=1 deriv=False"),
        (n1, n2, dim, 1, True,  f"profile-size {n1}x{n2} d={dim} order=1 deriv=True"),
    ]

    n_reps = args.reps
    all_pass = True
    for n1_c, n2_c, d, n_order, return_deriv, label in configs:
        X1 = np.random.rand(n1_c, d)
        X2 = np.random.rand(n2_c, d)

        print(f"\n--- {label} ---")

        # Correctness check (single run)
        diffs_orig = differences_by_dim_func_original(X1, X2, n_order, oti, return_deriv=return_deriv)
        diffs_fused = differences_by_dim_func_fused(X1, X2, n_order, oti, return_deriv=return_deriv)
        max_err = compare_results(diffs_orig, diffs_fused, label)

        status = "PASS" if max_err < 1e-12 else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  max error: {max_err:.2e}  [{status}]")

        # Timing (multiple reps, report mean)
        t_origs = []
        t_fuseds = []
        for _ in range(n_reps):
            t0 = time.perf_counter()
            _ = differences_by_dim_func_original(X1, X2, n_order, oti, return_deriv=return_deriv)
            t_origs.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            _ = differences_by_dim_func_fused(X1, X2, n_order, oti, return_deriv=return_deriv)
            t_fuseds.append(time.perf_counter() - t0)

        t_orig = np.mean(t_origs)
        t_fused = np.mean(t_fuseds)
        speedup = t_orig / t_fused if t_fused > 0 else float('inf')

        print(f"  original: {t_orig*1000:.1f}ms | fused: {t_fused*1000:.1f}ms | speedup: {speedup:.1f}x")
        print(f"  (mean of {n_reps} reps)")

    print(f"\n{'='*60}")
    print(f"  OVERALL: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
