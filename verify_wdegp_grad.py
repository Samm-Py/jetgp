"""
Finite-difference verification of nll_grad for all WDEGP kernels.
Uses the exact same data setup as WDEGP_unit_test_2 (six-hump camel, 2D, 4x4 grid).

Test points are chosen to be well within the valid parameter region to avoid
false failures from FD inaccuracy in highly-non-linear regions.

sigma_n gradients are excluded when both analytic and FD are < 1e-3 (effectively zero
at the test point) to avoid misleading relative-error comparisons.

Run with:
    python verify_wdegp_grad.py
"""
import numpy as np
import sympy as sp
import itertools

import jetgp.utils as utils
from jetgp.wdegp.wdegp import wdegp


# ─────────────────────────────────────────────────────────────────────────────
# Shared data setup (replicates WDEGP_unit_test_2)
# ─────────────────────────────────────────────────────────────────────────────

def _build_shared_data():
    np.random.seed(0)
    x_vals = np.linspace(-1.0, 1.0, 4)
    X_train = np.array(list(itertools.product(x_vals, x_vals)))

    submodel_indices = [
        [[0, 3, 12, 15]],
        [[1, 2, 4, 8, 7, 11, 13, 14], [1, 2, 4, 8, 7, 11, 13, 14]],
        [[5, 6, 9, 10], [5, 6, 9, 10], [5, 6, 9, 10], [5, 6, 9, 10], [5, 6, 9, 10]],
    ]
    der_indices = [
        [[[[1, 1]]]],
        [[[[1, 1]], [[2, 1]]]],
        [[[[1, 1]], [[2, 1]]], [[[1, 2]], [[1, 1], [2, 1]], [[2, 2]]]],
    ]

    x1s, x2s = sp.symbols('x1 x2', real=True)
    f_sym = (4 - 2.1*x1s**2 + x1s**4/3)*x1s**2 + x1s*x2s + (-4 + 4*x2s**2)*x2s**2
    f_n  = sp.lambdify((x1s, x2s), f_sym, 'numpy')
    d1_n = [sp.lambdify((x1s, x2s), sp.diff(f_sym, v), 'numpy') for v in (x1s, x2s)]
    d2_n = [[sp.lambdify((x1s, x2s), sp.diff(f_sym, a, b), 'numpy')
             for b in (x1s, x2s)] for a in (x1s, x2s)]

    def ev1(fn, pts):
        r = fn(pts[:, 0], pts[:, 1])
        r = np.atleast_1d(r)
        if r.size == 1 and pts.shape[0] > 1:
            r = np.full(pts.shape[0], r[0])
        return r.reshape(-1, 1)

    y_all = f_n(X_train[:, 0], X_train[:, 1]).reshape(-1, 1)
    c  = X_train[submodel_indices[0][0]]
    e  = X_train[submodel_indices[1][0]]
    p_ = X_train[submodel_indices[2][0]]

    submodel_data = [
        [y_all, ev1(d1_n[0], c)],
        [y_all, ev1(d1_n[0], e), ev1(d1_n[1], e)],
        [y_all, ev1(d1_n[0], p_), ev1(d1_n[1], p_),
         ev1(d2_n[0][0], p_), ev1(d2_n[0][1], p_), ev1(d2_n[1][1], p_)],
    ]
    return X_train, submodel_data, der_indices, submodel_indices


# ─────────────────────────────────────────────────────────────────────────────
# FD gradient (4th-order central differences)
# ─────────────────────────────────────────────────────────────────────────────

def fd_gradient(f, x0, h=1e-4):
    grad = np.zeros(len(x0))
    for i in range(len(x0)):
        p1, p2, m1, m2 = [x0.copy() for _ in range(4)]
        p1[i] += h;   p2[i] += 2*h
        m1[i] -= h;   m2[i] -= 2*h
        grad[i] = (-f(p2) + 8*f(p1) - 8*f(m1) + f(m2)) / (12 * h)
    return grad


# ─────────────────────────────────────────────────────────────────────────────
# Per-kernel test points
# Fixed at mild parameter values so the FD step lands in a smooth region.
# sigma_n is set to -4 (not too small) to keep the noise gradient measurable.
# For SineExp, period p=10^1 keeps the kernel smooth over the data range.
# ─────────────────────────────────────────────────────────────────────────────

TEST_POINTS = {
    ('SE',      'anisotropic'): [0.0, 0.0, 0.0, -4.0],
    ('SE',      'isotropic'):   [0.0, 0.0, -4.0],
    ('RQ',      'anisotropic'): [0.0, 0.0, 0.0, 0.0, -4.0],
    ('RQ',      'isotropic'):   [0.0, 0.0, 0.0, -4.0],
    ('SineExp', 'anisotropic'): [0.0, 0.0, 1.0, 1.0, 0.0, -4.0],
    ('SineExp', 'isotropic'):   [0.0, 1.0, 0.0, -4.0],
}


# ─────────────────────────────────────────────────────────────────────────────

def check_kernel(kernel, kernel_type, X_train, submodel_data, der_indices, submodel_indices):
    print(f"\n{'='*65}")
    print(f"  Kernel: {kernel}  ({kernel_type})")
    print(f"{'='*65}")

    model = wdegp(
        X_train, submodel_data, 2, 2, der_indices,
        derivative_locations=submodel_indices,
        normalize=True, kernel=kernel, kernel_type=kernel_type,
    )
    opt = model.optimizer
    x0  = np.array(TEST_POINTS[(kernel, kernel_type)], dtype=float)
    nll = opt.nll_wrapper(x0)

    print(f"  x0      = {np.round(x0, 3)}")
    print(f"  NLL(x0) = {nll:.4f}")

    if nll > 1e5:
        print("  Kernel matrix ill-conditioned at test point — skipping.")
        return False

    analytic = opt.nll_grad(x0)
    fd       = fd_gradient(opt.nll_wrapper, x0)

    print(f"\n  {'idx':>4}  {'analytic':>14}  {'FD':>14}  {'rel_err':>12}")
    print(f"  {'-'*55}")
    all_ok = True
    for i, (a, f_) in enumerate(zip(analytic, fd)):
        both_tiny = max(abs(a), abs(f_)) < 1e-3
        denom     = max(abs(a), abs(f_), 1e-10)
        rel       = abs(a - f_) / denom
        flag      = "" if both_tiny else ("  ← FAIL" if rel > 0.01 else "")
        if rel > 0.01 and not both_tiny:
            all_ok = False
        note = "  (near-zero, skipped)" if both_tiny else ""
        print(f"  {i:>4}  {a:>14.4e}  {f_:>14.4e}  {rel:>12.2e}{flag}{note}")

    print(f"\n  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == '__main__':
    X_train, submodel_data, der_indices, submodel_indices = _build_shared_data()

    results = {}
    for (kernel, ktype) in TEST_POINTS:
        key = f"{kernel}/{ktype}"
        try:
            results[key] = check_kernel(
                kernel, ktype,
                X_train, submodel_data, der_indices, submodel_indices,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            results[key] = False

    print(f"\n{'='*65}")
    print("  Summary")
    print(f"{'='*65}")
    for k, v in results.items():
        print(f"  {k:<28}  {'PASS' if v else 'FAIL'}")
