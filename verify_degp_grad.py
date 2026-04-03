"""
Finite-difference verification of nll_grad for DEGP, DDEGP, and GDDEGP kernels.

Uses a simple 2D test function f(x,y) = sin(x)*cos(y) on a 3x3 training grid.
All three model types are tested across all supported kernels (SE, RQ, SineExp,
Matern) in both anisotropic and isotropic forms.

Test points are chosen at mild parameter values to keep the NLL smooth and the
FD step well-behaved.

Run with:
    python verify_degp_grad.py
"""
import numpy as np
import itertools

from jetgp.full_degp.degp import degp
from jetgp.full_ddegp.ddegp import ddegp
from jetgp.full_gddegp.gddegp import gddegp


# ─────────────────────────────────────────────────────────────────────────────
# Training data builders
# ─────────────────────────────────────────────────────────────────────────────

def _build_degp_data():
    """2D DEGP: f=sin(x)cos(y), first-order derivs on 3x3 grid."""
    x_vals = np.linspace(-1.0, 1.0, 3)
    X = np.array(list(itertools.product(x_vals, x_vals)))
    N = len(X)
    f   = np.sin(X[:, 0]) * np.cos(X[:, 1])
    df1 = np.cos(X[:, 0]) * np.cos(X[:, 1])
    df2 = -np.sin(X[:, 0]) * np.sin(X[:, 1])
    y_train = [f.reshape(-1, 1), df1.reshape(-1, 1), df2.reshape(-1, 1)]
    # Format: [order_group[ direction[ [base, power], ... ], ... ], ...]
    der_indices = [[[[1, 1]], [[2, 1]]]]
    derivative_locations = [list(range(N)), list(range(N))]
    return X, y_train, der_indices, derivative_locations


def _build_ddegp_data():
    """2D DDEGP: f=sin(x)cos(y), 2 axis-aligned rays on 3x3 grid."""
    x_vals = np.linspace(-1.0, 1.0, 3)
    X = np.array(list(itertools.product(x_vals, x_vals)))
    N = len(X)
    rays = np.array([[1.0, 0.0],
                     [0.0, 1.0]]).T   # shape (2, 2): two axis-aligned rays
    f   = np.sin(X[:, 0]) * np.cos(X[:, 1])
    df1 = np.cos(X[:, 0]) * np.cos(X[:, 1])   # dir-deriv along [1,0]
    df2 = -np.sin(X[:, 0]) * np.sin(X[:, 1])  # dir-deriv along [0,1]
    y_train = [f.reshape(-1, 1), df1.reshape(-1, 1), df2.reshape(-1, 1)]
    # Two directional rays, each with a first-order deriv
    der_indices = [[[[1, 1]], [[2, 1]]]]
    derivative_locations = [list(range(N)), list(range(N))]
    return X, y_train, der_indices, derivative_locations, rays


def _build_gddegp_data():
    """2D GDDEGP: f=sin(x)cos(y), gradient-aligned per-point rays on 3x3 grid."""
    x_vals = np.linspace(-1.0, 1.0, 3)
    X = np.array(list(itertools.product(x_vals, x_vals)))
    N = len(X)
    f  = np.sin(X[:, 0]) * np.cos(X[:, 1])
    gx = np.cos(X[:, 0]) * np.cos(X[:, 1])
    gy = -np.sin(X[:, 0]) * np.sin(X[:, 1])

    # Gradient-aligned unit directions per training point
    grad = np.stack([gx, gy], axis=1)
    norms = np.linalg.norm(grad, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    dirs = grad / norms   # (N, 2)

    # Fallback: axes-aligned direction at near-zero gradient points
    for i in range(N):
        if np.linalg.norm(grad[i]) < 1e-10:
            dirs[i] = np.array([1.0, 0.0])

    rays_array = dirs.T   # shape (2, N) — one ray per training point
    dir_derivs = (gx * dirs[:, 0] + gy * dirs[:, 1]).reshape(-1, 1)

    y_train = [f.reshape(-1, 1), dir_derivs]
    der_indices = [[[[1, 1]]]]
    derivative_locations = [list(range(N))]
    return X, y_train, der_indices, derivative_locations, rays_array


# ─────────────────────────────────────────────────────────────────────────────
# 4th-order central finite differences
# ─────────────────────────────────────────────────────────────────────────────

def fd_gradient(f, x0, h=1e-4):
    grad = np.zeros(len(x0))
    for i in range(len(x0)):
        p1, p2, m1, m2 = [x0.copy() for _ in range(4)]
        p1[i] += h;   p2[i] += 2 * h
        m1[i] -= h;   m2[i] -= 2 * h
        grad[i] = (-f(p2) + 8 * f(p1) - 8 * f(m1) + f(m2)) / (12 * h)
    return grad


# ─────────────────────────────────────────────────────────────────────────────
# Test-point hyperparameter vectors (log10-scaled)
# Layout per the nll_grad docstring:
#   SE/Matern aniso: [log_ell_1..D, log_sf, log_sn]
#   SE/Matern iso:   [log_ell, log_sf, log_sn]
#   RQ aniso:        [log_ell_1..D, log_alpha, log_sf, log_sn]
#   RQ iso:          [log_ell, log_alpha(natural), log_sf, log_sn]
#   SineExp aniso:   [log_ell_1..D, log_p_1..D, log_sf, log_sn]
#   SineExp iso:     [log_ell, log_p, log_sf, log_sn]
# ─────────────────────────────────────────────────────────────────────────────

TEST_POINTS = {
    ('SE',      'anisotropic'): [0.0, 0.0, 0.0, -4.0],
    ('SE',      'isotropic'):   [0.0, 0.0, -4.0],
    ('RQ',      'anisotropic'): [0.0, 0.0, 0.0, 0.0, -4.0],
    ('RQ',      'isotropic'):   [0.0, 0.0, 0.0, -4.0],
    ('SineExp', 'anisotropic'): [0.0, 0.0, 1.0, 1.0, 0.0, -4.0],
    ('SineExp', 'isotropic'):   [0.0, 1.0, 0.0, -4.0],
    ('Matern',  'anisotropic'): [0.0, 0.0, 0.0, -4.0],
    ('Matern',  'isotropic'):   [0.0, 0.0, -4.0],
}


# ─────────────────────────────────────────────────────────────────────────────
# Per-model check functions
# ─────────────────────────────────────────────────────────────────────────────

def _report(kernel, kernel_type, opt, x0):
    """Run analytic vs FD comparison and print results. Returns True if PASS."""
    print(f"\n{'='*65}")
    print(f"  Kernel: {kernel}  ({kernel_type})")
    print(f"{'='*65}")

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
        flag      = "" if both_tiny else ("  <- FAIL" if rel > 0.01 else "")
        if rel > 0.01 and not both_tiny:
            all_ok = False
        note = "  (near-zero, skipped)" if both_tiny else ""
        print(f"  {i:>4}  {a:>14.4e}  {f_:>14.4e}  {rel:>12.2e}{flag}{note}")

    print(f"\n  Result: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


def check_degp(kernel, kernel_type, X, y_train, der_indices, derivative_locations):
    model = degp(
        X, y_train, n_order=1, n_bases=2,
        der_indices=der_indices,
        derivative_locations=derivative_locations,
        normalize=True, kernel=kernel, kernel_type=kernel_type,
    )
    x0 = np.array(TEST_POINTS[(kernel, kernel_type)], dtype=float)
    return _report(kernel, kernel_type, model.optimizer, x0)


def check_ddegp(kernel, kernel_type, X, y_train, der_indices, derivative_locations, rays):
    model = ddegp(
        X, y_train, n_order=1,
        der_indices=der_indices,
        rays=rays,
        derivative_locations=derivative_locations,
        normalize=True, kernel=kernel, kernel_type=kernel_type,
    )
    x0 = np.array(TEST_POINTS[(kernel, kernel_type)], dtype=float)
    return _report(kernel, kernel_type, model.optimizer, x0)


def check_gddegp(kernel, kernel_type, X, y_train, der_indices, derivative_locations, rays_array):
    model = gddegp(
        X, y_train, n_order=1,
        rays_list=[rays_array],
        der_indices=der_indices,
        derivative_locations=derivative_locations,
        normalize=True, kernel=kernel, kernel_type=kernel_type,
    )
    x0 = np.array(TEST_POINTS[(kernel, kernel_type)], dtype=float)
    return _report(kernel, kernel_type, model.optimizer, x0)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    np.random.seed(0)

    degp_data  = _build_degp_data()
    ddegp_data = _build_ddegp_data()
    gddegp_data = _build_gddegp_data()

    results = {}

    # ── DEGP ─────────────────────────────────────────────────────────────────
    print("\n" + "#" * 65)
    print("  DEGP")
    print("#" * 65)
    X, y, di, dl = degp_data
    for (kernel, ktype) in TEST_POINTS:
        key = f"DEGP/{kernel}/{ktype}"
        try:
            results[key] = check_degp(kernel, ktype, X, y, di, dl)
        except Exception:
            import traceback; traceback.print_exc()
            results[key] = False

    # ── DDEGP ────────────────────────────────────────────────────────────────
    print("\n" + "#" * 65)
    print("  DDEGP")
    print("#" * 65)
    X, y, di, dl, rays = ddegp_data
    for (kernel, ktype) in TEST_POINTS:
        key = f"DDEGP/{kernel}/{ktype}"
        try:
            results[key] = check_ddegp(kernel, ktype, X, y, di, dl, rays)
        except Exception:
            import traceback; traceback.print_exc()
            results[key] = False

    # ── GDDEGP ───────────────────────────────────────────────────────────────
    print("\n" + "#" * 65)
    print("  GDDEGP")
    print("#" * 65)
    X, y, di, dl, rays_arr = gddegp_data
    for (kernel, ktype) in TEST_POINTS:
        key = f"GDDEGP/{kernel}/{ktype}"
        try:
            results[key] = check_gddegp(kernel, ktype, X, y, di, dl, rays_arr)
        except Exception:
            import traceback; traceback.print_exc()
            results[key] = False

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("  Summary")
    print(f"{'='*65}")
    for k, v in results.items():
        print(f"  {k:<40}  {'PASS' if v else 'FAIL'}")
