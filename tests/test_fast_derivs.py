"""
Test get_all_derivs_fast (direct struct memory cast) against get_all_derivs.
Run after rebuilding a module with the new cmod_writer.py.

Usage: python test_fast_derivs.py
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from math import comb
from scipy.stats.qmc import LatinHypercube
from benchmark_functions import borehole, borehole_gradient, otl_circuit, otl_circuit_gradient
from jetgp.full_degp.degp import degp


def _enum_factors_with_counts(max_basis, ordi):
    from math import factorial
    from collections import Counter
    if ordi == 1:
        for i in range(1, max_basis + 1):
            yield 1.0, {i: 1}
        return
    for last in range(1, max_basis + 1):
        for _, prefix_counts in _enum_factors_with_counts(last, ordi - 1):
            counts = dict(prefix_counts)
            counts[last] = counts.get(last, 0) + 1
            f = 1
            for c in counts.values():
                f *= factorial(c)
            yield float(f), counts


def compute_deriv_factors(nbases, order):
    """Precompute derivative factorial scaling factors for any order."""
    from math import factorial
    from collections import Counter
    factors = [1.0]  # order 0: real part

    for ordi in range(1, order + 1):
        if ordi == 1:
            factors.extend([1.0] * nbases)
        else:
            for f, _ in _enum_factors_with_counts(nbases, ordi):
                factors.append(f)

    return np.array(factors, dtype=np.float64)


def test_fast(name, func, grad_func, dim, n_train=10, seed=42):
    print(f"\n{'='*60}")
    print(f"  {name}: DIM={dim}")
    print(f"{'='*60}")

    sampler = LatinHypercube(d=dim, seed=seed)
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

    diffs = model.differences_by_dim
    ell = np.zeros(dim + 1)
    phi = model.kernel_func(diffs, ell)

    n_bases = phi.get_active_bases()[-1]
    order = 2 * model.n_order
    ndir = comb(n_bases + order, order)

    print(f"  n_bases={n_bases}, order={order}, ndir={ndir}, phi.shape={phi.shape}")

    # Check that get_all_derivs_fast exists
    if not hasattr(phi, 'get_all_derivs_fast'):
        print("  ERROR: get_all_derivs_fast not found! Rebuild module with updated cmod_writer.py")
        return False

    # Reference: get_all_derivs (with get_item + dhelp_get_deriv_factor)
    ref = phi.get_all_derivs(n_bases, order)

    # Fast: get_all_derivs_fast (direct memory cast + precomputed factors)
    factors = compute_deriv_factors(n_bases, order)
    out = np.zeros((ndir, phi.shape[0], phi.shape[1]), dtype=np.float64)
    fast = phi.get_all_derivs_fast(factors, out)

    match = np.allclose(ref, fast)
    max_diff = np.max(np.abs(ref - fast))
    print(f"  get_all_derivs vs get_all_derivs_fast: match={match}, max_diff={max_diff:.2e}")

    if not match:
        # Find where they differ
        diff = np.abs(ref - fast)
        worst = np.unravel_index(np.argmax(diff), diff.shape)
        print(f"  Worst mismatch at index {worst}:")
        print(f"    ref={ref[worst]:.15e}, fast={fast[worst]:.15e}")
        print(f"    factor[{worst[0]}]={factors[worst[0]]}")

        # Check first few derivative planes
        for d in range(min(5, ndir)):
            plane_diff = np.max(np.abs(ref[d] - fast[d]))
            print(f"    plane {d}: max_diff={plane_diff:.2e}, factor={factors[d]:.1f}")

    # Also verify repeated calls work (buffer reuse)
    out2 = np.zeros_like(out)
    fast2 = phi.get_all_derivs_fast(factors, out2)
    match2 = np.allclose(ref, fast2)
    print(f"  Buffer reuse check: match={match2}")

    # Timing comparison
    import time

    N = 100
    t0 = time.perf_counter()
    for _ in range(N):
        phi.get_all_derivs(n_bases, order)
    t_ref = (time.perf_counter() - t0) / N

    t0 = time.perf_counter()
    for _ in range(N):
        phi.get_all_derivs_fast(factors, out)
    t_fast = (time.perf_counter() - t0) / N

    t0 = time.perf_counter()
    buf_into = np.zeros((ndir, phi.shape[0], phi.shape[1]), dtype=np.float64)
    for _ in range(N):
        phi.get_all_derivs_into(n_bases, order, buf_into)
    t_into = (time.perf_counter() - t0) / N

    speedup_vs_orig = t_ref / t_fast if t_fast > 0 else float('inf')
    speedup_vs_into = t_into / t_fast if t_fast > 0 else float('inf')
    print(f"\n  Timing ({N} calls):")
    print(f"    get_all_derivs:      {t_ref*1000:.3f} ms")
    print(f"    get_all_derivs_into: {t_into*1000:.3f} ms")
    print(f"    get_all_derivs_fast: {t_fast*1000:.3f} ms")
    print(f"    Speedup vs original: {speedup_vs_orig:.2f}x")
    print(f"    Speedup vs into:     {speedup_vs_into:.2f}x")

    return match and match2


all_pass = True
ok = test_fast("Borehole (m8n2)", borehole, borehole_gradient, dim=8)
if not ok: all_pass = False

ok = test_fast("OTL Circuit (m6n2)", otl_circuit, otl_circuit_gradient, dim=6)
if not ok: all_pass = False

print(f"\n{'='*60}")
print(f"  OVERALL: {'ALL PASS' if all_pass else 'SOME FAILED'}")
print(f"{'='*60}")
