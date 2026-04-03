"""Test fused_sqdist against the Python-loop reference implementation."""
import numpy as np
import time

def test_fused_sqdist(module_name='onumm8n2'):
    mod = __import__(f'pyoti.static.{module_name}', fromlist=[module_name])

    n, m = 50, 50
    dim = 8

    # Create random OTI diff arrays
    rng = np.random.default_rng(42)
    diffs = []
    for d in range(dim):
        arr = mod.zeros((n, m))
        # Set real parts to random values
        real_vals = rng.standard_normal((n, m))
        for i in range(n):
            for j in range(m):
                arr[i, j] = float(real_vals[i, j])
        diffs.append(arr)

    ell = rng.uniform(0.5, 2.0, size=dim)
    ell_sq = np.ascontiguousarray(ell ** 2, dtype=np.float64)

    # Reference: Python loop
    sqdist_ref = mod.zeros((n, m))
    tmp1 = mod.zeros((n, m))
    tmp2 = mod.zeros((n, m))
    for i in range(dim):
        mod.mul(ell[i], diffs[i], out=tmp1)
        mod.mul(tmp1, tmp1, out=tmp2)
        mod.sum(sqdist_ref, tmp2, out=sqdist_ref)

    # Check if fused_sqdist is available
    sqdist_fused = mod.zeros((n, m))
    if not hasattr(sqdist_fused, 'fused_sqdist'):
        print(f"fused_sqdist not available on {module_name} — need to recompile")
        return

    # Fused version
    sqdist_fused.fused_sqdist(diffs, ell_sq)

    # Compare: extract real parts
    ref_derivs = sqdist_ref.get_all_derivs(dim, 2)
    fused_derivs = sqdist_fused.get_all_derivs(dim, 2)

    max_diff = np.max(np.abs(ref_derivs - fused_derivs))
    print(f"[{module_name}] max_diff = {max_diff:.2e}")
    assert max_diff < 1e-10, f"Mismatch: max_diff = {max_diff}"

    # Timing
    n_iter = 100
    t0 = time.perf_counter()
    for _ in range(n_iter):
        mod.mul(0.0, sqdist_ref, out=sqdist_ref)
        for i in range(dim):
            mod.mul(ell[i], diffs[i], out=tmp1)
            mod.mul(tmp1, tmp1, out=tmp2)
            mod.sum(sqdist_ref, tmp2, out=sqdist_ref)
    t_loop = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(n_iter):
        sqdist_fused.fused_sqdist(diffs, ell_sq)
    t_fused = time.perf_counter() - t0

    print(f"  Loop:  {t_loop:.4f}s ({n_iter} iters)")
    print(f"  Fused: {t_fused:.4f}s ({n_iter} iters)")
    print(f"  Speedup: {t_loop/t_fused:.1f}x")


if __name__ == '__main__':
    test_fused_sqdist('onumm8n2')
