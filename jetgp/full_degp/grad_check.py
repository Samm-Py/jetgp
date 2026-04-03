"""
Finite-difference gradient check for DEGP optimizer.
Tests: get_all_derivs_fast vs get_all_derivs, rbf_kernel_fast vs rbf_kernel,
and overall nll_and_grad correctness.
"""
import numpy as np
from jetgp.full_degp.degp import degp

np.random.seed(42)
n, d = 15, 2
X = np.random.rand(n, d)
y = (np.sin(X[:, 0]) * np.cos(X[:, 1])).reshape(-1, 1)
dy_dx1 = (np.cos(X[:, 0]) * np.cos(X[:, 1])).reshape(-1, 1)
dy_dx2 = (-np.sin(X[:, 0]) * np.sin(X[:, 1])).reshape(-1, 1)
y_train = [y, dy_dx1, dy_dx2]
der_indices = [
    [[[1, 1]], [[2, 1]]],  # first-order
]
model = degp(X, y_train, n_order=1, n_bases=d, der_indices=der_indices,
             normalize=True, kernel='SE', kernel_type='anisotropic')
opt = model.optimizer

# --- Test 1: get_all_derivs_fast vs get_all_derivs ---
print("=" * 60)
print("Test 1: get_all_derivs_fast vs get_all_derivs")
print("=" * 60)
x0 = np.array([0.1, -0.2, 0.5, -3.0])
diffs = model.differences_by_dim
oti = model.kernel_factory.oti
phi = model.kernel_func(diffs, x0[:-1])
n_bases = phi.get_active_bases()[-1]
deriv_order = 2 * model.n_order

slow = phi.get_all_derivs(n_bases, deriv_order)
if hasattr(phi, 'get_all_derivs_fast'):
    factors = opt._get_deriv_factors(n_bases, deriv_order)
    buf = np.zeros_like(slow)
    fast = phi.get_all_derivs_fast(factors, buf)
    diff = np.max(np.abs(slow - fast))
    print(f"  max |slow - fast| = {diff:.2e}")
    if diff > 1e-10:
        print("  >>> get_all_derivs_fast DISAGREES with get_all_derivs!")
        # Show where they differ
        for i in range(slow.shape[0]):
            d = np.max(np.abs(slow[i] - fast[i]))
            if d > 1e-10:
                print(f"    slice [{i}]: max diff = {d:.2e}, slow max={np.max(np.abs(slow[i])):.2e}, fast max={np.max(np.abs(fast[i])):.2e}")
    else:
        print("  OK")
else:
    print("  get_all_derivs_fast not available, skipping")

# --- Test 2: Check if fused_scale_sq_mul matches manual computation ---
print()
print("=" * 60)
print("Test 2: fused_scale_sq_mul vs manual")
print("=" * 60)
phi = model.kernel_func(diffs, x0[:-1])  # fresh phi
if hasattr(phi, 'fused_scale_sq_mul'):
    ell = 10.0 ** x0[:d]
    ln10 = np.log(10.0)
    for dim in range(d):
        # Manual: -ln10 * ell[dim]^2 * diffs[dim]^2 * phi
        d_sq = oti.mul(diffs[dim], diffs[dim])
        manual = oti.mul(-ln10 * ell[dim] ** 2, oti.mul(d_sq, phi))
        manual_derivs = manual.get_all_derivs(n_bases, deriv_order)

        # Fused path
        phi2 = model.kernel_func(diffs, x0[:-1])  # fresh phi (buffer corruption!)
        dphi_buf = oti.zeros(phi2.shape)
        dphi_buf.fused_scale_sq_mul(diffs[dim], phi2, -ln10 * ell[dim] ** 2)
        fused_derivs = dphi_buf.get_all_derivs(n_bases, deriv_order)

        diff = np.max(np.abs(manual_derivs - fused_derivs))
        print(f"  dim {dim}: max |manual - fused| = {diff:.2e}", end="")
        if diff > 1e-10:
            print("  <--- MISMATCH")
        else:
            print("  OK")
else:
    print("  fused_scale_sq_mul not available, skipping")

# --- Test 3: Overall FD gradient check ---
print()
print("=" * 60)
print("Test 3: Analytic gradient vs finite differences")
print("=" * 60)
for x0 in [np.array([0.1, -0.2, 0.5, -3.0]),
            np.array([-0.5, 0.3, 0.0, -5.0]),
            np.array([0.0, 0.0, 0.0, -4.0])]:
    nll, grad_analytic = opt.nll_and_grad(x0)
    eps = 1e-5
    grad_fd = np.zeros_like(x0)
    for i in range(len(x0)):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        grad_fd[i] = (opt.nll_wrapper(xp) - opt.nll_wrapper(xm)) / (2 * eps)
    print(f"x0: {x0},  nll: {nll:.4f}")
    for i in range(len(x0)):
        rel = abs(grad_analytic[i] - grad_fd[i]) / (abs(grad_fd[i]) + 1e-12)
        flag = '  <--- BAD' if rel > 1e-3 else ''
        print(f"  x[{i}]  analytic={grad_analytic[i]:>12.4e}  fd={grad_fd[i]:>12.4e}  rel={rel:.2e}{flag}")
    print()

# --- Test 4: Force slow path and re-check gradient ---
print("=" * 60)
print("Test 4: Gradient with forced slow path (get_all_derivs only)")
print("=" * 60)
# Monkey-patch _expand_derivs to force slow path
original_expand = opt._expand_derivs
def slow_expand(phi, n_bases, deriv_order):
    return phi.get_all_derivs(n_bases, deriv_order)
opt._expand_derivs = slow_expand

x0 = np.array([0.1, -0.2, 0.5, -3.0])
nll, grad_analytic = opt.nll_and_grad(x0)
eps = 1e-5
grad_fd = np.zeros_like(x0)
for i in range(len(x0)):
    xp = x0.copy(); xp[i] += eps
    xm = x0.copy(); xm[i] -= eps
    grad_fd[i] = (opt.nll_wrapper(xp) - opt.nll_wrapper(xm)) / (2 * eps)
print(f"x0: {x0},  nll: {nll:.4f}")
for i in range(len(x0)):
    rel = abs(grad_analytic[i] - grad_fd[i]) / (abs(grad_fd[i]) + 1e-12)
    flag = '  <--- BAD' if rel > 1e-3 else ''
    print(f"  x[{i}]  analytic={grad_analytic[i]:>12.4e}  fd={grad_fd[i]:>12.4e}  rel={rel:.2e}{flag}")

# Restore
opt._expand_derivs = original_expand
