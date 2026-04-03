"""
Verify that get_all_derivs and get_all_derivs_into produce identical results
using real OTI matrices from actual benchmark problems.
"""
import sys
sys.path.insert(0, '.')
import numpy as np
from math import comb
from scipy.stats.qmc import LatinHypercube
from benchmark_functions import borehole, borehole_gradient, otl_circuit, otl_circuit_gradient
from jetgp.full_degp.degp import degp


def verify_on_model(name, func, grad_func, dim, n_train, seed=1000):
    """Build a real DEGP model and compare get_all_derivs vs get_all_derivs_into."""
    print(f"\n{'='*60}")
    print(f"  {name}: DIM={dim}, n_train={n_train}")
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

    # Get a real phi matrix from kernel computation
    diffs = model.differences_by_dim
    # length scales (dim) + signal variance (1) = dim+1 params, last is noise
    ell = np.zeros(dim + 1)  # log-scaled: dim length scales + 1 signal variance
    phi = model.kernel_func(diffs, ell)

    n_bases = phi.get_active_bases()[-1]
    order = 2 * model.n_order
    ndir = comb(n_bases + order, order)

    print(f"  n_bases={n_bases}, order={order}, ndir={ndir}")
    print(f"  phi shape: {phi.shape}")

    # Method 1: get_all_derivs (allocates new array)
    result1 = phi.get_all_derivs(n_bases, order)

    # Method 2: get_all_derivs_into (preallocated buffer)
    buf = np.zeros((ndir, phi.shape[0], phi.shape[1]), dtype=np.float64)
    result2 = phi.get_all_derivs_into(n_bases, order, buf)

    match = np.allclose(result1, result2)
    max_diff = np.max(np.abs(result1 - result2))
    print(f"  get_all_derivs vs get_all_derivs_into: match={match}, max_diff={max_diff:.2e}")
    print(f"  result shape: {result1.shape}, dtype: {result1.dtype}")
    print(f"  result range: [{result1.min():.4f}, {result1.max():.4f}]")

    # Also verify reuse: call get_all_derivs_into again with same buffer
    result3 = phi.get_all_derivs_into(n_bases, order, buf)
    match2 = np.allclose(result1, result3)
    print(f"  Buffer reuse check: match={match2}")

    return match and match2


all_pass = True

ok = verify_on_model("Borehole (m8n2)", borehole, borehole_gradient, dim=8, n_train=20)
if not ok: all_pass = False

ok = verify_on_model("OTL Circuit (m6n2)", otl_circuit, otl_circuit_gradient, dim=6, n_train=20)
if not ok: all_pass = False

print(f"\n{'='*60}")
print(f"  All pass: {all_pass}")
print(f"{'='*60}")
