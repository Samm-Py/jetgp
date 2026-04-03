import sys; sys.path.insert(0, '.')
import numpy as np
from benchmark_functions import morris, morris_gradient
from scipy.stats.qmc import LatinHypercube
from jetgp.full_degp.degp import degp

DIM=20; N=100
sampler = LatinHypercube(d=DIM, seed=42)
X = sampler.random(n=N)
y = morris(X); g = morris_gradient(X)
di = [[[[i+1,1]]] for i in range(DIM)]
dl = [list(range(N)) for _ in range(DIM)]
yt = [y.reshape(-1,1)] + [g[:,j:j+1] for j in range(DIM)]
m = degp(X, yt, n_order=1, n_bases=DIM, der_indices=di, derivative_locations=dl, normalize=True, kernel="SE", kernel_type="anisotropic")
x0 = np.zeros(len(m.bounds))
for i,b in enumerate(m.bounds): x0[i]=0.5*(b[0]+b[1])
oti = m.oti
diffs = m.differences_by_dim
phi = m.kernel_func(diffs, x0[:-1])
n_bases = phi.get_active_bases()[-1]
deriv_order = 2 * m.n_order
factors = m.optimizer._get_deriv_factors(n_bases, deriv_order)
buf = m.optimizer._get_deriv_buf(phi, n_bases, deriv_order)
phi_exp = phi.get_all_derivs_fast(factors, buf)
print(f"phi shape: {phi.shape}")
print(f"phi_exp shape: {phi_exp.shape}")
print(f"n_bases: {n_bases}")
print(f"n_deriv_types: {len(di)}")
print(f"K size: {N + DIM*N} x {N + DIM*N}")
print(f"dd block: {DIM}x{DIM} sub-blocks, each {N}x{N}")
print(f"dd iterations: {DIM*DIM*N*N:,}")
print(f"total K entries: {(N+DIM*N)**2:,}")
print(f"phi_exp_3d would be: ({phi_exp.shape[0]}, {N}, {N})")
print(f"phi_exp memory: {phi_exp.nbytes / 1e6:.1f} MB")

# Force kernel plan computation
m.optimizer._ensure_kernel_plan(n_bases)
plan = m.optimizer._kernel_plan
if plan is not None:
    print(f"\ndd_flat_indices shape: {plan['dd_flat_indices'].shape}")
    print(f"dd_flat_indices unique values: {np.unique(plan['dd_flat_indices'])}")
    print(f"dd_flat_indices max: {plan['dd_flat_indices'].max()}")
    print(f"index_sizes: {plan['index_sizes']}")
    print(f"All index sizes equal N? {np.all(plan['index_sizes'] == N)}")

    # Check if indices are trivial (0..N-1)
    for i, sz in enumerate(plan['index_sizes']):
        off = plan['idx_offsets'][i]
        indices = plan['idx_flat'][off:off+sz]
        is_trivial = np.array_equal(indices, np.arange(sz))
        print(f"  deriv type {i}: size={sz}, trivial={is_trivial}, indices[:5]={indices[:5]}")
        if i >= 3:
            print("  ...")
            break
else:
    print("No kernel plan!")
