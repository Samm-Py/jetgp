"""Debug GDDEGP: check block structure and K_sub consistency."""
import numpy as np
import sympy as sp
from scipy.stats import qmc
from scipy.linalg import cho_factor, cho_solve
from math import comb

def _branin_setup():
    D = 2
    x_sym = sp.symbols('x1 x2')
    a, b, c, r, s, t = 1, 5.1/(4*sp.pi**2), 5/sp.pi, 6, 10, 1/(8*sp.pi)
    f_expr = a*(x_sym[1] - b*x_sym[0]**2 + c*x_sym[0] - r)**2 + s*(1-t)*sp.cos(x_sym[0]) + s
    f_func = sp.lambdify(x_sym, f_expr, 'numpy')
    df_dx = [sp.lambdify(x_sym, sp.diff(f_expr, x_sym[d]), 'numpy') for d in range(D)]
    sampler = qmc.LatinHypercube(d=D, seed=42)
    X = sampler.random(n=12)
    X[:, 0] = X[:, 0] * 15 - 5
    X[:, 1] = X[:, 1] * 15
    y = np.array([f_func(X[i,0], X[i,1]) for i in range(len(X))]).reshape(-1, 1)
    g1 = np.array([df_dx[0](X[i,0], X[i,1]) for i in range(len(X))]).reshape(-1, 1)
    g2 = np.array([df_dx[1](X[i,0], X[i,1]) for i in range(len(X))]).reshape(-1, 1)
    return X, y, g1, g2

X, y, g1, g2 = _branin_setup()
from jetgp.full_gddegp_sparse.gddegp import gddegp

rays_list = []
dir_derivs = []
for i in range(len(X)):
    gx, gy = g1[i].item(), g2[i].item()
    mag = np.sqrt(gx**2 + gy**2)
    if mag < 1e-10:
        ray = np.array([[1.0], [0.0]])
    else:
        ray = np.array([[gx / mag], [gy / mag]])
    rays_list.append(ray)
    dir_derivs.append(gx * ray[0, 0] + gy * ray[1, 0])

rays_array = np.hstack(rays_list)
y_train = [y, np.array(dir_derivs).reshape(-1, 1)]
der_indices = [[[[1, 1]]]]
deriv_locs = [[i for i in range(len(X))]] * 1

model = gddegp(
    X, y_train, n_order=1,
    rays_list=[rays_array],
    der_indices=der_indices,
    derivative_locations=deriv_locs,
    normalize=True, kernel='SE', kernel_type='isotropic',
    rho=3.0, use_supernodes=False,
)

opt = model.optimizer
n = len(model.bounds)
x0 = np.array([0.1] * (n - 2) + [0.5, -3.0])

# Trigger lazy init
alpha_v, U, nll, phi, n_bases, oti, diffs = opt._sparse_nlml_direct(x0)

P_full = model.mmd_P_full
N_total = len(P_full)
block_maps = opt._block_phi_maps

print(f"N_total = {N_total}")
print(f"n_bases = {model.n_bases}")
print(f"block_size = {model.n_bases + 1}")
print(f"Number of blocks = {len(block_maps)}")

for i, bm in enumerate(block_maps):
    nb = bm['nb']
    m = len(nb)
    positions = bm['positions']
    print(f"\nBlock {i}: m={m}, start={bm['start']}, nb={nb}, positions={positions}")

    # Check what types are in this block
    k_type, k_phys, deriv_lookup, sign_lookup = opt._k_index_map
    orig_nb = P_full[nb]
    nb_type = k_type[orig_nb]
    nb_phys = k_phys[orig_nb]
    print(f"  nb_type = {nb_type}")
    print(f"  nb_phys = {nb_phys}")

# Also check deriv_lookup
print(f"\nderiv_lookup:\n{deriv_lookup}")
print(f"\nsign_lookup: {sign_lookup}")

# Check kernel plan indices
plan = opt._kernel_plan
print(f"\nfd_flat_indices: {plan['fd_flat_indices']}")
print(f"df_flat_indices: {plan['df_flat_indices']}")
print(f"dd_flat_indices:\n{plan['dd_flat_indices']}")
print(f"n_deriv_types: {plan['n_deriv_types']}")
print(f"index_sizes: {plan['index_sizes']}")
print(f"idx_offsets: {plan['idx_offsets']}")
print(f"idx_flat: {plan['idx_flat']}")

n_func = phi.shape[0]
row_off = plan.get('row_offsets_abs', plan['row_offsets'] + n_func)
col_off = plan.get('col_offsets_abs', plan['col_offsets'] + n_func)
print(f"row_offsets: {row_off}")
print(f"col_offsets: {col_off}")
