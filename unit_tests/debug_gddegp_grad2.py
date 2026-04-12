"""Diagnostic: compare W_proj from dense vs blockwise paths for GDDEGP at rho=3."""
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

# --- Dense path: extract W_proj ---
W, alpha_v, nll_d, phi, n_bases, oti, diffs = opt._dense_nll_and_W(x0)
deriv_order = 2 * model.n_order

# Compute W_proj via the dense path
from jetgp.full_gddegp_sparse import gddegp_utils as utils
phi_exp = opt._expand_derivs(phi, n_bases, deriv_order)
base_shape = phi.shape
phi_3d = phi_exp.reshape(phi_exp.shape[0], base_shape[0], base_shape[1])

ndir = comb(n_bases + deriv_order, deriv_order)
proj_shape = (ndir, base_shape[0], base_shape[1])
W_proj_dense = np.empty(proj_shape)

plan = opt._kernel_plan
row_off = plan.get('row_offsets_abs', plan['row_offsets'] + base_shape[0])
col_off = plan.get('col_offsets_abs', plan['col_offsets'] + base_shape[1])

utils._project_W_to_phi_space(
    W, W_proj_dense, base_shape[0], base_shape[1],
    plan['fd_flat_indices'], plan['df_flat_indices'],
    plan['dd_flat_indices'],
    plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
    plan['n_deriv_types'], row_off, col_off,
)
print(f"W_proj_dense shape: {W_proj_dense.shape}")
print(f"W_proj_dense range: [{W_proj_dense.min():.6e}, {W_proj_dense.max():.6e}]")
print(f"W_proj_dense abs sum: {np.abs(W_proj_dense).sum():.6e}")

# --- Blockwise path: extract W_proj ---
from jetgp.full_gddegp_sparse.optimizer import _project_G_to_W_proj

alpha_v2, U, nll_s, phi2, n_bases2, oti2, diffs2 = opt._sparse_nlml_direct(x0)

sigma_n_sq = (10.0 ** x0[-1]) ** 2
P_full = model.mmd_P_full
N_total = len(P_full)
n_func = phi2.shape[0]

k_type, k_phys, deriv_lookup, sign_lookup = opt._k_index_map

phi_exp2 = opt._expand_derivs(phi2, n_bases2, deriv_order)
phi_3d2 = phi_exp2.reshape(phi_exp2.shape[0], n_func, n_func)
phi_flat2 = phi_3d2.ravel()

block_maps = opt._block_phi_maps
y_ord = model.y_train[P_full]

W_proj_block = np.zeros(proj_shape)
noise_trace = 0.0

for bm in block_maps:
    nb   = bm['nb']
    m    = len(nb)
    positions = bm['positions']
    n_cols    = len(positions)

    K_sub = phi_flat2[bm['flat_idx']].reshape(m, m) * bm['sign_mat']
    diag_idx = np.arange(m)
    K_sub[diag_idx, diag_idx] += sigma_n_sq + bm['sd_diag']

    L_u, low_u = cho_factor(K_sub, lower=True)

    E = np.zeros((m, n_cols))
    E[positions, np.arange(n_cols)] = 1.0
    V = cho_solve((L_u, low_u), E)

    y_nb = y_ord[nb]
    alpha_b = cho_solve((L_u, low_u), y_nb)

    s     = V[positions, np.arange(n_cols)]
    a     = alpha_b[positions]
    beta  = a / s
    gamma = beta ** 2 + 1.0 / s

    M = V * gamma[np.newaxis, :] - 2.0 * alpha_b[:, np.newaxis] * beta[np.newaxis, :]
    noise_trace += np.sum(M * V)

    orig_nb  = P_full[nb]
    nb_type  = k_type[orig_nb]
    nb_phys  = k_phys[orig_nb]
    G_b = M @ V.T
    _project_G_to_W_proj(W_proj_block, G_b, nb_type, nb_phys,
                         deriv_lookup, sign_lookup, m)

print(f"\nW_proj_block shape: {W_proj_block.shape}")
print(f"W_proj_block range: [{W_proj_block.min():.6e}, {W_proj_block.max():.6e}]")
print(f"W_proj_block abs sum: {np.abs(W_proj_block).sum():.6e}")

# Compare
diff = W_proj_dense - W_proj_block
print(f"\nDiff range: [{diff.min():.6e}, {diff.max():.6e}]")
print(f"Diff abs sum: {np.abs(diff).sum():.6e}")
rel_diff = np.abs(diff) / np.maximum(np.abs(W_proj_dense), 1e-12)
print(f"Max relative diff: {rel_diff.max():.6e}")

# Check per-derivative-dim
for d in range(min(W_proj_dense.shape[0], 10)):
    d_dense = W_proj_dense[d]
    d_block = W_proj_block[d]
    if np.abs(d_dense).max() > 1e-10 or np.abs(d_block).max() > 1e-10:
        diff_d = np.abs(d_dense - d_block).max()
        scale_d = max(np.abs(d_dense).max(), np.abs(d_block).max(), 1e-12)
        print(f"  dim {d}: dense max={np.abs(d_dense).max():.4e}, block max={np.abs(d_block).max():.4e}, rel diff={diff_d/scale_d:.4e}")
