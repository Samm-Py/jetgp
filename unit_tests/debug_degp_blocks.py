"""Check block structure for DEGP vs GDDEGP to see if positions are always valid."""
import numpy as np
import sympy as sp
from scipy.stats import qmc

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

# --- DEGP ---
print("="*60)
print("DEGP block structure (rho=3.0)")
print("="*60)

from jetgp.full_degp_sparse.degp import degp
model_d = degp(
    X, [y, g1, g2], n_order=1, n_bases=2,
    der_indices=[[[[1, 1]], [[2, 1]]]],
    derivative_locations=[[i for i in range(len(X))]] * 2,
    normalize=True, kernel='SE', kernel_type='isotropic',
    rho=3.0, use_supernodes=False,
)
opt_d = model_d.optimizer
n = len(model_d.bounds)
x0 = np.array([0.1] * (n - 2) + [0.5, -3.0])
# trigger lazy init
alpha_v, U, nll, phi, n_bases, oti, diffs = opt_d._sparse_nlml_direct(x0)

print(f"N_total = {len(model_d.mmd_P_full)}, n_bases={model_d.n_bases}, block_size={model_d.n_bases+1}")
P_full = model_d.mmd_P_full
S = model_d.sparse_S_full_arr

for i, bm in enumerate(opt_d._block_phi_maps):
    nb = bm['nb']
    start = bm['start']
    end = min(start + model_d.n_bases + 1, len(P_full))
    block_pts = list(range(start, end))
    missing = [p for p in block_pts if p not in nb]
    if missing:
        print(f"  Block {i}: start={start}, m={len(nb)}, nb={nb}, MISSING: {missing}")

print("All DEGP blocks OK" if all(
    all(p in bm['nb'] for p in range(bm['start'], min(bm['start'] + model_d.n_bases + 1, len(P_full))))
    for bm in opt_d._block_phi_maps
) else "DEGP has MISSING block points")

# --- GDDEGP ---
print("\n" + "="*60)
print("GDDEGP block structure (rho=3.0)")
print("="*60)

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
y_train_g = [y, np.array(dir_derivs).reshape(-1, 1)]
der_indices = [[[[1, 1]]]]
deriv_locs = [[i for i in range(len(X))]] * 1

model_g = gddegp(
    X, y_train_g, n_order=1,
    rays_list=[rays_array],
    der_indices=der_indices,
    derivative_locations=deriv_locs,
    normalize=True, kernel='SE', kernel_type='isotropic',
    rho=3.0, use_supernodes=False,
)
opt_g = model_g.optimizer
n = len(model_g.bounds)
x0g = np.array([0.1] * (n - 2) + [0.5, -3.0])
alpha_v, U, nll, phi, n_bases, oti, diffs = opt_g._sparse_nlml_direct(x0g)

print(f"N_total = {len(model_g.mmd_P_full)}, n_bases={model_g.n_bases}, block_size={model_g.n_bases+1}")
P_full_g = model_g.mmd_P_full
S_g = model_g.sparse_S_full_arr

bad_blocks = 0
for i, bm in enumerate(opt_g._block_phi_maps):
    nb = bm['nb']
    start = bm['start']
    end = min(start + model_g.n_bases + 1, len(P_full_g))
    block_pts = list(range(start, end))
    missing = [p for p in block_pts if p not in nb]
    if missing:
        bad_blocks += 1
        print(f"  Block {i}: start={start}, m={len(nb)}, nb={nb}, MISSING: {missing}")

print(f"\nGDDEGP: {bad_blocks} blocks have missing points out of {len(opt_g._block_phi_maps)}")

# Check what S looks like for GDDEGP
print("\nGDDEGP S array (conditioning sets):")
for i in range(min(len(S_g), 24)):
    s = S_g[i] if isinstance(S_g[i], np.ndarray) else np.asarray(S_g[i])
    print(f"  S[{i}] = {s}")
