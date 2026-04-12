"""Diagnostic: compare dense _compute_grad vs blockwise _compute_grad_blockwise for GDDEGP."""
import numpy as np
import sympy as sp
from scipy.stats import qmc

# -- Branin setup (same as gradient_check_test.py) --
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

# -- Build GDDEGP model --
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

# Test with rho=1.0 (sparse) and rho=3.0 (dense fallback)
for rho_val in [3.0, 1.0]:
    print(f"\n{'='*60}")
    print(f"rho = {rho_val}")
    print(f"{'='*60}")

    model = gddegp(
        X, y_train, n_order=1,
        rays_list=[rays_array],
        der_indices=der_indices,
        derivative_locations=deriv_locs,
        normalize=True, kernel='SE', kernel_type='isotropic',
        rho=rho_val, use_supernodes=False,
    )

    opt = model.optimizer
    n = len(model.bounds)
    x0 = np.array([0.1] * (n - 2) + [0.5, -3.0])

    print(f"_use_dense_factor = {model._use_dense_factor}")
    print(f"n_order = {model.n_order}")
    print(f"x0 = {x0}")

    # Force both paths and compare
    # 1. Dense path
    try:
        old_flag = model._use_dense_factor
        model._use_dense_factor = True
        W, alpha_v, nll_d, phi, n_bases, oti, diffs = opt._dense_nll_and_W(x0)
        grad_dense = opt._compute_grad(x0, W, phi, n_bases, oti, diffs)
        model._use_dense_factor = old_flag
        print(f"\nDense grad:    {grad_dense}")
    except Exception as e:
        print(f"Dense path error: {e}")
        grad_dense = None

    # 2. Blockwise path
    try:
        alpha_v2, U, nll_s, phi2, n_bases2, oti2, diffs2 = opt._sparse_nlml_direct(x0)
        grad_block = opt._compute_grad_blockwise(x0, U, alpha_v2, phi2, n_bases2, oti2, diffs2)
        print(f"Blockwise grad: {grad_block}")
    except Exception as e:
        import traceback
        print(f"Blockwise path error: {e}")
        traceback.print_exc()
        grad_block = None

    # 3. FD gradient
    def _fd_gradient(opt, x0, eps=1e-5):
        grad = np.zeros_like(x0)
        for i in range(len(x0)):
            xp = x0.copy(); xp[i] += eps
            xm = x0.copy(); xm[i] -= eps
            grad[i] = (opt.nll_wrapper(xp) - opt.nll_wrapper(xm)) / (2 * eps)
        return grad

    grad_fd = _fd_gradient(opt, x0)
    print(f"FD grad:        {grad_fd}")

    if grad_dense is not None:
        rel_dense = np.abs(grad_dense - grad_fd) / np.maximum(np.abs(grad_fd), 1e-12)
        print(f"\nDense vs FD rel errors:    {rel_dense}")

    if grad_block is not None:
        rel_block = np.abs(grad_block - grad_fd) / np.maximum(np.abs(grad_fd), 1e-12)
        print(f"Blockwise vs FD rel errors: {rel_block}")

    if grad_dense is not None and grad_block is not None:
        rel_db = np.abs(grad_dense - grad_block) / np.maximum(np.abs(grad_dense), 1e-12)
        print(f"Dense vs Blockwise rel:     {rel_db}")
