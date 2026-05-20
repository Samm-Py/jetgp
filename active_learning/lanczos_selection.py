"""Lanczos-based derivative uncertainty helpers for adaptive DOE."""

import numpy as np


def _make_ray(direction, n_points):
    v = np.asarray(direction, dtype=float).reshape(-1)
    v = v / np.linalg.norm(v)
    return np.tile(v.reshape(-1, 1), (1, n_points))


def gradient_cov_matvec(gp_model, params, x_point, v):
    """
    Compute A·v where A = Cov[∇f(x_point) | data].

    Batches coordinate-axis queries into the GP's predict-time basis budget.
    """
    x_point = np.asarray(x_point, dtype=float).reshape(-1)
    d = x_point.size
    v = np.asarray(v, dtype=float).reshape(-1)
    nv = np.linalg.norm(v)
    if nv == 0.0:
        return np.zeros(d)
    v_unit = v / nv
    X = np.atleast_2d(x_point)
    v_ray = v_unit.reshape(-1, 1)

    n_bases = int(gp_model.n_bases)
    chunk = max(1, n_bases // 2 - 1)
    out = np.zeros(d)
    for start in range(0, d, chunk):
        idxs = list(range(start, min(start + chunk, d)))
        k = len(idxs)
        rays = []
        for i in idxs:
            e_i = np.zeros(d)
            e_i[i] = 1.0
            rays.append(e_i.reshape(-1, 1))
        rays.append(v_ray)
        derivs = [[[j + 1, 1]] for j in range(k + 1)]
        _, _, full_cov = gp_model.predict(
            X, params,
            rays_predict=rays,
            calc_cov=True, return_deriv=True,
            derivs_to_predict=derivs,
            return_full_cov=True,
        )
        for j, i in enumerate(idxs):
            out[i] = float(full_cov[j + 1, k + 1])
    return out * nv


def lanczos_top_k(matvec, d, k, m=None, tol=1e-10, seed=0):
    """
    Top-k eigenpairs of a symmetric PSD operator given only as a matvec.

    Uses Lanczos with full re-orthogonalisation. Returns eigenvalues in
    descending order and the matching eigenvectors as columns.
    """
    rng = np.random.default_rng(seed)
    m = int(m if m is not None else min(3 * k + 5, d))
    m = max(m, k)
    V = np.zeros((d, m + 1))
    alpha = np.zeros(m)
    beta = np.zeros(m + 1)
    v = rng.standard_normal(d)
    v /= np.linalg.norm(v)
    V[:, 0] = v
    j_final = m
    for j in range(m):
        w = matvec(V[:, j])
        if j > 0:
            w = w - beta[j] * V[:, j - 1]
        alpha[j] = float(V[:, j] @ w)
        w = w - alpha[j] * V[:, j]
        w = w - V[:, : j + 1] @ (V[:, : j + 1].T @ w)
        beta[j + 1] = float(np.linalg.norm(w))
        if beta[j + 1] < tol:
            j_final = j + 1
            break
        V[:, j + 1] = w / beta[j + 1]
    mm = j_final
    T = (np.diag(alpha[:mm])
         + np.diag(beta[1:mm], 1)
         + np.diag(beta[1:mm], -1))
    theta, Y = np.linalg.eigh(T)
    order = np.argsort(theta)[::-1]
    k_eff = min(k, mm)
    eigvals = theta[order[:k_eff]]
    eigvecs = V[:, :mm] @ Y[:, order[:k_eff]]
    norms = np.linalg.norm(eigvecs, axis=0, keepdims=True)
    norms[norms == 0.0] = 1.0
    eigvecs = eigvecs / norms
    return eigvals, eigvecs


def far_point(gp_model):
    """Reference point far from all training data for prior-scale queries."""
    if getattr(gp_model, "normalize", False):
        x_train_orig = (gp_model.x_train * gp_model.sigmas_x) + gp_model.mus_x
    else:
        x_train_orig = gp_model.x_train
    span = np.maximum(
        np.std(x_train_orig, axis=0),
        np.maximum(np.abs(x_train_orig).max(axis=0), 1.0),
    )
    return x_train_orig.mean(axis=0) + 1e3 * span


def prior_lambda_max_gradient(gp_model, params, m=6, seed=0):
    """Top eigenvalue of K_prior_∇ via Lanczos at a far point."""
    x_far = far_point(gp_model)
    d = x_far.size

    def matvec(v):
        return gradient_cov_matvec(gp_model, params, x_far, v)

    eigvals, _ = lanczos_top_k(matvec, d=d, k=1, m=max(m, 2), seed=seed)
    return float(eigvals[0])


def prior_function_variance(gp_model, params):
    """Scalar Var[f | prior] evaluated far from all data."""
    x_far = far_point(gp_model)
    _, var = gp_model.predict(
        np.atleast_2d(x_far), params,
        calc_cov=True, return_deriv=False)
    return float(var[0])


def second_order_variance_in_direction(gp_model, params, x_point, direction):
    """Posterior variance of d²f/dv² at x_point along unit direction v."""
    v = np.asarray(direction, dtype=float).reshape(-1)
    v = v / np.linalg.norm(v)
    ray = _make_ray(v, 1)
    _, var = gp_model.predict(
        np.atleast_2d(x_point), params,
        rays_predict=[ray],
        calc_cov=True,
        return_deriv=True,
        derivs_to_predict=[[[1, 2]]],
    )
    return float(var[1][0])


def prior_lambda_max_hessian(gp_model, params):
    """Conservative prior scale for pure second directional derivatives."""
    x_far = far_point(gp_model)
    d = x_far.size
    best = 0.0
    for i in range(d):
        e_i = np.zeros(d)
        e_i[i] = 1.0
        v = second_order_variance_in_direction(gp_model, params, x_far, e_i)
        if v > best:
            best = float(v)
    return best


def select_eigenpairs(gp_model, params, x_point, k, m=None, seed=0):
    """
    Top-k eigenpairs of Cov[∇f(x_point) | data] via Lanczos matvecs.
    """
    x_point = np.asarray(x_point).reshape(-1)
    d = x_point.size
    k = min(int(k), d)

    def matvec(v):
        return gradient_cov_matvec(gp_model, params, x_point, v)

    return lanczos_top_k(matvec, d=d, k=k, m=m, seed=seed)
