"""Global variance reduction (GVR) acquisition for GDDEGP models.

The criterion is the reduction in the *integrated* posterior variance of the
function value over an integration set ``Z``::

    GVR = (1/M) sum_z  b(z)^T S^-1 b(z)

    b(z) = Cov_post(c, f(z))          # (m,)   candidate-to-integration coupling
    S    = Cov_post(c, c) + sigma_n^2 I   # (m, m) candidate self-covariance

where ``c`` is the set of ``m`` new observations under consideration. Both
quantities are posterior covariances of the *current* model, so no fantasy model
has to be rebuilt: for fixed hyperparameters the variance reduction produced by a
new observation does not depend on the value that observation takes. This is the
rank-``m`` generalisation of the rank-1 identity already used by
``jetgp.full_degp.acquisition_funcs.imse_reduction``.

Three candidate kinds are supported:

    A ("d")   a directional derivative  d_v f(x_i)  at an existing design site
    B ("f")   a function value          f(x_new)    at a new site
    C ("fd")  both                      f(x_new) and d_v f(x_new) at a new site

Because ``d_v f = v^T grad f``, the direction enters ``b`` linearly and ``S``
quadratically, so at a fixed site the optimal ``v`` is the leading generalized
eigenvector of a Rayleigh quotient -- no search over directions is needed.
"""

import numpy as np
from scipy.linalg import eigh

from posterior_queries import _make_ray


# Relative jitter added to the denominator matrix of the generalized
# eigenproblem so that scipy's Cholesky-based solver stays well posed.
JITTER = 1e-10


def noise_variance(gp_model, params):
    """Observation-noise variance in the units of ``predict(return_full_cov=True)``.

    ``predict`` adds ``(10 ** sigma_n) ** 2`` to the normalised training kernel
    and returns ``f_cov_full = f_cov * sigma_y ** 2``, so the noise has to be
    scaled the same way to be commensurate with the returned blocks.
    """
    sigma_y = float(np.ravel(getattr(gp_model, "sigma_y", 1.0))[0])
    return (10.0 ** float(params[-1])) ** 2 * sigma_y ** 2


def joint_blocks(gp_model, params, x_c, Z):
    """Posterior blocks coupling one candidate site to the integration set.

    Returns ``(s_ff, b_f, s_fg, g, Sigma_g)``:

        s_ff    float      Var_post(f(x_c))
        b_f     (M,)       Cov_post(f(x_c), f(z))
        s_fg    (d,)       Cov_post(f(x_c), grad f(x_c))
        g       (d, M)     Cov_post(grad f(x_c), f(z))
        Sigma_g (d, d)     Cov_post(grad f(x_c), grad f(x_c))

    All in the original (unnormalised) units: ``normalize_directions_2`` divides
    the rays by ``sigmas_x`` while ``d/dx~_i = sigma_i d/dx_i``, so the two
    cancel and the derivative blocks are true real-space derivatives -- which is
    why ``transform_cov_directional`` applies only ``sigma_y ** 2``.
    """
    X_test = np.vstack([np.atleast_2d(np.asarray(x_c, dtype=float)), Z])
    n, d = X_test.shape
    eye = np.eye(d)
    rays = [_make_ray(eye[i], n) for i in range(d)]
    derivs = [[[i + 1, 1]] for i in range(d)]

    _, _, cov = gp_model.predict(
        X_test, params,
        rays_predict=rays,
        calc_cov=True,
        return_deriv=True,
        derivs_to_predict=derivs,
        return_full_cov=True,
    )

    # Block-major layout: row index = block * n + point; block 0 is f and block
    # i + 1 is d/dx_i. The candidate is point 0, the integration set is 1..n-1.
    zs = slice(1, n)
    grad_rows = [(i + 1) * n for i in range(d)]

    s_ff = float(cov[0, 0])
    b_f = np.asarray(cov[0, zs], dtype=float).reshape(-1)
    s_fg = np.array([cov[0, r] for r in grad_rows], dtype=float)
    g = np.array([np.asarray(cov[r, zs], dtype=float).reshape(-1)
                  for r in grad_rows])
    Sigma_g = np.array([[cov[r, c] for c in grad_rows] for r in grad_rows],
                       dtype=float)
    return s_ff, b_f, s_fg, g, Sigma_g


def generalized_spectrum(G, Sigma):
    """Eigenpairs of ``G v = lambda Sigma v``, descending, columns unit-norm.

    Returns ``(vals, vecs)`` or ``(None, None)`` for degenerate inputs, so
    callers can treat a failed candidate as worthless rather than catch errors.
    """
    d = G.shape[0]
    G = 0.5 * (G + G.T)
    Sigma = 0.5 * (Sigma + Sigma.T)
    if not (np.all(np.isfinite(G)) and np.all(np.isfinite(Sigma))):
        return None, None
    scale = max(float(np.trace(Sigma)) / d, 1e-300)
    Sigma = Sigma + JITTER * scale * np.eye(d)
    try:
        vals, vecs = eigh(G, Sigma)
    except (np.linalg.LinAlgError, ValueError):
        return None, None
    order = np.argsort(vals)[::-1]
    vals = np.asarray(vals, dtype=float)[order]
    vecs = np.asarray(vecs, dtype=float)[:, order]
    norms = np.linalg.norm(vecs, axis=0)
    norms[norms == 0.0] = 1.0
    return vals, vecs / norms


def best_direction(G, Sigma, exclude=(), cos_tol=0.999):
    """Top eigenpair of ``(G, Sigma)`` skipping directions already observed.

    ``exclude`` holds unit vectors; an eigenvector counted as parallel to one of
    them (|cos| > ``cos_tol``) is passed over in favour of the next one down.
    Returns ``(lambda, v_unit)``, or ``(0.0, None)`` when nothing is usable.
    """
    vals, vecs = generalized_spectrum(G, Sigma)
    if vals is None:
        return 0.0, None
    exclude = [np.asarray(u, dtype=float).reshape(-1) for u in exclude]
    for k in range(vals.size):
        lam = float(vals[k])
        if not np.isfinite(lam) or lam <= 0.0:
            continue
        v = vecs[:, k]
        if any(abs(float(u @ v)) > cos_tol for u in exclude):
            continue
        return lam, v
    return 0.0, None


def gvr_derivative_at_site(gp_model, params, x_i, Z, noise_var, exclude=(),
                           cos_tol=0.999):
    """Option A: best derivative-only GVR at an existing site. ``(gvr, v)``."""
    _, _, _, g, Sigma_g = joint_blocks(gp_model, params, x_i, Z)
    G = (g @ g.T) / g.shape[1]
    Sigma = Sigma_g + noise_var * np.eye(Sigma_g.shape[0])
    return best_direction(G, Sigma, exclude=exclude, cos_tol=cos_tol)


def gvr_at_new_site(gp_model, params, x_new, Z, noise_var):
    """Options B and C at a new site. Returns ``(gvr_b, gvr_c, v_c)``.

    C is obtained by conditioning the gradient block on the new function
    observation and then applying the option-A Rayleigh quotient to what is
    left, so ``gvr_c >= gvr_b`` holds by construction.
    """
    s_ff, b_f, s_fg, g, Sigma_g = joint_blocks(gp_model, params, x_new, Z)
    d = g.shape[0]
    denom_f = s_ff + noise_var
    if not np.isfinite(denom_f) or denom_f <= 0.0:
        return 0.0, 0.0, None

    gvr_b = float(np.mean(b_f ** 2) / denom_f)
    if not np.isfinite(gvr_b) or gvr_b < 0.0:
        return 0.0, 0.0, None

    g_cond = g - np.outer(s_fg, b_f) / denom_f
    Sigma_cond = Sigma_g - np.outer(s_fg, s_fg) / denom_f
    G_cond = (g_cond @ g_cond.T) / g_cond.shape[1]
    lam, v = best_direction(G_cond, Sigma_cond + noise_var * np.eye(d))
    if v is None:
        return gvr_b, gvr_b, None
    return gvr_b, gvr_b + lam, v
