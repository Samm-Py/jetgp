"""Matérn covariances of derivative data at coincident points.

JetGP once evaluated the Matérn kernel at r = sqrt(r^2 + eps^2); at r = 0 the OTI coefficients of
that square root grow like eps^(1-2k) and cancel in floating point only through second order, so
derivative data of order two and higher received corrupted diagonal blocks (Var f'' of order 1e15
for nu = 9/2). jetgp.utils.matern_from_sqdist evaluates coincident points by the even Taylor series
of the kernel in r^2 instead. Run with: python -m pytest tests/test_matern_coincident.py
"""
import math
import warnings

import numpy as np
import pytest
import sympy as sp

warnings.filterwarnings("ignore")
import jetgp.full_degp.optimizer as optimizer_module  # noqa: E402
from jetgp.full_degp.degp import degp  # noqa: E402


def exact_variances(nu, p):
    """Var of f^(j), j = 0..p, for a unit-variance, unit-length-scale 1-D Matérn process."""
    r = sp.symbols("r")
    nu = sp.Rational(int(round(2 * nu)), 2)
    z = sp.sqrt(2 * nu) * r
    k = sp.simplify(2 ** (1 - nu) / sp.gamma(nu) * z**nu * sp.besselk(nu, z))
    series = sp.series(k, r, 0, 2 * p + 1).removeO()
    return [float((-1) ** j * series.coeff(r, 2 * j) * math.factorial(2 * j)) for j in range(p + 1)]


def assembled_covariance(model, x0):
    """The covariance matrix the likelihood assembles at hyperparameters x0."""
    captured = {}
    fast, slow = optimizer_module.utils.rbf_kernel_fast, optimizer_module.utils.rbf_kernel

    def capture(function):
        def wrapped(*args, **kwargs):
            K = function(*args, **kwargs)
            captured["K"] = K.copy()
            return K
        return wrapped

    optimizer_module.utils.rbf_kernel_fast, optimizer_module.utils.rbf_kernel = capture(fast), capture(slow)
    try:
        model.optimizer.negative_log_marginal_likelihood(np.asarray(x0, dtype=float))
    finally:
        optimizer_module.utils.rbf_kernel_fast, optimizer_module.utils.rbf_kernel = fast, slow
    return captured["K"]


@pytest.mark.parametrize("p, smoothness", [(1, 1), (1, 3), (2, 2), (2, 4), (3, 3), (3, 5)])
def test_coincident_derivative_variances_are_exact(p, smoothness):
    y = [np.array([[1.0]])] + [np.array([[0.1 * (j + 1)]]) for j in range(p)]
    model = degp(np.array([[0.5]]), y, n_order=p, n_bases=1, der_indices=[[[[1, j]]] for j in range(1, p + 1)],
                 derivative_locations=[[0]] * p, normalize=False, kernel="Matern", kernel_type="anisotropic",
                 smoothness_parameter=smoothness)
    K = assembled_covariance(model, [0.0, 0.0, -16.0])
    np.testing.assert_allclose(np.diag(K), exact_variances(smoothness + 0.5, p), rtol=1e-12)


def test_likelihood_gradient_matches_finite_differences():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, (6, 2))
    f = lambda x: np.exp(-((x - 0.4) ** 2).sum(1))
    g = lambda x: -2 * (x - 0.4) * f(x)[:, None]
    hess = lambda x: np.stack([(4 * (x[:, 0] - 0.4) ** 2 - 2) * f(x), 4 * (x[:, 0] - 0.4) * (x[:, 1] - 0.4) * f(x),
                               (4 * (x[:, 1] - 0.4) ** 2 - 2) * f(x)], 1)
    y = [f(x)[:, None], g(x)[:, :1], g(x)[:, 1:], hess(x)[:, :1], hess(x)[:, 1:2], hess(x)[:, 2:]]
    der = [[[[1, 1]], [[2, 1]]], [[[1, 2]], [[1, 1], [2, 1]], [[2, 2]]]]
    model = degp(x, y, n_order=2, n_bases=2, der_indices=der, derivative_locations=[list(range(6))] * 5,
                 normalize=True, kernel="Matern", kernel_type="anisotropic", smoothness_parameter=4)
    x0 = np.array([-0.3, -0.2, 0.1, -3.0])
    analytic = model.optimizer.nll_grad(x0)
    h = 1e-5
    nll = model.optimizer.negative_log_marginal_likelihood
    central = np.array([(nll(x0 + h * e) - nll(x0 - h * e)) / (2 * h) for e in np.eye(len(x0))])
    np.testing.assert_allclose(analytic, central, rtol=1e-5, atol=1e-6 * np.abs(central).max())
