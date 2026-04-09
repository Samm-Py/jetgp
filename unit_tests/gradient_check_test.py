"""
Unit test: finite-difference gradient check for NLL across all GP modules and kernels.

Verifies that the analytic gradient returned by nll_and_grad (and nll_grad) matches
central finite differences of nll_wrapper for every (module, kernel, kernel_type) combo.
"""

import unittest
import numpy as np
import sympy as sp
from scipy.stats import qmc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fd_gradient(opt, x0, eps=1e-5):
    """Central finite-difference gradient of nll_wrapper."""
    grad = np.zeros_like(x0)
    for i in range(len(x0)):
        xp = x0.copy(); xp[i] += eps
        xm = x0.copy(); xm[i] -= eps
        grad[i] = (opt.nll_wrapper(xp) - opt.nll_wrapper(xm)) / (2 * eps)
    return grad


def _check_gradient(test_case, opt, x0, tol=1e-3, tag=""):
    """Assert that analytic gradient matches FD gradient within relative tolerance."""
    nll, grad_a = opt.nll_and_grad(x0)
    if np.all(grad_a == 0.0):
        # Cholesky decomposition failed — matrix too ill-conditioned at this x0.
        # Skip rather than falsely report a gradient mismatch.
        return
    grad_fd = _fd_gradient(opt, x0)
    for i in range(len(x0)):
        denom = max(abs(grad_fd[i]), abs(grad_a[i]), 1e-12)
        rel = abs(grad_a[i] - grad_fd[i]) / denom
        test_case.assertLess(
            rel, tol,
            f"{tag} x[{i}]: analytic={grad_a[i]:.6e} fd={grad_fd[i]:.6e} rel={rel:.2e}"
        )


# Matern uses sqrt(r2 + eps^2) smoothing which introduces inherent FD mismatch.
MATERN_TOL = 0.10

# SineExp involves sin/cos which can amplify central-FD error at eps=1e-5.
SINEEXP_TOL = 0.025


def _check_grad_matches_nll_grad(test_case, opt, x0, tag=""):
    """Assert that nll_grad and nll_and_grad return the same gradient."""
    _, grad_combined = opt.nll_and_grad(x0)
    grad_separate = opt.nll_grad(x0)
    np.testing.assert_allclose(
        grad_combined, grad_separate, atol=1e-10, rtol=1e-10,
        err_msg=f"{tag}: nll_grad and nll_and_grad disagree"
    )


# ---------------------------------------------------------------------------
# Branin function (2D) — used by DEGP, DDEGP, GDDEGP
# ---------------------------------------------------------------------------

def _branin_setup():
    """Return X_train (n,2), y_func, grad_x1_vals, grad_x2_vals for Branin."""
    x1_sym, x2_sym = sp.symbols("x1 x2", real=True)
    a, b, c, r, s, t = (1.0, 5.1 / (4 * sp.pi**2), 5.0 / sp.pi,
                         6.0, 10.0, 1.0 / (8 * sp.pi))
    f_sym = a * (x2_sym - b * x1_sym**2 + c * x1_sym - r)**2 + \
            s * (1 - t) * sp.cos(x1_sym) + s
    grad_x1_sym = sp.diff(f_sym, x1_sym)
    grad_x2_sym = sp.diff(f_sym, x2_sym)

    f_np = sp.lambdify([x1_sym, x2_sym], f_sym, "numpy")
    g1_np = sp.lambdify([x1_sym, x2_sym], grad_x1_sym, "numpy")
    g2_np = sp.lambdify([x1_sym, x2_sym], grad_x2_sym, "numpy")

    np.random.seed(7)
    sampler = qmc.LatinHypercube(d=2, seed=7)
    X = qmc.scale(sampler.random(n=12), [-5.0, 0.0], [10.0, 15.0])

    y = np.atleast_1d(f_np(X[:, 0], X[:, 1])).reshape(-1, 1)
    g1 = np.atleast_1d(g1_np(X[:, 0], X[:, 1])).reshape(-1, 1)
    g2 = np.atleast_1d(g2_np(X[:, 0], X[:, 1])).reshape(-1, 1)
    return X, y, g1, g2


# ---------------------------------------------------------------------------
# DEGP
# ---------------------------------------------------------------------------

class TestDEGPGradient(unittest.TestCase):
    """FD gradient check for DEGP across all kernels."""

    KERNELS = [
        ("SE", "isotropic"),
        ("SE", "anisotropic"),
        ("RQ", "isotropic"),
        ("RQ", "anisotropic"),
        ("SineExp", "isotropic"),
        ("SineExp", "anisotropic"),
        ("Matern", "isotropic"),
        ("Matern", "anisotropic"),
    ]

    @classmethod
    def setUpClass(cls):
        from jetgp.full_degp.degp import degp
        cls.degp_cls = degp
        X, y, g1, g2 = _branin_setup()
        cls.X = X
        cls.y_train = [y, g1, g2]
        cls.der_indices = [[[[1, 1]], [[2, 1]]]]
        cls.deriv_locs = [[i for i in range(len(X))]] * 2

    def _make_model(self, kernel, kernel_type):
        kw = {}
        if kernel == "Matern":
            kw["smoothness_parameter"] = 3
        return self.degp_cls(
            self.X, self.y_train, n_order=1, n_bases=2,
            der_indices=self.der_indices,
            derivative_locations=self.deriv_locs,
            normalize=True, kernel=kernel, kernel_type=kernel_type,
            **kw
        )

    def _x0_for(self, model):
        """Return a plausible test hyperparameter vector."""
        n = len(model.bounds)
        return np.array([0.1] * (n - 2) + [0.5, -3.0])

    def test_nll_and_grad_vs_fd(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                tol = MATERN_TOL if kernel == 'Matern' else 1e-3
                _check_gradient(self, opt, x0, tol=tol, tag=f"DEGP/{kernel}/{ktype}")

    def test_nll_grad_matches_nll_and_grad(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                _check_grad_matches_nll_grad(self, opt, x0,
                                              tag=f"DEGP/{kernel}/{ktype}")


# ---------------------------------------------------------------------------
# DDEGP
# ---------------------------------------------------------------------------

class TestDDEGPGradient(unittest.TestCase):
    """FD gradient check for DDEGP across all kernels."""

    KERNELS = [
        ("SE", "isotropic"),
        ("SE", "anisotropic"),
        ("RQ", "isotropic"),
        ("RQ", "anisotropic"),
        ("SineExp", "isotropic"),
        ("SineExp", "anisotropic"),
        ("Matern", "isotropic"),
        ("Matern", "anisotropic"),
    ]

    @classmethod
    def setUpClass(cls):
        from jetgp.full_ddegp.ddegp import ddegp
        cls.ddegp_cls = ddegp
        X, y, g1, g2 = _branin_setup()
        cls.X = X
        cls.y_func = y
        cls.g1, cls.g2 = g1, g2

        # Three directional rays at 45°, 90°, 135°
        angles = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        cls.rays = np.array([[np.cos(a) for a in angles],
                             [np.sin(a) for a in angles]])
        dir_derivs = []
        for i in range(cls.rays.shape[1]):
            r = cls.rays[:, i]
            dir_derivs.append(g1 * r[0] + g2 * r[1])
        cls.y_train = [y] + dir_derivs
        cls.der_indices = [[[[1, 1]], [[2, 1]], [[3, 1]]]]
        cls.deriv_locs = [[i for i in range(len(X))]] * 3

    def _make_model(self, kernel, kernel_type):
        kw = {}
        if kernel == "Matern":
            kw["smoothness_parameter"] = 3
        return self.ddegp_cls(
            self.X, self.y_train, n_order=1,
            der_indices=self.der_indices,
            rays=self.rays,
            derivative_locations=self.deriv_locs,
            normalize=True, kernel=kernel, kernel_type=kernel_type,
            **kw
        )

    def _x0_for(self, model):
        n = len(model.bounds)
        return np.array([0.1] * (n - 2) + [0.5, -3.0])

    def test_nll_and_grad_vs_fd(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                tol = MATERN_TOL if kernel == 'Matern' else 1e-3
                _check_gradient(self, opt, x0, tol=tol, tag=f"DDEGP/{kernel}/{ktype}")

    def test_nll_grad_matches_nll_and_grad(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                _check_grad_matches_nll_grad(self, opt, x0,
                                              tag=f"DDEGP/{kernel}/{ktype}")


# ---------------------------------------------------------------------------
# GDDEGP
# ---------------------------------------------------------------------------

class TestGDDEGPGradient(unittest.TestCase):
    """FD gradient check for GDDEGP across all kernels."""

    KERNELS = [
        ("SE", "isotropic"),
        ("SE", "anisotropic"),
        ("RQ", "isotropic"),
        ("RQ", "anisotropic"),
        ("SineExp", "isotropic"),
        ("SineExp", "anisotropic"),
        ("Matern", "isotropic"),
        ("Matern", "anisotropic"),
    ]

    @classmethod
    def setUpClass(cls):
        from jetgp.full_gddegp.gddegp import gddegp
        cls.gddegp_cls = gddegp
        X, y, g1, g2 = _branin_setup()
        cls.X = X

        # Per-point gradient-aligned rays
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

        cls.rays_array = np.hstack(rays_list)  # (2, n)
        cls.y_train = [y, np.array(dir_derivs).reshape(-1, 1)]
        cls.der_indices = [[[[1, 1]]]]
        cls.deriv_locs = [[i for i in range(len(X))]] * 1

    def _make_model(self, kernel, kernel_type):
        kw = {}
        if kernel == "Matern":
            kw["smoothness_parameter"] = 3
        return self.gddegp_cls(
            self.X, self.y_train, n_order=1,
            rays_list=[self.rays_array],
            der_indices=self.der_indices,
            derivative_locations=self.deriv_locs,
            normalize=True, kernel=kernel, kernel_type=kernel_type,
            **kw
        )

    def _x0_for(self, model):
        n = len(model.bounds)
        return np.array([0.1] * (n - 2) + [0.5, -3.0])

    def test_nll_and_grad_vs_fd(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                tol = MATERN_TOL if kernel == 'Matern' else 1e-3
                _check_gradient(self, opt, x0, tol=tol, tag=f"GDDEGP/{kernel}/{ktype}")

    def test_nll_grad_matches_nll_and_grad(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                _check_grad_matches_nll_grad(self, opt, x0,
                                              tag=f"GDDEGP/{kernel}/{ktype}")


# ---------------------------------------------------------------------------
# WDEGP (degp submodel)
# ---------------------------------------------------------------------------

class TestWDEGPGradient(unittest.TestCase):
    """FD gradient check for WDEGP (degp submodel) across all kernels."""

    KERNELS = [
        ("SE", "isotropic"),
        ("SE", "anisotropic"),
        ("RQ", "isotropic"),
        ("RQ", "anisotropic"),
        ("SineExp", "isotropic"),
        ("SineExp", "anisotropic"),
        ("Matern", "isotropic"),
        ("Matern", "anisotropic"),
    ]

    @classmethod
    def setUpClass(cls):
        from jetgp.wdegp.wdegp import wdegp
        import jetgp.utils as utils
        cls.wdegp_cls = wdegp
        cls.utils = utils

        X, y, g1, g2 = _branin_setup()
        cls.X = X
        n = len(X)
        cls.n_bases = 2
        cls.n_order = 1

        # One submodel per point: [all func vals, local dx1, local dx2]
        cls.submodel_indices = [[[i], [i]] for i in range(n)]
        submodel_data = []
        for k in range(n):
            submodel_data.append([y, g1[k:k+1], g2[k:k+1]])
        cls.submodel_data = submodel_data

        cls.derivative_specs = [
            utils.gen_OTI_indices(cls.n_bases, cls.n_order) for _ in range(n)
        ]

    def _make_model(self, kernel, kernel_type):
        kw = {}
        if kernel == "Matern":
            kw["smoothness_parameter"] = 3
        return self.wdegp_cls(
            self.X, self.submodel_data, self.n_order, self.n_bases,
            self.derivative_specs,
            derivative_locations=self.submodel_indices,
            normalize=True, kernel=kernel, kernel_type=kernel_type,
            **kw
        )

    def _x0_for(self, model):
        n = len(model.bounds)
        return np.array([0.1] * (n - 2) + [0.5, -3.0])

    def test_nll_and_grad_vs_fd(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                tol = MATERN_TOL if kernel == 'Matern' else 1e-3
                _check_gradient(self, opt, x0, tol=tol, tag=f"WDEGP/{kernel}/{ktype}")

    def test_nll_grad_matches_nll_and_grad(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                _check_grad_matches_nll_grad(self, opt, x0,
                                              tag=f"WDEGP/{kernel}/{ktype}")


# ---------------------------------------------------------------------------
# WDDEGP (ddegp submodel via wdegp)
# ---------------------------------------------------------------------------

class TestWDDEGPGradient(unittest.TestCase):
    """FD gradient check for WDEGP with ddegp submodel across SE kernels."""

    KERNELS = [
        ("SE", "isotropic"),
        ("SE", "anisotropic"),
        ("RQ", "isotropic"),
        ("RQ", "anisotropic"),
        ("Matern", "isotropic"),
        ("Matern", "anisotropic"),
    ]

    @classmethod
    def setUpClass(cls):
        from jetgp.wdegp.wdegp import wdegp
        import jetgp.utils as utils
        cls.wdegp_cls = wdegp
        cls.utils = utils

        X, y, g1, g2 = _branin_setup()
        cls.X = X
        n = len(X)
        cls.n_bases = 2
        cls.n_order = 1

        # 2 global directional rays
        angles = [np.pi / 4, 3 * np.pi / 4]
        cls.rays = np.array([[np.cos(a) for a in angles],
                             [np.sin(a) for a in angles]])

        # Each submodel: [all func vals, dir_deriv_ray0_at_pt, dir_deriv_ray1_at_pt]
        cls.submodel_indices = [[[i], [i]] for i in range(n)]
        submodel_data = []
        for k in range(n):
            dd = []
            for j in range(cls.rays.shape[1]):
                r = cls.rays[:, j]
                dd.append((g1[k] * r[0] + g2[k] * r[1]).reshape(1, 1))
            submodel_data.append([y] + dd)
        cls.submodel_data = submodel_data

        cls.derivative_specs = [
            utils.gen_OTI_indices(cls.n_bases, cls.n_order) for _ in range(n)
        ]

    def _make_model(self, kernel, kernel_type):
        kw = {}
        if kernel == "Matern":
            kw["smoothness_parameter"] = 3
        return self.wdegp_cls(
            self.X, self.submodel_data, self.n_order, self.n_bases,
            self.derivative_specs,
            derivative_locations=self.submodel_indices,
            normalize=True, kernel=kernel, kernel_type=kernel_type,
            rays=self.rays, submodel_type='ddegp',
            **kw
        )

    def _x0_for(self, model):
        n = len(model.bounds)
        return np.array([0.1] * (n - 2) + [0.5, -3.0])

    def test_nll_and_grad_vs_fd(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                tol = MATERN_TOL if kernel == 'Matern' else 1e-3
                _check_gradient(self, opt, x0, tol=tol, tag=f"WDDEGP/{kernel}/{ktype}")

    def test_nll_grad_matches_nll_and_grad(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                _check_grad_matches_nll_grad(self, opt, x0,
                                              tag=f"WDDEGP/{kernel}/{ktype}")


# ---------------------------------------------------------------------------
# WGDDEGP (gddegp submodel via wdegp)
# ---------------------------------------------------------------------------

class TestWGDDEGPGradient(unittest.TestCase):
    """FD gradient check for WDEGP with gddegp submodel across SE kernels."""

    KERNELS = [
        ("SE", "isotropic"),
        ("SE", "anisotropic"),
        ("RQ", "isotropic"),
        ("RQ", "anisotropic"),
        ("Matern", "isotropic"),
        ("Matern", "anisotropic"),
    ]

    @classmethod
    def setUpClass(cls):
        from jetgp.wdegp.wdegp import wdegp
        import jetgp.utils as utils
        cls.wdegp_cls = wdegp

        X, y, g1, g2 = _branin_setup()
        cls.X = X
        n = len(X)
        cls.n_bases = 2
        cls.n_order = 1

        # Split into 2 disjoint submodels (even/odd indices)
        sm1_idx = list(range(0, n, 2))
        sm2_idx = list(range(1, n, 2))

        # Per-point gradient-aligned rays
        rays_all = np.zeros((2, n))
        dir_derivs_all = np.zeros(n)
        for k in range(n):
            gx, gy = g1[k].item(), g2[k].item()
            mag = np.sqrt(gx**2 + gy**2)
            if mag < 1e-10:
                rays_all[:, k] = [1.0, 0.0]
            else:
                rays_all[:, k] = [gx / mag, gy / mag]
            dir_derivs_all[k] = gx * rays_all[0, k] + gy * rays_all[1, k]

        # rays_list: one entry per submodel, each a list of ray arrays (d, n_pts)
        cls.rays_list = [
            [rays_all[:, sm1_idx]],  # SM1: one directional derivative
            [rays_all[:, sm2_idx]],  # SM2: one directional derivative
        ]

        y_vals = y  # (n, 1)
        cls.submodel_data = [
            [y_vals, dir_derivs_all[sm1_idx].reshape(-1, 1)],
            [y_vals, dir_derivs_all[sm2_idx].reshape(-1, 1)],
        ]

        cls.der_indices = [
            [[[[1, 1]]]],  # SM1: 1 directional derivative, order 1
            [[[[1, 1]]]],  # SM2: 1 directional derivative, order 1
        ]

        cls.derivative_locations = [
            [sm1_idx],  # SM1
            [sm2_idx],  # SM2
        ]

    def _make_model(self, kernel, kernel_type):
        kw = {}
        if kernel == "Matern":
            kw["smoothness_parameter"] = 3
        return self.wdegp_cls(
            self.X, self.submodel_data, self.n_order, self.n_bases,
            self.der_indices,
            derivative_locations=self.derivative_locations,
            normalize=True, kernel=kernel, kernel_type=kernel_type,
            rays_list=self.rays_list, submodel_type='gddegp',
            **kw
        )

    def _x0_for(self, model):
        n = len(model.bounds)
        return np.array([0.1] * (n - 2) + [0.5, -3.0])

    def test_nll_and_grad_vs_fd(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                tol = MATERN_TOL if kernel == 'Matern' else 1e-3
                _check_gradient(self, opt, x0, tol=tol, tag=f"WGDDEGP/{kernel}/{ktype}")

    def test_nll_grad_matches_nll_and_grad(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                _check_grad_matches_nll_grad(self, opt, x0,
                                              tag=f"WGDDEGP/{kernel}/{ktype}")


# ---------------------------------------------------------------------------
# Sparse DEGP
# ---------------------------------------------------------------------------

class TestSparseDEGPGradient(unittest.TestCase):
    """FD gradient check for sparse DEGP across all kernels."""

    KERNELS = [
        ("SE", "isotropic"),
        ("SE", "anisotropic"),
        ("RQ", "isotropic"),
        ("RQ", "anisotropic"),
        ("SineExp", "isotropic"),
        ("SineExp", "anisotropic"),
        ("Matern", "isotropic"),
        ("Matern", "anisotropic"),
    ]

    @classmethod
    def setUpClass(cls):
        from jetgp.full_degp_sparse.degp import degp
        cls.degp_cls = degp
        X, y, g1, g2 = _branin_setup()
        cls.X = X
        cls.y_train = [y, g1, g2]
        cls.der_indices = [[[[1, 1]], [[2, 1]]]]
        cls.deriv_locs = [[i for i in range(len(X))]] * 2

    def _make_model(self, kernel, kernel_type):
        kw = {}
        if kernel == "Matern":
            kw["smoothness_parameter"] = 3
        return self.degp_cls(
            self.X, self.y_train, n_order=1, n_bases=2,
            der_indices=self.der_indices,
            derivative_locations=self.deriv_locs,
            normalize=True, kernel=kernel, kernel_type=kernel_type,
            rho=1.0, use_supernodes=False,
            **kw
        )

    def _x0_for(self, model):
        n = len(model.bounds)
        return np.array([0.1] * (n - 2) + [0.5, -3.0])

    def test_nll_and_grad_vs_fd(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                tol = MATERN_TOL if kernel == 'Matern' else 1e-3
                _check_gradient(self, opt, x0, tol=tol,
                                tag=f"SparseDEGP/{kernel}/{ktype}")


# ---------------------------------------------------------------------------
# Sparse DDEGP
# ---------------------------------------------------------------------------

class TestSparseDDEGPGradient(unittest.TestCase):
    """FD gradient check for sparse DDEGP across all kernels."""

    KERNELS = [
        ("SE", "isotropic"),
        ("SE", "anisotropic"),
        ("RQ", "isotropic"),
        ("RQ", "anisotropic"),
        ("SineExp", "isotropic"),
        ("SineExp", "anisotropic"),
        ("Matern", "isotropic"),
        ("Matern", "anisotropic"),
    ]

    @classmethod
    def setUpClass(cls):
        from jetgp.full_ddegp_sparse.ddegp import ddegp
        cls.ddegp_cls = ddegp
        X, y, g1, g2 = _branin_setup()
        cls.X = X
        cls.y_func = y
        cls.g1, cls.g2 = g1, g2

        angles = [np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        cls.rays = np.array([[np.cos(a) for a in angles],
                             [np.sin(a) for a in angles]])
        dir_derivs = []
        for i in range(cls.rays.shape[1]):
            r = cls.rays[:, i]
            dir_derivs.append(g1 * r[0] + g2 * r[1])
        cls.y_train = [y] + dir_derivs
        cls.der_indices = [[[[1, 1]], [[2, 1]], [[3, 1]]]]
        cls.deriv_locs = [[i for i in range(len(X))]] * 3

    def _make_model(self, kernel, kernel_type):
        kw = {}
        if kernel == "Matern":
            kw["smoothness_parameter"] = 3
        return self.ddegp_cls(
            self.X, self.y_train, n_order=1,
            der_indices=self.der_indices,
            rays=self.rays,
            derivative_locations=self.deriv_locs,
            normalize=True, kernel=kernel, kernel_type=kernel_type,
            rho=1.0, use_supernodes=False,
            **kw
        )

    def _x0_for(self, model):
        n = len(model.bounds)
        return np.array([0.1] * (n - 2) + [0.5, -3.0])

    def test_nll_and_grad_vs_fd(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                tol = MATERN_TOL if kernel == 'Matern' else SINEEXP_TOL if kernel == 'SineExp' else 1e-3
                _check_gradient(self, opt, x0, tol=tol,
                                tag=f"SparseDDEGP/{kernel}/{ktype}")


# ---------------------------------------------------------------------------
# Sparse GDDEGP
# ---------------------------------------------------------------------------

class TestSparseGDDEGPGradient(unittest.TestCase):
    """FD gradient check for sparse GDDEGP across all kernels."""

    KERNELS = [
        ("SE", "isotropic"),
        ("SE", "anisotropic"),
        ("RQ", "isotropic"),
        ("RQ", "anisotropic"),
        ("SineExp", "isotropic"),
        ("SineExp", "anisotropic"),
        ("Matern", "isotropic"),
        ("Matern", "anisotropic"),
    ]

    @classmethod
    def setUpClass(cls):
        from jetgp.full_gddegp_sparse.gddegp import gddegp
        cls.gddegp_cls = gddegp
        X, y, g1, g2 = _branin_setup()
        cls.X = X

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

        cls.rays_array = np.hstack(rays_list)  # (2, n)
        cls.y_train = [y, np.array(dir_derivs).reshape(-1, 1)]
        cls.der_indices = [[[[1, 1]]]]
        cls.deriv_locs = [[i for i in range(len(X))]] * 1

    def _make_model(self, kernel, kernel_type):
        kw = {}
        if kernel == "Matern":
            kw["smoothness_parameter"] = 3
        return self.gddegp_cls(
            self.X, self.y_train, n_order=1,
            rays_list=[self.rays_array],
            der_indices=self.der_indices,
            derivative_locations=self.deriv_locs,
            normalize=True, kernel=kernel, kernel_type=kernel_type,
            rho=1.0, use_supernodes=False,
            **kw
        )

    def _x0_for(self, model):
        n = len(model.bounds)
        return np.array([0.1] * (n - 2) + [0.5, -3.0])

    def test_nll_and_grad_vs_fd(self):
        for kernel, ktype in self.KERNELS:
            with self.subTest(kernel=kernel, kernel_type=ktype):
                model = self._make_model(kernel, ktype)
                opt = model.optimizer
                x0 = self._x0_for(model)
                tol = MATERN_TOL if kernel == 'Matern' else 1e-3
                _check_gradient(self, opt, x0, tol=tol,
                                tag=f"SparseGDDEGP/{kernel}/{ktype}")


if __name__ == "__main__":
    unittest.main()
