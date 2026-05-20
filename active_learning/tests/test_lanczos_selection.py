import numpy as np

from lanczos_selection import gradient_cov_matvec, lanczos_top_k


def test_lanczos_top_k_matches_diagonal_matrix():
    diagonal = np.array([5.0, 3.0, 1.0, 0.25])

    def matvec(v):
        return diagonal * v

    eigvals, eigvecs = lanczos_top_k(
        matvec, d=diagonal.size, k=2, m=4, seed=7)

    assert np.allclose(eigvals, [5.0, 3.0], atol=1e-10)

    residual = np.diag(diagonal) @ eigvecs - eigvecs * eigvals
    assert np.linalg.norm(residual) < 1e-9


class FakeGradientCovGP:
    def __init__(self, cov, n_bases=6):
        self.cov = np.asarray(cov, dtype=float)
        self.n_bases = n_bases

    def predict(self, X, params, rays_predict, calc_cov, return_deriv,
                derivs_to_predict, return_full_cov):
        n_rays = len(rays_predict)
        full_cov = np.zeros((n_rays + 1, n_rays + 1))
        for i, ray_i in enumerate(rays_predict):
            vi = ray_i[:, 0]
            for j, ray_j in enumerate(rays_predict):
                vj = ray_j[:, 0]
                full_cov[i + 1, j + 1] = vi @ self.cov @ vj
        return None, None, full_cov


def test_gradient_cov_matvec_matches_dense_product():
    cov = np.array([
        [3.0, 0.5, -0.25],
        [0.5, 2.0, 0.75],
        [-0.25, 0.75, 1.5],
    ])
    gp = FakeGradientCovGP(cov, n_bases=6)
    v = np.array([0.4, -1.2, 2.0])

    result = gradient_cov_matvec(gp, params=np.array([]), x_point=np.zeros(3), v=v)

    assert np.allclose(result, cov @ v)
