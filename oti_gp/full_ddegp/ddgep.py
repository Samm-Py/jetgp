import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize
import pyoti.sparse as oti
import utils
from kernel_funcs.kernel_funcs import KernelFactory
from full_ddegp.optimizer import Optimizer


class ddgep:
    def __init__(
        self,
        x_train,
        y_train,
        n_order,
        n_bases,
        der_indices,
        rays,
        sigma_n=0.0,
        nugget=1e-6,
        kernel="SE",
        kernel_type="anisotropic",
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.sigma_n = sigma_n
        self.nugget = nugget
        self.n_order = n_order
        self.n_bases = n_bases
        self.rays = rays
        self.n_rays = rays.shape[1]
        self.dim = x_train.shape[1]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.der_indices = der_indices
        self.kernel_func = self.create_kernel_function()

        self.kernel_factory = KernelFactory(
            dim=self.dim,
            normalize=self.normalize,
            differences_by_dim=self.differences_by_dim
        )
        self.kernel_func = self.kernel_factory.create_kernel(
            kernel_name=self.kernel,
            kernel_type=self.kernel_type
        )
        self.bounds = self.kernel_factory.bounds
        self.optimizer = Optimizer(self)

    def se_kernel_anisotropic(self, X1, X2, length_scales, n_order, index):
        X1 = oti.array(X1)
        X2 = oti.array(X2)

        n1, d = X1.shape
        n2, d = X2.shape

        ell = np.exp(length_scales[0:-1])
        sigma_f = length_scales[-1]
        # Prepare the output: a list of d arrays, each of shape (n, m)
        differences_by_dim = []

        # Loop over each dimension k
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = oti.zeros((n1, n2))

            # Nested loops to fill diffs_k
            for i in range(n1):
                for j in range(n2):
                    dire1 = 0
                    dire2 = 0
                    for l in range(self.n_rays):
                        dire1 = (
                            dire1
                            + oti.e(2 * l + 2, order=2 * n_order)
                            * self.rays[k, l]
                        )
                        dire2 = (
                            dire2
                            + oti.e(2 * l + 1, order=2 * n_order)
                            * self.rays[k, l]
                        )
                    diffs_k[i, j] = ((X1[i, k] + dire1)) - (X2[j, k] + dire2)

            # Append to our list
            differences_by_dim.append(diffs_k)

        sqdist = 0
        for i in range(d):
            sqdist = sqdist + (ell[i] * (differences_by_dim[i])) ** 2

        return sigma_f**2 * oti.exp(-0.5 * sqdist)

    def rq_kernel_anisotropic(self, X1, X2, length_scales, n_order, index):
        X1 = oti.array(X1)
        X2 = oti.array(X2)

        n1, d = X1.shape

        n2, d = X2.shape

        ell = np.exp(length_scales[: self.dim])
        alpha = np.exp(length_scales[self.dim:])
        sigma_f = length_scales[-1]

        # Prepare the output: a list of d arrays, each of shape (n, m)
        differences_by_dim = []

        # Loop over each dimension k
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = oti.zeros((n1, n2))

            # Nested loops to fill diffs_k
            for i in range(n1):
                for j in range(n2):
                    dire1 = 0
                    dire2 = 0
                    for l in range(self.n_rays):
                        dire1 = (
                            dire1
                            + oti.e(2 * l + 2, order=2 * n_order)
                            * self.rays[k, l]
                        )
                        dire2 = (
                            dire2
                            + oti.e(2 * l + 1, order=2 * n_order)
                            * self.rays[k, l]
                        )
                    diffs_k[i, j] = ((X1[i, k] + dire1)) - (X2[j, k] + dire2)

            # Append to our list
            differences_by_dim.append(diffs_k)

            # Distances scaled by each dimension's length scale

        # Distances scaled by each dimension's length scale

        sqdist = 1
        for i in range(d):
            sqdist *= (
                1 + (ell[i] * differences_by_dim[i]) ** 2 / (2 * alpha[i])
            ) ** (-alpha[i])

        return sigma_f**2 * sqdist

    def sine_exp_kernel_anisotropic(
        self, X1, X2, length_scales, n_order, index
    ):
        X1 = oti.array(X1)
        X2 = oti.array(X2)

        n1, d = X1.shape

        n2, d = X2.shape

        ell = np.exp(length_scales[: self.dim])
        p = length_scales[self.dim:]
        sigma_f = length_scales[-1]

        # Prepare the output: a list of d arrays, each of shape (n, m)
        differences_by_dim = []

        # Loop over each dimension k
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = oti.zeros((n1, n2))

            # Nested loops to fill diffs_k
            for i in range(n1):
                for j in range(n2):
                    dire1 = 0
                    dire2 = 0
                    for l in range(self.n_rays):
                        dire1 = (
                            dire1
                            + oti.e(2 * l + 2, order=2 * n_order)
                            * self.rays[k, l]
                        )
                        dire2 = (
                            dire2
                            + oti.e(2 * l + 1, order=2 * n_order)
                            * self.rays[k, l]
                        )
                    diffs_k[i, j] = ((X1[i, k] + dire1)) - (X2[j, k] + dire2)

            # Append to our list
            differences_by_dim.append(diffs_k)

        # Distances scaled by each dimension's length scale

        sqdist = np.sum(
            (ell[i] * oti.sin(np.pi / p[i] * differences_by_dim[i])) ** 2
            for i in range(d)
        )
        return sigma_f**2 * oti.exp(-2 * sqdist)

    def se_kernel_isotropic(self, X1, X2, length_scales, n_order, index):
        X1 = oti.array(X1)
        X2 = oti.array(X2)

        n1, d = X1.shape

        n2, d = X2.shape

        ell = np.exp(length_scales[0])
        sigma_f = length_scales[-1]

        # Prepare the output: a list of d arrays, each of shape (n, m)
        differences_by_dim = []

        # Loop over each dimension k
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = oti.zeros((n1, n2))

            # Nested loops to fill diffs_k
            for i in range(n1):
                for j in range(n2):
                    dire1 = 0
                    dire2 = 0
                    for l in range(self.n_rays):
                        dire1 = (
                            dire1
                            + oti.e(2 * l + 2, order=2 * n_order)
                            * self.rays[k, l]
                        )
                        dire2 = (
                            dire2
                            + oti.e(2 * l + 1, order=2 * n_order)
                            * self.rays[k, l]
                        )
                    diffs_k[i, j] = ((X1[i, k] + dire1)) - (X2[j, k] + dire2)

            # Append to our list
            differences_by_dim.append(diffs_k)

        # Distances scaled by each dimension's length scale
        sqdist = 0
        for i in range(d):
            sqdist = sqdist + (ell * (differences_by_dim[i])) ** 2

        return sigma_f**2 * oti.exp(-0.5 * sqdist)

    def rq_kernel_isotropic(self, X1, X2, length_scales, n_order, index):
        X1 = oti.array(X1)
        X2 = oti.array(X2)

        n1, d = X1.shape

        n2, d = X2.shape

        ell = np.exp(length_scales[0])
        alpha = np.exp(length_scales[1])
        sigma_f = length_scales[-1]

        # Prepare the output: a list of d arrays, each of shape (n, m)
        differences_by_dim = []

        # Loop over each dimension k
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = oti.zeros((n1, n2))

            # Nested loops to fill diffs_k
            for i in range(n1):
                for j in range(n2):
                    dire1 = 0
                    dire2 = 0
                    for l in range(self.n_rays):
                        dire1 = (
                            dire1
                            + oti.e(2 * l + 2, order=2 * n_order)
                            * self.rays[k, l]
                        )
                        dire2 = (
                            dire2
                            + oti.e(2 * l + 1, order=2 * n_order)
                            * self.rays[k, l]
                        )
                    diffs_k[i, j] = ((X1[i, k] + dire1)) - (X2[j, k] + dire2)

            # Append to our list
            differences_by_dim.append(diffs_k)

        # Distances scaled by each dimension's length scale
        sqdist = 1
        for i in range(d):
            sqdist *= (
                1 + (ell * differences_by_dim[i]) ** 2 / (2 * alpha)
            ) ** (-alpha)

        return sigma_f**2 * sqdist

    def sine_exp_kernel_isotropic(self, X1, X2, length_scales, n_order, index):
        X1 = oti.array(X1)
        X2 = oti.array(X2)

        n1, d = X1.shape

        n2, d = X2.shape

        ell = np.exp(length_scales[0])
        p = length_scales[1]
        sigma_f = length_scales[-1]

        # Prepare the output: a list of d arrays, each of shape (n, m)
        differences_by_dim = []

        # Loop over each dimension k
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = oti.zeros((n1, n2))

            # Nested loops to fill diffs_k
            for i in range(n1):
                for j in range(n2):
                    dire1 = 0
                    dire2 = 0
                    for l in range(self.n_rays):
                        dire1 = (
                            dire1
                            + oti.e(2 * l + 2, order=2 * n_order)
                            * self.rays[k, l]
                        )
                        dire2 = (
                            dire2
                            + oti.e(2 * l + 1, order=2 * n_order)
                            * self.rays[k, l]
                        )
                    diffs_k[i, j] = ((X1[i, k] + dire1)) - (X2[j, k] + dire2)

            # Append to our list
            differences_by_dim.append(diffs_k)

        sqdist = 0
        for i in range(d):
            sqdist = sqdist + (
                ell**2 * (oti.sin(np.pi * differences_by_dim[i] / p)) ** 2
            )

        return sigma_f**2 * oti.exp(-2 * sqdist)

    def negative_log_marginal_likelihood(
        self, ell, x_train, sigma_n, n_order, n_bases, der_indices
    ):
        """
        NLL for standard GP in multiple dimensions.

        NLL = 0.5 * y^T (K^-1) y + 0.5 * log|K| + 0.5*N*log(2*pi).
        """
        K = utils.rbf_kernel(
            x_train,
            x_train,
            ell,
            n_order,
            n_bases,
            der_indices,
            self.kernel_func,
        )
        K += self.nugget * np.eye(len(K))

        try:
            L = cholesky(K)
            alpha = solve(L.T, solve(L, self.y_train))

            data_fit = 0.5 * np.dot(self.y_train, alpha)
            log_det_K = np.sum(np.log(np.diag(L)))
            complexity = log_det_K

            N = len(self.y_train)
            const = 0.5 * N * np.log(2 * np.pi)

            return data_fit + complexity + const
        except:
            return 1000

    def optimize_hyperparameters(self):
        res = minimize(
            fun=self.negative_log_marginal_likelihood,
            x0=self.init,
            args=(
                self.x_train,
                self.sigma_n,
                self.n_order,
                self.n_bases,
                self.der_indices,
            ),
            method="L-BFGS-B",
            bounds=self.bounds,
        )
        return res.x

    def predict(
        self, X_test, length_scales, calc_cov=False, return_deriv=False
    ):
        """
        Compute posterior predictive mean and covariance at X_test
        under an ARD RBF kernel with given length_scales.
        """
        # Build K (train-train) and factor
        K = utils.rbf_kernel(
            self.x_train,
            self.x_train,
            length_scales,
            self.n_order,
            self.n_bases,
            self.der_indices,
            self.kernel_func,
        )
        K += self.sigma_n**2 * np.eye(len(K))
        L = cholesky(K)

        # alpha = K^-1 y

        alpha = solve(L.T, solve(L, self.y_train))

        K_s = utils.rbf_kernel(
            self.x_train,
            X_test,
            length_scales,
            self.n_order,
            self.n_bases,
            self.der_indices,
            self.kernel_func,
        )

        if not return_deriv:
            K_s = K_s[:, 0: len(X_test)]

            f_mean = K_s.T @ (alpha)

            if calc_cov:
                K_ss = utils.rbf_kernel(
                    X_test,
                    X_test,
                    length_scales,
                    self.n_order,
                    self.n_bases,
                    self.der_indices,
                    self.kernel_func,
                )  # shape (N_test, N_test)

                v = solve(L, K_s)
                f_cov = K_ss[0: len(X_test), 0: len(X_test)] - v.T.dot(v)

                return f_mean, f_cov
            else:
                return f_mean
        else:
            f_mean = K_s.T @ (alpha)

            if calc_cov:
                K_ss = utils.rbf_kernel(
                    X_test,
                    X_test,
                    length_scales,
                    self.n_order,
                    self.n_bases,
                    self.der_indices,
                    self.kernel_func,
                )  # shape (N_test, N_test)

                v = solve(L, K_s)
                f_cov = K_ss - v.T.dot(v)

                return f_mean, f_cov
            else:
                return f_mean
