import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize
import pyoti.sparse as oti
import pyoti.core as coti
import utils
from matplotlib import pyplot as plt
import itertools


class oti_gp_directional_weighted:
    def __init__(
        self,
        x_train,
        y_train,
        n_order,
        n_bases,
        index,
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
        self.index = index
        self.n_rays = rays.shape[1]
        self.dim = x_train.shape[1]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.der_indices = der_indices
        self.kernel_func = self.create_kernel_function()
        self.weight_kernel_func = self.create_weight_kernel()

    def create_weight_kernel(self):
        if self.kernel_type == "anisotropic":
            if self.kernel == "SE":
                return self.se_weight_kernel_anisotropic
            elif self.kernel == "RQ":
                return self.rq_weight_kernel_anisotropic
            elif self.kernel == "SineExp":
                return self.sine_exp_weight_kernel_anisotropic
            else:
                raise Exception("Kernel Not Implemented")
        else:
            raise Exception("Kernel Not Implemented")
            # if self.kernel == "SE":
            #     return self.se_weight_kernel_isotropic
            # elif self.kernel == "RQ":
            #     return self.rq_weight_kernel_isotropic
            # elif self.kernel == "SineExp":
            #     return self.sine_weight_exp_kernel_isotropic
            # else:

    def create_kernel_function(self):
        if self.kernel_type == "anisotropic":
            if self.kernel == "SE":
                theta_0 = np.zeros((self.dim,))
                self.init = np.concatenate((theta_0, np.array([1])))
                self.bounds = [(-4, 1e2)] * (len(theta_0)) + [(1e-4, None)]
                return self.se_kernel_anisotropic
            elif self.kernel == "RQ":
                theta_0 = np.zeros((self.dim,))
                alpha_0 = np.zeros((self.dim,))
                self.init = np.concatenate((theta_0, alpha_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(alpha_0))
                    + [(1e-4, None)]
                )

                return self.rq_kernel_anisotropic
            elif self.kernel == "SineExp":
                theta_0 = np.zeros((self.dim,))
                p_0 = 10 * np.ones((self.dim,))
                self.init = np.concatenate((theta_0, p_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(p_0))
                    + [(1e-4, None)]
                )

                return self.sine_exp_kernel_anisotropic
            else:
                raise Exception("Kernel Not Implemented")
        else:
            if self.kernel == "SE":
                theta_0 = np.zeros((1,))
                self.init = np.concatenate((theta_0, np.array([1])))
                self.bounds = [(-4, 1e2)] * (len(theta_0)) + [(1e-4, None)]
                return self.se_kernel_isotropic
            elif self.kernel == "RQ":
                theta_0 = np.zeros((1,))
                alpha_0 = np.zeros((1,))
                self.init = np.concatenate((theta_0, alpha_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(alpha_0))
                    + [(1e-4, None)]
                )
                return self.rq_kernel_isotropic
            elif self.kernel == "SineExp":
                theta_0 = np.zeros((1,))
                p_0 = 10 * np.ones((1,))
                self.init = np.concatenate((theta_0, p_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(p_0))
                    + [(1e-4, None)]
                )
                return self.sine_exp_kernel_isotropic
            else:
                raise Exception("Kernel Not Implemented")

    def se_kernel_anisotropic(self, X1, X2, length_scales, n_order, index=-1):
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
                    for l in index:
                        dire1 = (
                            dire1
                            + oti.e(2 * 0 + 2, order=2 * n_order)
                            * self.rays[k, l]
                        )
                        dire2 = (
                            dire2
                            + oti.e(2 * 0 + 1, order=2 * n_order)
                            * self.rays[k, l]
                        )
                    diffs_k[i, j] = ((X1[i, k] + dire1)) - (X2[j, k] + dire2)

            # Append to our list
            differences_by_dim.append(diffs_k)

        sqdist = 0
        for i in range(d):
            sqdist = sqdist + (ell[i] * (differences_by_dim[i])) ** 2

        return sigma_f**2 * oti.exp(-0.5 * sqdist)

    def se_weight_kernel_anisotropic(
        self, rays, x, length_scales, n_order, index=-1
    ):
        ell = np.exp(length_scales[0:-1])
        sigma_f = length_scales[-1]

        n1 = rays.shape[1]
        d = rays.shape[0]

        # Prepare the output: a list of d arrays, each of shape (n, m)
        differences_by_dim = []
        differences_by_dim_r = []

        # Loop over each dimension k
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = np.zeros((n1, n1))
            r = np.zeros((n1, 1))
            # Nested loops to fill diffs_k
            for i in range(n1):
                r[i, 0] = rays[k, i] - x[k]
                for j in range(n1):
                    diffs_k[i, j] = ((rays[k, i])) - (rays[k, j])

            # Append to our list
            differences_by_dim.append(diffs_k)
            differences_by_dim_r.append(r)
        # Distances scaled by each dimension's length scale

        sqdist = 0
        for i in range(d):
            sqdist = sqdist + (ell[i] * (differences_by_dim[i])) ** 2

        phi = sigma_f**2 * np.exp(-0.5 * sqdist)

        sqdist = 0
        for i in range(d):
            sqdist = sqdist + (ell[i] * (differences_by_dim_r[i])) ** 2

        r = sigma_f**2 * np.exp(-0.5 * sqdist)

        return phi, r

    def rq_kernel_anisotropic(self, X1, X2, length_scales, n_order, index=-1):
        X1 = oti.array(X1)
        X2 = oti.array(X2)

        n1, d = X1.shape

        n2, d = X2.shape

        ell = np.exp(length_scales[: self.dim])
        alpha = np.exp(length_scales[self.dim :])
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
                    for l in index:
                        dire1 = (
                            dire1
                            + oti.e(2 * 0 + 2, order=2 * n_order)
                            * self.rays[k, l]
                        )
                        dire2 = (
                            dire2
                            + oti.e(2 * 0 + 1, order=2 * n_order)
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

    def rq_weight_kernel_anisotropic(
        self, rays, x, length_scales, n_order, index=-1
    ):
        ell = np.exp(length_scales[: self.dim])
        alpha = np.exp(length_scales[self.dim : -1])
        sigma_f = length_scales[-1]

        n1 = rays.shape[1]
        d = rays.shape[0]

        # Prepare the output: a list of d arrays, each of shape (n, m)
        differences_by_dim = []
        differences_by_dim_r = []

        # Loop over each dimension k
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = np.zeros((n1, n1))
            r = np.zeros((n1, 1))
            # Nested loops to fill diffs_k
            for i in range(n1):
                r[i, 0] = rays[k, i] - x[k]
                for j in range(n1):
                    diffs_k[i, j] = ((rays[k, i])) - (rays[k, j])

            # Append to our list
            differences_by_dim.append(diffs_k)
            differences_by_dim_r.append(r)
        # Distances scaled by each dimension's length scale

        sqdist = 1
        for i in range(d):
            sqdist *= (
                1 + (ell[i] * differences_by_dim[i]) ** 2 / (2 * alpha[i])
            ) ** (-alpha[i])

        phi = sigma_f**2 * sqdist

        sqdist = 1
        for i in range(d):
            sqdist *= (
                1 + (ell[i] * differences_by_dim_r[i]) ** 2 / (2 * alpha[i])
            ) ** (-alpha[i])

        r = sigma_f**2 * sqdist

        return phi, r

    def sine_exp_kernel_anisotropic(
        self, X1, X2, length_scales, n_order, index=-1
    ):
        X1 = oti.array(X1)
        X2 = oti.array(X2)

        n1, d = X1.shape

        n2, d = X2.shape

        ell = np.exp(length_scales[: self.dim])
        p = length_scales[self.dim : -1]
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
                            + oti.e(2 * 0 + 2, order=2 * n_order)
                            * self.rays[k, l]
                        )
                        dire2 = (
                            dire2
                            + oti.e(2 * 0 + 1, order=2 * n_order)
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

    def sine_exp_weight_kernel_anisotropic(
        self, rays, x, length_scales, n_order, index=-1
    ):
        ell = np.exp(length_scales[: self.dim])
        p = np.exp(length_scales[self.dim : -1])
        sigma_f = length_scales[-1]

        n1 = rays.shape[1]
        d = rays.shape[0]

        # Prepare the output: a list of d arrays, each of shape (n, m)
        differences_by_dim = []
        differences_by_dim_r = []

        # Loop over each dimension k
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = np.zeros((n1, n1))
            r = np.zeros((n1, 1))
            # Nested loops to fill diffs_k
            for i in range(n1):
                r[i, 0] = rays[k, i] - x[k]
                for j in range(n1):
                    diffs_k[i, j] = ((rays[k, i])) - (rays[k, j])

            # Append to our list
            differences_by_dim.append(diffs_k)
            differences_by_dim_r.append(r)
        # Distances scaled by each dimension's length scale

        sqdist = np.sum(
            (ell[i] * np.sin(np.pi / p[i] * differences_by_dim[i])) ** 2
            for i in range(d)
        )

        phi = sigma_f**2 * sqdist

        sqdist = np.sum(
            (ell[i] * np.sin(np.pi / p[i] * differences_by_dim_r[i])) ** 2
            for i in range(d)
        )

        r = sigma_f**2 * sqdist

        return phi, r

    def se_kernel_isotropic(self, X1, X2, length_scales, n_order, index=-1):
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

    def rq_kernel_isotropic(self, X1, X2, length_scales, n_order, index=-1):
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

    def sine_exp_kernel_isotropic(
        self, X1, X2, length_scales, n_order, index=-1
    ):
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
        self,
        ell,
        x_train,
        y_train,
        sigma_n,
        n_order,
        n_bases,
        der_indices,
        index,
    ):
        """
        NLL for standard GP in multiple dimensions.

        NLL = 0.5 * y^T (K^-1) y + 0.5 * log|K| + 0.5*N*log(2*pi).
        """
        K = utils.rbf_kernel_directional(
            x_train,
            x_train,
            ell,
            n_order,
            n_bases,
            der_indices,
            self.kernel_func,
            index=index,
        )
        K += self.nugget * np.eye(len(K))

        try:
            L = cholesky(K)
            alpha = solve(L.T, solve(L, y_train))

            data_fit = 0.5 * np.dot(y_train, alpha)
            log_det_K = np.sum(np.log(np.diag(L)))
            complexity = log_det_K

            N = len(y_train)
            const = 0.5 * N * np.log(2 * np.pi)

            return data_fit + complexity + const
        except:
            return 1000

    def optimize_hyperparameters(self):
        res_submodel = []

        for i in range(0, len(self.index)):
            print(
                "Optimizing Parameters for submodel {0} out of {1}".format(
                    i + 1, len(self.index)
                )
            )
            y_train = self.y_train[i]
            res = minimize(
                fun=self.negative_log_marginal_likelihood,
                x0=self.init,
                args=(
                    self.x_train,
                    y_train,
                    self.sigma_n,
                    self.n_order,
                    self.n_bases,
                    self.der_indices,
                    self.index[i],
                ),
                method="L-BFGS-B",
                bounds=self.bounds,
            )
            res_submodel.append(res.x)
        return res_submodel

    def predict(
        self, X_test, length_scales, calc_cov=False, return_submodels=False
    ):
        """
        Compute posterior predictive mean and covariance at X_test
        under an ARD RBF kernel with given length_scales.
        """

        weights_matrix = np.zeros((X_test.shape[0], len(length_scales)))

        for k in range(0, X_test.shape[0]):
            weights = utils.determine_directional_weights(
                self.rays,
                X_test[k],
                length_scales[-1],
                self.n_order,
                self.n_bases,
                self.der_indices,
                self.weight_kernel_func,
            )
            weights_matrix[k, :] = weights[:, 0]

        y_val = 0
        y_var = 0

        submodel_vals = []
        submodel_cov = []
        for i in range(0, len(length_scales)):
            K = utils.rbf_kernel_directional(
                self.x_train,
                self.x_train,
                length_scales[i],
                self.n_order,
                self.n_bases,
                self.der_indices,
                self.kernel_func,
                index=self.index[i],
            )
            K += self.sigma_n**2 * np.eye(len(K))
            L = cholesky(K)

            # alpha = K^-1 y

            alpha = solve(L.T, solve(L, self.y_train[i]))

            K_s = utils.rbf_kernel_directional(
                self.x_train,
                X_test,
                length_scales[i],
                self.n_order,
                self.n_bases,
                self.der_indices,
                self.kernel_func,
                index=self.index[i],
            )
            K_s = K_s[:, 0 : len(X_test)]
            f_mean = K_s.T @ (alpha)
            if return_submodels:
                submodel_vals.append(f_mean)

            y_val = y_val + (weights_matrix[:, i] * f_mean)

            if calc_cov:
                K_ss = utils.rbf_kernel_weighted(
                    X_test,
                    X_test,
                    length_scales[i],
                    self.n_order,
                    self.n_bases,
                    self.der_indices,
                    self.kernel_func,
                    index=self.index[i],
                )  # shape (N_test, N_test)

                v = solve(L, K_s)
                f_cov = K_ss[0 : len(X_test), 0 : len(X_test)] - v.T.dot(v)
                y_var = y_var + (weights_matrix[:, i] ** 2 * f_cov)
                if return_submodels:
                    submodel_cov.append(f_cov)

        if return_submodels:
            if calc_cov:
                return y_val, y_var, submodel_vals, submodel_cov
            else:
                return y_val, submodel_vals
        else:
            if calc_cov:
                return y_val, y_var
            else:
                return y_val


class oti_gp_directional:
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

    def create_kernel_function(self):
        if self.kernel_type == "anisotropic":
            if self.kernel == "SE":
                theta_0 = np.zeros((self.dim,))
                self.init = np.concatenate((theta_0, np.array([1])))
                self.bounds = [(-4, 1e2)] * (len(theta_0)) + [(1e-4, None)]
                return self.se_kernel_anisotropic
            elif self.kernel == "RQ":
                theta_0 = np.zeros((self.dim,))
                alpha_0 = np.zeros((self.dim,))
                self.init = np.concatenate((theta_0, alpha_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(alpha_0))
                    + [(1e-4, None)]
                )

                return self.rq_kernel_anisotropic
            elif self.kernel == "SineExp":
                theta_0 = np.zeros((self.dim,))
                p_0 = 10 * np.ones((self.dim,))
                self.init = np.concatenate((theta_0, p_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(p_0))
                    + [(1e-4, None)]
                )

                return self.sine_exp_kernel_anisotropic
            else:
                raise Exception("Kernel Not Implemented")
        else:
            if self.kernel == "SE":
                theta_0 = np.zeros((1,))
                self.init = np.concatenate((theta_0, np.array([1])))
                self.bounds = [(-4, 1e2)] * (len(theta_0)) + [(1e-4, None)]
                return self.se_kernel_isotropic
            elif self.kernel == "RQ":
                theta_0 = np.zeros((1,))
                alpha_0 = np.zeros((1,))
                self.init = np.concatenate((theta_0, alpha_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(alpha_0))
                    + [(1e-4, None)]
                )
                return self.rq_kernel_isotropic
            elif self.kernel == "SineExp":
                theta_0 = np.zeros((1,))
                p_0 = 10 * np.ones((1,))
                self.init = np.concatenate((theta_0, p_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(p_0))
                    + [(1e-4, None)]
                )
                return self.sine_exp_kernel_isotropic
            else:
                raise Exception("Kernel Not Implemented")

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
        alpha = np.exp(length_scales[self.dim :])
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
        p = length_scales[self.dim :]
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
            K_s = K_s[:, 0 : len(X_test)]

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
                f_cov = K_ss[0 : len(X_test), 0 : len(X_test)] - v.T.dot(v)

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


class oti_gp_weighted:
    def __init__(
        self,
        x_train,
        y_train,
        n_order,
        n_bases,
        index,
        der_indices,
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
        self.index = index
        self.dim = x_train.shape[1]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.der_indices = der_indices
        self.kernel_func = self.create_kernel_function()

    def create_kernel_function(self):
        if self.kernel_type == "anisotropic":
            if self.kernel == "SE":
                theta_0 = np.zeros((self.dim,))
                self.init = np.concatenate((theta_0, np.array([1])))
                self.bounds = [(-4, 1e2)] * (len(theta_0)) + [(1e-4, None)]
                return self.se_kernel_anisotropic
            elif self.kernel == "RQ":
                theta_0 = np.zeros((self.dim,))
                alpha_0 = np.zeros((self.dim,))
                self.init = np.concatenate((theta_0, alpha_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(alpha_0))
                    + [(1e-4, None)]
                )

                return self.rq_kernel_anisotropic
            elif self.kernel == "SineExp":
                theta_0 = np.zeros((self.dim,))
                p_0 = 10 * np.ones((self.dim,))
                self.init = np.concatenate((theta_0, p_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(p_0))
                    + [(1e-4, None)]
                )

                return self.sine_exp_kernel_anisotropic
            else:
                raise Exception("Kernel Not Implemented")
        else:
            if self.kernel == "SE":
                theta_0 = np.zeros((1,))
                self.init = np.concatenate((theta_0, np.array([1])))
                self.bounds = [(-4, 1e2)] * (len(theta_0)) + [(1e-4, None)]
                return self.se_kernel_isotropic
            elif self.kernel == "RQ":
                theta_0 = np.zeros((1,))
                alpha_0 = np.zeros((1,))
                self.init = np.concatenate((theta_0, alpha_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(alpha_0))
                    + [(1e-4, None)]
                )
                return self.rq_kernel_isotropic
            elif self.kernel == "SineExp":
                theta_0 = np.zeros((1,))
                p_0 = 10 * np.ones((1,))
                self.init = np.concatenate((theta_0, p_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(p_0))
                    + [(1e-4, None)]
                )
                return self.sine_exp_kernel_isotropic
            else:
                raise Exception("Kernel Not Implemented")

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
                    if i in index and j in index:
                        diffs_k[i, j] = (
                            (X1[i, k] + oti.e(2 * k + 2, order=2 * n_order))
                        ) - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                    elif i in index and j not in index:
                        diffs_k[i, j] = (
                            (X1[i, k] + oti.e(2 * k + 2, order=2 * n_order))
                        ) - (X2[j, k])
                    elif i not in index and j in index:
                        diffs_k[i, j] = ((X1[i, k])) - (
                            X2[j, k] + oti.e(2 * k + 1, order=2 * n_order)
                        )
                    else:
                        diffs_k[i, j] = ((X1[i, k])) - (X2[j, k])
            # Append to our list
            differences_by_dim.append(diffs_k)

        # Distances scaled by each dimension's length scale
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
        alpha = np.exp(length_scales[self.dim :])
        sigma_f = length_scales[-1]

        # Prepare the output: a list of d arrays, each of shape (n, m)
        differences_by_dim = []

        # Loop over each dimension k
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = oti.zeros((n1, n1))

            # Nested loops to fill diffs_k
            for i in range(n1):
                for j in range(n2):
                    if i == index and j == index:
                        diffs_k[i, j] = (
                            (X1[i, k] + oti.e(2 * k + 2, order=2 * n_order))
                        ) - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                    elif i == index and j != index:
                        diffs_k[i, j] = (
                            (X1[i, k] + oti.e(2 * k + 2, order=2 * n_order))
                        ) - (X2[j, k])
                    elif i != index and j == index:
                        diffs_k[i, j] = ((X1[i, k])) - (
                            X2[j, k] + oti.e(2 * k + 1, order=2 * n_order)
                        )
                    else:
                        diffs_k[i, j] = ((X1[i, k])) - (X2[j, k])
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
        p = length_scales[self.dim :]
        sigma_f = length_scales[-1]

        # Prepare the output: a list of d arrays, each of shape (n, m)
        differences_by_dim = []

        # Loop over each dimension k
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = oti.zeros((n1, n1))

            # Nested loops to fill diffs_k
            for i in range(n1):
                for j in range(n2):
                    if i == index and j == index:
                        diffs_k[i, j] = (
                            (X1[i, k] + oti.e(2 * k + 2, order=2 * n_order))
                        ) - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                    elif i == index and j != index:
                        diffs_k[i, j] = (
                            (X1[i, k] + oti.e(2 * k + 2, order=2 * n_order))
                        ) - (X2[j, k])
                    elif i != index and j == index:
                        diffs_k[i, j] = ((X1[i, k])) - (
                            X2[j, k] + oti.e(2 * k + 1, order=2 * n_order)
                        )
                    else:
                        diffs_k[i, j] = ((X1[i, k])) - (X2[j, k])
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
            diffs_k = oti.zeros((n1, n1))

            # Nested loops to fill diffs_k
            for i in range(n1):
                for j in range(n2):
                    if i == index and j == index:
                        diffs_k[i, j] = (
                            (X1[i, k] + oti.e(2 * k + 2, order=2 * n_order))
                        ) - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                    elif i == index and j != index:
                        diffs_k[i, j] = (
                            (X1[i, k] + oti.e(2 * k + 2, order=2 * n_order))
                        ) - (X2[j, k])
                    elif i != index and j == index:
                        diffs_k[i, j] = ((X1[i, k])) - (
                            X2[j, k] + oti.e(2 * k + 1, order=2 * n_order)
                        )
                    else:
                        diffs_k[i, j] = ((X1[i, k])) - (X2[j, k])
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
            diffs_k = oti.zeros((n1, n1))

            # Nested loops to fill diffs_k
            for i in range(n1):
                for j in range(n2):
                    if i == index and j == index:
                        diffs_k[i, j] = (
                            (X1[i, k] + oti.e(2 * k + 2, order=2 * n_order))
                        ) - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                    elif i == index and j != index:
                        diffs_k[i, j] = (
                            (X1[i, k] + oti.e(2 * k + 2, order=2 * n_order))
                        ) - (X2[j, k])
                    elif i != index and j == index:
                        diffs_k[i, j] = ((X1[i, k])) - (
                            X2[j, k] + oti.e(2 * k + 1, order=2 * n_order)
                        )
                    else:
                        diffs_k[i, j] = ((X1[i, k])) - (X2[j, k])
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
            diffs_k = oti.zeros((n1, n1))

            # Nested loops to fill diffs_k
            for i in range(n1):
                for j in range(n2):
                    if i == index and j == index:
                        diffs_k[i, j] = (
                            (X1[i, k] + oti.e(2 * k + 2, order=2 * n_order))
                        ) - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                    elif i == index and j != index:
                        diffs_k[i, j] = (
                            (X1[i, k] + oti.e(2 * k + 2, order=2 * n_order))
                        ) - (X2[j, k])
                    elif i != index and j == index:
                        diffs_k[i, j] = ((X1[i, k])) - (
                            X2[j, k] + oti.e(2 * k + 1, order=2 * n_order)
                        )
                    else:
                        diffs_k[i, j] = ((X1[i, k])) - (X2[j, k])
            # Append to our list
            differences_by_dim.append(diffs_k)

        sqdist = 0
        for i in range(d):
            sqdist = sqdist + (
                ell**2 * (oti.sin(np.pi * differences_by_dim[i] / p)) ** 2
            )

        return sigma_f**2 * oti.exp(-2 * sqdist)

    def negative_log_marginal_likelihood(
        self,
        ell,
        x_train,
        y_train,
        sigma_n,
        n_order,
        n_bases,
        der_indices,
        index,
    ):
        """
        NLL for standard GP in multiple dimensions.

        NLL = 0.5 * y^T (K^-1) y + 0.5 * log|K| + 0.5*N*log(2*pi).
        """
        K = utils.rbf_kernel_weighted(
            x_train,
            x_train,
            ell,
            n_order,
            n_bases,
            der_indices,
            self.kernel_func,
            index=index,
        )
        K += self.nugget * np.eye(len(K))

        try:
            L = cholesky(K)
            alpha = solve(L.T, solve(L, y_train))

            data_fit = 0.5 * np.dot(y_train, alpha)
            log_det_K = np.sum(np.log(np.diag(L)))
            complexity = log_det_K

            N = len(y_train)
            const = 0.5 * N * np.log(2 * np.pi)

            return data_fit + complexity + const
        except:
            return 1000

    def optimize_hyperparameters(self):
        res_submodel = []

        for i in range(0, len(self.index)):
            print(
                "Optimizing Parameters for submodel {0} out of {1}".format(
                    i + 1, len(self.index)
                )
            )
            y_train = self.y_train[i]
            der_indices = self.der_indices[i]
            res = minimize(
                fun=self.negative_log_marginal_likelihood,
                x0=self.init,
                args=(
                    self.x_train,
                    y_train,
                    self.sigma_n,
                    self.n_order,
                    self.n_bases,
                    der_indices,
                    self.index[i],
                ),
                method="L-BFGS-B",
                bounds=self.bounds,
            )
            res_submodel.append(res.x)
        return res_submodel

    def predict(
        self,
        X_test,
        length_scales,
        calc_cov=False,
        return_submodels=False,
    ):
        """
        Compute posterior predictive mean and covariance at X_test
        under an ARD RBF kernel with given length_scales.
        """

        weights_matrix = np.zeros((X_test.shape[0], len(length_scales)))

        for k in range(0, X_test.shape[0]):
            weights = utils.determine_weights(
                self.x_train,
                X_test[k],
                length_scales[-1],
                self.n_order,
                self.n_bases,
                self.der_indices,
                self.kernel_func,
            )
            weights_matrix[k, :] = weights[:, 0]

        y_val = 0
        y_var = 0
        submodel_vals = []
        submodel_cov = []
        for i in range(0, len(length_scales)):
            K = utils.rbf_kernel_weighted(
                self.x_train,
                self.x_train,
                length_scales[i],
                self.n_order,
                self.n_bases,
                self.der_indices[i],
                self.kernel_func,
                index=self.index[i],
            )
            K += self.sigma_n**2 * np.eye(len(K))
            L = cholesky(K)

            # alpha = K^-1 y

            alpha = solve(L.T, solve(L, self.y_train[i]))

            K_s = utils.rbf_kernel_weighted(
                self.x_train,
                X_test,
                length_scales[i],
                self.n_order,
                self.n_bases,
                self.der_indices[i],
                self.kernel_func,
                index=self.index[i],
            )

            K_s = K_s[:, 0 : len(X_test)]
            f_mean = K_s.T @ (alpha)
            y_val = y_val + (weights_matrix[:, i] * f_mean)
            if return_submodels:
                submodel_vals.append(f_mean)

            if calc_cov:
                K_ss = utils.rbf_kernel_weighted(
                    X_test,
                    X_test,
                    length_scales[i],
                    self.n_order,
                    self.n_bases,
                    self.der_indices[i],
                    self.kernel_func,
                    index=self.index[i],
                )  # shape (N_test, N_test)

                v = solve(L, K_s)
                f_cov = K_ss[0 : len(X_test), 0 : len(X_test)] - v.T.dot(v)
                y_var = y_var + (weights_matrix[:, i] ** 2 * f_cov)
                if return_submodels:
                    submodel_cov.append(f_cov)

        if return_submodels:
            if calc_cov:
                return y_val, y_var, submodel_vals, submodel_cov
            else:
                return y_val, submodel_vals
        else:
            if calc_cov:
                return y_val, y_var
            else:
                return y_val


class oti_gp:
    def __init__(
        self,
        x_train,
        y_train,
        n_order,
        n_bases,
        der_indices,
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
        self.dim = x_train.shape[1]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.der_indices = der_indices
        self.kernel_func = self.create_kernel_function()

    def create_kernel_function(self):
        if self.kernel_type == "anisotropic":
            if self.kernel == "SE":
                theta_0 = np.zeros((self.dim,))
                self.init = np.concatenate((theta_0, np.array([1])))
                self.bounds = [(-4, 1e2)] * (len(theta_0)) + [(1e-4, None)]
                return self.se_kernel_anisotropic
            elif self.kernel == "RQ":
                theta_0 = np.zeros((self.dim,))
                alpha_0 = np.zeros((self.dim,))
                self.init = np.concatenate((theta_0, alpha_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(alpha_0))
                    + [(1e-4, None)]
                )

                return self.rq_kernel_anisotropic
            elif self.kernel == "SineExp":
                theta_0 = np.zeros((self.dim,))
                p_0 = 10 * np.ones((self.dim,))
                self.init = np.concatenate((theta_0, p_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(p_0))
                    + [(1e-4, None)]
                )

                return self.sine_exp_kernel_anisotropic
            else:
                raise Exception("Kernel Not Implemented")
        else:
            if self.kernel == "SE":
                theta_0 = np.zeros((1,))
                self.init = np.concatenate((theta_0, np.array([1])))
                self.bounds = [(-4, 1e2)] * (len(theta_0)) + [(1e-4, None)]
                return self.se_kernel_isotropic
            elif self.kernel == "RQ":
                theta_0 = np.zeros((1,))
                alpha_0 = np.zeros((1,))
                self.init = np.concatenate((theta_0, alpha_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(alpha_0))
                    + [(1e-4, None)]
                )
                return self.rq_kernel_isotropic
            elif self.kernel == "SineExp":
                theta_0 = np.zeros((1,))
                p_0 = 10 * np.ones((1,))
                self.init = np.concatenate((theta_0, p_0, np.array([1])))
                self.bounds = (
                    [(-4, 1e2)] * (len(theta_0))
                    + [(0, 1e2)] * (len(p_0))
                    + [(1e-4, None)]
                )
                return self.sine_exp_kernel_isotropic
            else:
                raise Exception("Kernel Not Implemented")

    def se_kernel_anisotropic(self, X1, X2, length_scales, n_order, index=-1):
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
                    diffs_k[i, j] = (
                        X1[i, k]
                        + oti.e(2 * k + 2, order=2 * n_order)
                        - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                    )

            # Append to our list
            differences_by_dim.append(diffs_k)

        # Distances scaled by each dimension's length scale
        sqdist = 0
        for i in range(d):
            sqdist = sqdist + (ell[i] * (differences_by_dim[i])) ** 2

        return sigma_f**2 * oti.exp(-0.5 * sqdist)

    def rq_kernel_anisotropic(self, X1, X2, length_scales, n_order, index=-1):
        X1 = oti.array(X1)
        X2 = oti.array(X2)

        n1, d = X1.shape

        n2, d = X2.shape

        ell = np.exp(length_scales[: self.dim])
        alpha = np.exp(length_scales[self.dim :])
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
                    diffs_k[i, j] = (
                        X1[i, k]
                        + oti.e(2 * k + 2, order=2 * n_order)
                        - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                    )

            # Append to our list
            differences_by_dim.append(diffs_k)

        # Distances scaled by each dimension's length scale

        sqdist = 1
        for i in range(d):
            sqdist *= (
                1 + (ell[i] * differences_by_dim[i]) ** 2 / (2 * alpha[i])
            ) ** (-alpha[i])

        return sigma_f**2 * sqdist

    def sine_exp_kernel_anisotropic(
        self, X1, X2, length_scales, n_order, index=-1
    ):
        X1 = oti.array(X1)
        X2 = oti.array(X2)

        n1, d = X1.shape

        n2, d = X2.shape

        ell = np.exp(length_scales[: self.dim])
        p = length_scales[self.dim :]
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
                    diffs_k[i, j] = (
                        X1[i, k]
                        + oti.e(2 * k + 2, order=2 * n_order)
                        - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                    )

            # Append to our list
            differences_by_dim.append(diffs_k)

        # Distances scaled by each dimension's length scale

        sqdist = np.sum(
            (ell[i] * oti.sin(np.pi / p[i] * differences_by_dim[i])) ** 2
            for i in range(d)
        )
        return sigma_f**2 * oti.exp(-2 * sqdist)

    def se_kernel_isotropic(self, X1, X2, length_scales, n_order, index=-1):
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
                    diffs_k[i, j] = (
                        X1[i, k]
                        + oti.e(2 * k + 2, order=2 * n_order)
                        - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                    )

            # Append to our list
            differences_by_dim.append(diffs_k)

        # Distances scaled by each dimension's length scale
        sqdist = 0
        for i in range(d):
            sqdist = sqdist + (ell * (differences_by_dim[i])) ** 2

        return sigma_f**2 * oti.exp(-0.5 * sqdist)

    def rq_kernel_isotropic(self, X1, X2, length_scales, n_order, index=-1):
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
                    diffs_k[i, j] = (
                        X1[i, k]
                        + oti.e(2 * k + 2, order=2 * n_order)
                        - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                    )

            # Append to our list
            differences_by_dim.append(diffs_k)

        # Distances scaled by each dimension's length scale
        sqdist = 1
        for i in range(d):
            sqdist *= (
                1 + (ell * differences_by_dim[i]) ** 2 / (2 * alpha)
            ) ** (-alpha)

        return sigma_f**2 * sqdist

    def sine_exp_kernel_isotropic(
        self, X1, X2, length_scales, n_order, index=-1
    ):
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
                    diffs_k[i, j] = (
                        X1[i, k]
                        + oti.e(2 * k + 2, order=2 * n_order)
                        - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order))
                    )

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
            K_s = K_s[:, 0 : len(X_test)]

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
                f_cov = K_ss[0 : len(X_test), 0 : len(X_test)] - v.T.dot(v)

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
