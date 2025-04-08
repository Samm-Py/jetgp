import numpy as np
from numpy.linalg import cholesky, solve
from scipy.optimize import minimize
import pyoti.sparse as oti
import utils
from line_profiler import profile
from pyswarm import pso


class oti_gp_directional_weighted_pts:
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
        # self.n_rays = rays.shape[1]
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
        rays = [self.rays[i] for i in index]

        # Loop over each dimension k

        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = oti.zeros((n1, n2))

            # Nested loops to fill diffs_k
            for i in range(n1):
                for j in range(n2):
                    for m in range(len(rays)):
                        dire1 = 0
                        dire2 = 0
                        for l in range(rays[m].shape[1]):
                            dire1 = (
                                dire1
                                + oti.e(2 * l + 2, order=2 * n_order)
                                * rays[m][k, l]
                            )
                            dire2 = (
                                dire2
                                + oti.e(2 * l + 1, order=2 * n_order)
                                * rays[m][k, l]
                            )
                        if i in index and j in index:
                            diffs_k[i, j] = ((X1[i, k] + dire1)) - (
                                X2[j, k] + dire2
                            )
                        elif i in index and j not in index:
                            diffs_k[i, j] = diffs_k[i, j] = (
                                (X1[i, k] + dire1)
                            ) - (X2[j, k])
                        elif i not in index and j in index:
                            diffs_k[i, j] = ((X1[i, k])) - (X2[j, k] + dire2)
                        else:
                            diffs_k[i, j] = ((X1[i, k])) - (X2[j, k])
            # Append to our list
            differences_by_dim.append(diffs_k)

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

    def sine_exp_kernel_anisotropic(
        self, X1, X2, length_scales, n_order, index=-1
    ):
        X1 = oti.array(X1)
        X2 = oti.array(X2)

        n1, d = X1.shape

        n2, d = X2.shape

        ell = np.exp(length_scales[: self.dim])
        p = length_scales[self.dim: -1]
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

            K_s = K_s[:, 0: len(X_test)]
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
                f_cov = K_ss[0: len(X_test), 0: len(X_test)] - v.T.dot(v)
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
        alpha = np.exp(length_scales[self.dim: -1])
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
        p = length_scales[self.dim: -1]
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
        p = np.exp(length_scales[self.dim: -1])
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
            K_s = K_s[:, 0: len(X_test)]
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
                f_cov = K_ss[0: len(X_test), 0: len(X_test)] - v.T.dot(v)
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


class oti_gp_weighted:
    def __init__(
        self,
        x_train,
        y_train,
        n_order,
        n_bases,
        index,
        der_indices,
        normalize=True,
        sigma_n=0.0,
        kernel="SE",
        kernel_type="anisotropic",
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.sigma_n = sigma_n
        self.n_order = n_order
        self.n_bases = n_bases
        self.index = index
        self.dim = x_train.shape[1]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.der_indices = der_indices
        self.kernel_func = self.create_kernel_function()
        self.differences_by_dim_submodels = []
        self.normalize = normalize

        self.flattened_der_indicies = []
        self.powers = []
        for k in range(0, len(der_indices)):
            indices = utils.transform_nested_list(der_indices[k])
            self.powers.append(utils.build_companion_array(
                n_bases, n_order, der_indices[k]))
            tmp = []
            for i in range(0, len(indices)):
                for j in range(0, len(indices[i])):
                    tmp.append(indices[i][j])
            self.flattened_der_indicies.append(tmp)
        if normalize:
            y_train_normalized = []
            for k in range(0, len(der_indices)):
                y_train_k, self.mu_y, self.sigma_y, self.sigmas_x, self.mus_x = utils.normalize_y_data(
                    x_train, y_train[k], self.flattened_der_indicies[k])
                y_train_normalized.append(y_train_k)
            self.y_train = y_train_normalized
            self.x_train = utils.normalize_x_data_train(x_train)
        else:
            y_train_flattened = []
            for k in range(0, len(der_indices)):
                y_train_flattened.append(utils.reshape_y_train(y_train[k]))
            self.x_train = x_train
            self.y_train = y_train_flattened
        for i in range(0, len(self.index)):
            self.submodel_index = self.index[i]
            self.differences_by_dim_submodels.append(
                self.differences_by_dim_func(
                    self.x_train,
                    self.x_train,
                    self.n_order,
                    self.submodel_index,
                )
            )

    def create_kernel_function(self):
        if self.kernel_type == "anisotropic":
            if self.kernel == "SE":
                self.bounds = self.bounds = [(-3, 3)]*self.dim + \
                    [(-1, 3)] + [(-16, -3)]
                return self.se_kernel_anisotropic
            elif self.kernel == "RQ":
                self.bounds = (
                    [(-6, 6)] * self.dim + [(0, 5)] + [(-9, 3)] + [(-16, -3)]
                )

                return self.rq_kernel_anisotropic
            elif self.kernel == "SineExp":
                self.bounds = (
                    [(-5, 5)] * (self.dim)
                    + [(0.0001, 1e2)] * (self.dim)
                    + [(1e-9, 1e2)]
                    + [(1e-16, 1e-3)]
                )

                return self.sine_exp_kernel_anisotropic
            else:
                raise Exception("Kernel Not Implemented")
        else:
            if self.kernel == "SE":
                self.bounds = [(-5, 5)] + [(-9, 4)] + [(-16, -3)]
                return self.se_kernel_isotropic
            elif self.kernel == "RQ":
                self.bounds = [(-4, 1e2)] + [(0, 5)] + [(-9, 2)] + [(-16, -3)]
                return self.rq_kernel_isotropic
            elif self.kernel == "SineExp":
                self.bounds = (
                    [(-4, 1e2)] + [(0, 1e2)] + [(-9, 2)] + [(-16, -3)]
                )
                return self.sine_exp_kernel_isotropic
            else:
                raise Exception("Kernel Not Implemented")

    def differences_by_dim_func(self, X1, X2, n_order, index=-1):
        X1 = oti.array(X1)
        X2 = oti.array(X2)

        n1, d = X1.shape
        n2, d = X2.shape

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
                            (X1[i, k] + oti.e(k + 1, order=2 * n_order))
                        ) - (X2[j, k])
                    elif i in index and j not in index:
                        diffs_k[i, j] = (
                            (X1[i, k] + oti.e(k + 1, order=2 * n_order))
                        ) - (X2[j, k])
                    elif i not in index and j in index:
                        diffs_k[i, j] = ((X1[i, k] + oti.e(k + 1, order=2 * n_order))) - (
                            X2[j, k]
                        )
                    else:
                        diffs_k[i, j] = ((X1[i, k])) - (X2[j, k])
            # Append to our list
            differences_by_dim.append(diffs_k)
        return differences_by_dim

    def se_kernel_anisotropic(self, differences_by_dim, length_scales, index):
        # Distances scaled by each dimension's length scale
        ell = np.exp(length_scales[0:-1])
        sigma_f = length_scales[length_scales.shape[0] - 1]
        sqdist = 0
        for i in range(self.dim):
            sqdist = sqdist + (ell[i] * (differences_by_dim[i])) ** 2

        return (10**sigma_f) ** 2 * oti.exp(-0.5 * sqdist)

    def rq_kernel_anisotropic(self, differences_by_dim, length_scales, index):
        ell = np.exp(length_scales[: self.dim])
        alpha = np.exp(length_scales[self.dim: self.dim + 1])[0]
        sigma_f = length_scales[length_scales.shape[0] - 1]

        sqdist = 1
        for i in range(self.dim):
            sqdist = sqdist + (ell[i] * (differences_by_dim[i])) ** 2

        return (10**sigma_f) ** 2 * (1 + sqdist / (2 * alpha)) ** (-alpha)

    def sine_exp_kernel_anisotropic(
        self, differences_by_dim, length_scales, index
    ):
        ell = np.exp(length_scales[: self.dim])
        p = length_scales[self.dim: -1]
        sigma_f = length_scales[length_scales.shape[0] - 1]

        sqdist = 1
        for i in range(self.dim):
            sqdist = (
                sqdist
                + (ell[i] * oti.sin((np.pi / p[i]) * differences_by_dim[i]))
                ** 2
            )

        return (10**sigma_f) ** 2 * oti.exp(-2 * sqdist)

    def se_kernel_isotropic(self, differences_by_dim, length_scales, index):
        ell = oti.exp(length_scales[0])
        sigma_f = length_scales[length_scales.shape[0] - 1]

        sqdist = 0
        for i in range(self.dim):
            sqdist = sqdist + (ell * (differences_by_dim[i])) ** 2

        return (10**sigma_f) ** 2 * oti.exp(-0.5 * sqdist)

    def rq_kernel_isotropic(self, differences_by_dim, length_scales, index):
        ell = np.exp(length_scales[0])
        alpha = np.exp(length_scales[1])
        sigma_f = length_scales[length_scales.shape[0] - 1]

        sqdist = 1
        for i in range(self.dim):
            sqdist = sqdist + (ell * (differences_by_dim[i])) ** 2

        return (10**sigma_f) ** 2 * (1 + sqdist / (2 * alpha)) ** (-alpha)

    def sine_exp_kernel_isotropic(
        self, differences_by_dim, length_scales, index
    ):
        ell = np.exp(length_scales[0])
        p = length_scales[1]
        sigma_f = length_scales[-1]

        sqdist = 0
        for i in range(self.dim):
            sqdist = sqdist + (
                ell**2 * (oti.sin(np.pi * differences_by_dim[i] / p)) ** 2
            )

        return (10**sigma_f) ** 2 * oti.exp(-2 * sqdist)

    def negative_log_marginal_likelihood(
        self,
        x0,
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
        ell = x0[:-1]
        sigma_n = x0[-1]
        llhood = 0
        for i in range(0, len(self.index)):
            y_train_submodel = self.y_train[i]
            der_indices_submodel = self.flattened_der_indicies[i]
            powers = self.powers[i]
            submodel_index = self.index[i]
            differences_by_dim_submodel = self.differences_by_dim_submodels[i]

            K = utils.rbf_kernel_weighted(
                differences_by_dim_submodel,
                ell,
                n_order,
                n_bases,
                self.kernel_func,
                der_indices_submodel,
                powers,
                index=submodel_index,
            )

            K += (10**sigma_n) ** 2 * np.eye(len(K))

            try:
                L = cholesky(K)
                alpha = solve(L.T, solve(L, y_train_submodel))

                data_fit = 0.5 * np.dot(y_train_submodel, alpha)
                log_det_K = np.sum(np.log(np.diag(L)))
                complexity = log_det_K

                N = len(y_train)
                const = 0.5 * N * np.log(2 * np.pi)

                llhood = llhood + data_fit + complexity + const
            except:
                llhood = llhood + 1e6
            return llhood

    def nll_wrapper(self, x0):
        return self.negative_log_marginal_likelihood(
            x0,
            self.x_train,
            self.y_train,
            # note: this may be unused in your method since x0[-1] is sigma_n
            self.sigma_n,
            self.n_order,
            self.n_bases,
            self.der_indices,
            self.submodel_index,
        )

    def optimize_hyperparameters(self, n_restart_optimizer=20, swarm_size=20):
        lb = [b[0] for b in self.bounds]
        ub = [b[1] for b in self.bounds]

        # Run PSO to minimize the NLL
        best_x, best_nll = pso(
            self.nll_wrapper,
            lb,
            ub,
            swarmsize=swarm_size,
            maxiter=n_restart_optimizer,
            debug=False,  # shows progress of the swarm
        )

        # Optionally: update model attributes with optimized values
        self.opt_x0 = best_x
        self.opt_nll = best_nll

        print("Best solution:", best_x)
        print("Objective value:", best_nll)
        return best_x

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
        ell = length_scales[:-1]
        sigma_n = length_scales[-1]
        weights_matrix = np.zeros((X_test.shape[0], self.x_train.shape[0]))
        diffs_by_dim = self.differences_by_dim_func(
            self.x_train, self.x_train, 0, index=[-1]
        )
        if self.normalize:
            X_test = utils.normalize_x_data_test(
                X_test, self.sigmas_x, self.mus_x)
        for k in range(0, X_test.shape[0]):
            diffs_by_dim_submodel = self.differences_by_dim_func(
                self.x_train, X_test[k].reshape(1, -1), 0, index=[-1]
            )
            weights = utils.determine_weights(
                diffs_by_dim,
                diffs_by_dim_submodel,
                ell,
                self.kernel_func,
            )
            weights_matrix[k, :] = weights[:, 0]

        y_val = 0
        y_var = 0
        submodel_vals = []
        submodel_cov = []

        for i in range(0, len(self.index)):
            differences_by_dim_train_test = self.differences_by_dim_func(
                self.x_train, X_test, self.n_order, index=self.index[i]
            )

            differences_by_dim_test_test = self.differences_by_dim_func(
                X_test, X_test, self.n_order, index=self.index[i]
            )

            diffs_by_dim = self.differences_by_dim_submodels[i]
            K = utils.rbf_kernel_weighted(
                diffs_by_dim,
                ell,
                self.n_order,
                self.n_bases,
                self.kernel_func,
                self.flattened_der_indicies[i],
                self.powers[i],
                index=self.index[i],
            )
            K += (10**sigma_n) ** 2 * np.eye(len(K))
            L = cholesky(K)

            # alpha = K^-1 y

            alpha = solve(L.T, solve(L, self.y_train[i]))

            K_s = utils.rbf_kernel_weighted(
                differences_by_dim_train_test,
                ell,
                self.n_order,
                self.n_bases,
                self.kernel_func,
                self.flattened_der_indicies[i],
                self.powers[i],
                index=self.index[i],
            )

            K_s = K_s[:, 0: len(X_test)]
            f_mean = K_s.T @ (alpha)
            if self.normalize:
                f_mean = self.mu_y + f_mean*self.sigma_y
            y_val = y_val + (weights_matrix[:, i] * f_mean)
            if return_submodels:
                submodel_vals.append(f_mean)

            if calc_cov:
                K_ss = utils.rbf_kernel_weighted(
                    differences_by_dim_test_test,
                    ell,
                    self.n_order,
                    self.n_bases,
                    self.kernel_func,
                    self.flattened_der_indicies[i],
                    self.powers[i],
                    index=self.index[i],
                )  # shape (N_test, N_test)

                v = solve(L, K_s)
                f_cov = K_ss[0: len(X_test), 0: len(X_test)] - v.T.dot(v)
                if self.normalize:
                    f_var = utils.transform_cov(
                        f_cov, self.sigma_y, self.sigmas_x, self.flattened_der_indicies[i], X_test)
                    y_var = y_var + (weights_matrix[:, i] ** 2 * f_var)
                else:
                    f_var = np.diag(np.abs(f_cov))
                    y_var = y_var + (weights_matrix[:, i] ** 2 * f_var)
                if return_submodels:
                    submodel_cov.append(f_var)

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
        normalize=True,
        sigma_n=0.0,
        nugget=1e-6,
        kernel="SE",
        kernel_type="anisotropic",
    ):

        self.sigma_n = sigma_n
        self.nugget = nugget
        self.n_order = n_order
        self.n_bases = n_bases
        self.dim = x_train.shape[1]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.der_indices = der_indices
        self.kernel_func = self.create_kernel_function()
        self.normalize = normalize

        indices = utils.transform_nested_list(der_indices)
        self.flattened_der_indicies = []
        for i in range(0, len(indices)):
            for j in range(0, len(indices[i])):
                self.flattened_der_indicies.append(indices[i][j])

        self.powers = utils.build_companion_array(
            n_bases, n_order, der_indices)

        if normalize:
            self.y_train, self.mu_y, self.sigma_y, self.sigmas_x, self.mus_x = utils.normalize_y_data(
                x_train, y_train, self.flattened_der_indicies)
            self.x_train = utils.normalize_x_data_train(x_train)
        else:
            self.x_train = x_train
            self.y_train = utils.reshape_y_train(y_train)

        self.differences_by_dim = self.differences_by_dim_func(
            self.x_train, self.x_train, n_order
        )

    def differences_by_dim_func(self, X1, X2, n_order, index=-1):
        X1 = oti.array(X1)
        X2 = oti.array(X2)

        n1, d = X1.shape

        n2, d = X2.shape

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
                        + oti.e(k + 1, order=2 * n_order)
                        - (X2[j, k])
                    )

            # Append to our list
            differences_by_dim.append(diffs_k)
        return differences_by_dim

    def create_kernel_function(self):
        if self.kernel_type == "anisotropic":
            if self.kernel == "SE":
                self.bounds = [(-3, 3)]*self.dim + \
                    [(-1, 3)] + [(-16, -3)]
                return self.se_kernel_anisotropic
            elif self.kernel == "RQ":
                self.bounds = (
                    [(-5, 5)] * self.dim + [(0, 10)] + [(-9, 6)] + [(-16, -3)]
                )

                return self.rq_kernel_anisotropic
            elif self.kernel == "SineExp":
                self.bounds = (
                    [(-5, 5)] * (self.dim)
                    + [(0.0, 1e2)] * (self.dim)
                    + [(-9, 5)]
                    + [(-16, -3)]
                )

                return self.sine_exp_kernel_anisotropic
            else:
                raise Exception("Kernel Not Implemented")
        else:
            if self.kernel == "SE":
                self.bounds = [(-4, 1)] + \
                    [(-1, 6)] + [(-16, -3)]
                return self.se_kernel_isotropic
            elif self.kernel == "RQ":
                self.bounds = [(-5, 5)] + [(0, 5)] + [(-9, 4)] + [(-16, -3)]
                return self.rq_kernel_isotropic
            elif self.kernel == "SineExp":
                self.bounds = (
                    [(-5, 5)] + [(0.0, 1e2)] + [(-9, 4)] + [(-16, -3)]
                )
                return self.sine_exp_kernel_isotropic
            else:
                raise Exception("Kernel Not Implemented")

    # @profile
    def se_kernel_anisotropic(
        self, differences_by_dim, length_scales, index=-1
    ):
        # Distances scaled by each dimension's length scale
        ell = 10**(length_scales[0:-1])
        sigma_f = length_scales[length_scales.shape[0] - 1]
        sqdist = 0
        for i in range(self.dim):
            sqdist = sqdist + (ell[i] * (differences_by_dim[i])) ** 2

        return ((10**sigma_f) ** 2) * oti.exp(-0.5 * sqdist)

    def rq_kernel_anisotropic(
        self, differences_by_dim, length_scales, n_order, index=-1
    ):
        ell = 10**(length_scales[: self.dim])
        alpha = np.exp(length_scales[self.dim: self.dim + 1])[0]
        sigma_f = length_scales[length_scales.shape[0] - 1]

        sqdist = 1
        for i in range(self.dim):
            sqdist = sqdist + (ell[i] * (differences_by_dim[i])) ** 2

        return ((10**sigma_f) ** 2) * (1 + sqdist / (2 * alpha)) ** (-alpha)

    def sine_exp_kernel_anisotropic(
        self, differences_by_dim, length_scales, n_order, index=-1
    ):
        ell = 10**(length_scales[: self.dim])
        p = length_scales[self.dim: -1]
        sigma_f = length_scales[length_scales.shape[0] - 1]

        sqdist = 1
        for i in range(self.dim):
            sqdist = (
                sqdist
                + (ell[i] * oti.sin((np.pi / p[i]) * differences_by_dim[i]))
                ** 2
            )

        return ((10**sigma_f) ** 2) * oti.exp(-2 * sqdist)

    def se_kernel_isotropic(self, differences_by_dim, length_scales, index=-1):
        ell = 10**(length_scales[0])
        sigma_f = length_scales[length_scales.shape[0] - 1]

        sqdist = 0
        for i in range(self.dim):
            sqdist = sqdist + (ell * (differences_by_dim[i])) ** 2

        return ((10**sigma_f) ** 2) * oti.exp(-0.5 * sqdist)

    def rq_kernel_isotropic(
        self, differences_by_dim, length_scales, n_order, index=-1
    ):
        ell = 10**(length_scales[0])
        alpha = np.exp(length_scales[1])
        sigma_f = length_scales[length_scales.shape[0] - 1]

        sqdist = 1
        for i in range(self.dim):
            sqdist = sqdist + (ell * (differences_by_dim[i])) ** 2

        return ((10**sigma_f) ** 2) * (1 + sqdist / (2 * alpha)) ** (-alpha)

    def sine_exp_kernel_isotropic(
        self, differences_by_dim, length_scales, n_order, index=-1
    ):
        ell = 10**(length_scales[0])
        p = length_scales[1]
        sigma_f = length_scales[-1]

        sqdist = 0
        for i in range(self.dim):
            sqdist = sqdist + (
                ell**2 * (oti.sin(np.pi * differences_by_dim[i] / p)) ** 2
            )

        return ((10**sigma_f) ** 2) * oti.exp(-2 * sqdist)

    def negative_log_marginal_likelihood(
        self, x0, x_train, sigma_n, n_order, n_bases, der_indices
    ):
        """
        NLL for standard GP in multiple dimensions.

        NLL = 0.5 * y^T (K^-1) y + 0.5 * log|K| + 0.5*N*log(2*pi).
        """

        ell = x0[:-1]
        sigma_n = x0[-1]
        K = utils.rbf_kernel(
            self.differences_by_dim,
            ell,
            n_order,
            n_bases,
            self.kernel_func,
            self.flattened_der_indicies,
            self.powers
        )
        K += ((10**sigma_n) ** 2) * np.eye(len(K))

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
            return 1e6

    def nll_wrapper(self, x0):
        return self.negative_log_marginal_likelihood(
            x0,
            self.x_train,
            # note: this may be unused in your method since x0[-1] is sigma_n
            self.sigma_n,
            self.n_order,
            self.n_bases,
            self.der_indices,
        )

    # @profile
    def optimize_hyperparameters(self, n_restart_optimizer=20, swarm_size=20):
        lb = [b[0] for b in self.bounds]
        ub = [b[1] for b in self.bounds]

        # Run PSO to minimize the NLL
        best_x, best_nll = pso(
            self.nll_wrapper,
            lb,
            ub,
            swarmsize=swarm_size,
            maxiter=n_restart_optimizer,
            debug=False,  # shows progress of the swarm
            minfunc=1e-8,
        )

        # Optionally: update model attributes with optimized values
        self.opt_x0 = best_x
        self.opt_nll = best_nll

        print("Best solution:", best_x)
        print("Objective value:", best_nll)

        return best_x

    def predict(
        self,
        X_test,
        params,
        calc_cov=False,
        return_deriv=False,
        n_restart_optimizer=100,
    ):
        """
        Compute posterior predictive mean and covariance at X_test
        under an ARD RBF kernel with given length_scales.
        """
        # Build K (train-train) and factor
        length_scales = params[0:-1]
        sigma_n = params[-1]
        K = utils.rbf_kernel(
            self.differences_by_dim,
            length_scales,
            self.n_order,
            self.n_bases,
            self.kernel_func,
            self.flattened_der_indicies,
            self.powers
        )
        for i in range(n_restart_optimizer):
            try:
                K += ((10**sigma_n) ** 2) * np.eye(K.shape[0])
                L = cholesky(K)
                break
            except np.linalg.LinAlgError:
                sigma_n *= 2  # Increase jitter if needed

        # alpha = K^-1 y
        print(sigma_n)
        alpha = solve(L.T, solve(L, self.y_train))

        if self.normalize:
            X_test = utils.normalize_x_data_test(
                X_test, self.sigmas_x, self.mus_x)

        diff_x_test_x_train = self.differences_by_dim_func(
            self.x_train, X_test, self.n_order
        )
        K_s = utils.rbf_kernel(
            diff_x_test_x_train,
            length_scales,
            self.n_order,
            self.n_bases,
            self.kernel_func,
            self.flattened_der_indicies,
            self.powers
        )

        if not return_deriv:
            K_s = K_s[:, 0: len(X_test)]

            f_mean = K_s.T @ (alpha)

            if self.normalize:
                f_mean = self.mu_y + f_mean*self.sigma_y

            if calc_cov:
                diff_x_test_x_test = self.differences_by_dim_func(
                    X_test, X_test, self.n_order
                )
                K_ss = utils.rbf_kernel(
                    diff_x_test_x_test,
                    length_scales,
                    self.n_order,
                    self.n_bases,
                    self.kernel_func,
                    self.flattened_der_indicies,
                    self.powers
                )  # shape (N_test, N_test)

                v = solve(L, K_s)
                f_cov = K_ss[0: len(X_test), 0: len(X_test)] - v.T.dot(v)

                if self.normalize:
                    f_var = np.diag(np.abs(f_cov))
                    return f_mean, self.sigma_y**2 * f_var
                else:
                    f_var = np.diag(np.abs(f_cov))
                    return f_mean, f_cov
            else:

                return f_mean
        else:
            f_mean = K_s.T @ (alpha)

            if self.normalize:
                f_mean = utils.transform_predictions(
                    f_mean, self.mu_y, self.sigma_y, self.sigmas_x, self.flattened_der_indicies, X_test)

            if calc_cov:
                diff_x_test_x_test = self.differences_by_dim_func(
                    X_test, X_test, self.n_order
                )
                K_ss = utils.rbf_kernel(
                    diff_x_test_x_test,
                    length_scales,
                    self.n_order,
                    self.n_bases,
                    self.kernel_func,
                    self.flattened_der_indicies,
                    self.powers
                )  # shape (N_test, N_test)

                v = solve(L, K_s)
                f_cov = K_ss - v.T.dot(v)

                if self.normalize:
                    f_var = utils.transform_cov(
                        f_cov, self.sigma_y, self.sigmas_x, self.flattened_der_indicies, X_test)
                    return f_mean, f_var
                else:
                    f_var = np.diag(np.abs(f_cov))
                    return f_mean, f_var
            else:
                return f_mean

    # def gradient_negative_log_marginal_likelihood(
    #     self, x0, x_train, sigma_n, n_order, n_bases, der_indices
    # ):
    #     """
    #     NLL for standard GP in multiple dimensions.

    #     NLL = 0.5 * y^T (K^-1) y + 0.5 * log|K| + 0.5*N*log(2*pi).
    #     """
    #     y_train = self.y_train.reshape(-1, 1)

    #     # K = utils.rbf_kernel(
    #     #     x_train,
    #     #     x_train,
    #     #     ell,
    #     #     n_order,
    #     #     n_bases,
    #     #     der_indices,
    #     #     self.kernel_func,
    #     # )
    #     # K += sigma_n**2 * np.eye(len(K))
    #     grads = []
    #     if not self.Flag:
    #         for i in range(0, len(x0) - 1):
    #             ell = oti.array(x0[:-1])
    #             ell[i, 0] = ell[i, 0] + oti.array(oti.e(2 * n_bases + (2)))
    #             sigma_n = x0[-1]
    #             K_ders = utils.rbf_kernel_der_params(
    #                 self.differences_by_dim_grad,
    #                 ell,
    #                 n_order,
    #                 n_bases,
    #                 der_indices,
    #                 self.kernel_func,
    #             )
    #             # L = cholesky(K)
    #             # alpha = solve(L.T, solve(L, self.y_train)).reshape(-1, 1)
    #             L = self.L_mat
    #             alpha = self.alpha

    #             # dK_dtheta = compute partial derivative of kernel matrix w.r.t. theta
    #             inv_K = np.linalg.solve(
    #                 L.T, np.linalg.solve(L, np.eye(L.shape[0]))
    #             )

    #             grad_L_theta = 0.5 * np.trace(
    #                 (alpha @ alpha.T - inv_K) @ K_ders
    #             )
    #             grads.append(grad_L_theta)

    #         # Compute derivative dL/dsigma_n
    #         trace_term = np.trace(inv_K)
    #         quad_term = y_train.T @ inv_K @ inv_K @ y_train
    #         grad_sigma_n = sigma_n * (trace_term - quad_term)
    #         grads.append(grad_sigma_n[0, 0])
    #         # print(grads)
    #         self.last_grad = -1 * np.array(grads)
    #         return self.last_grad
    #     else:
    #         try:
    #             return self.last_grad
    #         except:
    #             return np.ones_like(x0) * 1e6

    # def differences_by_dim_grad_func(self, X1, X2, n_order, index=-1):
    #     X1 = oti.array(X1)
    #     X2 = oti.array(X2)

    #     n1, d = X1.shape

    #     n2, d = X2.shape

    #     # Prepare the output: a list of d arrays, each of shape (n, m)
    #     differences_by_dim = []

    #     # Loop over each dimension k
    #     for k in range(d):
    #         # Create an empty (n, m) array for this dimension
    #         diffs_k = oti.zeros((n1, n2))

    #         # Nested loops to fill diffs_k
    #         for i in range(n1):
    #             for j in range(n2):
    #                 diffs_k[i, j] = (
    #                     X1[i, k]
    #                     + oti.e(2 * k + 2, order=2 * n_order + 1)
    #                     - (X2[j, k] + oti.e(2 * k + 1, order=2 * n_order + 1))
    #                 )

    #         # Append to our list
    #         differences_by_dim.append(diffs_k)
    #     return differences_by_dim
    # self.differences_by_dim_grad = self.differences_by_dim_grad_func(
    #     self.x_train, self.x_train, n_order
    # )
