import numpy as np
from numpy.linalg import cholesky, solve
from wddegp import wddegp_utils as wddegp_utils
import utils2 as utils
from kernel_funcs.kernel_funcs import KernelFactory
from wddegp.optimizer import Optimizer


class wddegp:
    def __init__(
        self,
        x_train,
        y_train,
        n_order,
        n_bases,
        index,
        der_indices,
        rays,
        normalize=False,
        sigma_n=0.0,
        kernel="SE",
        kernel_type="anisotropic",
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.sigma_n = sigma_n
        self.n_order = n_order
        self.n_bases = n_bases
        self.rays = rays
        self.index = index
        # self.n_rays = rays.shape[1]
        self.dim = x_train.shape[1]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.der_indices = der_indices
        self.flattened_der_indicies = []
        self.powers = []
        self.normalize = normalize

        for k, ders in enumerate(der_indices):
            indices = utils.transform_nested_list(ders)
            n_rays = rays[k].shape[1]
            self.powers.append(
                utils.build_companion_array(n_rays, n_order, ders))

            flat_indices = [i for sublist in indices for i in sublist]
            self.flattened_der_indicies.append(flat_indices)

        if normalize:
            self.y_train = []
            self.rays = []
            for k, ders in enumerate(self.der_indices):
                y_norm, self.mu_y, self.sigma_y, self.sigmas_x, self.mus_x = utils.normalize_y_data_directional(
                    x_train, y_train[k], self.flattened_der_indicies[k]
                )
                rays_norm = utils.normalize_directions(self.sigmas_x, rays[k])
                self.y_train.append(y_norm)
                self.rays.append(rays_norm)

            self.x_train = utils.normalize_x_data_train(x_train)
        else:
            self.y_train = [utils.reshape_y_train(y) for y in y_train]
            self.x_train = x_train

        self.differences_by_dim_submodels = [
            wddegp_utils.differences_by_dim_func(
                self.x_train, self.x_train, self.rays,  self.n_order, idx, idx_list
            )
            for idx, idx_list in enumerate(self.index)
        ]

        self.kernel_factory = KernelFactory(
            dim=self.dim,
            normalize=self.normalize,
            differences_by_dim=self.differences_by_dim_submodels[0],
        )
        self.kernel_func = self.kernel_factory.create_kernel(
            kernel_name=self.kernel,
            kernel_type=self.kernel_type,
        )
        self.bounds = self.kernel_factory.bounds
        self.optimizer = Optimizer(self)

    def optimize_hyperparameters(self, *args, **kwargs):
        return self.optimizer.optimize_hyperparameters(*args, **kwargs)

    def predict(self, X_test, length_scales, calc_cov=False, return_submodels=False):
        """
        Compute posterior predictive mean and covariance at X_test
        under an ARD RBF kernel with given length_scales.
        """
        ell = length_scales[:-1]
        sigma_n = length_scales[-1]
        n_test = X_test.shape[0]
        n_train = self.x_train.shape[0]

        # Normalize test input if required
        if self.normalize:
            X_test = utils.normalize_x_data_test(
                X_test, self.sigmas_x, self.mus_x)

        # Compute weights matrix
        weights_matrix = np.zeros((n_test, n_train))
        diffs_train_train = wddegp_utils.differences_by_dim_func(
            self.x_train, self.x_train, self.rays, 0, index=-1)

        for k in range(n_test):
            x_k = X_test[k].reshape(1, -1)
            diffs_train_test = wddegp_utils.differences_by_dim_func(
                self.x_train, x_k, self.rays, 0, index=-1)
            weights = wddegp_utils.determine_weights(
                diffs_train_train, diffs_train_test, ell, self.kernel_func)
            weights_matrix[k] = weights[:, 0]

        # Initialize outputs
        y_val = 0
        y_var = 0
        submodel_vals = []
        submodel_cov = []

        # Loop over submodels
        for i in range(len(self.index)):
            index_i = self.index[i]

            diffs_train_test = wddegp_utils.differences_by_dim_func(
                self.x_train, X_test, self.rays, self.n_order, index=i, index_list = index_i)
            diffs_train_train = self.differences_by_dim_submodels[i]

            # Kernel matrix (train/train) and Cholesky factor
            K = wddegp_utils.rbf_kernel(
                diffs_train_train, ell, self.n_order, self.n_bases, self.kernel_func,
                self.flattened_der_indicies[i], self.powers[i], index=index_i
            )
            K += (10**sigma_n)**2 * np.eye(len(K))
            L = cholesky(K)
            alpha = solve(L.T, solve(L, self.y_train[i]))

            # Predictive mean
            K_s = wddegp_utils.rbf_kernel(
                diffs_train_test, ell, self.n_order, self.n_bases, self.kernel_func,
                self.flattened_der_indicies[i], self.powers[i], index=index_i
            )
            f_mean = K_s[:, :n_test].T @ alpha
            if self.normalize:
                f_mean = self.mu_y + f_mean * self.sigma_y

            for j in range(len(self.index[i])):
                y_val += weights_matrix[:, self.index[i][j]] * f_mean
            if return_submodels:
                submodel_vals.append(f_mean)

            # Predictive covariance (optional)
            if calc_cov:
                diffs_test_test = wddegp_utils.differences_by_dim_func(
                    X_test, X_test, self.rays, self.n_order,  index=i, index_list = index_i)
                K_ss = wddegp_utils.rbf_kernel(
                    diffs_test_test, ell, self.n_order, self.n_bases, self.kernel_func,
                    self.flattened_der_indicies[i], self.powers[i], index=index_i
                )
                v = solve(L, K_s[:, :n_test])
                f_cov = K_ss[:n_test, :n_test] - v.T @ v

                if self.normalize:
                    f_var = utils.transform_cov(
                        f_cov, self.sigma_y, self.sigmas_x,
                        self.flattened_der_indicies[i], X_test
                    )
                else:
                    f_var = np.diag(np.abs(f_cov))
                for j in range(len(self.index[i])):
                    y_var += (weights_matrix[:, self.index[i][j]] ** 2) * f_var
                if return_submodels:
                    submodel_cov.append(f_var)

        # Return results
        if return_submodels:
            return (y_val, y_var, submodel_vals, submodel_cov) if calc_cov else (y_val, submodel_vals)
        else:
            return (y_val, y_var) if calc_cov else y_val
