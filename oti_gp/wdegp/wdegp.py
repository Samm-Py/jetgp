import numpy as np
from numpy.linalg import cholesky, solve
from wdegp import wdegp_utils as wdegp_utils
import utils as utils
from kernel_funcs.kernel_funcs import KernelFactory
from wdegp.optimizer import Optimizer


class wdegp:
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
        self.der_indices = der_indices
        self.dim = x_train.shape[1]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.normalize = normalize

        self.flattened_der_indicies = []
        self.powers = []

        for k, ders in enumerate(der_indices):
            indices = utils.transform_nested_list(ders)
            self.powers.append(
                utils.build_companion_array(n_bases, n_order, ders))

            flat_indices = [i for sublist in indices for i in sublist]
            self.flattened_der_indicies.append(flat_indices)

        if normalize:
            self.y_train = []
            for k, ders in enumerate(self.der_indices):
                y_norm, self.mu_y, self.sigma_y, self.sigmas_x, self.mus_x = utils.normalize_y_data(
                    x_train, y_train[k], self.flattened_der_indicies[k]
                )
                self.y_train.append(y_norm)

            self.x_train = utils.normalize_x_data_train(x_train)
        else:
            self.y_train = [utils.reshape_y_train(y) for y in y_train]
            self.x_train = x_train

        self.differences_by_dim_submodels = [
            wdegp_utils.differences_by_dim_func(
                self.x_train, self.x_train, self.n_order, idx
            )
            for idx in self.index
        ]

        self.kernel_factory = KernelFactory(
            dim=self.dim,
            normalize=self.normalize,
            n_order=self.n_order,
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
        diffs_train_train = wdegp_utils.differences_by_dim_func(
            self.x_train, self.x_train, 0, index=[-1])

        for k in range(n_test):
            x_k = X_test[k].reshape(1, -1)
            diffs_train_test = wdegp_utils.differences_by_dim_func(
                self.x_train, x_k, 0, index=[-1])
            weights = wdegp_utils.determine_weights(
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

            diffs_train_test = wdegp_utils.differences_by_dim_func(
                self.x_train, X_test, self.n_order, index=index_i)
            diffs_train_train = self.differences_by_dim_submodels[i]

            # Kernel matrix (train/train) and Cholesky factor
            K = wdegp_utils.rbf_kernel(
                diffs_train_train, ell, self.n_order, self.n_bases, self.kernel_func,
                self.flattened_der_indicies[i], self.powers[i], index=index_i
            )
            K += (10**sigma_n)**2 * np.eye(len(K))
            L = cholesky(K)
            alpha = solve(L.T, solve(L, self.y_train[i]))

            # Predictive mean
            K_s = wdegp_utils.rbf_kernel(
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
                diffs_test_test = wdegp_utils.differences_by_dim_func(
                    X_test, X_test, self.n_order, index=index_i)
                K_ss = wdegp_utils.rbf_kernel(
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
