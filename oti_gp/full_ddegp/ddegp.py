import numpy as np
from numpy.linalg import cholesky, solve
import utils2 as utils
from kernel_funcs.kernel_funcs import KernelFactory
from full_ddegp.optimizer import Optimizer
from full_ddegp import ddegp_utils


class ddegp:
    def __init__(
        self,
        x_train,
        y_train,
        n_order,
        n_bases,
        der_indices,
        rays,
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
        self.rays = rays
        self.n_rays = rays.shape[1]
        self.dim = x_train.shape[1]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.der_indices = der_indices
        self.normalize = normalize
        indices = utils.transform_nested_list(der_indices)
        self.flattened_der_indicies = utils.flatten_der_indices(indices)
        if normalize:
            self.y_train, self.mu_y, self.sigma_y, self.sigmas_x, self.mus_x = utils.normalize_y_data_directional(
                x_train, y_train, self.flattened_der_indicies)
            self.rays = utils.normalize_directions(self.sigmas_x, self.rays)
            self.x_train = utils.normalize_x_data_train(x_train)
        else:
            self.x_train = x_train
            self.y_train = utils.reshape_y_train(y_train)

        self.powers = utils.build_companion_array(
            self.n_rays, n_order, der_indices)
        self.differences_by_dim = ddegp_utils.differences_by_dim_func(
            self.x_train, self.x_train, self.rays, n_order)

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

    def optimize_hyperparameters(self, *args, **kwargs):
        return self.optimizer.optimize_hyperparameters(*args, **kwargs)

    def predict(
        self,
        X_test,
        params,
        calc_cov=False,
        return_deriv=False,
    ):
        """
        Compute posterior predictive mean and (optionally) covariance at X_test.
        """
        length_scales = params[:-1]
        sigma_n = params[-1]

        # Compute training kernel matrix and its Cholesky factor
        K = ddegp_utils.rbf_kernel(
            self.differences_by_dim,
            length_scales,
            self.n_order,
            self.n_bases,
            self.kernel_func,
            self.flattened_der_indicies,
            self.powers
        )
        K += (10**sigma_n) ** 2 * np.eye(K.shape[0])
        L = cholesky(K)
        alpha = solve(L.T, solve(L, self.y_train))

        if self.normalize:
            X_test = utils.normalize_x_data_test(
                X_test, self.sigmas_x, self.mus_x)

        # Compute train-test kernel
        diff_x_test_x_train = ddegp_utils.differences_by_dim_func(
            self.x_train, X_test, self.rays, self.n_order
        )
        K_s = ddegp_utils.rbf_kernel(
            diff_x_test_x_train,
            length_scales,
            self.n_order,
            self.n_bases,
            self.kernel_func,
            self.flattened_der_indicies,
            self.powers
        )

        # Compute posterior mean
        f_mean = (K_s[:, :len(X_test)].T @
                  alpha) if not return_deriv else (K_s.T @ alpha)

        if self.normalize:
            if return_deriv:
                f_mean = utils.transform_predictions_directional(
                    f_mean, self.mu_y, self.sigma_y, self.sigmas_x,
                    self.flattened_der_indicies, X_test
                )
            else:
                f_mean = self.mu_y + f_mean * self.sigma_y

        # If covariance is not requested, return mean only
        if not calc_cov:
            return f_mean

        # Compute test-test kernel and covariance
        diff_x_test_x_test = ddegp_utils.differences_by_dim_func(
            X_test, X_test, self.rays, self.n_order
        )
        K_ss = ddegp_utils.rbf_kernel(
            diff_x_test_x_test,
            length_scales,
            self.n_order,
            self.n_bases,
            self.kernel_func,
            self.flattened_der_indicies,
            self.powers
        )

        v = solve(L, K_s)
        f_cov = K_ss - \
            v.T @ v if return_deriv else K_ss[:len(X_test),
                                              :len(X_test)] - v.T @ v

        # Normalize or return raw covariance
        if self.normalize:
            if return_deriv:
                f_var = utils.transform_cov_directional(
                    f_cov, self.sigma_y, self.sigmas_x,
                    self.flattened_der_indicies, X_test
                )
            else:
                f_var = self.sigma_y**2 * np.diag(np.abs(f_cov))
        else:
            f_var = np.diag(np.abs(f_cov))

        return f_mean, f_var
