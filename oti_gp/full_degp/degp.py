import numpy as np
from numpy.linalg import cholesky, solve
from full_degp import degp_utils
import utils as utils
from kernel_funcs.kernel_funcs import KernelFactory
from full_degp.optimizer import Optimizer


class degp:
    def __init__(
        self,
        x_train,
        y_train,
        n_order,
        n_bases,
        der_indices,
        normalize=True,
        sigma_n=None,
        kernel="SE",
        kernel_type="anisotropic",
    ):

        self.sigma_n = sigma_n
        self.n_order = n_order
        self.n_bases = n_bases
        self.dim = x_train.shape[1]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.der_indices = der_indices
        self.normalize = normalize
        indices = utils.transform_nested_list(der_indices)
        self.flattened_der_indicies = utils.flatten_der_indices(indices)
        self.powers = utils.build_companion_array(
            n_bases, n_order, der_indices)

        if normalize:
            self.y_train, self.mu_y, self.sigma_y, self.sigmas_x, self.mus_x = utils.normalize_y_data(
                x_train, y_train, self.flattened_der_indicies)
            self.x_train = utils.normalize_x_data_train(x_train)
        else:
            self.x_train = x_train
            self.y_train = utils.reshape_y_train(y_train)
        self.differences_by_dim = degp_utils.differences_by_dim_func(
            self.x_train, self.x_train, n_order
        )
        self.kernel_factory = KernelFactory(
            dim=n_bases,
            normalize=True,
            differences_by_dim=self.differences_by_dim,
            true_noise_std=sigma_n  # <-- Based on the noise you injected into training data
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
        K = degp_utils.rbf_kernel(
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
        diff_x_test_x_train = degp_utils.differences_by_dim_func(
            self.x_train, X_test, self.n_order
        )
        K_s = degp_utils.rbf_kernel(
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
                f_mean = utils.transform_predictions(
                    f_mean, self.mu_y, self.sigma_y, self.sigmas_x,
                    self.flattened_der_indicies, X_test
                )
            else:
                f_mean = self.mu_y + f_mean * self.sigma_y

        # If covariance is not requested, return mean only
        if not calc_cov:
            return f_mean

        # Compute test-test kernel and covariance
        diff_x_test_x_test = degp_utils.differences_by_dim_func(
            X_test, X_test, self.n_order
        )
        K_ss = degp_utils.rbf_kernel(
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
                                              :len(X_test)] - v[:, :len(X_test)].T @ v[:, :len(X_test)]

        # Normalize or return raw covariance
        if self.normalize:
            if return_deriv:
                f_var = utils.transform_cov(
                    f_cov, self.sigma_y, self.sigmas_x,
                    self.flattened_der_indicies, X_test
                )
            else:
                f_var = self.sigma_y**2 * np.diag(np.abs(f_cov))
        else:
            f_var = np.diag(np.abs(f_cov))

        return f_mean, f_var

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
