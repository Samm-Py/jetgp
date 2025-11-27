import numpy as np
from numpy.linalg import cholesky, solve
from scipy.linalg import cho_solve, cho_factor, solve_triangular
from jetgp.full_degp import degp_utils
import jetgp.utils as utils
from jetgp.kernel_funcs.kernel_funcs import KernelFactory
from jetgp.full_degp.optimizer import Optimizer


class degp:
    def __init__(
        self,
        x_train,
        y_train,
        n_order,
        n_bases,
        der_indices,
        derivative_locations = None,
        normalize=True,
        sigma_data=None,
        kernel="SE",
        kernel_type="anisotropic",
        smoothness_parameter = None
    ):
        """
        Initialize the Derivative-Enhanced Gaussian Process (DEGP) model.

        Parameters:
        ----------
        x_train : ndarray
            Training input points.
        y_train : list of arrays
            Function values and derivatives.
        n_order : int
            Maximum derivative order.
        n_bases : int
            Total number of bases (function value + derivatives).
        der_indices : list of lists
            Derivative indices for each derivative component.
        normalize : bool, default=True
            Whether to normalize inputs and outputs.
        sigma_data : float or ndarray, optional
            Observational noise (can be None).
        kernel : str, default="SE"
            Kernel name.
        kernel_type : str, default="anisotropic"
            Kernel type (anisotropic or isotropic).
        """

        self.n_order = n_order
        self.n_bases = n_bases
        self.dim = x_train.shape[1]
        self.num_points = x_train.shape[0]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.der_indices = der_indices
        self.normalize = normalize
        self.derivative_locations = derivative_locations
        
        self.y_train_input = y_train
        self.x_train_input = x_train
        
        # Prepare indices and powers
        indices = der_indices
        self.flattened_der_indicies = utils.flatten_der_indices(indices)
        self.powers = utils.build_companion_array(
            n_bases, n_order, der_indices)
        # Normalize data if required
        if normalize:
            (
                self.y_train,
                self.mu_y,
                self.sigma_y,
                self.sigmas_x,
                self.mus_x,
                sigma_data,
            ) = utils.normalize_y_data(
                x_train, y_train, sigma_data, self.flattened_der_indicies
            )

            self.x_train = utils.normalize_x_data_train(x_train)
        else:
            self.x_train = x_train
            self.y_train = utils.reshape_y_train(y_train)

        # Compute differences for the kernel
        if kernel == 'SI':
            self.differences_by_dim = degp_utils.differences_by_dim_func_SI(
                self.x_train, self.x_train, n_order
            )
        else:
            self.differences_by_dim = degp_utils.differences_by_dim_func(
                self.x_train, self.x_train, n_order
            )

        # Initialize noise matrix
        self.sigma_data = (
            np.zeros((self.y_train.shape[0], self.y_train.shape[0]))
            if sigma_data is None
            else np.diag(sigma_data)
        )

        # Initialize kernel factory and optimizer
        self.kernel_factory = KernelFactory(
            dim=n_bases,
            normalize=normalize,
            differences_by_dim=self.differences_by_dim,
            n_order=n_order,
            smoothness_parameter=smoothness_parameter
        )
        self.kernel_func = self.kernel_factory.create_kernel(
            kernel_name=self.kernel, kernel_type=self.kernel_type
        )
        self.bounds = self.kernel_factory.bounds
        self.optimizer = Optimizer(self)

    def optimize_hyperparameters(self, *args, **kwargs):
        """
        Optimize model hyperparameters using the optimizer.
        """
        self.params = self.optimizer.optimize_hyperparameters(*args, **kwargs)
        return self.params

    def predict(self, X_test, params, calc_cov=False, return_deriv=False):
        """
        Compute posterior predictive mean and (optionally) covariance at X_test.

        Parameters:
        ----------
        X_test : ndarray
            Test input points.
        params : ndarray
            Hyperparameters (length scales and noise).
        calc_cov : bool, default=False
            Whether to compute the predictive covariance.
        return_deriv : bool, default=False
            Whether to return derivatives in the prediction.

        Returns:
        -------
        f_mean : ndarray
            Predictive mean.
        f_var : ndarray (if calc_cov=True)
            Predictive variance or covariance.
        """
        length_scales = params[:-1]
        sigma_n = params[-1]

        # Compute training kernel matrix and its Cholesky factor
        self.K = degp_utils.rbf_kernel(
            self.differences_by_dim,
            length_scales,
            self.n_order,
            self.n_bases,
            self.kernel_func,
            self.flattened_der_indicies,
            self.derivative_locations,
            self.powers,
        )
        self.K += (10**sigma_n) ** 2 * np.eye(self.K.shape[0])
        self.K += self.sigma_data**2
        try:
            cho_solve_failed = True
            L, low = cho_factor(self.K, lower=True)
            # L = cholesky(K)
            # low = True
            alpha = cho_solve(
                (L, low),
                self.y_train
            )
        except:
            cho_solve_failed = True
            alpha = np.linalg.solve(self.K, self.y_train)
            print(
                'Warning: Cholesky decomposition failed via scipy, using standard np solve instead.')
            # If Cholesky fails, fall back to standard solve

        if self.normalize:
            X_test = utils.normalize_x_data_test(
                X_test, self.sigmas_x, self.mus_x)
            
        if self.kernel == 'SI':
            diff_x_test_x_train = degp_utils.differences_by_dim_func_SI(
                self.x_train, X_test, self.n_order, return_deriv=return_deriv
            )
        else:
            diff_x_test_x_train = degp_utils.differences_by_dim_func(
                self.x_train, X_test, self.n_order, return_deriv=return_deriv
            )

        # Compute train-test kernel

        self.K_s = degp_utils.rbf_kernel_predictions(
            diff_x_test_x_train,
            length_scales,
            self.n_order,
            self.n_bases,
            self.kernel_func,
            self.flattened_der_indicies,
            self.derivative_locations,
            self.powers,
            return_deriv=return_deriv
        )

        # Compute posterior mean
        f_mean = self.K_s.T@alpha
        
        if self.normalize:
            if return_deriv:
                f_mean = utils.transform_predictions(
                    f_mean,
                    self.mu_y,
                    self.sigma_y,
                    self.sigmas_x,
                    self.flattened_der_indicies,
                    X_test,
                )
            else:
                f_mean = self.mu_y + f_mean * self.sigma_y
        n = X_test.shape[0]
        m = f_mean.shape[0]
        num_derivs = m // n
        # Reshape to (num_derivs, n), multiply by A, then flatten back
        reshaped_mean = f_mean.reshape(num_derivs, n)
        if not calc_cov:
            return reshaped_mean

        # Compute test-test kernel and covariance
        if self.kernel == 'SI':
            diff_x_test_x_test = degp_utils.differences_by_dim_func_SI(
                X_test, X_test, self.n_order, return_deriv=return_deriv
            )
        else:
            diff_x_test_x_test = degp_utils.differences_by_dim_func(
                X_test, X_test, self.n_order, return_deriv=return_deriv
            )

        self.K_ss = degp_utils.rbf_kernel_predictions(
            diff_x_test_x_test,
            length_scales,
            self.n_order,
            self.n_bases,
            self.kernel_func,
            self.flattened_der_indicies,
            self.derivative_locations,
            self.powers,
            return_deriv=return_deriv,
            calc_cov=calc_cov
        )

        if cho_solve_failed:
            f_cov = (
                self.K_ss - self.K_s.T @ np.linalg.solve(self.K, self.K_s)
                if return_deriv
                else self.K_ss[:len(X_test), :len(X_test)] - self.K_s[:, :len(X_test)].T @ np.linalg.solve(self.K, self.K_s[:, :len(X_test)])
            )
        else:
            v = solve_triangular(L, self.K_s, lower=low)

            f_cov = (
                self.K_ss - v.T @ v
                if return_deriv
                else self.K_ss[:len(X_test), :len(X_test)] - v[:, :len(X_test)].T @ v[:, :len(X_test)]
            )

        # Normalize or return raw covariance
        if self.normalize:
            if return_deriv:
                f_var = utils.transform_cov(
                    f_cov,
                    self.sigma_y,
                    self.sigmas_x,
                    self.flattened_der_indicies,
                    X_test,
                )
            else:
                f_var = self.sigma_y**2 * np.diag(np.abs(f_cov))
        else:
            f_var = np.diag(np.abs(f_cov))

        # Reshape to (num_derivs, n), multiply by A, then flatten back
        reshaped_var = f_var.reshape(num_derivs, n)
        return reshaped_mean, reshaped_var
