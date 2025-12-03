import numpy as np
import jetgp.utils as utils
from jetgp.kernel_funcs.kernel_funcs import KernelFactory
from jetgp.full_gddegp.optimizer import Optimizer
from jetgp.full_gddegp import gddegp_utils
from scipy.linalg import cho_solve, cho_factor, solve_triangular


class gddegp:
    """
    Global Directional Derivative-Enhanced Gaussian Process (GDDEGP) model.

    Supports point-wise directional derivatives with unique rays per point,
    hypercomplex representation, and automatic normalization. Includes methods
    for training, prediction, and uncertainty quantification using kernel methods.

    Parameters
    ----------
    x_train : ndarray
        Training input data of shape (n_samples, n_features).
    y_train : list or ndarray
        Training targets or list of directional derivatives.
    n_order : int
        Maximum derivative order.
    rays_list : list of ndarray
        List of ray arrays. rays_list[i] has shape (d, len(derivative_locations[i])).
    der_indices : list of lists
        Derivative multi-indices corresponding to each derivative term.
    derivative_locations : list of lists
        Which training points have which derivatives.
    normalize : bool, default=True
        Whether to normalize inputs and outputs.
    sigma_data : float or array-like, optional
        Observation noise standard deviation or diagonal noise values.
    kernel : str, default='SE'
        Kernel type ('SE', 'RQ', 'Matern', etc.).
    kernel_type : str, default='anisotropic'
        Kernel anisotropy ('anisotropic' or 'isotropic').
    smoothness_parameter : float, optional
        Smoothness parameter for Matern kernel.
    """

    def __init__(self, x_train, y_train, n_order, rays_list, der_indices,
                 derivative_locations=None, normalize=True, sigma_data=None,
                 kernel="SE", kernel_type="anisotropic", smoothness_parameter=None):

        if derivative_locations is None:
            raise Exception('Must provide derivative locations!')

        self.x_train = x_train
        self.y_train = y_train
        self.sigma_data = sigma_data
        self.n_order = n_order
        self.max_order = n_order
        self.rays_list = rays_list
        self.dim = x_train.shape[1]
        self.num_points = x_train.shape[0]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.normalize = normalize
        self.derivative_locations = derivative_locations

        self.flattened_der_indices = utils.flatten_der_indices(der_indices)

        if normalize:
            self.y_train, self.mu_y, self.sigma_y, self.sigmas_x, self.mus_x, sigma_data = \
                utils.normalize_y_data_directional(
                    x_train, y_train, sigma_data, self.flattened_der_indices)
            self.rays_list = utils.normalize_directions_2(self.sigmas_x, self.rays_list)
            self.x_train = utils.normalize_x_data_train(x_train)
        else:
            self.x_train = x_train
            self.y_train = utils.reshape_y_train(y_train)

        self.differences_by_dim = gddegp_utils.differences_by_dim_func(
            self.x_train, self.x_train,
            self.rays_list, self.rays_list,
            self.derivative_locations, self.derivative_locations,
            self.n_order
        )

        self.sigma_data = (
            np.zeros((self.y_train.shape[0], self.y_train.shape[0]))
            if sigma_data is None else 10 * np.diag(sigma_data)
        )

        self.kernel_factory = KernelFactory(
            dim=self.dim,
            normalize=self.normalize,
            n_order=self.max_order,
            differences_by_dim=self.differences_by_dim,
            smoothness_parameter=smoothness_parameter
        )
        self.kernel_func = self.kernel_factory.create_kernel(
            kernel_name=self.kernel,
            kernel_type=self.kernel_type
        )
        self.bounds = self.kernel_factory.bounds
        self.optimizer = Optimizer(self)

    def optimize_hyperparameters(self, *args, **kwargs):
        """
        Run the optimizer to find the best kernel hyperparameters.
        Returns optimized hyperparameter vector.
        """
        self.params = self.optimizer.optimize_hyperparameters(*args, **kwargs)
        return self.params

    def predict(self, X_test, params, rays_predict=None, calc_cov=False,
                return_deriv=False, derivs_to_predict=None):
        """
        Predict posterior mean and optional variance at test points.

        Parameters
        ----------
        X_test : ndarray
            Test input points of shape (n_test, n_features).
        params : ndarray
            Log-scaled kernel hyperparameters.
        rays_predict : list of ndarray, optional
            Rays at test points for derivative predictions.
        calc_cov : bool, default=False
            Whether to compute predictive variance.
        return_deriv : bool, default=False
            Whether to return derivative predictions.
        derivs_to_predict : list, optional
            Specific derivatives to predict. Must be subset of training derivatives.

        Returns
        -------
        f_mean : ndarray
            Predictive mean vector.
        f_var : ndarray, optional
            Predictive variance vector (only if calc_cov=True).
        """
        if return_deriv and rays_predict is None:
            raise Exception('Cannot make derivative predictions without rays')
        if not return_deriv and rays_predict is not None:
            raise Exception("No need to pass prediction rays if return_deriv is False")

        length_scales = params[:-1]
        sigma_n = params[-1]

        # Set up derivative prediction configuration
        if return_deriv:
            if derivs_to_predict is not None:
                invalid_derivs = [d for d in derivs_to_predict if d not in self.flattened_der_indices]
                if invalid_derivs:
                    raise ValueError(
                        f"The following derivative indices are not in the training set: {invalid_derivs}. "
                        f"Valid derivative indices are: {self.flattened_der_indices}"
                    )
                common_derivs = derivs_to_predict
            else:
                common_derivs = self.flattened_der_indices
                print(
                    f"Note: derivs_to_predict is None. Predictions will include all derivatives "
                    f"used in training: {self.flattened_der_indices}"
                )
        else:
            common_derivs = []

        # Build training kernel matrix
        phi_train = self.kernel_func(self.differences_by_dim, length_scales)
        self.n_bases = phi_train.get_active_bases()[-1]
        phi_exp_train = phi_train.get_all_derivs(self.n_bases, 2 * self.n_order)

        # Placeholder for powers (GDDEGP doesn't use sign powers like DEGP/DDEGP)
        powers = [0] * (len(self.flattened_der_indices) + 1)

        K = gddegp_utils.rbf_kernel(
            phi_train, phi_exp_train, self.n_order, self.n_bases,
            self.flattened_der_indices, 
            index=self.derivative_locations
        )
        K += (10 ** sigma_n) ** 2 * np.eye(K.shape[0])
        K += self.sigma_data ** 2

        # Solve linear system
        try:
            cho_solve_failed = False
            L, low = cho_factor(K, lower=True)
            alpha = cho_solve((L, low), self.y_train)
        except:
            cho_solve_failed = True
            alpha = np.linalg.solve(K, self.y_train)
            print('Warning: Cholesky decomposition failed via scipy, using standard np solve instead.')

        # Normalize test inputs and rays
        rays_test = rays_predict
        if self.normalize:
            X_test = utils.normalize_x_data_test(X_test, self.sigmas_x, self.mus_x)
            if rays_test is not None:
                rays_test = utils.normalize_directions_2(self.sigmas_x, rays_test)
                derivative_locations_test = [
                    list(range(X_test.shape[0])) for _ in range(len(common_derivs))
                ]
            else:
                derivative_locations_test = None
        else:
            if rays_test is not None:
                derivative_locations_test = [
                    list(range(X_test.shape[0])) for _ in range(len(common_derivs))
                ]
            else:
                derivative_locations_test = None

        # Compute train-test differences
        diff_x_train_x_test = gddegp_utils.differences_by_dim_func(
            self.x_train, X_test,
            self.rays_list, rays_test,
            self.derivative_locations, derivative_locations_test,
            self.n_order, return_deriv=return_deriv
        )

        # Compute train-test kernel
        phi_train_test = self.kernel_func(diff_x_train_x_test, length_scales)
        if return_deriv:
            phi_exp_train_test = phi_train_test.get_all_derivs(self.n_bases, 2 * self.n_order)
        else:
            phi_exp_train_test = phi_train_test.get_all_derivs(self.n_bases, self.n_order)

        K_s = gddegp_utils.rbf_kernel_predictions(
            phi_train_test, phi_exp_train_test, self.n_order, self.n_bases,
            self.flattened_der_indices, 
            return_deriv=return_deriv,
            index=self.derivative_locations,
            common_derivs=common_derivs
        )

        # Compute posterior mean
        f_mean = K_s @ alpha

        # Denormalize predictions
        if self.normalize:
            if return_deriv:
                f_mean = utils.transform_predictions_directional(
                    f_mean, self.mu_y, self.sigma_y, self.sigmas_x,
                    common_derivs, X_test)
            else:
                f_mean = self.mu_y + f_mean * self.sigma_y

        # Reshape predictions
        f_mean = f_mean.reshape(-1, 1)
        n = X_test.shape[0]
        m = f_mean.shape[0]
        num_derivs = m // n
        reshaped_mean = f_mean.reshape(num_derivs, n)

        if not calc_cov:
            return reshaped_mean

        # Compute test-test differences
        diff_x_test_x_test = gddegp_utils.differences_by_dim_func(
            X_test, X_test,
            rays_test, rays_test,
            derivative_locations_test, derivative_locations_test,
            self.n_order, return_deriv=return_deriv
        )

        # Compute test-test kernel
        phi_test_test = self.kernel_func(diff_x_test_x_test, length_scales)
        phi_exp_test_test = phi_test_test.get_all_derivs(self.n_bases, 2 * self.n_order)

        K_ss = gddegp_utils.rbf_kernel_predictions(
            phi_test_test, phi_exp_test_test, self.n_order, self.n_bases,
            self.flattened_der_indices,
            return_deriv=return_deriv,
            index=self.derivative_locations,
            common_derivs=common_derivs,
            calc_cov=True,
        )

        # Compute predictive covariance
        if cho_solve_failed:
            f_cov = K_ss - K_s @ np.linalg.inv(K) @ K_s.T
        else:
            v = solve_triangular(L, K_s.T, lower=low)
            f_cov = K_ss - v.T @ v

        # Transform covariance
        if self.normalize:
            if return_deriv:
                f_var = utils.transform_cov_directional(
                    f_cov, self.sigma_y, self.sigmas_x,
                    self.flattened_der_indices, X_test)
            else:
                f_var = self.sigma_y ** 2 * np.diag(np.abs(f_cov))
        else:
            f_var = np.diag(np.abs(f_cov))

        reshaped_var = f_var.reshape(num_derivs, n)
        return reshaped_mean, reshaped_var