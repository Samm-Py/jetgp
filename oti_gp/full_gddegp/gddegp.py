import numpy as np
from numpy.linalg import cholesky, solve
import utils as utils
from kernel_funcs.kernel_funcs import KernelFactory
from full_gddegp.optimizer import Optimizer
from full_gddegp import gddegp_utils
from line_profiler import profile


class gddegp:
    """
    Directional Derivative-Enhanced Gaussian Process (dDEGP) model.

    Supports multiple directional derivatives, hypercomplex representation,
    and automatic normalization. Includes methods for training, prediction,
    and uncertainty quantification using kernel methods.

    Parameters
    ----------
    x_train : ndarray
        Training input data of shape (n_samples, n_features).
    y_train : list or ndarray
        Training targets or list of directional derivatives.
    n_order : int
        Maximum derivative order.
    der_indices : list of lists
        Derivative multi-indices corresponding to each derivative term.
    rays : ndarray
        Array of shape (d, n_rays), where each column is a direction vector.
    normalize : bool, default=True
        Whether to normalize inputs and outputs.
    sigma_data : float or array-like, optional
        Observation noise standard deviation or diagonal noise values.
    kernel : str, default='SE'
        Kernel type ('SE', 'RQ', 'Matern', etc.).
    kernel_type : str, default='anisotropic'
        Kernel anisotropy ('anisotropic' or 'isotropic').
    """

    def __init__(self, x_train, y_train, n_order, rays_array,
                 normalize=True, sigma_data=None, kernel="SE", kernel_type="anisotropic"):
        self.x_train = x_train
        self.y_train = y_train
        self.sigma_data = sigma_data
        self.n_order = n_order
        self.rays_array = rays_array
        self.dim = x_train.shape[1]
        self.num_points = x_train.shape[0]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.normalize = normalize
        # self.num_directions = rays_array.shape[1]
        indices = gddegp_utils.make_der_indices(
            1, self.n_order)

        self.flattened_der_indicies = utils.flatten_der_indices(indices)

        if normalize:
            self.y_train, self.mu_y, self.sigma_y, self.sigmas_x, self.mus_x, sigma_data = utils.normalize_y_data_directional(
                x_train, y_train, sigma_data, self.flattened_der_indicies)
            self.rays_array = utils.normalize_directions_2(
                self.sigmas_x, self.rays_array)
            self.x_train = utils.normalize_x_data_train(x_train)
        else:
            self.x_train = x_train
            self.y_train = utils.reshape_y_train(y_train)

        self.differences_by_dim = gddegp_utils.differences_by_dim_func(
            self.x_train, self.x_train, self.rays_array, self.rays_array, n_order)

        self.sigma_data = (
            np.zeros((self.y_train.shape[0], self.y_train.shape[0]))
            if sigma_data is None else 10*np.diag(sigma_data))

        self.kernel_factory = KernelFactory(
            dim=self.dim,
            normalize=self.normalize,
            n_order=self.n_order,
            differences_by_dim=self.differences_by_dim)
        self.kernel_func = self.kernel_factory.create_kernel(
            kernel_name=self.kernel,
            kernel_type=self.kernel_type)
        self.bounds = self.kernel_factory.bounds
        self.optimizer = Optimizer(self)

    def optimize_hyperparameters(self, *args, **kwargs):
        """
        Run the optimizer to find the best kernel hyperparameters.
        Returns optimized hyperparameter vector.
        """
        return self.optimizer.optimize_hyperparameters(*args, **kwargs)

    @profile
    def predict(self, X_test, rays_predict, params, calc_cov=False, return_deriv=False):
        """
        Predict posterior mean and optional variance at test points.

        Parameters
        ----------
        X_test : ndarray
            Test input points of shape (n_test, n_features).
        params : ndarray
            Log-scaled kernel hyperparameters.
        calc_cov : bool, default=False
            Whether to compute predictive variance.
        return_deriv : bool, default=False
            Whether to return derivative predictions.

        Returns
        -------
        f_mean : ndarray
            Predictive mean vector.
        f_var : ndarray, optional
            Predictive variance vector (only if calc_cov=True).
        """
        length_scales = params[:-1]
        sigma_n = params[-1]

        K = gddegp_utils.rbf_kernel(
            self.differences_by_dim,
            length_scales,
            self.n_order,
            self.num_points,
            self.kernel_func,
        )
        K += (10**sigma_n) ** 2 * np.eye(K.shape[0])
        K += self.sigma_data**2

        L = cholesky(K)
        alpha = solve(L.T, solve(L, self.y_train))

        # theta = np.pi/4
        # unit_v = np.array([[np.cos(theta)], [np.sin(theta)]])  # shape (2,1)
        rays_test = rays_predict
        if self.normalize:
            X_test = utils.normalize_x_data_test(
                X_test, self.sigmas_x, self.mus_x)
            rays_test = utils.normalize_directions_2(
                self.sigmas_x, rays_test)
        diff_x_test_x_train = gddegp_utils.differences_by_dim_func(
            self.x_train, X_test, self.rays_array, rays_test, self.n_order, return_deriv=return_deriv)
        K_s = gddegp_utils.rbf_kernel(
            diff_x_test_x_train, length_scales, self.n_order,
            X_test.shape[0],
            self.kernel_func, return_deriv=return_deriv)

        f_mean = K_s.T@alpha

        if self.normalize:
            if return_deriv:
                f_mean = utils.transform_predictions_directional(
                    f_mean, self.mu_y, self.sigma_y, self.sigmas_x,
                    self.flattened_der_indicies, X_test)
            else:
                f_mean = self.mu_y + f_mean * self.sigma_y

        if not calc_cov:
            return f_mean

        diff_x_test_x_test = gddegp_utils.differences_by_dim_func(
            X_test, X_test, rays_test, rays_test, self.n_order)
        K_ss = gddegp_utils.rbf_kernel(
            diff_x_test_x_test, length_scales, self.n_order, X_test.shape[0],
            self.kernel_func, return_deriv=return_deriv)

        v = solve(L, K_s)
        if not return_deriv:
            f_cov = K_ss[:X_test.shape[0], :] - v.T @ v
        else:
            f_cov = K_ss - v.T @ v

        if self.normalize:
            if return_deriv:
                f_var = utils.transform_cov_directrional(
                    f_cov, self.sigma_y, self.sigmas_x,
                    self.flattened_der_indicies, X_test)
            else:
                f_var = self.sigma_y**2 * np.diag(np.abs(f_cov))
        else:
            f_var = np.diag(np.abs(f_cov))

        return f_mean, f_var
