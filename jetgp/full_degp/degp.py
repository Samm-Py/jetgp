import numpy as np
from numpy.linalg import cholesky, solve
from scipy.linalg import cho_solve, cho_factor, solve_triangular
from jetgp.full_degp import degp_utils
import jetgp.utils as utils
from jetgp.kernel_funcs.kernel_funcs import KernelFactory, get_oti_module
from jetgp.full_degp.optimizer import Optimizer


class degp:
    """
    Derivative-Enhanced Gaussian Process (DEGP) model.

    Supports coordinate-aligned partial derivatives, hypercomplex representation,
    and automatic normalization. Includes methods for training, prediction,
    and uncertainty quantification using kernel methods.

    Parameters
    ----------
    x_train : ndarray
        Training input data of shape (n_samples, n_features).
    y_train : list or ndarray
        Training targets or list of partial derivatives.
    n_order : int
        Maximum derivative order.
    n_bases : int
        Number of input dimensions.
    der_indices : list of lists
        Derivative multi-indices corresponding to each derivative term.
    derivative_locations : list of lists
        Which training points have which derivatives.
    normalize : bool, default=True
        Whether to normalize inputs and outputs.
    sigma_data : float or array-like, optional
        Observation noise standard deviation or diagonal noise values.
    kernel : str, default='SE'
        Kernel type ('SE', 'RQ', 'Matern', 'SI', etc.).
    kernel_type : str, default='anisotropic'
        Kernel anisotropy ('anisotropic' or 'isotropic').
    smoothness_parameter : float, optional
        Smoothness parameter for Matern kernel.
    """

    def __init__(
        self,
        x_train,
        y_train,
        n_order,
        n_bases,
        der_indices,
        derivative_locations=None,
        normalize=True,
        sigma_data=None,
        kernel="SE",
        kernel_type="anisotropic",
        smoothness_parameter=None
    ):
        if n_order > 0 and derivative_locations is None:
            import warnings
            # Count total number of derivative components across all orders
            n_derivs = sum(len(order_derivs) for order_derivs in der_indices)
            n_train = len(x_train)
            derivative_locations = [[i for i in range(n_train)] for _ in range(n_derivs)]
            warnings.warn(
                f"derivative_locations not provided. Assuming all {n_derivs} derivative(s) "
                f"are available at all {n_train} training point(s).",
                UserWarning
        )
            
        elif der_indices is None and n_order == 0:
            der_indices = []
            derivative_locations = []
        self.n_order = n_order
        self.n_bases = n_bases
        self.dim = x_train.shape[1]
        self.num_points = x_train.shape[0]
        self.kernel = kernel
        self.kernel_type = kernel_type
        self.der_indices = der_indices
        self.normalize = normalize
        self.derivative_locations = derivative_locations
        self.oti = get_oti_module(n_bases, n_order)
        self.y_train_input = y_train
        self.x_train_input = x_train

        # Prepare indices and powers
        self.flattened_der_indices = utils.flatten_der_indices(der_indices)
        self.powers = utils.build_companion_array(n_bases, n_order, der_indices)

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
                x_train, y_train, sigma_data, self.flattened_der_indices
            )
            self.x_train = utils.normalize_x_data_train(x_train)
        else:
            self.x_train = x_train
            self.y_train = utils.reshape_y_train(y_train)

        # Compute differences for the kernel
        # if kernel == 'SI':
        #     self.differences_by_dim = degp_utils.differences_by_dim_func_SI(
        #         self.x_train, self.x_train, n_order
        #     )
        # else:
        self.differences_by_dim = degp_utils.differences_by_dim_func(
            self.x_train, self.x_train, n_order, self.oti
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
            smoothness_parameter=smoothness_parameter,
            oti_module=self.oti
        )
        self.kernel_func = self.kernel_factory.create_kernel(
            kernel_name=self.kernel, kernel_type=self.kernel_type
        )
        self.bounds = self.kernel_factory.bounds
        self.optimizer = Optimizer(self)

    def optimize_hyperparameters(self, *args, **kwargs):
        """
        Optimize model hyperparameters using the optimizer.
        Returns optimized hyperparameter vector.
        """
        self.params = self.optimizer.optimize_hyperparameters(*args, **kwargs)
        return self.params

    def predict(self, X_test, params, calc_cov=False, return_deriv=False, derivs_to_predict=None):
        """
        Compute posterior predictive mean and (optionally) covariance at X_test.

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
        derivs_to_predict : list, optional
            Specific derivatives to predict. Can include derivatives not present in the
            training set — the cross-covariance K_* is constructed from kernel derivatives
            and does not require the requested derivative to have been observed during
            training. Each entry must be a valid derivative spec within n_bases and n_order
            (e.g. ``[[3, 1]]`` for df/dx3 in a first-order model).
            If None, defaults to all derivatives used in training.

        Returns
        -------
        f_mean : ndarray
            Predictive mean vector.
        f_var : ndarray, optional
            Predictive variance vector (only if calc_cov=True).
        """
        length_scales = params[:-1]
        sigma_n = params[-1]

        # Set up derivative prediction configuration
        if return_deriv:
            if derivs_to_predict is not None:
                common_derivs = derivs_to_predict
            else:
                common_derivs = self.flattened_der_indices
                print(
                    f"Note: derivs_to_predict is None. Predictions will include all derivatives "
                    f"used in training: {self.flattened_der_indices}"
                )
            self.powers_predict = utils.build_companion_array_predict(
                self.n_bases, self.n_order, common_derivs)
        else:
            common_derivs = []
            self.powers_predict = None

        # Build training kernel matrix
        phi_train = self.kernel_func(self.differences_by_dim, length_scales)
        
        if self.n_order > 0:
            phi_exp_train = phi_train.get_all_derivs(self.n_bases, 2 * self.n_order)
        else:
            phi_exp_train = phi_train.real
            phi_exp_train = phi_exp_train[np.newaxis, :, :]

        K = degp_utils.rbf_kernel(
            phi_train, phi_exp_train, self.n_order, self.n_bases,
            self.flattened_der_indices, self.powers,
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

        # Normalize test inputs
        if self.normalize:
            X_test = utils.normalize_x_data_test(X_test, self.sigmas_x, self.mus_x)

        # Set up test derivative locations
        if return_deriv:
            derivative_locations_test = [
                list(range(X_test.shape[0])) for _ in range(len(common_derivs))]
        else:
            derivative_locations_test = None

        # Compute train-test differences
        # if self.kernel == 'SI':
        #     diff_x_test_x_train = degp_utils.differences_by_dim_func_SI(
        #         self.x_train, X_test, self.n_order, return_deriv=return_deriv
        #     )
        # else:
        diff_x_test_x_train = degp_utils.differences_by_dim_func(
            self.x_train, X_test, self.n_order, self.oti, return_deriv=return_deriv
        )

        # Compute train-test kernel
        phi_train_test = self.kernel_func(diff_x_test_x_train, length_scales)
        if self.n_order > 0:
            if return_deriv:
                phi_exp_train_test = phi_train_test.get_all_derivs(self.n_bases, 2 * self.n_order)
            else:
                phi_exp_train_test = phi_train_test.get_all_derivs(self.n_bases, self.n_order)
        else:
            phi_exp_train_test = phi_train_test.real
            phi_exp_train_test =  phi_exp_train_test[np.newaxis, :, :]

        K_s = degp_utils.rbf_kernel_predictions(
            phi_train_test, phi_exp_train_test, self.n_order, self.n_bases,
            self.flattened_der_indices, self.powers,
            return_deriv=return_deriv,
            index=self.derivative_locations,
            common_derivs=common_derivs,
            powers_predict=self.powers_predict
        )

        # Compute posterior mean
        f_mean = K_s.T @ alpha

        # Denormalize predictions
        if self.normalize:
            if return_deriv:
                f_mean = utils.transform_predictions(
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
        # if self.kernel == 'SI':
        #     diff_x_test_x_test = degp_utils.differences_by_dim_func_SI(
        #         X_test, X_test, self.n_order, return_deriv=return_deriv
        #     )
        # else:
        diff_x_test_x_test = degp_utils.differences_by_dim_func(
            X_test, X_test, self.n_order, self.oti, return_deriv=return_deriv
        )

        # Compute test-test kernel
        phi_test_test = self.kernel_func(diff_x_test_x_test, length_scales)
        if self.n_order > 0:
            phi_exp_test_test = phi_test_test.get_all_derivs(self.n_bases, 2 * self.n_order)
        else:
            phi_exp_test_test = phi_test_test.real
            phi_exp_test_test = phi_exp_test_test[np.newaxis,:,:]

        K_ss = degp_utils.rbf_kernel_predictions(
            phi_test_test, phi_exp_test_test, self.n_order, self.n_bases,
            self.flattened_der_indices, self.powers,
            return_deriv=return_deriv,
            index=derivative_locations_test,
            common_derivs=common_derivs,
            calc_cov=True,
            powers_predict=self.powers_predict
        )

        # Compute predictive covariance
        if cho_solve_failed:
            f_cov = K_ss - K_s.T @ np.linalg.inv(K) @ K_s
        else:
            v = solve_triangular(L, K_s, lower=low)
            f_cov = K_ss - v.T @ v

        # Transform covariance
        if self.normalize:
            if return_deriv:
                f_var = utils.transform_cov(
                    f_cov, self.sigma_y, self.sigmas_x,
                    common_derivs, X_test)
            else:
                f_var = self.sigma_y ** 2 * np.diag(np.abs(f_cov))
        else:
            f_var = np.diag(np.abs(f_cov))

        reshaped_var = f_var.reshape(num_derivs, n)
        return reshaped_mean, reshaped_var