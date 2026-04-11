import numpy as np
import jetgp.utils as utils
from jetgp.kernel_funcs.kernel_funcs import KernelFactory, get_oti_module
from jetgp.full_gddegp_sparse.optimizer import Optimizer
from jetgp.full_gddegp_sparse import gddegp_utils
from scipy.linalg import cho_solve, cho_factor, solve_triangular
import warnings


class gddegp:
    """
    Sparse Cholesky variant of the GDDEGP model.

    Adds sparse inverse-Cholesky acceleration for NLML evaluation during
    hyperparameter optimisation.  Prediction always uses the exact dense
    Cholesky solve.

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
    n_bases : int, optional
        Override the OTI space size. By default ``2 * n_direction_types``.
    normalize : bool, default=True
        Whether to normalize inputs and outputs.
    sigma_data : float or array-like, optional
        Observation noise standard deviation or diagonal noise values.
    kernel : str, default='SE'
        Kernel type.
    kernel_type : str, default='anisotropic'
        Kernel anisotropy.
    smoothness_parameter : float, optional
        Smoothness parameter for Matern kernel.
    rho : float, default=3.0
        Sparsity radius multiplier.
    use_supernodes : bool, default=True
        If True, aggregate columns into supernodes.
    supernode_lam : float, default=1.5
        Lambda parameter for supernode merging.
    """

    def __init__(self, x_train, y_train, n_order, rays_list, der_indices,
                 derivative_locations=None, n_bases=None, normalize=True,
                 sigma_data=None, kernel="SE", kernel_type="anisotropic",
                 smoothness_parameter=None,
                 rho=3.0, use_supernodes=True, supernode_lam=1.5):

        if n_order > 0 and derivative_locations is None:
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
        self.der_indices = der_indices

        self.flattened_der_indices = utils.flatten_der_indices(der_indices)
        if n_bases is not None:
            self.n_bases = n_bases
        else:
            self.n_bases = 2 * len(self.flattened_der_indices)
        self.oti = get_oti_module(self.n_bases, n_order)

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
            n_order, self.oti, return_deriv=True
        )

        self.sigma_data = (
            np.zeros((self.y_train.shape[0], self.y_train.shape[0]))
            if sigma_data is None else np.diag(sigma_data)
        )
        self.sigma_data_sq_diag = (
            np.zeros(self.y_train.shape[0])
            if sigma_data is None
            else np.asarray(sigma_data) ** 2
        )

        self.kernel_factory = KernelFactory(
            dim=self.dim,
            normalize=self.normalize,
            n_order=self.max_order,
            differences_by_dim=self.differences_by_dim,
            smoothness_parameter=smoothness_parameter,
            oti_module=self.oti,
            sparse_diffs=False
        )
        self.kernel_func = self.kernel_factory.create_kernel(
            kernel_name=self.kernel,
            kernel_type=self.kernel_type
        )
        self.bounds = self.kernel_factory.bounds
        self.optimizer = Optimizer(self)

        # Sparse Cholesky setup
        self.rho = rho
        self.use_supernodes = use_supernodes
        self.supernode_lam = supernode_lam
        self._setup_sparse_cholesky()

    def _setup_sparse_cholesky(self):
        """Precompute MMD ordering, fill-distances, and sparsity pattern."""
        from jetgp.full_gddegp_sparse.sparse_cholesky import (
            mmd_ordering, build_sparsity_pattern, build_supernodes,
            expand_mmd_permutation, expand_sparsity_to_blocks,
            expand_supernodes_to_blocks,
        )
        X = self.x_train
        self.mmd_P, self.mmd_l = mmd_ordering(X)
        X_ord = X[self.mmd_P]
        self.sparse_S = build_sparsity_pattern(X_ord, self.mmd_l, self.rho)

        self.mmd_P_full, self._phys_to_rows = expand_mmd_permutation(
            self.mmd_P, self.num_points, self.derivative_locations
        )
        self.sparse_S_full = expand_sparsity_to_blocks(self.sparse_S, self._phys_to_rows)
        self.sparse_S_full_arr = {
            j: np.asarray(s, dtype=np.intp) for j, s in self.sparse_S_full.items()
        }

        N_total = len(self.mmd_P_full)
        total_nb = sum(len(s) for s in self.sparse_S_full.values())
        self.sparse_fill_fraction = total_nb / (N_total * N_total)
        self._use_dense_factor = self.sparse_fill_fraction > 0.25

        if self.use_supernodes:
            phys_sns = build_supernodes(
                X_ord, self.mmd_l, self.sparse_S, lam=self.supernode_lam
            )
            self.sparse_supernodes = phys_sns
            self.sparse_supernodes_full = expand_supernodes_to_blocks(
                phys_sns, self._phys_to_rows
            )
            for sn in self.sparse_supernodes_full:
                sn['children_arr'] = np.asarray(sn['children'])
                ch_pos = {c: i for i, c in enumerate(sn['children'])}
                sn['ch_pos'] = ch_pos
                sn['parent_positions'] = np.array(
                    [ch_pos[p] for p in sn['parents']]
                )
        else:
            self.sparse_supernodes = None
            self.sparse_supernodes_full = None

    def optimize_hyperparameters(self, *args, **kwargs):
        """Run the optimizer. Returns optimized hyperparameter vector."""
        self.params = self.optimizer.optimize_hyperparameters(*args, **kwargs)
        return self.params

    def predict(self, X_test, params, rays_predict=None, calc_cov=False,
                return_deriv=False, derivs_to_predict=None):
        """
        Predict posterior mean and optional variance at test points.
        Uses exact dense Cholesky solve (not sparse approximation).
        """
        n_predict = X_test.shape[0]

        # Handle missing rays_predict when derivatives are requested
        if return_deriv and rays_predict is None:
            n_rays = len(self.flattened_der_indices)
            warnings.warn(
                f"No rays_predict provided for derivative predictions. "
                f"Predictions will be made along coordinate axes.",
                UserWarning
            )
            rays_predict = []
            for i in range(n_rays):
                axis_idx = i % self.dim
                ray_array = np.zeros((self.dim, n_predict))
                ray_array[axis_idx, :] = 1.0
                rays_predict.append(ray_array)

        if not return_deriv and rays_predict is not None:
            warnings.warn(
                "rays_predict was provided but return_deriv=False. "
                "The provided rays will be ignored.",
                UserWarning
            )

        if return_deriv and rays_predict is not None:
            if len(self.rays_list) > 0 and len(rays_predict) > len(self.rays_list):
                raise ValueError(
                    f"Number of prediction rays ({len(rays_predict)}) exceeds the number of "
                    f"training rays ({len(self.rays_list)})."
                )
            for i, ray_array in enumerate(rays_predict):
                if not isinstance(ray_array, np.ndarray):
                    raise TypeError(
                        f"Ray array {i} must be a numpy ndarray, got {type(ray_array).__name__}."
                    )
                if ray_array.ndim != 2:
                    raise ValueError(
                        f"Ray array {i} must be 2-dimensional, got {ray_array.ndim} dimensions."
                    )
                if ray_array.shape[0] != self.dim:
                    raise ValueError(
                        f"Ray array {i} has {ray_array.shape[0]} rows, expected {self.dim}."
                    )
                if ray_array.shape[1] != n_predict:
                    raise ValueError(
                        f"Ray array {i} has {ray_array.shape[1]} columns, expected {n_predict}."
                    )

        length_scales = params[:-1]
        sigma_n = params[-1]

        if return_deriv:
            if derivs_to_predict is not None:
                common_derivs = derivs_to_predict
            else:
                common_derivs = self.flattened_der_indices

            required_order = max(
                sum(pair[1] for pair in deriv_spec)
                for deriv_spec in common_derivs
            )
            predict_order = max(required_order, self.n_order)

            if predict_order > self.n_order:
                predict_oti = get_oti_module(self.n_bases, predict_order)
                smoothness_param = getattr(self.kernel_factory, 'alpha', None)
                predict_kernel_factory = KernelFactory(
                    dim=self.dim,
                    normalize=self.normalize,
                    differences_by_dim=self.differences_by_dim,
                    n_order=predict_order,
                    smoothness_parameter=smoothness_param,
                    oti_module=predict_oti,
                    sparse_diffs=False
                )
                predict_kernel_func = predict_kernel_factory.create_kernel(
                    kernel_name=self.kernel, kernel_type=self.kernel_type
                )
            else:
                predict_oti = self.oti
                predict_kernel_func = self.kernel_func
        else:
            common_derivs = []
            predict_order = self.n_order
            predict_oti = self.oti
            predict_kernel_func = self.kernel_func

        _cache_hit = (
            hasattr(self, '_cached_params')
            and self._cached_params is not None
            and np.array_equal(self._cached_params, params)
            and getattr(self, '_cached_L', None) is not None
        )

        if _cache_hit:
            L = self._cached_L
            low = self._cached_low
            alpha = self._cached_alpha
            self.n_bases = self._cached_n_bases
            cho_solve_failed = False
        else:
            phi_train = self.kernel_func(self.differences_by_dim, length_scales)
            if self.n_order == 0:
                self.n_bases = 0
                phi_exp_train = phi_train.real[np.newaxis, :, :]
            else:
                active = phi_train.get_active_bases()
                self.n_bases = max(self.n_bases, active[-1] if active else 0)
                phi_exp_train = phi_train.get_all_derivs(self.n_bases, 2 * self.n_order)

            powers = [0] * (len(self.flattened_der_indices) + 1)

            K = gddegp_utils.rbf_kernel(
                phi_train, phi_exp_train, self.n_order, self.n_bases,
                self.flattened_der_indices,
                index=self.derivative_locations
            )
            K.flat[::K.shape[0] + 1] += (10 ** sigma_n) ** 2
            K += self.sigma_data ** 2

            try:
                cho_solve_failed = False
                L, low = cho_factor(K, lower=True)
                alpha = cho_solve((L, low), self.y_train)
            except Exception:
                cho_solve_failed = True
                alpha = np.linalg.solve(K, self.y_train)
                L, low = None, None

            self._cached_L = L
            self._cached_low = low
            self._cached_alpha = alpha
            self._cached_n_bases = self.n_bases
            self._cached_params = params.copy()

        rays_test = rays_predict

        if self.normalize:
            X_test = utils.normalize_x_data_test(X_test, self.sigmas_x, self.mus_x)

        if not return_deriv:
            rays_test = None
            derivative_locations_test = None
        else:
            derivative_locations_test = [
                list(range(X_test.shape[0])) for _ in range(len(common_derivs))]
            if self.normalize:
                rays_test = utils.normalize_directions_2(self.sigmas_x, rays_test)

        diff_x_train_x_test = gddegp_utils.differences_by_dim_func(
            self.x_train, X_test,
            self.rays_list, rays_test,
            self.derivative_locations, derivative_locations_test,
            predict_order, predict_oti, return_deriv=return_deriv
        )

        phi_train_test = predict_kernel_func(diff_x_train_x_test, length_scales)
        if predict_order > 0:
            if return_deriv:
                phi_exp_train_test = phi_train_test.get_all_derivs(self.n_bases, 2 * predict_order)
            else:
                phi_exp_train_test = phi_train_test.get_all_derivs(self.n_bases, predict_order)
        else:
            phi_exp_train_test = phi_train_test.real[np.newaxis, :, :]

        K_s = gddegp_utils.rbf_kernel_predictions(
            phi_train_test, phi_exp_train_test, predict_order, self.n_bases,
            self.flattened_der_indices,
            return_deriv=return_deriv,
            index=self.derivative_locations,
            common_derivs=common_derivs
        )

        f_mean = K_s @ alpha

        if self.normalize:
            if return_deriv:
                f_mean = utils.transform_predictions_directional(
                    f_mean, self.mu_y, self.sigma_y, self.sigmas_x,
                    common_derivs, X_test)
            else:
                f_mean = self.mu_y + f_mean * self.sigma_y

        f_mean = f_mean.reshape(-1, 1)
        n = X_test.shape[0]
        m = f_mean.shape[0]
        num_derivs = m // n
        reshaped_mean = f_mean.reshape(num_derivs, n)

        if not calc_cov:
            return reshaped_mean

        diff_x_test_x_test = gddegp_utils.differences_by_dim_func(
            X_test, X_test,
            rays_test, rays_test,
            derivative_locations_test, derivative_locations_test,
            predict_order, predict_oti, return_deriv=return_deriv
        )

        phi_test_test = predict_kernel_func(diff_x_test_x_test, length_scales)
        if predict_order > 0:
            phi_exp_test_test = phi_test_test.get_all_derivs(self.n_bases, 2 * predict_order)
        else:
            phi_exp_test_test = phi_test_test.real[np.newaxis, :, :]

        K_ss = gddegp_utils.rbf_kernel_predictions(
            phi_test_test, phi_exp_test_test, predict_order, self.n_bases,
            self.flattened_der_indices,
            return_deriv=return_deriv,
            index=derivative_locations_test,
            common_derivs=common_derivs,
            calc_cov=True,
        )

        if cho_solve_failed:
            v_fallback = np.linalg.solve(K, K_s.T)
            f_cov = K_ss - K_s @ v_fallback
        else:
            v = solve_triangular(L, K_s.T, lower=low)
            f_cov = K_ss - v.T @ v

        if self.normalize:
            if return_deriv:
                f_var = utils.transform_cov_directional(
                    f_cov, self.sigma_y, self.sigmas_x,
                    common_derivs, X_test)
            else:
                f_var = self.sigma_y ** 2 * np.diag(np.abs(f_cov))
        else:
            f_var = np.diag(np.abs(f_cov))

        reshaped_var = f_var.reshape(num_derivs, n)
        return reshaped_mean, reshaped_var
