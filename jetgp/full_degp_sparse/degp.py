import numpy as np
from numpy.linalg import cholesky, solve
from scipy.linalg import cho_solve, cho_factor, solve_triangular
from jetgp.full_degp_sparse import degp_utils  # noqa: E402 (sparse copy)
import jetgp.utils as utils
from jetgp.kernel_funcs.kernel_funcs import KernelFactory, get_oti_module
from jetgp.full_degp_sparse.optimizer import Optimizer
from jetgp.full_degp_sparse.sparse_cholesky import mmd_ordering, build_sparsity_pattern


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
    rho : float, default=3.0
        Sparsity radius multiplier for the geometric criterion
        dist(x_P(i), x_P(j)) <= rho * l(j). Larger values give denser
        sparsity patterns and more accurate (but slower) approximations.
    use_supernodes : bool, default=True
        If True, aggregate columns into supernodes to reduce the number of
        local factorisations during sparse U construction.
    supernode_lam : float, default=1.5
        Merging threshold for supernode construction.
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
        smoothness_parameter=None,
        rho=1.0,
        use_supernodes=True,
        supernode_lam=1.5,
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

        # Sparse Cholesky: precompute MMD ordering and sparsity pattern once.
        # The pattern depends only on (x_train, rho) and NOT on hyperparameters,
        # so it is safe to compute here and reuse across all NLML evaluations.
        self.rho = rho
        self.use_supernodes = use_supernodes
        self.supernode_lam = supernode_lam
        self._setup_sparse_cholesky()

    def _setup_sparse_cholesky(self):
        """
        Precompute the MMD ordering, fill-distances, and sparsity pattern.

        Called once during __init__. Stores:
            self.mmd_P             : physical permutation (size N)
            self.mmd_l             : fill-distances (size N)
            self.sparse_S          : physical sparsity pattern
            self.mmd_P_full        : full K-matrix permutation (size N_total)
            self.sparse_S_full     : sparsity pattern in P_full-indexed space
            self.sparse_supernodes_full : supernodes in P_full-indexed space
        """
        from jetgp.full_degp_sparse.sparse_cholesky import (
            mmd_ordering, build_sparsity_pattern, build_supernodes,
            expand_mmd_permutation, expand_sparsity_to_blocks,
            expand_supernodes_to_blocks,
        )
        X = self.x_train  # already normalised if normalize=True
        self.mmd_P, self.mmd_l = mmd_ordering(X)
        X_ord = X[self.mmd_P]
        self.sparse_S = build_sparsity_pattern(X_ord, self.mmd_l, self.rho)

        # Expand physical ordering to cover all K-matrix rows (function + derivatives)
        self.mmd_P_full, self._phys_to_rows = expand_mmd_permutation(
            self.mmd_P, self.num_points, self.derivative_locations
        )
        self.sparse_S_full = expand_sparsity_to_blocks(self.sparse_S, self._phys_to_rows)
        # Pre-convert sparsity sets to numpy arrays for build_U
        self.sparse_S_full_arr = {
            j: np.asarray(s, dtype=np.intp) for j, s in self.sparse_S_full.items()
        }

        # Compute fill fraction to decide sparse vs dense factorisation path.
        # When neighbourhoods are nearly full, dense Cholesky is faster than
        # many overlapping block factorisations.
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
            # Pre-convert supernode index lists to numpy arrays and build
            # position lookups so build_U_supernodes avoids per-call overhead.
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

            # Determine prediction order from requested derivatives
            required_order = max(
                sum(pair[1] for pair in deriv_spec)
                for deriv_spec in common_derivs
            )
            predict_order = max(required_order, self.n_order)

            if predict_order > self.n_order:
                predict_oti = get_oti_module(self.n_bases, predict_order)
                smoothness_param = getattr(self.kernel_factory, 'alpha', None)
                predict_kernel_factory = KernelFactory(
                    dim=self.n_bases,
                    normalize=self.normalize,
                    differences_by_dim=self.differences_by_dim,
                    n_order=predict_order,
                    smoothness_parameter=smoothness_param,
                    oti_module=predict_oti
                )
                predict_kernel_func = predict_kernel_factory.create_kernel(
                    kernel_name=self.kernel, kernel_type=self.kernel_type
                )
            else:
                predict_oti = self.oti
                predict_kernel_func = self.kernel_func

            self.powers_predict = utils.build_companion_array_predict(
                self.n_bases, predict_order, common_derivs)
        else:
            common_derivs = []
            self.powers_predict = None
            predict_order = self.n_order
            predict_oti = self.oti
            predict_kernel_func = self.kernel_func

        # Reuse cached exact Cholesky + alpha from a previous predict call
        # if available.  Skip the cache when _cached_L is None — that means
        # the cache was set by the sparse optimiser path (approximate alpha).
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
            cho_solve_failed = False
        else:
            # Build training kernel matrix (no cache available)
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

            # Final prediction always uses exact K solve (the sparse
            # approximation is only for NLML during hyperparameter optimisation).
            try:
                L, low = cho_factor(K, lower=True)
                alpha = cho_solve((L, low), self.y_train)
                cho_solve_failed = False
            except Exception:
                alpha = np.linalg.solve(K, self.y_train)
                L, low = None, None
                cho_solve_failed = True

            # Cache the exact solve for subsequent predict calls
            self._cached_L = L
            self._cached_low = low
            self._cached_alpha = alpha
            self._cached_params = params.copy()

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
            self.x_train, X_test, predict_order, predict_oti, return_deriv=return_deriv
        )

        # Compute train-test kernel
        phi_train_test = predict_kernel_func(diff_x_test_x_train, length_scales)
        if predict_order > 0:
            if return_deriv:
                phi_exp_train_test = phi_train_test.get_all_derivs(self.n_bases, 2 * predict_order)
            else:
                phi_exp_train_test = phi_train_test.get_all_derivs(self.n_bases, predict_order)
        else:
            phi_exp_train_test = phi_train_test.real
            phi_exp_train_test =  phi_exp_train_test[np.newaxis, :, :]

        K_s = degp_utils.rbf_kernel_predictions(
            phi_train_test, phi_exp_train_test, predict_order, self.n_bases,
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
        diff_x_test_x_test = degp_utils.differences_by_dim_func(
            X_test, X_test, predict_order, predict_oti, return_deriv=return_deriv
        )

        # Compute test-test kernel
        phi_test_test = predict_kernel_func(diff_x_test_x_test, length_scales)
        if predict_order > 0:
            phi_exp_test_test = phi_test_test.get_all_derivs(self.n_bases, 2 * predict_order)
        else:
            phi_exp_test_test = phi_test_test.real
            phi_exp_test_test = phi_exp_test_test[np.newaxis,:,:]

        K_ss = degp_utils.rbf_kernel_predictions(
            phi_test_test, phi_exp_test_test, predict_order, self.n_bases,
            self.flattened_der_indices, self.powers,
            return_deriv=return_deriv,
            index=derivative_locations_test,
            common_derivs=common_derivs,
            calc_cov=True,
            powers_predict=self.powers_predict
        )

        # Compute predictive covariance using sparse U: K^{-1} ≈ U U^T,
        # so K_s^T K^{-1} K_s ≈ (U^T K_s)^T (U^T K_s).
        # U is stored in original (non-permuted) index space as self._cached_U
        # with column order following P; apply the same permutation to K_s rows.
        if cho_solve_failed:
            if hasattr(self, '_cached_U') and self._cached_U is not None:
                P_full = self.mmd_P_full
                U = self._cached_U
                K_s_ord = K_s[P_full, :]  # reorder rows to MMD order
                v = U.T @ K_s_ord    # shape (N, n_test*)
                f_cov = K_ss - v.T @ v
            else:
                # Last resort: rebuild K and invert (only hit if U was never cached)
                phi_train_fb = self.kernel_func(self.differences_by_dim, length_scales)
                if self.n_order > 0:
                    phi_exp_fb = phi_train_fb.get_all_derivs(self.n_bases, 2 * self.n_order)
                else:
                    phi_exp_fb = phi_train_fb.real[np.newaxis, :, :]
                K_fb = degp_utils.rbf_kernel(
                    phi_train_fb, phi_exp_fb, self.n_order, self.n_bases,
                    self.flattened_der_indices, self.powers,
                    index=self.derivative_locations
                )
                K_fb += (10 ** sigma_n) ** 2 * np.eye(K_fb.shape[0])
                K_fb += self.sigma_data ** 2
                f_cov = K_ss - K_s.T @ np.linalg.inv(K_fb) @ K_s
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