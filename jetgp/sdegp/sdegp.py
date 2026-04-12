"""
Sliced Derivative-Enhanced Gaussian Process (SDEGP)
====================================================

Implements Sliced GE-Kriging (SGE-Kriging) from Cheng & Zimmermann (2024).
Training points are partitioned into m slices along the coordinate with the
largest sensitivity index; the likelihood is approximated as a signed sum
of pair-block and single-slice-block log-likelihoods (eq. 20).

The partitioning strategy (currently sensitivity-based coordinate slicing)
is pluggable via the ``partition_indices`` function in ``sliced_partition.py``.
"""

import numpy as np
import jetgp.utils as utils
from jetgp.kernel_funcs.kernel_funcs import KernelFactory, get_oti_module


class sdegp:
    """
    Sliced Derivative-Enhanced Gaussian Process (SDEGP) regression model.

    Implements the sliced GE-Kriging (SGE-Kriging) method from
    Cheng & Zimmermann (2024). Training points are partitioned into *m*
    slices along the coordinate with the largest sensitivity index; the
    likelihood is approximated as a signed sum of pair-block and
    single-slice-block log-likelihoods (eq. 20).

    Parameters
    ----------
    x_train : ndarray of shape (n_samples, n_features)
        Input training points.
    y_train : ndarray of shape (n_samples,) or (n_samples, 1)
        Function values at training points.
    grads : ndarray of shape (n_samples, n_features)
        Gradient observations at all training points.
    n_order : int
        Maximum derivative order (typically 1).
    m : int
        Number of slices (must be >= 3).  Produces 2*m - 3 submodels.
    normalize : bool, default=True
        If True, normalizes the input and output data.
    sigma_data : float or ndarray, optional
        Known observation noise or covariance matrix.
    kernel : str, default='SE'
        Type of kernel to use: 'SE', 'RQ', 'Matern', or 'SineExp'.
    kernel_type : str, default='anisotropic'
        Whether kernel is 'anisotropic' or 'isotropic'.
    smoothness_parameter : float, optional
        Smoothness parameter for Matern kernel.
    """

    def __init__(
        self,
        x_train,
        y_train,
        grads,
        n_order,
        m,
        normalize=True,
        sigma_data=None,
        kernel="SE",
        kernel_type="anisotropic",
        smoothness_parameter=None,
    ):
        from jetgp.sdegp.sliced_partition import build_sliced_submodels

        # Build submodel structures from raw data
        y_vals = np.asarray(y_train).ravel()
        grads = np.asarray(grads)
        (
            submodel_data,
            der_specs_list,
            der_locs_list,
            func_locs_list,
            weights,
            slice_dim,
            slices,
        ) = build_sliced_submodels(x_train, y_vals, grads, m, n_order=n_order)

        # Store slicing metadata
        self.m = m
        self.slices = slices
        self.slice_dim = slice_dim

        # Store basic parameters
        self.x_train = x_train
        self.y_train = submodel_data
        self.n_order = n_order
        self.n_bases = x_train.shape[1]
        self.der_indices = der_specs_list
        self.derivative_locations = der_locs_list
        self.function_locations = [list(fl) for fl in func_locs_list]
        self.submodel_weights = weights
        self.submodel_type = 'degp'
        self.rays = None
        self.rays_list = None
        self.normalize = normalize
        self.kernel = kernel
        self.kernel_type = kernel_type

        self.oti = get_oti_module(self.n_bases, n_order)

        self.num_points = len(x_train)
        self.dim = x_train.shape[1]
        self.num_submodels = len(submodel_data)

        # Store original input for reference
        self.y_train_input = [yt.copy() if hasattr(yt, 'copy') else yt
                              for yt in submodel_data]
        self.x_train_input = x_train.copy()

        # Validate configuration
        self._validate_config()
        
        # Set up derivative_locations defaults
        self._setup_derivative_locations()
        
        # Process derivative indices
        self._setup_derivative_indices()

        # Precompute kernel plans (purely structural — no hyperparameters)
        self._precompute_kernel_plans()

        # Handle sigma_data
        if sigma_data is None:
            sigma_data = np.zeros(self._compute_total_constraints())
        self.sigma_data = np.diag(sigma_data)
        self.sigma_data_sq_diag = np.asarray(sigma_data) ** 2

        # Normalize if requested
        if normalize:
            self._normalize_data()
        else:
            self.y_train_normalized = [utils.reshape_y_train(submodel) for submodel in self.y_train]
            self.x_train_normalized = x_train
            
        # Precompute differences
        self._precompute_differences()
        
        # Set up kernel
        self.kernel_factory = KernelFactory(
            dim=self.dim,
            normalize=self.normalize,
            n_order=self.n_order,
            differences_by_dim=self.differences_by_dim,
            smoothness_parameter=smoothness_parameter,
            oti_module=self.oti,
            sparse_diffs=(self.submodel_type == 'degp')
        )
        self.kernel_func = self.kernel_factory.create_kernel(
            kernel_name=self.kernel,
            kernel_type=self.kernel_type,
        )
        self.bounds = self.kernel_factory.bounds
        
        # Set up optimizer
        from jetgp.sdegp.optimizer import Optimizer
        self.optimizer = Optimizer(self)

        # Lazy-built full prediction model (built on first predict call)
        self._full_predict_model = None

    def _validate_config(self):
        """Validate the configuration parameters."""
        pass  # submodel_type is always 'degp'; partition validated by build_sliced_submodels

    def _setup_derivative_locations(self):
        """Set up derivative_locations with defaults.

        For SDEGP, derivative_locations are always provided by
        build_sliced_submodels, so this is a no-op safety net.
        """
        if self.derivative_locations is None:
            self.derivative_locations = []
            for submodel_idx in range(self.num_submodels):
                n_derivs = self._count_derivatives_for_submodel(submodel_idx)
                self.derivative_locations.append(
                    [list(range(self.num_points))] * n_derivs
                )

    def _count_derivatives_for_submodel(self, submodel_idx):
        """Count the number of derivative types for a submodel."""
        # Flatten der_indices for this submodel
        der_idx = self.der_indices[submodel_idx]
        count = 0
        for group in der_idx:
            count += len(group)
        return count

    def _setup_derivative_indices(self):
        """Process and flatten derivative indices for each submodel."""
        self.flattened_der_indices = []
        self.powers = []
        
        base_der_indices = utils.gen_OTI_indices(self.n_bases, self.n_order)
        
        for submodel_idx, ders in enumerate(self.der_indices):
            self.powers.append(
                utils.build_companion_array(self.n_bases, self.n_order, ders)
            )
            flat_indices = [i for sublist in ders for i in sublist]
            self.flattened_der_indices.append(flat_indices)

    def _compute_total_constraints(self):
        """Compute total number of constraints across all submodels."""
        total = 0
        for submodel_idx in range(self.num_submodels):
            # Function values (per-submodel)
            total += len(self.function_locations[submodel_idx])
            # Derivative values
            for locs in self.derivative_locations[submodel_idx]:
                total += len(locs)
        return total

    def _normalize_data(self):
        """Normalize input and output data.

        Normalization statistics (mu_y, sigma_y, mus_x, sigmas_x) are
        computed from the full x_train once, then applied consistently to
        each submodel's (potentially different) function-value subset.
        """
        # Compute input normalization from full x_train
        self.mus_x = np.mean(self.x_train, axis=0).reshape(1, -1)
        self.sigmas_x = np.std(self.x_train, axis=0).reshape(1, -1)

        # Compute output normalization from full y dataset.
        # Collect all unique function values across submodels.
        all_func_vals = np.concatenate([
            np.asarray(self.y_train[k][0]).flatten()
            for k in range(self.num_submodels)
        ])
        # Deduplicate: take unique values to avoid bias from overlapping slices
        all_func_vals_unique = np.unique(all_func_vals)
        self.mu_y = np.mean(all_func_vals_unique).reshape(-1, 1)
        self.sigma_y = np.std(all_func_vals_unique).reshape(-1, 1)

        self.y_train_normalized = []
        for k, submodel_data in enumerate(self.y_train):
            # Normalize function values with global stats
            y_func_norm = (submodel_data[0] - self.mu_y) / self.sigma_y
            y_norm = y_func_norm
            # Normalize derivatives using chain rule
            for i, der_idx in enumerate(self.flattened_der_indices[k]):
                factor = 1.0 / self.sigma_y
                for j in range(len(der_idx)):
                    factor = factor * self.sigmas_x[0][der_idx[j][0] - 1] ** der_idx[j][1]
                y_norm = np.vstack((y_norm.reshape(-1, 1),
                                   submodel_data[i + 1] * factor[0, 0]))
            self.y_train_normalized.append(y_norm.flatten())

        self.x_train_normalized = utils.normalize_x_data_train(self.x_train)
        
    def _precompute_differences(self):
        """Precompute differences_by_dim for DEGP (coordinate-aligned)."""
        x = self.x_train_normalized if self.normalize else self.x_train
        from jetgp.sdegp import sdegp_utils
        self.differences_by_dim = sdegp_utils.differences_by_dim_func(
            x, x, self.n_order, self.oti
        )

    def _precompute_kernel_plans(self):
        """Precompute structural kernel plans once per model.

        Plans depend only on (n_order, n_bases, derivative layout), not on
        hyperparameters, so they are built here and reused by the optimizer
        across every nll/nll_and_grad call.
        """
        self.kernel_plans = None
        gp_utils = self._get_utils_module()
        if gp_utils is None or not hasattr(gp_utils, 'precompute_kernel_plan'):
            return
        plan_n_bases = self.n_bases
        plans = []
        for i in range(len(self.derivative_locations)):
            func_idx = np.asarray(self.function_locations[i], dtype=np.int64)
            plan = gp_utils.precompute_kernel_plan(
                self.n_order, plan_n_bases,
                self.flattened_der_indices[i],
                self.powers[i],
                self.derivative_locations[i],
                function_locations=func_idx,
            )
            # Pre-bake everything the optimizer's hot path needs so it can
            # call _assemble_kernel_numba directly without any per-call dict
            # lookups on the plan.
            n_func = len(func_idx)
            plan['row_offsets_abs'] = plan['row_offsets'] + n_func
            plan['col_offsets_abs'] = plan['col_offsets'] + n_func
            plan['_numba_args'] = (
                plan['fd_flat_indices'], plan['df_flat_indices'], plan['dd_flat_indices'],
                plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
                plan['signs'], plan['n_deriv_types'],
                plan['row_offsets_abs'], plan['col_offsets_abs'],
                func_idx,
            )
            plans.append(plan)
        self.kernel_plans = plans

    def _get_utils_module(self):
        """Get the appropriate utils module."""
        from jetgp.sdegp import sdegp_utils
        return sdegp_utils

    def _check_full_predict_compat(self):
        """Validate that submodel observations can be merged for prediction.

        Requires identical ``der_indices`` across submodels (so we can union
        observations per component without re-mapping derivative types).

        The full function-value vector is reconstructed by merging all
        submodels' function values at their respective global indices.
        """
        for k in range(1, self.num_submodels):
            if self.der_indices[k] != self.der_indices[0]:
                raise ValueError(
                    "Prediction requires every submodel to share the "
                    "same der_indices."
                )

        # Reconstruct full function-value vector from per-submodel slices
        y_func_full = np.empty((self.num_points, 1))
        y_func_full[:] = np.nan
        for k in range(self.num_submodels):
            func_locs = self.function_locations[k]
            y_k = np.asarray(self.y_train_input[k][0]).ravel()
            for local_i, pt in enumerate(func_locs):
                val = float(y_k[local_i])
                if not np.isnan(y_func_full[pt, 0]) and not np.isclose(y_func_full[pt, 0], val):
                    raise ValueError(
                        f"Inconsistent function value at point {pt} across submodels."
                    )
                y_func_full[pt, 0] = val

        if np.any(np.isnan(y_func_full)):
            missing = np.where(np.isnan(y_func_full.ravel()))[0]
            raise ValueError(
                f"Prediction requires function values at all training "
                f"points; missing at indices {missing.tolist()}"
            )
        return y_func_full

    def _union_degp_derivs(self):
        """Union per-component (point -> value) observations across submodels.

        Returns
        -------
        full_locs : list of list of int
        full_y_derivs : list of ndarray (n_obs, 1)
        """
        n_components = len(self.flattened_der_indices[0])
        full_locs = []
        full_y_derivs = []
        for comp in range(n_components):
            merged = {}
            for k in range(self.num_submodels):
                locs_k = self.derivative_locations[k][comp]
                y_k = np.asarray(self.y_train_input[k][comp + 1]).ravel()
                for local_i, pt in enumerate(locs_k):
                    val = float(y_k[local_i])
                    if pt in merged and not np.isclose(merged[pt], val):
                        raise ValueError(
                            f"Inconsistent derivative value at point {pt}, "
                            f"component {comp} across submodels."
                        )
                    merged[pt] = val
            sorted_pts = sorted(merged.keys())
            full_locs.append(sorted_pts)
            full_y_derivs.append(
                np.array([merged[p] for p in sorted_pts], dtype=float).reshape(-1, 1)
            )
        return full_locs, full_y_derivs

    def _build_full_predict_model(self):
        """Build a full DEGP model over the union of submodel observations.

        Prediction uses a single dense GP with the SDEGP-optimized
        hyperparameters — the sliced decomposition is only for training.
        """
        smoothness_param = getattr(self.kernel_factory, 'alpha', None)
        y_func_ref = self._check_full_predict_compat()
        full_locs, full_y_derivs = self._union_degp_derivs()
        y_train_full = [y_func_ref.reshape(-1, 1)] + full_y_derivs

        from jetgp.full_degp.degp import degp
        return degp(
            self.x_train_input,
            y_train_full,
            n_order=self.n_order,
            n_bases=self.n_bases,
            der_indices=self.der_indices[0],
            derivative_locations=full_locs,
            normalize=self.normalize,
            kernel=self.kernel,
            kernel_type=self.kernel_type,
            smoothness_parameter=smoothness_param,
        )

    def optimize_hyperparameters(self, *args, **kwargs):
        """
        Optimize hyperparameters via the configured optimizer.

        Returns
        -------
        ndarray
            Optimized hyperparameter vector.
        """
        return self.optimizer.optimize_hyperparameters(*args, **kwargs)
    
    def predict(
        self,
        X_test,
        length_scales,
        calc_cov=False,
        return_deriv=False,
        derivs_to_predict=None,
    ):
        """
        Compute posterior predictive mean and (optionally) covariance at test points.

        SDEGP uses a sliced likelihood for fast *training* (hyperparameter
        optimization), but prediction is always exact: a full DEGP model is
        built over the union of all submodel observations and queried with the
        SDEGP-optimized hyperparameters.

        Parameters
        ----------
        X_test : ndarray of shape (n_test, n_features)
            Test input points.
        length_scales : ndarray
            Log-scaled kernel hyperparameters including noise level.
        calc_cov : bool, default=False
            If True, also compute and return predictive covariance.
        return_deriv : bool, default=False
            If True, also predict coordinate-aligned derivatives.
        derivs_to_predict : list, optional
            Specific derivatives to predict. If None, defaults to all
            derivatives within n_bases and n_order.

        Returns
        -------
        y_val : ndarray
            Predicted mean values. Shape depends on return_deriv.
        y_var : ndarray, optional
            Predictive variances (only if calc_cov=True).
        """
        if self._full_predict_model is None:
            self._full_predict_model = self._build_full_predict_model()
        return self._full_predict_model.predict(
            X_test, length_scales,
            calc_cov=calc_cov,
            return_deriv=return_deriv,
            derivs_to_predict=derivs_to_predict,
        )

