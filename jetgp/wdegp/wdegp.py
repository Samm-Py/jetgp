"""
Unified Weighted Derivative-Enhanced Gaussian Process (WDEGP)
==============================================================

Supports DEGP, DDEGP, or GDDEGP mode for all submodels.

Submodel Types:
- 'degp': Coordinate-aligned derivatives (standard DEGP)
- 'ddegp': Global directional derivatives (same rays at all points)
- 'gddegp': Point-wise directional derivatives (unique rays per point)
"""

import numpy as np
from numpy.linalg import cholesky
from scipy.linalg import cho_solve, cho_factor, solve_triangular
import jetgp.utils as utils
from jetgp.kernel_funcs.kernel_funcs import KernelFactory, get_oti_module


class wdegp:
    """
    Unified Weighted Derivative-Enhanced Gaussian Process (WDEGP) regression model.

    Supports multiple submodels with DEGP, DDEGP, or GDDEGP derivative structure.
    All submodels use the same derivative type.

    Parameters
    ----------
    x_train : ndarray of shape (n_samples, n_features)
        Input training points.
    y_train : list of lists of arrays
        Each element is a submodel's data: [y_func, y_der1, y_der2, ...]
    n_order : int
        Maximum derivative order to be supported.
    n_bases : int
        Number of OTI basis terms used.
    der_indices : list of lists
        Multi-indices of derivatives for each submodel.
    derivative_locations : list of lists of lists, optional
        For each submodel, which points have which derivatives.
        derivative_locations[submodel][deriv_type] = [point_indices]
        If None, all points have all derivatives.
    submodel_type : str, default='degp'
        Type of derivative structure: 'degp', 'ddegp', or 'gddegp'.
    rays : ndarray, optional
        For 'ddegp' mode: global ray directions, shape (d, n_directions).
        All submodels share these rays.
    rays_list : list of list of ndarray, optional
        For 'gddegp' mode: point-wise rays organized by submodel.
        rays_list[submodel_idx][dir_idx] has shape (d, n_points_with_dir).
        Example: rays_list[0] = [rays_dir1_sm1, rays_dir2_sm1] for submodel 1.
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
        n_order, 
        n_bases, 
        der_indices,
        derivative_locations=None,
        submodel_type='degp',
        rays=None,
        rays_list=None,
        normalize=True, 
        sigma_data=None, 
        kernel="SE", 
        kernel_type="anisotropic",
        smoothness_parameter=None
    ):
        # Store basic parameters
        self.x_train = x_train
        self.y_train = y_train
        self.n_order = n_order
        self.n_bases = n_bases
        self.der_indices = der_indices
        self.derivative_locations = derivative_locations
        self.submodel_type = submodel_type
        self.rays = rays
        self.rays_list = rays_list
        self.normalize = normalize
        self.kernel = kernel
        self.kernel_type = kernel_type
        
        if submodel_type == 'degp' or submodel_type == 'ddegp':
            self.oti = get_oti_module(self.n_bases, n_order)
        elif submodel_type == 'gddegp':
            self.oti = get_oti_module(2*self.n_bases, n_order)
            
        self.num_points = len(x_train)
        self.dim = x_train.shape[1]
        self.num_submodels = len(y_train)
        
        # Store original input for reference
        self.y_train_input = [yt.copy() if hasattr(yt, 'copy') else yt for yt in y_train]
        self.x_train_input = x_train.copy()

        # Validate configuration
        self._validate_config()
        
        # Set up derivative_locations defaults
        self._setup_derivative_locations()
        
        # Process derivative indices
        self._setup_derivative_indices()
        
        # Handle sigma_data
        if sigma_data is None:
            sigma_data = np.zeros(self._compute_total_constraints())
        self.sigma_data = np.diag(sigma_data)
        self.sigma_data_sq_diag = np.asarray(sigma_data) ** 2

        # Normalize if requested
        if normalize:
            self._normalize_data()
        else:
            self.y_train_normalized = [utils.reshape_y_train(submodel) for submodel in y_train]
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
        from jetgp.wdegp.optimizer import Optimizer
        self.optimizer = Optimizer(self)

    def _validate_config(self):
        """Validate the configuration parameters."""
        valid_types = ['degp', 'ddegp', 'gddegp']
        if self.submodel_type not in valid_types:
            raise ValueError(f"submodel_type must be one of {valid_types}, got '{self.submodel_type}'")
        
        if self.submodel_type == 'ddegp':
            if self.rays is None:
                raise ValueError("rays parameter is required for submodel_type='ddegp'")
            if self.rays.shape[0] != self.dim:
                raise ValueError(f"rays must have shape (d, n_directions), got {self.rays.shape}")
                
        if self.submodel_type == 'gddegp':
            if self.rays_list is None:
                raise ValueError("rays_list parameter is required for submodel_type='gddegp'")
            if len(self.rays_list) != self.num_submodels:
                raise ValueError(
                    f"rays_list must have {self.num_submodels} entries (one per submodel), "
                    f"got {len(self.rays_list)}"
                )
            for sm_idx, sm_rays in enumerate(self.rays_list):
                for dir_idx, r in enumerate(sm_rays):
                    if r.shape[0] != self.dim:
                        raise ValueError(
                            f"rays_list[{sm_idx}][{dir_idx}] must have shape (d, n_points), "
                            f"got {r.shape}"
                        )

    def _setup_derivative_locations(self):
        """Set up derivative_locations with defaults."""
        if self.derivative_locations is None:
            # Default: all points have all derivatives for all submodels
            self.derivative_locations = []
            for submodel_idx in range(self.num_submodels):
                n_derivs = self._count_derivatives_for_submodel(submodel_idx)
                self.derivative_locations.append(
                    [list(range(self.num_points))] * n_derivs
                )
        
        # For GDDEGP, validate rays_list matches derivative_locations per submodel
        if self.submodel_type == 'gddegp':
            for sm_idx, sm_rays in enumerate(self.rays_list):
                submodel_locs = self.derivative_locations[sm_idx]
                for dir_idx, r in enumerate(sm_rays):
                    if dir_idx < len(submodel_locs):
                        expected_size = len(submodel_locs[dir_idx])
                        if r.shape[1] != expected_size:
                            raise ValueError(
                                f"rays_list[{sm_idx}][{dir_idx}] has {r.shape[1]} columns but "
                                f"derivative_locations[{sm_idx}][{dir_idx}] expects {expected_size} points"
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
            # Function values
            total += self.num_points
            # Derivative values
            for locs in self.derivative_locations[submodel_idx]:
                total += len(locs)
        return total

    def _normalize_data(self):
        """Normalize input and output data."""
        self.y_train_normalized = []
        
        if self.submodel_type == 'degp':
            # Standard DEGP normalization
            for k, submodel_data in enumerate(self.y_train):
                y_norm, self.mu_y, self.sigma_y, self.sigmas_x, self.mus_x, _ = \
                    utils.normalize_y_data(
                        self.x_train, 
                        submodel_data,
                        np.zeros(len(submodel_data)),  # placeholder
                        self.flattened_der_indices[k]
                    )
                self.y_train_normalized.append(y_norm)
        
        elif self.submodel_type == 'ddegp':
            # DDEGP uses directional normalization
            for k, submodel_data in enumerate(self.y_train):
                y_norm, self.mu_y, self.sigma_y, self.sigmas_x, self.mus_x, _ = \
                    utils.normalize_y_data_directional(
                        self.x_train, 
                        submodel_data,
                        np.zeros(len(submodel_data)),  # placeholder
                        self.flattened_der_indices[k]
                    )
                
                self.y_train_normalized.append(y_norm)
            self.rays =  self.rays / self.sigmas_x.flatten()[:, None]
        elif self.submodel_type == 'gddegp':
            # GDDEGP uses directional normalization
            for k, submodel_data in enumerate(self.y_train):
                y_norm, self.mu_y, self.sigma_y, self.sigmas_x, self.mus_x, _ = \
                    utils.normalize_y_data_directional(
                        self.x_train, 
                        submodel_data,
                        np.zeros(len(submodel_data)),  # placeholder
                        self.flattened_der_indices[k]
                    )
                self.y_train_normalized.append(y_norm)
            # Normalize rays_list using normalize_directions_2
            self.rays_list = [
                utils.normalize_directions_2(self.sigmas_x, sm_rays)
                for sm_rays in self.rays_list
            ]
        
        self.x_train_normalized = utils.normalize_x_data_train(self.x_train)
        
    def _precompute_differences(self):
        """Precompute differences_by_dim based on submodel_type."""
        x = self.x_train_normalized if self.normalize else self.x_train
        
        if self.submodel_type == 'degp':
            from jetgp.wdegp import wdegp_utils
            self.differences_by_dim = wdegp_utils.differences_by_dim_func(
                x, x, self.n_order, self.oti
            )
        elif self.submodel_type == 'ddegp':
            from jetgp.full_ddegp import wddegp_utils
            # Use first submodel's derivative_locations as reference
            # (all submodels share same rays structure)
            self.differences_by_dim = wddegp_utils.differences_by_dim_func(
                x, x,
                self.rays,
                self.n_order,
                self.oti,
                return_deriv=True
            )
        elif self.submodel_type == 'gddegp':
            from jetgp.full_gddegp import wgddegp_utils
            
            # For GDDEGP, combine rays from all submodels into global structure
            # Each submodel may have a different number of direction types
            max_n_dirs = max(len(sm_rays) for sm_rays in self.rays_list)
            
            global_rays = []
            global_derivative_locations = []
            
            for dir_idx in range(max_n_dirs):
                # Concatenate rays and locations for this direction from all submodels
                rays_for_dir = []
                locs_for_dir = []
                for sm_idx in range(self.num_submodels):
                    sm_rays = self.rays_list[sm_idx]
                    sm_locs = self.derivative_locations[sm_idx]
                    # Only include if this submodel has this direction type
                    if dir_idx < len(sm_rays):
                        rays_for_dir.append(sm_rays[dir_idx])
                        locs_for_dir.extend(sm_locs[dir_idx])
                
                if rays_for_dir:  # Only add if at least one submodel has this direction
                    global_rays.append(np.hstack(rays_for_dir))
                    global_derivative_locations.append(locs_for_dir)
            
            # Store global structures for use in kernel computations
            self.global_rays = global_rays
            self.global_derivative_locations = global_derivative_locations
            
            self.differences_by_dim = wgddegp_utils.differences_by_dim_func(
                x, x,
                global_rays, global_rays,
                global_derivative_locations, global_derivative_locations,
                self.n_order,
                self.oti,
                return_deriv=True
            )

    def _get_utils_module(self):
        """Get the appropriate utils module based on submodel_type."""
        if self.submodel_type == 'degp':
            from jetgp.wdegp import wdegp_utils
            return wdegp_utils
        elif self.submodel_type == 'ddegp':
            from jetgp.full_ddegp import wddegp_utils
            return wddegp_utils
        elif self.submodel_type == 'gddegp':
            from jetgp.full_gddegp import wgddegp_utils
            return wgddegp_utils

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
        return_submodels=False,
        rays_predict=None,
        derivs_to_predict=None
    ):
        """
        Compute posterior predictive mean and (optionally) covariance at test points.
    
        Parameters
        ----------
        X_test : ndarray of shape (n_test, n_features)
            Test input points.
        length_scales : ndarray
            Log-scaled kernel hyperparameters including noise level.
        calc_cov : bool, default=False
            If True, also compute and return predictive covariance.
        return_deriv : bool, default=False
            If True, also predict derivatives (requires rays_predict for GDDEGP).
        return_submodels : bool, default=False
            If True, return submodel-specific contributions.
        rays_predict : list of ndarray, optional
            For 'gddegp' mode with return_deriv=True: rays at test points.
            rays_predict[dir_idx] has shape (d, n_test).
        derivs_to_predict : list, optional
            Specific derivatives to predict. Can include derivatives not present in the
            training set of any submodel — each submodel constructs K_* from kernel
            derivatives directly. If None, defaults to all derivatives common to all
            submodels.
    
        Returns
        -------
        y_val : ndarray
            Predicted mean values. Shape depends on return_deriv.
        y_var : ndarray, optional
            Predictive variances (only if calc_cov=True).
        submodel_vals : list of ndarrays, optional
            Submodel predictions (only if return_submodels=True).
        submodel_cov : list of ndarrays, optional
            Submodel variances (only if calc_cov and return_submodels are True).
        """
        import warnings
        
        gp_utils = self._get_utils_module()
        
        ell = length_scales[:-1]
        sigma_n = length_scales[-1]
        n_test = X_test.shape[0]
        n_train = self.x_train.shape[0]
        
        # =========================================================================
        # Validation checks for rays_predict parameter
        # =========================================================================
        
        # Check 1: Warning when rays_predict provided but return_deriv=False
        if not return_deriv and rays_predict is not None:
            warnings.warn(
                "rays_predict provided but return_deriv=False. "
                "The rays will be ignored.",
                UserWarning
            )
        
        # Check 2: Handle missing rays_predict for GDDEGP/DDEGP with return_deriv=True
        if return_deriv and rays_predict is None:
            if self.submodel_type == 'gddegp':
                # Determine number of ray directions from training
                n_rays = len(self.global_rays)
                warnings.warn(
                    f"No rays_predict provided for GDDEGP with return_deriv=True. "
                    f"Using coordinate axes as default prediction rays ({n_rays} directions).",
                    UserWarning
                )
                # Build default coordinate axis rays
                rays_predict = []
                for i in range(n_rays):
                    axis_idx = i % self.n_bases
                    ray_array = np.zeros((self.n_bases, n_test))
                    ray_array[axis_idx, :] = 1.0
                    rays_predict.append(ray_array)
                    
            elif self.submodel_type == 'ddegp':
                # DDEGP uses global rays, no rays_predict needed
                pass
        
        # Check 3: Validate rays_predict structure when provided and return_deriv=True
        if return_deriv and rays_predict is not None:
            if self.submodel_type == 'gddegp':
                # Validate number of ray directions
                n_expected_rays = len(self.global_rays)
                if len(rays_predict) != n_expected_rays:
                    raise ValueError(
                        f"Number of prediction ray directions ({len(rays_predict)}) "
                        f"does not match training ray directions ({n_expected_rays})"
                    )
                
                # Validate each ray array
                for i, ray_array in enumerate(rays_predict):
                    if not isinstance(ray_array, np.ndarray):
                        raise TypeError(
                            f"rays_predict[{i}] must be a numpy ndarray, "
                            f"got {type(ray_array).__name__}"
                        )
                    if ray_array.ndim != 2:
                        raise ValueError(
                            f"rays_predict[{i}] must be 2-dimensional, "
                            f"got {ray_array.ndim} dimensions"
                        )
                    if ray_array.shape[0] != self.dim:
                        raise ValueError(
                            f"rays_predict[{i}] has {ray_array.shape[0]} rows, "
                            f"expected {self.dim} (number of input dimensions)"
                        )
                    if ray_array.shape[1] != n_test:
                        raise ValueError(
                            f"rays_predict[{i}] has {ray_array.shape[1]} columns, "
                            f"expected {n_test} (number of test points)"
                        )
                        
            elif self.submodel_type == 'ddegp':
                # For DDEGP, rays_predict should match global rays structure
                if not isinstance(rays_predict, np.ndarray):
                    raise TypeError(
                        f"rays_predict for DDEGP must be a numpy ndarray, "
                        f"got {type(rays_predict).__name__}"
                    )
                if rays_predict.shape[0] != self.dim:
                    raise ValueError(
                        f"rays_predict has {rays_predict.shape[0]} rows, "
                        f"expected {self.dim} (number of input dimensions)"
                    )
                    
            elif self.submodel_type == 'degp':
                # DEGP doesn't use rays, warn if provided
                warnings.warn(
                    "rays_predict provided but submodel_type='degp' does not use rays. "
                    "The rays will be ignored.",
                    UserWarning
                )
        
        # =========================================================================
        # End of validation checks
        # =========================================================================
        
        # Normalize test inputs
        if self.normalize:
            X_test_norm = utils.normalize_x_data_test(X_test, self.sigmas_x, self.mus_x)
            if rays_predict is not None and self.submodel_type == 'gddegp':
                rays_predict = utils.normalize_directions_2(
                    self.sigmas_x, rays_predict)
        else:
            X_test_norm = X_test
        
        x_train = self.x_train_normalized if self.normalize else self.x_train
    
        # Determine which derivatives to predict across submodels
        if return_deriv:
            if derivs_to_predict is not None:
                common_derivs = derivs_to_predict
            else:
                if self.submodel_type in ('gddegp', 'wgddegp'):
                    raise ValueError(
                        "derivs_to_predict and rays_predict must be provided explicitly "
                        "for WDEGP with GDDEGP submodels when return_deriv=True, "
                        "because the length of rays_predict must match derivs_to_predict."
                    )
                # Default: predict all derivatives within n_bases and n_order,
                # deterministically sorted by total derivative order
                common_derivs = utils.flatten_der_indices(
                    utils.gen_OTI_indices(self.n_bases, self.n_order)
                )

            # Determine prediction order from requested derivatives
            required_order = max(
                sum(pair[1] for pair in deriv_spec)
                for deriv_spec in common_derivs
            )
            predict_order = max(required_order, self.n_order)

            if predict_order > self.n_order:
                if self.submodel_type == 'degp' or self.submodel_type == 'ddegp':
                    predict_oti = get_oti_module(self.n_bases, predict_order)
                elif self.submodel_type == 'gddegp':
                    predict_oti = get_oti_module(2 * self.n_bases, predict_order)
                smoothness_param = getattr(self.kernel_factory, 'alpha', None)
                predict_kernel_factory = KernelFactory(
                    dim=self.dim,
                    normalize=self.normalize,
                    differences_by_dim=self.differences_by_dim,
                    n_order=predict_order,
                    smoothness_parameter=smoothness_param,
                    oti_module=predict_oti,
                    sparse_diffs=(self.submodel_type == 'degp')
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
    
        y_val = 0
        y_var = 0
        submodel_vals = []
        submodel_cov = []
    
        # For multiple submodels, compute weights (using function-only differences)
        if self.num_submodels > 1:
            if self.submodel_type == 'degp':
                from jetgp.wdegp import wdegp_utils
                diffs_for_weights = wdegp_utils.differences_by_dim_func(
                    X_test_norm, x_train, 0,self.oti, return_deriv=False
                )
            else:
                diffs_for_weights = self._compute_weight_differences(X_test_norm, x_train)
            
            # Use train-train differences for weight computation
            diffs_train_for_weights = self.differences_by_dim
            
            weights_matrix = gp_utils.determine_weights(
                diffs_train_for_weights, diffs_for_weights, ell, self.kernel_func, sigma_n
            )
    
        # Hoist shared computations out of the submodel loop.
        # OTI kernel_func reuses internal buffers, so a second call corrupts
        # the imaginary parts of previously returned OTI objects.  We use
        # .copy() to snapshot each phi into independent memory before the
        # next kernel_func call overwrites the shared buffer.
        diffs_train_train = self.differences_by_dim
        diffs_train_test = self._compute_train_test_differences(
            x_train, X_test_norm, return_deriv, rays_predict,
            predict_order=predict_order, predict_oti=predict_oti
        )

        phi_train_train = self.kernel_func(diffs_train_train, ell).copy()
        if self.n_order == 0:
            self.n_bases_rays = 0
            phi_exp_train_train = phi_train_train.real
            phi_exp_train_train = phi_exp_train_train[np.newaxis, :, :]
        else:
            self.n_bases_rays = phi_train_train.get_active_bases()[-1]
            phi_exp_train_train = phi_train_train.get_all_derivs(self.n_bases_rays, 2 * self.n_order)

        phi_train_test = predict_kernel_func(diffs_train_test, ell).copy()
        if predict_order > 0:
            if return_deriv:
                phi_exp_train_test = phi_train_test.get_all_derivs(self.n_bases_rays, 2 * predict_order)
            else:
                phi_exp_train_test = phi_train_test.get_all_derivs(self.n_bases_rays, predict_order)
        else:
            phi_exp_train_test = phi_train_test.real
            phi_exp_train_test = phi_exp_train_test[np.newaxis, :, :]

        # Pre-compute test-test kernel if covariance is requested
        phi_test_test = None
        phi_exp_test_test = None
        if calc_cov:
            diffs_test_test = self._compute_test_test_differences(
                X_test_norm, return_deriv, rays_predict,
                predict_order=predict_order, predict_oti=predict_oti
            )
            phi_test_test = predict_kernel_func(diffs_test_test, ell).copy()
            if predict_order == 0:
                phi_exp_test_test = phi_test_test.real[np.newaxis, :, :]
            else:
                phi_exp_test_test = phi_test_test.get_all_derivs(self.n_bases_rays, 2 * predict_order)

        # Loop over submodels
        for i in range(self.num_submodels):
            deriv_locs_i = self.derivative_locations[i]

            # Build training kernel matrix (per-submodel: different indices/powers)
            K = gp_utils.rbf_kernel(
                phi_train_train, phi_exp_train_train, self.n_order, self.n_bases,
                self.flattened_der_indices[i], self.powers[i],
                index=deriv_locs_i
            )
            K.flat[::K.shape[0] + 1] += (10 ** sigma_n) ** 2

            # Solve linear system
            try:
                cho_solve_failed = False
                L, low = cho_factor(K, lower=True)
                alpha = cho_solve((L, low), self.y_train_normalized[i])
            except Exception as e:
                cho_solve_failed = True
                alpha = np.linalg.solve(K, self.y_train_normalized[i])
                print('Warning: Cholesky decomposition failed, using standard solve.')

            K_s = gp_utils.rbf_kernel_predictions(
                phi_train_test, phi_exp_train_test, predict_order, self.n_bases_rays,
                self.flattened_der_indices[i], self.powers[i],
                return_deriv=return_deriv,
                index=deriv_locs_i,
                common_derivs=common_derivs,
                powers_predict=self.powers_predict
            )
            
            # Compute predictive mean
            if self.submodel_type == 'gddegp':
                K_s = K_s.T
                f_mean = K_s.T @ alpha
            else:
                f_mean = K_s.T @ alpha
            f_mean = f_mean.reshape(-1, 1)
            
            # Denormalize predictions
            if self.normalize:
                if return_deriv:
                    if self.submodel_type == 'gddegp' or self.submodel_type == 'ddegp':
                        f_mean = utils.transform_predictions_directional(
                           f_mean, self.mu_y, self.sigma_y, self.sigmas_x,
                           common_derivs, X_test)
                    else:
                        f_mean = utils.transform_predictions(
                            f_mean, self.mu_y, self.sigma_y, self.sigmas_x,
                            common_derivs, X_test
                        )
                else:
                    f_mean = self.mu_y + f_mean * self.sigma_y
    
            # Reshape predictions
            n = X_test.shape[0]
            m = f_mean.shape[0]
            num_derivs = m // n
            reshaped = f_mean.reshape(num_derivs, n)
            
            if self.n_order == 0 and not calc_cov:
                return reshaped
                
            # Apply weights for multiple submodels
            if self.num_submodels > 1:
                # Compute weight for this submodel
                unique_indices = set()
                for subindex in deriv_locs_i:
                    unique_indices.update(subindex)
                unique_indices = sorted(unique_indices)
                
                weight = np.zeros(weights_matrix.shape[0])
                for idx in unique_indices:
                    weight += weights_matrix[:, idx]
                
                if return_submodels:
                    submodel_vals.append(reshaped.copy())
                
                reshaped_weighted = reshaped * weight
            else:
                reshaped_weighted = reshaped
                if return_submodels:
                    raise ValueError('Cannot return submodels for a single model')
            
            y_val += reshaped_weighted
    
            # Compute covariance if requested
            if calc_cov:
                f_var = self._compute_predictive_variance(
                    X_test_norm, deriv_locs_i, common_derivs, self.powers_predict,
                    ell, i, K, K_s, L if not cho_solve_failed else None,
                    low if not cho_solve_failed else None, cho_solve_failed,
                    return_deriv, rays_predict,
                    predict_order=predict_order, predict_oti=predict_oti,
                    predict_kernel_func=predict_kernel_func,
                    phi_test_test_cached=phi_test_test,
                    phi_exp_test_test_cached=phi_exp_test_test
                )
                
                if self.num_submodels > 1:
                    f_var_reshaped = f_var.reshape(num_derivs, n)
                    if self.n_order == 0:
                        return reshaped, f_var_reshaped
                    if return_submodels:
                        submodel_cov.append(f_var_reshaped.copy())
                    f_var_reshaped = f_var_reshaped * weight
                    y_var += f_var_reshaped
                else:
                    y_var = f_var.reshape(num_derivs, n)
    
        # Return results
        if self.num_submodels == 1:
            if calc_cov:
                return (y_val, y_var ** 2)
            return y_val
        else:
            if return_submodels:
                if calc_cov:
                    return (y_val, y_var ** 2, submodel_vals, submodel_cov)
                return (y_val, submodel_vals)
            else:
                if calc_cov:
                    return (y_val, y_var ** 2)
                return y_val

    def _compute_train_test_differences(self, x_train, X_test, return_deriv, rays_predict,
                                         submodel_idx=0, predict_order=None, predict_oti=None):
        """Compute train-test differences based on submodel_type."""
        p_order = predict_order if predict_order is not None else self.n_order
        p_oti = predict_oti if predict_oti is not None else self.oti
        if self.submodel_type == 'degp':
            from jetgp.wdegp import wdegp_utils
            return wdegp_utils.differences_by_dim_func(
                x_train, X_test, p_order, p_oti, return_deriv=return_deriv
            )
        elif self.submodel_type == 'ddegp':
            from jetgp.full_ddegp import wddegp_utils
            deriv_locs_test = [list(range(len(X_test)))] * self.rays.shape[1] if return_deriv else None
            return wddegp_utils.differences_by_dim_func(
                x_train, X_test,
                self.rays,
                p_order,
                p_oti,
                return_deriv=return_deriv
            )
        elif self.submodel_type == 'gddegp':
            from jetgp.full_gddegp import gddegp_utils
            n_dirs = len(self.global_rays)
            rays_test = rays_predict if return_deriv and rays_predict is not None else None
            deriv_locs_test = [list(range(len(X_test)))] * n_dirs if return_deriv else None
            return gddegp_utils.differences_by_dim_func(
                x_train, X_test,
                self.global_rays, rays_test,
                self.global_derivative_locations, deriv_locs_test,
                p_order,
                p_oti,
                return_deriv=return_deriv
            )

    def _compute_weight_differences(self, X_test, x_train):
        """Compute differences for weight calculation (always without derivatives)."""
        if self.submodel_type == 'degp':
            from jetgp.wdegp import wdegp_utils
            return wdegp_utils.differences_by_dim_func(X_test, x_train, 0,self.oti, return_deriv=False)
        elif self.submodel_type == 'ddegp':
            from jetgp.full_ddegp import wddegp_utils
            return wddegp_utils.differences_by_dim_func(
                X_test, x_train, self.rays, 0,self.oti, return_deriv=False
            )
        elif self.submodel_type == 'gddegp':
            from jetgp.full_gddegp import wgddegp_utils
            return wgddegp_utils.differences_by_dim_func(
                X_test, x_train, None, None, None, None, 0,self.oti, return_deriv=False
            )

    def _compute_predictive_variance(
        self, X_test, deriv_locs_i, common_derivs, powers_predict, ell, submodel_idx, K, K_s, L, low, cho_solve_failed,
        return_deriv, rays_predict, predict_order=None, predict_oti=None, predict_kernel_func=None,
        phi_test_test_cached=None, phi_exp_test_test_cached=None
    ):
        """Compute predictive variance for a submodel."""
        gp_utils = self._get_utils_module()
        p_order = predict_order if predict_order is not None else self.n_order

        # Use pre-computed test-test kernel if available, otherwise compute
        if phi_test_test_cached is not None:
            phi_test_test = phi_test_test_cached
            phi_exp_test_test = phi_exp_test_test_cached
        else:
            p_kernel_func = predict_kernel_func if predict_kernel_func is not None else self.kernel_func
            diffs_test_test = self._compute_test_test_differences(
                X_test, return_deriv, rays_predict, submodel_idx=submodel_idx,
                predict_order=p_order, predict_oti=predict_oti
            )
            phi_test_test = p_kernel_func(diffs_test_test, ell).copy()
            if p_order == 0:
                phi_exp_test_test = phi_test_test.real[np.newaxis, :, :]
            else:
                bases = phi_test_test.get_active_bases()
                n_bases = bases[-1]
                phi_exp_test_test = phi_test_test.get_all_derivs(n_bases, 2 * p_order)

        deriv_locs_test = [list(range(len(X_test)))] * len(self.derivative_locations[submodel_idx]) if return_deriv else None

        K_ss = gp_utils.rbf_kernel_predictions(
            phi_test_test, phi_exp_test_test, p_order, self.n_bases_rays,
            self.flattened_der_indices[submodel_idx], self.powers[submodel_idx],
            return_deriv=return_deriv,
            index=deriv_locs_test,
            common_derivs=common_derivs,
            calc_cov=True,
            powers_predict=powers_predict
        )
        
        n_test = len(X_test)
        
        if cho_solve_failed:
                f_cov = K_ss - K_s.T @ np.linalg.solve(K, K_s)

        else:
            v = solve_triangular(L, K_s, lower=low)
            f_cov = K_ss - v.T @ v

        if self.normalize:
            if self.submodel_type == 'gddegp' or 'ddegp':
                f_var = utils.transform_cov_directional(
                    f_cov, self.sigma_y, self.sigmas_x,
                    common_derivs, X_test)
            else:
                f_var = utils.transform_cov(
                    f_cov, self.sigma_y, self.sigmas_x,
                    common_derivs, X_test
                )
        else:
            f_var = np.diag(np.abs(f_cov))
        
        return np.sqrt(f_var)

    def _compute_test_test_differences(self, X_test, return_deriv, rays_predict,
                                        submodel_idx=0, predict_order=None, predict_oti=None):
        """Compute test-test differences for covariance."""
        p_order = predict_order if predict_order is not None else self.n_order
        p_oti = predict_oti if predict_oti is not None else self.oti
        if self.submodel_type == 'degp':
            from jetgp.wdegp import wdegp_utils
            return wdegp_utils.differences_by_dim_func(
                X_test, X_test, p_order, p_oti, return_deriv=return_deriv
            )
        elif self.submodel_type == 'ddegp':
            from jetgp.full_ddegp import wddegp_utils
            deriv_locs = [list(range(len(X_test)))] * self.rays.shape[1] if return_deriv else None
            return wddegp_utils.differences_by_dim_func(
                X_test, X_test,
                self.rays,
                p_order,
                p_oti,
                return_deriv=return_deriv
            )
        elif self.submodel_type == 'gddegp':
            from jetgp.full_gddegp import wgddegp_utils
            rays_test = rays_predict if return_deriv else None
            n_dirs = len(self.global_rays) if rays_test is None else len(rays_test)
            deriv_locs = [list(range(len(X_test)))] * n_dirs if return_deriv else None
            return wgddegp_utils.differences_by_dim_func(
                X_test, X_test,
                rays_test, rays_test,
                deriv_locs, deriv_locs,
                p_order,
                p_oti,
                return_deriv=return_deriv
            )