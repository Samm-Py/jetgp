# acquisition_functions.py

import numpy as np
from jetgp.full_degp import degp_utils
import jetgp.utils as utils
from line_profiler import profile
from scipy.stats.qmc import Sobol
from scipy.optimize import minimize
def find_next_point_batch(
    gp,
    params,
    dist_params,
    acquisition_func,
    integration_points=None,
    candidate_points=None,
    n_candidate_points: int = 1024,
    n_local_starts: int = 1,
    n_batch_points: int = 10,
    local_opt = True,
    **acq_kwargs
):
    """
    Batch active learning with support for aggregated mode (multi-GP).

    If use_agg_al=False: single GP loop.
    If use_agg_al=True: use gp_list + params_list and acquisition_func must handle them.
    Returns: np.ndarray of selected points, shape (n_selected, d).
    """

    selected_points = []
    n_integration_points = integration_points.shape[0]
    # helper to generate candidates
    def _generate_candidates(dimension, seed_offset=0):
        sampler = Sobol(d=dimension, scramble=True, seed=acq_kwargs.get('seed', 0) + seed_offset)
        coarse_points_unit = sampler.random(n=n_candidate_points)
        return utils.get_inverse(dist_params, coarse_points_unit)

    if type(gp) is not list or type(params) is not list:
        raise ValueError("gp and params must be provided as list")



    for n in range(n_batch_points):
        if candidate_points is None:
            candidate_points = _generate_candidates(gp[0].n_bases,seed_offset=n)
        # Normalize once before the loop
        cand_scores_all = []
        
        if gp[0].normalize:
            sigmas_x = gp[0].sigmas_x
            mus_x = gp[0].mus_x
            
            candidate_points_norm = utils.normalize_x_data_test(candidate_points.copy(), sigmas_x, mus_x)
            if integration_points is not None:
                integration_points_norm = utils.normalize_x_data_test(integration_points.copy(), sigmas_x, mus_x)
                n_integration_points = integration_points.shape[0]
        else:
            candidate_points_norm = candidate_points.copy()
            integration_points_norm = integration_points.copy()
            n_integration_points = None
        
        # Precompute all differences once
        precomputed_diffs = {
            'train_cand': degp_utils.differences_by_dim_func(
                gp[0].x_train, candidate_points_norm, gp[0].n_order, return_deriv=False
            ),
            'cand_cand': degp_utils.differences_by_dim_func(
                candidate_points_norm, candidate_points_norm, gp[0].n_order, return_deriv=False
            ),
            'domain_cand': degp_utils.differences_by_dim_func(
                integration_points_norm, candidate_points_norm, gp[0].n_order, return_deriv=False
            ),
            'train_domain': degp_utils.differences_by_dim_func(
                gp[0].x_train, integration_points_norm, gp[0].n_order, return_deriv=False
            )
        }
        # Loop over time steps - much simpler now!
        for gp_t, params_t in zip(gp, params):
            scores_t = acquisition_func(
                gp_t,
                params_t,
                precomputed_diffs,
                n_integration_points = n_integration_points
            )
            
                    
                
            cand_scores_all.append(scores_t.reshape(-1, 1))
            
            
        cand_scores = np.mean(np.hstack(cand_scores_all), axis=1)
        # --- Pick top starting points from aggregated candidate scores ---
        top_indices = np.argsort(cand_scores.ravel())[-n_local_starts:]
        top_starting_points = candidate_points[top_indices]
        
        if local_opt:
            # --- Get bounds for local optimization ---
            bounds = utils.get_optimization_bounds(dist_params)
            all_unbounded = all(b[0] is None for b in bounds)
            
            # --- Define objective function for local refinement ---
            def objective_function(x: np.ndarray) -> float:
                """
                Objective for local optimization. 
                Note: Can't use precomputed diffs here since x is varying during optimization.
                """
                x_cand = x.reshape(1, -1)
                
                # Normalize the candidate point if needed
                if gp[0].normalize:
                    x_cand_norm = utils.normalize_x_data_test(x_cand.copy(), sigmas_x, mus_x)
                else:
                    x_cand_norm = x_cand.copy()
                
                # Compute differences for this single candidate
                diff_train_cand = degp_utils.differences_by_dim_func(
                    gp[0].x_train, x_cand_norm, gp[0].n_order, return_deriv=False
                )
                diff_cand_cand = degp_utils.differences_by_dim_func(
                    x_cand_norm, x_cand_norm, gp[0].n_order, return_deriv=False
                )
                diff_domain_cand = degp_utils.differences_by_dim_func(
                    integration_points_norm, x_cand_norm, gp[0].n_order, return_deriv=False
                )
                
                # Precomputed diffs for this candidate
                local_precomputed_diffs = {
                    'train_cand': diff_train_cand,
                    'cand_cand': diff_cand_cand,
                    'domain_cand': diff_domain_cand,
                    'train_domain': precomputed_diffs['train_domain']  # Reuse this one
                }
                
                # Compute aggregated score across all GPs
                score_list = [
                    acquisition_func(
                        gp_t,
                        params_t,
                        precomputed_diffs=local_precomputed_diffs,
                        n_integration_points = n_integration_points
                    )
                    for gp_t, params_t in zip(gp, params)
                ]
                return -np.mean([s[0] if isinstance(s, np.ndarray) else s for s in score_list])
            
            # --- Local optimization loop ---
            best_x, best_score = None, -np.inf
            for start_point in top_starting_points:
                if all_unbounded:
                    res = minimize(fun=objective_function, x0=start_point, method="BFGS")
                else:
                    res = minimize(fun=objective_function, x0=start_point, method="L-BFGS-B", bounds=bounds)
                
                if -res.fun > best_score:
                    best_score, best_x = -res.fun, res.x
            
            best_x = best_x.reshape(1, -1)
            selected_points.append(best_x)
        else:
            top_index = np.argmax(cand_scores.ravel())
            best_x = candidate_points[top_index].reshape(1, -1)
            selected_points.append(best_x)
            candidate_points = np.delete(candidate_points, top_index, axis=0)

        # Augment each GP with predicted outputs at best_x
        for t, gp_t in enumerate(gp):
            params_t = params[t]
            y_pred = gp_t.predict(best_x, params_t, calc_cov=False, return_deriv=True)
            y_to_add = utils._format_y_to_add(y_pred)

            x_aug = np.vstack([gp_t.x_train_input, best_x])
            y_aug = []
            for i_y, y_block in enumerate(gp_t.y_train_input):
                if i_y < len(y_to_add):
                    y_aug.append(np.vstack([y_block, y_to_add[i_y]]))
                else:
                    y_aug.append(y_block)

            gp[t] = gp_t.__class__(
                x_aug, y_aug,
                gp_t.n_order,
                n_bases=gp_t.n_bases,
                der_indices=gp_t.der_indices,
                normalize=gp_t.normalize,
                kernel=gp_t.kernel,
                kernel_type=gp_t.kernel_type,
            )

    return np.vstack(selected_points)

# def imse_reduction_verify(gp, params, x_train, y_train_list,der_indices, candidate_points, full_domain, noise_var=None):
#     """
#     Brute-force IMSE verification using temporary DE-GP for each candidate point.
#     Handles flattened y_train_candidates output from gp.predict correctly.

#     Args:
#         gp: Original degp object.
#         params: Kernel hyperparameters.
#         y_train_list: List of arrays (function + derivatives) for the original training points.
#         candidate_points: Candidate points to evaluate (n_candidates, d).
#         full_domain: Domain points for IMSE computation (n_domain, d).
#         noise_var: Optional noise variance for function observations.

#     Returns:
#         imse_brute: Array of IMSE reductions for each candidate.
#     """
#     n_candidates = candidate_points.shape[0]
#     n_domain = full_domain.shape[0]
#     n_derivs = len(y_train_list) - 1  # number of derivatives
#     imse_brute = np.zeros(n_candidates)

#     # 1. Original GP posterior variance at full_domain (function values only)
#     _, sigma2_orig = gp.predict(
#         full_domain, params, calc_cov=True, return_deriv=False)
#     sigma2_orig = np.maximum(sigma2_orig, 0.0)

#     # 2. Predict all function + derivatives over full domain
#     y_train_candidates_flat = gp.predict(
#         full_domain, params, calc_cov=False, return_deriv=True)

#     for i_cand, x_cand in enumerate(candidate_points):
#         # 3. Extract candidate outputs from the flattened array
#         y_cand_list = []
#         for k in range(n_derivs + 1):  # include function
#             start = k * n_domain + i_cand
#             end = start + 1
#             y_cand_list.append(y_train_candidates_flat[start:end])

#         # 4. Construct temporary training outputs (list of arrays)
#         y_train_temp = []
#         for j in range(len(y_train_list)):
#             y_train_temp.append(np.vstack([y_train_list[j], y_cand_list[j]]))

#         # 5. Construct augmented training inputs
#         X_train_temp = np.vstack([x_train.copy(), x_cand])

#         # 6. Initialize temporary GP with augmented training data
#         gp_temp = degp(
#             X_train_temp, y_train_temp, gp.n_order, gp.n_bases,
#             normalize=gp.normalize,
#             kernel=gp.kernel,
#             der_indices=gp.der_indices,
#             kernel_type=gp.kernel_type
#         )

#         # 7. Predict posterior variance at full domain (function values only)
#         _, sigma2_new = gp_temp.predict(
#             full_domain, params, calc_cov=True, return_deriv=False)
#         sigma2_new = np.maximum(sigma2_new, 0.0)

#         # 8. Compute IMSE reduction for this candidate
#         imse_brute[i_cand] = np.mean((sigma2_orig - sigma2_new))
#     return imse_brute

@profile
def mse_reduction(
    gp, params, precomputed_diffs,
    noise_var=None, return_deriv=False, **kwargs
):
    """
    Compute MSE reduction (posterior variance) at candidate points.
    
    Args:
        gp: GP object
        params: Kernel hyperparameters
        x_train: Training inputs
        y_train_list: Training outputs
        candidate_points: Normalized candidate points to evaluate
        precomputed_diffs: Dict with precomputed difference matrices:
            - 'train_cand': differences between training and candidate points
            - 'cand_cand': differences between candidate points
        noise_var: Observation noise variance
        return_deriv: Whether to return derivatives
        normalize: Whether to normalize y data (x data assumed already normalized)
    
    Returns:
        1D array of posterior variances at candidate points
    """

    
    
    length_scales = params[:-1]
    sigma_n = params[-1]
    
    # Training kernel
    diff_train_train = gp.differences_by_dim
    K = degp_utils.rbf_kernel(
        diff_train_train, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=True
    )
    K += (10**sigma_n) ** 2 * np.eye(K.shape[0])
    
    # Use precomputed differences
    diff_train_cand = precomputed_diffs['train_cand']
    diff_cand_cand = precomputed_diffs['cand_cand']
    
    # Compute kernels from differences
    K_s = degp_utils.rbf_kernel(
        diff_train_cand, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=return_deriv
    )
    
    n = K_s.shape[1]
    
    K_ss = degp_utils.rbf_kernel(
        diff_cand_cand, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=return_deriv
    )
    
    # Compute posterior covariance
    f_cov = (
        K_ss - K_s.T @ np.linalg.solve(K, K_s)
        if return_deriv
        else K_ss[:n, :n] - K_s[:, :n].T @ np.linalg.solve(K, K_s[:, :n])
    )
    
    return np.diag(f_cov)


def imse_reduction(
    gp, params, precomputed_diffs,
    noise_var=None, return_deriv=False, **kwargs
):
    """
    Vectorized exact IMSE reduction with precomputed differences.
    
    Args:
        gp: GP object
        params: Kernel hyperparameters
        x_train: Training inputs
        y_train_list: Training outputs
        candidate_points: Normalized candidate points to evaluate
        integration_points: Normalized integration points for IMSE
        precomputed_diffs: Dict with precomputed difference matrices:
            - 'train_domain': differences between training and integration points
            - 'train_cand': differences between training and candidate points
            - 'cand_cand': differences between candidate points
            - 'domain_cand': differences between integration and candidate points
        noise_var: Observation noise variance
        return_deriv: Whether to return derivatives
        normalize: Whether to normalize y data (x data assumed already normalized)
    
    Returns:
        1D array of IMSE reductions for each candidate point
    """

   
    
    
    length_scales = params[:-1]
    sigma_n = params[-1]
    n_integration_points = kwargs['n_integration_points']
    # Training kernel
    diff_train_train = gp.differences_by_dim
    K = degp_utils.rbf_kernel(
        diff_train_train, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=True
    )
    K += (10**sigma_n) ** 2 * np.eye(K.shape[0])
    
    # Use precomputed differences
    diff_train_domain = precomputed_diffs['train_domain']
    diff_train_cand = precomputed_diffs['train_cand']
    diff_cand_cand = precomputed_diffs['cand_cand']
    diff_domain_cand = precomputed_diffs['domain_cand']
    
    # Compute kernels from differences
    K_s = degp_utils.rbf_kernel(
        diff_train_domain, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=return_deriv
    )
    
    K_train_cand = degp_utils.rbf_kernel(
        diff_train_cand, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=return_deriv
    )
    
    K_cand_cand = np.diag(degp_utils.rbf_kernel(
        diff_cand_cand, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=return_deriv
    ))
    

    K_domain_cand = degp_utils.rbf_kernel( diff_domain_cand, length_scales, gp.n_order, gp.n_bases, gp.kernel_func, gp.flattened_der_indicies, gp.powers, return_deriv=return_deriv ) 
    # shape: (n_domain, n_candidates) # Solve K^{-1} @ K_train_cand for all candidates 
    v = np.linalg.solve(K, K_train_cand) # shape: (n_train, n_candidates) # Numerator and denominator of variance reduction 
    numerator = (K_domain_cand[:n_integration_points,:] - K_s.T @ v) ** 2 # shape: (n_domain, n_candidates) 
    denominator = K_cand_cand- np.sum(K_train_cand * v, axis=0) # shape: (n_candidates,) denominator = np.maximum(denominator, 1e-16) 
    variance_reduction = (numerator / denominator) # shape: (n_domain, n_candidates) # IMSE reduction for each candidate = mean over domain points
    imse_reductions = variance_reduction.mean(axis=0) # shape: (n_candidates,)
    return imse_reductions
