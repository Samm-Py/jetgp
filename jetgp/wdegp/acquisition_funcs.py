# acquisition_functions.py

import numpy as np
from jetgp.wdegp import wdegp_utils
import jetgp.utils as utils
from line_profiler import profile
from scipy.stats.qmc import Sobol
from scipy.optimize import minimize
import copy
from scipy.stats import norm
def finite_difference_gradient(gp, x, params, h=1e-5):
    """
    Compute central finite difference approximation of GP mean gradient at x.

    Parameters
    ----------
    gp : object
        Trained GP model instance
    x : ndarray, shape (1, d)
        Point at which to compute finite difference gradient
    params : array-like
        GP hyperparameters
    h : float
        Step size

    Returns
    -------
    grad_fd : ndarray, shape (d, 1)
        Central finite difference gradient estimate
    """
    x = np.atleast_2d(x)
    d = x.shape[1]
    grad_fd = np.zeros((d, 1))

    # Evaluate GP mean at central point and shifted points
    for i in range(d):
        dx = np.zeros_like(x)
        dx[0, i] = h

        f_plus = gp.predict(x + dx, params,
                            calc_cov=False, return_submodels=False)
        f_minus = gp.predict(x - dx, params,
                             calc_cov=False, return_submodels = False)
        grad_fd[i, 0] = (f_plus[0] - f_minus[0]) / (2 * h)
    return grad_fd
def find_next_point_batch(
    gp,
    params,
    submodel_indices,
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
            'train_cand': wdegp_utils.differences_by_dim_func(
                gp[0].x_train,candidate_points_norm, gp[0].n_order
            ),
            'cand_train': wdegp_utils.differences_by_dim_func(
                candidate_points_norm,gp[0].x_train, gp[0].n_order
            ),
            'cand_cand': wdegp_utils.differences_by_dim_func(
                candidate_points_norm, candidate_points_norm, gp[0].n_order
            ),
            'domain_cand': wdegp_utils.differences_by_dim_func(
                integration_points_norm, candidate_points_norm, gp[0].n_order
            ),
            'train_domain': wdegp_utils.differences_by_dim_func(
                gp[0].x_train, integration_points_norm, gp[0].n_order
            )
        }
        # Loop over time steps - much simpler now!
        for gp_t, params_t in zip(gp, params):
            scores_t = acquisition_func(
                gp_t,
                params_t,
                candidate_points,
                precomputed_diffs = precomputed_diffs,
                n_integration_points = n_integration_points
            )
            
                    
                
            cand_scores_all.append(scores_t.reshape(-1, 1))
            
            
        cand_scores = np.mean(np.hstack(cand_scores_all), axis=1)
        # --- Pick top starting points from aggregated candidate scores ---
        top_indices = np.argsort(cand_scores.ravel())[::-1][:n_local_starts]
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
                diff_train_cand = wdegp_utils.differences_by_dim_func(
                     gp[0].x_train,x_cand_norm,  gp[0].n_order
                )
                diff_cand_train = wdegp_utils.differences_by_dim_func(
                     x_cand_norm,gp[0].x_train,  gp[0].n_order
                )
                diff_cand_cand = wdegp_utils.differences_by_dim_func(
                    x_cand_norm, x_cand_norm, gp[0].n_order
                )
                diff_domain_cand = wdegp_utils.differences_by_dim_func(
                    integration_points_norm, x_cand_norm, gp[0].n_order
                )
                
                # Precomputed diffs for this candidate
                local_precomputed_diffs = {
                    'train_cand': diff_train_cand,
                    'cand_train': diff_cand_train,
                    'cand_cand': diff_cand_cand,
                    'domain_cand': diff_domain_cand,
                    'train_domain': precomputed_diffs['train_domain']  # Reuse this one
                }
                
                # Compute aggregated score across all GPs
                score_list = [
                    np.sqrt(abs(acquisition_func(
                        gp_t,
                        params_t,
                        x_cand,
                        precomputed_diffs=local_precomputed_diffs,
                        n_integration_points = n_integration_points
                    )))

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
        if n_batch_points > 1:
            for t, gp_t in enumerate(gp):
                params_t = params[t]
                y_pred = gp_t.predict(best_x, params_t, calc_cov=False, return_submodels=False)
                y_to_add = utils._format_y_to_add(y_pred)
    
                x_aug = np.vstack([gp_t.x_train_input.copy(), best_x])
                y_aug= copy.deepcopy(gp_t.y_train_input)
                for i in range(len(y_aug)):
                        y_aug[i][0] = np.vstack([y_aug[i][0].copy(), y_to_add[0]])
    
                gp[t] = gp_t.__class__(
                    x_aug, y_aug,
                    gp_t.n_order,
                    n_bases=gp_t.n_bases,
                    index = gp_t.index,
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
    gp, params,candidate_points, precomputed_diffs = None,
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
    diff_train_cand = precomputed_diffs['train_cand']
    diff_cand_train = precomputed_diffs['cand_train']
    diff_cand_cand = precomputed_diffs['cand_cand']
    weights_matrix = wdegp_utils.determine_weights(
        diff_train_train, diff_cand_train, length_scales, gp.kernel_func, sigma_n)
    
    phi_train_train = gp.kernel_func(
    diff_train_train, length_scales)

    # Extract ALL derivative components into a single flat array (highly efficient)
    phi_exp_train_train = phi_train_train.get_all_derivs(
    gp.n_bases, 2 * gp.n_order)   
    # If Cholesky fails, fall back to standard solve
    phi_train_test = gp.kernel_func(diff_train_cand, length_scales)
    
    
    phi_test_test = gp.kernel_func(
        diff_cand_cand, length_scales)

    # Extract ALL derivative components into a single flat array (highly efficient)
    phi_exp_test_test = phi_test_test.get_all_derivs(
        gp.n_bases, 2 * gp.n_order)
    # Extract ALL derivative components into a single flat array (highly efficient)
    phi_exp_train_test = phi_train_test.get_all_derivs(
        gp.n_bases, 2 * gp.n_order)
    y_var = 0
    
    submodel_indices = gp.index
    for i in range(len(submodel_indices)):
        index_i = submodel_indices[i]
        K = wdegp_utils.rbf_kernel(
            phi_train_train, phi_exp_train_train, gp.n_order, gp.n_bases,
            gp.flattened_der_indicies[i], gp.powers[i], index=index_i
        )
        K.flat[::K.shape[0] + 1] += (10 ** sigma_n) ** 2
        
        # Use precomputed differences
    
        
        # Compute kernels from differences
        K_s = wdegp_utils.rbf_kernel_predictions(
            phi_train_test, phi_exp_train_test, gp.n_order, gp.n_bases,
            gp.flattened_der_indicies[i], gp.powers[i],return_deriv, index=index_i
        )

        n = weights_matrix.shape[0]
        
        K_ss = wdegp_utils.rbf_kernel_predictions(
            phi_test_test, phi_exp_test_test, gp.n_order, gp.n_bases,
            gp.flattened_der_indicies[i], gp.powers[i],return_deriv, index=index_i, calc_cov=True
        )
        
        # Compute posterior covariance
        f_cov = K_ss[:n, :n] -K_s[:, :n].T @ np.linalg.solve(K, K_s[:, :n])
        f_var = np.diag(np.abs(f_cov))
        unique_indices = set()
        for subindex in submodel_indices[i]:
            unique_indices.update(subindex)
        unique_indices = sorted(unique_indices)
        
        # Sum weights for all unique indices
        weight = np.zeros(weights_matrix.shape[0])  # Initialize weight vector
        for idx in unique_indices:
            weight = weight + weights_matrix[:, idx]
        y_var += weight * np.sqrt(f_var)
        
    
    return y_var**2

def gradient_selection(gp, next_points, params, integration_points, candidate_points, dist_params, method='PIUR', n_local_starts = 10, local_opt = True):
    if method == 'PIUR':
        params_t = params
        
        # ===== GP_T_1: Add function value at next_points[0] =====
        y_pred_1 = gp.predict(next_points[0], params_t, calc_cov=False, return_submodels=False, return_deriv = True)
        y_to_add_1 = utils._format_y_to_add(y_pred_1[0])

        x_aug_1 = np.vstack([gp.x_train_input.copy(), next_points[0]])
        y_aug_1 = copy.deepcopy(gp.y_train_input)
        for i in range(len(y_aug_1)):
            y_aug_1[i][0] = np.vstack([y_aug_1[i][0].copy(), y_to_add_1[0]])

        gp_t_1 = gp.__class__(
            x_aug_1, y_aug_1,
            gp.n_order,
            n_bases=gp.n_bases,
            index=gp.index,
            der_indices=gp.der_indices,
            normalize=gp.normalize,
            kernel=gp.kernel,
            kernel_type=gp.kernel_type,
        )
        
        # Calculate integrated variance for gp_t_1
        _, y_cov_1 = gp_t_1.predict(
            integration_points,
            params_t,
            calc_cov=True,
            return_submodels=False
        )
        integrated_var_1 = np.mean((abs(np.sqrt(y_cov_1))))  # or np.trapz if you have weights
        
        # ===== GP_T_2: Add function values at both points =====
        y_pred_2 = gp.predict(next_points[1], params_t, calc_cov=False, return_submodels=False, return_deriv= False)
        y_to_add_2 = utils._format_y_to_add(y_pred_2)

        x_aug_2 = np.vstack([x_aug_1.copy(), next_points[1]])
        y_aug_2 = copy.deepcopy(y_aug_1)
        for i in range(len(y_aug_2)):
            y_aug_2[i][0] = np.vstack([y_aug_2[i][0].copy(), y_to_add_2[0]])
                
        gp_t_2 = gp.__class__(
            x_aug_2, y_aug_2,
            gp.n_order,
            n_bases=gp.n_bases,
            index=gp.index,
            der_indices=gp.der_indices,
            normalize=gp.normalize,
            kernel=gp.kernel,
            kernel_type=gp.kernel_type,
        )
        
        # Calculate integrated variance for gp_t_2
        mu, y_cov_2 = gp_t_2.predict(
            integration_points,
            params_t,
            calc_cov=True,
            return_submodels=False
        )
        integrated_var_2 = np.mean((np.sqrt(abs(y_cov_2))))
        
        # ===== GP_T_3: Add function value + gradient at next_points[0] =====
        grad_x_next = y_pred_1[1]
        y_aug_3 = copy.deepcopy(gp.y_train_input)
        x_aug_3 = copy.deepcopy(x_aug_1)
        
        index_t = copy.deepcopy(gp.index)

        new_point_idx = len(x_aug_1) - 1
        
        base_der_indices = utils.gen_OTI_indices(gp.n_bases, gp.n_order)
        insert_pos = len(gp.index[0][0])
        
        for i in range(len(y_aug_3)):
            y_aug_3[i][0] = np.vstack([y_aug_3[i][0].copy(), y_to_add_1[0]])
                    
        
        # Append derivatives to first submodel
        deriv_idx = 1
        for i in range(len(base_der_indices)):
            for j in range(len(base_der_indices[i])):
                deriv = y_pred_1[j + 1].reshape(-1, 1)
                y_aug_3[0][deriv_idx] = np.vstack([
                    y_aug_3[0][deriv_idx],
                    deriv
                ])
                deriv_idx += 1
        
        index_t[0][0].append(insert_pos + gp.index[0][0][0])
        
        gp_t_3 = gp.__class__(
            x_aug_3, y_aug_3,
            gp.n_order,
            n_bases=gp.n_bases,
            index=index_t,
            der_indices=gp.der_indices,
            normalize=gp.normalize,
            kernel=gp.kernel,
            kernel_type=gp.kernel_type,
        )
        
        # Calculate integrated variance for gp_t_3
        _, y_cov_3 = gp_t_3.predict(
            integration_points,
            params_t,
            calc_cov=True,
            return_submodels=False
        )
        integrated_var_3 = np.mean((np.sqrt(abs(y_cov_3))))
        
        print(f"Integrated variance - GP1 (fn only): {integrated_var_1:.6f}")
        print(f"Integrated variance - GP2 (2 fns): {integrated_var_2:.6f}")
        print(f"Integrated variance - GP3 (fn+grad): {integrated_var_3:.6f}")
        
        PIUR_G = integrated_var_1 - integrated_var_3
        PIUR_R = integrated_var_1 - integrated_var_2
        T_G = 1.1
        rho = (PIUR_G/T_G)/(PIUR_R/1)
        
        AGS = rho
        
        # ===== Plotting Routine =====
        import matplotlib.pyplot as plt
        
        # Create a fine grid for plotting (assuming 1D or 2D input)
        if True:
          # 1D case
            x_plot = np.linspace(candidate_points.min(), candidate_points.max(), 200).reshape(-1, 1)
            
            # Get predictions from all four GPs
            mu_orig, cov_orig = gp.predict(x_plot, params_t, calc_cov=True, return_submodels=False)
            std_orig = np.sqrt(cov_orig)
            
            mu_1, cov_1 = gp_t_1.predict(x_plot, params_t, calc_cov=True, return_submodels=False)
            std_1 = np.sqrt(abs(cov_1))
            
            mu_2, cov_2 = gp_t_2.predict(x_plot, params_t, calc_cov=True, return_submodels=False)
            std_2 = np.sqrt(abs(cov_2))
            
            mu_3, cov_3 = gp_t_3.predict(x_plot, params_t, calc_cov=True, return_submodels=False)
            std_3 = np.sqrt(abs(cov_3))
            
            # Create output directory
            import os
            output_dir = 'gp_evolution_plots'
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine iteration number from current GP training data size
            n_training = len(gp.x_train_input)
            
            # Larger font sizes
            TITLE_SIZE = 22
            LABEL_SIZE = 20
            LEGEND_SIZE = 16
            TICK_SIZE = 16
            
            # Get x-axis limits for consistent scaling
            x_min, x_max = x_plot.min(), x_plot.max()
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            # ===== STEP 1: Plot Original GP =====
            ax1.plot(x_plot, mu_orig.flatten(), 'k-', label='Original GP', linewidth=3)
            ax1.fill_between(x_plot.ravel(), 
                            mu_orig.flatten() - 2*std_orig, 
                            mu_orig.flatten() + 2*std_orig, 
                            alpha=0.2, color='black')
            
            # Plot training points
            ax1.scatter(gp.x_train_input, gp.y_train_input[0][0], 
                       c='black', s=150, marker='o', edgecolors='white', 
                       linewidths=2, label='Training data', zorder=5)
            
            # Plot next selected points on original GP
            ax1.scatter(next_points[0], gp.predict(next_points[0], params_t, calc_cov=False)[0], 
                       c='blue', s=200, marker='*', edgecolors='white', 
                       linewidths=2, label='Next x₁', zorder=5, alpha=0.7)
            ax1.scatter(next_points[1], gp.predict(next_points[1], params_t, calc_cov=False)[0], 
                       c='green', s=200, marker='*', edgecolors='white', 
                       linewidths=2, label='Next x₂', zorder=5, alpha=0.7)
            
            ax1.set_xlabel('x', fontsize=LABEL_SIZE)
            ax1.set_ylabel('f(x)', fontsize=LABEL_SIZE)
            ax1.set_xlim(x_min, x_max)
            ax1.set_title('GP Predictions with 95% Confidence Intervals', fontsize=TITLE_SIZE, fontweight='bold')
            ax1.legend(loc='upper left', fontsize=LEGEND_SIZE)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=TICK_SIZE)
            
            # Empty second plot for now
            ax2.set_xlabel('x', fontsize=LABEL_SIZE)
            ax2.set_ylabel('Error Reduction (ME)', fontsize=LABEL_SIZE)
            ax2.set_xlim(x_min, x_max)
            ax2.set_title('Predictive Error Comparison', fontsize=TITLE_SIZE, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=TICK_SIZE)
            ax2.set_yscale('log')
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step1_original.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            # ===== STEP 2: Add GP + f(x₁) =====
            ax1.plot(x_plot, mu_1.flatten(), 'b-', label='GP + f(x₁)', linewidth=3)
            ax1.fill_between(x_plot.ravel(), 
                            mu_1.flatten() - 2*std_1, 
                            mu_1.flatten() + 2*std_1, 
                            alpha=0.2, color='blue')
            
            # Update the marker for x₁ to show it's been added
            ax1.collections[-2].set_alpha(1.0)  # Make x₁ marker fully opaque
            
            ax1.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            # Add to error plot
            ax2.axvline(next_points[0][0], color='blue', linestyle='--', alpha=0.5, 
                       linewidth=2, label='x₁ location')
            ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step2_add_x1.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            # ===== STEP 3: Add GP + f(x₁) + f(x₂) =====
            ax1.plot(x_plot, mu_2.flatten(), 'g-', label='GP + f(x₁) + f(x₂)', linewidth=3)
            ax1.fill_between(x_plot.ravel(), 
                            mu_2.flatten() - 2*std_2, 
                            mu_2.flatten() + 2*std_2, 
                            alpha=0.2, color='green')
            
            # Update the marker for x₂ to show it's been added
            ax1.collections[-1].set_alpha(1.0)  # Make x₂ marker fully opaque
            
            ax1.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            # Add to error plot
            ax2.axvline(next_points[1][0], color='green', linestyle='--', alpha=0.5, 
                       linewidth=2, label='x₂ location')
            ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step3_add_x2.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            # ===== STEP 4: Add GP + f(x₁) + ∇f(x₁) =====
            ax1.plot(x_plot, mu_3.flatten(), 'r-', label='GP + f(x₁) + ∇f(x₁)', linewidth=3)
            ax1.fill_between(x_plot.ravel(), 
                            mu_3.flatten() - 2*std_3, 
                            mu_3.flatten() + 2*std_3, 
                            alpha=0.2, color='red')
            
            ax1.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step4_add_gradient.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            # ===== STEP 5: Add first error curve (MSE₁ - MSE∇) =====
            ax2.plot(x_plot, abs(std_1 - std_3), 'b-', label='MSE₁ - MSE∇', linewidth=3)
            ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step5_error1.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            # ===== STEP 6: Add second error curve (MSE₁ - MSE₂) =====
            ax2.plot(x_plot, abs(std_1 - std_2), 'g-', label='MSE₁ - MSE₂', linewidth=3)
            ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step6_error2.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            # ===== STEP 7: Add ratio curve =====
            ax2.plot(x_plot, abs(std_1 - std_3)/abs(std_1 - std_2), 'r-', 
                    label=r'(MSE₁ - MSE∇)/(MSE₁ - MSE₂)', linewidth=3)
            ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step7_error_ratio.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            plt.show()
            plt.close()
            
            # Print summary statistics
            print("\n" + "="*60)
            print(f"GP Comparison Summary - Iteration {n_training}")
            print("="*60)
            print(f"Original GP - Integrated Variance: {np.mean(std_orig**2):.6f}")
            print(f"GP + f(x₁) - Integrated Variance: {np.mean(std_1**2):.6f}")
            print(f"GP + f(x₁) + f(x₂) - Integrated Variance: {np.mean(std_2**2):.6f}")
            print(f"GP + f(x₁) + ∇f(x₁) - Integrated Variance: {np.mean(std_3**2):.6f}")
            print(f"\nMax Variance Reduction (function only): {np.mean(std_orig**2) - np.mean(std_1**2):.6f}")
            print(f"Max Variance Reduction (two functions): {np.mean(std_orig**2) - np.mean(std_2**2):.6f}")
            print(f"Max Variance Reduction (function + gradient): {np.mean(std_orig**2) - np.mean(std_3**2):.6f}")
            print(f"\nPIUR_G (gradient): {PIUR_G:.6f}")
            print(f"PIUR_R (function): {PIUR_R:.6f}")
            print(f"AGS ratio (ρ): {AGS:.6f}")
            print("="*60)
        return AGS
    elif method == "PUR":
        params_t = params
        
        # ===== GP_T_1: Add function value at next_points[0] =====
        y_pred_1 = gp.predict(next_points[0], params_t, calc_cov=False, return_submodels=False)
        y_to_add_1 = utils._format_y_to_add(y_pred_1)

        x_aug_1 = np.vstack([gp.x_train_input.copy(), next_points[0]])
        y_aug_1 = copy.deepcopy(gp.y_train_input)
        for i in range(len(y_aug_1)):
            y_aug_1[i][0] = np.vstack([y_aug_1[i][0].copy(), y_to_add_1[0]])

        gp_t_1 = gp.__class__(
            x_aug_1, y_aug_1,
            gp.n_order,
            n_bases=gp.n_bases,
            index=gp.index,
            der_indices=gp.der_indices,
            normalize=gp.normalize,
            kernel=gp.kernel,
            kernel_type=gp.kernel_type,
        )
        
        
        # ===== GP_T_2: Add function values at both points =====
        y_pred_2 = gp.predict(next_points[1], params_t, calc_cov=False, return_submodels=False)
        y_to_add_2 = utils._format_y_to_add(y_pred_2)

        x_aug_2 = np.vstack([x_aug_1.copy(), next_points[1]])
        y_aug_2 = copy.deepcopy(y_aug_1)
        for i in range(len(y_aug_2)):
            y_aug_2[i][0] = np.vstack([y_aug_2[i][0].copy(), y_to_add_2[0]])
                
        gp_t_2 = gp.__class__(
            x_aug_2, y_aug_2,
            gp.n_order,
            n_bases=gp.n_bases,
            index=gp.index,
            der_indices=gp.der_indices,
            normalize=gp.normalize,
            kernel=gp.kernel,
            kernel_type=gp.kernel_type,
        )
        
       
        # ===== GP_T_3: Add function value + gradient at next_points[0] =====
        grad_x_next = finite_difference_gradient(gp, next_points[0], params)
        y_aug_3 = copy.deepcopy(gp.y_train_input)
        
    
        index_t = copy.deepcopy(gp.index)


        new_point_idx = len(x_aug_1) - 1
        base_der_indices = utils.gen_OTI_indices(gp.n_bases, gp.n_order)
        insert_pos = len(gp.index[0])
        
        for i in range(len(y_aug_3)):
            y_aug_3[i][0] = np.vstack([y_aug_3[i][0].copy(), y_to_add_1[0]])
                    
        if new_point_idx != insert_pos:
            temp1 = x_aug_1[[insert_pos, new_point_idx]].copy() 
            temp2 = x_aug_1[[new_point_idx, insert_pos]].copy()
            x_aug_1[[insert_pos, new_point_idx]] = temp2
            x_aug_1[[new_point_idx, insert_pos]] = temp1
            temp1 = y_aug_3[0][0][new_point_idx].copy() 
            temp2 = y_aug_3[0][0][insert_pos].copy()
            y_aug_3[0][0][new_point_idx] = temp2
            y_aug_3[0][0][insert_pos] = temp1
        
        deriv_idx = 1
        for i in range(len(base_der_indices)):
            for j in range(len(base_der_indices[i])):
                deriv = grad_x_next.reshape(-1, 1)
                y_aug_3[0][deriv_idx] = np.vstack([
                    y_aug_3[0][deriv_idx],
                    deriv
                ])
                deriv_idx += 1
        
        index_t[0].append(insert_pos + gp.index[0][0])
        
        gp_t_3 = gp.__class__(
            x_aug_1, y_aug_3,
            gp.n_order,
            n_bases=gp.n_bases,
            index=index_t,
            der_indices=gp.der_indices,
            normalize=gp.normalize,
            kernel=gp.kernel,
            kernel_type=gp.kernel_type,
        )
        
        _, y_cov_1 = gp_t_1.predict(
            candidate_points,
            params_t,
            calc_cov=True,
            return_submodels=False
        )
        _, y_cov_2 = gp_t_2.predict(
            candidate_points,
            params_t,
            calc_cov=True,
            return_submodels=False
        )
        _, y_cov_3 = gp_t_3.predict(
            candidate_points,
            params_t,
            calc_cov=True,
            return_submodels=False
        )
        T_G = 1.1
        PIUR_G = np.sqrt(abs(y_cov_1)) - np.sqrt(abs(y_cov_3))
        PIUR_R = np.sqrt(abs(y_cov_1)) - np.sqrt(abs(y_cov_2))
        rho = (PIUR_G / T_G) / (PIUR_R / 1.0)
        cand_scores_G = PIUR_G
        cand_scores_R = PIUR_R
        top_indices_G = np.argsort(cand_scores_G.ravel())[::-1][:n_local_starts]
        top_indices_R = np.argsort(cand_scores_R.ravel())[::-1][:n_local_starts]
        top_starting_points_G = candidate_points[top_indices_G]
        top_starting_points_R = candidate_points[top_indices_R]
        if local_opt:
            bounds = utils.get_optimization_bounds(dist_params)
            all_unbounded = all(b[0] is None for b in bounds)
            
            def objective_function_G(x: np.ndarray) -> float:
                x_cand = x.reshape(1, -1)
                
                _, y_cov_1 = gp_t_1.predict(
                    x_cand,
                    params_t,
                    calc_cov=True,
                    return_submodels=False
                )
                _, y_cov_3 = gp_t_3.predict(
                    x_cand,
                    params_t,
                    calc_cov=True,
                    return_submodels=False
                )
                score = np.sqrt(abs(y_cov_1)) - np.sqrt(abs(y_cov_3))
                
                return -score if not isinstance(score, np.ndarray) else -score[0]
            
            best_x, best_score = None, -np.inf
            for start_point in top_starting_points_G:
                if all_unbounded:
                    res = minimize(fun=objective_function_G, x0=start_point, method="BFGS")
                else:
                    res = minimize(fun=objective_function_G, x0=start_point, method="L-BFGS-B", bounds=bounds)
                
                if -res.fun > best_score:
                    best_score, best_x = -res.fun, res.x
            
            best_score_G = best_score
            
            def objective_function_R(x: np.ndarray) -> float:
                x_cand = x.reshape(1, -1)
                
                _, y_cov_1 = gp_t_1.predict(
                    x_cand,
                    params_t,
                    calc_cov=True,
                    return_submodels=False
                )
                _, y_cov_2 = gp_t_2.predict(
                    x_cand,
                    params_t,
                    calc_cov=True,
                    return_submodels=False
                )
                score = np.sqrt(abs(y_cov_1)) - np.sqrt(abs(y_cov_2))
                
                return -score if not isinstance(score, np.ndarray) else -score[0]
            
            best_x, best_score = None, -np.inf
            for start_point in top_starting_points_R:
                if all_unbounded:
                    res = minimize(fun=objective_function_R, x0=start_point, method="BFGS")
                else:
                    res = minimize(fun=objective_function_R, x0=start_point, method="L-BFGS-B", bounds=bounds)
                
                if -res.fun > best_score:
                    best_score, best_x = -res.fun, res.x
            
            best_score_R = best_score
        else:
            best_score_R = np.max(cand_scores_G)
            best_score_G = np.max(cand_scores_G)

        T_G = 1.1
        PIUR_G = best_score_G
        PIUR_R = best_score_R
        rho = (PIUR_G / T_G) / (PIUR_R / 1.0)
        
        AGS = rho
        
        # ===== Plotting Routine =====
        import matplotlib.pyplot as plt
        
        # Create a fine grid for plotting (assuming 1D or 2D input)
        if True:
          # 1D case
            x_plot = np.linspace(candidate_points.min(), candidate_points.max(), 200).reshape(-1, 1)
            
            # Get predictions from all four GPs
            mu_orig, cov_orig = gp.predict(x_plot, params_t, calc_cov=True, return_submodels=False)
            std_orig = np.sqrt(cov_orig)
            
            mu_1, cov_1 = gp_t_1.predict(x_plot, params_t, calc_cov=True, return_submodels=False)
            std_1 = np.sqrt(abs(cov_1))
            
            mu_2, cov_2 = gp_t_2.predict(x_plot, params_t, calc_cov=True, return_submodels=False)
            std_2 = np.sqrt(abs(cov_2))
            
            mu_3, cov_3 = gp_t_3.predict(x_plot, params_t, calc_cov=True, return_submodels=False)
            std_3 = np.sqrt(abs(cov_3))
            
            # Create output directory
            import os
            output_dir = 'gp_evolution_plots_PUR'
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine iteration number from current GP training data size
            n_training = len(gp.x_train_input)
            
            # Larger font sizes
            TITLE_SIZE = 22
            LABEL_SIZE = 20
            LEGEND_SIZE = 16
            TICK_SIZE = 16
            
            # Get x-axis limits for consistent scaling
            x_min, x_max = x_plot.min(), x_plot.max()
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
            
            # ===== STEP 1: Plot Original GP =====
            ax1.plot(x_plot, mu_orig.flatten(), 'k-', label='Original GP', linewidth=3)
            ax1.fill_between(x_plot.ravel(), 
                            mu_orig.flatten() - 2*std_orig, 
                            mu_orig.flatten() + 2*std_orig, 
                            alpha=0.2, color='black')
            
            # Plot training points
            ax1.scatter(gp.x_train_input, gp.y_train_input[0][0], 
                       c='black', s=150, marker='o', edgecolors='white', 
                       linewidths=2, label='Training data', zorder=5)
            
            # Plot next selected points on original GP
            ax1.scatter(next_points[0], gp.predict(next_points[0], params_t, calc_cov=False)[0], 
                       c='blue', s=200, marker='*', edgecolors='white', 
                       linewidths=2, label='Next x₁', zorder=5, alpha=0.7)
            ax1.scatter(next_points[1], gp.predict(next_points[1], params_t, calc_cov=False)[0], 
                       c='green', s=200, marker='*', edgecolors='white', 
                       linewidths=2, label='Next x₂', zorder=5, alpha=0.7)
            
            ax1.set_xlabel('x', fontsize=LABEL_SIZE)
            ax1.set_ylabel('f(x)', fontsize=LABEL_SIZE)
            ax1.set_xlim(x_min, x_max)
            ax1.set_title('GP Predictions with 95% Confidence Intervals', fontsize=TITLE_SIZE, fontweight='bold')
            ax1.legend(loc='upper left', fontsize=LEGEND_SIZE)
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(labelsize=TICK_SIZE)
            
            # Empty second plot for now
            ax2.set_xlabel('x', fontsize=LABEL_SIZE)
            ax2.set_ylabel('Error Reduction (ME)', fontsize=LABEL_SIZE)
            ax2.set_xlim(x_min, x_max)
            ax2.set_title('Predictive Error Comparison', fontsize=TITLE_SIZE, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.tick_params(labelsize=TICK_SIZE)
            ax2.set_yscale('log')
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step1_original.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            # ===== STEP 2: Add GP + f(x₁) =====
            ax1.plot(x_plot, mu_1.flatten(), 'b-', label='GP + f(x₁)', linewidth=3)
            ax1.fill_between(x_plot.ravel(), 
                            mu_1.flatten() - 2*std_1, 
                            mu_1.flatten() + 2*std_1, 
                            alpha=0.2, color='blue')
            
            # Update the marker for x₁ to show it's been added
            ax1.collections[-2].set_alpha(1.0)  # Make x₁ marker fully opaque
            
            ax1.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            # Add to error plot
            ax2.axvline(next_points[0][0], color='blue', linestyle='--', alpha=0.5, 
                       linewidth=2, label='x₁ location')
            ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step2_add_x1.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            # ===== STEP 3: Add GP + f(x₁) + f(x₂) =====
            ax1.plot(x_plot, mu_2.flatten(), 'g-', label='GP + f(x₁) + f(x₂)', linewidth=3)
            ax1.fill_between(x_plot.ravel(), 
                            mu_2.flatten() - 2*std_2, 
                            mu_2.flatten() + 2*std_2, 
                            alpha=0.2, color='green')
            
            # Update the marker for x₂ to show it's been added
            ax1.collections[-1].set_alpha(1.0)  # Make x₂ marker fully opaque
            
            ax1.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            # Add to error plot
            ax2.axvline(next_points[1][0], color='green', linestyle='--', alpha=0.5, 
                       linewidth=2, label='x₂ location')
            ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step3_add_x2.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            # ===== STEP 4: Add GP + f(x₁) + ∇f(x₁) =====
            ax1.plot(x_plot, mu_3.flatten(), 'r-', label='GP + f(x₁) + ∇f(x₁)', linewidth=3)
            ax1.fill_between(x_plot.ravel(), 
                            mu_3.flatten() - 2*std_3, 
                            mu_3.flatten() + 2*std_3, 
                            alpha=0.2, color='red')
            
            ax1.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step4_add_gradient.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            # ===== STEP 5: Add first error curve (MSE₁ - MSE∇) =====
            ax2.plot(x_plot, abs(std_1 - std_3), 'b-', label='MSE₁ - MSE∇', linewidth=3)
            ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step5_error1.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            # ===== STEP 6: Add second error curve (MSE₁ - MSE₂) =====
            ax2.plot(x_plot, abs(std_1 - std_2), 'g-', label='MSE₁ - MSE₂', linewidth=3)
            ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step6_error2.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            # ===== STEP 7: Add ratio curve =====
            ax2.plot(x_plot, abs(std_1 - std_3)/abs(std_1 - std_2), 'r-', 
                    label=r'(MSE₁ - MSE∇)/(MSE₁ - MSE₂)', linewidth=3)
            ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
            
            plt.tight_layout()
            filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step7_error_ratio.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved: {filename}")
            
            plt.show()
            plt.close()
            
            # Print summary statistics
            print("\n" + "="*60)
            print(f"GP Comparison Summary - Iteration {n_training}")
            print("="*60)
            print(f"Original GP - Integrated Variance: {np.mean(std_orig**2):.6f}")
            print(f"GP + f(x₁) - Integrated Variance: {np.mean(std_1**2):.6f}")
            print(f"GP + f(x₁) + f(x₂) - Integrated Variance: {np.mean(std_2**2):.6f}")
            print(f"GP + f(x₁) + ∇f(x₁) - Integrated Variance: {np.mean(std_3**2):.6f}")
            print(f"\nMax Variance Reduction (function only): {np.mean(std_orig**2) - np.mean(std_1**2):.6f}")
            print(f"Max Variance Reduction (two functions): {np.mean(std_orig**2) - np.mean(std_2**2):.6f}")
            print(f"Max Variance Reduction (function + gradient): {np.mean(std_orig**2) - np.mean(std_3**2):.6f}")
            print(f"\nPIUR_G (gradient): {PIUR_G:.6f}")
            print(f"PIUR_R (function): {PIUR_R:.6f}")
            print(f"AGS ratio (ρ): {AGS:.6f}")
            print("="*60)
        return AGS
    elif method == "PEIR":
        params_t = params
        
        # ===== GP_T_1: Add function value at next_points[0] =====
        y_pred_1 = gp.predict(next_points[0], params_t, calc_cov=False, return_submodels=False, return_deriv = True)
        y_to_add_1 = utils._format_y_to_add(y_pred_1[0,0])
   
        x_aug_1 = np.vstack([gp.x_train_input.copy(), next_points[0]])
        y_aug_1 = copy.deepcopy(gp.y_train_input)
        for i in range(len(y_aug_1)):
            y_aug_1[i][0] = np.vstack([y_aug_1[i][0].copy(), y_to_add_1[0]])
   
        gp_t_1 = gp.__class__(
            x_aug_1, y_aug_1,
            gp.n_order,
            n_bases=gp.n_bases,
            index=gp.index,
            der_indices=gp.der_indices,
            normalize=gp.normalize,
            kernel=gp.kernel,
            kernel_type=gp.kernel_type,
        )
        
        # Calculate integrated variance for gp_t_1
        expected_improvement_1 = EI(gp_t_1, params,
            next_points[1],
        )
     
         
        # ===== GP_T_3: Add function value + gradient at next_points[0] =====
        grad_x_next = y_pred_1[1]
        y_aug_3 = copy.deepcopy(gp.y_train_input)
        x_aug_3 = copy.deepcopy(x_aug_1)
        
        index_t = copy.deepcopy(gp.index)

        new_point_idx = len(x_aug_1) - 1
        
        base_der_indices = utils.gen_OTI_indices(gp.n_bases, gp.n_order)
        insert_pos = len(gp.index[0])
        
        for i in range(len(y_aug_3)):
            y_aug_3[i][0] = np.vstack([y_aug_3[i][0].copy(), y_to_add_1[0]])
                    
        
        # Append derivatives to first submodel
        deriv_idx = 1
        for i in range(len(base_der_indices)):
            for j in range(len(base_der_indices[i])):
                deriv = y_pred_1[j + 1].reshape(-1, 1)
                y_aug_3[0][deriv_idx] = np.vstack([
                    y_aug_3[0][deriv_idx],
                    deriv
                ])
                deriv_idx += 1
        
        index_t[0][0].append(insert_pos + gp.index[0][0][0])
        
        gp_t_3 = gp.__class__(
            x_aug_3, y_aug_3,
            gp.n_order,
            n_bases=gp.n_bases,
            index=index_t,
            der_indices=gp.der_indices,
            normalize=gp.normalize,
            kernel=gp.kernel,
            kernel_type=gp.kernel_type,
        )
        
        # Calculate integrated variance for gp_t_1
        expected_improvement_2 = EI(gp_t_3, params,
           next_points[1])
        
     
    PIUR_G = expected_improvement_1 - expected_improvement_2
    PIUR_R = expected_improvement_1
    
    T_G = 1.1
    rho = (PIUR_G / T_G) / (PIUR_R / 1.0)
    
    AGS = rho
    
    # ===== Plotting Routine =====
    import matplotlib.pyplot as plt
    
    # Create a fine grid for plotting (assuming 1D or 2D input)
    if True:
          # 1D case
          x_plot = np.linspace(candidate_points.min(), candidate_points.max(), 200).reshape(-1, 1)
          
          # Get predictions from all four GPs
          mu_orig, cov_orig = gp.predict(x_plot, params_t, calc_cov=True, return_submodels=False)
          std_orig = np.sqrt(cov_orig)
          EI_ord = EI(gp, params_t, x_plot)
          mu_1, cov_1 = gp_t_1.predict(x_plot, params_t, calc_cov=True, return_submodels=False)
          std_1 = np.sqrt(abs(cov_1))
          EI_1 = EI(gp_t_1, params_t, x_plot)
          mu_2, cov_2 = gp_t_3.predict(x_plot, params_t, calc_cov=True, return_submodels=False)
          std_2 = np.sqrt(abs(cov_2))
          EI_2 = EI(gp_t_3, params_t, x_plot)
    
          import os
          output_dir = 'gp_evolution_plots_EI'
          os.makedirs(output_dir, exist_ok=True)
          
          # Determine iteration number from current GP training data size
          n_training = len(gp.x_train_input)
          
          # Larger font sizes
          TITLE_SIZE = 22
          LABEL_SIZE = 20
          LEGEND_SIZE = 16
          TICK_SIZE = 16
          
          # Get x-axis limits for consistent scaling
          x_min, x_max = x_plot.min(), x_plot.max()
          
          # Create figure with 2 subplots
          fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
          
          # ===== STEP 1: Plot Original GP =====
          ax1.plot(x_plot, mu_orig.flatten(), 'k-', label='Original GP', linewidth=3)
          ax1.fill_between(x_plot.ravel(), 
                          mu_orig.flatten() - 2*std_orig, 
                          mu_orig.flatten() + 2*std_orig, 
                          alpha=0.2, color='black')
          
          # Plot training points
          ax1.scatter(gp.x_train_input, gp.y_train_input[0][0], 
                     c='black', s=150, marker='o', edgecolors='white', 
                     linewidths=2, label='Training data', zorder=5)
          
          # Plot next selected points on original GP
          ax1.scatter(next_points[0], gp.predict(next_points[0], params_t, calc_cov=False)[0], 
                     c='blue', s=200, marker='*', edgecolors='white', 
                     linewidths=2, label='Next x₁', zorder=5, alpha=0.7)
          ax1.scatter(next_points[1], gp.predict(next_points[1], params_t, calc_cov=False)[0], 
                     c='green', s=200, marker='*', edgecolors='white', 
                     linewidths=2, label='Next x₂', zorder=5, alpha=0.7)
          
          ax1.set_xlabel('x', fontsize=LABEL_SIZE)
          ax1.set_ylabel('f(x)', fontsize=LABEL_SIZE)
          ax1.set_xlim(x_min, x_max)
          ax1.set_title('GP Predictions with 95% Confidence Intervals', fontsize=TITLE_SIZE, fontweight='bold')
          ax1.legend(loc='upper left', fontsize=LEGEND_SIZE)
          ax1.grid(True, alpha=0.3)
          ax1.tick_params(labelsize=TICK_SIZE)
          
          # Empty second plot for now
          ax2.set_xlabel('x', fontsize=LABEL_SIZE)
          ax2.set_ylabel('Expected Improvement (EI)', fontsize=LABEL_SIZE)
          ax2.set_xlim(x_min, x_max)
          ax2.set_ylim(top = 10*max(EI_ord), bottom=1e-6)  # Set lower bound to 1e-7
          ax2.set_title('Predictive Error Comparison', fontsize=TITLE_SIZE, fontweight='bold')
          ax2.grid(True, alpha=0.3)
          ax2.tick_params(labelsize=TICK_SIZE)
          ax2.set_yscale('log')
          
          plt.tight_layout()
          filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step1_original.png'
          plt.savefig(filename, dpi=300, bbox_inches='tight')
          print(f"Saved: {filename}")
          
          # ===== STEP 2: Add GP + f(x₁) =====
          ax1.plot(x_plot, mu_1.flatten(), 'b-', label='GP + f(x₁)', linewidth=3)
          ax1.fill_between(x_plot.ravel(), 
                          mu_1.flatten() - 2*std_1, 
                          mu_1.flatten() + 2*std_1, 
                          alpha=0.2, color='blue')
          
          # Update the marker for x₁ to show it's been added
          ax1.collections[-2].set_alpha(1.0)  # Make x₁ marker fully opaque
          
          ax1.legend(loc='upper left', fontsize=LEGEND_SIZE)
          
          # Add to error plot
          ax2.axvline(next_points[0][0], color='blue', linestyle='--', alpha=0.5, 
                     linewidth=2, label='x₁ location')
          ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
          
          plt.tight_layout()
          filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step2_add_x1.png'
          plt.savefig(filename, dpi=300, bbox_inches='tight')
          print(f"Saved: {filename}")
          
         
          # ===== STEP 4: Add GP + f(x₁) + ∇f(x₁) =====
          ax1.plot(x_plot, mu_2.flatten(), 'r-', label='GP + f(x₁) + ∇f(x₁)', linewidth=3)
          ax1.fill_between(x_plot.ravel(), 
                          mu_2.flatten() - 2*std_2, 
                          mu_2.flatten() + 2*std_2, 
                          alpha=0.2, color='red')
          
          ax1.legend(loc='upper left', fontsize=LEGEND_SIZE)
          
          plt.tight_layout()
          filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step4_add_gradient.png'
          plt.savefig(filename, dpi=300, bbox_inches='tight')
          print(f"Saved: {filename}")
          
          # ===== STEP 5: Add first error curve (MSE₁ - MSE∇) =====
          ax2.plot(x_plot, EI_ord, 'b-', label='Ord EI', linewidth=3)
          ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
          
          plt.tight_layout()
          filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step5_error1.png'
          plt.savefig(filename, dpi=300, bbox_inches='tight')
          print(f"Saved: {filename}")
          
          # ===== STEP 6: Add second error curve (MSE₁ - MSE₂) =====
          ax2.plot(x_plot, EI_1, 'g-', label='EI Regular Augemented', linewidth=3)
          ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
          
          plt.tight_layout()
          filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step6_error2.png'
          plt.savefig(filename, dpi=300, bbox_inches='tight')
          print(f"Saved: {filename}")
          
          # ===== STEP 7: Add ratio curve =====
          ax2.plot(x_plot,EI_2 , 'r-', 
                  label=r'EI Gradient Augemented', linewidth=3)
          ax2.legend(loc='upper left', fontsize=LEGEND_SIZE)
          
          plt.tight_layout()
          filename = f'{output_dir}/gp_comparison_iter_{n_training:03d}_step7_error_ratio.png'
          plt.savefig(filename, dpi=300, bbox_inches='tight')
          print(f"Saved: {filename}")
          
          plt.show()
          plt.close()
          
          # Print summary statistics
          print("\n" + "="*60)
          print(f"GP Comparison Summary - Iteration {n_training}")
          print("="*60)
          print(f"Original GP - Integrated Variance: {np.mean(std_orig**2):.6f}")
          print(f"GP + f(x₁) - Integrated Variance: {np.mean(std_1**2):.6f}")
          print(f"GP + f(x₁) + ∇f(x₁) - Integrated Variance: {np.mean(std_2**2):.6f}")
          # print(f"\nMax Variance Reduction (function only): {np.mean(std_orig**2) - np.mean(std_1**2):.6f}")
          # print(f"Max Variance Reduction (two functions): {np.mean(std_orig**2) - np.mean(std_2**2):.6f}")
          # print(f"Max Variance Reduction (function + gradient): {np.mean(std_orig**2) - np.mean(std_3**2):.6f}")
          # print(f"\nPIUR_G (gradient): {PIUR_G:.6f}")
          # print(f"PIUR_R (function): {PIUR_R:.6f}")
          # print(f"AGS ratio (ρ): {AGS:.6f}")
          print("="*60)
    return AGS

def imse_reduction(
    gp, params,candidate_points, precomputed_diffs = None,
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
    K = wdegp_utils.rbf_kernel(
        diff_train_train, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=True
    )
    K.flat[::K.shape[0] + 1] += (10 ** sigma_n) ** 2
    
    # Use precomputed differences
    diff_train_domain = precomputed_diffs['train_domain']
    diff_train_cand = precomputed_diffs['train_cand']
    diff_cand_cand = precomputed_diffs['cand_cand']
    diff_domain_cand = precomputed_diffs['domain_cand']
    
    # Compute kernels from differences
    K_s = wdegp_utils.rbf_kernel(
        diff_train_domain, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=return_deriv
    )
    
    K_train_cand = wdegp_utils.rbf_kernel(
        diff_train_cand, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=return_deriv
    )
    
    K_cand_cand = np.diag(wdegp_utils.rbf_kernel(
        diff_cand_cand, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=return_deriv
    ))
    

    K_domain_cand = wdegp_utils.rbf_kernel( diff_domain_cand, length_scales, gp.n_order, gp.n_bases, gp.kernel_func, gp.flattened_der_indicies, gp.powers, return_deriv=return_deriv ) 
    # shape: (n_domain, n_candidates) # Solve K^{-1} @ K_train_cand for all candidates 
    v = np.linalg.solve(K, K_train_cand) # shape: (n_train, n_candidates) # Numerator and denominator of variance reduction 
    numerator = (K_domain_cand[:n_integration_points,:] - K_s.T @ v) ** 2 # shape: (n_domain, n_candidates) 
    denominator = K_cand_cand- np.sum(K_train_cand * v, axis=0) # shape: (n_candidates,) denominator = np.maximum(denominator, 1e-16) 
    variance_reduction = (numerator / denominator) # shape: (n_domain, n_candidates) # IMSE reduction for each candidate = mean over domain points
    imse_reductions = variance_reduction.mean(axis=0) # shape: (n_candidates,)
    return imse_reductions


def EI(gp_model,params, x_test, precomputed_diffs = None,
                n_integration_points = None):
    """
    Compute Expected Improvement at test points.
    
    EI(x) = E[max(f_min - f(x), 0)]
          = (f_min - μ(x)) * Φ(Z) + σ(x) * φ(Z)
    where Z = (f_min - μ(x)) / σ(x)
    
    Args:
        X_test: Test points (n_test, n_dim)
        gp_model: Current GP model
        params: Optimized hyperparameters
        
    Returns:
        ei: Expected improvement values (n_test,)
    """
    # Get predictions (y_cov is variance, not covariance matrix)
    y_pred, y_cov = gp_model.predict(
        x_test, params, calc_cov=True, return_submodels=False
    )
    
    mu = y_pred.flatten()
    sigma = np.sqrt(abs(y_cov.flatten()))
    
    # Expected Improvement calculation
    f_min = np.min(gp_model.y_train_input[0][0])
    
    # Improvement
    improvement = f_min - mu
    
    # Standardized improvement
    Z = improvement / sigma
    
    # Expected Improvement
    ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
    
    # Set EI to 0 where sigma is very small
    #ei[sigma < 1e-10] = 0.0
    
    return ei