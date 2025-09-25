# acquisition_functions.py

import numpy as np
from full_degp import degp_utils
from scipy.linalg import cholesky, solve_triangular, LinAlgError
from full_degp.degp import degp
from scipy.linalg import cho_solve, cho_factor
import utils
from line_profiler import profile

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
def imse_reduction_efficient(
    gp, params, x_train, y_train_list,y_var,
    candidate_points, full_domain, noise_var=None,
    return_deriv=False, normalize=True
):
    """
    Vectorized exact IMSE reduction for scalar GP (function only).
    Computes IMSE reduction for all candidates at once without looping.
    """
    X_train = x_train
    flattened_der_indicies = gp.flattened_der_indicies
    num_integraion_points= full_domain.shape[0]
    if normalize: 
        (
        y_train,
        mu_y,
        sigma_y,
        sigmas_x,
        mus_x,
        sigma_data,
        ) = utils.normalize_y_data(x_train, y_train_list, noise_var, flattened_der_indicies)
        X_train = utils.normalize_x_data_test(x_train, sigmas_x, mus_x)
        candidate_points = utils.normalize_x_data_test(candidate_points, sigmas_x, mus_x)
        full_domain = utils.normalize_x_data_test(full_domain, sigmas_x, mus_x)
    
    candidate_points = np.sort(candidate_points, axis = 0)
    length_scales = params[:-1]
    sigma_n = params[-1]

    # --- 1. Training kernel ---
    #diff_train_train = gp.differences_by_dim
    K = gp.K

    # --- 2. Kernel matrices for full domain ---
    #diff_train_domain = gp.diff_x_test_x_train
    K_s = gp.K_s

    # --- 3. Vectorized differences ---
    # Candidate to training
    diff_train_cand = degp_utils.differences_by_dim_func(X_train, candidate_points, gp.n_order, return_deriv=False)
    K_train_cand = degp_utils.rbf_kernel(
        diff_train_cand, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=False
    )  # shape: (n_train, n_candidates)

    # Candidate self-covariance
    diff_cand_cand = degp_utils.differences_by_dim_func(candidate_points, candidate_points, gp.n_order, return_deriv=False)
    K_cand_cand = np.diag(degp_utils.rbf_kernel(
        diff_cand_cand, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=False
    ))  # shape: (n_candidates,)

    # Candidate to full domain
    diff_domain_cand = degp_utils.differences_by_dim_func(full_domain, candidate_points, gp.n_order, return_deriv=False)
    K_domain_cand = degp_utils.rbf_kernel(
        diff_domain_cand, length_scales, gp.n_order, gp.n_bases, gp.kernel_func,
        gp.flattened_der_indicies, gp.powers, return_deriv=False
    )  # shape: (n_domain, n_candidates)

    # Solve K^{-1} @ K_train_cand for all candidates
    v = np.linalg.solve(K, K_train_cand)  # shape: (n_train, n_candidates)

    # Numerator and denominator of variance reduction
    numerator = (K_domain_cand[:num_integraion_points,:]  - K_s.T @ v) ** 2  # shape: (n_domain, n_candidates)
    denominator = K_cand_cand- np.sum(K_train_cand * v, axis=0)  # shape: (n_candidates,)
    denominator = np.maximum(denominator, 1e-16)

    variance_reduction = (numerator / denominator)  # shape: (n_domain, n_candidates)

    # IMSE reduction for each candidate = mean over domain points
    imse_reductions = variance_reduction.mean(axis=0)  # shape: (n_candidates,)

    return imse_reductions
