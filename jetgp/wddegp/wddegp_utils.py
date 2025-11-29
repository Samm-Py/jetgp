import pyoti.core as coti  # Required for the advanced implementation
import numpy as np
import pyoti.sparse as oti
from line_profiler import profile


def differences_by_dim_func(X1, X2, rays, n_order, index=-1, index_list=[]):
    """
    Compute directional differences between input arrays X1 and X2 along specified rays.

    Parameters
    ----------
    X1 : ndarray
        Array of shape (n_samples1, n_features) representing the first set of input points.
    X2 : ndarray
        Array of shape (n_samples2, n_features) representing the second set of input points.
    rays : ndarray
        Array of shape (n_features, n_rays) representing direction vectors.
    n_order : int
        Maximum derivative order for hypercomplex expansion.
    index : int, default=-1
        Index of the current submodel (used to select directional rays specific to this submodel).
    index_list : list, default=[]
        List of indices corresponding to the training points included in the current submodel.

    Returns
    -------
    differences_by_dim : list of ndarray
        List containing the directional differences for each feature dimension.
    """
    X1 = oti.array(X1)
    X2 = oti.array(X2)
    n1, d = X1.shape
    n2, d = X2.shape

    differences_by_dim = []
    rays = rays[index]

    for k in range(d):
        diffs_k = oti.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                dire1 = 0
                for l in range(rays.shape[1]):
                    dire1 += oti.e(l + 1, order=2 * n_order) * rays[k, l]
                if i in index_list or j in index_list:
                    diffs_k[i, j] = (X1[i, k] + dire1) - X2[j, k]
                else:
                    diffs_k[i, j] = X1[i, k] - X2[j, k]
        differences_by_dim.append(diffs_k)

    return differences_by_dim


def deriv_map(nbases, order):
    """
    Creates a mapping from (order, index_within_order) to a single
    flattened index for all derivative components.
    """
    k = 0
    map_deriv = []
    for ordi in range(order + 1):
        ndir = coti.ndir_order(nbases, ordi)
        map_deriv_i = [0] * ndir
        for idx in range(ndir):
            map_deriv_i[idx] = k
            k += 1
        map_deriv.append(map_deriv_i)
    return map_deriv


def transform_der_indices(der_indices, der_map):
    """
    Transforms a list of user-facing derivative specifications into the
    internal (order, index) format and the final flattened index.
    """
    deriv_ind_transf = []
    deriv_ind_order = []
    for deriv in der_indices:
        imdir = coti.imdir(deriv)
        idx, order = imdir
        deriv_ind_transf.append(der_map[order][idx])
        deriv_ind_order.append(imdir)
    return deriv_ind_transf, deriv_ind_order


@profile
def rbf_kernel(
    differences,
    length_scales,
    n_order,
    n_bases,
    kernel_func,
    der_indices,
    powers,
    index=-1,
):
    """
    Constructs the RBF kernel matrix with derivative entries using an
    efficient pre-allocation strategy combined with a single call to
    extract all derivative components.

    Parameters
    ----------
    differences : list of ndarray
        Precomputed pairwise differences by dimension.
    length_scales : ndarray
        Kernel hyperparameters.
    n_order : int
        Maximum derivative order.
    n_bases : int
        Number of hypercomplex bases (input dimensions).
    kernel_func : callable
        Base kernel function that returns a hypercomplex result.
    der_indices : list
        List of derivative multi-indices to include.
    powers : list
        Parity powers for applying sign conventions.
    index : list or int, optional
        Indices of points that have derivative information.

    Returns
    -------
    K : ndarray
        Fully assembled kernel matrix with function and derivative blocks.
    """
    # --- 1. Initial Setup and Efficient Derivative Extraction ---
    dh = coti.get_dHelp()
    phi = kernel_func(differences, length_scales)

    # Extract ALL derivative components into a single flat array (highly efficient)
    phi_exp = phi.get_all_derivs(n_bases, 2 * n_order)

    # Create maps to translate derivative specifications to flat indices
    der_map = deriv_map(n_bases, 2 * n_order)
    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)

    # --- 2. Determine Block Sizes and Pre-allocate Matrix ---
    n_rows_func, n_cols_func = phi.shape
    n_deriv_types = len(der_indices)
    n_pts_with_derivs_cols = len([i for i in range(n_cols_func) if i in index])
    n_pts_with_derivs_rows = len([i for i in range(n_rows_func) if i in index])
    total_rows = n_rows_func + n_pts_with_derivs_rows * n_deriv_types
    total_cols = n_cols_func + n_pts_with_derivs_cols * n_deriv_types

    K = np.zeros((total_rows, total_cols))

    # The full content blocks are always the size of the original phi matrix
    base_shape = (n_rows_func, n_cols_func)

    # --- 3. Fill the Matrix Block by Block ---

    # Block (0,0): Function-Function (K_ff)
    # The real part is always at index 0 of the flat array
    content_full = phi_exp[0].reshape(base_shape)
    K[:n_rows_func, :n_cols_func] = content_full * ((-1) ** powers[0])

    # First Block-Row: Function-Derivative (K_fd)
    col_offset = n_cols_func
    for j in range(n_deriv_types):
        flat_idx = der_indices_tr[j]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        # Slice the content to match the number of derivative points
        content_sliced = content_full[:, index[0]:index[-1]+1]

        K[:n_rows_func, col_offset: col_offset +
            n_pts_with_derivs_cols ] = content_sliced * ((-1) ** powers[j + 1])
        col_offset += n_pts_with_derivs_cols

    # First Block-Column: Derivative-Function (K_df)
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        flat_idx = der_indices_tr[i]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        # Slice the content to match the number of derivative points
        content_sliced = content_full[index[0]:index[-1]+1, :]

        K[row_offset: row_offset +  n_pts_with_derivs_rows,
            :n_cols_func] = content_sliced * ((-1) ** powers[0])
        row_offset += n_pts_with_derivs_rows

    # Inner Blocks: Derivative-Derivative (K_dd)
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        col_offset = n_cols_func
        for j in range(n_deriv_types):
            # Multiply the derivative indices to find the correct flat index
            imdir1 = der_ind_order[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(
                imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            flat_idx = der_map[new_ord][new_idx]

            content_full = phi_exp[flat_idx].reshape(base_shape)
            # Slice the content for the derivative-derivative block
            content_sliced = content_full[index[0]
                :index[-1]+1, index[0]:index[-1]+1]

            K[row_offset: row_offset +  n_pts_with_derivs_rows, col_offset: col_offset +
                n_pts_with_derivs_cols] = content_sliced * ((-1) ** powers[j + 1])
            col_offset += n_pts_with_derivs_cols
        row_offset += n_pts_with_derivs_rows

    return K

# def rbf_kernel(differences, length_scales, n_order, n_bases, kernel_func, der_indices, powers, index=-1, index_list=[]):
#     """
#     Construct the kernel matrix for the weighted directional derivative-enhanced GP.

#     Parameters
#     ----------
#     differences : list of ndarray
#         Precomputed pairwise differences by dimension.
#     length_scales : ndarray
#         Array of length scales (hyperparameters).
#     n_order : int
#         Maximum derivative order.
#     n_bases : int
#         Number of OTI bases.
#     kernel_func : callable
#         Kernel function evaluated over differences.
#     der_indices : list
#         Indices indicating derivative orders.
#     powers : list
#         Powers of each term based on OTI basis structure.
#     index : int, default=-1
#         Dummy argument for compatibility.
#     index_list : list, default=[]
#         List identifying index subsets for directional derivatives.

#     Returns
#     -------
#     K : ndarray
#         Fully assembled kernel matrix including derivatives.
#     """
#     phi = kernel_func(differences, length_scales, index)

#     for i in range(0, len(der_indices) + 1):
#         row_j = 0
#         for j in range(0, len(der_indices) + 1):
#             if j == 0 and i == 0:
#                 row_j = phi.real * (-1)**(powers[j])
#             elif j > 0 and i == 0:
#                 row_j = np.hstack((
#                     row_j,
#                     (-1)**(powers[j]) * phi[:, index[0]: index[-1] + 1].get_deriv(der_indices[j - 1]),
#                 ))
#             elif j == 0 and i > 0:
#                 row_j = phi[index[0]: index[-1] + 1,
#                             :].get_deriv(der_indices[i - 1])
#             else:
#                 row_j = np.hstack((
#                     row_j,
#                     (-1)**(powers[j]) * np.array(
#                         phi[index[0]: index[-1] + 1, index[0]: index[-1] + 1].get_deriv(
#                             der_indices[j - 1] + der_indices[i - 1]
#                         )
#                     ),
#                 ))
#         if i == 0:
#             K = row_j
#         else:
#             K = np.vstack((K, row_j))

#     return K



def determine_weights(diffs_by_dim, diffs_test, length_scales, kernel_func, sigma_n):
    """
    Vectorized version: compute interpolation weights for multiple test points at once.
    
    Parameters
    ----------
    diffs_by_dim : list of ndarray
        Pairwise differences between training points (by dimension).
    diffs_test : list of ndarray
        Pairwise differences between test points and training points (by dimension).
        Shape: each array is (n_test, n_train) or similar batch dimension.
    length_scales : array-like
        Kernel hyperparameters.
    kernel_func : callable
        Kernel function.
    sigma_n : float
        Noise parameter (if needed).
    
    Returns
    -------
    weights_matrix : ndarray of shape (n_test, n_train)
        Interpolation weights for each test point.
    """
    
    # Compute K matrix (training covariance) - same for all test points
    K = kernel_func(diffs_by_dim, length_scales).real
    n_train = K.shape[0]
    
    # Compute r vectors (test-train covariances) for all test points at once
    r_all = kernel_func(diffs_test, length_scales).real  # shape: (n_test, n_train)
    n_test = r_all.shape[0]
    
    # Build augmented system matrix M (same for all test points)
    M = np.zeros((n_train + 1, n_train + 1))
    M[:n_train, :n_train] = K
    M[:n_train, n_train] = 1
    M[n_train, :n_train] = 1
    M[n_train, n_train] = 0
    
    # Build augmented RHS for all test points
    r_augmented = np.zeros((n_test, n_train + 1))
    r_augmented[:, :n_train] = r_all
    r_augmented[:, n_train] = 1
    
    # Solve for all test points at once: M @ solution.T = r_augmented.T
    # solution.T has shape (n_train+1, n_test)
    solution = np.linalg.solve(M, r_augmented.T)  # shape: (n_train+1, n_test)
    
    # Extract weights (exclude Lagrange multiplier)
    weights_matrix = solution[:n_train, :].T  # shape: (n_test, n_train)
    
    return weights_matrix
