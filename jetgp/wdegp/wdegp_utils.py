import pyoti.core as coti  # Required for the advanced implementation
import numpy as np
import pyoti.sparse as oti
from line_profiler import profile


# @profile
# def differences_by_dim_func(X1, X2, n_order, index=-1):
#     """
#     Compute dimension-wise differences between two sets of points X1 and X2,
#     introducing hypercomplex perturbation for selected indices.

#     Parameters
#     ----------
#     X1 : ndarray of shape (n1, d)
#         First input array.
#     X2 : ndarray of shape (n2, d)
#         Second input array.
#     n_order : int
#         Derivative order used to set the hypercomplex type.
#     index : list or int, optional
#         Indices in X1 or X2 to apply hypercomplex tagging (default is -1, no tagging).

#     Returns
#     -------
#     differences_by_dim : list of ndarray
#         List of length d. Each entry is an (n1, n2) array of differences for a single dimension.
#     """
#     X1 = oti.array(X1)
#     X2 = oti.array(X2)
#     n1, d = X1.shape
#     n2, _ = X2.shape
#     differences_by_dim = []

#     for k in range(d):
#         diffs_k = oti.zeros((n1, n2))
#         for i in range(n1):
#             for j in range(n2):
#                 if i in index or j in index:
#                     diffs_k[i, j] = (
#                         X1[i, k] + oti.e(k + 1, order=2 * n_order)) - X2[j, k]
#                 else:
#                     diffs_k[i, j] = X1[i, k] - X2[j, k]
#         differences_by_dim.append(diffs_k)
#     return differences_by_dim


def differences_by_dim_func(X1, X2, n_order, return_deriv=True):
    """
    Compute pairwise differences between two input arrays X1 and X2 for each dimension,
    embedding hypercomplex units along each dimension for automatic differentiation.

    For each dimension k, this function computes:
        diff_k[i, j] = X1[i, k] + e_{k+1} - X2[j, k]
    where e_{k+1} is a hypercomplex unit for the (k+1)-th dimension with order 2 * n_order.

    Parameters
    ----------
    X1 : array_like of shape (n1, d)
        First set of input points with n1 samples in d dimensions.

    X2 : array_like of shape (n2, d)
        Second set of input points with n2 samples in d dimensions.

    n_order : int
        The base order used to construct hypercomplex units (e_{k+1}) with order 2 * n_order.

    index : int, optional
        Currently unused. Reserved for future enhancements.

    Returns
    -------
    differences_by_dim : list of length d
        A list where each element is an array of shape (n1, n2), containing the differences
        between corresponding dimensions of X1 and X2, augmented with hypercomplex units.

    Notes
    -----
    - The function leverages hypercomplex arithmetic from the pyOTI library.
    - This routine is typically used in the construction of hypercomplex kernels for Gaussian processes
      or other applications involving automatic differentiation.

    Example
    -------
    >>> X1 = [[1.0, 2.0], [3.0, 4.0]]
    >>> X2 = [[1.5, 2.5], [3.5, 4.5]]
    >>> n_order = 1
    >>> diffs = differences_by_dim_func(X1, X2, n_order)
    >>> len(diffs)
    2
    >>> diffs[0].shape
    (2, 2)
    """
    X1 = oti.array(X1)
    X2 = oti.array(X2)
    n1, d = X1.shape

    n2, d = X2.shape

    # Prepare the output: a list of d arrays, each of shape (n, m)
    differences_by_dim = []

    # Loop over each dimension k
    if n_order == 0:
        for k in range(d):
            diffs_k = oti.zeros((n1, n2))
            for i in range(n1):
                diffs_k[i, :] = (
                    X1[i, k]
                    - (X2[:, k].T)
                )
            differences_by_dim.append(diffs_k)
    elif not return_deriv:

        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = oti.zeros((n1, n2))

            # Nested loops to fill diffs_k
            for i in range(n1):
                diffs_k[i, :] = (
                    X1[i, k]
                    + oti.e(k + 1, order=n_order)
                    - (X2[:, k].T)
                )

            # Append to our list
            differences_by_dim.append(diffs_k)
    else:
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = oti.zeros((n1, n2))

            # Nested loops to fill diffs_k
            for i in range(n1):
                diffs_k[i, :] = (
                    X1[i, k]
                    - (X2[:, k].T)
                )

            # Append to our list
            differences_by_dim.append(diffs_k+ oti.e(k + 1, order=2*n_order))
    return differences_by_dim

# @profile
# def rbf_kernel(differences, length_scales, n_order, n_bases, kernel_func, der_indices, powers, index=-1):
#     """
#     Constructs the RBF kernel matrix with derivative entries using hypercomplex representation.

#     Parameters
#     ----------
#     differences : list of ndarrays
#         Differences between training points by dimension.
#     length_scales : array-like
#         Kernel hyperparameters in log10 space.
#     n_order : int
#         Maximum derivative order.
#     n_bases : int
#         Number of OTI basis terms used in the analysis.
#     kernel_func : callable
#         Base kernel function (e.g., SE, RQ).
#     der_indices : list of lists
#         Derivative multi-indices.
#     powers : list of ints
#         Parity powers for each derivative term.
#     index : list of int
#         Indices used for slicing derivative blocks.

#     Returns
#     -------
#     K : ndarray
#         Full RBF kernel matrix with mixed function and derivative entries.
#     """
#     phi = kernel_func(differences, length_scales, index)

#     # Preallocate K

#     for i in range(0, len(der_indices) + 1):
#         row_j = 0
#         for j in range(0, len(der_indices) + 1):
#             if j == 0 and i == 0:
#                 row_j = phi.real * (-1)**(powers[j])
#             elif j > 0 and i == 0:
#                 row_j = np.hstack((
#                     row_j,
#                     (-1)**(powers[j]) * phi[:, index[0]                                            : index[-1] + 1].get_deriv(der_indices[j - 1])
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
#                     )
#                 ))
#         if i == 0:
#             K = row_j
#         else:
#             K = np.vstack((K, row_j))
#     return K


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
        phi,
    phi_exp,
    n_order,
    n_bases,
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
    (Parameters are the same as the original function)

    Returns
    -------
    K : ndarray
        Full RBF kernel matrix with mixed function and derivative entries.
    """
    # --- 1. Initial Setup and Efficient Derivative Extraction ---
    dh = coti.get_dHelp()
    # Create maps to translate derivative specifications to flat indices
    der_map = deriv_map(n_bases, 2 * n_order)
    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)

    # --- 2. Determine Block Sizes and Pre-allocate Matrix ---
    n_rows_func, n_cols_func = phi.shape
    n_deriv_types = len(der_indices)
    n_pts_with_derivs_cols = sum(len(order_indices) for order_indices in index)
    n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index)
    total_rows = n_rows_func + n_pts_with_derivs_rows
    total_cols = n_cols_func + n_pts_with_derivs_cols

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
        
        # Get the indices for this derivative order
        current_indices = index[j]
        
        # Use fancy indexing to select only the columns at the specified indices
        content_sliced = content_full[:, current_indices]
        
        # Number of points with this derivative order
        n_pts_this_order = len(current_indices)
        
        K[:n_rows_func, col_offset: col_offset + n_pts_this_order] = content_sliced * ((-1) ** powers[j + 1])
        col_offset += n_pts_this_order
    # First Block-Column: Derivative-Function (K_df)
    row_offset = n_rows_func
    for i in range(n_deriv_types):
         flat_idx = der_indices_tr[i]
         content_full = phi_exp[flat_idx].reshape(base_shape)
         
         # Get the indices for this derivative order
         current_indices = index[i]
         
         # Use fancy indexing to select only the rows at the specified indices
         content_sliced = content_full[current_indices, :]
         
         # Number of points with this derivative order
         n_pts_this_order = len(current_indices)
         
         K[row_offset: row_offset + n_pts_this_order, :n_cols_func] = content_sliced * ((-1) ** powers[0])
         row_offset += n_pts_this_order

    # Inner Blocks: Derivative-Derivative (K_dd)
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        col_offset = n_cols_func
        
        # Get row indices for this derivative order
        row_indices = index[i]
        n_pts_row = len(row_indices)
        
        for j in range(n_deriv_types):
            # Get column indices for this derivative order
            col_indices = index[j]
            n_pts_col = len(col_indices)
            
            # Multiply the derivative indices to find the correct flat index
            imdir1 = der_ind_order[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(
                imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            flat_idx = der_map[new_ord][new_idx]
            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Slice both rows and columns using non-contiguous indices
            # Use numpy's ix_ for efficient 2D fancy indexing
            content_sliced = content_full[np.ix_(row_indices, col_indices)]
            
            K[row_offset: row_offset + n_pts_row, 
              col_offset: col_offset + n_pts_col] = content_sliced * ((-1) ** powers[j + 1])
            
            col_offset += n_pts_col
        
        row_offset += n_pts_row

    return K

@profile
def rbf_kernel_predictions(
        phi,
    phi_exp,
    n_order,
    n_bases,
    der_indices,
    powers,
    return_deriv,
    index=-1,
    common_derivs = None,
    calc_cov = False,
    powers_predict = None
):
    """
    Constructs the RBF kernel matrix with derivative entries using an
    efficient pre-allocation strategy combined with a single call to
    extract all derivative components.

    Parameters
    ----------
    (Parameters are the same as the original function)

    Returns
    -------
    K : ndarray
        Full RBF kernel matrix with mixed function and derivative entries.
    """
    # --- 1. Initial Setup and Efficient Derivative Extraction ---
    if calc_cov and not return_deriv:
        return phi.real
    dh = coti.get_dHelp()
    # Create maps to translate derivative specifications to flat indices
    # --- 2. Determine Block Sizes and Pre-allocate Matrix ---
    n_rows_func, n_cols_func = phi.shape
    n_deriv_types = len(der_indices)
    n_deriv_types_pred = len(common_derivs)
    if return_deriv:
        der_map = deriv_map(n_bases, 2 * n_order)
        index_2 = [i for i in range(phi_exp.shape[-1])]
        if calc_cov:
            index = [i for i in range(phi_exp.shape[-1])]
            n_deriv_types = n_deriv_types_pred
            n_pts_with_derivs_rows = n_deriv_types * len([i for i in range(n_cols_func) if i in index_2])
        else:
            n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index)
    else:
        der_map = deriv_map(n_bases, n_order)
        index_2 = []
        n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index)

    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)
    der_indices_tr_pred, der_ind_order_pred = transform_der_indices(common_derivs, der_map)
    n_pts_with_derivs_cols = n_deriv_types_pred * len([i for i in range(n_cols_func) if i in index_2])

    total_rows = n_rows_func + n_pts_with_derivs_rows 
    total_cols = n_cols_func + n_pts_with_derivs_cols 

    K = np.zeros((total_rows, total_cols))

    # The full content blocks are always the size of the original phi matrix
    base_shape = (n_rows_func, n_cols_func)

    # --- 3. Fill the Matrix Block by Block ---

    # Block (0,0): Function-Function (K_ff)
    # The real part is always at index 0 of the flat array
    content_full = phi_exp[0].reshape(base_shape)
    K[:n_rows_func, :n_cols_func] = content_full * ((-1) ** powers[0])
    
    if not return_deriv:
        # First Block-Column: Derivative-Function (K_df)
        row_offset = n_rows_func
        for i in range(n_deriv_types):
            # Get row indices for this derivative order

            row_indices = index[i]
            n_pts_row = len(row_indices)
            
            flat_idx = der_indices_tr[i]
            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Slice rows using non-contiguous indices
            content_sliced = content_full[row_indices, :]
            
            K[row_offset: row_offset + n_pts_row, :n_cols_func] = content_sliced * ((-1) ** powers[0])
            row_offset += n_pts_row
        return K
    else:
        # First Block-Row: Function-Derivative (K_fd)
        col_offset = n_cols_func
        for j in range(n_deriv_types_pred):
            # Get column indices for this derivative order
            col_indices = index_2
            n_pts_col = len(col_indices)
            
            flat_idx = der_indices_tr_pred[j]
            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Slice columns using non-contiguous indices
            content_sliced = content_full[:, col_indices]
    
            K[:n_rows_func, col_offset: col_offset + n_pts_col] = content_sliced * ((-1) ** powers_predict[j + 1])
            col_offset += n_pts_col
    
        # First Block-Column: Derivative-Function (K_df)
        row_offset = n_rows_func
        for i in range(n_deriv_types):
            # Get row indices for this derivative order
            if calc_cov:
                row_indices = index
                flat_idx = der_indices_tr_pred[i]
            else:
                row_indices =index[i]
                flat_idx = der_indices_tr[i]
            n_pts_row = len(row_indices)
            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Slice rows using non-contiguous indices
            content_sliced = content_full[row_indices, :]
    
            K[row_offset: row_offset + n_pts_row, :n_cols_func] = content_sliced * ((-1) ** powers[0])
            row_offset += n_pts_row
    
        # Inner Blocks: Derivative-Derivative (K_dd)
        row_offset = n_rows_func
        for i in range(n_deriv_types):
            # Get row indices for this derivative order
            if calc_cov:
                row_indices = index
            else:
                row_indices =index[i]
            n_pts_row = len(row_indices)
            
            col_offset = n_cols_func
            for j in range(n_deriv_types_pred):
                # Get column indices for this derivative order
                col_indices = index_2
                n_pts_col = len(col_indices)
                
                # Multiply the derivative indices to find the correct flat index
                imdir1 = der_ind_order_pred[j]
                imdir2 = der_ind_order[i]
                new_idx, new_ord = dh.mult_dir(
                    imdir1[0], imdir1[1], imdir2[0], imdir2[1])
                flat_idx = der_map[new_ord][new_idx]
    
                content_full = phi_exp[flat_idx].reshape(base_shape)
                
                # Slice both rows and columns using non-contiguous indices
                content_sliced = content_full[np.ix_(row_indices, col_indices)]
    
                K[row_offset: row_offset + n_pts_row, 
                  col_offset: col_offset + n_pts_col] = content_sliced * ((-1) ** powers_predict[j + 1])
                col_offset += n_pts_col
            row_offset += n_pts_row
    
        return K

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

def to_tuple(item):
    if isinstance(item, list):
        return tuple(to_tuple(x) for x in item)
    return item

def to_list(x):
    if isinstance(x, tuple):
        return [to_list(i) for i in x]
    return x

def find_common_derivatives(all_indices):
    """Find derivative indices common to all submodels."""
    sets = [set(to_tuple(elem) for elem in idx_list) for idx_list in all_indices]
    return sets[0].intersection(*sets[1:])

