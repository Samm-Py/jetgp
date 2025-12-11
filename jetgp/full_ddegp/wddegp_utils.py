import numpy as np
import pyoti.sparse as oti
import pyoti.core as coti
from line_profiler import profile
import numba


# =============================================================================
# Numba-accelerated helper functions for efficient matrix slicing
# =============================================================================

@numba.jit(nopython=True, cache=True)
def extract_rows(content_full, row_indices, n_cols):
    """
    Extract rows from content_full at specified indices.
    
    Parameters
    ----------
    content_full : ndarray of shape (n_rows_full, n_cols)
        Source matrix.
    row_indices : ndarray of int64
        Row indices to extract.
    n_cols : int
        Number of columns.
        
    Returns
    -------
    result : ndarray of shape (len(row_indices), n_cols)
        Extracted rows.
    """
    n_rows = len(row_indices)
    result = np.empty((n_rows, n_cols))
    for i in range(n_rows):
        ri = row_indices[i]
        for j in range(n_cols):
            result[i, j] = content_full[ri, j]
    return result


@numba.jit(nopython=True, cache=True)
def extract_cols(content_full, col_indices, n_rows):
    """
    Extract columns from content_full at specified indices.
    
    Parameters
    ----------
    content_full : ndarray of shape (n_rows, n_cols_full)
        Source matrix.
    col_indices : ndarray of int64
        Column indices to extract.
    n_rows : int
        Number of rows.
        
    Returns
    -------
    result : ndarray of shape (n_rows, len(col_indices))
        Extracted columns.
    """
    n_cols = len(col_indices)
    result = np.empty((n_rows, n_cols))
    for i in range(n_rows):
        for j in range(n_cols):
            result[i, j] = content_full[i, col_indices[j]]
    return result


@numba.jit(nopython=True, cache=True)
def extract_submatrix(content_full, row_indices, col_indices):
    """
    Extract submatrix from content_full at specified row and column indices.
    Replaces the expensive np.ix_ operation.
    
    Parameters
    ----------
    content_full : ndarray of shape (n_rows_full, n_cols_full)
        Source matrix.
    row_indices : ndarray of int64
        Row indices to extract.
    col_indices : ndarray of int64
        Column indices to extract.
        
    Returns
    -------
    result : ndarray of shape (len(row_indices), len(col_indices))
        Extracted submatrix.
    """
    n_rows = len(row_indices)
    n_cols = len(col_indices)
    result = np.empty((n_rows, n_cols))
    for i in range(n_rows):
        ri = row_indices[i]
        for j in range(n_cols):
            result[i, j] = content_full[ri, col_indices[j]]
    return result


@numba.jit(nopython=True, cache=True, parallel=False)
def extract_and_assign(content_full, row_indices, col_indices, K, 
                       row_start, col_start, sign):
    """
    Extract submatrix and assign directly to K with sign multiplication.
    Combines extraction and assignment in one pass for better performance.
    
    Parameters
    ----------
    content_full : ndarray of shape (n_rows_full, n_cols_full)
        Source matrix.
    row_indices : ndarray of int64
        Row indices to extract.
    col_indices : ndarray of int64
        Column indices to extract.
    K : ndarray
        Target matrix to fill.
    row_start : int
        Starting row index in K.
    col_start : int
        Starting column index in K.
    sign : float
        Sign multiplier (+1.0 or -1.0).
    """
    n_rows = len(row_indices)
    n_cols = len(col_indices)
    for i in range(n_rows):
        ri = row_indices[i]
        for j in range(n_cols):
            K[row_start + i, col_start + j] = content_full[ri, col_indices[j]] * sign


@numba.jit(nopython=True, cache=True)
def extract_rows_and_assign(content_full, row_indices, K, 
                            row_start, col_start, n_cols, sign):
    """
    Extract rows and assign directly to K with sign multiplication.
    
    Parameters
    ----------
    content_full : ndarray of shape (n_rows_full, n_cols)
        Source matrix.
    row_indices : ndarray of int64
        Row indices to extract.
    K : ndarray
        Target matrix to fill.
    row_start : int
        Starting row index in K.
    col_start : int
        Starting column index in K.
    n_cols : int
        Number of columns to copy.
    sign : float
        Sign multiplier (+1.0 or -1.0).
    """
    n_rows = len(row_indices)
    for i in range(n_rows):
        ri = row_indices[i]
        for j in range(n_cols):
            K[row_start + i, col_start + j] = content_full[ri, j] * sign


@numba.jit(nopython=True, cache=True)
def extract_cols_and_assign(content_full, col_indices, K, 
                            row_start, col_start, n_rows, sign):
    """
    Extract columns and assign directly to K with sign multiplication.
    
    Parameters
    ----------
    content_full : ndarray of shape (n_rows, n_cols_full)
        Source matrix.
    col_indices : ndarray of int64
        Column indices to extract.
    K : ndarray
        Target matrix to fill.
    row_start : int
        Starting row index in K.
    col_start : int
        Starting column index in K.
    n_rows : int
        Number of rows to copy.
    sign : float
        Sign multiplier (+1.0 or -1.0).
    """
    n_cols = len(col_indices)
    for i in range(n_rows):
        for j in range(n_cols):
            K[row_start + i, col_start + j] = content_full[i, col_indices[j]] * sign


# =============================================================================
# Difference computation functions
# =============================================================================

def differences_by_dim_func(X1, X2, rays, n_order, return_deriv=True):
    """
    Compute dimension-wise pairwise differences between X1 and X2,
    including hypercomplex perturbations in the directions specified by `rays`.
    
    This optimized version pre-calculates the perturbation and uses a single
    efficient loop for subtraction, avoiding broadcasting issues with OTI arrays.
    
    Parameters
    ----------
    X1 : ndarray of shape (n1, d)
        First set of input points with n1 samples in d dimensions.
    X2 : ndarray of shape (n2, d)
        Second set of input points with n2 samples in d dimensions.
    rays : ndarray of shape (d, n_rays)
        Directional vectors for derivative computation.
    n_order : int
        The base order used to construct hypercomplex units.
        When return_deriv=True, uses order 2*n_order.
        When return_deriv=False, uses order n_order.
    return_deriv : bool, optional (default=True)
        If True, use order 2*n_order for hypercomplex units (needed for 
        derivative-derivative blocks in training kernel).
        If False, use order n_order (sufficient for prediction without 
        derivative outputs).
        
    Returns
    -------
    differences_by_dim : list of length d
        A list where each element is an array of shape (n1, n2), containing 
        the differences between corresponding dimensions of X1 and X2, 
        augmented with directional hypercomplex perturbations.
    """
    X1 = oti.array(X1)
    X2 = oti.array(X2)
    n1, d = X1.shape
    n2, _ = X2.shape
    n_rays = rays.shape[1]
    
    differences_by_dim = []
    
    # Case 1: n_order == 0 (no hypercomplex perturbation)
    if n_order == 0:
        for k in range(d):
            diffs_k = oti.zeros((n1, n2))
            for i in range(n1):
                diffs_k[i, :] = X1[i, k] - X2[:, k].T
            differences_by_dim.append(diffs_k)
        return differences_by_dim
    
    # Determine the order for hypercomplex units based on return_deriv
    if return_deriv:
        hc_order = 2 * n_order
    else:
        hc_order = n_order
    
    # Pre-calculate the perturbation vector using directional rays
    e_bases = [oti.e(i + 1, order=hc_order) for i in range(n_rays)]
    perts = np.dot(rays, e_bases)
    
    # Case 2: return_deriv=False (prediction without derivative outputs)
    if not return_deriv:
        for k in range(d):
            # Add the pre-calculated perturbation for the current dimension to all points in X1
            X1_k_tagged = X1[:, k] + perts[k]
            X2_k = X2[:, k]
            
            # Pre-allocate the result matrix for this dimension
            diffs_k = oti.zeros((n1, n2))
            
            # Use an efficient single loop for subtraction
            for i in range(n1):
                diffs_k[i, :] = X1_k_tagged[i, 0] - X2_k[:, 0].T
            
            differences_by_dim.append(diffs_k)
    
    # Case 3: return_deriv=True (training kernel with derivative-derivative blocks)
    else:
        for k in range(d):
            X2_k = X2[:, k]
            
            # Pre-allocate the result matrix for this dimension
            diffs_k = oti.zeros((n1, n2))
            
            # Compute differences without perturbation first
            for i in range(n1):
                diffs_k[i, :] = X1[i, k] - X2_k[:, 0].T
            
            # Add perturbation to the entire matrix (more efficient)
            differences_by_dim.append(diffs_k + perts[k])
    
    return differences_by_dim


# =============================================================================
# Derivative mapping utilities
# =============================================================================

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


# =============================================================================
# RBF Kernel Assembly Functions (Optimized with Numba)
# =============================================================================

def rbf_kernel(
    phi,
    phi_exp,
    n_order,
    n_bases,
    der_indices,
    powers,
    index=-1
):
    """
    Assembles the full DD-GP covariance matrix using an efficient, pre-computed
    derivative array and block-wise matrix filling.
    
    This version uses Numba-accelerated functions for efficient matrix slicing,
    replacing expensive np.ix_ operations.
    
    Parameters
    ----------
    phi : OTI array
        Base kernel matrix from kernel_func(differences, length_scales).
    phi_exp : ndarray
        Expanded derivative array from phi.get_all_derivs().
    n_order : int
        Maximum derivative order considered.
    n_bases : int
        Total number of bases (function value + derivative terms).
    der_indices : list of lists
        Multi-index derivative structures for each derivative component.
    powers : list of int
        Powers of (-1) applied to each term (for symmetry or sign conventions).
    index : list of lists
        Specifies which training point indices have each derivative type.
        
    Returns
    -------
    K : ndarray
        Full kernel matrix with function values and derivative blocks.
    """
    dh = coti.get_dHelp()
    
    # Create maps to translate derivative specifications to flat indices
    der_map = deriv_map(n_bases, 2 * n_order)
    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)

    # Determine Block Sizes and Pre-allocate Matrix
    n_rows_func, n_cols_func = phi.shape
    n_deriv_types = len(der_indices)
    n_pts_with_derivs_cols = sum(len(order_indices) for order_indices in index)
    n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index)
    total_rows = n_rows_func + n_pts_with_derivs_rows
    total_cols = n_cols_func + n_pts_with_derivs_cols

    K = np.zeros((total_rows, total_cols))
    base_shape = (n_rows_func, n_cols_func)
    
    # Pre-compute signs (avoid repeated exponentiation)
    signs = np.array([(-1.0) ** p for p in powers], dtype=np.float64)
    
    # Convert index lists to numpy arrays for numba
    index_arrays = [np.asarray(idx, dtype=np.int64) for idx in index]
    
    # Block (0,0): Function-Function (K_ff)
    content_full = phi_exp[0].reshape(base_shape)
    K[:n_rows_func, :n_cols_func] = content_full * signs[0]
    
    # First Block-Column: Derivative-Function (K_df)
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        flat_idx = der_indices_tr[i]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        
        current_indices = index_arrays[i]
        n_pts_this_order = len(current_indices)
        
        # Use numba for efficient row extraction and assignment
        extract_rows_and_assign(content_full, current_indices, K,
                                row_offset, 0, n_cols_func, signs[0])
        row_offset += n_pts_this_order
    
    # First Block-Row: Function-Derivative (K_fd)
    col_offset = n_cols_func
    for j in range(n_deriv_types):
        flat_idx = der_indices_tr[j]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        
        current_indices = index_arrays[j]
        n_pts_this_order = len(current_indices)
        
        # Use numba for efficient column extraction and assignment
        extract_cols_and_assign(content_full, current_indices, K,
                                0, col_offset, n_rows_func, signs[j + 1])
        col_offset += n_pts_this_order
    
    # Inner Blocks: Derivative-Derivative (K_dd)
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        col_offset = n_cols_func
        
        row_indices = index_arrays[i]
        n_pts_row = len(row_indices)
        
        for j in range(n_deriv_types):
            col_indices = index_arrays[j]
            n_pts_col = len(col_indices)
            
            # Multiply derivative indices to find correct flat index
            imdir1 = der_ind_order[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            flat_idx = der_map[new_ord][new_idx]
            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Use numba for efficient submatrix extraction and assignment (replaces np.ix_)
            extract_and_assign(content_full, row_indices, col_indices, K,
                               row_offset, col_offset, signs[j + 1])
            
            col_offset += n_pts_col
        
        row_offset += n_pts_row
    
    return K


def rbf_kernel_predictions(
    phi,
    phi_exp,
    n_order,
    n_bases,
    der_indices,
    powers,
    return_deriv,
    index=-1,
    common_derivs=None,
    calc_cov=False,
    powers_predict=None
):
    """
    Constructs the RBF kernel matrix for predictions with directional derivative entries.
    
    This version uses Numba-accelerated functions for efficient matrix slicing.

    Parameters
    ----------
    phi : OTI array
        Base kernel matrix between test and training points.
    phi_exp : ndarray
        Expanded derivative array from phi.get_all_derivs().
    n_order : int
        Maximum derivative order.
    n_bases : int
        Number of input dimensions.
    der_indices : list
        Derivative specifications.
    powers : list of int
        Sign powers for each derivative type.
    return_deriv : bool
        If True, predict derivatives at ALL test points.
    index : list of lists
        Training point indices for each derivative type.
    common_derivs : list
        Common derivative indices to predict.
    calc_cov : bool
        If True, computing covariance (use all indices for rows).
    powers_predict : list of int, optional
        Sign powers for prediction derivatives.

    Returns
    -------
    K : ndarray
        Prediction kernel matrix.
    """
    if calc_cov and not return_deriv:
        return phi.real
    
    dh = coti.get_dHelp()
    
    n_rows_func, n_cols_func = phi.shape
    n_deriv_types = len(der_indices)
    n_deriv_types_pred = len(common_derivs) if common_derivs else 0
    
    # Pre-compute signs
    signs = np.array([(-1.0) ** p for p in powers], dtype=np.float64)
    if powers_predict is not None:
        signs_predict = np.array([(-1.0) ** p for p in powers_predict], dtype=np.float64)
    else:
        signs_predict = signs
    
    if return_deriv:
        der_map = deriv_map(n_bases, 2 * n_order)
        index_2 = np.arange(phi_exp.shape[-1], dtype=np.int64)
        if calc_cov:
            index_cov = np.arange(phi_exp.shape[-1], dtype=np.int64)
            n_deriv_types = n_deriv_types_pred
            n_pts_with_derivs_rows = n_deriv_types * len([i for i in range(n_cols_func) if i < len(index_2)])
        else:
            n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index)
    else:
        der_map = deriv_map(n_bases, n_order)
        index_2 = np.array([], dtype=np.int64)
        n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index)

    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)
    der_indices_tr_pred, der_ind_order_pred = transform_der_indices(common_derivs, der_map) if common_derivs else ([], [])
    n_pts_with_derivs_cols = n_deriv_types_pred * len([i for i in range(n_cols_func) if i < len(index_2)])

    total_rows = n_rows_func + n_pts_with_derivs_rows 
    total_cols = n_cols_func + n_pts_with_derivs_cols 

    K = np.zeros((total_rows, total_cols))
    base_shape = (n_rows_func, n_cols_func)

    # Convert index lists to numpy arrays for numba
    if index != -1 and isinstance(index, list) and len(index) > 0:
        index_arrays = [np.asarray(idx, dtype=np.int64) for idx in index]
    else:
        index_arrays = []

    # Block (0,0): Function-Function (K_ff)
    content_full = phi_exp[0].reshape(base_shape)
    K[:n_rows_func, :n_cols_func] = content_full * signs[0]
    
    if not return_deriv:
        # First Block-Column: Derivative-Function (K_df)
        row_offset = n_rows_func
        for i in range(n_deriv_types):
            if calc_cov:
                row_indices = index_cov
            else:
                if not index_arrays:
                    break
                row_indices = index_arrays[i]
            n_pts_row = len(row_indices)
            
            flat_idx = der_indices_tr[i]
            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Use numba for efficient row extraction
            extract_rows_and_assign(content_full, row_indices, K,
                                    row_offset, 0, n_cols_func, signs[0])
            row_offset += n_pts_row
        return K
    
    # --- return_deriv=True case ---
    
    # First Block-Row: Function-Derivative (K_fd)
    col_offset = n_cols_func
    for j in range(n_deriv_types_pred):
        col_indices = index_2
        n_pts_col = len(col_indices)
        
        flat_idx = der_indices_tr_pred[j]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        
        # Use numba for efficient column extraction
        extract_cols_and_assign(content_full, col_indices, K,
                                0, col_offset, n_rows_func, signs_predict[j + 1])
        col_offset += n_pts_col

    # First Block-Column: Derivative-Function (K_df)
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        if calc_cov:
            row_indices = index_cov
            flat_idx = der_indices_tr_pred[i]
        else:
            if not index_arrays:
                break
            row_indices = index_arrays[i]
            flat_idx = der_indices_tr[i]
        n_pts_row = len(row_indices)
        
        content_full = phi_exp[flat_idx].reshape(base_shape)
        
        # Use numba for efficient row extraction
        extract_rows_and_assign(content_full, row_indices, K,
                                row_offset, 0, n_cols_func, signs[0])
        row_offset += n_pts_row

    # Inner Blocks: Derivative-Derivative (K_dd)
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        if calc_cov:
            row_indices = index_cov
        else:
            if not index_arrays:
                break
            row_indices = index_arrays[i]
        n_pts_row = len(row_indices)
        
        col_offset = n_cols_func
        for j in range(n_deriv_types_pred):
            col_indices = index_2
            n_pts_col = len(col_indices)
            
            # Multiply the derivative indices to find the correct flat index
            imdir1 = der_ind_order_pred[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(
                imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            flat_idx = der_map[new_ord][new_idx]

            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Use numba for efficient submatrix extraction and assignment (replaces np.ix_)
            extract_and_assign(content_full, row_indices, col_indices, K,
                               row_offset, col_offset, signs_predict[j + 1])
            col_offset += n_pts_col
        row_offset += n_pts_row

    return K


# =============================================================================
# Utility functions
# =============================================================================

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
    r_all = kernel_func(diffs_test, length_scales).real
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
    
    # Solve for all test points at once
    solution = np.linalg.solve(M, r_augmented.T)
    
    # Extract weights (exclude Lagrange multiplier)
    weights_matrix = solution[:n_train, :].T
    
    return weights_matrix


def to_list(x):
    """Convert tuple to list recursively."""
    if isinstance(x, tuple):
        return [to_list(i) for i in x]
    return x


def to_tuple(item):
    """Convert list to tuple recursively."""
    if isinstance(item, list):
        return tuple(to_tuple(x) for x in item)
    return item


def find_common_derivatives(all_indices):
    """Find derivative indices common to all submodels."""
    sets = [set(to_tuple(elem) for elem in idx_list) for idx_list in all_indices]
    return sets[0].intersection(*sets[1:])