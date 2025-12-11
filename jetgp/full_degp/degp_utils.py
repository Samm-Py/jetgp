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
    return_deriv : bool, optional
        If True, use 2*n_order for derivative predictions.

    Returns
    -------
    differences_by_dim : list of length d
        A list where each element is an array of shape (n1, n2), containing the differences
        between corresponding dimensions of X1 and X2, augmented with hypercomplex units.
    """
    X1 = oti.array(X1)
    X2 = oti.array(X2)
    n1, d = X1.shape
    n2, d = X2.shape

    differences_by_dim = []

    if n_order == 0:
        for k in range(d):
            diffs_k = oti.zeros((n1, n2))
            for i in range(n1):
                diffs_k[i, :] = X1[i, k] - (X2[:, k].T)
            differences_by_dim.append(diffs_k)
    elif not return_deriv:
        for k in range(d):
            diffs_k = oti.zeros((n1, n2))
            for i in range(n1):
                diffs_k[i, :] = (
                    X1[i, k]
                    + oti.e(k + 1, order=n_order)
                    - (X2[:, k].T)
                )
            differences_by_dim.append(diffs_k)
    else:
        for k in range(d):
            diffs_k = oti.zeros((n1, n2))
            for i in range(n1):
                diffs_k[i, :] = X1[i, k] - (X2[:, k].T)
            differences_by_dim.append(diffs_k + oti.e(k + 1, order=2 * n_order))

    return differences_by_dim


# =============================================================================
# Derivative mapping utilities
# =============================================================================

def deriv_map(nbases, order):
    """
    Create mapping from (order, index) to flattened index.
    
    Parameters
    ----------
    nbases : int
        Number of base dimensions.
    order : int
        Maximum derivative order.
        
    Returns
    -------
    map_deriv : list of lists
        Mapping where map_deriv[order][idx] gives the flattened index.
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
    Transform derivative indices to flattened format.
    
    Parameters
    ----------
    der_indices : list
        User-facing derivative specifications.
    der_map : list of lists
        Derivative mapping from deriv_map().
        
    Returns
    -------
    deriv_ind_transf : list
        Flattened indices for each derivative.
    deriv_ind_order : list
        (index, order) tuples for each derivative.
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

@profile
def rbf_kernel(
    phi,
    phi_exp,
    n_order,
    n_bases,
    der_indices,
    powers,
    index=None
):
    """
    Compute the derivative-enhanced RBF kernel matrix (optimized version).
    
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
        Number of input dimensions.
    der_indices : list of lists
        Multi-index derivative structures for each derivative component.
    powers : list of int
        Powers of (-1) applied to each term.
    index : list of lists or None, optional
        If empty list, assumes uniform blocks.
        If provided, specifies which training point indices have each derivative type.
        
    Returns
    -------
    K : ndarray
        Kernel matrix including function values and derivative terms.
    """
    dh = coti.get_dHelp()

    n_rows_func, n_cols_func = phi.shape
    n_deriv_types = len(der_indices)

    der_map = deriv_map(n_bases, 2 * n_order)
    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)

    # Pre-compute signs (avoid repeated exponentiation)
    signs = np.array([(-1.0) ** p for p in powers], dtype=np.float64)

    # =========================================================================
    # CASE 1: Uniform blocks (original behavior) - index is None or empty
    # =========================================================================
    if index is None or len(index) == 0:
        K = np.zeros((n_rows_func * (n_deriv_types + 1), n_cols_func * (n_deriv_types + 1)))
        outer_loop_index = n_deriv_types + 1

        for j in range(outer_loop_index):
            signj = signs[j]
            for i in range(n_deriv_types + 1):
                Klocal = K[i * n_rows_func: (i + 1) * n_rows_func,
                           j * n_cols_func: (j + 1) * n_cols_func]
                if j == 0 and i == 0:
                    Klocal[:, :] = phi_exp[0] * signj

        return K

    # =========================================================================
    # CASE 2: Non-contiguous indices - index is provided
    # =========================================================================
    n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index)
    total_rows = n_rows_func + n_pts_with_derivs_rows
    n_pts_with_derivs_cols = sum(len(order_indices) for order_indices in index)
    total_cols = n_cols_func + n_pts_with_derivs_cols

    K = np.zeros((total_rows, total_cols))
    base_shape = (n_rows_func, n_cols_func)

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
        row_indices = index_arrays[i]
        n_pts_this_order = len(row_indices)
        
        # Use numba for efficient row extraction and assignment
        extract_rows_and_assign(content_full, row_indices, K,
                                row_offset, 0, n_cols_func, signs[0])
        row_offset += n_pts_this_order

    # First Block-Row: Function-Derivative (K_fd)
    col_offset = n_cols_func
    for j in range(n_deriv_types):
        flat_idx = der_indices_tr[j]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        col_indices = index_arrays[j]
        n_pts_this_order = len(col_indices)
        
        # Use numba for efficient column extraction and assignment
        extract_cols_and_assign(content_full, col_indices, K,
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

            imdir1 = der_ind_order[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            flat_idx = der_map[new_ord][new_idx]
            content_full = phi_exp[flat_idx].reshape(base_shape)

            # Use numba for direct extraction and assignment (replaces np.ix_)
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
    index=None,
    common_derivs=None,
    calc_cov=False,
    powers_predict=None
):
    """
    Constructs the RBF kernel matrix for predictions with derivative entries.
    
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
        Derivative specifications for training data.
    powers : list of int
        Sign powers for each derivative type.
    return_deriv : bool
        If True, predict derivatives at ALL test points.
    index : list of lists or None
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
    # Early return for covariance-only case
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

    # Determine derivative map and index structures
    if return_deriv:
        der_map = deriv_map(n_bases, 2 * n_order)
        index_2 = np.arange(n_cols_func, dtype=np.int64)
        if calc_cov:
            index_cov = np.arange(n_cols_func, dtype=np.int64)
            n_deriv_types = n_deriv_types_pred
            n_pts_with_derivs_rows = n_deriv_types * n_cols_func
        else:
            n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index) if index else 0
    else:
        der_map = deriv_map(n_bases, n_order)
        index_2 = np.array([], dtype=np.int64)
        n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index) if index else 0

    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)
    der_indices_tr_pred, der_ind_order_pred = transform_der_indices(common_derivs, der_map) if common_derivs else ([], [])
    n_pts_with_derivs_cols = n_deriv_types_pred * len(index_2)

    total_rows = n_rows_func + n_pts_with_derivs_rows
    total_cols = n_cols_func + n_pts_with_derivs_cols

    K = np.zeros((total_rows, total_cols))
    base_shape = (n_rows_func, n_cols_func)

    # Convert index lists to numpy arrays for numba
    if index is not None and len(index) > 0 and isinstance(index[0], (list, np.ndarray)):
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
        n_pts_col = len(index_2)

        flat_idx = der_indices_tr_pred[j]
        content_full = phi_exp[flat_idx].reshape(base_shape)

        # Use numba for efficient column extraction
        extract_cols_and_assign(content_full, index_2, K,
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
            n_pts_col = len(index_2)

            imdir1 = der_ind_order_pred[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            flat_idx = der_map[new_ord][new_idx]

            content_full = phi_exp[flat_idx].reshape(base_shape)

            # Use numba for efficient submatrix extraction and assignment (replaces np.ix_)
            extract_and_assign(content_full, row_indices, index_2, K,
                              row_offset, col_offset, signs_predict[j + 1])
            col_offset += n_pts_col
        row_offset += n_pts_row

    return K