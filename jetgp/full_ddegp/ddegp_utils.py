import numpy as np
import numba
import pyoti.core as coti
from line_profiler import profile


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

def differences_by_dim_func(X1, X2, rays, n_order, oti_module, return_deriv=True, index=-1):
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
    oti_module : module
        The PyOTI static module (e.g., pyoti.static.onumm4n2).
    return_deriv : bool, optional (default=True)
        If True, use order 2*n_order for hypercomplex units (needed for 
        derivative-derivative blocks in training kernel).
        If False, use order n_order (sufficient for prediction without 
        derivative outputs).
    index : int, optional
        Currently unused. Reserved for future enhancements.
        
    Returns
    -------
    differences_by_dim : list of length d
        A list where each element is an array of shape (n1, n2), containing 
        the differences between corresponding dimensions of X1 and X2, 
        augmented with directional hypercomplex perturbations.
        
    Notes
    -----
    - The function leverages hypercomplex arithmetic from the pyOTI library.
    - The directional perturbation is computed as: perts = rays @ e_bases
      where e_bases are the hypercomplex units for each ray direction.
    - This routine is typically used in the construction of directional 
      derivative kernels for Gaussian processes.
      
    Example
    -------
    >>> X1 = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> X2 = np.array([[1.5, 2.5], [3.5, 4.5]])
    >>> rays = np.eye(2)  # Standard basis directions
    >>> n_order = 1
    >>> oti_module = get_oti_module(2, 1)  # dim=2, n_order=1
    >>> diffs = differences_by_dim_func(X1, X2, rays, n_order, oti_module)
    >>> len(diffs)
    2
    >>> diffs[0].shape
    (2, 2)
    """
    X1 = oti_module.array(X1)
    X2 = oti_module.array(X2)
    n1, d = X1.shape
    n2, _ = X2.shape
    n_rays = rays.shape[1]
    
    differences_by_dim = []
    
    # Case 1: n_order == 0 (no hypercomplex perturbation)
    if n_order == 0:
        for k in range(d):
            diffs_k = oti_module.zeros((n1, n2))
            for i in range(n1):
                diffs_k[i, :] = X1[i, k] - oti_module.transpose(X2[:, k])
            differences_by_dim.append(diffs_k)
        return differences_by_dim
    
    # Determine the order for hypercomplex units based on return_deriv
    if return_deriv:
        hc_order = 2 * n_order
    else:
        hc_order = n_order
    
    # Pre-calculate the perturbation vector using directional rays
    e_bases = [oti_module.e(i + 1, order=hc_order) for i in range(n_rays)]
    perts = np.dot(rays, e_bases)
    
    # Case 2: return_deriv=False (prediction without derivative outputs)
    if not return_deriv:
        for k in range(d):
            # Add the pre-calculated perturbation for the current dimension to all points in X1
            X1_k_tagged = X1[:, k] + perts[k]
            X2_k = X2[:, k]
            
            # Pre-allocate the result matrix for this dimension
            diffs_k = oti_module.zeros((n1, n2))
            
            # Use an efficient single loop for subtraction
            for i in range(n1):
                diffs_k[i, :] = X1_k_tagged[i, 0] - X2_k[:, 0].T
            
            differences_by_dim.append(diffs_k)
    
    # Case 3: return_deriv=True (training kernel with derivative-derivative blocks)
    else:
        for k in range(d):
            X2_k = X2[:, k]
            
            # Pre-allocate the result matrix for this dimension
            diffs_k = oti_module.zeros((n1, n2))
            
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
    Transforms a list of user-facing derivative specifications into the
    internal (order, index) format and the final flattened index.
    
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
    index=-1
):
    """
    Assembles the full DD-GP covariance matrix using an efficient, pre-computed
    derivative array and block-wise matrix filling.
    
    Supports both uniform blocks (all derivatives at all points) and non-contiguous
    indices (different derivatives at different subsets of points).
    
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
        Number of input dimensions (rays).
    der_indices : list of lists
        Multi-index derivative structures for each derivative component.
    powers : list of int
        Powers of (-1) applied to each term (for symmetry or sign conventions).
    index : list of lists or int, optional (default=-1)
        If empty list, assumes all derivative types apply to all training points.
        If provided, specifies which training point indices have each derivative type,
        allowing non-contiguous index support and variable block sizes.
        
    Returns
    -------
    K : ndarray
        Full kernel matrix with function values and derivative blocks.
    """
    # --- 1. Initial Setup and Efficient Derivative Extraction ---
    dh = coti.get_dHelp()
    
    # Create maps to translate derivative specifications to flat indices
    der_map = deriv_map(n_bases, 2 * n_order)
    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)

    # --- 2. Determine Block Sizes and Pre-allocate Matrix ---
    n_rows_func, n_cols_func = phi.shape
    n_deriv_types = len(der_indices)
    
    # Pre-compute signs (avoid repeated exponentiation)
    signs = np.array([(-1.0) ** p for p in powers], dtype=np.float64)
    
    # Convert index lists to numpy arrays for numba (if provided)
    if isinstance(index, list) and len(index) > 0:
        index_arrays = [np.asarray(idx, dtype=np.int64) for idx in index]
    else:
        index_arrays = []
    
    n_pts_with_derivs_cols = sum(len(idx) for idx in index_arrays) if index_arrays else 0
    n_pts_with_derivs_rows = n_pts_with_derivs_cols
    total_rows = n_rows_func + n_pts_with_derivs_rows
    total_cols = n_cols_func + n_pts_with_derivs_cols

    K = np.zeros((total_rows, total_cols))
    base_shape = (n_rows_func, n_cols_func)
    
    # --- 3. Fill the Matrix Block by Block ---
    
    # Block (0,0): Function-Function (K_ff)
    content_full = phi_exp[0].reshape(base_shape)
    K[:n_rows_func, :n_cols_func] = content_full * signs[0]
    
    if not index_arrays:
        # No derivative indices provided, return early
        return K
    
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
            
            # Multiply derivative indices to find correct flat index
            imdir1 = der_ind_order[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            flat_idx = der_map[new_ord][new_idx]
            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Use numba for efficient submatrix extraction and assignment
            # This replaces the expensive np.ix_ operation
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
    
    This handles the asymmetric case where:
    - Rows: Test points (predictions)
    - Columns: Training points (with derivative structure from index)
    
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
        Number of input dimensions (rays).
    der_indices : list
        Derivative specifications for training data.
    powers : list of int
        Sign powers for each derivative type.
    return_deriv : bool
        If True, predict derivatives at ALL test points.
    index : list of lists or int, optional (default=-1)
        Training point indices for each derivative type.
    common_derivs : list, optional
        Common derivative indices to predict (intersection of training and requested).
    calc_cov : bool, optional (default=False)
        If True, computing covariance (use all indices for rows).
    powers_predict : list of int, optional
        Sign powers for prediction derivatives.

    Returns
    -------
    K : ndarray
        Prediction kernel matrix.
    """
    # --- 1. Initial Setup ---
    if calc_cov and not return_deriv:
        return phi.real
    
    dh = coti.get_dHelp()
    
    # Pre-compute signs
    signs = np.array([(-1.0) ** p for p in powers], dtype=np.float64)
    if powers_predict is not None:
        signs_predict = np.array([(-1.0) ** p for p in powers_predict], dtype=np.float64)
    else:
        signs_predict = signs
    
    # --- 2. Determine Block Sizes and Pre-allocate Matrix ---
    n_rows_func, n_cols_func = phi.shape
    n_deriv_types = len(der_indices)
    n_deriv_types_pred = len(common_derivs) if common_derivs else 0
    
    # Convert index to numpy arrays
    if isinstance(index, list) and len(index) > 0 and isinstance(index[0], (list, np.ndarray)):
        index_arrays = [np.asarray(idx, dtype=np.int64) for idx in index]
    else:
        index_arrays = []
    
    if return_deriv:
        der_map = deriv_map(n_bases, 2 * n_order)
        index_2 = np.arange(n_cols_func, dtype=np.int64)
        if calc_cov:
            index_cov = np.arange(n_cols_func, dtype=np.int64)
            n_deriv_types = n_deriv_types_pred
            n_pts_with_derivs_rows = n_deriv_types * n_cols_func
        else:
            n_pts_with_derivs_rows = sum(len(idx) for idx in index_arrays) if index_arrays else 0
    else:
        der_map = deriv_map(n_bases, n_order)
        index_2 = np.array([], dtype=np.int64)
        n_pts_with_derivs_rows = sum(len(idx) for idx in index_arrays) if index_arrays else 0

    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)
    
    if common_derivs:
        der_indices_tr_pred, der_ind_order_pred = transform_der_indices(common_derivs, der_map)
    else:
        der_indices_tr_pred, der_ind_order_pred = [], []
    
    n_pts_with_derivs_cols = n_deriv_types_pred * len(index_2)

    total_rows = n_rows_func + n_pts_with_derivs_rows 
    total_cols = n_cols_func + n_pts_with_derivs_cols 

    K = np.zeros((total_rows, total_cols))
    base_shape = (n_rows_func, n_cols_func)

    # --- 3. Fill the Matrix Block by Block ---

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
            
            # Multiply derivative indices to find correct flat index
            imdir1 = der_ind_order_pred[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            flat_idx = der_map[new_ord][new_idx]

            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Use numba for efficient submatrix extraction and assignment
            extract_and_assign(content_full, row_indices, index_2, K,
                              row_offset, col_offset, signs_predict[j + 1])
            col_offset += n_pts_col
        row_offset += n_pts_row

    return K