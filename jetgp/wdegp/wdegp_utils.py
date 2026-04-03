import pyoti.core as coti
import numpy as np
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

def differences_by_dim_func(X1, X2, n_order, oti_module, return_deriv=True):
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
    oti_module : module
        The PyOTI static module (e.g., pyoti.static.onumm4n2).
    return_deriv : bool, optional
        If True, use 2*n_order for derivative predictions.

    Returns
    -------
    differences_by_dim : list of length d
        A list where each element is an array of shape (n1, n2), containing the differences
        between corresponding dimensions of X1 and X2, augmented with hypercomplex units.
    """
    X1 = oti_module.array(X1)
    X2 = oti_module.array(X2)
    n1, d = X1.shape
    n2, d = X2.shape

    differences_by_dim = []

    if n_order == 0:
        for k in range(d):
            diffs_k = oti_module.zeros((n1, n2))
            for i in range(n1):
                diffs_k[i, :] = X1[i, k] - (oti_module.transpose(X2[:, k]))
            differences_by_dim.append(diffs_k)
    elif not return_deriv:
        for k in range(d):
            diffs_k = oti_module.zeros((n1, n2))
            for i in range(n1):
                diffs_k[i, :] = (
                    X1[i, k]
                    + oti_module.e(k + 1, order=n_order)
                    - (X2[:, k].T)
                )
            differences_by_dim.append(diffs_k)
    else:
        for k in range(d):
            diffs_k = oti_module.zeros((n1, n2))
            for i in range(n1):
                diffs_k[i, :] = X1[i, k] - (X2[:, k].T)
            differences_by_dim.append(diffs_k + oti_module.e(k + 1, order=2 * n_order))

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
    
    This version uses Numba-accelerated functions for efficient matrix slicing,
    replacing expensive np.ix_ operations.

    Parameters
    ----------
    phi : OTI array
        Base kernel matrix from kernel_func(differences, length_scales).
    phi_exp : ndarray
        Expanded derivative array from phi.get_all_derivs().
    n_order : int
        Maximum derivative order.
    n_bases : int
        Number of OTI bases.
    der_indices : list
        Derivative specifications.
    powers : list of int
        Sign powers for each derivative type.
    index : list of lists
        Training point indices for each derivative type.

    Returns
    -------
    K : ndarray
        Full RBF kernel matrix with mixed function and derivative entries.
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

    # Inner Blocks: Derivative-Derivative (K_dd)
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        col_offset = n_cols_func
        row_indices = index_arrays[i]
        n_pts_row = len(row_indices)
        
        for j in range(n_deriv_types):
            col_indices = index_arrays[j]
            n_pts_col = len(col_indices)
            
            # Multiply the derivative indices to find the correct flat index
            imdir1 = der_ind_order[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(
                imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            flat_idx = der_map[new_ord][new_idx]
            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Use numba for efficient submatrix extraction and assignment (replaces np.ix_)
            extract_and_assign(content_full, row_indices, col_indices, K,
                               row_offset, col_offset, signs[j + 1])
            
            col_offset += n_pts_col
        row_offset += n_pts_row

    return K


@numba.jit(nopython=True, cache=True)
def _assemble_kernel_numba(phi_exp_3d, K, n_rows_func, n_cols_func,
                           fd_flat_indices, df_flat_indices, dd_flat_indices,
                           idx_flat, idx_offsets, idx_sizes,
                           signs, n_deriv_types, row_offsets, col_offsets):
    """
    Fused numba kernel that assembles the entire K matrix in a single call.
    Handles ff, fd, df, and dd blocks without Python-level loop overhead.
    """
    # Block (0,0): Function-Function
    s0 = signs[0]
    for r in range(n_rows_func):
        for c in range(n_cols_func):
            K[r, c] = phi_exp_3d[0, r, c] * s0

    # First Block-Row: Function-Derivative (fd)
    for j in range(n_deriv_types):
        fi = fd_flat_indices[j]
        sj = signs[j + 1]
        co = col_offsets[j]
        off_j = idx_offsets[j]
        sz_j = idx_sizes[j]
        for r in range(n_rows_func):
            for k in range(sz_j):
                ci = idx_flat[off_j + k]
                K[r, co + k] = phi_exp_3d[fi, r, ci] * sj

    # First Block-Column: Derivative-Function (df)
    for i in range(n_deriv_types):
        fi = df_flat_indices[i]
        ro = row_offsets[i]
        off_i = idx_offsets[i]
        sz_i = idx_sizes[i]
        for k in range(sz_i):
            ri = idx_flat[off_i + k]
            for c in range(n_cols_func):
                K[ro + k, c] = phi_exp_3d[fi, ri, c] * s0

    # Inner Blocks: Derivative-Derivative (dd)
    for i in range(n_deriv_types):
        ro = row_offsets[i]
        off_i = idx_offsets[i]
        sz_i = idx_sizes[i]
        for j in range(n_deriv_types):
            fi = dd_flat_indices[i, j]
            sj = signs[j + 1]
            co = col_offsets[j]
            off_j = idx_offsets[j]
            sz_j = idx_sizes[j]
            for ki in range(sz_i):
                ri = idx_flat[off_i + ki]
                for kj in range(sz_j):
                    ci = idx_flat[off_j + kj]
                    K[ro + ki, co + kj] = phi_exp_3d[fi, ri, ci] * sj


def precompute_kernel_plan(n_order, n_bases, der_indices, powers, index):
    """
    Precompute all structural information needed by rbf_kernel so it can be
    reused across repeated calls with different phi_exp values.

    Returns a dict containing flat indices, signs, index arrays, precomputed
    offsets/sizes, and mult_dir results for the dd block.
    """
    dh = coti.get_dHelp()
    der_map = deriv_map(n_bases, 2 * n_order)
    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)

    n_deriv_types = len(der_indices)
    signs = np.array([(-1.0) ** p for p in powers], dtype=np.float64)
    index_arrays = [np.asarray(idx, dtype=np.int64) for idx in index]

    # Precompute sizes and offsets
    index_sizes = np.array([len(idx) for idx in index_arrays], dtype=np.int64)
    n_pts_with_derivs = int(index_sizes.sum())

    # Pack all index arrays into a single flat array with offsets
    idx_flat = np.concatenate(index_arrays) if n_deriv_types > 0 else np.array([], dtype=np.int64)
    idx_offsets = np.zeros(n_deriv_types, dtype=np.int64)
    for i in range(1, n_deriv_types):
        idx_offsets[i] = idx_offsets[i - 1] + index_sizes[i - 1]

    # Precompute row/col offsets in K for each deriv type
    row_offsets = np.zeros(n_deriv_types, dtype=np.int64)
    col_offsets = np.zeros(n_deriv_types, dtype=np.int64)
    # Note: n_rows_func == n_cols_func for training kernel, but we store
    # offsets relative to n_rows_func which is added at call time
    cumsum = 0
    for i in range(n_deriv_types):
        row_offsets[i] = cumsum  # relative to n_rows_func
        col_offsets[i] = cumsum  # relative to n_cols_func
        cumsum += index_sizes[i]

    # Precompute mult_dir results for dd blocks
    dd_flat_indices = np.empty((n_deriv_types, n_deriv_types), dtype=np.int64)
    for i in range(n_deriv_types):
        for j in range(n_deriv_types):
            imdir1 = der_ind_order[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(
                imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            dd_flat_indices[i, j] = der_map[new_ord][new_idx]

    # fd and df flat indices as arrays
    fd_flat_indices = np.array(der_indices_tr, dtype=np.int64)
    df_flat_indices = np.array(der_indices_tr, dtype=np.int64)

    return {
        'der_indices_tr': der_indices_tr,
        'signs': signs,
        'index_arrays': index_arrays,
        'index_sizes': index_sizes,
        'n_pts_with_derivs': n_pts_with_derivs,
        'dd_flat_indices': dd_flat_indices,
        'n_deriv_types': n_deriv_types,
        # Fused kernel data
        'idx_flat': idx_flat,
        'idx_offsets': idx_offsets,
        'row_offsets': row_offsets,
        'col_offsets': col_offsets,
        'fd_flat_indices': fd_flat_indices,
        'df_flat_indices': df_flat_indices,
    }


def rbf_kernel_fast(phi_exp_3d, plan, out=None):
    """
    Fast kernel assembly using a precomputed plan and fused numba kernel.

    Parameters
    ----------
    phi_exp_3d : ndarray of shape (n_derivs, n_rows_func, n_cols_func)
        Pre-reshaped expanded derivative array.
    plan : dict
        Precomputed plan from precompute_kernel_plan().
    out : ndarray, optional
        Pre-allocated output array. If None, a new array is allocated.

    Returns
    -------
    K : ndarray
        Full kernel matrix.
    """
    n_rows_func = phi_exp_3d.shape[1]
    n_cols_func = phi_exp_3d.shape[2]
    total = n_rows_func + plan['n_pts_with_derivs']
    if out is not None:
        K = out
    else:
        K = np.empty((total, total))

    if 'row_offsets_abs' in plan:
        row_off = plan['row_offsets_abs']
        col_off = plan['col_offsets_abs']
    else:
        row_off = plan['row_offsets'] + n_rows_func
        col_off = plan['col_offsets'] + n_cols_func

    _assemble_kernel_numba(
        phi_exp_3d, K, n_rows_func, n_cols_func,
        plan['fd_flat_indices'], plan['df_flat_indices'], plan['dd_flat_indices'],
        plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
        plan['signs'], plan['n_deriv_types'], row_off, col_off,
    )

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
        Number of OTI bases.
    der_indices : list
        Derivative specifications for training data.
    powers : list of int
        Sign powers for each derivative type.
    return_deriv : bool
        If True, predict derivatives at test points.
    index : list of lists
        Training point indices for each derivative type.
    common_derivs : list
        Common derivative indices to predict.
    calc_cov : bool
        If True, computing covariance.
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
            n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index) if isinstance(index, list) else 0
    else:
        der_map = deriv_map(n_bases, n_order)
        index_2 = np.array([], dtype=np.int64)
        n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index) if isinstance(index, list) else 0

    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)
    der_indices_tr_pred, der_ind_order_pred = transform_der_indices(common_derivs, der_map) if common_derivs else ([], [])
    n_pts_with_derivs_cols = n_deriv_types_pred * len([i for i in range(n_cols_func) if i < len(index_2)])

    total_rows = n_rows_func + n_pts_with_derivs_rows 
    total_cols = n_cols_func + n_pts_with_derivs_cols 

    K = np.zeros((total_rows, total_cols))
    base_shape = (n_rows_func, n_cols_func)

    # Convert index lists to numpy arrays for numba
    if isinstance(index, list) and len(index) > 0:
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
            imdir2 = der_ind_order_pred[i] if calc_cov else der_ind_order[i]
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


def to_tuple(item):
    """Convert list to tuple recursively."""
    if isinstance(item, list):
        return tuple(to_tuple(x) for x in item)
    return item


def to_list(x):
    """Convert tuple to list recursively."""
    if isinstance(x, tuple):
        return [to_list(i) for i in x]
    return x


def find_common_derivatives(all_indices):
    """Find derivative indices common to all submodels."""
    sets = [set(to_tuple(elem) for elem in idx_list) for idx_list in all_indices]
    return sets[0].intersection(*sets[1:])