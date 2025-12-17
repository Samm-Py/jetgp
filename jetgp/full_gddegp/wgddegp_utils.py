import numpy as np
from line_profiler import profile
import pyoti.core as coti
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


@numba.jit(nopython=True, cache=True)
def extract_submatrix_transposed(content_full, row_indices, col_indices):
    """
    Extract submatrix and return its transpose.
    Replaces content_full[np.ix_(row_indices, col_indices)].T
    
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
    result : ndarray of shape (len(col_indices), len(row_indices))
        Transposed extracted submatrix.
    """
    n_rows = len(row_indices)
    n_cols = len(col_indices)
    result = np.empty((n_cols, n_rows))
    for i in range(n_rows):
        ri = row_indices[i]
        for j in range(n_cols):
            result[j, i] = content_full[ri, col_indices[j]]
    return result


@numba.jit(nopython=True, cache=True)
def extract_rows_transposed(content_full, row_indices, n_cols):
    """
    Extract rows and return transposed result.
    Replaces content_full[row_indices, :].T
    
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
    result : ndarray of shape (n_cols, len(row_indices))
        Transposed extracted rows.
    """
    n_rows = len(row_indices)
    result = np.empty((n_cols, n_rows))
    for i in range(n_rows):
        ri = row_indices[i]
        for j in range(n_cols):
            result[j, i] = content_full[ri, j]
    return result


@numba.jit(nopython=True, cache=True)
def extract_cols_transposed(content_full, col_indices, n_rows):
    """
    Extract columns and return transposed result.
    Replaces content_full[:, col_indices].T
    
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
    result : ndarray of shape (len(col_indices), n_rows)
        Transposed extracted columns.
    """
    n_cols = len(col_indices)
    result = np.empty((n_cols, n_rows))
    for i in range(n_rows):
        for j in range(n_cols):
            result[j, i] = content_full[i, col_indices[j]]
    return result


@numba.jit(nopython=True, cache=True, parallel=False)
def extract_and_assign(content_full, row_indices, col_indices, K, 
                       row_start, col_start):
    """
    Extract submatrix and assign directly to K.
    
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
    """
    n_rows = len(row_indices)
    n_cols = len(col_indices)
    for i in range(n_rows):
        ri = row_indices[i]
        for j in range(n_cols):
            K[row_start + i, col_start + j] = content_full[ri, col_indices[j]]


@numba.jit(nopython=True, cache=True, parallel=False)
def extract_and_assign_transposed(content_full, row_indices, col_indices, K, 
                                  row_start, col_start):
    """
    Extract submatrix and assign its transpose directly to K.
    Replaces K[...] = content_full[np.ix_(row_indices, col_indices)].T
    
    Parameters
    ----------
    content_full : ndarray of shape (n_rows_full, n_cols_full)
        Source matrix.
    row_indices : ndarray of int64
        Row indices to extract from content_full.
    col_indices : ndarray of int64
        Column indices to extract from content_full.
    K : ndarray
        Target matrix to fill.
    row_start : int
        Starting row index in K.
    col_start : int
        Starting column index in K.
    """
    n_rows = len(row_indices)
    n_cols = len(col_indices)
    for i in range(n_rows):
        ri = row_indices[i]
        for j in range(n_cols):
            # Transposed assignment: K[col_idx, row_idx] = content[row_idx, col_idx]
            K[row_start + j, col_start + i] = content_full[ri, col_indices[j]]


@numba.jit(nopython=True, cache=True)
def extract_rows_and_assign(content_full, row_indices, K, 
                            row_start, col_start, n_cols):
    """
    Extract rows and assign directly to K.
    
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
    """
    n_rows = len(row_indices)
    for i in range(n_rows):
        ri = row_indices[i]
        for j in range(n_cols):
            K[row_start + i, col_start + j] = content_full[ri, j]


@numba.jit(nopython=True, cache=True)
def extract_cols_and_assign(content_full, col_indices, K, 
                            row_start, col_start, n_rows):
    """
    Extract columns and assign directly to K.
    
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
    """
    n_cols = len(col_indices)
    for i in range(n_rows):
        for j in range(n_cols):
            K[row_start + i, col_start + j] = content_full[i, col_indices[j]]


@numba.jit(nopython=True, cache=True)
def extract_rows_and_assign_transposed(content_full, row_indices, K, 
                                       row_start, col_start, n_cols):
    """
    Extract rows and assign transposed result directly to K.
    Replaces K[...] = content_full[row_indices, :].T
    
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
        Number of columns in content_full.
    """
    n_rows = len(row_indices)
    for i in range(n_rows):
        ri = row_indices[i]
        for j in range(n_cols):
            K[row_start + j, col_start + i] = content_full[ri, j]


@numba.jit(nopython=True, cache=True)
def extract_cols_and_assign_transposed(content_full, col_indices, K, 
                                       row_start, col_start, n_rows):
    """
    Extract columns and assign transposed result directly to K.
    Replaces K[...] = content_full[:, col_indices].T
    
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
        Number of rows in content_full.
    """
    n_cols = len(col_indices)
    for i in range(n_rows):
        for j in range(n_cols):
            K[row_start + j, col_start + i] = content_full[i, col_indices[j]]

# =============================================================================
# Difference computation functions
# =============================================================================

def compute_dimension_differences(k, X1, X2, n1, n2, rays_X1, rays_X2, 
                                   derivative_locations_X1, derivative_locations_X2,
                                   e_tags_1, e_tags_2, oti_module):
    """
    Compute differences for a single dimension k.
    Only perturbs points at specified derivative_locations with their corresponding rays.
    
    Parameters
    ----------
    k : int
        Dimension index
    X1, X2 : oti.array
        Input point arrays of shape (n1, d) and (n2, d)
    n1, n2 : int
        Number of points in X1, X2
    rays_X1 : list of ndarray or None
        rays_X1[i] has shape (d, len(derivative_locations_X1[i]))
        Column j corresponds to point derivative_locations_X1[i][j]
    rays_X2 : list of ndarray or None
        rays_X2[i] has shape (d, len(derivative_locations_X2[i]))
    derivative_locations_X1 : list of list
        derivative_locations_X1[i] contains indices of X1 points with direction i
    derivative_locations_X2 : list of list
        derivative_locations_X2[i] contains indices of X2 points with direction i
    e_tags_1, e_tags_2 : list
        OTI basis elements for each direction
    oti_module : module
        The PyOTI static module (e.g., pyoti.static.onumm4n2).
        
    Returns
    -------
    diffs_k : oti.array
        Differences for dimension k with shape (n1, n2).
    """
    # Build perturbation vector for X1
    perturb_X1_values = [0.0] * n1
    if rays_X1 is not None:
        for dir_idx in range(len(rays_X1)):
            locs = derivative_locations_X1[dir_idx]
            rays = rays_X1[dir_idx]
            for j, pt_idx in enumerate(locs):
                perturb_X1_values[pt_idx] = perturb_X1_values[pt_idx] + e_tags_1[dir_idx] * rays[k, j]
    
    # Build perturbation vector for X2
    perturb_X2_values = [0.0] * n2
    if rays_X2 is not None:
        for dir_idx in range(len(rays_X2)):
            locs = derivative_locations_X2[dir_idx]
            rays = rays_X2[dir_idx]
            for j, pt_idx in enumerate(locs):
                perturb_X2_values[pt_idx] = perturb_X2_values[pt_idx] + e_tags_2[dir_idx] * rays[k, j]
    
    # Convert to OTI arrays
    perturb_X1 = oti_module.array(perturb_X1_values)
    perturb_X2 = oti_module.array(perturb_X2_values)
    
    # Tag coordinates
    X1_k_tagged = X1[:, k] + perturb_X1
    X2_k_tagged = X2[:, k] + perturb_X2
    
    # Compute differences
    diffs_k = oti_module.zeros((n1, n2))
    for i in range(n1):
        diffs_k[i, :] = X1_k_tagged[i, 0] - oti_module.transpose(X2_k_tagged[:, 0])
    
    return diffs_k


def differences_by_dim_func(X1, X2, rays_X1, rays_X2, derivative_locations_X1, derivative_locations_X2, 
                            n_order, oti_module, return_deriv=True):
    """
    Compute dimension-wise differences with OTI tagging on both X1 and X2.
    Only perturbs points at specified derivative_locations with their corresponding rays.
    
    Parameters
    ----------
    X1 : ndarray of shape (n1, d)
        First set of input points
    X2 : ndarray of shape (n2, d)
        Second set of input points
    rays_X1 : list of ndarray or None
        List of ray arrays for X1. rays_X1[i] has shape (d, len(derivative_locations_X1[i]))
        where column j contains the ray direction for point derivative_locations_X1[i][j]
    rays_X2 : list of ndarray or None
        List of ray arrays for X2. rays_X2[i] has shape (d, len(derivative_locations_X2[i]))
    derivative_locations_X1 : list of list
        derivative_locations_X1[i] contains indices of X1 points that have derivative direction i
    derivative_locations_X2 : list of list
        derivative_locations_X2[i] contains indices of X2 points that have derivative direction i
    n_order : int
        Derivative order for OTI tagging
    oti_module : module
        The PyOTI static module (e.g., pyoti.static.onumm4n2).
    return_deriv : bool, optional
        If True, use order 2*n_order (for training kernel with derivative-derivative blocks)
        If False, use order n_order (for prediction without derivative outputs)
    
    Returns
    -------
    differences_by_dim : list of oti.array
        List of length d, each element is an (n1, n2) OTI array of differences for that dimension
    """
    X1 = oti_module.array(X1)
    X2 = oti_module.array(X2)
    n1, d = X1.shape
    n2, _ = X2.shape
    
    # Determine number of derivative directions from rays arrays
    m1 = len(rays_X1) if rays_X1 is not None else 0
    m2 = len(rays_X2) if rays_X2 is not None else 0
    m = max(m1, m2)
    
    # Pre-compute OTI basis elements
    e_tags_1 = []
    e_tags_2 = []
    
    if n_order == 0:
        e_tags_1 = [0] * m
        e_tags_2 = [0] * m
    elif not return_deriv:
        for i in range(m):
            e_tags_1.append(oti_module.e((2 * i + 1), order=n_order))
            e_tags_2.append(oti_module.e((2 * i + 2), order=n_order))
    else:
        for i in range(m):
            e_tags_1.append(oti_module.e((2 * i + 1), order=2 * n_order))
            e_tags_2.append(oti_module.e((2 * i + 2), order=2 * n_order))
    
    # Compute differences for each dimension
    differences_by_dim = []
    for k in range(d):
        diffs_k = compute_dimension_differences(
            k, X1, X2, n1, n2, rays_X1, rays_X2, 
            derivative_locations_X1, derivative_locations_X2,
            e_tags_1, e_tags_2, oti_module
        )
        differences_by_dim.append(diffs_k)
    
    return differences_by_dim
# =============================================================================
# Derivative index transformation utilities
# =============================================================================

def make_first_odd(der_indices):
    """Transform derivative indices to use odd bases (1, 3, 5, ...)."""
    result = []
    for group in der_indices:
        new_group = []
        for pair in group:
            first = pair[0]
            new_group.append([2 * first - 1, pair[1]])
        result.append(new_group)
    return result


def make_first_even(der_indices):
    """Transform derivative indices to use even bases (2, 4, 6, ...)."""
    result = []
    for group in der_indices:
        new_group = []
        for pair in group:
            first = pair[0]
            new_group.append([2 * first, pair[1]])
        result.append(new_group)
    return result



# =============================================================================
# Derivative mapping utilities
# =============================================================================

def deriv_map(nbases, order):
    """Create mapping from (order, index) to flattened index."""
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
    """Transform derivative indices to flattened format."""
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
    Assembles the full GDDEGP covariance matrix with support for selective
    derivative coverage via derivative_locations.
    
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
        Number of OTI bases (must be even).
    der_indices : list
        Derivative index specifications.
    powers : list of int
        Powers of (-1) applied to each term (unused but kept for API consistency).
    index : list of list
        index[i] contains indices of points with derivative direction i.

    Returns
    -------
    K : ndarray
        Kernel matrix with block structure based on derivative_locations.
    """
    dh = coti.get_dHelp()

    highest_order = n_order
    if n_order == 0:
        n_bases = 0
        phi_exp = phi.real
        phi_exp = phi_exp[np.newaxis, :, :]
    else:
        n_bases = phi.get_active_bases()[-1]
        phi_exp = phi.get_all_derivs(n_bases, 2 * highest_order)
    assert n_bases % 2 == 0, "n_bases must be an even number."
    PHIrows, PHIcols = phi.shape
    total_derivs = len(der_indices)

    # Compute output matrix dimensions
    n_deriv_rows = sum(len(locs) for locs in index)
    n_deriv_cols = sum(len(locs) for locs in index)
    n_output_rows = PHIrows + n_deriv_rows
    n_output_cols = PHIcols + n_deriv_cols
    

    der_map = deriv_map(n_bases, 2 * highest_order)
    
    row_iters = total_derivs + 1
    col_iters = total_derivs + 1

    # Pre-compute derivative index transformations
    der_indices_even = make_first_even(der_indices)
    der_indices_odd = make_first_odd(der_indices)
    der_indices_tr_even, der_ind_order_even = transform_der_indices(der_indices_even, der_map)
    der_indices_tr_odd, der_ind_order_odd = transform_der_indices(der_indices_odd, der_map)

    # Convert index lists to numpy arrays for numba
    index_arrays = [np.asarray(locs, dtype=np.int64) for locs in index]

    # Compute block offsets
    row_offsets = [0, PHIrows]
    for i in range(total_derivs):
        row_offsets.append(row_offsets[-1] + len(index[i]))
    
    col_offsets = [0, PHIcols]
    for i in range(total_derivs):
        col_offsets.append(col_offsets[-1] + len(index[i]))

    # Allocate output matrix
    K = np.zeros((n_output_rows, n_output_cols))

    # Fill blocks
    for i in range(row_iters):
        for j in range(col_iters):
            
            if i == 0 and j == 0:
                # K_ff: Full function-function block (all points)
                idx = 0
                K[0:PHIrows, 0:PHIcols] = phi_exp[idx]

            elif i == 0 and j > 0:
                # K_fd: Function rows (all), derivative j columns (at derivative_locations[j-1])
                idx = der_indices_tr_even[j - 1]
                col_locs = index_arrays[j - 1]
                col_start = col_offsets[j]
                
                # Use numba for efficient column extraction
                extract_cols_and_assign(phi_exp[idx], col_locs, K,
                                        0, col_start, PHIrows)

            elif i > 0 and j == 0:
                # K_df: Derivative i rows (at derivative_locations[i-1]), function columns (all)
                idx = der_indices_tr_odd[i - 1]
                row_locs = index_arrays[i - 1]
                row_start = row_offsets[i]
                
                # Use numba for efficient row extraction
                extract_rows_and_assign(phi_exp[idx], row_locs, K,
                                        row_start, 0, PHIcols)

            else:
                # K_dd: Derivative i rows, derivative j columns
                imdir1 = der_ind_order_even[j - 1]
                imdir2 = der_ind_order_odd[i - 1]
                new_idx, new_ord = dh.mult_dir(
                    imdir1[0], imdir1[1], imdir2[0], imdir2[1])
                idx = der_map[new_ord][new_idx]

                row_locs = index_arrays[i - 1]
                col_locs = index_arrays[j - 1]
                row_start = row_offsets[i]
                col_start = col_offsets[j]
                
                # Use numba for efficient submatrix extraction (replaces np.ix_)
                extract_and_assign(phi_exp[idx], row_locs, col_locs, K,
                                   row_start, col_start)

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
    Constructs the RBF kernel matrix for predictions with selective derivative coverage.
    
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
        Sign powers (unused but kept for API consistency).
    return_deriv : bool
        If True, predict derivatives at test points.
    index : list of list
        Training point indices for each derivative type.
    common_derivs : list
        Common derivative indices to predict.
    calc_cov : bool
        If True, computing covariance.
    powers_predict : list of int, optional
        Sign powers for prediction derivatives (unused but kept for API consistency).

    Returns
    -------
    K : ndarray
        Prediction kernel matrix.
    """
    if calc_cov and not return_deriv:
        return phi.real
    
    dh = coti.get_dHelp()
    
    n_train, n_test = phi.shape
    n_deriv_types = len(der_indices)
    n_deriv_types_pred = len(common_derivs) if common_derivs else 0
    
    # Handle n_order = 0 case
    if n_order == 0:
        return phi.real.T

    # Convert index lists to numpy arrays for numba
    index_arrays = [np.asarray(locs, dtype=np.int64) for locs in index]
    
    # Extract derivative components based on return_deriv
    if not return_deriv:
        phi_exp = phi.get_all_derivs(n_bases, n_order)
        der_map = deriv_map(n_bases, n_order)
    else:
        phi_exp = phi.get_all_derivs(n_bases, 2 * n_order)
        der_map = deriv_map(n_bases, 2 * n_order)
    
    # Create derivative index transformations
    der_indices_even = make_first_even(der_indices)
    der_indices_odd = make_first_odd(der_indices)
    der_indices_tr_odd, der_ind_order_odd = transform_der_indices(der_indices_odd, der_map)
    
    # Compute matrix dimensions
    n_rows_func = n_test
    if return_deriv:
        derivative_locations_test = [np.arange(n_test, dtype=np.int64)] * n_deriv_types_pred
        n_rows_derivs = sum(len(locs) for locs in derivative_locations_test)
    else:
        n_rows_derivs = 0
    total_rows = n_rows_func + n_rows_derivs
    
    if return_deriv and calc_cov:
        n_cols_func = n_train
        n_deriv_types = n_deriv_types_pred 
        n_cols_derivs = sum(len(locs) for locs in derivative_locations_test)
        total_cols = n_cols_func + n_cols_derivs
    else:
        n_cols_func = n_train
        n_cols_derivs = sum(len(locs) for locs in index)
        total_cols = n_cols_func + n_cols_derivs
    
    # Compute block offsets
    row_offsets = [0, n_test]
    if return_deriv:
        for i in range(n_deriv_types_pred):
            row_offsets.append(row_offsets[-1] + len(derivative_locations_test[i]))
    
    col_offsets = [0, n_train]
    for i in range(n_deriv_types):
        col_offsets.append(col_offsets[-1] + len(index[i]))
    
    # Allocate output matrix
    K = np.zeros((total_rows, total_cols))
    base_shape = (n_train, n_test)
    
    # Block (0,0): Function-Function (K_ff)
    content_full = phi_exp[0].reshape(base_shape)
    K[:n_test, :n_train] = content_full.T
    
    # First Block-Row: Function-Derivative (K_fd)
    for j in range(n_deriv_types):
        train_locs = index_arrays[j]
        col_start = col_offsets[j + 1]
        
        flat_idx = der_indices_tr_odd[j]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        
        # Use numba for efficient row extraction with transpose
        extract_rows_and_assign_transposed(content_full, train_locs, K,
                                           0, col_start, n_test)
    
    if not return_deriv:
        return K
    
    # First Block-Column: Derivative-Function (K_df)
    der_indices_tr_even, der_ind_order_even = transform_der_indices(der_indices_even, der_map)
    der_indices_even_pred = make_first_even(common_derivs)
    der_indices_tr_even_pred, der_ind_order_even_pred = transform_der_indices(der_indices_even_pred, der_map)
    
    for i in range(n_deriv_types_pred):
        test_locs = derivative_locations_test[i]
        row_start = row_offsets[i + 1]
        
        flat_idx = der_indices_tr_even_pred[i]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        
        # Use numba for efficient column extraction with transpose
        extract_cols_and_assign_transposed(content_full, test_locs, K,
                                           row_start, 0, n_train)
    
    # Inner Blocks: Derivative-Derivative (K_dd)
    for i in range(n_deriv_types_pred):
        test_locs = derivative_locations_test[i]
        row_start = row_offsets[i + 1]
        
        for j in range(n_deriv_types):
            train_locs = index_arrays[j]
            col_start = col_offsets[j + 1]
            
            imdir_train = der_ind_order_odd[j]
            imdir_test = der_ind_order_even_pred[i]
            new_idx, new_ord = dh.mult_dir(
                imdir_train[0], imdir_train[1], 
                imdir_test[0], imdir_test[1]
            )
            flat_idx = der_map[new_ord][new_idx]
            
            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Use numba for efficient submatrix extraction with transpose (replaces np.ix_ + .T)
            extract_and_assign_transposed(content_full, train_locs, test_locs, K,
                                          row_start, col_start)
    
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