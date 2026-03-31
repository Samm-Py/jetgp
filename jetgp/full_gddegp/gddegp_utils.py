import numpy as np
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
        Dimension index.
    X1, X2 : oti.array
        Input point arrays of shape (n1, d) and (n2, d).
    n1, n2 : int
        Number of points in X1, X2.
    rays_X1 : list of ndarray or None
        rays_X1[i] has shape (d, len(derivative_locations_X1[i])).
    rays_X2 : list of ndarray or None
        rays_X2[i] has shape (d, len(derivative_locations_X2[i])).
    derivative_locations_X1 : list of list
        derivative_locations_X1[i] contains indices of X1 points with direction i.
    derivative_locations_X2 : list of list
        derivative_locations_X2[i] contains indices of X2 points with direction i.
    e_tags_1, e_tags_2 : list
        OTI basis elements for each direction.
    oti_module : module
        The PyOTI static module.

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

    GDDEGP uses a dual-tag OTI scheme: X1 points are tagged with odd bases
    (e_1, e_3, e_5, ...) and X2 points with even bases (e_2, e_4, e_6, ...).
    This requires ``n_bases = 2 * n_direction_types``.

    The dual-tag approach is necessary because each point can have a unique
    directional ray, and the kernel matrix requires derivatives with respect to
    *both* sets of directions simultaneously. In the difference X1 - X2, the
    OTI coefficient for basis e_i at position (a, b) encodes only the ray of
    the point that was tagged with e_i. A single-tag scheme (tagging both X1
    and X2 with the same basis) would conflate the two rays in the difference,
    making it impossible to recover the correct cross-derivative
    ``v_i(a)^T H v_j(b)`` needed for K_dd blocks, and producing an asymmetric
    K_fd block when rays vary per point.

    Parameters
    ----------
    X1 : ndarray of shape (n1, d)
        First set of input points.
    X2 : ndarray of shape (n2, d)
        Second set of input points.
    rays_X1 : list of ndarray or None
        List of ray arrays for X1. rays_X1[i] has shape (d, len(derivative_locations_X1[i])).
    rays_X2 : list of ndarray or None
        List of ray arrays for X2. rays_X2[i] has shape (d, len(derivative_locations_X2[i])).
    derivative_locations_X1 : list of list
        derivative_locations_X1[i] contains indices of X1 points with derivative direction i.
    derivative_locations_X2 : list of list
        derivative_locations_X2[i] contains indices of X2 points with derivative direction i.
    n_order : int
        Derivative order for OTI tagging.
    oti_module : module
        The PyOTI static module (e.g., pyoti.static.onumm4n2).
    return_deriv : bool, optional
        If True, use order 2*n_order for derivative-derivative blocks.

    Returns
    -------
    differences_by_dim : list of oti.array
        List of length d, each element is an (n1, n2) OTI array.
    """
    X1 = oti_module.array(X1)
    X2 = oti_module.array(X2)
    n1, d = X1.shape
    n2, _ = X2.shape

    # Determine number of derivative directions
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
    index=None
):
    """
    Assembles the full GDDEGP covariance matrix with selective derivative coverage.
    
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
    index : list of list
        index[i] contains indices of points with derivative direction i.

    Returns
    -------
    K : ndarray
        Kernel matrix with block structure based on derivative locations.
    """
    dh = coti.get_dHelp()

    assert n_bases % 2 == 0, "n_bases must be an even number."
    PHIrows, PHIcols = phi.shape
    total_derivs = len(der_indices)

    # Compute output matrix dimensions
    n_deriv_rows = sum(len(locs) for locs in index)
    n_deriv_cols = sum(len(locs) for locs in index)
    n_output_rows = PHIrows + n_deriv_rows
    n_output_cols = PHIcols + n_deriv_cols

    der_map = deriv_map(n_bases, 2 * n_order)

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
    for i in range(total_derivs + 1):
        for j in range(total_derivs + 1):

            if i == 0 and j == 0:
                # K_ff: Full function-function block
                K[0:PHIrows, 0:PHIcols] = phi_exp[0]

            elif i == 0 and j > 0:
                # K_fd: Function rows, derivative j columns
                idx = der_indices_tr_even[j - 1]
                col_locs = index_arrays[j - 1]
                col_start = col_offsets[j]
                
                # Use numba for efficient column extraction
                extract_cols_and_assign(phi_exp[idx], col_locs, K,
                                        0, col_start, PHIrows)

            elif i > 0 and j == 0:
                # K_df: Derivative i rows, function columns
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


@numba.jit(nopython=True, cache=True)
def _assemble_kernel_numba(phi_exp_3d, K, n_rows_func, n_cols_func,
                           fd_flat_indices, df_flat_indices, dd_flat_indices,
                           idx_flat, idx_offsets, idx_sizes,
                           n_deriv_types, row_offsets, col_offsets):
    """Fused numba kernel for GDDEGP K matrix assembly (no signs, even/odd bases)."""
    # ff block
    for r in range(n_rows_func):
        for c in range(n_cols_func):
            K[r, c] = phi_exp_3d[0, r, c]
    # fd block (even indices)
    for j in range(n_deriv_types):
        fi = fd_flat_indices[j]
        co = col_offsets[j]
        off_j = idx_offsets[j]
        sz_j = idx_sizes[j]
        for r in range(n_rows_func):
            for k in range(sz_j):
                ci = idx_flat[off_j + k]
                K[r, co + k] = phi_exp_3d[fi, r, ci]
    # df block (odd indices)
    for i in range(n_deriv_types):
        fi = df_flat_indices[i]
        ro = row_offsets[i]
        off_i = idx_offsets[i]
        sz_i = idx_sizes[i]
        for k in range(sz_i):
            ri = idx_flat[off_i + k]
            for c in range(n_cols_func):
                K[ro + k, c] = phi_exp_3d[fi, ri, c]
    # dd block (even × odd)
    for i in range(n_deriv_types):
        ro = row_offsets[i]
        off_i = idx_offsets[i]
        sz_i = idx_sizes[i]
        for j in range(n_deriv_types):
            fi = dd_flat_indices[i, j]
            co = col_offsets[j]
            off_j = idx_offsets[j]
            sz_j = idx_sizes[j]
            for ki in range(sz_i):
                ri = idx_flat[off_i + ki]
                for kj in range(sz_j):
                    ci = idx_flat[off_j + kj]
                    K[ro + ki, co + kj] = phi_exp_3d[fi, ri, ci]


def precompute_kernel_plan(n_order, n_bases, der_indices, powers, index):
    """Precompute structural info for rbf_kernel_fast (GDDEGP even/odd variant)."""
    dh = coti.get_dHelp()
    assert n_bases % 2 == 0, "n_bases must be an even number."
    der_map = deriv_map(n_bases, 2 * n_order)

    n_deriv_types = len(der_indices)
    index_arrays = [np.asarray(idx, dtype=np.int64) for idx in index]

    index_sizes = np.array([len(idx) for idx in index_arrays], dtype=np.int64)
    n_pts_with_derivs = int(index_sizes.sum())

    idx_flat = np.concatenate(index_arrays) if n_deriv_types > 0 else np.array([], dtype=np.int64)
    idx_offsets = np.zeros(n_deriv_types, dtype=np.int64)
    for i in range(1, n_deriv_types):
        idx_offsets[i] = idx_offsets[i - 1] + index_sizes[i - 1]

    row_offsets = np.zeros(n_deriv_types, dtype=np.int64)
    col_offsets = np.zeros(n_deriv_types, dtype=np.int64)
    cumsum = 0
    for i in range(n_deriv_types):
        row_offsets[i] = cumsum
        col_offsets[i] = cumsum
        cumsum += index_sizes[i]

    # Even/odd derivative transforms
    der_indices_even = make_first_even(der_indices)
    der_indices_odd = make_first_odd(der_indices)
    der_indices_tr_even, der_ind_order_even = transform_der_indices(der_indices_even, der_map)
    der_indices_tr_odd, der_ind_order_odd = transform_der_indices(der_indices_odd, der_map)

    fd_flat_indices = np.array(der_indices_tr_even, dtype=np.int64)
    df_flat_indices = np.array(der_indices_tr_odd, dtype=np.int64)

    dd_flat_indices = np.empty((n_deriv_types, n_deriv_types), dtype=np.int64)
    for i in range(n_deriv_types):
        for j in range(n_deriv_types):
            imdir1 = der_ind_order_even[j]
            imdir2 = der_ind_order_odd[i]
            new_idx, new_ord = dh.mult_dir(imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            dd_flat_indices[i, j] = der_map[new_ord][new_idx]

    return {
        'signs': np.ones(n_deriv_types + 1, dtype=np.float64),  # unused, kept for API
        'index_arrays': index_arrays,
        'index_sizes': index_sizes,
        'n_pts_with_derivs': n_pts_with_derivs,
        'dd_flat_indices': dd_flat_indices,
        'n_deriv_types': n_deriv_types,
        'idx_flat': idx_flat,
        'idx_offsets': idx_offsets,
        'row_offsets': row_offsets,
        'col_offsets': col_offsets,
        'fd_flat_indices': fd_flat_indices,
        'df_flat_indices': df_flat_indices,
    }


def rbf_kernel_fast(phi_exp_3d, plan):
    """Fast kernel assembly using precomputed plan and fused numba kernel."""
    n_rows_func = phi_exp_3d.shape[1]
    n_cols_func = phi_exp_3d.shape[2]
    total = n_rows_func + plan['n_pts_with_derivs']
    K = np.empty((total, total))

    row_off = plan['row_offsets'] + n_rows_func
    col_off = plan['col_offsets'] + n_cols_func

    _assemble_kernel_numba(
        phi_exp_3d, K, n_rows_func, n_cols_func,
        plan['fd_flat_indices'], plan['df_flat_indices'], plan['dd_flat_indices'],
        plan['idx_flat'], plan['idx_offsets'], plan['index_sizes'],
        plan['n_deriv_types'], row_off, col_off,
    )
    return K


@profile
def rbf_kernel_predictions(
    phi,
    phi_exp,
    n_order,
    n_bases,
    der_indices,
    return_deriv,
    index=None,
    common_derivs=None,
    calc_cov=False,
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
    return_deriv : bool
        If True, predict derivatives at test points.
    index : list of list
        Training point indices for each derivative type.
    common_derivs : list
        Common derivative indices to predict.
    calc_cov : bool
        If True, computing covariance.

    Returns
    -------
    K : ndarray
        Prediction kernel matrix.
    """
    # Early return for covariance-only case
    if calc_cov and not return_deriv:
        return phi.real.T

    dh = coti.get_dHelp()

    n_train, n_test = phi.shape
    n_deriv_types = len(der_indices)
    n_deriv_types_pred = len(common_derivs) if common_derivs else 0

    # Handle n_order = 0 case
    if n_order == 0:
        return phi.real.T

    # Convert index lists to numpy arrays for numba
    index_arrays = [np.asarray(locs, dtype=np.int64) for locs in index]

    # Determine derivative map
    if return_deriv:
        der_map = deriv_map(n_bases, 2 * n_order)
        derivative_locations_test = [np.arange(n_test, dtype=np.int64)] * n_deriv_types_pred
    else:
        der_map = deriv_map(n_bases, n_order)

    # Create derivative index transformations
    der_indices_even = make_first_even(der_indices)
    der_indices_odd = make_first_odd(der_indices)
    der_indices_tr_odd, der_ind_order_odd = transform_der_indices(der_indices_odd, der_map)
    der_indices_odd_pred = make_first_odd(common_derivs) if common_derivs else []
    der_indices_tr_odd_pred, der_ind_order_odd_pred = transform_der_indices(der_indices_odd_pred, der_map) if common_derivs else ([], [])

    # Compute matrix dimensions
    n_rows_func = n_test
    if return_deriv:
        n_rows_derivs = sum(len(locs) for locs in derivative_locations_test)
    else:
        n_rows_derivs = 0
    total_rows = n_rows_func + n_rows_derivs

    if return_deriv and calc_cov:
        n_deriv_types = n_deriv_types_pred
        n_cols_derivs = sum(len(locs) for locs in derivative_locations_test)
        total_cols = n_train + n_cols_derivs
    else:
        n_cols_derivs = sum(len(locs) for locs in index)
        total_cols = n_train + n_cols_derivs

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

        flat_idx = der_indices_tr_odd_pred[j] if calc_cov else der_indices_tr_odd[j]
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

            imdir_train = der_ind_order_odd_pred[j] if calc_cov else der_ind_order_odd[j]
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