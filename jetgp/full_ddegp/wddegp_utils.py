import numpy as np
import pyoti.sparse as oti
import pyoti.core as coti  # Required for the advanced implementation
from line_profiler import profile


# @profile
# def differences_by_dim_func(X1, X2, rays, n_order, index=-1):
#     """
#     Compute dimension-wise pairwise differences between X1 and X2,
#     including hypercomplex perturbations in the directions specified by `rays`.

#     Parameters
#     ----------
#     X1 : ndarray of shape (n1, d)
#         First input array.
#     X2 : ndarray of shape (n2, d)
#         Second input array.
#     rays : ndarray of shape (d, n_rays)
#         Direction vectors applied for each dimension using hypercomplex algebra.
#     n_order : int
#         Derivative order used to define the hypercomplex perturbation.
#     index : list or int, optional
#         Index to determine which points receive tagging. Default is -1 (all tagged).

#     Returns
#     -------
#     differences_by_dim : list of ndarray
#         List of length d. Each entry is an (n1, n2) array of directional differences for one dimension.
#     """
#     X1 = oti.array(X1)
#     X2 = oti.array(X2)

#     n1, d = X1.shape
#     n2, _ = X2.shape
#     n_rays = rays.shape[1]

#     differences_by_dim = []

#     for k in range(d):
#         diffs_k = oti.zeros((n1, n2))
#         for i in range(n1):
#             for j in range(n2):
#                 dire1 = sum(oti.e(l + 1, order=2 * n_order) *
#                             rays[k, l] for l in range(n_rays))
#                 diffs_k[i, j] = (X1[i, k] + dire1) - X2[j, k]
#         differences_by_dim.append(diffs_k)
#     return differences_by_dim


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
    >>> diffs = differences_by_dim_func(X1, X2, rays, n_order)
    >>> len(diffs)
    2
    >>> diffs[0].shape
    (2, 2)
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


# def rbf_kernel(differences, length_scales, n_order, kernel_func, der_indices, powers, index=-1):
#     """
#     Compute a radial basis function (RBF) kernel matrix with hypercomplex derivative augmentation.

#     Parameters
#     ----------
#     differences : list of ndarray
#         List of difference arrays for each input dimension.
#     length_scales : ndarray
#         Log-scaled length scale parameters.
#     n_order : int
#         Maximum derivative order.
#     kernel_func : callable
#         Kernel function to be used for evaluating pairwise differences.
#     der_indices : list of lists
#         Multi-index derivatives specifying derivative evaluation directions.
#     powers : list of int
#         Parity powers used to scale contributions.
#     index : list or int, optional
#         Index for submodel selection or evaluation region.

#     Returns
#     -------
#     K : ndarray
#         Full kernel matrix with function values and derivative blocks.
#     """
#     phi = kernel_func(differences, length_scales, index)

#     for i in range(0, len(der_indices) + 1):
#         row_j = 0
#         for j in range(0, len(der_indices) + 1):
#             if j == 0 and i == 0:
#                 row_j = phi.real * (-1) ** powers[j]
#             elif j > 0 and i == 0:
#                 row_j = np.hstack(
#                     (row_j, (-1) ** powers[j] * phi.get_deriv(der_indices[j - 1])))
#             elif j == 0 and i > 0:
#                 row_j = phi.get_deriv(der_indices[i - 1])
#             else:
#                 row_j = np.hstack((
#                     row_j,
#                     (-1) ** powers[j] *
#                     phi.get_deriv(der_indices[j - 1] + der_indices[i - 1])
#                 ))
#         if i == 0:
#             K = row_j
#         else:
#             K = np.vstack((K, row_j))

#     return K

# @profile
# def rbf_kernel(differences, length_scales, n_order, n_rays, kernel_func, der_indices, powers, index=-1):
#     """
#     Assembles the full DD-GP covariance matrix with hypercomplex derivatives.

#     This is an optimized version that pre-allocates the final matrix and fills
#     it block by block, avoiding inefficient stacking operations.

#     Parameters
#     ----------
#     differences : list of ndarray
#         List of difference arrays for each input dimension.
#     length_scales : ndarray
#         Log-scaled length scale parameters.
#     n_order : int
#         Maximum derivative order.
#     kernel_func : callable
#         Kernel function to be used for evaluating pairwise differences.
#     der_indices : list of lists
#         Multi-index derivatives specifying derivative evaluation directions.
#     powers : list of int
#         Parity powers used to scale contributions.
#     index : list or int, optional
#         Index for submodel selection or evaluation region.

#     Returns
#     -------
#     K : ndarray
#         Full kernel matrix with function values and derivative blocks.
#     """
#     # Evaluate the kernel once to get the hypercomplex result
#     phi = kernel_func(differences, length_scales, index)

#     # Determine the dimensions of the final matrix
#     n_blocks = len(der_indices) + 1
#     block_rows, block_cols = phi.shape

#     # Pre-allocate the full covariance matrix with zeros
#     K = np.zeros((n_blocks * block_rows, n_blocks * block_cols))

#     # Fill the matrix block by block
#     for i in range(n_blocks):  # Block-row index
#         for j in range(n_blocks):  # Block-column index

#             # Get a view of the current block to be filled
#             K_block = K[i*block_rows: (i+1)*block_rows,
#                         j*block_cols: (j+1)*block_cols]

#             # Fill the block based on its position in the matrix
#             if i == 0 and j == 0:
#                 # Top-left block: K_ff (function-function)
#                 content = phi.real
#             elif i == 0 and j > 0:
#                 # Top row: K_fd (function-derivative)
#                 content = phi.get_deriv(der_indices[j - 1])
#             elif i > 0 and j == 0:
#                 # First column: K_df (derivative-function)
#                 content = phi.get_deriv(der_indices[i - 1])
#             else:
#                 # Inner blocks: K_dd (derivative-derivative)
#                 content = phi.get_deriv(
#                     der_indices[i - 1] + der_indices[j - 1])

#             # Apply the sign-flipping power and assign to the block
#             K_block[:, :] = content * ((-1) ** powers[j])

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


def rbf_kernel(
        phi,
    phi_exp,
    n_order,
    n_bases,
    der_indices,
    powers,
    index = -1
):
    """
    Assembles the full DD-GP covariance matrix using an efficient, pre-computed
    derivative array and block-wise matrix filling.
    
    Supports both uniform blocks (all derivatives at all points) and non-contiguous
    indices (different derivatives at different subsets of points).
    
    Parameters
    ----------
    differences : ndarray of shape (N, M, D)
        Pairwise differences between input points (X - X').
    length_scales : ndarray of shape (D,)
        Length scales for each input dimension (ARD).
    n_order : int
        Maximum derivative order considered.
    n_bases : int
        Total number of bases (function value + derivative terms).
    kernel_func : callable
        Function that computes the base RBF kernel and its derivatives.
    der_indices : list of lists
        Multi-index derivative structures for each derivative component.
    powers : list of int
        Powers of (-1) applied to each term (for symmetry or sign conventions).
    index : list of lists or None, optional (default=None)
        If None, assumes all derivative types apply to all training points (uniform blocks).
        If provided, specifies which training point indices have each derivative type,
        allowing non-contiguous index support and variable block sizes.
    direction_index : int, optional (default=-1)
        Index for selecting directional kernels or subspaces.
    return_deriv : bool, optional (default=True)
        If True, build full matrix with derivative-derivative blocks.
        If False, only build function and function-derivative blocks.
        
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
    n_pts_with_derivs_cols = sum(len(order_indices) for order_indices in index)
    n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index)
    total_rows = n_rows_func + n_pts_with_derivs_rows
    total_cols = n_cols_func + n_pts_with_derivs_cols

    K = np.zeros((total_rows, total_cols))
    base_shape = (n_rows_func, n_cols_func)
    
    # Block (0,0): Function-Function (K_ff)
    content_full = phi_exp[0].reshape(base_shape)
    K[:n_rows_func, :n_cols_func] = content_full * ((-1) ** powers[0])
    
    # First Block-Column: Derivative-Function (K_df)
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        flat_idx = der_indices_tr[i]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        
        current_indices = index[i]
        n_pts_this_order = len(current_indices)
        
        # Slice rows at specific indices
        content_sliced = content_full[current_indices, :]
        
        K[row_offset: row_offset + n_pts_this_order, :n_cols_func] = content_sliced * ((-1) ** powers[0])
        row_offset += n_pts_this_order
    
    
    # First Block-Row: Function-Derivative (K_fd)
    col_offset = n_cols_func
    for j in range(n_deriv_types):
        flat_idx = der_indices_tr[j]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        
        current_indices = index[j]
        n_pts_this_order = len(current_indices)
        
        # Slice columns at specific indices
        content_sliced = content_full[:, current_indices]
        
        K[:n_rows_func, col_offset: col_offset + n_pts_this_order] = content_sliced * ((-1) ** powers[j + 1])
        col_offset += n_pts_this_order
    
    # Inner Blocks: Derivative-Derivative (K_dd)
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        col_offset = n_cols_func
        
        row_indices = index[i]
        n_pts_row = len(row_indices)
        
        for j in range(n_deriv_types):
            col_indices = index[j]
            n_pts_col = len(col_indices)
            
            # Multiply derivative indices to find correct flat index
            imdir1 = der_ind_order[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            flat_idx = der_map[new_ord][new_idx]
            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Slice both rows and columns using non-contiguous indices
            content_sliced = content_full[np.ix_(row_indices, col_indices)]
            
            K[row_offset: row_offset + n_pts_row,
              col_offset: col_offset + n_pts_col] = content_sliced * ((-1) ** powers[j + 1])
            
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
    index = -1,
    calc_cov=False
):

    """
    Constructs the RBF kernel matrix for predictions with directional derivative entries.
    
    This handles the asymmetric case where:
    - Rows: Test points (predictions)
    - Columns: Training points (with derivative structure from index)

    Parameters
    ----------
    differences : ndarray of shape (N_test, N_train, D)
        Pairwise differences between test and training points.
    length_scales : ndarray of shape (D,)
        Length scales for each input dimension (ARD).
    n_order : int
        Maximum derivative order.
    n_bases : int
        Number of input dimensions.
    kernel_func : callable
        Function that computes the base RBF kernel and its derivatives.
    der_indices : list
        Derivative specifications.
    powers : list of int
        Sign powers for each derivative type.
    index : list of lists
        Training point indices for each derivative type.
    direction_index : int, optional (default=-1)
        Index for selecting directional kernels or subspaces.
    return_deriv : bool, optional (default=False)
        If True, predict derivatives at ALL test points.
    calc_cov : bool, optional (default=False)
        If True, computing covariance (use all indices for rows).

    Returns
    -------
    K : ndarray
        Prediction kernel matrix.
    """
    if calc_cov and not return_deriv:
        return phi.real
    dh = coti.get_dHelp()
    
    # 1. Evaluate the kernel once to get the hypercomplex result
    n_rows_func, n_cols_func = phi.shape  # (N_test, N_train)
    n_deriv_types = len(der_indices)
    
    if n_order == 0:
        return phi.real
    
    # 2. Extract derivative components based on return_deriv
    if not return_deriv:
        phi_exp = phi.get_all_derivs(n_bases, n_order)
        der_map = deriv_map(n_bases, n_order)
    else:
        phi_exp = phi.get_all_derivs(n_bases, 2 * n_order)
        der_map = deriv_map(n_bases, 2 * n_order)
    
    # 3. Create maps to translate derivative specifications to flat indices
    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)
    
    # --- Determine column indices (training points) ---
    # When return_deriv=True, we need K_fd blocks which use ALL test columns
    if return_deriv:
        col_indices_all = list(range(n_cols_func))  # All test points for columns
        n_pts_with_derivs_cols = n_deriv_types * n_cols_func
    else:
        n_pts_with_derivs_cols = 0
    
    # --- Determine row indices (test/prediction points) ---
    if calc_cov:
        # Covariance: use all indices for all derivative types
        row_indices_per_type = [list(range(n_rows_func)) for _ in range(n_deriv_types)]
        n_pts_with_derivs_rows = n_deriv_types * n_rows_func
    else:
        # Non-contiguous: different indices per derivative type
        row_indices_per_type = index
        n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index)
    
    # --- Pre-allocate matrix ---
    total_rows = n_rows_func + n_pts_with_derivs_rows
    total_cols = n_cols_func + n_pts_with_derivs_cols
    K = np.zeros((total_rows, total_cols))
    
    base_shape = (n_rows_func, n_cols_func)
    
    # =========================================================================
    # Block (0,0): Function-Function (K_ff)
    # =========================================================================
    content_full = phi_exp[0].reshape(base_shape)
    K[:n_rows_func, :n_cols_func] = content_full * ((-1) ** powers[0])
    
    # =========================================================================
    # First Block-Column: Derivative-Function (K_df)
    # Rows = derivative predictions, Cols = function training values
    # =========================================================================
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        row_indices = row_indices_per_type[i]
        n_pts_row = len(row_indices)
        
        flat_idx = der_indices_tr[i]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        
        # Slice rows at training point indices
        content_sliced = content_full[row_indices, :]
        
        K[row_offset: row_offset + n_pts_row, :n_cols_func] = content_sliced * ((-1) ** powers[0])
        row_offset += n_pts_row
    
    if not return_deriv:
        return K
    
    # =========================================================================
    # First Block-Row: Function-Derivative (K_fd)
    # Rows = function predictions, Cols = derivative training values
    # =========================================================================
    col_offset = n_cols_func
    for j in range(n_deriv_types):
        n_pts_col = n_cols_func  # All test points
        
        flat_idx = der_indices_tr[j]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        
        # Use all columns (all test points)
        K[:n_rows_func, col_offset: col_offset + n_pts_col] = content_full * ((-1) ** powers[j + 1])
        col_offset += n_pts_col
    
    # =========================================================================
    # Inner Blocks: Derivative-Derivative (K_dd)
    # =========================================================================
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        row_indices = row_indices_per_type[i]
        n_pts_row = len(row_indices)
        
        col_offset = n_cols_func
        for j in range(n_deriv_types):
            n_pts_col = n_cols_func  # All test points for columns
            
            # Multiply derivative indices to find correct flat index
            imdir1 = der_ind_order[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            flat_idx = der_map[new_ord][new_idx]
            
            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Slice rows at training indices, use all columns
            content_sliced = content_full[row_indices, :]
            
            K[row_offset: row_offset + n_pts_row,
              col_offset: col_offset + n_pts_col] = content_sliced * ((-1) ** powers[j + 1])
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

def find_common_derivatives(all_indices):
    """Find derivative indices common to all submodels."""
    sets = [set(to_tuple(elem) for elem in idx_list) for idx_list in all_indices]
    return sets[0].intersection(*sets[1:])

def extract_common_predictions(predictions, indices, common_tuples, include_fvals=True):
    """
    Extract rows corresponding to common derivatives.
    
    predictions: shape (num_ders, num_funcs) where row 0 is f_vals
    indices: list of derivative indices for this submodel
    common_tuples: set of common derivative indices (as tuples)
    """
    offset = 1 if include_fvals else 0
    extract_rows = [0] if include_fvals else []
    
    # Build a lookup: tuple -> row index
    idx_to_row = {to_tuple(idx): i + offset for i, idx in enumerate(indices)}
    
    # Extract rows in a consistent order (sorted by the tuple for reproducibility)
    for common_idx in sorted(common_tuples):
        extract_rows.append(idx_to_row[common_idx])
    
    return predictions[extract_rows]