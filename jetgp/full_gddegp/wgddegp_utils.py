import numpy as np
import pyoti.sparse as oti
from line_profiler import profile
import pyoti.core as coti
def make_first_odd(der_indices):

    # Nested structure: [[[1, 1]], [[2, 1]], ...]
    result = []
    for group in der_indices:
        new_group = []
        for pair in group:
            first = pair[0]
            new_group.append([2*first - 1, pair[1]])
        result.append(new_group)
    return result

def make_first_even(der_indices):

    # Nested structure: [[[1, 1]], [[2, 1]], ...]
    result = []
    for group in der_indices:
        new_group = []
        for pair in group:
            first = pair[0]
            new_group.append([2*first, pair[1]])
        result.append(new_group)
    return result

def compute_dimension_differences(k, X1, X2, n1, n2, rays_X1, rays_X2, 
                                   derivative_locations_X1, derivative_locations_X2,
                                   e_tags_1, e_tags_2):
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
    """
    # Build perturbation vector for X1
    # Use Python list to accumulate OTI values at each point index
    perturb_X1_values = [0.0] * n1
    if rays_X1 is not None:
        for dir_idx in range(len(rays_X1)):
            locs = derivative_locations_X1[dir_idx]
            rays = rays_X1[dir_idx]  # Shape: (d, len(locs))
            for j, pt_idx in enumerate(locs):
                # rays[:, j] is the ray direction for point at index pt_idx
                perturb_X1_values[pt_idx] = perturb_X1_values[pt_idx] + e_tags_1[dir_idx] * rays[k, j]
    
    # Build perturbation vector for X2
    perturb_X2_values = [0.0] * n2
    if rays_X2 is not None:
        for dir_idx in range(len(rays_X2)):
            locs = derivative_locations_X2[dir_idx]
            rays = rays_X2[dir_idx]  # Shape: (d, len(locs))
            for j, pt_idx in enumerate(locs):
                perturb_X2_values[pt_idx] = perturb_X2_values[pt_idx] + e_tags_2[dir_idx] * rays[k, j]
    
    # Convert to OTI arrays and reshape for broadcasting
    perturb_X1 = oti.array(perturb_X1_values)
    perturb_X2 = oti.array(perturb_X2_values)
    
    # Tag coordinates
    X1_k_tagged = X1[:, k] + perturb_X1
    X2_k_tagged = X2[:, k] + perturb_X2
    
    # Compute differences
    diffs_k = oti.zeros((n1, n2))
    for i in range(n1):
        diffs_k[i, :] = X1_k_tagged[i, 0] - X2_k_tagged[:, 0].T
    
    return diffs_k


def differences_by_dim_func(X1, X2, rays_X1, rays_X2, derivative_locations_X1, derivative_locations_X2, 
                            n_order, return_deriv=True):
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
    return_deriv : bool, optional
        If True, use order 2*n_order (for training kernel with derivative-derivative blocks)
        If False, use order n_order (for prediction without derivative outputs)
    
    Returns
    -------
    differences_by_dim : list of oti.array
        List of length d, each element is an (n1, n2) OTI array of differences for that dimension
    
    Example
    -------
    # Two derivative directions with different point coverage:
    # Direction 1 at points [0,1,2,3,4,5], Direction 2 at points [2,3,4]
    derivative_locations_X1 = [[0, 1, 2, 3, 4, 5], [2, 3, 4]]
    rays_X1 = [
        np.array(...),  # Shape: (d, 6) - one column per point in derivative_locations_X1[0]
        np.array(...)   # Shape: (d, 3) - one column per point in derivative_locations_X1[1]
    ]
    """
    X1 = oti.array(X1)
    X2 = oti.array(X2)
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
        # No perturbation when n_order is 0
        e_tags_1 = [0] * m
        e_tags_2 = [0] * m
    elif not return_deriv:
        # Prediction without derivatives: use order n_order
        for i in range(m):
            e_tags_1.append(oti.e((2*i + 1), order=n_order))
            e_tags_2.append(oti.e((2*i + 2), order=n_order))
    else:
        # Training or prediction with derivatives: use order 2*n_order
        for i in range(m):
            e_tags_1.append(oti.e((2*i + 1), order=2*n_order))
            e_tags_2.append(oti.e((2*i + 2), order=2*n_order))
    
    # Compute differences for each dimension
    differences_by_dim = []
    for k in range(d):
        diffs_k = compute_dimension_differences(
            k, X1, X2, n1, n2, rays_X1, rays_X2, 
            derivative_locations_X1, derivative_locations_X2,
            e_tags_1, e_tags_2
        )
        differences_by_dim.append(diffs_k)
    
    return differences_by_dim

def make_der_indices(num_directions: int, max_order: int):
    """
    Build the list of derivative-index specs:
        [[tag, order], â€¦]  for every order = 1..max_order
                            and every tag  = 1..num_directions
    """
    der_indices = []
    for order in range(1, max_order + 1):
        for tag in range(1, num_directions + 1):
            der_indices.append([[tag, order]])
    return der_indices


# def build_first_block_row(phi, max_order):
#     """
#     Build the first block row of a DD-GP kernel matrix.

#     Parameters
#     ----------
#     phi : ndarray (n x n), hypercomplex
#         Full kernel matrix with all tags embedded.
#     num_directions : int
#         Number of distinct directional tags.
#     max_order : int
#         Maximum derivative order to include.

#     Returns
#     -------
#     row : ndarray
#         Horizontally concatenated first row: K_ff | K_fd (order 1) | ... | K_fd (order R)
#     """
#     row = phi.real  # K_ff block

#     for order in range(1, max_order + 1):
#         deriv_block = phi.get_deriv([[2, order]])
#         row = np.hstack((row, deriv_block))

#     return row


# def build_first_block_column(phi, max_order):
#     """
#     Build the first block row of a DD-GP kernel matrix.

#     Parameters
#     ----------
#     phi : ndarray (n x n), hypercomplex
#         Full kernel matrix with all tags embedded.
#     num_directions : int
#         Number of distinct directional tags.
#     max_order : int
#         Maximum derivative order to include.

#     Returns
#     -------
#     row : ndarray
#         Horizontally concatenated first row: K_ff | K_fd (order 1) | ... | K_fd (order R)
#     """
#     column = phi.real  # K_ff block

#     for order in range(1, max_order + 1):
#         deriv_block = phi.get_deriv([[1, order]])
#         column = np.vstack((column, deriv_block))

#     return column


# def build_K_dd_full(phi, order_1, order_2):
#     """
#     Build the full (n * num_directions) x (n * num_directions) second derivative block matrix.

#     Parameters
#     ----------
#     phi : ndarray of shape (n, n), with hypercomplex entries
#         Kernel matrix with directional tagging.
#     num_directions : int
#         Number of directional directions (tags).

#     Returns
#     -------
#     K_dd : ndarray of shape (n * num_directions, n * num_directions)
#     """
#     mixed_tag = [[1, order_1], [2, order_2]]
#     # If the *array* object supports .get_deriv, this is one call:
#     K_dd = phi.get_deriv(mixed_tag).copy()

#     # If phi is a plain NumPy array of objects, fall back to:
#     # K_dd = np.empty_like(phi, dtype=float)
#     # for i in range(phi.shape[0]):
#     #     for j in range(phi.shape[1]):
#     #         K_dd[i, j] = phi[i, j].get_deriv(mixed_tag)

#     # ------------------------------------------------------------------
#     # 2) Replace diagonal entries with the special same-direction term
#     #    (-1)^{order_1} · d^{order_1+order_2}/d(tag1)
#     # ------------------------------------------------------------------
#     # diag_len = min(phi.shape)          # works whether or not phi is square
#     # diag_idx = np.arange(diag_len)

#     # # Compute new diagonal values once, then bulk-assign
#     # new_diag = [
#     #     (-1) ** order_1
#     #     * phi[int(k), int(k)].get_deriv([[1, order_1 + order_2]])
#     #     for k in diag_idx
#     # ]
#     # K_dd[diag_idx, diag_idx] = new_diag

#     return K_dd

# @profile
# def rbf_kernel(differences, length_scales, max_order, kernel_func, index=-1, return_deriv=True):
#     """
#     Assemble the full DD-GP covariance matrix, blockâ€“row by blockâ€“row,
#     strictly using the derivative calls and conventions you verified.

#     Parameters
#     ----------
#     phi : (n Ã— n) hypercomplex ndarray
#         Kernel matrix with all directional tags embedded.
#     num_directions : int
#     max_order      : int

#     Returns
#     -------
#     K : ndarray   # square, size  n Â· (1 + num_directions Â· max_order)
#     """

#     # ---------- 0)  top block-row  ---------------------------------
#     #      [ K_ff  |  K_fd(order=1) | K_fd(order=2) | ... ]
#     # shape (n , n + n*T*max_order)

#     phi = kernel_func(differences, length_scales, index)
#     nderivs = max_order
#     PHIrows = phi.shape[0]
#     PHIcols = phi.shape[1]
#     if max_order == 0:
#         return phi.real
#     if not return_deriv:
#         nderivs = max_order
#         K = np.zeros((PHIrows * (nderivs + 1), PHIcols))
#         iters = 1
#     else:
#         K = np.zeros((PHIrows * (nderivs + 1), PHIcols * (nderivs + 1)))
#         iters = max_order + 1

#     for i in range(0, max_order + 1):
#         for j in range(0, iters):
#             # Get local view of the global array.
#             Klocal = K[i*PHIrows: (i+1)*PHIrows, j*PHIcols: (j+1)*PHIcols]
#             if j == 0 and i == 0:
#                 Klocal[:, :] = phi.real
#             elif j > 0 and i == 0:
#                 Klocal[:, :] = phi.get_deriv([[2, j]])
#             elif j == 0 and i > 0:
#                 Klocal[:, :] = phi.get_deriv([[1, i]])
#             else:

#                 # imdir1 = der_ind_order[j-1]
#                 # imdir2 = der_ind_order[i-1]
#                 # idx, ord = dh.mult_dir(
#                 #     imdir1[0], imdir1[1], imdir2[0], imdir2[1])
#                 # idx = der_map[ord][idx]

#                 Klocal[:, :] = phi.get_deriv([[1, i], [2, j]])

#     # print(f'Shape final: ({K.shape})')
#     # print(f'Shape final factors: ')
#     # print(f' -> nrows: {K.shape[0]/phi.shape[0]}x')
#     # print(f' -> ncols: {K.shape[1]/phi.shape[1]}x')
#     # print(80*'-')

#     return K


def deriv_map(nbases, order):
    import pyoti.core as coti
    # dh = coti.get_dHelp()
    k = 0
    map_deriv = []
    # np.arange(coti.ndir_total(nbases,order),dtype=np.int64)
    for ordi in range(order+1):
        ndir = coti.ndir_order(nbases, ordi)
        map_deriv_i = [0] * ndir
        for idx in range(ndir):
            map_deriv_i[idx] = k
            k += 1
        map_deriv.append(map_deriv_i)
    return map_deriv


def transform_der_indices(der_indices, der_map):
    import pyoti.core as coti
    deriv_ind_transf = []
    deriv_ind_order = []
    for deriv in der_indices:
        # deriv_ind_transf_i = []
        # for deriv in der_list:
        #     deriv_ind_transf_i.append(coti.imdir(deriv))
        imdir = coti.imdir(deriv)
        idx, order = imdir
        deriv_ind_transf.append(der_map[order][idx])
        deriv_ind_order.append(imdir)
    return deriv_ind_transf, deriv_ind_order



# Assuming 'coti' and 'deriv_map' are defined elsewhere in your project
# import coti
# from some_module import deriv_map
@profile
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
    Assembles the full GDDEGP covariance matrix with support for selective
    derivative coverage via derivative_locations.

    Parameters
    ----------
    differences : array-like
        The differences between input points (OTI arrays with hypercomplex tags).
    length_scales : array-like
        The length scales for the RBF kernel.
    max_orders_per_dim : int
        Maximum derivative order for hypercomplex expansion.
    kernel_func : callable
        The function that evaluates the kernel.
    der_indices : list
        Derivative index specifications.
    derivative_locations : list of list or None
        derivative_locations[i] contains indices of points with derivative direction i.
        If None, assumes all derivatives at all points (uniform coverage).
        Example: [[0,1,2,3,4,5], [2,3,4]] means direction 1 at 6 points, direction 2 at 3 points.
    return_deriv : bool, optional
        If False, returns only K_ff and K_df blocks.
        If True, returns full matrix including K_fd and K_dd blocks.

    Returns
    -------
    K : ndarray
        Kernel matrix with block structure based on derivative_locations.
        
    Notes
    -----
    Output matrix structure with selective coverage:
    
           | f (all)    | d1 (locs1) | d2 (locs2) | ...
    -------+------------+------------+------------+----
    f(all) | K_ff       | K_fd1      | K_fd2      | ...
           | (n, n)     | (n, |l1|)  | (n, |l2|)  | ...
    -------+------------+------------+------------+----
    d1     | K_d1f      | K_d1d1     | K_d1d2     | ...
    (locs1)| (|l1|, n)  | (|l1|,|l1|)| (|l1|,|l2|)| ...
    -------+------------+------------+------------+----
    d2     | K_d2f      | K_d2d1     | K_d2d2     | ...
    (locs2)| (|l2|, n)  | (|l2|,|l1|)| (|l2|,|l2|)| ...
    
    Where n = number of training points, |li| = len(derivative_locations[i])
    """
    # Get the pyoti derivative helper
    dh = coti.get_dHelp()


    # --- VALIDATION AND SETUP ---
    
    highest_order = n_order
    # 1. Evaluate the kernel once (full n x n matrix)
    n_bases = phi.get_active_bases()[-1]
    assert n_bases % 2 == 0, "n_bases must be an even number."
    PHIrows, PHIcols = phi.shape
    total_derivs = len(der_indices)

   
    # --- COMPUTE OUTPUT MATRIX DIMENSIONS ---
    # Row dimension: n_points (function) + sum of derivative location lengths
    n_deriv_rows = sum(len(locs) for locs in index)

    # Full matrix with derivative columns too
    n_deriv_cols = sum(len(locs) for locs in index)
    n_output_rows = PHIrows + n_deriv_rows
    n_output_cols = PHIcols + n_deriv_cols
    
    phi_exp = phi.get_all_derivs(n_bases, 2 * highest_order)
    der_map = deriv_map(n_bases, 2 * highest_order)
    
    row_iters = total_derivs + 1
    col_iters = total_derivs + 1

    # --- PRE-COMPUTE DERIVATIVE INDEX TRANSFORMATIONS ---
    der_indices_even = make_first_even(der_indices)
    der_indices_odd = make_first_odd(der_indices)
    der_indices_tr_even, der_ind_order_even = transform_der_indices(der_indices_even, der_map)
    der_indices_tr_odd, der_ind_order_odd = transform_der_indices(der_indices_odd, der_map)

    # --- COMPUTE BLOCK OFFSETS ---
    # Row offsets: [0, PHIrows, PHIrows + |l0|, PHIrows + |l0| + |l1|, ...]
    row_offsets = [0, PHIrows]
    for i in range(total_derivs):
        row_offsets.append(row_offsets[-1] + len(index[i]))
    
    # Column offsets (only needed if return_deriv=True)
    col_offsets = [0, PHIcols]
    for i in range(total_derivs):
        col_offsets.append(col_offsets[-1] + len(index[i]))

    # --- ALLOCATE OUTPUT MATRIX ---
    K = np.zeros((n_output_rows, n_output_cols))

    # --- FILL BLOCKS ---
    for i in range(row_iters):
        for j in range(col_iters):
            
            if i == 0 and j == 0:
                # K_ff: Full function-function block (all points)
                idx = 0
                K[0:PHIrows, 0:PHIcols] = phi_exp[idx]

            elif i == 0 and j > 0:
                # K_fd: Function rows (all), derivative j columns (at derivative_locations[j-1])
                idx = der_indices_tr_even[j-1]
                col_locs = index[j-1]
                col_start = col_offsets[j]
                col_end = col_start + len(col_locs)
                # Extract columns at specified locations
                K[0:PHIrows, col_start:col_end] = phi_exp[idx][:, col_locs]

            elif i > 0 and j == 0:
                # K_df: Derivative i rows (at derivative_locations[i-1]), function columns (all)
                idx = der_indices_tr_odd[i-1]
                row_locs = index[i-1]
                row_start = row_offsets[i]
                row_end = row_start + len(row_locs)
                # Extract rows at specified locations
                K[row_start:row_end, 0:PHIcols] = phi_exp[idx][row_locs, :]

            else:
                # K_dd: Derivative i rows, derivative j columns
                imdir1 = der_ind_order_even[j-1]
                imdir2 = der_ind_order_odd[i-1]
                new_idx, new_ord = dh.mult_dir(
                    imdir1[0], imdir1[1], imdir2[0], imdir2[1])
                idx = der_map[new_ord][new_idx]

                row_locs = index[i-1]
                col_locs = index[j-1]
                row_start = row_offsets[i]
                row_end = row_start + len(row_locs)
                col_start = col_offsets[j]
                col_end = col_start + len(col_locs)
                
                # Use np.ix_ for 2D fancy indexing to extract submatrix
                K[row_start:row_end, col_start:col_end] = phi_exp[idx][np.ix_(row_locs, col_locs)]

    return K

# def rbf_kernel(differences, length_scales, max_order, kernel_func, der_indices, index=-1, return_deriv=True):
#     """
#     Assembles the full DD-GP covariance matrix using an efficient, pre-computed
#     derivative array and block-wise matrix filling.

#     Parameters
#     ----------
#     (Parameters are the same as the original function, with n_bases added)

#     Returns
#     -------
#     K : ndarray
#         Full kernel matrix with function values and derivative blocks.
#     """
#     # Get the pyoti derivative helper for multiplying derivative indices
#     dh = coti.get_dHelp()
#     n_bases = len(differences[0].get_active_bases())
#     # 1. Evaluate the kernel once to get the hypercomplex result
#     phi = kernel_func(differences, length_scales, index)

#     # 2. Extract ALL derivative components into a single flat array (highly efficient)
#     # The order must be 2 * max_order to capture mixed derivative terms
#     phi_exp = phi.get_all_derivs(n_bases, 2 * max_order)

#     # 3. Create maps to translate derivative specifications to flat indices
#     der_map = deriv_map(n_bases, 2 * max_order)
#     # 4. Pre-allocate the full covariance matrix
#     PHIrows, PHIcols = phi.shape

#     if max_order == 0:
#         return phi.real

#     if not return_deriv:
#         nderivs = max_order
#         K = np.zeros((PHIrows * (nderivs + 1), PHIcols))
#         iters = 1
#     else:
#         nderivs = max_order
#         K = np.zeros((PHIrows * (nderivs + 1), PHIcols * (nderivs + 1)))
#         iters = len(der_indices) + 1

#     # 5. Fill the matrix block by block using the pre-computed derivative array
#     for i in range(iters):  # Block-row index (corresponds to derivatives of X1)
#         for j in range(iters):      # Block-column index (corresponds to derivatives of X2)

#             Klocal = K[i*PHIrows:(i+1)*PHIrows, j*PHIcols:(j+1)*PHIcols]

#             if j == 0 and i == 0:
#                 # K_ff: Access the real part (index 0 of the flat array)
#                 idx = 0
#             elif j > 0 and i == 0:
#                 # K_fd: Derivative w.r.t. X2's tag, e(2)
#                 imdir = coti.imdir([[2, j]])
#                 idx = der_map[imdir[1]][imdir[0]]
#             elif j == 0 and i > 0:
#                 # K_df: Derivative w.r.t. X1's tag, e(1)
#                 imdir = coti.imdir([[1, i]])
#                 idx = der_map[imdir[1]][imdir[0]]
#             else:
#                 # K_dd: Mixed derivative w.r.t. e(1) and e(2)
#                 imdir1 = coti.imdir([[1, i]])
#                 imdir2 = coti.imdir([[2, j]])
#                 new_idx, new_ord = dh.mult_dir(
#                     imdir1[0], imdir1[1], imdir2[0], imdir2[1])
#                 idx = der_map[new_ord][new_idx]

#             # Assign content from the flat array using the calculated index
#             Klocal[:, :] = phi_exp[idx]

#     return K

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
    Constructs the RBF kernel matrix for predictions with selective derivative coverage.
    
    This handles the asymmetric case where:
    - Rows: Test points (predictions)
    - Columns: Training points (with derivative structure from derivative_locations_train)

    Parameters
    ----------
    differences : list of oti.array
        Dimension-wise differences between training and test points.
        Each element has shape (n_train, n_test).
    length_scales : ndarray of shape (D,)
        Length scales for each input dimension (ARD).
    n_order : int
        Maximum derivative order.
    kernel_func : callable
        Function that computes the base RBF kernel.
    der_indices : list
        Derivative specifications for each direction.
    derivative_locations_train : list of list
        derivative_locations_train[i] contains indices of TRAINING points 
        that have derivative direction i.
    derivative_locations_test : list of list or None
        derivative_locations_test[i] contains indices of TEST points 
        where we want to predict derivative direction i.
        If None and return_deriv=True, predicts derivatives at all test points.
    return_deriv : bool, optional (default=False)
        If True, predict derivatives at test points.
    calc_cov : bool, optional (default=False)
        If True, computing covariance - use all test indices for derivative rows.

    Returns
    -------
    K : ndarray
        Prediction kernel matrix.
        
    Notes
    -----
    OTI tagging convention:
    - X_train (rows of phi) → odd bases (1, 3, 5, ...) → der_indices_odd
    - X_test (cols of phi) → even bases (2, 4, 6, ...) → der_indices_even
    
    Output K structure (transposed from phi):
    - Rows = test points
    - Cols = training points
    """
    dh = coti.get_dHelp()
    
    # 1. Evaluate the kernel once to get the hypercomplex result

    n_train, n_test = phi.shape  # NOTE: rows=train (odd), cols=test (even)
    n_deriv_types = len(der_indices)
    n_bases =phi.get_active_bases()[-1]
    
    # Handle n_order = 0 case (no derivatives)
    if n_order == 0:
        return phi.real.T  # Transpose to get (n_test, n_train)
    
    
    # 2. Extract derivative components based on return_deriv
    if not return_deriv:
        phi_exp = phi.get_all_derivs(n_bases, n_order)
        der_map = deriv_map(n_bases, n_order)
    else:
        phi_exp = phi.get_all_derivs(n_bases, 2 * n_order)
        der_map = deriv_map(n_bases, 2 * n_order)
    
    # 3. Create maps to translate derivative specifications to flat indices
    # IMPORTANT: Use even/odd convention matching the OTI tagging
    # - Training (rows of phi) → odd bases → der_indices_odd
    # - Test (cols of phi) → even bases → der_indices_even
    der_indices_even = make_first_even(der_indices)  # For test (cols of phi)
    der_indices_odd = make_first_odd(der_indices)    # For train (rows of phi)
    
   
    der_indices_tr_odd, der_ind_order_odd = transform_der_indices(der_indices_odd, der_map)
    
    # --- COMPUTE MATRIX DIMENSIONS ---
    # Output rows = test side (transposed from phi cols)
    n_rows_func = n_test
    if return_deriv:
        derivative_locations_test = [i for i in range(n_test)]
        n_rows_derivs = sum(len(locs) for locs in derivative_locations_test)
    else:
        n_rows_derivs = 0
    total_rows = n_rows_func + n_rows_derivs
    
    # Output cols = training side (transposed from phi rows)
    n_cols_func = n_train
    n_cols_derivs = sum(len(locs) for locs in index)
    total_cols = n_cols_func + n_cols_derivs
    
    # --- COMPUTE BLOCK OFFSETS ---
    # Row offsets (test side): [0, n_test, n_test + |l0_te|, ...]
    row_offsets = [0, n_test]
    if return_deriv:
        for i in range(n_deriv_types):
            row_offsets.append(row_offsets[-1] + len(derivative_locations_test[i]))
    
    # Column offsets (training side): [0, n_train, n_train + |l0_tr|, ...]
    col_offsets = [0, n_train]
    for i in range(n_deriv_types):
        col_offsets.append(col_offsets[-1] + len(index[i]))
    
    # --- ALLOCATE OUTPUT MATRIX ---
    K = np.zeros((total_rows, total_cols))
    
    base_shape = (n_train, n_test)  # phi_exp shape: rows=train (odd), cols=test (even)
    
    # =========================================================================
    # Block (0,0): Function-Function (K_ff)
    # Output: (n_test, n_train) - need to transpose from phi_exp
    # =========================================================================
    content_full = phi_exp[0].reshape(base_shape)
    K[:n_test, :n_train] = content_full.T
    
    # =========================================================================
    # First Block-Row: Function-Derivative (K_fd)
    # Output rows: all test function values
    # Output cols: training derivatives at derivative_locations_train
    # Training is in rows of phi (odd tags) → use der_indices_odd
    # =========================================================================
    for j in range(n_deriv_types):
        train_locs = index[j]
        col_start = col_offsets[j + 1]
        col_end = col_start + len(train_locs)
        
        # Training derivatives use ODD indices (rows of phi)
        flat_idx = der_indices_tr_odd[j]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        
        # Extract training rows at derivative locations, then transpose
        # content_full[train_locs, :] has shape (|train_locs|, n_test)
        # Transpose to get (n_test, |train_locs|)
        K[:n_test, col_start:col_end] = content_full[train_locs, :].T
    
    if not return_deriv:
        return K
    
    # =========================================================================
    # First Block-Column: Derivative-Function (K_df)
    # Output rows: test derivatives at derivative_locations_test
    # Output cols: all training function values
    # Test is in cols of phi (even tags) → use der_indices_even
    # =========================================================================
    der_indices_tr_even, der_ind_order_even = transform_der_indices(der_indices_even, der_map)
    for i in range(n_deriv_types):
        test_locs = derivative_locations_test[i]
        row_start = row_offsets[i + 1]
        row_end = row_start + len(test_locs)
        
        # Test derivatives use EVEN indices (cols of phi)
        flat_idx = der_indices_tr_even[i]
        content_full = phi_exp[flat_idx].reshape(base_shape)
        
        # Extract test columns at derivative locations, then transpose
        # content_full[:, test_locs] has shape (n_train, |test_locs|)
        # Transpose to get (|test_locs|, n_train)
        K[row_start:row_end, :n_train] = content_full[:, test_locs].T
    
    # =========================================================================
    # Inner Blocks: Derivative-Derivative (K_dd)
    # Output rows: test derivatives (even) at derivative_locations_test
    # Output cols: training derivatives (odd) at derivative_locations_train
    # =========================================================================
    for i in range(n_deriv_types):
        test_locs = derivative_locations_test[i]
        row_start = row_offsets[i + 1]
        row_end = row_start + len(test_locs)
        
        for j in range(n_deriv_types):
            train_locs = index[j]
            col_start = col_offsets[j + 1]
            col_end = col_start + len(train_locs)
            
            # Multiply derivative indices:
            # - Training (cols of output K, rows of phi) → ODD
            # - Test (rows of output K, cols of phi) → EVEN
            imdir_train = der_ind_order_odd[j]   # Training uses odd
            imdir_test = der_ind_order_even[i]   # Test uses even
            new_idx, new_ord = dh.mult_dir(
                imdir_train[0], imdir_train[1], 
                imdir_test[0], imdir_test[1]
            )
            flat_idx = der_map[new_ord][new_idx]
            
            content_full = phi_exp[flat_idx].reshape(base_shape)
            
            # Extract submatrix: train rows at train_locs, test cols at test_locs
            # content_full[train_locs, :][:, test_locs] has shape (|train_locs|, |test_locs|)
            # Transpose to get (|test_locs|, |train_locs|)
            submatrix = content_full[np.ix_(train_locs, test_locs)]  # (|train_locs|, |test_locs|)
            K[row_start:row_end, col_start:col_end] = submatrix.T
    
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