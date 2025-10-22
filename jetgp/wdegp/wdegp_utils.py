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


def differences_by_dim_func(X1, X2, n_order, index=-1):
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
    for k in range(d):
        # Create an empty (n, m) array for this dimension
        diffs_k = oti.zeros((n1, n2))

        # Nested loops to fill diffs_k
        for i in range(n1):
            diffs_k[i, :] = (
                X1[i, k]
                + oti.e(k + 1, order=2 * n_order)
                - (X2[:, k].T)
            )

        # Append to our list
        differences_by_dim.append(diffs_k)
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
    n_pts_deriv = len(index)
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


def determine_weights(diffs_by_dim, diffs_test, length_scales, kernel_func, sigma_n):
    """
    Compute interpolation weights for Weighted Coefficient Kriging (WCK)
    using a radial basis function (RBF) kernel.

    This method constructs a modified system to solve for interpolation weights
    that minimize the kriging variance, including a Lagrange multiplier to enforce unbiasedness.

    Parameters
    ----------
    diffs_by_dim : list of ndarray
        Pairwise differences between training points (by dimension).
    diffs_test : list of ndarray
        Pairwise differences between training and test point (by dimension).
    length_scales : array-like
        Kernel hyperparameters (log-scaled or raw depending on kernel_func).
    kernel_func : callable
        Kernel function (e.g., squared exponential, Matérn) accepting differences and returning
        a scalar covariance matrix.

    Returns
    -------
    w : ndarray of shape (n_train,)
        Interpolation weights for combining weighted Taylor basis coefficients
        in the kriging predictor.

    Notes
    -----
    This method solves the augmented kriging system:

        [ K   F ] [w ] = [r]
        [ F^T 0 ] [μ ]   [1]

    where:
      - K is the covariance matrix of the training basis coefficients,
      - r is the covariance vector between the test point and training coefficients,
      - F is a column vector of ones enforcing unbiasedness,
      - μ is the Lagrange multiplier,
      - w contains the interpolation weights used in Weighted Coefficient Kriging.
    """
    n1 = diffs_test[0].shape[1]
    index = [-1]

    phi = kernel_func(diffs_by_dim, length_scales, index)
    r = kernel_func(diffs_test, length_scales, index)

    K = phi.real
    F = np.ones((n1, 1))
    r = r.real.reshape(-1, 1)
    r = np.vstack((r, [1]))

    M = np.zeros((n1 + 1, n1 + 1))
    M[:n1, :n1] = K
    M[:n1, n1] = F.flatten()
    M[n1, :n1] = F.flatten()
    M[n1, n1] = 0

    solution = np.linalg.solve(M, r)
    w = solution[:n1]

    return w
