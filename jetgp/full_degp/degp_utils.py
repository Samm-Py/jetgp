import numpy as np
import pyoti.sparse as oti
import pyoti.core as coti
from line_profiler import profile


@profile
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


def differences_by_dim_func_SI(X1, X2, n_order, return_deriv=True, index=-1):
    """
    Compute pairwise differences for sine-exponential kernel (periodic).
    """
    n1, d = X1.shape
    n2, d = X2.shape

    differences_by_dim = []

    if n_order == 0:
        for k in range(d):
            diffs_k = np.zeros((n1, n2))
            diffs_k[:, :] = (
                X1[:, k].reshape(-1, 1) - X2[:, k].reshape(1, -1)
            ) % 1
            differences_by_dim.append(oti.array(diffs_k))
    elif not return_deriv:
        for k in range(d):
            diffs_k = np.zeros((n1, n2))
            diffs_k[:, :] = (
                X1[:, k].reshape(-1, 1) - X2[:, k].reshape(1, -1)
            ) % 1
            differences_by_dim.append(oti.array(diffs_k) + oti.e(1, order=n_order))
    else:
        for k in range(d):
            diffs_k = np.zeros((n1, n2))
            diffs_k[:, :] = (
                X1[:, k].reshape(-1, 1) - X2[:, k].reshape(1, -1)
            ) % 1
            differences_by_dim.append(oti.array(diffs_k) + oti.e(k + 1, order=2 * n_order))

    return differences_by_dim


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
    Compute the derivative-enhanced RBF kernel matrix.

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
    return_deriv : bool, optional (default=True)
        If True, build full matrix with derivative-derivative blocks.
    index : list of lists or None, optional (default=None)
        If None, assumes uniform blocks. If provided, specifies which
        training point indices have each derivative type.

    Returns
    -------
    K : ndarray
        Kernel matrix including function values and derivative terms.
    """
    dh = coti.get_dHelp()

    n_rows_func, n_cols_func = phi.shape
    n_deriv_types = len(der_indices)

    # Determine derivative map based on return_deriv


    der_map = deriv_map(n_bases, 2 * n_order)

    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)

    # =========================================================================
    # CASE 1: Uniform blocks (original behavior) - index is None
    # =========================================================================
    if index is None:
       
        K = np.zeros((n_rows_func * (n_deriv_types + 1), n_cols_func * (n_deriv_types + 1)))
        outer_loop_index = n_deriv_types + 1

        for j in range(outer_loop_index):
            signj = (-1) ** powers[j]
            for i in range(n_deriv_types + 1):
                Klocal = K[i * n_rows_func: (i + 1) * n_rows_func,
                           j * n_cols_func: (j + 1) * n_cols_func]

                if j == 0 and i == 0:
                    Klocal[:, :] = phi_exp[0] * signj
                elif j > 0 and i == 0:
                    Klocal[:, :] = signj * phi_exp[der_indices_tr[j - 1]]
                elif j == 0 and i > 0:
                    Klocal[:, :] = phi_exp[der_indices_tr[i - 1]]
                else:
                    imdir1 = der_ind_order[j - 1]
                    imdir2 = der_ind_order[i - 1]
                    idx, ord = dh.mult_dir(imdir1[0], imdir1[1], imdir2[0], imdir2[1])
                    idx = der_map[ord][idx]
                    Klocal[:, :] = signj * phi_exp[idx]

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

            imdir1 = der_ind_order[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            flat_idx = der_map[new_ord][new_idx]
            content_full = phi_exp[flat_idx].reshape(base_shape)

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
    index=None,
    common_derivs=None,
    calc_cov=False,
    powers_predict=None
):
    """
    Constructs the RBF kernel matrix for predictions with derivative entries.

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

    # Determine derivative map
    if return_deriv:
        der_map = deriv_map(n_bases, 2 * n_order)
        index_2 = list(range(phi_exp.shape[-1]))
        if calc_cov:
            index = list(range(phi_exp.shape[-1]))
            n_deriv_types = n_deriv_types_pred
            n_pts_with_derivs_rows = n_deriv_types * len([i for i in range(n_cols_func) if i in index_2])
        else:
            n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index)
    else:
        der_map = deriv_map(n_bases, n_order)
        index_2 = []
        n_pts_with_derivs_rows = sum(len(order_indices) for order_indices in index)

    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)
    der_indices_tr_pred, der_ind_order_pred = transform_der_indices(common_derivs, der_map) if common_derivs else ([], [])
    n_pts_with_derivs_cols = n_deriv_types_pred * len([i for i in range(n_cols_func) if i in index_2])

    total_rows = n_rows_func + n_pts_with_derivs_rows
    total_cols = n_cols_func + n_pts_with_derivs_cols

    K = np.zeros((total_rows, total_cols))
    base_shape = (n_rows_func, n_cols_func)

    # Block (0,0): Function-Function (K_ff)
    content_full = phi_exp[0].reshape(base_shape)
    K[:n_rows_func, :n_cols_func] = content_full * ((-1) ** powers[0])

    if not return_deriv:
        # First Block-Column: Derivative-Function (K_df)
        row_offset = n_rows_func
        for i in range(n_deriv_types):
            if calc_cov:
                row_indices = index_2
            else:
                row_indices = index[i]
            n_pts_row = len(row_indices)

            flat_idx = der_indices_tr[i]
            content_full = phi_exp[flat_idx].reshape(base_shape)
            content_sliced = content_full[row_indices, :]

            K[row_offset: row_offset + n_pts_row, :n_cols_func] = content_sliced * ((-1) ** powers[0])
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
        content_sliced = content_full[:, col_indices]

        K[:n_rows_func, col_offset: col_offset + n_pts_col] = content_sliced * ((-1) ** powers_predict[j + 1])
        col_offset += n_pts_col

    # First Block-Column: Derivative-Function (K_df)
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        if calc_cov:
            row_indices = index
            flat_idx = der_indices_tr_pred[i]
        else:
            row_indices = index[i]
            flat_idx = der_indices_tr[i]
        n_pts_row = len(row_indices)

        content_full = phi_exp[flat_idx].reshape(base_shape)
        content_sliced = content_full[row_indices, :]

        K[row_offset: row_offset + n_pts_row, :n_cols_func] = content_sliced * ((-1) ** powers[0])
        row_offset += n_pts_row

    # Inner Blocks: Derivative-Derivative (K_dd)
    row_offset = n_rows_func
    for i in range(n_deriv_types):
        if calc_cov:
            row_indices = index
        else:
            row_indices = index[i]
        n_pts_row = len(row_indices)

        col_offset = n_cols_func
        for j in range(n_deriv_types_pred):
            col_indices = index_2
            n_pts_col = len(col_indices)

            imdir1 = der_ind_order_pred[j]
            imdir2 = der_ind_order[i]
            new_idx, new_ord = dh.mult_dir(imdir1[0], imdir1[1], imdir2[0], imdir2[1])
            flat_idx = der_map[new_ord][new_idx]

            content_full = phi_exp[flat_idx].reshape(base_shape)
            content_sliced = content_full[np.ix_(row_indices, col_indices)]

            K[row_offset: row_offset + n_pts_row,
              col_offset: col_offset + n_pts_col] = content_sliced * ((-1) ** powers_predict[j + 1])
            col_offset += n_pts_col
        row_offset += n_pts_row

    return K


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





