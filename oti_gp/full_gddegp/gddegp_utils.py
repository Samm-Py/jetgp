import numpy as np
import pyoti.sparse as oti
from line_profiler import profile


def compute_dimension_differences(k, X1, X2, n1, n2, rays_X1, rays_X2, e_tags_1, e_tags_2):
    """
    Compute differences for a single dimension k.
    """
    n1, n2 = X1.shape[0], X2.shape[0]

    # Tag current dimension
    X1_k_tagged = X1[:, k] + e_tags_1 * rays_X1[k, :].T
    X2_k_tagged = X2[:, k] + e_tags_2 * rays_X2[k, :].T
    # Compute differences for this dimension
    diffs_k = oti.zeros((n1, n2))
    for i in range(n1):
        diffs_k[i, :] = X1_k_tagged[i, 0] - X2_k_tagged[:, 0].T

    return diffs_k


def differences_by_dim_func(X1, X2, rays_X1, rays_X2, n_order):
    """
    Compute dimension-wise differences with OTI tagging on both X1 and X2.
    Optimized version that pre-computes tags to avoid redundant calculations.
    """
    X1 = oti.array(X1)
    X2 = oti.array(X2)
    rays_X1 = oti.array(rays_X1)
    rays_X2 = oti.array(rays_X2)
    n1, d = X1.shape
    n2, _ = X2.shape

    m1 = 1  # directions per point in X1
    m2 = 1  # directions per point in X2

    # Pre-compute OTI basis elements (avoid recomputing in loops)
    e_tags_1 = oti.e(1, order=2 * n_order)
    e_tags_2 = oti.e(2, order=2 * n_order)

    if n_order == 0:
        e_tags_1 = oti.zero()
        e_tags_2 = oti.zero()

    differences_by_dim = []

    for k in range(d):
        diffs_k = compute_dimension_differences(
            k, X1, X2, n1, n2, rays_X1, rays_X2, e_tags_1, e_tags_2)

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


def build_first_block_row(phi, num_directions, max_order):
    """
    Build the first block row of a DD-GP kernel matrix.

    Parameters
    ----------
    phi : ndarray (n x n), hypercomplex
        Full kernel matrix with all tags embedded.
    num_directions : int
        Number of distinct directional tags.
    max_order : int
        Maximum derivative order to include.

    Returns
    -------
    row : ndarray
        Horizontally concatenated first row: K_ff | K_fd (order 1) | ... | K_fd (order R)
    """
    row = phi.real  # K_ff block

    for order in range(1, max_order + 1):
        deriv_block = phi.get_deriv([[2, order]])
        row = np.hstack((row, deriv_block))

    return row


def build_first_block_column(phi, num_directions, max_order):
    """
    Build the first block row of a DD-GP kernel matrix.

    Parameters
    ----------
    phi : ndarray (n x n), hypercomplex
        Full kernel matrix with all tags embedded.
    num_directions : int
        Number of distinct directional tags.
    max_order : int
        Maximum derivative order to include.

    Returns
    -------
    row : ndarray
        Horizontally concatenated first row: K_ff | K_fd (order 1) | ... | K_fd (order R)
    """
    column = phi.real  # K_ff block

    for order in range(1, max_order + 1):
        deriv_block = phi.get_deriv([[1, order]])
        column = np.vstack((column, deriv_block))

    return column


def build_K_dd_full(phi, num_directions, order_1, order_2):
    """
    Build the full (n * num_directions) x (n * num_directions) second derivative block matrix.

    Parameters
    ----------
    phi : ndarray of shape (n, n), with hypercomplex entries
        Kernel matrix with directional tagging.
    num_directions : int
        Number of directional directions (tags).

    Returns
    -------
    K_dd : ndarray of shape (n * num_directions, n * num_directions)
    """
    mixed_tag = [[1, order_1], [2, order_2]]
    # If the *array* object supports .get_deriv, this is one call:
    K_dd = phi.get_deriv(mixed_tag).copy()

    # If phi is a plain NumPy array of objects, fall back to:
    # K_dd = np.empty_like(phi, dtype=float)
    # for i in range(phi.shape[0]):
    #     for j in range(phi.shape[1]):
    #         K_dd[i, j] = phi[i, j].get_deriv(mixed_tag)

    # ------------------------------------------------------------------
    # 2) Replace diagonal entries with the special same-direction term
    #    (-1)^{order_1} · d^{order_1+order_2}/d(tag1)
    # ------------------------------------------------------------------
    # diag_len = min(phi.shape)          # works whether or not phi is square
    # diag_idx = np.arange(diag_len)

    # # Compute new diagonal values once, then bulk-assign
    # new_diag = [
    #     (-1) ** order_1
    #     * phi[int(k), int(k)].get_deriv([[1, order_1 + order_2]])
    #     for k in diag_idx
    # ]
    # K_dd[diag_idx, diag_idx] = new_diag

    return K_dd


@profile
def rbf_kernel(differences, length_scales, max_order, num_directions, kernel_func, index=-1, return_deriv=True):
    """
    Assemble the full DD-GP covariance matrix, blockâ€“row by blockâ€“row,
    strictly using the derivative calls and conventions you verified.

    Parameters
    ----------
    phi : (n Ã— n) hypercomplex ndarray
        Kernel matrix with all directional tags embedded.
    num_directions : int
    max_order      : int

    Returns
    -------
    K : ndarray   # square, size  n Â· (1 + num_directions Â· max_order)
    """

    # ---------- 0)  top block-row  ---------------------------------
    #      [ K_ff  |  K_fd(order=1) | K_fd(order=2) | ... ]
    # shape (n , n + n*T*max_order)

    phi = kernel_func(differences, length_scales, index)
    if max_order == 0:
        return phi.real
    if not return_deriv:
        K = build_first_block_column(phi, num_directions, max_order)
        return K

    K_top = build_first_block_row(phi, num_directions, max_order)
    block_rows = [K_top]

    # ---------- subsequent block-rows (order_1 = 1 .. max_order) ---
    for order_1 in range(1, max_order + 1):

        K_df = phi.get_deriv([[1, order_1]])      # (n,)

        # ---- (b)  concatenate all derivativeâ€“derivative blocks for this row
        dd_blocks = []
        for order_2 in range(1, max_order + 1):
            # uses your untouched helper
            dd_block = build_K_dd_full(phi, num_directions, order_1, order_2)
            # each (n*T , n*T)
            dd_blocks.append(dd_block)
        # (n*T , n*T*max_order)
        K_dd_row = np.hstack(dd_blocks)

        # ---- (c)  complete this block-row and append --------------
        # (n , n*T + n*T*max_order)
        full_row = np.hstack([K_df, K_dd_row])
        block_rows.append(full_row)

    # ---------- final v-stack --------------------------------------
    K_global = np.vstack(block_rows)
    return K_global
