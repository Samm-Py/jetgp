import numpy as np
from numpy.linalg import cholesky, solve
import utils as utils
from kernel_funcs.kernel_funcs import KernelFactory
from full_ddegp.optimizer import Optimizer
from full_gddegp import gddegp_utils
import pyoti.sparse as oti  # Hyper-complex AD


def assemble_global_cov(phi, num_directions, max_order):
    """
    Assemble the complete DD-GP covariance matrix using the three
    already-tested helpers you wrote earlier.

    ┌────────────┐
    │ K_ff  K_fd │   ← build_first_block_row(…)
    │ K_df  K_dd │   ← built below (uses build_K_fd_matrix + build_K_dd_full)
    └────────────┘
    """
    n = phi.shape[0]

    # ---------- 1) top block row  ----------
    K_top = build_first_block_row(
        phi, num_directions, max_order)   # shape (n , n + T·R)

    # ---------- 2) K_df  (all derivative-vs-function rows) ----------
    # one row for every (order,tag) pair
    K_df_rows = []
    for order in range(1, max_order + 1):
        for tag in range(1, num_directions + 1):
            # NOTE: keep exactly the derivative call you verified
            row_vec = (-phi[:, tag - 1].get_deriv([[1, order]])).T   # (1 , n)
            K_df_rows.append(row_vec)
    # shape (T·R·n_deriv_rows , n)
    K_df = np.vstack(K_df_rows)

    # ---------- 3) K_dd  (all derivative-vs-derivative blocks) ----------
    dd_row_blocks = []
    for ord1 in range(1, max_order + 1):
        dd_col_blocks = []
        for ord2 in range(1, max_order + 1):
            # uses your unmodified build_K_dd_full exactly once per (ord1, ord2)
            dd_block = build_K_dd_full(phi, num_directions, ord1, ord2)
            dd_col_blocks.append(dd_block)
        dd_row_blocks.append(np.hstack(dd_col_blocks))
    K_dd = np.vstack(dd_row_blocks)

    # ---------- 4) final assembly ----------
    top_width = K_top.shape[1]
    left_height = K_df.shape[0]

    upper = K_top                                   # ( n , n + T·R )
    lower = np.hstack([K_df, K_dd])                 # ([T·R·n] , same width)

    K_full = np.vstack([upper, lower])              # square matrix

    return K_full


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
        for tag in range(1, num_directions + 1):
            deriv_block = phi[:, tag-1].get_deriv([[1, order]])
            row = np.hstack((row, deriv_block))

    return row


def build_K_fd_matrix(phi, num_directions, max_order, order_1):
    """
    Build the full K_fd block matrix:
        Rows = all (tag, order) pairs (∂^order f / ∂v_tag^order)
        Columns = function values

    Parameters
    ----------
    phi : ndarray (n x n), hypercomplex
        Kernel matrix with directional tagging.
    num_directions : int
        Number of directional tags.
    max_order : int
        Highest derivative order.

    Returns
    -------
    K_fd : ndarray (n * num_directions * max_order, n)
    """
    n = phi.shape[0]
    K_fd_rows = []

    for tag in range(1, num_directions + 1):
        block = (-1)**(order_1) * \
            phi[:, tag - 1].get_deriv([[1, order_1]])  # shape: (n,)
        block = block.reshape(-1, 1)  # ensure 2D
        K_fd_rows.append(block)

    # Stack all blocks vertically to get (n * num_directions * max_order, 1)
    K_fd = np.hstack(K_fd_rows)
    return K_fd


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
    n = phi.shape[0]
    K_dd = []

    for i in range(n):
        row = []
        for j in range(n):
            if i == j:
                val = (-1)**(order_1 + order_2) * \
                    phi[i, j].get_deriv([[1, order_1 + order_2]])
            else:
                val = (-1)**(order_2) * phi[i, j].get_deriv(
                    [[i + 1, order_1]] + [[j + 1, order_2]])
            row.append(val)
        K_dd.append(row)

    return np.array(K_dd)


def assemble_ddgp_cov(phi, num_directions, max_order):
    """
    Assemble the full DD-GP covariance matrix, block–row by block–row,
    strictly using the derivative calls and conventions you verified.

    Parameters
    ----------
    phi : (n × n) hypercomplex ndarray
        Kernel matrix with all directional tags embedded.
    num_directions : int
    max_order      : int

    Returns
    -------
    K : ndarray   # square, size  n · (1 + num_directions · max_order)
    """
    n = phi.shape[0]

    # ---------- 0)  top block-row  ---------------------------------
    #      [ K_ff  |  K_fd(order=1) | K_fd(order=2) | ... ]
    # shape (n , n + n*T*max_order)
    K_top = build_first_block_row(phi, num_directions, max_order)
    block_rows = [K_top]

    # ---------- subsequent block-rows (order_1 = 1 .. max_order) ---
    for order_1 in range(1, max_order + 1):

        # ---- (a)  K_df  =  rows of −∂^{order_1}f / ∂v_tag^{order_1} ----
        K_df_cols = []
        for tag in range(1, num_directions + 1):
            # *** keep this call exactly as confirmed ***
            col = (-1)**(order_1)*phi[:, tag -
                                      1].get_deriv([[1, order_1]])      # (n,)
            # make it (n,1)
            K_df_cols.append(col.reshape(-1, 1))
        # (n , n_directions)
        K_df = np.hstack(K_df_cols)

        # ---- (b)  concatenate all derivative–derivative blocks for this row
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


def kernel_func(differences_by_dim, length_scales=np.array([1, 1, 1]), index=-1):
    """
    Anisotropic Squared Exponential (SE) kernel.

    Parameters
    ----------
    differences_by_dim : list of ndarray
    length_scales : list
    index : int, optional

    Returns
    -------
    ndarray
    """
    dim = 2
    ell = (length_scales[:-1])
    sigma_f = length_scales[-1]
    sqdist = sum((ell[i] * ell[i] * differences_by_dim[i]*differences_by_dim[i])
                 for i in range(dim))
    return (sigma_f) ** 2 * oti.exp(-0.5 * sqdist)


def generate_rays(order, num_points):
    """Generate unit vectors (rays) and their hypercomplex perturbations."""
    thetas_list = [[np.pi/6], [np.pi/3]]
    rays_list = []
    perts_list = []
    for idx, thetas in enumerate(thetas_list):
        rays = np.column_stack([[np.cos(t), np.sin(t)] for t in thetas])
        e = [oti.e(i + 1, order=order) for i in range(rays.shape[1])]
        perts = np.dot(rays, e)
        rays_list.append(rays)
        perts_list.append(perts)
    return rays_list, perts_list


rays_list, perts = generate_rays(2, 2)

x_train = np.array([[0, 0], [1, 1]])
n_order = 2
differences_by_dim = gddegp_utils.differences_by_dim_func(
    x_train, x_train, rays_list, rays_list, n_order)


phi = kernel_func(differences_by_dim)

# row_j = build_first_block_row(phi, 2, 2)
# row_k = build_K_fd_matrix(phi, 2, n_order)
# row_i = build_K_dd_full(phi, 2, 1, 1)
# row_i = build_K_dd_full(phi, 2, 1, 2)


K = assemble_ddgp_cov(phi, 2, n_order)
print("Global covariance matrix shape:", K.shape)
