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


def differences_by_dim_func(X1, X2, rays, n_order, index=-1):
    """
    Compute dimension-wise pairwise differences between X1 and X2,
    including hypercomplex perturbations in the directions specified by `rays`.

    This optimized version pre-calculates the perturbation and uses a single
    efficient loop for subtraction, avoiding broadcasting issues with OTI arrays.

    Parameters
    ----------
    X1 : ndarray of shape (n1, d)
    X2 : ndarray of shape (n2, d)
    rays : ndarray of shape (d, n_rays)
    n_order : int
    index : list or int, optional

    Returns
    -------
    differences_by_dim : list of ndarray
    """
    X1 = oti.array(X1)
    X2 = oti.array(X2)

    n1, d = X1.shape
    n2, _ = X2.shape
    n_rays = rays.shape[1]

    # --- OPTIMIZATION 1: Pre-calculate the perturbation vector ---
    e_bases = [oti.e(i + 1, order=2 * n_order) for i in range(n_rays)]
    perts = np.dot(rays, e_bases)

    differences_by_dim = []

    for k in range(d):
        # Add the pre-calculated perturbation for the current dimension to all points in X1
        X1_k_tagged = (X1[:, k] + perts[k])
        X2_k = X2[:, k]

        # Pre-allocate the result matrix for this dimension
        diffs_k = oti.zeros((n1, n2))

        # --- OPTIMIZATION 2: Use an efficient single loop for subtraction ---
        # This avoids broadcasting and is much faster than a double loop.
        for i in range(n1):
            diffs_k[i, :] = X1_k_tagged[i, 0] - X2_k[:, 0].T

        differences_by_dim.append(diffs_k)

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


@profile
def rbf_kernel(
    differences,
    length_scales,
    n_order,
    n_bases,
    kernel_func,
    der_indices,
    powers,
    index=-1,
):
    """
    Assembles the full DD-GP covariance matrix using an efficient, pre-computed
    derivative array and block-wise matrix filling.

    Parameters
    ----------
    (Parameters are the same as your original function, with n_bases added)

    Returns
    -------
    K : ndarray
        Full kernel matrix with function values and derivative blocks.
    """
    # Get the pyoti derivative helper for multiplying derivative indices
    dh = coti.get_dHelp()

    # 1. Evaluate the kernel once to get the hypercomplex result
    phi = kernel_func(differences, length_scales, index)

    # 2. Extract ALL derivative components into a single flat array (highly efficient)
    phi_exp = phi.get_all_derivs(n_bases, 2 * n_order)

    # 3. Create maps to translate derivative specifications to flat indices
    der_map = deriv_map(n_bases, 2 * n_order)
    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)

    # 4. Pre-allocate the full covariance matrix
    n_blocks = len(der_indices) + 1
    block_rows, block_cols = phi.shape
    K = np.zeros((n_blocks * block_rows, n_blocks * block_cols))

    # 5. Fill the matrix block by block using the pre-computed derivative array
    for i in range(n_blocks):  # Block-row index
        for j in range(n_blocks):  # Block-column index

            # Get a view of the current block
            K_block = K[i*block_rows:(i+1)*block_rows,
                        j*block_cols:(j+1)*block_cols]
            sign_j = (-1)**(powers[j])

            if i == 0 and j == 0:
                # K_ff: Access the real part (index 0 of the flat array)
                idx = 0
            elif i == 0 and j > 0:
                # K_fd: Access the j-th derivative using its flat index
                idx = der_indices_tr[j-1]
            elif i > 0 and j == 0:
                # K_df: Access the i-th derivative using its flat index
                idx = der_indices_tr[i-1]
            else:
                # K_dd: Multiply the derivative indices to find the correct flat index
                imdir1 = der_ind_order[j-1]
                imdir2 = der_ind_order[i-1]
                new_idx, new_ord = dh.mult_dir(
                    imdir1[0], imdir1[1], imdir2[0], imdir2[1])
                idx = der_map[new_ord][new_idx]

            # Assign content from the flat array using the calculated index
            K_block[:, :] = sign_j * phi_exp[idx]

    return K
