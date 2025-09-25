import numpy as np
import pyoti.sparse as oti
from line_profiler import profile


def differences_by_dim_func(X1, X2, n_order, return_deriv=True, index=-1):
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
    if n_order == 0:
        for k in range(d):
            diffs_k = oti.zeros((n1, n2))
            for i in range(n1):
                diffs_k[i, :] = (
                    X1[i, k]
                    - (X2[:, k].T)
                )
            differences_by_dim.append(diffs_k)
    elif not return_deriv:

        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = oti.zeros((n1, n2))

            # Nested loops to fill diffs_k
            for i in range(n1):
                diffs_k[i, :] = (
                    X1[i, k]
                    + oti.e(k + 1, order=n_order)
                    - (X2[:, k].T)
                )

            # Append to our list
            differences_by_dim.append(diffs_k)
    else:
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = oti.zeros((n1, n2))

            # Nested loops to fill diffs_k
            for i in range(n1):
                diffs_k[i, :] = (
                    X1[i, k]
                    - (X2[:, k].T)
                )

            # Append to our list
            differences_by_dim.append(diffs_k+ oti.e(k + 1, order=2*n_order))
    return differences_by_dim

def differences_by_dim_func_SI(X1, X2, n_order, return_deriv=True, index=-1):
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
    #X1 = oti.array(X1)
    #X2 = oti.array(X2)
    n1, d = X1.shape

    n2, d = X2.shape

    # Prepare the output: a list of d arrays, each of shape (n, m)
    differences_by_dim = []

    # Loop over each dimension k
    if n_order == 0:
        for k in range(d):
            diffs_k = np.zeros((n1, n2))
            diffs_k[:, :] = (
                X1[:, k].reshape(-1,1)
                - (X2[:, k].reshape(1,-1))
            )%1
            differences_by_dim.append(oti.array(diffs_k))
    elif not return_deriv:

        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = np.zeros((n1, n2))
            # Nested loops to fill diffs_k
            for i in range(n1):
                diffs_k[:, :] = (
                    X1[:, k].reshape(-1,1)
                    - (X2[:, k].reshape(1,-1))
                )%1

            # Append to our list
            differences_by_dim.append(oti.array(diffs_k) + oti.e(1, order = n_order))
    else:
        for k in range(d):
            # Create an empty (n, m) array for this dimension
            diffs_k = np.zeros((n1, n2))
            # Nested loops to fill diffs_k
            diffs_k[:, :] = (
                    X1[:, k].reshape(-1,1)
                    - (X2[:, k].reshape(1,-1))
                )%1

            # Append to our list
            differences_by_dim.append(oti.array(diffs_k)+ oti.e(k + 1, order=2*n_order))
    return differences_by_dim


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
    return_deriv=True
):
    """
    Compute the derivative-enhanced Radial Basis Function (RBF) kernel matrix 
    with Automatic Relevance Determination (ARD) for multi-dimensional inputs.

    The kernel supports function values and mixed **partial derivatives** up to a specified order.

    General form of the ARD RBF kernel:
    k(x, x') = exp(-0.5 * sum_d((x_d - x'_d)^2 / length_scale_d^2))

    Parameters:
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
        Must return an object with:
        - `.real`: the base kernel.
        - `.get_deriv(multi_index)`: derivatives of the kernel.
    der_indices : list of lists
        Multi-index derivative structures for each derivative component.
    powers : list of int
        Powers of (-1) applied to each term (for symmetry or sign conventions).
    index : int, optional (default=-1)
        Index for selecting directional kernels or subspaces (if applicable).

    Returns:
    -------
    K : ndarray of shape (n_total, n_total)
        Kernel matrix including function values and derivative terms.

    Notes:
    -----
    - The kernel matrix is built by stacking:
      - Function value blocks.
      - Derivative blocks (including mixed partial derivatives).
    - The sign convention for derivatives is controlled by `powers`.

    Example:
    --------
    >>> K = rbf_kernel(differences, length_scales, n_order, n_bases, kernel_func, der_indices, powers)
    >>> K.shape == (n_bases, n_bases)
    """
    import pyoti.core as coti
    dh = coti.get_dHelp()

    phi = kernel_func(differences, length_scales, index)
    PHIrows = phi.shape[0]
    PHIcols = phi.shape[1]
    nderivs = len(der_indices)

    if n_order == 0:
        return phi.real
    elif not return_deriv:
        phi_exp = phi.get_all_derivs(n_bases, n_order)
        # print(phi_exp.shape)
        der_map = deriv_map(n_bases, n_order)
        K = np.zeros((PHIrows * (nderivs + 1), PHIcols))
        outer_loop_index = 1
    else:
        phi_exp = phi.get_all_derivs(n_bases, 2*n_order)
        # print(phi_exp.shape)
        der_map = deriv_map(n_bases, 2*n_order)
        K = np.zeros((PHIrows * (nderivs + 1), PHIcols * (nderivs + 1)))
        outer_loop_index = len(der_indices) + 1
    # print(der_map)
    # for i in range(0, len(der_indices) ):
    #     print(der_indices[i])
    der_indices_tr, der_ind_order = transform_der_indices(der_indices, der_map)
    # print("")
    # for i in range(0, len(der_indices_tr) ):
    #     print(der_indices[i] , der_indices_tr[i])
    # print("")

    # TODO: Preallocate matrix (and indices) to optimize matrix generation:

    # print(f'len(der_indices): ({len(der_indices)})')
    # print(f'Shape before: ({phi.shape})')

    for j in range(0, outer_loop_index):
        signj = (-1)**(powers[j])
        for i in range(0, len(der_indices) + 1):
            # Get local view of the global array.
            Klocal = K[i*PHIrows: (i+1)*PHIrows, j*PHIcols: (j+1)*PHIcols]
            if j == 0 and i == 0:
                Klocal[:, :] = phi_exp[0] * signj
            elif j > 0 and i == 0:
                Klocal[:, :] = signj * phi_exp[der_indices_tr[j-1]]
            elif j == 0 and i > 0:
                Klocal[:, :] = phi_exp[der_indices_tr[i-1]]
            else:

                imdir1 = der_ind_order[j-1]
                imdir2 = der_ind_order[i-1]
                idx, ord = dh.mult_dir(
                    imdir1[0], imdir1[1], imdir2[0], imdir2[1])
                idx = der_map[ord][idx]

                Klocal[:, :] = signj * phi_exp[idx]

    # print(f'Shape final: ({K.shape})')
    # print(f'Shape final factors: ')
    # print(f' -> nrows: {K.shape[0]/phi.shape[0]}x')
    # print(f' -> ncols: {K.shape[1]/phi.shape[1]}x')
    # print(80*'-')

    return K


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
