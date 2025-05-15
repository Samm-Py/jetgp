import numpy as np
import pyoti.sparse as oti
from line_profiler import profile

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
            for j in range(n2):
                diffs_k[i, j] = (
                    X1[i, k]
                    + oti.e(k + 1, order=2 * n_order)
                    - (X2[j, k])
                )

        # Append to our list
        differences_by_dim.append(diffs_k)
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

    phi = kernel_func(differences, length_scales, index)

    for i in range(0, len(der_indices) + 1):
        row_j = 0
        for j in range(0, len(der_indices) + 1):
            if j == 0 and i == 0:
                row_j = phi.real * (-1)**(powers[j])
            elif j > 0 and i == 0:
                row_j = np.hstack(
                    (row_j,  (-1)**(powers[j]) * phi.get_deriv(der_indices[j - 1])))
            elif j == 0 and i > 0:
                row_j = phi.get_deriv(der_indices[i - 1])
            else:
                row_j = np.hstack(
                    (row_j, (-1)**(powers[j]) *
                     phi.get_deriv(der_indices[j - 1] + der_indices[i - 1]))
                )
        if i == 0:
            K = row_j
        else:
            K = np.vstack((K, row_j))

    return K
