import numpy as np
import pyoti.sparse as oti


def differences_by_dim_func(X1, X2, rays, n_order, index=-1):
    """
    Compute dimension-wise pairwise differences between X1 and X2,
    including hypercomplex perturbations in the directions specified by `rays`.

    Parameters
    ----------
    X1 : ndarray of shape (n1, d)
        First input array.
    X2 : ndarray of shape (n2, d)
        Second input array.
    rays : ndarray of shape (d, n_rays)
        Direction vectors applied for each dimension using hypercomplex algebra.
    n_order : int
        Derivative order used to define the hypercomplex perturbation.
    index : list or int, optional
        Index to determine which points receive tagging. Default is -1 (all tagged).

    Returns
    -------
    differences_by_dim : list of ndarray
        List of length d. Each entry is an (n1, n2) array of directional differences for one dimension.
    """
    X1 = oti.array(X1)
    X2 = oti.array(X2)

    n1, d = X1.shape
    n2, _ = X2.shape
    n_rays = rays.shape[1]

    differences_by_dim = []

    for k in range(d):
        diffs_k = oti.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                dire1 = sum(oti.e(l + 1, order=2 * n_order) *
                            rays[k, l] for l in range(n_rays))
                diffs_k[i, j] = (X1[i, k] + dire1) - X2[j, k]
        differences_by_dim.append(diffs_k)
    return differences_by_dim


def rbf_kernel(differences, length_scales, n_order, kernel_func, der_indices, powers, index=-1):
    """
    Compute a radial basis function (RBF) kernel matrix with hypercomplex derivative augmentation.

    Parameters
    ----------
    differences : list of ndarray
        List of difference arrays for each input dimension.
    length_scales : ndarray
        Log-scaled length scale parameters.
    n_order : int
        Maximum derivative order.
    kernel_func : callable
        Kernel function to be used for evaluating pairwise differences.
    der_indices : list of lists
        Multi-index derivatives specifying derivative evaluation directions.
    powers : list of int
        Parity powers used to scale contributions.
    index : list or int, optional
        Index for submodel selection or evaluation region.

    Returns
    -------
    K : ndarray
        Full kernel matrix with function values and derivative blocks.
    """
    phi = kernel_func(differences, length_scales, index)

    for i in range(0, len(der_indices) + 1):
        row_j = 0
        for j in range(0, len(der_indices) + 1):
            if j == 0 and i == 0:
                row_j = phi.real * (-1) ** powers[j]
            elif j > 0 and i == 0:
                row_j = np.hstack(
                    (row_j, (-1) ** powers[j] * phi.get_deriv(der_indices[j - 1])))
            elif j == 0 and i > 0:
                row_j = phi.get_deriv(der_indices[i - 1])
            else:
                row_j = np.hstack((
                    row_j,
                    (-1) ** powers[j] *
                    phi.get_deriv(der_indices[j - 1] + der_indices[i - 1])
                ))
        if i == 0:
            K = row_j
        else:
            K = np.vstack((K, row_j))

    return K
