import numpy as np
import pyoti.sparse as oti


def differences_by_dim_func(X1, X2, rays, n_order, index=-1, index_list=[]):
    """
    Compute directional differences between input arrays X1 and X2 along specified rays.

    Parameters
    ----------
    X1 : ndarray
        Array of shape (n_samples1, n_features) representing the first set of input points.
    X2 : ndarray
        Array of shape (n_samples2, n_features) representing the second set of input points.
    rays : ndarray
        Array of shape (n_features, n_rays) representing direction vectors.
    n_order : int
        Maximum derivative order for hypercomplex expansion.
    index : int, default=-1
        Index of the current submodel (used to select directional rays specific to this submodel).
    index_list : list, default=[]
        List of indices corresponding to the training points included in the current submodel.

    Returns
    -------
    differences_by_dim : list of ndarray
        List containing the directional differences for each feature dimension.
    """
    X1 = oti.array(X1)
    X2 = oti.array(X2)
    n1, d = X1.shape
    n2, d = X2.shape

    differences_by_dim = []
    rays = rays[index]

    for k in range(d):
        diffs_k = oti.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                dire1 = 0
                for l in range(rays.shape[1]):
                    dire1 += oti.e(l + 1, order=2 * n_order) * rays[k, l]
                if i in index_list or j in index_list:
                    diffs_k[i, j] = (X1[i, k] + dire1) - X2[j, k]
                else:
                    diffs_k[i, j] = X1[i, k] - X2[j, k]
        differences_by_dim.append(diffs_k)

    return differences_by_dim


def rbf_kernel(differences, length_scales, n_order, n_bases, kernel_func, der_indices, powers, index=-1, index_list=[]):
    """
    Construct the kernel matrix for the weighted directional derivative-enhanced GP.

    Parameters
    ----------
    differences : list of ndarray
        Precomputed pairwise differences by dimension.
    length_scales : ndarray
        Array of length scales (hyperparameters).
    n_order : int
        Maximum derivative order.
    n_bases : int
        Number of OTI bases.
    kernel_func : callable
        Kernel function evaluated over differences.
    der_indices : list
        Indices indicating derivative orders.
    powers : list
        Powers of each term based on OTI basis structure.
    index : int, default=-1
        Dummy argument for compatibility.
    index_list : list, default=[]
        List identifying index subsets for directional derivatives.

    Returns
    -------
    K : ndarray
        Fully assembled kernel matrix including derivatives.
    """
    phi = kernel_func(differences, length_scales, index)

    for i in range(0, len(der_indices) + 1):
        row_j = 0
        for j in range(0, len(der_indices) + 1):
            if j == 0 and i == 0:
                row_j = phi.real * (-1)**(powers[j])
            elif j > 0 and i == 0:
                row_j = np.hstack((
                    row_j,
                    (-1)**(powers[j]) * phi[:, index[0]: index[-1] + 1].get_deriv(der_indices[j - 1]),
                ))
            elif j == 0 and i > 0:
                row_j = phi[index[0]: index[-1] + 1,
                            :].get_deriv(der_indices[i - 1])
            else:
                row_j = np.hstack((
                    row_j,
                    (-1)**(powers[j]) * np.array(
                        phi[index[0]: index[-1] + 1, index[0]: index[-1] + 1].get_deriv(
                            der_indices[j - 1] + der_indices[i - 1]
                        )
                    ),
                ))
        if i == 0:
            K = row_j
        else:
            K = np.vstack((K, row_j))

    return K


def determine_weights(diffs_by_dim, diffs_test, length_scales, kernel_func):
    """
    Solve for Kriging weights using a Weighted Coefficient Kriging model.

    This method solves an augmented linear system that enforces unbiasedness
    and ensures the interpolation is consistent with the weighted directional
    derivative-enhanced GP structure.

    Parameters
    ----------
    diffs_by_dim : list of ndarray
        Differences between training points.
    diffs_test : list of ndarray
        Differences between training points and test point.
    length_scales : ndarray
        Kernel length scales.
    kernel_func : callable
        Kernel function evaluated over pairwise differences.

    Returns
    -------
    w : ndarray
        Optimal weights vector for Weighted Coefficient Kriging interpolation.
    """
    n1 = diffs_test[0].shape[0]

    index = [-1]

    phi = kernel_func(diffs_by_dim, length_scales, index)
    r = kernel_func(diffs_test, length_scales, index)

    K = phi.real
    F = np.ones((n1, 1))
    r = r.real[:, 0].reshape(-1, 1)
    r = np.vstack((r, [1]))

    M = np.zeros((n1 + 1, n1 + 1))
    M[:n1, :n1] = K
    M[:n1, n1] = F.flatten()
    M[n1, :n1] = F.flatten()
    M[n1, n1] = 0

    solution = np.linalg.solve(M, r)

    w = solution[:n1]

    return w
