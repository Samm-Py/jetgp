import numpy as np
import pyoti.sparse as oti
from line_profiler import profile

def differences_by_dim_func(X1, X2, n_order, index=-1):
    """
    Compute dimension-wise differences between two sets of points X1 and X2,
    introducing hypercomplex perturbation for selected indices.

    Parameters
    ----------
    X1 : ndarray of shape (n1, d)
        First input array.
    X2 : ndarray of shape (n2, d)
        Second input array.
    n_order : int
        Derivative order used to set the hypercomplex type.
    index : list or int, optional
        Indices in X1 or X2 to apply hypercomplex tagging (default is -1, no tagging).

    Returns
    -------
    differences_by_dim : list of ndarray
        List of length d. Each entry is an (n1, n2) array of differences for a single dimension.
    """
    X1 = oti.array(X1)
    X2 = oti.array(X2)
    n1, d = X1.shape
    n2, _ = X2.shape
    differences_by_dim = []

    for k in range(d):
        diffs_k = oti.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                if i in index or j in index:
                    diffs_k[i, j] = (
                        X1[i, k] + oti.e(k + 1, order=2 * n_order)) - X2[j, k]
                else:
                    diffs_k[i, j] = X1[i, k] - X2[j, k]
        differences_by_dim.append(diffs_k)
    return differences_by_dim

@profile
def rbf_kernel(differences, length_scales, n_order, n_bases, kernel_func, der_indices, powers, index=-1):
    """
    Constructs the RBF kernel matrix with derivative entries using hypercomplex representation.

    Parameters
    ----------
    differences : list of ndarrays
        Differences between training points by dimension.
    length_scales : array-like
        Kernel hyperparameters in log10 space.
    n_order : int
        Maximum derivative order.
    n_bases : int
        Number of OTI basis terms used in the analysis.
    kernel_func : callable
        Base kernel function (e.g., SE, RQ).
    der_indices : list of lists
        Derivative multi-indices.
    powers : list of ints
        Parity powers for each derivative term.
    index : list of int
        Indices used for slicing derivative blocks.

    Returns
    -------
    K : ndarray
        Full RBF kernel matrix with mixed function and derivative entries.
    """
    phi = kernel_func(differences, length_scales, index)

    # Preallocate K
    
    

    for i in range(0, len(der_indices) + 1):
        row_j = 0
        for j in range(0, len(der_indices) + 1):
            if j == 0 and i == 0:
                row_j = phi.real * (-1)**(powers[j])
            elif j > 0 and i == 0:
                row_j = np.hstack((
                    row_j,
                    (-1)**(powers[j]) * phi[:, index[0]: index[-1] + 1].get_deriv(der_indices[j - 1])
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
                    )
                ))
        if i == 0:
            K = row_j
        else:
            K = np.vstack((K, row_j))
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
