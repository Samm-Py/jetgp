import numpy as np
import pyoti.sparse as oti


def differences_by_dim_func(X1, X2, n_order, index=-1):
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
                if i in index or j in index:
                    diffs_k[i, j] = (
                        (X1[i, k] + oti.e(k + 1, order=2 * n_order))
                    ) - (X2[j, k])
                else:
                    diffs_k[i, j] = ((X1[i, k])) - (X2[j, k])
        # Append to our list
        differences_by_dim.append(diffs_k)
    return differences_by_dim


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

    phi = kernel_func(differences, length_scales, index)

    for i in range(0, len(der_indices) + 1):
        row_j = 0
        for j in range(0, len(der_indices) + 1):
            if j == 0 and i == 0:
                row_j = phi.real * (-1)**(powers[j])
            elif j > 0 and i == 0:
                row_j = np.hstack(
                    (
                        row_j,
                        (-1)**(powers[j]) * phi[:, index[0]: index[-1] + 1].get_deriv(
                            der_indices[j - 1]
                        ),
                    )
                )
            elif j == 0 and i > 0:
                row_j = phi[index[0]: index[-1] + 1, :].get_deriv(
                    der_indices[i-1]
                )
            else:
                row_j = np.hstack(
                    (
                        row_j,
                        (-1)**(powers[j]) * np.array(
                            phi[
                                index[0]: index[-1] + 1,
                                index[0]: index[-1] + 1,
                            ].get_deriv(der_indices[j - 1] + der_indices[i - 1])
                        ),
                    )
                )
        if i == 0:
            K = row_j
        else:
            K = np.vstack((K, row_j))

    return K


def determine_weights(
    diffs_by_dim,
    diffs_test,
    length_scales,
    kernel_func,
):
    """
    ARD RBF kernel for multi-dimensional inputs:
      k(x, x') = sigma_f^2 * exp( -0.5 * sum_{d}((x_d - x'_d)^2 / ell_d^2) ).

    Parameters
    ----------
    X1 : array of shape (N, D)
    X2 : array of shape (M, D)
    length_scales : array of shape (D,), each dimension's length scale
    sigma_f : float (signal amplitude)

    Returns
    -------
    K : array of shape (N, M)


    """
    # Allocate the output array

    n1 = diffs_test[0].shape[0]

    index = [-1]

    phi = kernel_func(diffs_by_dim, length_scales, index)
    r = kernel_func(diffs_test, length_scales, index)

    K = phi.real
    F = np.ones((n1, 1))
    r = r.real[:, 0].reshape(-1, 1)
    r = np.vstack((r, [1]))

    # Construct the augmented matrix of size (n+1) x (n+1)
    M = np.zeros((n1 + 1, n1 + 1))
    M[:n1, :n1] = K  # top-left block is R
    M[:n1, n1] = F.flatten()  # top-right block is F (as a column)
    M[n1, :n1] = F.flatten()  # bottom-left block is F^T
    M[n1, n1] = 0  # bottom-right block is 0

    # Solve the system for [w; mu]
    solution = np.linalg.solve(M, r)

    # Extract w and mu from the solution vector
    w = solution[:n1]  # w is the first n entries (n x 1)

    return w
