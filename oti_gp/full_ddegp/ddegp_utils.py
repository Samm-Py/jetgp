import numpy as np
import pyoti.sparse as oti


def differences_by_dim_func(X1, X2, rays, n_order, index=-1):
    X1 = oti.array(X1)
    X2 = oti.array(X2)

    n1, d = X1.shape
    n2, d = X2.shape
    n_rays = rays.shape[1]

    # Prepare the output: a list of d arrays, each of shape (n, m)
    differences_by_dim = []

    # Loop over each dimension k
    for k in range(d):
        # Create an empty (n, m) array for this dimension
        diffs_k = oti.zeros((n1, n2))

        # Nested loops to fill diffs_k
        for i in range(n1):
            for j in range(n2):
                dire1 = 0
                for l in range(n_rays):
                    dire1 = (
                        dire1
                        + oti.e(l + 1, order=2 * n_order)
                        * rays[k, l]
                    )
                diffs_k[i, j] = ((X1[i, k] + dire1)) - (X2[j, k])

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
