import numpy as np
from matplotlib import pyplot as plt
import pyoti.sparse as oti
from scipy.linalg import lu


def squared_exponential_kernel(x, theta):
    """Compute squared exponential kernel matrix with length scale theta."""
    x = np.atleast_2d(x).T
    sq_dist = (x - x.T) ** 2
    K = np.exp(-0.5 * sq_dist * theta**2)
    return K


def dK_dtheta(x, theta):
    """Compute analytical derivative of the squared exponential kernel with respect to theta."""
    x = np.atleast_2d(x).T
    sq_dist = (x - x.T) ** 2
    dK = -0.5 * (sq_dist * 2 * theta) * np.exp(-0.5 * sq_dist * theta**2)
    return dK


def analytic_derivative_of_det(x, theta):
    """Compute derivative of det(K) using the analytic trace identity."""
    K = squared_exponential_kernel(x, theta)
    dK = dK_dtheta(x, theta)
    K_inv = np.linalg.inv(K)
    trace_term = np.trace(K_inv @ dK)
    return trace_term


# Also include the complex-step version from before
def squared_exponential_kernel_param(x, theta, epsilon=1e-10, jitter=0.0):
    theta = theta + 1j * epsilon
    x = np.atleast_2d(x).T
    sq_dist = (x - x.T) ** 2
    K = np.exp(-0.5 * sq_dist * theta**2)
    K += jitter * np.eye(len(x))
    return K


def det_derivative_complex_step(x, theta):
    K = squared_exponential_kernel_param(x, theta)
    det_val = np.log(np.linalg.det(K))
    deriv = np.imag(det_val) / 1e-10
    return det_val.real, deriv


def lu_decomposition(A):
    n = A.shape[0]
    L = oti.zeros(A.shape)
    U = oti.zeros(A.shape)

    for i in range(n):
        # Upper triangular matrix U
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))

        # Lower triangular matrix L
        L[i, i] = 1
        for k in range(i + 1, n):
            L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[
                i, i
            ]

    return L, U


def diagonal_product(A):
    n = A.shape[0]
    product = 1
    for i in range(n):
        product *= A[i, i]
    return product


def ker_func_oti(X1, X2, length_scales, n_order, index=-1):
    X1 = oti.array(X1)
    X2 = oti.array(X2)

    n1, d = X1.shape

    n2, d = X2.shape

    ell = length_scales[0:-1]
    sigma_f = length_scales[-1]

    # Prepare the output: a list of d arrays, each of shape (n, m)
    differences_by_dim = []

    # Loop over each dimension k
    for k in range(d):
        # Create an empty (n, m) array for this dimension
        diffs_k = oti.zeros((n1, n2))

        # Nested loops to fill diffs_k
        for i in range(n1):
            for j in range(n2):
                diffs_k[i, j] = X1[i, k] - (X2[j, k])

        # Append to our list
        differences_by_dim.append(diffs_k)

    # Distances scaled by each dimension's length scale
    sqdist = 0
    for i in range(d):
        sqdist = sqdist + (ell[i] * (differences_by_dim[i])) ** 2

    return sigma_f**2 * oti.exp(-0.5 * sqdist)


# Evaluate and compare
x = np.linspace(-1, 1, 10)

ders_ana = []
ders_complex = []
ders_oti_ana = []
ders_oti = []
thetas = np.linspace(1.0, 10, 100)


for theta in thetas:
    K = ker_func_oti(x, x, [theta + oti.e(1, order=1), 1], 1)
    L, U = lu_decomposition(K)
    p2 = diagonal_product(U)
    log_det = oti.log(p2)
    d_K_d_theta_oti = log_det.get_deriv([[1, 1]])
    ders_oti.append(d_K_d_theta_oti)
    K_inv = np.linalg.inv(K.real)
    dk_dtheta = K.get_deriv([[1, 1]])
    mat = K_inv @ dk_dtheta
    der_oti_ana = np.trace(mat)
    ders_oti_ana.append(der_oti_ana)
    d_det_analytic = analytic_derivative_of_det(x, theta)
    det_val_complex, d_det_complex = det_derivative_complex_step(x, theta)

    d_det_analytic, det_val_complex, d_det_complex
    ders_ana.append(d_det_analytic)
    ders_complex.append(d_det_complex)


plt.plot(thetas, ders_ana)
plt.plot(thetas, ders_oti_ana)
plt.plot(thetas, ders_oti)
