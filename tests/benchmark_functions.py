"""
Benchmark test functions and their analytical gradients.

Functions from:
    Erickson, C.B., Ankenman, B.E., Sanchez, S.M. (2018).
    "Comparison of Gaussian process modeling software."
    European Journal of Operational Research, 266(1), 179-192.

All functions accept inputs in [0, 1]^d and internally scale
to the original domain. Gradients are computed analytically
and returned with respect to the [0, 1]^d inputs.
"""

import numpy as np
from scipy.stats.qmc import LatinHypercube


# ============================================================================
# Borehole function (8D)
# ============================================================================

# Original domain bounds
_BOREHOLE_BOUNDS = np.array([
    [0.05,    0.15],     # rw: radius of borehole (m)
    [100.0,   50000.0],  # r:  radius of influence (m)
    [63070.0, 115600.0], # Tu: transmissivity of upper aquifer (m^2/yr)
    [990.0,   1110.0],   # Hu: potentiometric head of upper aquifer (m)
    [63.1,    116.0],    # Tl: transmissivity of lower aquifer (m^2/yr)
    [700.0,   820.0],    # Hl: potentiometric head of lower aquifer (m)
    [1120.0,  1680.0],   # L:  length of borehole (m)
    [9855.0,  12045.0],  # Kw: hydraulic conductivity of borehole (m/yr)
])


def _scale_to_original(X_unit, bounds):
    """Scale from [0,1]^d to original domain."""
    lb = bounds[:, 0]
    ub = bounds[:, 1]
    return lb + X_unit * (ub - lb)


def borehole(X_unit):
    """
    Borehole function.

    Parameters
    ----------
    X_unit : ndarray of shape (n, 8)
        Input points in [0, 1]^8.

    Returns
    -------
    y : ndarray of shape (n,)
        Function values.
    """
    X = _scale_to_original(X_unit, _BOREHOLE_BOUNDS)
    rw = X[:, 0]
    r  = X[:, 1]
    Tu = X[:, 2]
    Hu = X[:, 3]
    Tl = X[:, 4]
    Hl = X[:, 5]
    L  = X[:, 6]
    Kw = X[:, 7]

    log_r_rw = np.log(r / rw)
    numer = 2 * np.pi * Tu * (Hu - Hl)
    denom = log_r_rw * (1.0 + (2 * L * Tu) / (log_r_rw * rw**2 * Kw) + Tu / Tl)
    y = numer / denom
    return y


def borehole_gradient(X_unit):
    """
    Analytical gradient of the borehole function w.r.t. [0,1]^8 inputs.

    Parameters
    ----------
    X_unit : ndarray of shape (n, 8)
        Input points in [0, 1]^8.

    Returns
    -------
    grad : ndarray of shape (n, 8)
        Gradient w.r.t. the unit-cube inputs.
    """
    bounds = _BOREHOLE_BOUNDS
    scale = bounds[:, 1] - bounds[:, 0]
    X = _scale_to_original(X_unit, bounds)

    rw = X[:, 0]
    r  = X[:, 1]
    Tu = X[:, 2]
    Hu = X[:, 3]
    Tl = X[:, 4]
    Hl = X[:, 5]
    L  = X[:, 6]
    Kw = X[:, 7]

    log_r_rw = np.log(r / rw)
    A = 2 * np.pi * Tu * (Hu - Hl)
    B = (2 * L * Tu) / (log_r_rw * rw**2 * Kw)
    C = Tu / Tl
    D = 1.0 + B + C
    denom = log_r_rw * D
    f = A / denom

    grad_orig = np.zeros_like(X)

    # df/drw
    dlog_drw = -1.0 / rw
    # B = 2*L*Tu / (log_r_rw * rw^2 * Kw)
    # Let u = log_r_rw * rw^2, then B = 2*L*Tu / (u * Kw)
    # du/drw = (-1/rw)*rw^2 + log_r_rw*2*rw = rw*(2*log_r_rw - 1)
    # dB/drw = -2*L*Tu / (u^2 * Kw) * du/drw
    u = log_r_rw * rw**2
    du_drw = rw * (2.0 * log_r_rw - 1.0)
    dB_drw = -(2.0 * L * Tu) / (u**2 * Kw) * du_drw
    dD_drw = dB_drw
    d_denom_drw = dlog_drw * D + log_r_rw * dD_drw
    grad_orig[:, 0] = -A / denom**2 * d_denom_drw

    # df/dr
    dlog_dr = 1.0 / r
    dB_dr = -(2 * L * Tu) / (rw**2 * Kw * log_r_rw**2) * (1.0 / r)
    dD_dr = dB_dr
    d_denom_dr = dlog_dr * D + log_r_rw * dD_dr
    grad_orig[:, 1] = -A / denom**2 * d_denom_dr

    # df/dTu
    dA_dTu = 2 * np.pi * (Hu - Hl)
    dB_dTu = (2 * L) / (log_r_rw * rw**2 * Kw)
    dC_dTu = 1.0 / Tl
    dD_dTu = dB_dTu + dC_dTu
    grad_orig[:, 2] = (dA_dTu * denom - A * log_r_rw * dD_dTu) / denom**2

    # df/dHu
    dA_dHu = 2 * np.pi * Tu
    grad_orig[:, 3] = dA_dHu / denom

    # df/dTl
    dC_dTl = -Tu / Tl**2
    dD_dTl = dC_dTl
    grad_orig[:, 4] = -A / denom**2 * log_r_rw * dD_dTl

    # df/dHl
    dA_dHl = -2 * np.pi * Tu
    grad_orig[:, 5] = dA_dHl / denom

    # df/dL
    dB_dL = (2 * Tu) / (log_r_rw * rw**2 * Kw)
    dD_dL = dB_dL
    grad_orig[:, 6] = -A / denom**2 * log_r_rw * dD_dL

    # df/dKw
    dB_dKw = -(2 * L * Tu) / (log_r_rw * rw**2 * Kw**2)
    dD_dKw = dB_dKw
    grad_orig[:, 7] = -A / denom**2 * log_r_rw * dD_dKw

    # Chain rule: df/dx_unit = df/dx_orig * scale
    grad = grad_orig * scale[np.newaxis, :]
    return grad


# ============================================================================
# OTL Circuit function (6D)
# ============================================================================

_OTL_BOUNDS = np.array([
    [50.0,   150.0],   # Rb1
    [25.0,   70.0],    # Rb2
    [0.5,    3.0],     # Rf
    [1.2,    2.5],     # Rc1
    [0.25,   1.2],     # Rc2
    [50.0,   300.0],   # beta
])


def otl_circuit(X_unit):
    """
    OTL circuit function.

    Parameters
    ----------
    X_unit : ndarray of shape (n, 6)
        Input points in [0, 1]^6.

    Returns
    -------
    y : ndarray of shape (n,)
        Function values.
    """
    X = _scale_to_original(X_unit, _OTL_BOUNDS)
    Rb1  = X[:, 0]
    Rb2  = X[:, 1]
    Rf   = X[:, 2]
    Rc1  = X[:, 3]
    Rc2  = X[:, 4]
    beta = X[:, 5]

    Vb1 = 12.0 * Rb2 / (Rb1 + Rb2)
    term = beta * (Rc2 + 9.0)
    Vm = ((Vb1 + 0.74) * term) / (term + Rf) + \
         (11.35 * Rf) / (term + Rf) + \
         (0.74 * Rf * term) / ((term + Rf) * Rc1)
    return Vm


def otl_circuit_gradient(X_unit):
    """
    Analytical gradient of the OTL circuit function w.r.t. [0,1]^6 inputs.

    Parameters
    ----------
    X_unit : ndarray of shape (n, 6)
        Input points in [0, 1]^6.

    Returns
    -------
    grad : ndarray of shape (n, 6)
        Gradient w.r.t. the unit-cube inputs.
    """
    bounds = _OTL_BOUNDS
    scale = bounds[:, 1] - bounds[:, 0]
    X = _scale_to_original(X_unit, bounds)

    Rb1  = X[:, 0]
    Rb2  = X[:, 1]
    Rf   = X[:, 2]
    Rc1  = X[:, 3]
    Rc2  = X[:, 4]
    beta = X[:, 5]

    Vb1 = 12.0 * Rb2 / (Rb1 + Rb2)
    S = beta * (Rc2 + 9.0)  # common subexpression
    T = S + Rf              # denominator

    grad_orig = np.zeros_like(X)

    # dVm/dRb1 — only Vb1 depends on Rb1
    dVb1_dRb1 = -12.0 * Rb2 / (Rb1 + Rb2)**2
    grad_orig[:, 0] = dVb1_dRb1 * S / T

    # dVm/dRb2 — only Vb1 depends on Rb2
    dVb1_dRb2 = 12.0 * Rb1 / (Rb1 + Rb2)**2
    grad_orig[:, 1] = dVb1_dRb2 * S / T

    # dVm/dRf
    # Term1 = (Vb1 + 0.74) * S / T
    # Term2 = 11.35 * Rf / T
    # Term3 = 0.74 * Rf * S / (T * Rc1)
    dT1_dRf = -(Vb1 + 0.74) * S / T**2
    dT2_dRf = 11.35 * (T - Rf) / T**2
    dT3_dRf = 0.74 * S / (Rc1) * (T - Rf) / T**2
    grad_orig[:, 2] = dT1_dRf + dT2_dRf + dT3_dRf

    # dVm/dRc1
    grad_orig[:, 3] = -0.74 * Rf * S / (T * Rc1**2)

    # dVm/dRc2 — S and T depend on Rc2
    dS_dRc2 = beta
    # d/dRc2 [(Vb1+0.74)*S/T] = (Vb1+0.74) * (dS*T - S*dS) / T^2
    #                          = (Vb1+0.74) * dS * Rf / T^2
    dT1_dRc2 = (Vb1 + 0.74) * dS_dRc2 * Rf / T**2
    dT2_dRc2 = -11.35 * Rf * dS_dRc2 / T**2
    dT3_dRc2 = 0.74 * Rf / Rc1 * dS_dRc2 * Rf / T**2
    grad_orig[:, 4] = dT1_dRc2 + dT2_dRc2 + dT3_dRc2

    # dVm/dbeta — S and T depend on beta
    dS_dbeta = Rc2 + 9.0
    dT1_dbeta = (Vb1 + 0.74) * dS_dbeta * Rf / T**2
    dT2_dbeta = -11.35 * Rf * dS_dbeta / T**2
    dT3_dbeta = 0.74 * Rf / Rc1 * dS_dbeta * Rf / T**2
    grad_orig[:, 5] = dT1_dbeta + dT2_dbeta + dT3_dbeta

    grad = grad_orig * scale[np.newaxis, :]
    return grad


# ============================================================================
# Morris function (20D)
# ============================================================================

def _morris_w(x):
    """
    Compute the w transformation for the Morris function.
    w_i = 2*(1.1*x_i/(x_i + 0.1) - 0.5) for i in {3,5,7} (0-indexed: {2,4,6})
    w_i = 2*(x_i - 0.5) for all other i
    """
    n, d = x.shape
    w = 2.0 * (x - 0.5)
    for i in [2, 4, 6]:  # 0-indexed versions of i=3,5,7
        w[:, i] = 2.0 * (1.1 * x[:, i] / (x[:, i] + 0.1) - 0.5)
    return w


def morris(X_unit):
    """
    Morris function (20D).

    Input is in [0, 1]^20 (no rescaling needed).

    Parameters
    ----------
    X_unit : ndarray of shape (n, 20)
        Input points in [0, 1]^20.

    Returns
    -------
    y : ndarray of shape (n,)
        Function values.
    """
    x = X_unit
    n = x.shape[0]
    d = 20
    w = _morris_w(x)

    # Coefficients
    beta_i = np.array([20.0 if i < 10 else (-1.0)**(i+1) for i in range(d)])
    beta_ij = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1, d):
            if i < 6 and j < 6:
                beta_ij[i, j] = -15.0
            else:
                beta_ij[i, j] = (-1.0)**(i+j+2)

    beta_ijl = np.zeros((d, d, d))
    for i in range(5):
        for j in range(i+1, 5):
            for l in range(j+1, 5):
                beta_ijl[i, j, l] = 10.0

    # First-order terms
    y = w @ beta_i

    # Second-order terms
    for i in range(d):
        for j in range(i+1, d):
            y += beta_ij[i, j] * w[:, i] * w[:, j]

    # Third-order terms
    for i in range(5):
        for j in range(i+1, 5):
            for l in range(j+1, 5):
                y += beta_ijl[i, j, l] * w[:, i] * w[:, j] * w[:, l]

    # Four-way interaction
    y += 5.0 * w[:, 0] * w[:, 1] * w[:, 2] * w[:, 3]

    return y


def morris_gradient(X_unit):
    """
    Analytical gradient of the Morris function w.r.t. [0,1]^20 inputs.

    Parameters
    ----------
    X_unit : ndarray of shape (n, 20)
        Input points in [0, 1]^20.

    Returns
    -------
    grad : ndarray of shape (n, 20)
        Gradient w.r.t. the [0,1]^20 inputs.
    """
    x = X_unit
    n = x.shape[0]
    d = 20
    w = _morris_w(x)

    # dw_i/dx_i
    dw_dx = 2.0 * np.ones_like(x)
    for i in [2, 4, 6]:
        # w_i = 2*(1.1*x/(x+0.1) - 0.5)
        # dw_i/dx_i = 2 * 1.1 * 0.1 / (x+0.1)^2 = 0.22 / (x+0.1)^2
        dw_dx[:, i] = 0.22 / (x[:, i] + 0.1)**2

    # Coefficients
    beta_i = np.array([20.0 if i < 10 else (-1.0)**(i+1) for i in range(d)])
    beta_ij = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1, d):
            if i < 6 and j < 6:
                beta_ij[i, j] = -15.0
            else:
                beta_ij[i, j] = (-1.0)**(i+j+2)

    beta_ijl = np.zeros((d, d, d))
    for i in range(5):
        for j in range(i+1, 5):
            for l in range(j+1, 5):
                beta_ijl[i, j, l] = 10.0

    # df/dw_k for each k
    df_dw = np.zeros((n, d))

    # First-order: df/dw_k = beta_k
    df_dw += beta_i[np.newaxis, :]

    # Second-order: df/dw_k += sum_{j>k} beta_ij[k,j]*w_j + sum_{i<k} beta_ij[i,k]*w_i
    for k in range(d):
        for j in range(k+1, d):
            df_dw[:, k] += beta_ij[k, j] * w[:, j]
        for i in range(k):
            df_dw[:, k] += beta_ij[i, k] * w[:, i]

    # Third-order terms
    for k in range(5):
        for i in range(5):
            for j in range(i+1, 5):
                for l in range(j+1, 5):
                    if k == i:
                        df_dw[:, k] += beta_ijl[i, j, l] * w[:, j] * w[:, l]
                    elif k == j:
                        df_dw[:, k] += beta_ijl[i, j, l] * w[:, i] * w[:, l]
                    elif k == l:
                        df_dw[:, k] += beta_ijl[i, j, l] * w[:, i] * w[:, j]

    # Four-way: 5 * w0 * w1 * w2 * w3
    df_dw[:, 0] += 5.0 * w[:, 1] * w[:, 2] * w[:, 3]
    df_dw[:, 1] += 5.0 * w[:, 0] * w[:, 2] * w[:, 3]
    df_dw[:, 2] += 5.0 * w[:, 0] * w[:, 1] * w[:, 3]
    df_dw[:, 3] += 5.0 * w[:, 0] * w[:, 1] * w[:, 2]

    # Chain rule: df/dx_k = df/dw_k * dw_k/dx_k
    grad = df_dw * dw_dx
    return grad


# ============================================================================
# Active subspace function (10D, exact 2D active subspace)
# ============================================================================

# Known active subspace directions (orthonormal)
_AS_W1 = np.array([1.0, 2.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
_AS_W1 = _AS_W1 / np.linalg.norm(_AS_W1)

_AS_W2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0])
_AS_W2 = _AS_W2 / np.linalg.norm(_AS_W2)


def active_subspace_10d(X_unit):
    """
    10D function with exact 2D active subspace.

    f(x) = exp(-u1^2) * sin(2*pi*u2) + 0.1*sin(2*pi*u1)

    where u1 = w1^T x, u2 = w2^T x and w1, w2 are known orthonormal
    vectors. The gradient lies entirely in span{w1, w2}.

    Parameters
    ----------
    X_unit : ndarray of shape (n, 10)
        Input points in [0, 1]^10.

    Returns
    -------
    y : ndarray of shape (n,)
        Function values.
    """
    u1 = X_unit @ _AS_W1
    u2 = X_unit @ _AS_W2
    return np.exp(-u1**2) * np.sin(2 * np.pi * u2) + 0.1 * np.sin(2 * np.pi * u1)


def active_subspace_10d_gradient(X_unit):
    """
    Analytical gradient of active_subspace_10d w.r.t. [0,1]^10 inputs.

    df/dx = df/du1 * w1 + df/du2 * w2

    Parameters
    ----------
    X_unit : ndarray of shape (n, 10)
        Input points in [0, 1]^10.

    Returns
    -------
    grad : ndarray of shape (n, 10)
        Gradient w.r.t. the unit-cube inputs.
    """
    u1 = X_unit @ _AS_W1
    u2 = X_unit @ _AS_W2
    df_du1 = (-2 * u1 * np.exp(-u1**2) * np.sin(2 * np.pi * u2)
              + 0.1 * 2 * np.pi * np.cos(2 * np.pi * u1))
    df_du2 = 2 * np.pi * np.exp(-u1**2) * np.cos(2 * np.pi * u2)
    return df_du1[:, np.newaxis] * _AS_W1 + df_du2[:, np.newaxis] * _AS_W2


def active_subspace_10d_directional(X_unit):
    """
    Directional derivatives of active_subspace_10d along w1 and w2.

    Returns
    -------
    dirs : ndarray of shape (n, 2)
        Column 0: df/d(w1), column 1: df/d(w2).
    """
    u1 = X_unit @ _AS_W1
    u2 = X_unit @ _AS_W2
    df_du1 = (-2 * u1 * np.exp(-u1**2) * np.sin(2 * np.pi * u2)
              + 0.1 * 2 * np.pi * np.cos(2 * np.pi * u1))
    df_du2 = 2 * np.pi * np.exp(-u1**2) * np.cos(2 * np.pi * u2)
    return np.column_stack([df_du1, df_du2])


def get_active_subspace_directions():
    """Return the two known active subspace directions as (10, 2) matrix."""
    return np.column_stack([_AS_W1, _AS_W2])


# ============================================================================
# Data generation utilities
# ============================================================================

def generate_training_data(func, grad_func, n_samples, dim, seed=42):
    """
    Generate training data using maximin Latin Hypercube Sampling.

    Parameters
    ----------
    func : callable
        Function mapping (n, d) -> (n,).
    grad_func : callable
        Gradient function mapping (n, d) -> (n, d).
    n_samples : int
        Number of training points.
    dim : int
        Input dimension.
    seed : int
        Random seed.

    Returns
    -------
    X_train : ndarray of shape (n_samples, dim)
        Training inputs in [0, 1]^d.
    y_train : ndarray of shape (n_samples,)
        Function values.
    grad_train : ndarray of shape (n_samples, dim)
        Gradient values.
    """
    sampler = LatinHypercube(d=dim, seed=seed)
    X_train = sampler.random(n=n_samples)
    y_train = func(X_train)
    grad_train = grad_func(X_train)
    return X_train, y_train, grad_train


def generate_test_data(func, n_test, dim, seed=99):
    """
    Generate test data using Latin Hypercube Sampling.

    Parameters
    ----------
    func : callable
        Function mapping (n, d) -> (n,).
    n_test : int
        Number of test points.
    dim : int
        Input dimension.
    seed : int
        Random seed.

    Returns
    -------
    X_test : ndarray of shape (n_test, dim)
        Test inputs in [0, 1]^d.
    y_test : ndarray of shape (n_test,)
        True function values at test points.
    """
    sampler = LatinHypercube(d=dim, seed=seed)
    X_test = sampler.random(n=n_test)
    y_test = func(X_test)
    return X_test, y_test


def compute_metrics(y_true, y_pred):
    """
    Compute RMSE and normalized RMSE.

    Parameters
    ----------
    y_true : ndarray
        True values.
    y_pred : ndarray
        Predicted values.

    Returns
    -------
    metrics : dict
        Dictionary with 'rmse' and 'nrmse' (normalized by range).
    """
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    y_range = np.max(y_true) - np.min(y_true)
    nrmse = rmse / y_range if y_range > 0 else np.inf
    return {'rmse': rmse, 'nrmse': nrmse}


# ============================================================================
# Verification
# ============================================================================

def verify_gradients(func, grad_func, dim, bounds=None, n_points=5,
                     eps=1e-6, seed=42):
    """
    Verify analytical gradients against finite differences.

    Parameters
    ----------
    func : callable
        Function mapping (n, d) -> (n,).
    grad_func : callable
        Gradient function mapping (n, d) -> (n, d).
    dim : int
        Input dimension.
    bounds : ndarray, optional
        Not used (functions take [0,1]^d input).
    n_points : int
        Number of test points.
    eps : float
        Finite difference step size.
    seed : int
        Random seed.

    Returns
    -------
    max_rel_error : float
        Maximum relative error across all points and dimensions.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(0.1, 0.9, size=(n_points, dim))  # stay away from boundaries

    grad_analytical = grad_func(X)
    grad_fd = np.zeros_like(X)

    for j in range(dim):
        X_plus = X.copy()
        X_minus = X.copy()
        X_plus[:, j] += eps
        X_minus[:, j] -= eps
        grad_fd[:, j] = (func(X_plus) - func(X_minus)) / (2 * eps)

    abs_error = np.abs(grad_analytical - grad_fd)
    scale = np.maximum(np.abs(grad_fd), 1e-10)
    rel_error = abs_error / scale
    max_rel_error = np.max(rel_error)

    print(f"  Max absolute error: {np.max(abs_error):.6e}")
    print(f"  Max relative error: {max_rel_error:.6e}")
    return max_rel_error


if __name__ == "__main__":
    print("Verifying Borehole gradients...")
    err = verify_gradients(borehole, borehole_gradient, 8)
    print(f"  PASS: {err < 1e-4}\n")

    print("Verifying OTL circuit gradients...")
    err = verify_gradients(otl_circuit, otl_circuit_gradient, 6)
    print(f"  PASS: {err < 1e-4}\n")

    print("Verifying Morris gradients...")
    err = verify_gradients(morris, morris_gradient, 20)
    print(f"  PASS: {err < 1e-4}\n")

    print("Verifying active subspace 10D gradients...")
    err = verify_gradients(active_subspace_10d, active_subspace_10d_gradient, 10)
    print(f"  PASS: {err < 1e-4}\n")