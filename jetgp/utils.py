import numpy as np
import pyoti.core as coti
import pyoti.sparse as oti
import sympy as sp
from scipy.stats import qmc
from scipy.optimize import minimize
from scipy.stats import norm
import sys
from scipy.linalg import null_space
from full_gddegp import gddegp
from line_profiler import profile
from scipy.stats.qmc import Sobol
import scipy.stats as stats
from fractions import Fraction
from scipy.special import comb
from numpy.polynomial import Polynomial
import math
import utils
from full_degp import degp_utils


def scale_samples(samples, lower_bounds, upper_bounds):
    """
    Scale each column of samples from the unit interval [0, 1] to user-defined bounds [lb_j, ub_j].

    Parameters:
    ----------
    samples : ndarray of shape (d, n)
        A 2D array where each column represents a sample in [0, 1]^n.
    lower_bounds : array-like of length n
        Lower bounds for each dimension.
    upper_bounds : array-like of length n
        Upper bounds for each dimension.

    Returns:
    -------
    ndarray of shape (d, n)
        Scaled samples where each column is mapped from [0, 1] to [lb_j, ub_j] for each dimension.

    Notes:
    -----
    This function assumes that each sample is a column vector, and bounds are applied column-wise.

    Example:
    --------
    >>> samples = np.array([[0.5, 0.2], [0.8, 0.4]])
    >>> lower_bounds = [0, 1]
    >>> upper_bounds = [1, 3]
    >>> scale_samples(samples, lower_bounds, upper_bounds)
    array([[0.5, 1.4],
           [0.8, 1.8]])
    """

    samples = np.asarray(samples)
    lower_bounds = np.asarray(lower_bounds)
    upper_bounds = np.asarray(upper_bounds)

    # Ensure correct shapes
    assert (
        samples.shape[1] == len(lower_bounds) == len(upper_bounds)
    ), "Dimension mismatch between samples and bounds"

    # Reshape bounds to broadcast across rows
    lb = lower_bounds[np.newaxis, :]
    ub = upper_bounds[np.newaxis, :]

    return lb + samples * (ub - lb)


def flatten_der_indices(indices):
    """
    Flatten a nested list of derivative indices.

    Parameters:
    ----------
    indices : list of lists
        A nested list where each sublist contains derivative index specifications.

    Returns:
    -------
    list
        A single flattened list containing all derivative index entries.

    Example:
    --------
    >>> indices = [[[1, 1]], [[1, 2], [2, 1]]]
    >>> flatten_der_indices(indices)
    [[1, 1], [1, 2], [2, 1]]
    """

    flattened_indices = []
    for i in range(0, len(indices)):
        for j in range(0, len(indices[i])):
            flattened_indices.append(indices[i][j])
    return flattened_indices


def nrmse(y_true, y_pred, norm_type="minmax"):
    """
    Compute the Normalized Root Mean Squared Error (NRMSE) between true and predicted values.

    Parameters:
    ----------
    y_true : array-like
        Ground truth or reference values.
    y_pred : array-like
        Predicted values to compare against the ground truth.
    norm_type : str, default="minmax"
        The method used to normalize the RMSE:
        - 'minmax': Normalize by the range (max - min) of `y_true`.
        - 'mean': Normalize by the mean of `y_true`.
        - 'std': Normalize by the standard deviation of `y_true`.

    Returns:
    -------
    float
        The normalized root mean squared error.

    Raises:
    -------
    ValueError
        If `norm_type` is not one of {'minmax', 'mean', 'std'}.

    Example:
    --------
    >>> y_true = np.array([3, 5, 2, 7])
    >>> y_pred = np.array([2.5, 5.5, 2, 8])
    >>> nrmse(y_true, y_pred, norm_type="mean")
    0.1443  # Example value (varies depending on input)
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)

    if norm_type == "minmax":
        norm = np.max(y_true) - np.min(y_true)
    elif norm_type == "mean":
        norm = np.mean(y_true)
    elif norm_type == "std":
        norm = np.std(y_true)
    else:
        raise ValueError("norm_type must be 'minmax', 'mean', or 'std'")

    return rmse / norm if norm != 0 else np.inf


def convert_index_to_exponent_form(lst):
    """
   Convert a list of indices into exponent form.

   For a given list of integers, the function compresses consecutive identical elements
   into pairs [value, count], where 'value' is the element and 'count' is its occurrence.

   Parameters:
   ----------
   lst : list of int
       A list of integers representing variable indices.

   Returns:
   -------
   list of [int, int]
       A compressed list where each entry is [value, count], representing the multiplicity
       of each unique value in the original list.

   Example:
   --------
   >>> lst = [1, 1, 2, 2, 2, 3]
   >>> convert_index_to_exponent_form(lst)
   [[1, 2], [2, 3], [3, 1]]
   """

    compressed = []
    current_num = None
    count = 0

    for num in lst:
        if num != current_num:
            if current_num is not None:
                compressed.append([current_num, count])
            current_num = num
            count = 1
        else:
            count += 1

    if current_num is not None:
        compressed.append([current_num, count])

    return compressed


def build_companion_array(nvars, order, der_indices):
    """
    Build a companion array that maps each derivative index to its corresponding order in the OTI basis.

    The array represents the order of each component in the OTI (Order-Truncated Imaginary) number system:
    - 0 for function values.
    - Corresponding derivative order for each derivative term.

    Parameters:
    ----------
    nvars : int
        Number of variables (input dimensions).
    order : int
        Maximum derivative order considered.
    der_indices : list of lists
        Derivative indices in exponent form, where each sublist represents the derivative multi-index
        for a specific derivative term.

    Returns:
    -------
    companion_array : ndarray
        A 1D array of length (1 + total derivatives), where:
        - The first entry is 0 (function value).
        - Each subsequent entry indicates the derivative order (e.g., 1 for first derivatives).

    Example:
    --------
    >>> nvars = 2
    >>> order = 2
    >>> der_indices = [[[1, 1]], [[1, 2]], [[2, 1]]]
    >>> build_companion_array(nvars, order, der_indices)
    array([0, 1, 2, 1])
    """
    companion_list = [0]
    for i in range(len(der_indices)):
        for j in range(0, len(der_indices[i])):
            companion_list.append(
                compare_OTI_indices(nvars, order, der_indices[i][j]))

    companion_array = np.array(companion_list)
    return companion_array


def compare_OTI_indices(nvars, order, term_check):
    """
    Compare a given multi-index term against all basis terms in the OTI number system.

    This function searches through all OTI basis terms (multi-indices) up to the specified order
    and identifies the **order** of the matching term.

    Parameters:
    ----------
    nvars : int
        Number of variables (input dimensions).
    order : int
        Maximum derivative order considered.
    term_check : list of [int, int]
        The multi-index to check, given in exponent form [[var_index, exponent], ...].

    Returns:
    -------
    int
        The order of the term in the OTI basis (1, 2, ..., order).
        Returns -1 if no matching term is found.

    Example:
    --------
    >>> nvars = 2
    >>> order = 2
    >>> term_check = [[1, 2]]  # Represents ∂²/∂x₁²
    >>> compare_OTI_indices(nvars, order, term_check)
    2
    """
    dH = coti.get_dHelp()

    for ordi in range(1, order + 1):
        nterms = coti.ndir_order(nvars, ordi)
        for idx in range(nterms):
            term = convert_index_to_exponent_form(dH.get_fulldir(idx, ordi))
            if term == term_check:
                return ordi
    return -1


def gen_OTI_indices(nvars, order):
    """
    Generate the list of OTI (Order-Truncated Imaginary) basis indices in exponent form.

    For a given number of variables and maximum derivative order, this function produces the
    multi-index representations for all basis terms in the OTI number system.

    Parameters:
    ----------
    nvars : int
        Number of variables (input dimensions).
    order : int
        Maximum derivative order considered.

    Returns:
    -------
    list of lists
        A nested list where:
        - The outer list has length `order` (one entry per derivative order).
        - Each inner list contains multi-indices in exponent form for that order.

    Example:
    --------
    >>> nvars = 2
    >>> order = 2
    >>> gen_OTI_indices(nvars, order)
    [
        [[[1, 1]], [[2, 1]]],         # First-order derivatives: ∂/∂x₁, ∂/∂x₂
        # Second-order: ∂²/∂x₁², ∂²/∂x₁∂x₂, ∂²/∂x₂²
        [[[1, 2]], [[1, 1], [2, 1]], [[2, 2]]]
    ]
    """
    dH = coti.get_dHelp()

    ind = [0] * order

    for ordi in range(1, order + 1):
        nterms = coti.ndir_order(nvars, ordi)
        i = [0] * nterms
        for idx in range(nterms):
            i[idx] = convert_index_to_exponent_form(dH.get_fulldir(idx, ordi))
        ind[ordi - 1] = i
    return ind


def reshape_y_train(y_train):
    """
    Flatten and concatenate function values and derivative observations into a single 1D array.

    Parameters:
    ----------
    y_train : list of arrays
        A list where:
        - y_train[0] contains the function values (shape: (n_samples,))
        - y_train[1:], if present, contain derivative values (shape: (n_samples,)) for each derivative component.

    Returns:
    -------
    ndarray of shape (n_total,)
        A flattened 1D array concatenating function values and all derivatives.

    Example:
    --------
    >>> y_train = [np.array([1.0, 2.0]), np.array([0.5, 1.0])]
    >>> reshape_y_train(y_train)
    array([1.0, 2.0, 0.5, 1.0])
    """
    y_train_flattened = y_train[0]
    for i in range(1, len(y_train)):
        y_train_flattened = np.vstack(
            (y_train_flattened.reshape(-1, 1), y_train[i].reshape(-1, 1)))
    return y_train_flattened.flatten()


def transform_cov(cov, sigma_y, sigmas_x, der_indices, X_test):
    """
    Rescale the diagonal of a covariance matrix to reflect the original (unnormalized) variance
    of function values and derivatives.

    This function transforms the variance estimates from normalized space back to the original scale.

    Parameters:
    ----------
    cov : ndarray of shape (n_total, n_total)
        Covariance matrix from the GP model (including function values and derivatives).
    sigma_y : float
        Standard deviation used to normalize the function values.
    sigmas_x : ndarray of shape (1, nvars)
        Standard deviations used to normalize each input dimension.
    der_indices : list of lists
        Derivative multi-indices, where each sublist represents the derivative directions and orders.
    X_test : ndarray of shape (n_samples, nvars)
        Test input points corresponding to the covariance matrix blocks.

    Returns:
    -------
    y_var_normalized : ndarray of shape (n_total,)
        Rescaled variances for function values and derivatives in the original space.

    Example:
    --------
    >>> cov.shape = (n_total, n_total)
    >>> transform_cov(cov, sigma_y, sigmas_x, der_indices, X_test).shape == (n_total,)
    """
    var = abs(np.diag(cov))
    y_var_normalized = np.zeros((var.shape[0],))
    y_var_normalized[0:X_test.shape[0]] = var[0:X_test.shape[0]]*sigma_y**2
    for i in range(len(der_indices)):
        factor = 1
        for j in range(len(der_indices[i])):
            factor = factor * \
                (sigmas_x[0][der_indices[i][j][0] - 1])**(der_indices[i][j][1])
        factor = sigma_y**2 / factor**2
        y_var_normalized[(i+1) * X_test.shape[0]: (i+2) * X_test.shape[0]
                         ] = var[(i+1) * X_test.shape[0]: (i+2) * X_test.shape[0]]*factor
    return y_var_normalized


def transform_cov_directrional(cov, sigma_y, sigmas_x, der_indices, X_test):
    """
    Rescale the diagonal of a covariance matrix for function values and directional derivatives.

    Unlike `transform_cov`, this function assumes **directional derivatives** (not multi-index derivatives),
    so no input scaling (`sigmas_x`) is applied to derivative terms.

    Parameters:
    ----------
    cov : ndarray of shape (n_total, n_total)
        Covariance matrix from the GP model (including function values and directional derivatives).
    sigma_y : float
        Standard deviation used to normalize the function values.
    sigmas_x : ndarray of shape (1, nvars)
        Standard deviations used to normalize each input dimension (unused for derivatives here).
    der_indices : list of lists
        Derivative directions (for directional derivatives).
    X_test : ndarray of shape (n_samples, nvars)
        Test input points corresponding to the covariance matrix blocks.

    Returns:
    -------
    y_var_normalized : ndarray of shape (n_total,)
        Rescaled variances for function values and directional derivatives in the original space.

    Example:
    --------
    >>> cov.shape = (n_total, n_total)
    >>> transform_cov_directrional(cov, sigma_y, sigmas_x, der_indices, X_test).shape == (n_total,)
    """
    var = abs(np.diag(cov))
    y_var_normalized = np.zeros((var.shape[0],))
    y_var_normalized[0:X_test.shape[0]] = var[0:X_test.shape[0]]*sigma_y**2
    for i in range(len(der_indices)):
        factor = sigma_y**2
        y_var_normalized[(i+1) * X_test.shape[0]: (i+2) * X_test.shape[0]
                         ] = var[(i+1) * X_test.shape[0]: (i+2) * X_test.shape[0]]*factor
    return y_var_normalized


def transform_predictions(y_pred, mu_y, sigma_y, sigmas_x, der_indices, X_test):
    """
    Rescale predicted function values and derivatives from normalized space back to their original scale.

    This function transforms both function value predictions and **multi-index derivatives**
    back to the original units after GP prediction.

    Parameters:
    ----------
    y_pred : ndarray of shape (n_total,)
        Predicted mean values from the GP model in normalized space
        (includes function values and derivatives).
    mu_y : float
        Mean of the original function values (before normalization).
    sigma_y : float
        Standard deviation of the original function values (before normalization).
    sigmas_x : ndarray of shape (1, nvars)
        Standard deviations of the input variables (used for rescaling derivatives).
    der_indices : list of lists
        Multi-index derivative structures for each derivative component.
    X_test : ndarray of shape (n_samples, nvars)
        Test input points corresponding to the prediction blocks.

    Returns:
    -------
    y_train_normalized : ndarray of shape (n_total, 1)
        Rescaled function values and derivatives in the original scale.

    Example:
    --------
    >>> y_pred.shape = (n_total,)
    >>> transform_predictions(y_pred, mu_y, sigma_y, sigmas_x, der_indices, X_test).shape
    (n_total, 1)
    """
    y_train_normalized = (
        mu_y + y_pred[0:X_test.shape[0]]*sigma_y).reshape(-1, 1)
    for i in range(len(der_indices)):
        factor = sigma_y
        for j in range(len(der_indices[i])):
            factor = factor / \
                (sigmas_x[0][der_indices[i][j][0] - 1])**(der_indices[i][j][1])
        y_train_normalized = np.vstack((y_train_normalized, (y_pred[(
            i + 1) * X_test.shape[0]: (i + 2) * X_test.shape[0]] * factor[0, 0]).reshape(-1, 1)))
    return y_train_normalized


def transform_predictions_directional(y_pred, mu_y, sigma_y, sigmas_x, der_indices, X_test):
    """
    Rescale predicted function values and **directional derivatives** from normalized space
    back to their original scale.

    This function assumes the derivatives are **directional** (not multi-index),
    so it applies only output scaling (sigma_y) to derivatives.

    Parameters:
    ----------
    y_pred : ndarray of shape (n_total,)
        Predicted mean values from the GP model in normalized space
        (includes function values and directional derivatives).
    mu_y : float
        Mean of the original function values (before normalization).
    sigma_y : float
        Standard deviation of the original function values (before normalization).
    sigmas_x : ndarray of shape (1, nvars)
        Standard deviations of the input variables (not used here but included for compatibility).
    der_indices : list of lists
        Directions for directional derivatives (each sublist represents a direction vector).
    X_test : ndarray of shape (n_samples, nvars)
        Test input points corresponding to the prediction blocks.

    Returns:
    -------
    y_train_normalized : ndarray of shape (n_total, 1)
        Rescaled function values and directional derivatives in the original scale.

    Example:
    --------
    >>> y_pred.shape = (n_total,)
    >>> transform_predictions_directional(y_pred, mu_y, sigma_y, sigmas_x, der_indices, X_test).shape
    (n_total, 1)
    """
    y_train_normalized = (
        mu_y + y_pred[0:X_test.shape[0]]*sigma_y).reshape(-1, 1)
    for i in range(len(der_indices)):
        factor = sigma_y
        y_train_normalized = np.vstack((y_train_normalized, (y_pred[(
            i + 1) * X_test.shape[0]: (i + 2) * X_test.shape[0]] * factor[0, 0]).reshape(-1, 1)))
    return y_train_normalized


def normalize_x_data_test(X_test, sigmas_x, mus_x):
    """
    Normalize test input data using the mean and standard deviation from the training inputs.

    Parameters:
    ----------
    X_test : ndarray of shape (n_samples, nvars)
        Test input points to be normalized.
    sigmas_x : ndarray of shape (1, nvars)
        Standard deviations of the training inputs for each variable (used for scaling).
    mus_x : ndarray of shape (1, nvars)
        Means of the training inputs for each variable (used for centering).

    Returns:
    -------
    X_train_normalized : ndarray of shape (n_samples, nvars)
        Normalized test inputs.

    Example:
    --------
    >>> X_test = np.array([[2.0, 3.0]])
    >>> sigmas_x = np.array([[1.0, 2.0]])
    >>> mus_x = np.array([[0.0, 1.0]])
    >>> normalize_x_data_test(X_test, sigmas_x, mus_x)
    array([[2.0, 1.0]])
    """

    X_train_normalized = (X_test - mus_x) / sigmas_x

    return X_train_normalized


def normalize_x_data_train(X_train):
    """
    Normalize training input data by centering and scaling each variable.

    Parameters:
    ----------
    X_train : ndarray of shape (n_samples, nvars)
        Training input points.

    Returns:
    -------
    X_train_normalized : ndarray of shape (n_samples, nvars)
        Normalized training inputs.
    mean_vec_x : ndarray of shape (1, nvars)
        Mean values for each input variable (used for centering).
    std_vec_x : ndarray of shape (1, nvars)
        Standard deviations for each input variable (used for scaling).

    Example:
    --------
    >>> X_train = np.array([[1.0, 2.0], [3.0, 4.0]])
    >>> normalize_x_data_train(X_train)
    (array([[-1., -1.], [ 1.,  1.]]), array([[2., 3.]]), array([[1., 1.]]))
    """
    mean_vec_x = np.mean(X_train, axis=0).reshape(1, -1)  # shape: (m, 1)
    std_vec_x = np.std(X_train, axis=0).reshape(1, -1)    # shape: (m, 1)

    X_train_normalized = (X_train - mean_vec_x) / std_vec_x

    return X_train_normalized


def normalize_directions(sigmas_x, rays):
    """
    Normalize direction vectors (rays) based on input scaling.

    This function rescales direction vectors used for **directional derivatives** so that
    they are consistent with normalized input space.

    Parameters:
    ----------
    sigmas_x : ndarray of shape (1, nvars)
        Standard deviations of the input variables (used for scaling each direction).
    rays : ndarray of shape (nvars, n_directions)
        Direction vectors (columns) in the original input space.

    Returns:
    -------
    transformed_rays : ndarray of shape (nvars, n_directions)
        Normalized direction vectors.

    Example:
    --------
    >>> sigmas_x = np.array([[2.0, 1.0]])
    >>> rays = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> normalize_directions(sigmas_x, rays)
    array([[0.5, 0. ],
           [0. , 1. ]])
    """
    transformed_rays = np.zeros(rays.shape)
    for i in range(rays.shape[1]):
        transformed_rays[:, i] = np.diag(1/sigmas_x.flatten()) @ rays[:, i]
    return transformed_rays


# def normalize_directions_2(sigmas_x, rays):
#     """
#     Normalize direction vectors (rays) based on input scaling.

#     This function rescales direction vectors used for **directional derivatives** so that
#     they are consistent with normalized input space.

#     Parameters:
#     ----------
#     sigmas_x : ndarray of shape (1, nvars)
#         Standard deviations of the input variables (used for scaling each direction).
#     rays : ndarray of shape (nvars, n_directions)
#         Direction vectors (columns) in the original input space.

#     Returns:
#     -------
#     transformed_rays : ndarray of shape (nvars, n_directions)
#         Normalized direction vectors.

#     Example:
#     --------
#     >>> sigmas_x = np.array([[2.0, 1.0]])
#     >>> rays = np.array([[1.0, 0.0], [0.0, 1.0]])
#     >>> normalize_directions(sigmas_x, rays)
#     array([[0.5, 0. ],
#            [0. , 1. ]])
#     """
#     transformed_rays_list = []
#     for i in range(len(rays)):

#         transformed_rays = np.zeros(rays[i].shape)
#         for j in range(transformed_rays.shape[1]):
#             transformed_rays[:, j] = np.diag(
#                 1/sigmas_x.flatten()) @ rays[i][:, j]
#         transformed_rays_list.append(transformed_rays)
#     return transformed_rays_list


def normalize_directions_2(sigmas_x, rays_array):
    """
    Normalize direction vectors (rays) based on input scaling.

    This function rescales direction vectors used for **directional derivatives** so that
    they are consistent with normalized input space.

    Parameters:
    ----------
    sigmas_x : ndarray of shape (1, nvars)
        Standard deviations of the input variables (used for scaling each direction).
    rays : ndarray of shape (nvars, n_directions)
        Direction vectors (columns) in the original input space.

    Returns:
    -------
    transformed_rays : ndarray of shape (nvars, n_directions)
        Normalized direction vectors.

    Example:
    --------
    >>> sigmas_x = np.array([[2.0, 1.0]])
    >>> rays = np.array([[1.0, 0.0], [0.0, 1.0]])
    >>> normalize_directions(sigmas_x, rays)
    array([[0.5, 0. ],
            [0. , 1. ]])
    """

    return [rays_array[i] / sigmas_x.flatten()[:, None] for i in range(len(rays_array))]


def normalize_y_data(X_train, y_train, sigma_data, der_indices):
    """
    Normalize function values, derivatives, and observational noise for training data.

    This function:
    - Normalizes function values (`y_train[0]`) to have zero mean and unit variance.
    - Scales **derivatives** using the chain rule, considering input normalization.
    - Scales **observational noise** (`sigma_data`) accordingly.

    Parameters:
    ----------
    X_train : ndarray of shape (n_samples, nvars)
        Training input points (used to compute input normalization statistics).
    y_train : list of arrays
        List where:
        - y_train[0] contains function values.
        - y_train[1:], if present, contain derivative values for each derivative component.
    sigma_data : float or None
        Standard deviation of the observational noise (for function values).
        If provided, it will be normalized.
    der_indices : list of lists
        Multi-index derivative structures for each derivative component.

    Returns:
    -------
    y_train_normalized : ndarray of shape (n_total,)
        Normalized function values and derivatives (flattened).
    mean_vec_y : ndarray of shape (m, 1)
        Mean of function values before normalization.
    std_vec_y : ndarray of shape (m, 1)
        Standard deviation of function values before normalization.
    std_vec_x : ndarray of shape (1, nvars)
        Standard deviations of input variables.
    mean_vec_x : ndarray of shape (1, nvars)
        Means of input variables.
    noise_std_normalized : float or None
        Normalized observational noise standard deviation.

    Example:
    --------
    >>> normalize_y_data(X_train, y_train, sigma_data=0.75, der_indices=[[[1, 1]]])
    (y_train_normalized, mean_vec_y, std_vec_y,
     std_vec_x, mean_vec_x, noise_std_normalized)
    """

    num_points = len(X_train)
    mean_vec_x = np.mean(X_train, axis=0).reshape(1, -1)  # shape: (m, 1)
    std_vec_x = np.std(X_train, axis=0).reshape(1, -1)    # shape: (m, 1)

    mean_vec_y = np.mean(y_train[0], axis=0).reshape(-1, 1)  # shape: (m, 1)
    std_vec_y = np.std(y_train[0], axis=0).reshape(-1, 1)    # shape: (m, 1)

    y_train_normalized = (y_train[0] - mean_vec_y)/std_vec_y
    noise_std_normalized = sigma_data
    for i in range(len(der_indices)):
        factor = 1/std_vec_y
        for j in range(len(der_indices[i])):
            factor = factor * \
                (std_vec_x[0][der_indices[i][j][0] - 1]
                 )**(der_indices[i][j][1])
        y_train_normalized = np.vstack(
            (y_train_normalized.reshape(-1, 1), y_train[i + 1] * factor[0, 0]))
        if sigma_data is not None:
            noise_std_normalized[(
                i + 1)*num_points:(i+2)*num_points] = sigma_data[(
                    i + 1)*num_points:(i+2)*num_points] * factor[0, 0]
        else:
            noise_std_normalized = None
        # Scale noise if provided

    return y_train_normalized.flatten(), mean_vec_y, std_vec_y, std_vec_x, mean_vec_x, noise_std_normalized


def normalize_y_data_directional(X_train, y_train, sigma_data, der_indices):
    """
    Normalize function values and **directional derivatives** for training data.

    This function:
    - Normalizes **function values (`y_train[0]`)** to have zero mean and unit variance.
    - Scales **directional derivatives** (`y_train[1:]`) by the **function value standard deviation (`std_vec_y`)**.

    Parameters:
    ----------
    X_train : ndarray of shape (n_samples, nvars)
        Training input points (used to compute input normalization statistics).
    y_train : list of arrays
        List where:
        - y_train[0] contains function values.
        - y_train[1:], if present, contain directional derivative values for each direction.
    der_indices : list of lists
        Directions for directional derivatives (each sublist represents a direction vector).

    Returns:
    -------
    y_train_normalized : ndarray of shape (n_total,)
        Normalized function values and directional derivatives (flattened).
    mean_vec_y : ndarray of shape (m, 1)
        Mean of function values before normalization.
    std_vec_y : ndarray of shape (m, 1)
        Standard deviation of function values before normalization.
    std_vec_x : ndarray of shape (1, nvars)
        Standard deviations of input variables.
    mean_vec_x : ndarray of shape (1, nvars)
        Means of input variables.

    Example:
    --------
    >>> normalize_y_data_directional(X_train, y_train, der_indices=[[[1, 0.5], [2, 0.5]]])
    (y_train_normalized, mean_vec_y, std_vec_y, std_vec_x, mean_vec_x)
    """
    mean_vec_x = np.mean(X_train, axis=0).reshape(1, -1)  # shape: (m, 1)
    std_vec_x = np.std(X_train, axis=0).reshape(1, -1)    # shape: (m, 1)

    mean_vec_y = np.mean(y_train[0], axis=0).reshape(-1, 1)  # shape: (m, 1)
    std_vec_y = np.std(y_train[0], axis=0).reshape(-1, 1)    # shape: (m, 1)

    y_train_normalized = (y_train[0] - mean_vec_y)/std_vec_y

    for i in range(len(der_indices)):
        factor = 1/std_vec_y
        y_train_normalized = np.vstack(
            (y_train_normalized.reshape(-1, 1), y_train[i + 1] * factor[0, 0]))
    if sigma_data is not None:
        noise_std_normalized = sigma_data / std_vec_y[0, 0]
    else:
        noise_std_normalized = None

    return y_train_normalized.flatten(), mean_vec_y, std_vec_y, std_vec_x, mean_vec_x, noise_std_normalized


def generate_submodel_noise_matricies(sigma_data, index, der_indices, num_points, base_der_indices):
    """
    Generate diagonal noise covariance matrices for each submodel component
    (including function values and their associated derivatives).

    This function constructs a list of diagonal matrices where each matrix corresponds
    to a specific group of training indices (e.g., for a submodel), and includes
    both function value noise and associated derivative noise.

    Parameters
    ----------
    sigma_data : ndarray of shape (n_total, n_total)
        Full covariance (typically diagonal) matrix for all training data, including function values and derivatives.
    index : list of lists of int
        List where each sublist contains indices of the function values for one submodel (e.g., a cluster or partition).
    der_indices : list of lists
        Each sublist contains derivative directions for the submodel, corresponding to base_der_indices.
    num_points : int
        Number of training points per function (used to compute index offsets for derivatives).
    base_der_indices : list of lists
        Master list of derivative indices that define the ordering of blocks in the covariance matrix.

    Returns
    -------
    sub_model_matricies : list of ndarray
        List of diagonal noise matrices (ndarray of shape (n_submodel_total, n_submodel_total)) for each submodel,
        combining noise contributions from function values and all applicable derivative components.

    Raises
    ------
    Exception
        If a derivative index in `der_indices` is not found in `base_der_indices`.

    Example
    -------
    >>> sigma_data.shape = (300, 300)
    >>> index = [[0, 1, 2], [3, 4, 5]]
    >>> der_indices = [[[1, 1]], [[2, 1], [1, 2]]]
    >>> base_der_indices = [[[1, 1]], [[2, 1]], [[1, 2]]]
    >>> num_points = 100
    >>> generate_submodel_noise_matricies(sigma_data, index, der_indices, num_points, base_der_indices)
    [array of shape (6, 6), array of shape (9, 9)]
    """

    sub_model_matricies = []
    for i, idx in enumerate(index):
        values = np.diag(sigma_data[:num_points, :num_points])
        for j in range(len(der_indices[i])):
            for k, item in enumerate(base_der_indices):
                if item == der_indices[i][j]:
                    scale_factor = k
                    break
                else:
                    scale_factor = -1
                    continue

            if scale_factor == -1:
                raise Exception('Unknown Error')
            scale_factor = k + 1
            indices = (scale_factor*num_points)+np.array(idx)
            values = np.concatenate(
                (values, sigma_data[indices[0:], indices[0:]].flatten()))
        sub_model_matricies.append(np.diag(values))

    return sub_model_matricies


def matern_kernel_builder(nu):
    """
    Symbolically builds the Matérn kernel function with given smoothness ν.

    Parameters
    ----------
    nu : float
        Smoothness parameter of the Matérn kernel. Should be a half-integer (e.g., 0.5, 1.5, 2.5, ...).

    Returns
    -------
    callable
        A lambdified function that evaluates the Matérn kernel as a function of distance r.
    """
    r = sp.symbols('r')
    nu = sp.Rational(2 * nu, 2)
    prefactor = (2 ** (1 - nu)) / sp.gamma(nu)
    z = sp.sqrt(2 * nu) * r
    k_r = prefactor * z**nu * sp.simplify(sp.besselk(nu, z))
    expr = sp.simplify(k_r)
    custom_dict = {"exp": oti.exp}
    matern_kernel_func = sp.lambdify((r), expr, modules=[custom_dict, "numpy"])
    return matern_kernel_func



def generate_bernoulli_numbers(n_max: int) -> list[Fraction]:
    """
    Generates Bernoulli numbers B_0 to B_n_max using their recurrence relation.

    Args:
        n_max: The maximum order of the Bernoulli number to generate.

    Returns:
        A list of Bernoulli numbers as Fraction objects.
    """
    if n_max < 0:
        return []

    B = [Fraction(0)] * (n_max + 1)
    B[0] = Fraction(1)  # B_0 = 1

    for n in range(1, n_max + 1):
        # The recurrence relation sum: sum_{k=0 to n} C(n+1, k) * B_k = 0
        # We solve for B_n: B_n = - (1 / C(n+1, n)) * sum_{k=0 to n-1} C(n+1, k) * B_k
        sum_val = Fraction(0)
        for k in range(n):
            # Binomial coefficient C(n+1, k)
            binomial_coeff = comb(n + 1, k, exact=True)
            sum_val += binomial_coeff * B[k]
        
        # C(n+1, n) is just n+1
        B[n] = -sum_val / (n + 1)
        
        # B_k is zero for all odd k > 1
        if n > 1 and n % 2 != 0:
            B[n] = Fraction(0)
            
    return B

def generate_bernoulli_polynomial(alpha: int) -> Polynomial:
    """
    Generates the (2*alpha)-th Bernoulli polynomial, B_{2*alpha}(x).

    Args:
        alpha: A non-negative integer.

    Returns:
        A numpy.polynomial.Polynomial object representing the polynomial.
    """
    if not isinstance(alpha, int) or alpha < 0:
        raise ValueError("alpha must be a non-negative integer.")

    n = 2 * alpha
    
    # Step 1: Generate the required Bernoulli numbers
    bernoulli_nums = generate_bernoulli_numbers(n)
    
    # Step 2: Construct the polynomial coefficients
    # The polynomial is sum_{k=0 to n} C(n, k) * B_k * x^(n-k)
    # The coefficient of x^j is C(n, n-j) * B_{n-j}
    # numpy.polynomial expects coeffs from lowest power (x^0) to highest (x^n)
    
    coeffs = np.zeros(n + 1, dtype=object)
    
    for j in range(n + 1): # j represents the power of x
        k = n - j
        # Coefficient for x^j is C(n, k) * B_k
        binomial_coeff = comb(n, k, exact=True)
        term_coeff = binomial_coeff * bernoulli_nums[k]
        coeffs[j] = float(term_coeff) # Convert fraction to float for Polynomial class
        
    # The Polynomial class takes coefficients from lowest power to highest
    return Polynomial(coeffs)

def generate_bernoulli_lambda(alpha: int) -> callable:
    """
    Generates a callable (lambda) function for the (2*alpha)-th Bernoulli polynomial.

    Args:
        alpha: A non-negative integer.

    Returns:
        A callable function that evaluates B_{2*alpha}(x).
    """
    # The numpy Polynomial object is already a callable function.
    return ((-1)**(alpha + 1) * (2*np.pi)**(2 * alpha)) / (math.factorial(2 * alpha)) * generate_bernoulli_polynomial(alpha)


def robust_local_optimization(func, x0, args=(), lb=None, ub=None, debug=False):
    """Robust L-BFGS-B optimization with abnormal termination handling"""
    bounds = list(zip(lb, ub)) if lb is not None and ub is not None else None

    result = minimize(
        func, x0, args=args, method="L-BFGS-B", bounds=bounds,
        options={"disp": False, "maxiter": 1000, "gtol": 1e-7}
    )
    sys.stdout.flush()

    if "ABNORMAL_TERMINATION_IN_LNSRCH" in result.message:
        result.recovered_from_abnormal = False
        if debug:
            print(f"L-BFGS-B terminated abnormally: {result.message:}")
            print("Using current point as recovered solution...")

        # Create result with current point
        class RecoveredResult:
            def __init__(self, x, fun, message):
                self.x = x
                self.fun = fun
                self.success = False
                self.message = message
                self.recovered_from_abnormal = True

        current_f = func(result.x, *args)
        return RecoveredResult(result.x, current_f, f"Recovered from abnormal termination: {result.message}")
    else:
        return result


def should_accept_local_result(local_res, current_best_f, is_feasible, debug=False):
    """Check if local optimization result should be accepted"""
    if not local_res.success and not local_res.recovered_from_abnormal:
        return False

    if not is_feasible(local_res.x):
        if debug:
            print("Local optimization result is infeasible - rejecting")
        return False

    if local_res.fun >= current_best_f:
        if debug:
            print(
                "Local optimization did not improve:")
        return False

    return True

def jade(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
         pop_size=100, maxiter=100, p=0.1, c=0.1,
         minfunc=1e-8, minstep=1e-8, debug=False,
         local_opt_every=15, initial_positions=None, seed=42):
    """
    JADE (Adaptive Differential Evolution) with optional local refinement
    and minstep stopping criterion applied only when g_best improves.
    
    JADE: Adaptive Differential Evolution With Optional External Archive
    https://ieeexplore.ieee.org/abstract/document/5208221
    
    """

    lb = np.asarray(lb)
    ub = np.asarray(ub)
    D = len(lb)
    np.random.seed(seed)

    # Constraint setup
    if f_ieqcons is not None:
        def cons(x): return np.asarray(f_ieqcons(x, *args, **kwargs))
    elif ieqcons:
        def cons(x): return np.asarray([c(x, *args, **kwargs) for c in ieqcons])
    else:
        def cons(x): return np.array([0.0])

    def is_feasible(x):
        return np.all(cons(x) >= 0)

    # Initialize population using Sobol
    sampler = qmc.Sobol(d=D, scramble=True, seed=seed)
    sobol_sample = sampler.random_base2(m=int(np.ceil(np.log2(pop_size))))
    pop = qmc.scale(sobol_sample[:pop_size], lb, ub)
    if initial_positions is not None:
        n_init = min(len(initial_positions), pop_size)
        pop[:n_init] = initial_positions[:n_init]

    # Fitness and feasibility
    fitness = np.array([func(ind, *args, **kwargs) for ind in pop])
    feasible = np.array([is_feasible(ind) for ind in pop])

    # Initialize global best
    g_idx = np.argmin(fitness * feasible + (~feasible)*1e10)
    g_best = pop[g_idx].copy()
    f_best = fitness[g_idx]

    mu_F = 0.5
    mu_CR = 0.5
    archive = []

    # Store previous best for minstep comparison
    prev_g_best = g_best.copy()
    f_prev_best = f_best

    for gen in range(1, maxiter+1):
        new_pop = np.zeros_like(pop)
        new_fitness = np.zeros(pop_size)
        new_F_list = []
        new_CR_list = []

        for i in range(pop_size):
            # Adaptive F and CR
            F = np.random.standard_cauchy() * 0.1 + mu_F
            F = np.clip(F, 0, 1)
            CR = np.random.normal(mu_CR, 0.1)
            CR = np.clip(CR, 0, 1)

            # p-best selection
            p_num = max(2, int(np.ceil(p*pop_size)))
            pbest_idx = np.random.choice(np.argsort(fitness)[:p_num])
            x_pbest = pop[pbest_idx]

            # Mutation: current-to-pbest
            idxs = [idx for idx in range(pop_size) if idx != i]
            r1, r2 = np.random.choice(idxs, 2, replace=False)
            x_mut = pop[i] + F*(x_pbest - pop[i]) + F*(pop[r1] - pop[r2])
            x_mut = np.clip(x_mut, lb, ub)

            # Crossover
            jrand = np.random.randint(D)
            trial = np.array([x_mut[j] if np.random.rand() < CR or j == jrand else pop[i][j] for j in range(D)])
            trial = np.clip(trial, lb, ub)

            f_trial = func(trial, *args, **kwargs)
            if is_feasible(trial) and f_trial < fitness[i]:
                new_pop[i] = trial
                new_fitness[i] = f_trial
                archive.append(pop[i].copy())
                new_F_list.append(F)
                new_CR_list.append(CR)
            else:
                new_pop[i] = pop[i]
                new_fitness[i] = fitness[i]

        # Update population
        pop = new_pop
        fitness = new_fitness

        # Update global best
        feasible_idx = [idx for idx in range(pop_size) if is_feasible(pop[idx])]
        if feasible_idx:
            idx_best = feasible_idx[np.argmin(fitness[feasible_idx])]
            if fitness[idx_best] < f_best:
                # Check minstep only when g_best improves
                step_size = np.linalg.norm(pop[idx_best] - prev_g_best)
                if step_size <= minstep:
                    if debug:
                        print(f"Stopping: Position change < {minstep} at generation {gen}")
                    g_best = pop[idx_best].copy()
                    f_best = fitness[idx_best]
                    break

                g_best = pop[idx_best].copy()
                f_best = fitness[idx_best]
                prev_g_best = g_best.copy()
                f_prev_best = f_best

        # Adapt F and CR
        if new_F_list:
            mu_F = (1-c)*mu_F + c*np.mean(new_F_list)
            mu_CR = (1-c)*mu_CR + c*np.mean(new_CR_list)

        # Periodic local refinement
        if gen % local_opt_every == 0:
            res = minimize(func, g_best, args=args, bounds=np.stack((lb, ub), axis=1))
            if is_feasible(res.x) and res.fun < f_best:
                g_best = res.x.copy()
                f_best = res.fun
                prev_g_best = g_best.copy()
                f_prev_best = f_best
                if debug:
                    print(f"Local refinement improved at gen {gen}: f={f_best}")

        # Debug print
        if debug:
            print(f"Gen {gen}: best f={f_best}")

        # Check minfunc stopping criterion
        if f_best <= minfunc:
            if debug:
                print(f"Stopping: Objective improvement < {minfunc} at generation {gen}")
            break

    return g_best, f_best

def pso(func, lb, ub, ieqcons=[], f_ieqcons=None, args=(), kwargs={},
        pop_size=100, omega=0.5, phip=0.5, phig=0.5, maxiter=100,
        minstep=1e-8, minfunc=1e-8, debug=False, seed=42,
        local_opt_every=15, initial_positions=None):
    """
    Particle Swarm Optimization with periodic local refinement
    R. C. Eberhart, Y. Shi and J. Kennedy, Swarm Intelligence, CA, San Mateo:Morgan Kaufmann, 2001.
    https://theswissbay.ch/pdf/Gentoomen%20Library/Artificial%20Intelligence/Swarm%20Intelligence/Swarm%20intelligence%20-%20James%20Kennedy.pdf
    Parameters:
    -----------
    func : callable
        Objective function to minimize
    lb : array_like
        Lower bounds for variables
    ub : array_like  
        Upper bounds for variables
    ieqcons : list, optional
        List of inequality constraint functions
    f_ieqcons : callable, optional
        Function returning array of inequality constraints
    args : tuple, optional
        Extra arguments passed to objective function
    kwargs : dict, optional
        Extra keyword arguments passed to objective function
    swarmsize : int, optional
        Number of particles in swarm
    omega : float, optional
        Inertia weight
    phip : float, optional
        Personal best weight
    phig : float, optional
        Global best weight
    maxiter : int, optional
        Maximum number of iterations
    minstep : float, optional
        Minimum step size for convergence
    minfunc : float, optional
        Minimum function improvement for convergence
    debug : bool, optional
        Whether to print debug information
    seed : int, optional
        Random seed for reproducibility
    local_opt_every : int, optional
        Frequency of local optimization (every N iterations)

    Returns:
    --------
    best_position : ndarray
        Best position found
    best_value : float
        Best function value found
    """

    # Input validation
    lb = np.asarray(lb)
    ub = np.asarray(ub)
    D = len(lb)

    assert len(lb) == len(
        ub), "Lower- and upper-bounds must be the same length"
    assert np.all(
        ub > lb), "All upper-bound values must be greater than lower-bound values"
    assert callable(func), "Invalid function handle"

    # Set random seed
    np.random.seed(seed)

    # Velocity bounds
    vhigh = np.abs(ub - lb)
    vlow = -vhigh

    # Constraint setup
    if f_ieqcons is not None:
        def cons(x):
            return np.asarray(f_ieqcons(x, *args, **kwargs))
    elif ieqcons:
        def cons(x):
            return np.asarray([c(x, *args, **kwargs) for c in ieqcons])
    else:
        def cons(x):
            return np.array([0.0])

    def is_feasible(x):
        return np.all(cons(x) >= 0)

    # Initialize swarm using Sobol sequence
    sampler = qmc.Sobol(d=D, scramble=True, seed=seed)
    sobol_sample = sampler.random_base2(m=int(np.ceil(np.log2(pop_size))))
    x = qmc.scale(sobol_sample[:pop_size], lb, ub)
    if initial_positions is not None:
        n_init = min(len(initial_positions), pop_size)
        x[:n_init] = initial_positions[:n_init]
    # Initialize velocities and personal bests
    v = vlow + np.random.rand(pop_size, D) * (vhigh - vlow)
    p = np.copy(x)
    fp = np.array([func(p[i], *args, **kwargs) for i in range(pop_size)])
    feasibles = np.array([is_feasible(p[i]) for i in range(pop_size)])

    # Initialize global best
    fg = np.inf
    g = None

    for i in range(pop_size):
        if feasibles[i] and fp[i] < fg:
            fg = fp[i]
            g = p[i].copy()

    # Main PSO loop
    it = 1
    while it <= maxiter:
        # Update velocities and positions
        rp = np.random.rand(pop_size, D)
        rg = np.random.rand(pop_size, D)
        v = omega * v + phip * rp * (p - x) + phig * rg * (g - x)
        x = np.clip(x + v, lb, ub)

        # Evaluate particles and update personal/global bests
        for i in range(pop_size):
            fx = func(x[i], *args, **kwargs)

            if fx < fp[i] and is_feasible(x[i]):
                p[i] = x[i].copy()
                fp[i] = fx

                if fx < fg:
                    stepsize = np.linalg.norm(g - x[i])

                    if debug:
                        print(
                            f'New best for swarm at iteration {it}: {x[i]} {fx}')

                    # Check convergence criteria
                    if np.abs(fg - fx) <= minfunc:
                        print(f'Stopping: Objective improvement < {minfunc}')
                        return x[i], fx
                    if stepsize <= minstep:
                        print(f'Stopping: Position change < {minstep}')
                        return x[i], fx

                    g = x[i].copy()
                    fg = fx

        # Periodic local refinement - CLEANED UP VERSION
        if it % local_opt_every == 0:
            local_res = robust_local_optimization(
                func, g, args=args, lb=lb, ub=ub, debug=False
            )

            if should_accept_local_result(local_res, fg, is_feasible, debug):
                if debug:
                    print(
                        "Gradient refinement improved")
                g = local_res.x.copy()
                fg = local_res.fun

        if debug:
            print(f'Best after iteration {it}: {g} {fg}')
        it += 1

    # Final checks
    print(f'Stopping: maximum iterations reached --> {maxiter}')
    if g is not None and not is_feasible(g):
        print("Warning: Optimization finished without a feasible solution.")

    return g, fg


def ecl_acquisition(mu_N, var_N, threshold=0.0):
    """
    Entropy Contour Learning (ECL) acquisition function for gddegp models.

    Implements the ECL acquisition function from equation (8):
    ECL(x | S_N, g) = -P(g(Y(x)) > 0) log P(g(Y(x)) > 0) - P(g(Y(x)) ≤ 0) log P(g(Y(x)) ≤ 0)

    Where g is the affine limit state function g(Y(x)) = Y(x) - T and the failure 
    region G is defined by g(Y(x)) ≤ 0.

    Parameters:
    -----------
    gp_model : gddegp
        Trained gp model instance
    X : array-like, shape (n_points, n_features) or (n_features,)
        Points to evaluate the acquisition function
    rays_predict : ndarray
        Ray directions for prediction. Shape should be (n_dims, n_rays)
    params : ndarray
        Hyperparameters for gddegp prediction
    threshold : float, default=0.0
        Threshold T for the affine limit state function g(Y(x)) = Y(x) - T
        Default is 0.0, meaning failure region is defined by Y(x) ≤ 0

    Returns:
    --------
    ecl_values : array-like, shape (n_points,) or float
        ECL acquisition function values (higher values indicate more informative points)

    Examples:
    ---------
    >>> # After training your gddegp model
    >>> ecl_vals = ecl_acquisition(gp_model, candidate_points, rays, params, threshold=0.0)
    >>> next_point_idx = np.argmax(ecl_vals)
    >>> next_point = candidate_points[next_point_idx]
    """

    # Convert variance to standard deviation
    sigma_N = np.sqrt(np.maximum(var_N, 1e-15))  # Ensure positive variance

    # Handle single point case
    if mu_N.ndim == 0:
        mu_N = np.array([mu_N])
    if sigma_N.ndim == 0:
        sigma_N = np.array([sigma_N])

    # Compute standardized values: (μ_N(x) - T) / σ_N(x)
    sigma_N = np.maximum(sigma_N, 1e-10)  # Numerical stability
    z = (mu_N - threshold) / sigma_N

    # Compute probabilities using standard normal CDF Φ
    # P(g(Y(x)) > 0) = P(Y(x) > T) = 1 - Φ((μ_N(x) - T)/σ_N(x))
    p_safe = 1 - norm.cdf(z)  # Probability of being in safe region

    # Compute ECL using binary entropy: H(p) = -p*log(p) - (1-p)*log(1-p)
    # Clip probabilities for numerical stability
    p_safe = np.clip(p_safe, 1e-15, 1 - 1e-15)
    ecl_values = -p_safe * np.log(p_safe) - (1 - p_safe) * np.log(1 - p_safe)

    # Return scalar if single point, array otherwise
    return ecl_values[0] if len(ecl_values) == 1 else ecl_values


def ecl_batch_acquisition(gp_model, X, rays_predict, params, threshold=0.0, batch_size=1):
    """
    Batch ECL acquisition function that selects multiple points simultaneously.

    Uses a greedy approach to select the next batch_size points that maximize
    the ECL criterion while maintaining diversity.

    Parameters:
    -----------
    gp_model : gddegp
        Trained gddegp model instance
    X : array-like, shape (n_candidates, n_features)
        Candidate points to select from
    rays_predict : ndarray
        Ray directions for prediction
    params : ndarray
        Hyperparameters for gddegp prediction
    threshold : float, default=0.0
        Threshold for the limit state function
    batch_size : int, default=1
        Number of points to select in the batch

    Returns:
    --------
    selected_indices : array-like, shape (batch_size,)
        Indices of selected points from X
    selected_points : array-like, shape (batch_size, n_features)
        The selected points

    Examples:
    ---------
    >>> indices, points = ecl_batch_acquisition(gp_model, candidates, rays, params, batch_size=3)
    >>> next_experiments = points
    """
    X = np.atleast_2d(X)
    n_candidates = X.shape[0]

    if batch_size >= n_candidates:
        return np.arange(n_candidates), X

    selected_indices = []
    remaining_indices = list(range(n_candidates))

    for _ in range(batch_size):
        if not remaining_indices:
            break

        # Evaluate ECL for remaining candidates
        remaining_X = X[remaining_indices]
        ecl_values = ecl_acquisition(
            gp_model, remaining_X, rays_predict, params, threshold)

        # Select point with highest ECL value
        best_local_idx = np.argmax(ecl_values)
        best_global_idx = remaining_indices[best_local_idx]

        selected_indices.append(best_global_idx)
        remaining_indices.remove(best_global_idx)

    selected_indices = np.array(selected_indices)
    selected_points = X[selected_indices]

    return selected_indices, selected_points


def get_surrogate_gradient_ray(gp, x, params, fallback_axis=0, normalize=True, threshold=0.0):
    """
    Returns a normalized surrogate gradient direction (d x 1 column vector)
    at location x using the current GP model (any input dimension).
    If the GP mean at x is above threshold, returns -grad; else returns grad.

    Parameters
    ----------
    gp : object
        Trained GP model instance with .predict (supports arbitrary input dim)
    x : array-like, shape (1, d)
        The input location where to compute the surrogate gradient direction
    params : array-like
        GP hyperparameters
    fallback_axis : int, default=0
        Axis to use if gradient norm is zero (default: 0)
    normalize : bool, default=True
        If True, return a unit vector; else, return unnormalized gradient
    threshold : float, default=0.0
        Threshold value for sign flip

    Returns
    -------
    ray : ndarray, shape (d, 1)
        The chosen direction (as a column vector)
    grad : ndarray, shape (d, 1)
        The predicted gradient as a column vector (signed as above)
    """
    x = np.atleast_2d(x)
    d = x.shape[1]
    grad = np.zeros((d, 1))

    # Predict each directional derivative along standard basis
    for i in range(d):
        basis = np.zeros((d, 1))
        basis[i, 0] = 1.0
        mu = gp.predict(x, basis, params, calc_cov=False, return_deriv=True)
        grad[i, 0] = mu[1]  # Assumes mu[0]=f(x), mu[1]=first derivative

    # Predict the GP mean at x (using any direction, e.g., first axis)
    basis0 = np.zeros((d, 1))
    basis0[0, 0] = 1.0
    mu_mean = gp.predict(x, basis0, params, calc_cov=False,
                         return_deriv=False)[0]

    # Flip sign if above threshold
    if mu_mean > threshold:
        grad = -grad

    norm = np.linalg.norm(grad)
    if normalize:
        if norm < 1e-16:
            fallback = np.zeros((d, 1))
            fallback[fallback_axis, 0] = 1.0
            return fallback.reshape(-1, 1)
        else:
            return (grad / norm).reshape(-1, 1)
    else:
        return grad.reshape(-1, 1)


def finite_difference_gradient(gp, x, params, h=1e-5):
    """
    Compute central finite difference approximation of GP mean gradient at x.

    Parameters
    ----------
    gp : object
        Trained GP model instance
    x : ndarray, shape (1, d)
        Point at which to compute finite difference gradient
    params : array-like
        GP hyperparameters
    h : float
        Step size

    Returns
    -------
    grad_fd : ndarray, shape (d, 1)
        Central finite difference gradient estimate
    """
    x = np.atleast_2d(x)
    d = x.shape[1]
    grad_fd = np.zeros((d, 1))
    ray0 = np.zeros((d, 1))
    # Direction for mean prediction; can be any since return_deriv=False
    ray0[0, 0] = 1.0

    # Evaluate GP mean at central point and shifted points
    for i in range(d):
        dx = np.zeros_like(x)
        dx[0, i] = h

        f_plus = gp.predict(x + dx, ray0, params,
                            calc_cov=False, return_deriv=False)
        f_minus = gp.predict(x - dx, ray0, params,
                             calc_cov=False, return_deriv=False)
        grad_fd[i, 0] = (f_plus[0] - f_minus[0]) / (2 * h)
    return grad_fd


def check_gp_gradient(gp, x, params, h=1e-5, fallback_axis=0):
    """
    Compare GP-predicted gradient vs finite-difference at x.
    Prints and returns both.
    """
    # Surrogate-predicted gradient
    grad_sur = get_surrogate_gradient_ray(
        gp, x, params, fallback_axis=fallback_axis, normalize=False)
    # Finite difference gradient
    grad_fd = finite_difference_gradient(gp, x, params, h=h)

    print("Surrogate GP gradient:\n", grad_sur.flatten())
    print("Finite-difference gradient:\n", grad_fd.flatten())
    print("Absolute error:\n", np.abs(grad_sur.flatten() - grad_fd.flatten()))
    print("Relative error:\n", np.abs(grad_sur.flatten() -
          grad_fd.flatten()) / (np.abs(grad_fd.flatten()) + 1e-15))

    return grad_sur, grad_fd


def get_entropy_ridge_direction_nd(gp, x, params, threshold=0.0, h=1e-5,
                                   fallback_axis=0, normalize=True, random_dir=False, seed=None):
    """
    Get a direction tangent to the entropy level set ("ridge direction") at x.
    In higher dimensions, returns either a single direction or an orthonormal basis for the tangent space.

    Parameters
    ----------
    gp : object
        Trained GP model instance with .predict
    x : array-like, shape (1, d)
        The input location
    params : array-like
        GP hyperparameters
    threshold : float
        ECL threshold
    h : float
        Finite difference step
    fallback_axis : int
        Axis for fallback if gradient is zero
    normalize : bool
        Normalize output direction
    random_dir : bool
        If True, return a random direction in the ridge (level set). If False, returns first basis vector.
    seed : int or None
        For reproducible random direction

    Returns
    -------
    ridge_dir : ndarray, shape (d, 1)
        Ridge direction (tangent to entropy level set) at x
    grad_H : ndarray, shape (d, 1)
        Gradient of entropy at x
    basis : ndarray, shape (d, d-1)
        (If requested) Orthonormal basis for tangent space to entropy level set at x
    """
    from utils import ecl_acquisition

    x = np.atleast_2d(x)
    d = x.shape[1]

    # Entropy at x
    def entropy_func(x0):
        x0 = np.atleast_2d(x0)
        ray0 = np.zeros((d, 1))
        ray0[0, 0] = 1.0
        mu, var = gp.predict(
            x0, ray0, params, calc_cov=True, return_deriv=False)
        return float(ecl_acquisition(mu, var, threshold=threshold))

    grad_H = np.zeros((d, 1))
    for i in range(d):
        dx = np.zeros_like(x)
        dx[0, i] = h
        grad_H[i, 0] = (entropy_func(x + dx) - entropy_func(x - dx)) / (2 * h)

    # Edge case: gradient nearly zero
    # grad_norm = np.linalg.norm(grad_H)
    # if grad_norm < 1e-12:
    #     fallback = np.zeros((d, 1))
    #     fallback[fallback_axis, 0] = 1.0
    #     return fallback, grad_H, None

    # Orthonormal basis for null space (tangent space to level set)
    # null_space returns (d, d-1) matrix: each column is a basis vector
    basis = null_space(grad_H.T)  # shape (d, d-1)

    # Select a direction from the tangent space
    if random_dir:
        rng = np.random.default_rng(seed)
        coeffs = rng.standard_normal(basis.shape[1])
        ridge_dir = basis @ coeffs
        if normalize:
            ridge_dir = ridge_dir / np.linalg.norm(ridge_dir)
        ridge_dir = ridge_dir.reshape(-1, 1)
    else:
        # Return first basis vector (deterministic)
        ridge_dir = basis[:, 0]
        if normalize:
            ridge_dir = ridge_dir / np.linalg.norm(ridge_dir)
        ridge_dir = ridge_dir.reshape(-1, 1)

    return ridge_dir


def get_entropy_ridge_direction_nd_2(gp, x, params, threshold=0.0, h=1e-5,
                                     fallback_axis=0, normalize=True, random_dir=False, seed=None):
    """
    Get a direction tangent to the entropy level set ("ridge direction") at x.
    In higher dimensions, returns either a single direction or an orthonormal basis for the tangent space.

    Parameters
    ----------
    gp : object
        Trained GP model instance with .predict
    x : array-like, shape (1, d)
        The input location
    params : array-like
        GP hyperparameters
    threshold : float
        ECL threshold
    h : float
        Finite difference step
    fallback_axis : int
        Axis for fallback if gradient is zero
    normalize : bool
        Normalize output direction
    random_dir : bool
        If True, return a random direction in the ridge (level set). If False, returns first basis vector.
    seed : int or None
        For reproducible random direction

    Returns
    -------
    ridge_dir : ndarray, shape (d, 1)
        Ridge direction (tangent to entropy level set) at x
    grad_H : ndarray, shape (d, 1)
        Gradient of entropy at x
    basis : ndarray, shape (d, d-1)
        (If requested) Orthonormal basis for tangent space to entropy level set at x
    """
    grad_H = get_surrogate_gradient_ray(
        gp, x, params, fallback_axis=0, normalize=True, threshold=0.0)

    # Edge case: gradient nearly zero
    # grad_norm = np.linalg.norm(grad_H)
    # if grad_norm < 1e-12:
    #     fallback = np.zeros((d, 1))
    #     fallback[fallback_axis, 0] = 1.0
    #     return fallback, grad_H, None

    # Orthonormal basis for null space (tangent space to level set)
    # null_space returns (d, d-1) matrix: each column is a basis vector
    basis = null_space(grad_H.T)  # shape (d, d-1)

    best_grad = np.inf
    best_dir = None
    for idx in range(basis.shape[1]):
        v = basis[:, idx].reshape(-1, 1)
        mu = gp.predict(x, v, params, calc_cov=False, return_deriv=True)
        if gp.n_order >= 2:
            second_deriv = mu[2]
        else:
            # Use central finite difference on the GP's directional derivative
            x_fwd = x + h * v.T
            x_bwd = x - h * v.T
            # Predict first derivative at forward and backward positions
            mu_fwd = gp.predict(
                x_fwd, v, params, calc_cov=False, return_deriv=True)
            mu_bwd = gp.predict(
                x_bwd, v, params, calc_cov=False, return_deriv=True)
            # mu_fwd[1] and mu_bwd[1] are first directional derivatives at shifted positions
            second_deriv = (mu_fwd[1] - mu_bwd[1]) / (2 * h)
        if abs(second_deriv) < best_grad:
            best_grad = abs(second_deriv)
            best_dir = v

    return best_dir


def sobol_points(n_points, box, seed=0):
    d = len(box)
    sampler = qmc.Sobol(d=d, scramble=True, seed=seed)
    m = int(np.ceil(np.log2(n_points)))
    pts = sampler.random_base2(m=m)[:n_points]
    for j, (lo, hi) in enumerate(box):
        pts[:, j] = lo + (hi - lo) * pts[:, j]
    return pts


def local_box_around_point(x_next, delta):
    # x_next: shape (1, d) or (d,)
    x_next = np.atleast_2d(x_next).reshape(-1)
    d = x_next.shape[0]
    return [(float(x_next[j] - delta), float(x_next[j] + delta)) for j in range(d)]


@profile
def maximize_ier_direction(
    gp, x_next, x_train, y_blocks, rays_array, params, box, threshold=0.0, n_integration=500, seed=123, delta=.5
):
    """
    Maximizes Integrated Entropy Reduction (IER) at x_next over directions v (unit norm).
    """
    d = x_next.shape[1]
    # 1. Integration points for Monte Carlo estimate
    local_box = local_box_around_point(x_next, delta)
    integration_points = sobol_points(n_integration, local_box, seed)
    # Use a default ray for integration points (e.g., all along first axis)
    ray0 = np.zeros((d, 1))
    ray0[0, 0] = 1.0
    integration_rays = np.tile(ray0, n_integration)

    # 2. Get current mean/variance at integration points
    mu_before, var_before = gp.predict(
        integration_points, integration_rays, params, calc_cov=True, return_deriv=False
    )
    from utils import ecl_acquisition
    ecl_before = ecl_acquisition(mu_before, var_before, threshold=threshold)

    @profile
    def negative_ier(w):
        v = w.reshape(-1, 1)
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            return 1e6
        v = v / norm_v

        Y_blocks = [y.copy() for y in y_blocks]
        # --- Normalize if needed ---
        v_ = v
        integration_rays_ = integration_rays
        # utils.check_gp_gradient(gp, x_next, previous_params)
        # Hypercomplex construction for new point

        # Compute GP predictions (function value and derivatives)
        y_pred = gp.predict(x_next, v_, params,
                            calc_cov=False, return_deriv=True).ravel()

        # How many derivatives do you want to include?
        n_order = gp.n_order

        # Rebuild y_blocks_next to include the GP predictions
        Y_blocks_next = []

        # Append the function prediction
        Y_blocks_next.append(y_pred[0].reshape(-1, 1))

        # Append derivatives in order (1st, 2nd, ..., n_order)
        for i in range(1, n_order + 1):
            Y_blocks_next.append(y_pred[i].reshape(-1, 1))

        # Augment training set
        X_train = np.vstack([x_train, x_next])
        for k in range(len(y_blocks)):
            Y_blocks[k] = np.vstack([Y_blocks[k], Y_blocks_next[k]])
        rays_array_tmp = np.concatenate([rays_array, v_], axis=1)

        gp_temp = gddegp.gddegp(X_train, Y_blocks,
                                n_order=gp.n_order,
                                rays_array=rays_array_tmp,
                                normalize=True,
                                kernel="SE",
                                kernel_type="anisotropic",)

        mu, var_after = gp_temp.predict(
            integration_points, integration_rays_, params, calc_cov=True, return_deriv=False)
        ecl_after = ecl_acquisition(
            mu, var_after, threshold=threshold)
        ier = np.mean(ecl_before - ecl_after)
        return -ier
    # 4. Optimize using L-BFGS-B, with a couple random restarts for robustness

    D = d  # dimension of direction
    lb = np.full(D, -2.0)
    ub = np.full(D, 2.0)
    # Optionally, can try [-1, 1] or any sufficiently wide box (PSO will normalize inside objective)

    best_w, best_val = pso(
        negative_ier, lb, ub,
        pop_size=10*d, maxiter=11, debug=True, seed=seed + 77
    )
    v_opt = best_w.reshape(-1, 1)
    v_opt /= np.linalg.norm(v_opt)

    return v_opt


# utils.py


# UPDATED function to use the new bounds logic
# def find_next_point(
#     gp,
#     params,
#     x_train,
#     y_train_list,
#     dist_params,
#     acquisition_func,
#     integration_points = None,
#     n_candidate_points: int = 1024,
#     n_local_starts: int = 10,
#     use_agg_al = False,
#     var_hist = [],
#     **acq_kwargs
# ):
#     """Finds the next point using distribution-aware optimization bounds."""
    
#     if not use_agg_al:
#         # --- Stage 1: Coarse Global Search ---
#         shape, scale = get_pdf_params(dist_params)
        
#         sampler = Sobol(d=x_train.shape[1], scramble=True)
#         coarse_points_unit = sampler.random(n=n_candidate_points)
#         candidate_points = get_inverse(dist_params, coarse_points_unit
#         )
        
#         candidate_scores = acquisition_func(
#             gp, params, x_train, y_train_list,
#             candidate_points, integration_points = integration_points, normalize=gp.normalize
#         )
        
#         top_indices = np.argsort(candidate_scores.ravel())[-n_local_starts:]
#         top_starting_points = candidate_points[top_indices]
    
#         # --- Stage 2: Local Optimization ---
        
#         # Get the correct bounds for the optimizer
#         bounds = get_optimization_bounds(dist_params)
#         all_unbounded = all(b[0] is None for b in bounds)
    
#         def objective_function(x: np.ndarray) -> float:
#             x_candidate = x.reshape(1, -1)
#             score = acquisition_func(
#                 gp, params, x_train, y_train_list,
#                 x_candidate, integration_points = integration_points, normalize=gp.normalize
#             )
#             return -score[0]
    
#         best_x = None
#         best_score = -np.inf
        
#         for start_point in top_starting_points:
#             if all_unbounded:
#                 res = minimize(fun=objective_function, x0=start_point, method="BFGS")
#             else:
#                 res = minimize(fun=objective_function, x0=start_point, method="L-BFGS-B", bounds=bounds)
            
#             if -res.fun > best_score:
#                 best_score = -res.fun
#                 best_x = res.x
                
#         return best_x.reshape(1, -1), best_score
#     else:
#         agg_variance = acquisition_func(var_hist)
#         max_index = np.argmax(agg_variance)
#         return integration_points[max_index,:], agg_variance[0,max_index]

def _ensure_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        return x.reshape(1, -1)
    return x

def _format_y_to_add(y_pred):
    """Convert gp.predict output at one x to list of 2D arrays for y_train_list append."""
    if isinstance(y_pred, (list, tuple)):
        return [np.atleast_2d(np.asarray(e)) for e in y_pred]
    else:
        return [np.atleast_2d(np.asarray(y_pred))]

def find_next_point_batch(
    gp,
    params,
    dist_params,
    acquisition_func,
    integration_points=None,
    candidate_points=None,
    n_candidate_points: int = 1024,
    n_local_starts: int = 1,
    n_batch_points: int = 10,
    local_opt = True,
    **acq_kwargs
):
    """
    Batch active learning with support for aggregated mode (multi-GP).

    If use_agg_al=False: single GP loop.
    If use_agg_al=True: use gp_list + params_list and acquisition_func must handle them.
    Returns: np.ndarray of selected points, shape (n_selected, d).
    """

    selected_points = []
    n_integration_points = integration_points.shape[0]
    # helper to generate candidates
    def _generate_candidates(dimension, seed_offset=0):
        sampler = Sobol(d=dimension, scramble=True, seed=acq_kwargs.get('seed', 0) + seed_offset)
        coarse_points_unit = sampler.random(n=n_candidate_points)
        return get_inverse(dist_params, coarse_points_unit)

    if type(gp) is not list or type(params) is not list:
        raise ValueError("gp and params must be provided as list")



    for n in range(n_batch_points):
        if candidate_points is None:
            candidate_points = _generate_candidates(gp[0].n_bases,seed_offset=n)
        # Normalize once before the loop
        cand_scores_all = []
        
        if gp[0].normalize:
            sigmas_x = gp[0].sigmas_x
            mus_x = gp[0].mus_x
            
            candidate_points_norm = utils.normalize_x_data_test(candidate_points.copy(), sigmas_x, mus_x)
            if integration_points is not None:
                integration_points_norm = utils.normalize_x_data_test(integration_points.copy(), sigmas_x, mus_x)
                n_integration_points = integration_points.shape[0]
        else:
            candidate_points_norm = candidate_points.copy()
            integration_points_norm = integration_points.copy()
            n_integration_points = None
        
        # Precompute all differences once
        precomputed_diffs = {
            'train_cand': degp_utils.differences_by_dim_func(
                gp[0].x_train, candidate_points_norm, gp[0].n_order, return_deriv=False
            ),
            'cand_cand': degp_utils.differences_by_dim_func(
                candidate_points_norm, candidate_points_norm, gp[0].n_order, return_deriv=False
            ),
            'domain_cand': degp_utils.differences_by_dim_func(
                integration_points_norm, candidate_points_norm, gp[0].n_order, return_deriv=False
            ),
            'train_domain': degp_utils.differences_by_dim_func(
                gp[0].x_train, integration_points_norm, gp[0].n_order, return_deriv=False
            )
        }
        # Loop over time steps - much simpler now!
        for gp_t, params_t in zip(gp, params):
            scores_t = acquisition_func(
                gp_t,
                params_t,
                precomputed_diffs,
                n_integration_points = n_integration_points
            )
            
                    
                
            cand_scores_all.append(scores_t.reshape(-1, 1))
            
            
        cand_scores = np.mean(np.hstack(cand_scores_all), axis=1)
        # --- Pick top starting points from aggregated candidate scores ---
        top_indices = np.argsort(cand_scores.ravel())[-n_local_starts:]
        top_starting_points = candidate_points[top_indices]
        
        if local_opt:
            # --- Get bounds for local optimization ---
            bounds = get_optimization_bounds(dist_params)
            all_unbounded = all(b[0] is None for b in bounds)
            
            # --- Define objective function for local refinement ---
            def objective_function(x: np.ndarray) -> float:
                """
                Objective for local optimization. 
                Note: Can't use precomputed diffs here since x is varying during optimization.
                """
                x_cand = x.reshape(1, -1)
                
                # Normalize the candidate point if needed
                if gp[0].normalize:
                    x_cand_norm = utils.normalize_x_data_test(x_cand.copy(), sigmas_x, mus_x)
                else:
                    x_cand_norm = x_cand.copy()
                
                # Compute differences for this single candidate
                diff_train_cand = degp_utils.differences_by_dim_func(
                    gp[0].x_train, x_cand_norm, gp[0].n_order, return_deriv=False
                )
                diff_cand_cand = degp_utils.differences_by_dim_func(
                    x_cand_norm, x_cand_norm, gp[0].n_order, return_deriv=False
                )
                diff_domain_cand = degp_utils.differences_by_dim_func(
                    integration_points_norm, x_cand_norm, gp[0].n_order, return_deriv=False
                )
                
                # Precomputed diffs for this candidate
                local_precomputed_diffs = {
                    'train_cand': diff_train_cand,
                    'cand_cand': diff_cand_cand,
                    'domain_cand': diff_domain_cand,
                    'train_domain': precomputed_diffs['train_domain']  # Reuse this one
                }
                
                # Compute aggregated score across all GPs
                score_list = [
                    acquisition_func(
                        gp_t,
                        params_t,
                        precomputed_diffs=local_precomputed_diffs,
                        n_integration_points = n_integration_points
                    )
                    for gp_t, params_t in zip(gp, params)
                ]
                return -np.mean([s[0] if isinstance(s, np.ndarray) else s for s in score_list])
            
            # --- Local optimization loop ---
            best_x, best_score = None, -np.inf
            for start_point in top_starting_points:
                if all_unbounded:
                    res = minimize(fun=objective_function, x0=start_point, method="BFGS")
                else:
                    res = minimize(fun=objective_function, x0=start_point, method="L-BFGS-B", bounds=bounds)
                
                if -res.fun > best_score:
                    best_score, best_x = -res.fun, res.x
            
            best_x = best_x.reshape(1, -1)
            selected_points.append(best_x)
        else:
            top_index = np.argmax(cand_scores.ravel())
            best_x = candidate_points[top_index].reshape(1, -1)
            selected_points.append(best_x)
            print(best_x)
            candidate_points = np.delete(candidate_points, top_index, axis=0)

        # Augment each GP with predicted outputs at best_x
        for t, gp_t in enumerate(gp):
            params_t = params[t]
            y_pred = gp_t.predict(best_x, params_t, calc_cov=False, return_deriv=True)
            y_to_add = _format_y_to_add(y_pred)

            x_aug = np.vstack([gp_t.x_train_input, best_x])
            y_aug = []
            for i_y, y_block in enumerate(gp_t.y_train_input):
                if i_y < len(y_to_add):
                    y_aug.append(np.vstack([y_block, y_to_add[i_y]]))
                else:
                    y_aug.append(y_block)

            gp[t] = gp_t.__class__(
                x_aug, y_aug,
                gp_t.n_order,
                n_bases=gp_t.n_bases,
                der_indices=gp_t.der_indices,
                normalize=gp_t.normalize,
                kernel=gp_t.kernel,
                kernel_type=gp_t.kernel_type,
            )

    return np.vstack(selected_points)

# def sample_from_dist(dist, means, var, shape, scale, size):
#     num_cols = len(dist)
#     num_rows = int(size)
#     samples = np.zeros((num_rows, num_cols))
#     for i in range(0, num_cols):
#         if dist[i] == 'N':  # normal (Gaussian)
#             samples[:, i] = stats.norm.rvs(
#                 loc=shape[i], scale=scale[i], size=size)
#         elif dist[i] == 'U':  # uniform
#             samples[:, i] = stats.uniform.rvs(
#                 loc=shape[i], scale=scale[i], size=size)
#         elif dist[i] == 'LN':  # log-normal
#             samples[:, i] = stats.lognorm.rvs(
#                 scale[i], scale=np.exp(shape[i]), size=size)
#         # elif dist[i] == 'B':  # beta
#         #     frozen_dist = stats.beta(a=shape[i], b=scale[i])
#         #     transformed_samples[:, i] = frozen_dist.ppf(samples[:, i])
#         elif dist[i] == 'T':  # triangle (symmetric)
#             c = .5
#             samples[:, i] = stats.triang.rvs(
#                 c, loc=shape[i], scale=scale[i], size=size)
#         elif dist[i] == 'UTN':
#             # samples[:, i] = stats.uniform.rvs(
#             #     loc=shape[i], scale=scale[i] - shape[i], size=size)
#             cf = 3
#             # lb = shape[i] + cf*shape[i]
#             # ub = scale[i] - cf*shape[i]
#             # transformed_samples[:, i] = (
#             #     ub - lb)*samples[:, i] + lb
#             a, b = (shape[i] - means[i]) / \
#                 (cf*var[i]*means[i]), (scale[i] -
#                                        means[i]) / (cf*var[i]*means[i])
#             samples[:, i] = stats.truncnorm.rvs(
#                 a, b, loc=means[i], scale=cf*var[i]*means[i], size=size)

#     return samples
def get_pdf_params(dist_params):
    """
    Computes scipy-specific loc/scale parameters. Prioritizes explicit bounds if provided.
    """
    dists = dist_params['dists']
    nVar = len(dists)
    rv_a = np.zeros(nVar)  # loc
    rv_b = np.zeros(nVar)  # scale

    # Check for optional explicit bounds
    has_bounds = 'lower_bounds' in dist_params and 'upper_bounds' in dist_params

    for i, dist_name in enumerate(dists):
        # --- Logic for when explicit bounds ARE provided ---
        if has_bounds and dist_name in ['U', 'TN']:
            lb = dist_params['lower_bounds'][i]
            ub = dist_params['upper_bounds'][i]
            if dist_name == 'U':
                rv_a[i] = lb       # loc = lower bound
                rv_b[i] = ub - lb  # scale = width
            elif dist_name == 'TN':
                rv_a[i] = dist_params['means'][i]
                rv_b[i] = np.sqrt(dist_params['vars'][i])
        
        # --- Fallback logic when bounds are NOT provided ---
        else:
            stdev = np.sqrt(dist_params['vars'][i])
            mean = dist_params['means'][i]
            if dist_name == 'U':
                rv_a[i] = mean - np.sqrt(3) * stdev
                rv_b[i] = 2 * np.sqrt(3) * stdev
            elif dist_name == 'N':
                rv_a[i] = mean
                rv_b[i] = stdev
            # (Add other fallbacks here if needed)
            
    return rv_a, rv_b

def get_optimization_bounds(dist_params):
    """
    Determines optimization bounds. Prioritizes explicit bounds if provided.
    """
    bounds = []
    has_bounds = 'lower_bounds' in dist_params and 'upper_bounds' in dist_params

    for i, dist_name in enumerate(dist_params['dists']):
        # --- Use explicit bounds if available for bounded distributions ---
        if has_bounds and dist_name in ['U', 'TN', 'B']:
            bounds.append((dist_params['lower_bounds'][i], dist_params['upper_bounds'][i]))
        
        # --- Fallback logic or for other distributions ---
        elif dist_name == 'B': # Beta is on [0, 1] by default
            bounds.append((0, 1))
        elif dist_name in ['N', 'LN']: # Unbounded
            bounds.append((None, None))
        else: # Fallback for U/TN if bounds were not explicitly given
            stdev = np.sqrt(dist_params['vars'][i])
            mean = dist_params['means'][i]
            if dist_name == 'U':
                lb = mean - np.sqrt(3) * stdev
                ub = mean + np.sqrt(3) * stdev
                bounds.append((lb, ub))
            elif dist_name == 'TN':
                lb = mean - 3 * stdev
                ub = mean + 3 * stdev
                bounds.append((lb, ub))
            else:
                bounds.append((None, None))
    return bounds


def get_inverse(dist_params, samples):
    """
    Transforms uniform samples to a specified distribution via inverse CDF.
    """
    dists = dist_params['dists']
    loc_params, scale_params = get_pdf_params(dist_params)
    has_bounds = 'lower_bounds' in dist_params and 'upper_bounds' in dist_params
    
    transformed_samples = np.zeros(samples.shape)
    for i in range(samples.shape[1]):
        dist_name = dists[i]
        loc = loc_params[i]
        scale = scale_params[i]

        if dist_name == 'TN':
            mean = dist_params['means'][i]
            stdev = np.sqrt(dist_params['vars'][i])
            if has_bounds:
                lb = dist_params['lower_bounds'][i]
                ub = dist_params['upper_bounds'][i]
            else: # Fallback
                lb = mean - 3 * stdev
                ub = mean + 3 * stdev
            
            a, b = (lb - mean) / stdev, (ub - mean) / stdev
            frozen_dist = stats.truncnorm(a, b, loc=mean, scale=stdev)
        
        elif dist_name == 'U':
            frozen_dist = stats.uniform(loc=loc, scale=scale)
        
        elif dist_name == 'N':
            frozen_dist = stats.norm(loc=loc, scale=scale)

        # (Add other distributions here)

        transformed_samples[:, i] = frozen_dist.ppf(samples[:, i])
    return transformed_samples

# (The find_next_point function does not need to be changed from the previous version)