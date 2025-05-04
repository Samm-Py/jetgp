import numpy as np
import pyoti.core as coti
import pyoti.sparse as oti
import sympy as sp


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
    noise_std_normalized = np.zeros(sigma_data.shape)
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


# def convert_noise_to_matrix(sigma_data, der_):
