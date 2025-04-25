import numpy as np
import pyoti.core as coti


def scale_samples(samples, lower_bounds, upper_bounds):
    """
    Scale each column of samples from [0, 1] to [lb_j, ub_j].

    Parameters:
        samples (ndarray): A (d, n) array of samples in [0, 1]^n.
        lower_bounds (array-like): Length-n array of lower bounds.
        upper_bounds (array-like): Length-n array of upper bounds.

    Returns:
        ndarray: A (d, n) array with each column scaled to its corresponding bounds.
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
    flattened_indices = []
    for i in range(0, len(indices)):
        for j in range(0, len(indices[i])):
            flattened_indices.append(indices[i][j])
    return flattened_indices


def nrmse(y_true, y_pred, norm_type="minmax"):
    """
    Compute the Normalized Root Mean Squared Error (NRMSE).

    Parameters:
        y_true (array-like): Ground truth values.
        y_pred (array-like): Predicted values.
        norm_type (str): Normalization type:
                         - 'minmax': divide by (max - min) of y_true
                         - 'mean': divide by mean of y_true
                         - 'std': divide by standard deviation of y_true

    Returns:
        float: NRMSE value.
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


def transform_nested_list(nested_list):
    """
    Recursively traverse a nested list. Whenever we encounter
    a two-element list [k, v] with both k and v as integers,
    transform k -> 2*k.
    """
    # If it's not a list, return it as is.
    if not isinstance(nested_list, list):
        return nested_list

    # If it's exactly two integers [k, v], apply the transformation.
    if len(nested_list) == 2 and all(
        (isinstance(x, np.uint16) or isinstance(x, int)) for x in nested_list
    ):
        k, v = nested_list
        return [k, v]  # Transform k -> 2*k

    # Otherwise, apply the transformation to each element recursively.
    return [transform_nested_list(item) for item in nested_list]


def convert_index_to_exponent_form(lst):
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
    companion_list = [0]
    for i in range(len(der_indices)):
        for j in range(0, len(der_indices[i])):
            companion_list.append(
                compare_OTI_indices(nvars, order, der_indices[i][j]))

    companion_array = np.array(companion_list)
    return companion_array


def compare_OTI_indices(nvars, order, term_check):
    """
    Generate list of lists that contain indices of bases in an OTI number
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
    Generate list of lists that contain indices of bases in an OTI number
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
    y_train_flattened = y_train[0]
    for i in range(1, len(y_train)):
        y_train_flattened = np.vstack(
            (y_train_flattened.reshape(-1, 1), y_train[i].reshape(-1, 1)))
    return y_train_flattened.flatten()


def transform_cov(cov, sigma_y, sigmas_x, der_indices, X_test):
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
    var = abs(np.diag(cov))
    y_var_normalized = np.zeros((var.shape[0],))
    y_var_normalized[0:X_test.shape[0]] = var[0:X_test.shape[0]]*sigma_y**2
    for i in range(len(der_indices)):
        factor = sigma_y**2
        y_var_normalized[(i+1) * X_test.shape[0]: (i+2) * X_test.shape[0]
                         ] = var[(i+1) * X_test.shape[0]: (i+2) * X_test.shape[0]]*factor
    return y_var_normalized


def transform_predictions(y_pred, mu_y, sigma_y, sigmas_x, der_indices, X_test):
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
    y_train_normalized = (
        mu_y + y_pred[0:X_test.shape[0]]*sigma_y).reshape(-1, 1)
    for i in range(len(der_indices)):
        factor = sigma_y
        y_train_normalized = np.vstack((y_train_normalized, (y_pred[(
            i + 1) * X_test.shape[0]: (i + 2) * X_test.shape[0]] * factor[0, 0]).reshape(-1, 1)))
    return y_train_normalized


def normalize_x_data_test(X_test, sigmas_x, mus_x):

    X_train_normalized = (X_test - mus_x) / sigmas_x

    return X_train_normalized


def normalize_x_data_train(X_train):
    mean_vec_x = np.mean(X_train, axis=0).reshape(1, -1)  # shape: (m, 1)
    std_vec_x = np.std(X_train, axis=0).reshape(1, -1)    # shape: (m, 1)

    X_train_normalized = (X_train - mean_vec_x) / std_vec_x

    return X_train_normalized


def normalize_directions(sigmas_x, rays):
    transformed_rays = np.zeros(rays.shape)
    for i in range(rays.shape[1]):
        transformed_rays[:, i] = np.diag(1/sigmas_x.flatten()) @ rays[:, i]
    return transformed_rays


def normalize_y_data(X_train, y_train, sigma_data, der_indices):
    mean_vec_x = np.mean(X_train, axis=0).reshape(1, -1)  # shape: (m, 1)
    std_vec_x = np.std(X_train, axis=0).reshape(1, -1)    # shape: (m, 1)

    mean_vec_y = np.mean(y_train[0], axis=0).reshape(-1, 1)  # shape: (m, 1)
    std_vec_y = np.std(y_train[0], axis=0).reshape(-1, 1)    # shape: (m, 1)

    y_train_normalized = (y_train[0] - mean_vec_y)/std_vec_y

    for i in range(len(der_indices)):
        factor = 1/std_vec_y
        for j in range(len(der_indices[i])):
            factor = factor * \
                (std_vec_x[0][der_indices[i][j][0] - 1]
                 )**(der_indices[i][j][1])
        y_train_normalized = np.vstack(
            (y_train_normalized.reshape(-1, 1), y_train[i + 1] * factor[0, 0]))

        # Scale noise if provided
    if sigma_data is not None:
        noise_std_normalized = sigma_data / std_vec_y[0, 0]
    else:
        noise_std_normalized = None

    return y_train_normalized.flatten(), mean_vec_y, std_vec_y, std_vec_x, mean_vec_x, noise_std_normalized


def normalize_y_data_directional(X_train, y_train, der_indices):
    mean_vec_x = np.mean(X_train, axis=0).reshape(1, -1)  # shape: (m, 1)
    std_vec_x = np.std(X_train, axis=0).reshape(1, -1)    # shape: (m, 1)

    mean_vec_y = np.mean(y_train[0], axis=0).reshape(-1, 1)  # shape: (m, 1)
    std_vec_y = np.std(y_train[0], axis=0).reshape(-1, 1)    # shape: (m, 1)

    y_train_normalized = (y_train[0] - mean_vec_y)/std_vec_y

    for i in range(len(der_indices)):
        factor = 1/std_vec_y
        y_train_normalized = np.vstack(
            (y_train_normalized.reshape(-1, 1), y_train[i + 1] * factor[0, 0]))
    return y_train_normalized.flatten(), mean_vec_y, std_vec_y, std_vec_x, mean_vec_x
