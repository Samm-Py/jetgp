import numpy as np
import pyoti.core as coti
from matplotlib import pyplot as plt
import pyoti.sparse as oti
import itertools
from line_profiler import profile


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


def transform_nested_list_row(nested_list):
    # If `nested_list` is not a list, just return it as is (base case).
    if not isinstance(nested_list, list):
        return nested_list

    # If we have a list of length 2 (e.g. [k, v]),
    # and both k and v are integers, apply the condition.
    if len(nested_list) == 2 and all(
        (isinstance(x, np.uint16) or isinstance(x, int)) for x in nested_list
    ):
        k, v = nested_list
        # If k == 1, leave it alone; otherwise transform k -> 2*k - 1.
        if k == 1:
            return [k, v]
        else:
            return [2 * k - 1, v]

    # Otherwise, recursively apply transformation to each element
    # in this list.
    return [transform_nested_list_row(item) for item in nested_list]


def transform_nested_list_column(nested_list):
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
        return [2 * k, v]  # Transform k -> 2*k

    # Otherwise, apply the transformation to each element recursively.
    return [transform_nested_list_column(item) for item in nested_list]


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
    len_array = 0
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

    ind = [0] * order

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


# @profile
def rbf_kernel_der_params(
    differences,
    length_scales,
    n_order,
    n_bases,
    der_indices,
    kernel_func,
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

    indices_row = transform_nested_list_row(der_indices)
    indices_column = transform_nested_list_column(der_indices)
    rows = []
    columns = []
    for i in range(0, len(indices_row)):
        for j in range(0, len(indices_row[i])):
            rows.append(indices_row[i][j])
            columns.append(indices_column[i][j])

    k = 2 * n_bases + 2
    for i in range(0, len(rows) + 1):
        row_j = 0
        for j in range(0, len(columns) + 1):
            if j == 0 and i == 0:
                row_j = phi.get_deriv([[k, 1]])
            elif j > 0 and i == 0:
                row_j = np.hstack(
                    (row_j, phi.get_deriv(rows[j - 1] + [[k, 1]]))
                )
            elif j == 0 and i > 0:
                row_j = phi.get_deriv(columns[i - 1] + [[k, 1]])
            else:
                row_j = np.hstack(
                    (
                        row_j,
                        phi.get_deriv(rows[j - 1] + columns[i - 1] + [[k, 1]]),
                    )
                )
        if i == 0:
            K = row_j
        else:
            K = np.vstack((K, row_j))
    return K


def rbf_kernel_weighted_testing(
    differences,
    length_scales,
    n_order,
    n_bases,
    der_indices,
    kernel_func,
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

    indices_row = transform_nested_list_row(der_indices)
    indices_column = transform_nested_list_column(der_indices)
    rows = []
    columns = []
    for i in range(0, len(indices_row)):
        for j in range(0, len(indices_row[i])):
            rows.append(indices_row[i][j])
            columns.append(indices_column[i][j])

    for i in range(0, len(rows) + 1):
        row_j = 0
        for j in range(0, len(columns) + 1):
            if j == 0 and i == 0:
                row_j = phi.real
            elif j > 0 and i == 0:
                row_j = np.hstack(
                    (
                        row_j,
                        phi[:, index[0]: index[-1] + 1].get_deriv(
                            rows[j - 1]
                        ),
                    )
                )
            elif j == 0 and i > 0:
                row_j = phi[index[0]: index[-1] + 1, :].get_deriv(
                    columns[i - 1]
                )
            else:
                row_j = np.hstack(
                    (
                        row_j,
                        np.array(
                            phi[
                                index[0]: index[-1] + 1,
                                index[0]: index[-1] + 1,
                            ].get_deriv(rows[j - 1] + columns[i - 1])
                        ),
                    )
                )
        if i == 0:
            K = row_j
        else:
            K = np.vstack((K, row_j))

    return K


def rbf_kernel_weighted(
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


def rbf_kernel_directional(
    X1, X2, length_scales, n_order, n_bases, der_indices, kernel_func, index=-1
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
    phi = kernel_func(X1, X2, length_scales, n_order, index)

    indices_row = transform_nested_list_row(der_indices)
    indices_column = transform_nested_list_column(der_indices)
    rows = []
    columns = []
    for i in range(0, len(indices_row)):
        for j in range(0, len(indices_row[i])):
            rows.append(indices_row[i][j])
            columns.append(indices_column[i][j])

    for i in range(0, len(rows) + 1):
        row_j = 0
        for j in range(0, len(columns) + 1):
            if j == 0 and i == 0:
                row_j = phi.real
            elif j > 0 and i == 0:
                row_j = np.hstack(
                    (
                        row_j,
                        phi[:, 0].get_deriv(rows[j - 1]),
                    )
                )
            elif j == 0 and i > 0:
                row_j = phi[0, :].get_deriv(columns[i - 1])
            else:
                row_j = np.hstack(
                    (
                        row_j,
                        np.array(
                            phi[
                                0,
                                0,
                            ].get_deriv(rows[j - 1] + columns[i - 1])
                        ).reshape(-1, 1),
                    )
                )
        if i == 0:
            K = row_j
        else:
            K = np.vstack((K, row_j))

    return K


def determine_directional_weights(
    rays, x, length_scales, n_order, n_bases, der_indices, kernel_func
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

    index = [0]
    n1, d = rays.T.shape
    K, r = kernel_func(rays, x, length_scales, n_order, index)

    F = np.ones((n1, 1))
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


def make_plots(
    X_train,
    y_train,
    X_test,
    y_pred,
    true_function,
    cov=-1,
    X1_grid=0,
    X2_grid=0,
    n_order=1,
    n_bases=1,
    plot_derivative_surrogates=False,
    der_indices=[],
):
    if plot_derivative_surrogates:
        assert (
            plot_derivative_surrogates and y_pred.shape[0] > X_test.shape[0]
        ), "Can not plot derivative information without returning values from gaussian proccess"
        assert (
            len(der_indices) != 0 and plot_derivative_surrogates
        ), "Must pass derivative indices to plot derivative surrogates"
    # Standard deviation is the square root of the diagonal of the covariance matrix

    assert X_train.shape[1] <= 400, "Plots not implemented for dimension >= 2"

    if isinstance(cov, int):
        cov = np.zeros((y_pred.shape[0], y_pred.shape[0]))
    sigma = np.sqrt(abs(np.diag(cov)))
    if not plot_derivative_surrogates:
        if X_train.shape[1] == 1:
            sigma = np.sqrt(abs(np.diag(cov)))

            if X_train.shape[1] == 1:
                # ----- Plot Results -----
                X_test_pert = oti.array(X_test)
                for i in range(1, n_bases + 1):
                    X_test_pert[:, i - 1] = X_test_pert[:, i - 1] + oti.e(
                        i, order=n_order
                    )

                true_values = true_function(X_test_pert, alg=oti).real
                plt.figure(0, figsize=(12, 6))
                plt.plot(
                    X_test,
                    true_values,  # Evaluate the same function with numpy
                    "r--",
                    label="True function",
                )

                # Plot the training points. Note that 'y_train[:num_points]' contains
                # the original function values (as opposed to the stacked derivative entries).
                plt.scatter(
                    X_train,
                    y_train[: X_train.shape[0]],
                    color="k",
                    label="Training points",
                )

                # GP mean prediction
                plt.plot(
                    X_test,
                    y_pred[: X_test.shape[0]],
                    "b",
                    label="GP Prediction",
                )

                # 95% confidence interval (approx. ±1.96 * std dev)
                plt.fill_between(
                    X_test.ravel(),
                    y_pred[0: X_test.shape[0]]
                    - 1.96 * sigma[0: X_test.shape[0]],
                    y_pred[0: X_test.shape[0]]
                    + 1.96 * sigma[0: X_test.shape[0]],
                    color="b",
                    alpha=0.2,
                    label="95% Confidence Interval",
                )

                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                plt.tight_layout(
                    pad=2.0
                )  # Adjust the layout to make room for the legend
                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.title(
                    "Order {} Enhanced Gaussian Process\n True Function Prediction".format(
                        n_order
                    )
                )
                plt.show()
                # rmse = np.sqrt(
                #     np.mean(
                #         (y_pred[: X_test.shape[0]] - true_values.flatten())
                #         ** 2
                #     )
                # )
                # print("RMSE between model and true function: {}".format(rmse))
        else:
            # Reshape the predicted mean back into a 2D grid for contour plotting.

            # Compute the min and max values of the grids
            x1_min, x1_max = np.min(X1_grid), np.max(X1_grid)
            x2_min, x2_max = np.min(X2_grid), np.max(X2_grid)

            # Create a mask for points within the grid bounds
            mask = (
                (X_train[:, 0] > x1_min)
                & (X_train[:, 0] < x1_max)
                & (X_train[:, 1] > x2_min)
                & (X_train[:, 1] < x2_max)
            )

            # Filtered points
            X_filtered = X_train[mask]
            N_grid = X1_grid.shape[0]
            num_points = X_test.shape[0]
            y_pred = y_pred[:num_points]
            f_mean_2d = y_pred.reshape(N_grid, N_grid)
            X_test_pert = oti.array(X_test)
            for i in range(1, n_bases + 1):
                X_test_pert[:, i - 1] = X_test_pert[:, i - 1] + oti.e(
                    i, order=n_order
                )
            # Compute the true function values on the test grid (using numpy operations for a plain evaluation)
            true_values = true_function(X_test_pert, alg=oti).real.reshape(
                (N_grid, N_grid)
            )

            # ----- Plotting the Results -----
            # Create a figure with two subplots: one for the GP prediction and one for the true function.
            plt.figure(figsize=(8, 6))
            plt.rcParams.update({"font.size": 12})

            # Subplot (a): GP Prediction
            vmin = min(f_mean_2d.min(), true_values.min())
            vmax = max(f_mean_2d.max(), true_values.max())
            # plt.subplot(1, 2, 1)
            # plt.title(
            #     "Order {} Enhanced Gaussian Process\n True Function Prediction".format(
            #         n_order
            #     )
            # )
            # Contour plot of the GP predicted mean
            # plt.subplot(1, 2, 1)
            contour1 = plt.contourf(
                X1_grid,
                X2_grid,
                f_mean_2d,
                levels=50,
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            plt.colorbar(contour1)
            plt.scatter(
                X_filtered[:, 0],
                X_filtered[:, 1],
                c="white",
                edgecolors="k",
                label="Train pts",
            )
            # plt.xlabel("X1")
            # plt.ylabel("X2")
            plt.xticks([])  # no x ticks
            plt.yticks([])  # no y ticks
            # plt.legend()
            plt.tight_layout(pad=2.0)

            # # Subplot (b)
            # plt.subplot(1, 2, 2)
            # plt.title("True Function", fontsize=12)
            # contour2 = plt.contourf(
            #     X1_grid,
            #     X2_grid,
            #     true_values,
            #     levels=25,
            #     cmap="viridis",
            #     vmin=vmin,
            #     vmax=vmax,
            # )
            # plt.colorbar(contour2)
            # # Overlay the training points on the prediction plot
            # plt.scatter(
            #     X_filtered[:, 0], X_filtered[:, 1], c="white", edgecolors="k"
            # )
            # plt.xticks([])  # no x ticks
            # plt.yticks([])  # no y ticks
            # plt.show()

            # plt.tight_layout(pad=2.0)

            # ----- Performance Evaluation -----
            # Compute the root mean squared error (RMSE) between the GP prediction and the true function.
            rmse = np.sqrt(np.mean((f_mean_2d - true_values) ** 2))
            print("RMSE between model and true function: {}".format(rmse))
    else:
        sigma = np.sqrt(abs(np.diag(cov)))

        if X_train.shape[1] == 1:
            # ----- Plot Results -----
            X_test_pert = oti.array(X_test)
            for i in range(1, n_bases + 1):
                X_test_pert[:, i - 1] = X_test_pert[:, i - 1] + oti.e(
                    i, order=n_order
                )

            plt.figure(0, figsize=(12, 6))
            true_values = true_function(X_test_pert, alg=oti).real

            plt.plot(
                X_test,
                true_values,  # Evaluate the same function with numpy
                "r--",
                label="True function",
            )

            # Plot the training points. Note that 'y_train[:num_points]' contains
            # the original function values (as opposed to the stacked derivative entries).
            plt.scatter(
                X_train,
                y_train[: X_train.shape[0]],
                color="k",
                label="Training points",
            )

            # GP mean prediction
            plt.plot(
                X_test, y_pred[: X_test.shape[0]], "b", label="GP Prediction"
            )

            # 95% confidence interval (approx. ±1.96 * std dev)
            plt.fill_between(
                X_test.ravel(),
                y_pred[0: X_test.shape[0]]
                - 1.96 * sigma[0: X_test.shape[0]],
                y_pred[0: X_test.shape[0]]
                + 1.96 * sigma[0: X_test.shape[0]],
                color="b",
                alpha=0.2,
                label="95% Confidence Interval",
            )

            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

            plt.tight_layout(
                pad=2.0
            )  # Adjust the layout to make room for the legend
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title(
                "Order {} Enhanced Gaussian Process\n True Function Prediction".format(
                    n_order
                )
            )
            plt.show()

            rmse = np.sqrt(
                np.mean(
                    (y_pred[: X_test.shape[0]] - true_values.flatten()) ** 2
                )
            )
            print("RMSE between model and true function: {}".format(rmse))

            # Plot the true function (using the non-OTI version of exp, sin, cos, etc.)
            for i in range(len(der_indices)):
                for j in range(len(der_indices[i])):
                    plt.figure(i + 1, figsize=(12, 6))
                    true_values = true_function(
                        X_test_pert, alg=oti
                    ).get_deriv(der_indices[i][j])
                    plt.plot(
                        X_test,
                        true_values,  # Evaluate the same function with numpy
                        "r--",
                        label="True function\nderivative index {}".format(
                            der_indices[i][j]
                        ),
                    )

                    # Plot the training points. Note that 'y_train[:num_points]' contains
                    # the original function values (as opposed to the stacked derivative entries).
                    plt.scatter(
                        X_train,
                        y_train[
                            (i + 1)
                            * X_train.shape[0]: (i + 2)
                            * X_train.shape[0]
                        ],
                        color="k",
                        label="Training points",
                    )

                    # GP mean prediction
                    plt.plot(
                        X_test,
                        y_pred[
                            (i + 1)
                            * X_test.shape[0]: (i + 2)
                            * X_test.shape[0]
                        ],
                        "b",
                        label="GP Prediction",
                    )

                    # 95% confidence interval (approx. ±1.96 * std dev)
                    plt.fill_between(
                        X_test.ravel(),
                        y_pred[
                            (i + 1)
                            * X_test.shape[0]: (i + 2)
                            * X_test.shape[0]
                        ]
                        - 1.96
                        * sigma[
                            (i + 1)
                            * X_test.shape[0]: (i + 2)
                            * X_test.shape[0]
                        ],
                        y_pred[
                            (i + 1)
                            * X_test.shape[0]: (i + 2)
                            * X_test.shape[0]
                        ]
                        + 1.96
                        * sigma[
                            (i + 1)
                            * X_test.shape[0]: (i + 2)
                            * X_test.shape[0]
                        ],
                        color="b",
                        alpha=0.2,
                        label="95% Confidence Interval",
                    )

                    # Place the legend outside the plot area (to the right)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                    plt.tight_layout(
                        pad=2.0
                    )  # Adjust the layout to make room for the legend
                    plt.xlabel("x")
                    plt.ylabel("f(x)")
                    plt.title(
                        "Order {0} Enhanced Gaussian Process\n True Function Derivative {1} Prediction".format(
                            n_order, der_indices[i][j]
                        )
                    )
                    plt.show()
                    rmse = np.sqrt(
                        np.mean(
                            (
                                y_pred[
                                    (i + 1)
                                    * X_test.shape[0]: (i + 2)
                                    * X_test.shape[0]
                                ]
                                - true_values.flatten()
                            )
                            ** 2
                        )
                    )
                    print(
                        "RMSE between model and derivative {0} of true function: {1}".format(
                            der_indices[i][j], rmse
                        )
                    )
        else:
            # Reshape the predicted mean back into a 2D grid for contour plotting.
            N_grid = X1_grid.shape[0]
            num_points = X_test.shape[0]
            y_pred_real = y_pred[:num_points]
            f_mean_2d = y_pred_real.reshape(N_grid, N_grid)
            X_test_pert = oti.array(X_test)
            for i in range(1, n_bases + 1):
                X_test_pert[:, i - 1] = X_test_pert[:, i - 1] + oti.e(
                    i, order=n_order
                )
            # Compute the true function values on the test grid (using numpy operations for a plain evaluation)
            true_values = true_function(X_test_pert, alg=oti).real.reshape(
                (N_grid, N_grid)
            )

            # ----- Plotting the Results -----
            # Create a figure with two subplots: one for the GP prediction and one for the true function.
            plt.figure(0, figsize=(12, 6))

            # Subplot (a): GP Prediction
            plt.subplot(1, 2, 1)
            plt.title(
                "Order {} Enhanced Gaussian Process\n True Function Prediction".format(
                    n_order
                )
            )
            # Contour plot of the GP predicted mean
            plt.contourf(
                X1_grid, X2_grid, f_mean_2d, levels=50, cmap="viridis"
            )
            plt.colorbar()
            # Overlay the training points on the prediction plot
            plt.scatter(
                X_train[:, 0],
                X_train[:, 1],
                c="white",
                edgecolors="k",
                label="Train pts",
            )
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()
            plt.tight_layout(pad=2.0)

            # Subplot (b): True Function
            plt.subplot(1, 2, 2)
            title_str = r"True Function"
            plt.title(title_str, fontsize=12)
            # Contour plot of the true function values
            plt.contourf(
                X1_grid, X2_grid, true_values, levels=50, cmap="viridis"
            )
            plt.colorbar()
            # Overlay the training points on the true function plot
            plt.scatter(
                X_train[:, 0], X_train[:, 1], c="white", edgecolors="k"
            )
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.show()

            plt.tight_layout(pad=2.0)

            # ----- Performance Evaluation -----
            # Compute the root mean squared error (RMSE) between the GP prediction and the true function.
            rmse = np.sqrt(np.mean((f_mean_2d - true_values) ** 2))
            print("RMSE between model and true function: {}".format(rmse))

            # Plot the true function (using the non-OTI version of exp, sin, cos, etc.)
            der_index_counter = 0
            for i in range(len(der_indices)):
                for j in range(len(der_indices[i])):
                    predicted_values = y_pred[
                        (der_index_counter + 1)
                        * X_test.shape[0]: (der_index_counter + 2)
                        * X_test.shape[0]
                    ]
                    f_mean_2d = predicted_values.reshape(N_grid, N_grid)
                    plt.figure(der_index_counter + 1, figsize=(12, 6))
                    true_values = (
                        true_function(X_test_pert, alg=oti)
                        .get_deriv(der_indices[i][j])
                        .reshape(N_grid, N_grid)
                    )

                    # ----- Plotting the Results -----
                    # Create a figure with two subplots: one for the GP prediction and one for the true function.

                    # Subplot (a): GP Prediction
                    plt.subplot(1, 2, 1)
                    plt.title(
                        "Order {0} Enhanced Gaussian Process\n True Function Derivative {1} Prediction".format(
                            n_order, der_indices[i][j]
                        )
                    )
                    # Contour plot of the GP predicted mean
                    plt.contourf(
                        X1_grid, X2_grid, f_mean_2d, levels=50, cmap="viridis"
                    )
                    plt.colorbar()
                    # Overlay the training points on the prediction plot
                    plt.scatter(
                        X_train[:, 0],
                        X_train[:, 1],
                        c="white",
                        edgecolors="k",
                        label="Train pts",
                    )
                    plt.xlabel("X1")
                    plt.ylabel("X2")
                    plt.legend()
                    plt.tight_layout(pad=2.0)

                    # Subplot (b): True Function
                    plt.subplot(1, 2, 2)
                    plt.title(
                        "True Function Derivative {0}".format(
                            der_indices[i][j]
                        )
                    )
                    # Contour plot of the true function values
                    plt.contourf(
                        X1_grid,
                        X2_grid,
                        true_values,
                        levels=50,
                        cmap="viridis",
                    )
                    plt.colorbar()
                    # Overlay the training points on the true function plot
                    plt.scatter(
                        X_train[:, 0], X_train[:, 1], c="white", edgecolors="k"
                    )
                    plt.xlabel("X1")
                    plt.ylabel("X2")
                    plt.show()

                    plt.tight_layout(pad=2.0)

                    # ----- Performance Evaluation -----
                    # Compute the root mean squared error (RMSE) between the GP prediction and the true function.
                    rmse = np.sqrt(np.mean((f_mean_2d - true_values) ** 2))
                    print(
                        "RMSE between model and true function derivative {0} : {1}".format(
                            der_indices[i][j], rmse
                        )
                    )

                    der_index_counter = der_index_counter + 1


def make_submodel_plots(
    X_train,
    y_train,
    X_test,
    y_pred,
    true_function,
    cov=-1,
    X1_grid=0,
    X2_grid=0,
    n_order=1,
    n_bases=1,
    plot_submodels=False,
    submodel_vals=[],
    submodel_cov=[],
):
    if plot_submodels:
        assert (
            len(submodel_vals) != 0
        ), "Can not plot submodels without returning values from gaussian proccess"

        if len(submodel_cov) == 0:
            for i in range(len(submodel_vals)):
                submodel_cov.append(
                    np.zeros((y_pred.shape[0], y_pred.shape[0]))
                )
    # Standard deviation is the square root of the diagonal of the covariance matrix

    assert X_train.shape[1] <= 400, "Plots not implemented for dimension >= 2"

    if isinstance(cov, int):
        cov = np.zeros((y_pred.shape[0], y_pred.shape[0]))
    sigma = np.sqrt(abs(np.diag(cov)))
    if not plot_submodels:
        if X_train.shape[1] == 1:
            sigma = np.sqrt(abs(np.diag(cov)))

            if X_train.shape[1] == 1:
                true_values = true_function(X_test, alg=np)
                plt.figure(0, figsize=(12, 6))
                plt.plot(
                    X_test,
                    true_values,  # Evaluate the same function with numpy
                    "r--",
                    label="True function",
                )

                # Plot the training points. Note that 'y_train[:num_points]' contains
                # the original function values (as opposed to the stacked derivative entries).
                plt.scatter(
                    X_train,
                    y_train[: X_train.shape[0]],
                    color="k",
                    label="Training points",
                )

                # GP mean prediction
                plt.plot(
                    X_test,
                    y_pred[: X_test.shape[0]],
                    "b",
                    label="GP Prediction",
                )

                # 95% confidence interval (approx. ±1.96 * std dev)
                plt.fill_between(
                    X_test.ravel(),
                    y_pred[0: X_test.shape[0]]
                    - 1.96 * sigma[0: X_test.shape[0]],
                    y_pred[0: X_test.shape[0]]
                    + 1.96 * sigma[0: X_test.shape[0]],
                    color="b",
                    alpha=0.2,
                    label="95% Confidence Interval",
                )

                plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

                plt.tight_layout(
                    pad=2.0
                )  # Adjust the layout to make room for the legend
                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.title("Gaussian Process Regression Fit")
                plt.show()
                rmse = np.sqrt(
                    np.mean(
                        (y_pred[: X_test.shape[0]] - true_values.flatten())
                        ** 2
                    )
                )
                print("RMSE between model and true function: {}".format(rmse))
        else:
            # Reshape the predicted mean back into a 2D grid for contour plotting.
            N_grid = X1_grid.shape[0]
            f_mean_2d = y_pred.reshape(N_grid, N_grid)
            # Compute the true function values on the test grid (using numpy operations for a plain evaluation)
            true_values = true_function(X_test, alg=np).reshape(
                (N_grid, N_grid)
            )

            # ----- Plotting the Results -----
            # Create a figure with two subplots: one for the GP prediction and one for the true function.
            plt.figure(figsize=(12, 6))

            # Subplot (a): GP Prediction
            plt.subplot(1, 2, 1)
            plt.title(
                "Order {} Enhanced Gaussian Process\nTrue Function Prediction".format(
                    n_order
                )
            )
            # Contour plot of the GP predicted mean
            plt.contourf(
                X1_grid, X2_grid, f_mean_2d, levels=50, cmap="viridis"
            )
            plt.colorbar()
            # Overlay the training points on the prediction plot
            plt.scatter(
                X_train[:, 0],
                X_train[:, 1],
                c="white",
                edgecolors="k",
                label="Train pts",
            )
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()
            plt.tight_layout(pad=2.0)

            # Subplot (b): True Function
            plt.subplot(1, 2, 2)
            title_str = r"True Function"
            plt.title(title_str, fontsize=12)
            # Contour plot of the true function values
            plt.contourf(
                X1_grid, X2_grid, true_values, levels=50, cmap="viridis"
            )
            plt.colorbar()
            # Overlay the training points on the true function plot
            plt.scatter(
                X_train[:, 0], X_train[:, 1], c="white", edgecolors="k"
            )
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.show()

            plt.tight_layout(pad=2.0)

            # ----- Performance Evaluation -----
            # Compute the root mean squared error (RMSE) between the GP prediction and the true function.
            rmse = np.sqrt(np.mean((f_mean_2d - true_values) ** 2))
            print("RMSE between model and true function: {}".format(rmse))
    else:
        sigma = np.sqrt(abs(np.diag(cov)))

        if X_train.shape[1] == 1:
            true_values = true_function(X_test, alg=np)
            plt.figure(0, figsize=(12, 6))
            plt.plot(
                X_test,
                true_values,  # Evaluate the same function with numpy
                "r--",
                label="True function",
            )

            # Plot the training points. Note that 'y_train[:num_points]' contains
            # the original function values (as opposed to the stacked derivative entries).
            plt.scatter(
                X_train,
                y_train[: X_train.shape[0]],
                color="k",
                label="Training points",
            )

            # GP mean prediction
            plt.plot(
                X_test,
                y_pred[: X_test.shape[0]],
                "b",
                label="Weighted GP Prediction",
            )

            # 95% confidence interval (approx. ±1.96 * std dev)
            plt.fill_between(
                X_test.ravel(),
                y_pred[0: X_test.shape[0]]
                - 1.96 * sigma[0: X_test.shape[0]],
                y_pred[0: X_test.shape[0]]
                + 1.96 * sigma[0: X_test.shape[0]],
                color="b",
                alpha=0.2,
                label="95% Confidence Interval",
            )

            plt.legend(bbox_to_anchor=(1.0, 1), loc="upper left")

            plt.tight_layout(
                pad=2.0
            )  # Adjust the layout to make room for the legend
            plt.xlabel("x")
            plt.ylabel("f(x)")
            plt.title(
                "Order {} Enhanced Gaussian Process\nTrue Function Prediction".format(
                    n_order
                )
            )
            plt.show()
            rmse = np.sqrt(
                np.mean(
                    (y_pred[: X_test.shape[0]] - true_values.flatten()) ** 2
                )
            )
            print("RMSE between model and true function: {}".format(rmse))

            for i in range(0, len(submodel_vals)):
                sigma = np.sqrt(abs(np.diag(submodel_cov[i])))
                y_pred = submodel_vals[i]
                plt.figure(i + 1, figsize=(12, 6))
                plt.plot(
                    X_test,
                    true_values,  # Evaluate the same function with numpy
                    "r--",
                    label="True function",
                )

                # Plot the training points. Note that 'y_train[:num_points]' contains
                # the original function values (as opposed to the stacked derivative entries).
                plt.scatter(
                    X_train,
                    y_train[: X_train.shape[0]],
                    color="k",
                    label="Training points",
                )

                # GP mean prediction
                plt.plot(
                    X_test,
                    submodel_vals[i],
                    "b",
                    label="GP Prediction",
                )

                # 95% confidence interval (approx. ±1.96 * std dev)
                plt.fill_between(
                    X_test.ravel(),
                    y_pred[0: X_test.shape[0]]
                    - 1.96 * sigma[0: X_test.shape[0]],
                    y_pred[0: X_test.shape[0]]
                    + 1.96 * sigma[0: X_test.shape[0]],
                    color="b",
                    alpha=0.2,
                    label="95% Confidence Interval",
                )

                plt.legend(bbox_to_anchor=(1.0, 1), loc="upper left")

                plt.tight_layout(pad=2.0)
                plt.xlabel("x")
                plt.ylabel("f(x)")
                plt.title(
                    "Gaussian Process Regression Fit: Submodel {}".format(
                        i + 1
                    )
                )
                plt.show()

        else:
            # Reshape the predicted mean back into a 2D grid for contour plotting.
            N_grid = X1_grid.shape[0]
            f_mean_2d = y_pred.reshape(N_grid, N_grid)
            # Compute the true function values on the test grid (using numpy operations for a plain evaluation)
            true_values = true_function(X_test, alg=np).reshape(
                (N_grid, N_grid)
            )

            # ----- Plotting the Results -----
            # Create a figure with two subplots: one for the GP prediction and one for the true function.
            plt.figure(0, figsize=(12, 6))

            # Subplot (a): GP Prediction
            plt.subplot(1, 2, 1)
            plt.title(
                "Order {} Enhanced Gaussian Process\nTrue Function Prediction".format(
                    n_order
                )
            )
            # Contour plot of the GP predicted mean
            plt.contourf(
                X1_grid, X2_grid, f_mean_2d, levels=50, cmap="viridis"
            )
            plt.colorbar()
            # Overlay the training points on the prediction plot
            plt.scatter(
                X_train[:, 0],
                X_train[:, 1],
                c="white",
                edgecolors="k",
                label="Train pts",
            )
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.legend()
            plt.tight_layout(pad=2.0)

            # Subplot (b): True Function
            plt.subplot(1, 2, 2)
            title_str = r"True Function"
            plt.title(title_str, fontsize=12)
            # Contour plot of the true function values
            plt.contourf(
                X1_grid, X2_grid, true_values, levels=50, cmap="viridis"
            )
            plt.colorbar()
            # Overlay the training points on the true function plot
            plt.scatter(
                X_train[:, 0], X_train[:, 1], c="white", edgecolors="k"
            )
            plt.xlabel("X1")
            plt.ylabel("X2")
            plt.show()

            plt.tight_layout(pad=2.0)

            # ----- Performance Evaluation -----
            # Compute the root mean squared error (RMSE) between the GP prediction and the true function.
            nrmse_vals = nrmse(true_values, f_mean_2d)
            print(
                "NRMSE between model and true function: {}".format(nrmse_vals)
            )

            for i in range(0, len(submodel_vals)):
                sigma = np.sqrt(abs(np.diag(submodel_cov[i])))
                y_pred = submodel_vals[i]
                f_mean_2d = y_pred.reshape(N_grid, N_grid)
                # ----- Plotting the Results -----
                # Create a figure with two subplots: one for the GP prediction and one for the true function.
                plt.figure(i + 1, figsize=(12, 6))

                # Subplot (a): GP Prediction
                plt.subplot(1, 2, 1)
                plt.title(
                    "Order {0} Enhanced Gaussian Process\nSubmodel {1}, Datapoint {2}".format(
                        n_order, i + 1, X_train[i]
                    )
                )
                # Contour plot of the GP predicted mean
                plt.contourf(
                    X1_grid, X2_grid, f_mean_2d, levels=25, cmap="viridis"
                )
                plt.colorbar()
                # Overlay the training points on the prediction plot
                plt.scatter(
                    X_train[:, 0],
                    X_train[:, 1],
                    c="white",
                    edgecolors="k",
                    label="Train pts",
                )
                plt.xlabel("X1")
                plt.ylabel("X2")
                plt.legend()
                plt.tight_layout(pad=2.0)

                # Subplot (b): True Function
                plt.subplot(1, 2, 2)
                title_str = r"True Function"
                plt.title(title_str, fontsize=12)
                # Contour plot of the true function values
                plt.contourf(
                    X1_grid, X2_grid, true_values, levels=25, cmap="viridis"
                )
                plt.colorbar()
                # Overlay the training points on the true function plot
                plt.scatter(
                    X_train[:, 0], X_train[:, 1], c="white", edgecolors="k"
                )
                plt.xlabel("X1")
                plt.ylabel("X2")
                plt.show()

                plt.tight_layout(pad=2.0)


def make_weighted_directional_plots(
    X_train,
    y_train,
    X_test,
    y_pred,
    true_function,
    cov=-1,
    X1_grid=0,
    X2_grid=0,
    n_order=1,
    n_bases=1,
    plot_directional_ders=False,
    submodel_vals=[],
    thetas_list=[],
):
    N_grid = X1_grid.shape[0]
    f_mean_2d = y_pred.reshape(
        N_grid, N_grid
    )  # Reshape predictions for plotting

    # ----- Plotting -----
    # Evaluate the true function on the test grid.
    true_values = true_function(X_test, alg=np).reshape((N_grid, N_grid))

    plt.figure(figsize=(12, 5))

    # (a) Plot the GP predicted mean as a contour plot.
    plt.subplot(1, 2, 1)
    plt.title(
        "Order {} Directional Derivative Enhanced \n Gaussian Process Regression Prediction".format(
            n_order
        )
    )
    plt.contourf(X1_grid, X2_grid, f_mean_2d, levels=50, cmap="viridis")
    plt.colorbar()
    plt.scatter(
        X_train[:, 0],
        X_train[:, 1],
        c="white",
        edgecolors="k",
        label="Train pts",
    )
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.tight_layout()

    # (b) Plot the true function.
    plt.subplot(1, 2, 2)
    title_str = r"$f(x_1, x_2) = \cos(x_1) + \cos(x_2)$"
    plt.title(title_str, fontsize=12)
    plt.contourf(X1_grid, X2_grid, true_values, levels=50, cmap="viridis")
    plt.colorbar()
    plt.scatter(X_train[:, 0], X_train[:, 1], c="white", edgecolors="k")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

    plt.tight_layout()

    # Compute RMSE between GP prediction and true function.
    rmse = np.sqrt(np.mean((f_mean_2d - true_values) ** 2))
    print(
        "RMSE between Gaussian Process Prediction and True Function: {}".format(
            rmse
        )
    )

    # ----- 3D Surface Plot with Directional Derivative Rays -----
    # Create a 3D plot to visualize the GP response surface and the directional derivative rays.
    fig = plt.figure(0, figsize=(12, 5))
    ax = fig.add_subplot(projection="3d")

    # Plot the GP predicted surface.
    surf = ax.plot_surface(
        X1_grid, X2_grid, f_mean_2d, linewidth=0, zorder=1, alpha=0.7
    )

    # For plotting directional derivative rays:
    # Parameterize t for evaluating along each ray.
    t = np.linspace(0.0, 1.0, 1000)
    x_plot = []
    y_plot = []
    f_plot = []

    n_dim = X_test.shape[1]
    n_rays = len(
        thetas_list
    )  # Number of rays equals the number of angles provided.
    # Initialize an array to store the directional unit vectors (rays).
    rays = np.zeros((n_dim, n_rays))

    # Reset the thetas list for plotting directional rays.
    for i, theta in enumerate(thetas_list):
        rays[:, i] = [np.cos(theta), np.sin(theta)]
    nrays = rays.shape[1]

    # Compute elementary perturbations along each ray.
    e = [oti.e(i + 1, order=n_order) for i in range(nrays)]
    x_p, y_p = np.dot(rays, e)
    X = oti.array([x_p, y_p]).T
    f_p = true_function(X, alg=oti)[0, 0]
    # Evaluate the perturbation functions along the parameter t.
    for i in range(nrays):
        x_plot.append(x_p.rom_eval_object([i + 1], [t]) * np.ones(t.size))
        y_plot.append(y_p.rom_eval_object([i + 1], [t]) * np.ones(t.size))
        f_plot.append(f_p.rom_eval_object([i + 1], [t]) * np.ones(t.size))

    # Plot directional derivative lines on the 3D surface.
    lines = []
    labels = ["Gaussian Process\nResponse Surface"]
    proxy = plt.Line2D(
        [0],
        [0],
        linestyle="none",
        marker="s",
        markersize=10,
        color="tab:blue",
        alpha=0.5,
    )
    lines.append(proxy)  # Proxy for the GP surface
    for i in range(len(x_plot)):
        (line,) = ax.plot3D(
            x_plot[i],
            y_plot[i],
            f_plot[i],
            zorder=100,
            label=f"Directional Derivative $v_{i+1}$",
        )
        lines.append(line)
        labels.append(f"Directional Derivative $v_{i+1}$")

    # Place legend outside the 3D plot.
    ax.legend(
        lines,
        labels,
        loc="center left",
        bbox_to_anchor=(1.2, 0.5),
        frameon=False,
    )
    # Set axes labels.
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("f(x,y)")
    plt.tight_layout()
    plt.show()

    # Create figure

    for i in range(0, len(submodel_vals)):
        fig = plt.figure(i + 100, figsize=(12, 5))
        ax = fig.add_subplot(projection="3d")
        f_mean = submodel_vals[i]
        ax.plot_surface(
            X1_grid,
            X2_grid,
            f_mean.reshape((N_grid, N_grid)),
            linewidth=0,
            zorder=1,
            alpha=0.7,
        )

        # For plotting directional derivative rays
        t = np.linspace(0.0, 1, 1000)
        x_plot = []
        y_plot = []
        f_plot = []

        e = [oti.e(i + 1, order=n_order) for i in range(1)]
        x_p, y_p = np.dot(rays[:, i].reshape(-1, 1), e)

        X = oti.array([x_p, y_p]).T

        f_p = true_function(X, alg=oti)[0, 0]

        for comb in itertools.combinations(range(1, n_rays + 1), 2):
            # This removes any combination between directions e_i * e_j = 0
            # Truncate all possible combinations between imdirections.
            f_p = f_p.truncate(comb)
        for j in range(n_rays):
            x_plot.append(x_p.rom_eval_object([j + 1], [t]) * np.ones(t.size))
            y_plot.append(y_p.rom_eval_object([j + 1], [t]) * np.ones(t.size))
            f_plot.append(f_p.rom_eval_object([j + 1], [t]) * np.ones(t.size))

        # Plot directional derivative lines with labels
        lines = []
        labels = ["Gaussian Process\nResponse Surface"]
        proxy = plt.Line2D(
            [0],
            [0],
            linestyle="none",
            marker="s",
            markersize=10,
            color="tab:blue",
            alpha=0.5,
        )
        lines.append(proxy)  # Proxy for surface

        for j in range(len(x_plot)):
            (line,) = ax.plot3D(
                x_plot[j],
                y_plot[j],
                f_plot[j],
                zorder=100,
                label=f"Directional Derivative $v_{i+1}$",
            )
            if j == 0:
                lines.append(line)
                labels.append(f"Directional Derivative $v_{i+1}$")

        # Show legend in the margin
        ax.legend(
            lines,
            labels,
            loc="center left",
            bbox_to_anchor=(1.2, 0.5),
            frameon=False,
        )

        # Labels and formatting
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x,y)")
        plt.tight_layout()
        plt.show()


def make_directional_plots(
    X_train,
    y_train,
    X_test,
    y_pred,
    true_function,
    cov=-1,
    X1_grid=0,
    X2_grid=0,
    n_order=1,
    n_bases=1,
    plot_directional_ders=False,
    der_indices=[],
    thetas_list=[],
):
    if plot_directional_ders:
        assert (
            len(thetas_list) != 0 and plot_directional_ders
        ), "Must pass thetas to plot derivative surrogates"
    # Standard deviation is the square root of the diagonal of the covariance matrix

    assert X_train.shape[1] <= 2, "Plots not implemented for dimension >= 2"

    if isinstance(cov, int):
        cov = np.zeros((y_pred.shape[0], y_pred.shape[0]))
    if not plot_directional_ders:
        # Reshape the predicted mean back into a 2D grid for contour plotting.
        N_grid = X1_grid.shape[0]
        num_points = X_test.shape[0]
        y_pred = y_pred[:num_points]
        f_mean_2d = y_pred.reshape(N_grid, N_grid)
        X_test_pert = oti.array(X_test)
        for i in range(1, n_bases + 1):
            X_test_pert[:, i - 1] = X_test_pert[:, i - 1] + oti.e(
                i, order=n_order
            )
        # Compute the true function values on the test grid (using numpy operations for a plain evaluation)
        true_values = true_function(X_test_pert, alg=oti).real.reshape(
            (N_grid, N_grid)
        )

        # ----- Plotting the Results -----
        # Create a figure with two subplots: one for the GP prediction and one for the true function.
        plt.figure(figsize=(12, 6))

        # Subplot (a): GP Prediction
        plt.subplot(1, 2, 1)
        plt.title(
            "Order {} Directional Enhanced Gaussian Process\n True Function Prediction".format(
                n_order
            )
        )
        # Contour plot of the GP predicted mean
        plt.contourf(X1_grid, X2_grid, f_mean_2d, levels=50, cmap="viridis")
        plt.colorbar()
        # Overlay the training points on the prediction plot
        plt.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c="white",
            edgecolors="k",
            label="Train pts",
        )
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.tight_layout(pad=2.0)

        # Subplot (b): True Function
        plt.subplot(1, 2, 2)
        title_str = r"True Function"
        plt.title(title_str, fontsize=12)
        # Contour plot of the true function values
        plt.contourf(X1_grid, X2_grid, true_values, levels=50, cmap="viridis")
        plt.colorbar()
        # Overlay the training points on the true function plot
        plt.scatter(X_train[:, 0], X_train[:, 1], c="white", edgecolors="k")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()

        plt.tight_layout(pad=2.0)

        # ----- Performance Evaluation -----
        # Compute the root mean squared error (RMSE) between the GP prediction and the true function.
        rmse = np.sqrt(np.mean((f_mean_2d - true_values) ** 2))
        print("RMSE between model and true function: {}".format(rmse))
    else:
        # Compute the true function values on the test grid (using numpy operations for a plain evaluation)
        colors = ["tab:red", "tab:orange", "tab:green"]
        N_grid = X1_grid.shape[0]
        num_points = X_test.shape[0]
        true_values = true_function(X_test, alg=np).reshape((N_grid, N_grid))

        # ----- Plotting the Results -----
        # Create a figure with two subplots: one for the GP prediction and one for the true function.
        plt.figure(0, figsize=(12, 6))

        # Subplot (a): GP Prediction
        plt.subplot(1, 2, 1)
        plt.title(
            "Order {} Enhanced Gaussian Process\n True Function Prediction".format(
                n_order
            )
        )
        # Contour plot of the GP predicted mean
        f_mean_2d = y_pred[:num_points].reshape(N_grid, N_grid)
        plt.contourf(X1_grid, X2_grid, f_mean_2d, levels=50, cmap="viridis")
        plt.colorbar()
        # Overlay the training points on the prediction plot
        plt.scatter(
            X_train[:, 0],
            X_train[:, 1],
            c="white",
            edgecolors="k",
            label="Train pts",
        )
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.legend()
        plt.tight_layout(pad=2.0)

        # Subplot (b): True Function
        plt.subplot(1, 2, 2)
        title_str = r"True Function"
        plt.title(title_str, fontsize=12)
        # Contour plot of the true function values
        plt.contourf(X1_grid, X2_grid, true_values, levels=50, cmap="viridis")
        plt.colorbar()
        # Overlay the training points on the true function plot
        plt.scatter(X_train[:, 0], X_train[:, 1], c="white", edgecolors="k")
        plt.xlabel("X1")
        plt.ylabel("X2")
        plt.show()

        plt.tight_layout(pad=2.0)

        # ----- Performance Evaluation -----
        # Compute the root mean squared error (RMSE) between the GP prediction and the true function.
        rmse = np.sqrt(np.mean((f_mean_2d - true_values) ** 2))
        print("RMSE between model and true function: {}".format(rmse))

        # Plot the true function (using the non-OTI version of exp, sin, cos, etc.)
        fig = plt.figure(2, figsize=(12, 5))
        ax = fig.add_subplot(projection="3d")
        # Plot surface
        ax.plot_surface(
            X1_grid, X2_grid, f_mean_2d, linewidth=0, zorder=1, alpha=0.7
        )
        lines = []
        labels = ["Gaussian Process\nResponse Surface"]

        nrays = len(thetas_list)
        ndim = X_train.shape[1]
        order = n_order
        # Vector with the dimensions of the rays.
        rays = np.zeros((ndim, nrays))
        for i, theta in enumerate(thetas_list):
            rays[:, i] = [np.cos(theta), np.sin(theta)]

        nrays = rays.shape[1]

        for i in range(X_train.shape[0]):
            e = [oti.e(i + 1, order=order) for i in range(nrays)]
            x_p, y_p = np.dot(rays, e)

            # For plotting directional derivative rays
            t = np.linspace(-np.pi / 3, np.pi / 3, 1000)
            x_plot = []
            y_plot = []
            f_plot = []

            x_p = X_train[i, 0] + x_p
            y_p = X_train[i, 1] + y_p
            X = oti.array([x_p, y_p]).T
            f_p = true_function(X, alg=oti)[0, 0]
            for j in range(nrays):
                x_plot.append(
                    x_p.rom_eval_object([j + 1], [t]) * np.ones(t.size)
                )
                y_plot.append(
                    y_p.rom_eval_object([j + 1], [t]) * np.ones(t.size)
                )
                f_plot.append(
                    f_p.rom_eval_object([j + 1], [t]) * np.ones(t.size)
                )

            # Plot directional derivative lines with labels
            proxy = plt.Line2D(
                [0],
                [0],
                linestyle="none",
                marker="s",
                markersize=10,
                color="tab:blue",
                alpha=0.5,
            )
            if i == 0:
                lines.append(proxy)  # Proxy for surface
            if i == 0:
                for j in range(len(x_plot)):
                    (line,) = ax.plot3D(
                        x_plot[j],
                        y_plot[j],
                        f_plot[j],
                        zorder=100,
                        label=f"Directional Derivative $v_{i+1}$",
                        color=colors[j],
                    )
                    lines.append(line)
                    labels.append(f"Directional Derivative $v_{i+1}$")
            else:
                for j in range(len(x_plot)):
                    ax.plot3D(
                        x_plot[j],
                        y_plot[j],
                        f_plot[j],
                        zorder=100,
                        color=colors[j],
                    )

        # Show legend in the margin
        ax.legend(
            lines,
            labels,
            loc="center left",
            bbox_to_anchor=(1.2, 0.5),
            frameon=False,
        )

        # Labels and formatting
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("f(x,y)")
        plt.tight_layout()
        plt.show()

        e = [oti.e(i + 1, order=order) for i in range(nrays)]
        x_p, y_p = np.dot(rays, e)

        # For plotting directional derivative rays
        t = np.linspace(-np.pi / 3, np.pi / 3, 1000)
        x_plot = []
        y_plot = []
        f_plot = []

        x_p = X_test[:, 0] + x_p
        y_p = X_test[:, 1] + y_p
        X = oti.array([x_p, y_p]).T
        f_p = true_function(X, alg=oti)

        # ----- Plotting the Results -----
        # Create a figure with two subplots: one for the GP prediction and one for the true function.
        for j in range(0, len(der_indices)):
            for i, der_index in enumerate(der_indices[j]):
                plt.figure(i + 1000, figsize=(12, 6))

                # Subplot (a): GP Prediction
                plt.subplot(1, 2, 1)
                plt.title(
                    "Direction {0} Enhanced Gaussian Process\n True Function Prediction".format(
                        der_index
                    )
                )
                # Contour plot of the GP predicted mean
                f_mean_2d = y_pred[
                    (i + 1) * num_points: (i + 2) * num_points
                ].reshape(N_grid, N_grid)
                plt.contourf(
                    X1_grid, X2_grid, f_mean_2d, levels=50, cmap="viridis"
                )
                plt.colorbar()
                # Overlay the training points on the prediction plot
                plt.scatter(
                    X_train[:, 0],
                    X_train[:, 1],
                    c="white",
                    edgecolors="k",
                    label="Train pts",
                )
                plt.xlabel("X1")
                plt.ylabel("X2")
                plt.legend()
                plt.tight_layout(pad=2.0)

                # Subplot (b): True Function
                plt.subplot(1, 2, 2)
                title_str = r"True Function Directional Derivative {}".format(
                    der_index
                )
                plt.title(title_str, fontsize=12)
                # Contour plot of the true function values
                true_values = f_p.get_deriv(der_index).reshape(N_grid, N_grid)
                plt.contourf(
                    X1_grid,
                    X2_grid,
                    true_values,
                    levels=50,
                    cmap="viridis",
                )
                plt.colorbar()
                # Overlay the training points on the true function plot
                plt.scatter(
                    X_train[:, 0], X_train[:, 1], c="white", edgecolors="k"
                )
                plt.xlabel("X1")
                plt.ylabel("X2")
                plt.show()

                plt.tight_layout(pad=2.0)
                rmse = np.sqrt(
                    np.mean((f_mean_2d.flatten() - true_values.flatten()) ** 2)
                )
                print(
                    "RMSE between model and directional derivative {0} of true function: {1}".format(
                        der_index, rmse
                    )
                )
