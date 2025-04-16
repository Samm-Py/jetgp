import numpy as np
from matplotlib import pyplot as plt
import pyoti.sparse as oti
import utils


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
    y_train = utils.reshape_y_train(y_train)
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
    sigma = np.sqrt(cov).flatten()
    if not plot_derivative_surrogates:
        if X_train.shape[1] == 1:

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
                    y_pred[: X_test.shape[0]].reshape(-1, 1),
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

    else:
        sigma = np.sqrt(cov)

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
    sigma = np.sqrt(cov)
    if not plot_submodels:
        if X_train.shape[1] == 1:
            sigma = np.sqrt(cov)

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
    else:
        sigma = np.sqrt(cov)

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
                y_train[0][0][: X_train.shape[0]],
                color="k",
                label="Training points",
            )

            # GP mean prediction
            plt.plot(
                X_test,
                y_pred.flatten(),
                "b",
                label="Weighted GP Prediction",
            )

            # 95% confidence interval (approx. ±1.96 * std dev)
            plt.fill_between(
                X_test.ravel(),
                y_pred.flatten()
                - 1.96 * sigma[0: X_test.shape[0]],
                y_pred.flatten()
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

            for i in range(0, len(submodel_vals)):
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
                    y_train[0][0][: X_train.shape[0]],
                    color="k",
                    label="Training points",
                )

                # GP mean prediction
                plt.plot(
                    X_test,
                    submodel_vals[i].flatten(),
                    "b",
                    label="GP Prediction",
                )

                # 95% confidence interval (approx. ±1.96 * std dev)
                plt.fill_between(
                    X_test.ravel(),
                    y_pred.flatten()
                    - 1.96 * sigma[0: X_test.shape[0]],
                    y_pred.flatten()
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
                    "Order {0} Enhanced Gaussian Process\nSubmodel {1}, Index Set {2}".format(
                        n_order, i + 1, i
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
