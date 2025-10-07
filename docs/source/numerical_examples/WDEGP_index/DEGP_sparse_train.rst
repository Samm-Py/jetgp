================================================================================
Sparse Derivative-Enhanced GP Tutorial: 1D Example
================================================================================

This tutorial demonstrates a **sparse, derivative-enhanced
Gaussian Process model**. Note the sparse implementation is a special case of the 
weighted derivative enhanced gaussian proccess model outlined in the theory section.
In particuar, we focus on the special case of creating a single
submodel with derivatives at :math:`m` training points out of :math:`n` total points 
:math:`(m \leq n)`.

This approach is useful for:
- Demonstrating a practical, computationally efficient DEGP implementation.
- Incorporating derivative information at selected points.
- Highlighting the connection to the general weighted DEGP framework.

.. contents::
   :local:
   :depth: 2

Setup
-----

We begin by importing necessary libraries, including `pyoti` for hypercomplex
automatic differentiation and standard scientific Python packages.

.. jupyter-execute::

    import numpy as np
    import pyoti.sparse as oti
    from wdegp.wdegp import wdegp
    import utils
    import plotting_helper
    from typing import List, Dict, Callable

Configuration
-------------

All configuration parameters are defined as simple variables. This includes
the number of training points, derivative order, kernel type, and other
hyperparameters.

.. jupyter-execute::

    n_order = 2
    n_bases = 1
    num_training_pts = 10
    test_points = 250
    lb_x, ub_x = 0.5, 2.5
    normalize_data = True
    kernel = "SE"
    kernel_type = "anisotropic"
    n_restarts = 15
    swarm_size = 200
    np.random.seed(42)

True Function
-------------

We define a simple 1D function for demonstration purposes. This function
will be used to generate both function values and derivatives.

.. jupyter-execute::

    def example_function(X, alg=np):
        x1 = X[:,0]
        return alg.sin(10 * np.pi * x1) / (2 * x1) + (x1 - 1)**4

Derivative Selection
--------------------

We select \(m\) points from the total \(n\) training points to include derivative
information, illustrating a sparse derivative-enhanced strategy.

.. jupyter-execute::

    derivative_indices = [2, 3, 4, 5]  # m points for derivative observations

Training Data Generation
------------------------

Training data is generated for all \(n\) points, and derivative information
is computed at the selected \(m\) points using pyOTI.

.. jupyter-execute::

    def generate_training_data(lb_x, ub_x, num_points, n_order, n_bases,
                               derivative_indices, true_function):
        X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1,1)
        y_real = true_function(X_train, alg=np)
        base_der_indices = utils.gen_OTI_indices(n_bases, n_order)
        y_train_data = []

        X_sub = oti.array(X_train[derivative_indices])
        for i in range(n_bases):
            for j in range(X_sub.shape[0]):
                X_sub[j, i] += oti.e(i+1, order=n_order)

        y_hc = oti.array([true_function(x, alg=oti) for x in X_sub])
        y_sub = [y_real.reshape(-1,1)]
        for i in range(len(base_der_indices)):
            for j in range(len(base_der_indices[i])):
                deriv = y_hc.get_deriv(base_der_indices[i][j]).reshape(-1,1)
                y_sub.append(deriv)

        y_train_data.append(y_sub)
        return X_train, y_train_data, base_der_indices

Model Construction
------------------

We create a single-submodel **weighted DEGP** using the prepared data.

.. jupyter-execute::

    def build_wdegp_model(X_train, y_train_data, n_order, n_bases,
                          derivative_indices, der_indices,
                          normalize=True, kernel="SE", kernel_type="anisotropic"):
        model = wdegp(
            X_train,
            y_train_data,
            n_order,
            n_bases,
            [derivative_indices],  # single submodel
            [der_indices],
            normalize=normalize,
            kernel=kernel,
            kernel_type=kernel_type
        )
        return model

Hyperparameter Optimization
---------------------------

.. jupyter-execute::

    def optimize_model(model, n_restarts=15, swarm_size=200):
        print("Optimizing hyperparameters...")
        params = model.optimize_hyperparameters(
            n_restart_optimizer=n_restarts,
            swarm_size=swarm_size, verbose = False
        )
        print("Optimization complete.")
        return params

Prediction and Evaluation
-------------------------

.. jupyter-execute::

    def predict_and_evaluate(model, true_function, lb_x, ub_x, test_points, params):
        X_test = np.linspace(lb_x, ub_x, test_points).reshape(-1,1)
        y_pred, y_cov, _, _ = model.predict(X_test, params, calc_cov=True, return_submodels=True)
        y_true = true_function(X_test, alg=np)
        nrmse = utils.nrmse(y_true, y_pred)
        print(f"NRMSE: {nrmse:.6f}")
        return {'X_test': X_test, 'y_pred': y_pred, 'y_true': y_true, 'y_cov': y_cov, 'nrmse': nrmse}

Enhanced Visualization
-----------------------

The following plotting function highlights the points where derivative
information is included. These points are shown in red and labeled with
their corresponding indices. This helps visualize how derivative data
constrains the GP model in specific regions.

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import numpy as np

    def plot_sparse_degps(
        X_train: np.ndarray,
        y_train_data: list,
        X_test: np.ndarray,
        y_pred: np.ndarray,
        y_true: np.ndarray,
        y_cov: np.ndarray,
        derivative_indices: list,
        true_function: callable,
        n_order: int,
        show_confidence: bool = True,
        figsize: tuple = (8, 5)
    ):
        """
        Plot sparse derivative-enhanced GP predictions with labeled derivative points.
        """
        plt.figure(figsize=figsize)

        # Predictive uncertainty (95% confidence interval)
        if show_confidence:
            std = np.sqrt(y_cov)
            plt.fill_between(
                X_test.flatten(),
                y_pred.flatten() - 2 * std,
                y_pred.flatten() + 2 * std,
                color="lightblue", alpha=0.4, label="95% Confidence Interval"
            )

        # True function
        plt.plot(X_test, y_true, "k--", lw=1.5, label="True function")

        # GP mean prediction
        plt.plot(X_test.flatten(), y_pred.flatten(), "b", lw=2, label="GP mean prediction")

        # Training points
        plt.scatter(X_train, y_train_data[0][0], color="black", s=40, label="Training points")

        # Derivative points
        X_deriv = X_train[derivative_indices]
        y_deriv = y_train_data[0][0][derivative_indices]
        plt.scatter(X_deriv, y_deriv, color="red", s=80, marker="D", label=f"Derivative points (order <= {n_order})")

        # Annotate derivative points
        for i, idx in enumerate(derivative_indices):
            plt.text(
                X_train[idx] + 0.02, y_train_data[0][0][idx],
                f"d@{idx}", color="red", fontsize=9,
                verticalalignment="bottom", horizontalalignment="left"
            )

        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Sparse Derivative-Enhanced GP with Labeled Derivative Points")
        plt.legend(loc="best", frameon=True)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

Full Workflow
-------------

Finally, we combine all steps to train, optimize, and visualize the sparse
derivative-enhanced Gaussian Process model.

.. jupyter-execute::

    X_train, y_train_data, der_indices = generate_training_data(
        lb_x, ub_x, num_training_pts, n_order, n_bases,
        derivative_indices, example_function
    )

    model = build_wdegp_model(
        X_train, y_train_data, n_order, n_bases,
        derivative_indices, der_indices, normalize_data, kernel, kernel_type
    )

    params = optimize_model(model, n_restarts, swarm_size)
    results = predict_and_evaluate(model, example_function, lb_x, ub_x, test_points, params)

    # Use the new visualization
    plot_sparse_degps(
        X_train,
        y_train_data,
        results['X_test'],
        results['y_pred'],
        results['y_true'],
        results['y_cov'],
        derivative_indices,
        example_function,
        n_order
    )
