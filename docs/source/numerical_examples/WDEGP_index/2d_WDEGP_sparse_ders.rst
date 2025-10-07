2D Grouped Submodel Derivative-Enhanced Gaussian Process
=========================================================

This tutorial demonstrates a **heterogeneous derivative-enhanced GP (DEGP)** 
for a 2D function where training points are grouped into submodels. Each submodel 
can have a different derivative order. Function values from all training points 
are used for the global fit, but derivatives are selectively computed only for 
points in `submodel_point_groups`.

Key Features
------------

- Approximation of a 2D function.
- Submodel creation from arbitrary groups of training points.
- **Selective Derivative Utilization**: Only points included in submodel groups contribute derivative information.
- **Heterogeneous Derivative Orders**: Different submodels can use different derivative orders.
- **Automatic Data Reordering**: Training data is reordered for compatibility with the GP indexing scheme.

Configuration
-------------

.. jupyter-execute::

    import numpy as np

    # Random seed
    random_seed = 0
    np.random.seed(random_seed)

    # GP configuration
    n_order = 3  # Maximum derivative order for the most complex submodel
    n_bases = 2
    lb_x, ub_x = -1.0, 1.0
    lb_y, ub_y = -1.0, 1.0
    points_per_axis = 4
    kernel = "RQ"
    kernel_type = "isotropic"
    normalize = True
    n_restart_optimizer = 15
    swarm_size = 100
    test_points_per_axis = 35

    # Submodel point groups
    submodel_point_groups = [
        [0, 3, 12, 15],                 # Corners: no derivatives
        [1, 2, 4, 8, 7, 11, 13, 14],    # Edges: 1st order derivatives
        [5, 6, 9, 10]                   # Center: full derivatives
    ]

Example Function
----------------

.. jupyter-execute::

    def six_hump_camel_function(X, alg=np):
        x1 = X[:, 0]
        x2 = X[:, 1]
        return ((4 - 2.1 * x1**2 + (x1**4)/3.0) * x1**2 +
                x1*x2 + (-4 + 4*x2**2) * x2**2)

Data Generation
---------------

.. jupyter-execute::

    import itertools

    def generate_training_points():
        x_vals = np.linspace(lb_x, ub_x, points_per_axis)
        y_vals = np.linspace(lb_y, ub_y, points_per_axis)
        return np.array(list(itertools.product(x_vals, y_vals)))

    X_train_initial = generate_training_points()
    print("Training points shape:", X_train_initial.shape)

Data Reordering
---------------

.. jupyter-execute::

    def reorder_training_data(X_train_initial, submodel_point_groups):
        import itertools
        arbitrary_flat = list(itertools.chain.from_iterable(submodel_point_groups))
        all_indices = set(range(X_train_initial.shape[0]))
        used_indices = set(arbitrary_flat)
        unused_indices = sorted(list(all_indices - used_indices))
        reorder_map = arbitrary_flat + unused_indices
        X_train_reordered = X_train_initial[reorder_map]

        sequential_indices = []
        current_pos = 0
        for group in submodel_point_groups:
            group_len = len(group)
            sequential_indices.append(list(range(current_pos, current_pos + group_len)))
            current_pos += group_len

        return X_train_reordered, sequential_indices

    X_train_reordered, sequential_indices = reorder_training_data(X_train_initial, submodel_point_groups)
    print("Reordered training points shape:", X_train_reordered.shape)

Submodel Data Preparation (Heterogeneous Derivatives)
-----------------------------------------------------

.. jupyter-execute::

    import pyoti.sparse as oti
    import utils

    def prepare_submodel_data(X_train, submodel_indices):
        # Define derivative structures for each submodel
        base_der_indices = utils.gen_OTI_indices(n_bases, n_order)
        derivative_specs = [
            [],  # Submodel 1: no derivatives
            utils.gen_OTI_indices(n_bases, 1),  # Submodel 2: 1st order
            base_der_indices  # Submodel 3: full derivatives
        ]

        y_function_values = six_hump_camel_function(X_train, alg=np)
        submodel_data = []

        for k, point_indices in enumerate(submodel_indices):
            X_sub_oti = oti.array(X_train[point_indices])
            for i in range(n_bases):
                for j in range(X_sub_oti.shape[0]):
                    X_sub_oti[j, i] += oti.e(i + 1, order=n_order)

            y_with_derivatives = oti.array([six_hump_camel_function(x, alg=oti) for x in X_sub_oti])

            current_submodel_data = [y_function_values]
            current_derivative_spec = derivative_specs[k]
            for i in range(len(current_derivative_spec)):
                for j in range(len(current_derivative_spec[i])):
                    deriv = y_with_derivatives.get_deriv(current_derivative_spec[i][j]).reshape(-1, 1)
                    current_submodel_data.append(deriv)
            submodel_data.append(current_submodel_data)

        return submodel_data, derivative_specs

    submodel_data, derivative_specs = prepare_submodel_data(X_train_reordered, sequential_indices)
    print("Number of submodels:", len(submodel_data))

Build and Optimize GP
--------------------

.. jupyter-execute::

    from wdegp.wdegp import wdegp

    def build_and_optimize_gp(X_train, submodel_data, submodel_indices, derivative_specs):
        gp_model = wdegp(
            X_train, submodel_data, n_order, n_bases,
            submodel_indices, derivative_specs, normalize=normalize,
            kernel=kernel, kernel_type=kernel_type
        )
        params = gp_model.optimize_hyperparameters(
            n_restart_optimizer=n_restart_optimizer,
            swarm_size=swarm_size, verbose = False
        )
        return gp_model, params

    gp_model, params = build_and_optimize_gp(X_train_reordered, submodel_data, sequential_indices, derivative_specs)
    print("GP model built and optimized.")

Evaluation and Visualization
----------------------------

.. jupyter-execute::

    import matplotlib.pyplot as plt
    import numpy as np

    x_lin = np.linspace(lb_x, ub_x, test_points_per_axis)
    y_lin = np.linspace(lb_y, ub_y, test_points_per_axis)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    y_pred, submodel_vals = gp_model.predict(X_test, params, calc_cov=False, return_submodels=True)
    y_true = six_hump_camel_function(X_test, alg=np)
    nrmse = utils.nrmse(y_true, y_pred)
    abs_error = np.abs(y_true - y_pred)

    print("NRMSE:", nrmse)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True, sharey=True)

    c1 = axes[0].contourf(X1_grid, X2_grid, y_true.reshape(test_points_per_axis, test_points_per_axis), levels=50, cmap="viridis")
    fig.colorbar(c1, ax=axes[0])
    axes[0].set_title("True Function")
    axes[0].scatter(X_train_initial[:,0], X_train_initial[:,1], c="red", edgecolor="k", s=50, label="Training Points")

    c2 = axes[1].contourf(X1_grid, X2_grid, y_pred.reshape(test_points_per_axis, test_points_per_axis), levels=50, cmap="viridis")
    fig.colorbar(c2, ax=axes[1])
    axes[1].set_title("GP Prediction")
    axes[1].scatter(X_train_initial[:,0], X_train_initial[:,1], c="red", edgecolor="k", s=50)

    c3 = axes[2].contourf(X1_grid, X2_grid, abs_error.reshape(test_points_per_axis, test_points_per_axis), levels=50, cmap="magma")
    fig.colorbar(c3, ax=axes[2])
    axes[2].set_title("Absolute Error")
    axes[2].scatter(X_train_initial[:,0], X_train_initial[:,1], c="red", edgecolor="k", s=50)

    for ax in axes:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
    axes[0].legend()
    plt.tight_layout()
    plt.show()
