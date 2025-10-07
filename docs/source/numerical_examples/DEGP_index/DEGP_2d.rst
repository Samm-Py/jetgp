2D Arbitrary Order Derivative-Enhanced Gaussian Process Numerical Example
==========================================================================

This example demonstrates **2D Derivative-Enhanced Gaussian Process (DEGP)** regression
for multi-dimensional function approximation. We'll include partial derivatives
up to a specified order to improve predictions using very few training points.

Key concepts covered:
- 2D hypercomplex automatic differentiation
- Full inclusion of first- and second-order partial derivatives
- Multi-dimensional GP regression with derivative constraints
- 2D visualization and spatial error analysis

.. contents::
   :local:
   :depth: 2

Setup
-----

Import necessary packages and set plotting parameters.

.. jupyter-execute::

    import numpy as np
    import itertools
    import time
    import pyoti.sparse as oti
    from full_degp.degp import degp
    import utils
    import plotting_helper
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 12})

Target Function
---------------

We model the following 2D function:

.. math::

    f(x_1, x_2) = x_1^2 x_2 + \cos(10 x_1) + \cos(10 x_2)

over the domain :math:`x_1, x_2 \in [-1, 1]`. The following plots show the function behavior across the domain.

.. jupyter-execute::

    # Create a fine evaluation grid for plotting
    lb_x, ub_x = -1.0, 1.0
    lb_y, ub_y = -1.0, 1.0
    plot_resolution = 100
    x1_plot = np.linspace(lb_x, ub_x, plot_resolution)
    x2_plot = np.linspace(lb_y, ub_y, plot_resolution)
    X1_plot, X2_plot = np.meshgrid(x1_plot, x2_plot)
    X_plot = np.column_stack([X1_plot.ravel(), X2_plot.ravel()])

    # Compute true function values
    def true_function(X, alg=np):
        x1, x2 = X[:, 0], X[:, 1]
        return x1**2 * x2 + alg.cos(10*x1) + alg.cos(10*x2)

    y_plot = true_function(X_plot).reshape(plot_resolution, plot_resolution)

    # 3D surface plot
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(14,6))

    ax1 = fig.add_subplot(1,2,1, projection='3d')
    surf = ax1.plot_surface(X1_plot, X2_plot, y_plot, cmap='viridis', edgecolor='none', alpha=0.9)
    ax1.set_xlabel('$x_1$')
    ax1.set_ylabel('$x_2$')
    ax1.set_zlabel('$f(x_1, x_2)$')
    ax1.set_title('3D Surface of Target Function')
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10)

    # 2D contour plot
    ax2 = fig.add_subplot(1,2,2)
    contour = ax2.contourf(X1_plot, X2_plot, y_plot, levels=50, cmap='viridis')
    ax2.set_xlabel('$x_1$')
    ax2.set_ylabel('$x_2$')
    ax2.set_title('Contour of Target Function')
    fig.colorbar(contour, ax=ax2)
    plt.tight_layout()
    plt.show()

Define Parameters
-----------------

.. jupyter-execute::

    # Domain
    lb_x, ub_x = -1.0, 1.0
    lb_y, ub_y = -1.0, 1.0

    # Training grid
    num_pts_per_axis = 5
    sampling_strategy = 'uniform'  # 'uniform', 'random', or 'chebyshev'

    # DEGP hyperparameters
    n_order = 2
    n_bases = 2
    kernel = "SE"
    kernel_type = "anisotropic"
    normalize_data = True
    n_restarts = 15
    swarm_size = 100

    # Test grid resolution
    test_grid_resolution = 25

Parameter Explanation
---------------------

The following parameters control the DEGP training and evaluation:

- **`lb_x`, `ub_x`, `lb_y`, `ub_y`**: Bounds of the 2D input domain.
- **`num_pts_per_axis`**: Number of training points per axis (total points = num_pts_per_axis²).
- **`sampling_strategy`**: How training points are distributed ('uniform', 'random', 'chebyshev').
- **`n_order`**: Maximum derivative order included in the DEGP.
- **`n_bases`**: Number of hypercomplex bases, equal to input dimension.
- **`kernel` and `kernel_type`**: Covariance kernel and isotropic/anisotropic choice.
- **`normalize_data`**: If True, observations are normalized before fitting the GP.
- **`n_restarts` and `swarm_size`**: Settings for hyperparameter optimization via particle swarm.
- **`test_grid_resolution`**: Number of points per axis in the evaluation grid.

Increasing the derivative order or the number of training points increases observations per point, improving the model’s ability to capture local variations but also increasing computational cost.

Generate 2D Training Grid
-------------------------

.. jupyter-execute::

    def create_training_grid(lb_x, ub_x, lb_y, ub_y, num_pts_per_axis, strategy='uniform'):
        if strategy == 'uniform':
            x_vals = np.linspace(lb_x, ub_x, num_pts_per_axis)
            y_vals = np.linspace(lb_y, ub_y, num_pts_per_axis)
            return np.array(list(itertools.product(x_vals, y_vals)))
        elif strategy == 'random':
            np.random.seed(42)
            return np.random.uniform([lb_x, lb_y], [ub_x, ub_y], (num_pts_per_axis**2, 2))
        else:  # Chebyshev
            k = np.arange(1, num_pts_per_axis+1)
            x_cheb = 0.5 * (lb_x + ub_x) + 0.5 * (ub_x - lb_x) * np.cos((2*k - 1)*np.pi/(2*num_pts_per_axis))
            y_cheb = 0.5 * (lb_y + ub_y) + 0.5 * (ub_y - lb_y) * np.cos((2*k - 1)*np.pi/(2*num_pts_per_axis))
            return np.array(list(itertools.product(x_cheb, y_cheb)))

    X_train = create_training_grid(lb_x, ub_x, lb_y, ub_y, num_pts_per_axis, sampling_strategy)
    print(f"Training points shape: {X_train.shape}")

Analyze Derivative Structure
----------------------------

.. jupyter-execute::

    def analyze_derivative_structure(n_bases, n_order):
        der_indices = utils.gen_OTI_indices(n_bases, n_order)
        total_derivatives = sum(len(group) for group in der_indices)
        print(f"Including all derivatives up to order {n_order}")
        print(f"Total derivative types per point: {total_derivatives} (including mixed partials)")
        return der_indices

    der_indices = analyze_derivative_structure(n_bases, n_order)

Generate 2D Training Data with Derivatives
------------------------------------------

.. jupyter-execute::

    def generate_training_data(X_train, n_order, n_bases, der_indices, true_function):
        X_train_pert = oti.array(X_train)
        for i in range(n_bases):
            X_train_pert[:, i] += oti.e(i+1, order=n_order)
        y_train_hc = true_function(X_train_pert)
        y_train_list = [y_train_hc.real]
        for group in der_indices:
            for sub_group in group:
                y_train_list.append(y_train_hc.get_deriv(sub_group))
        total_obs = sum(d.shape[0] for d in y_train_list)
        print(f"Total observations created: {total_obs}")
        return y_train_list

    def true_function(X, alg=oti):
        x1, x2 = X[:, 0], X[:, 1]
        return x1**2 * x2 + alg.cos(10*x1) + alg.cos(10*x2)

    y_train_list = generate_training_data(X_train, n_order, n_bases, der_indices, true_function)

Train and Evaluate 2D DEGP Models
---------------------------------

For the 2D DEGP, we train a model using both function values and partial derivatives. The workflow is as follows:

1. **Derivative Index Generation**:  
   Multi-index sets `der_indices` enumerate all partial derivatives up to `n_order`, including mixed partials

2. **2D Hypercomplex Automatic Differentiation**:  
   `oti.array` represents the 2D training inputs in hypercomplex form. Adding `oti.e(i+1, order=n_order)` to each input dimension tags points for automatic differentiation, and evaluating `true_function` returns both function values and derivatives via `get_deriv`.

3. **Training Data Construction**:  
   - `y_train_hc.real` contains function values.  
   - `y_train_hc.get_deriv(sub_group)` extracts each derivative.  
   - Observations are collected in `y_train_list` and passed to the DEGP constructor.

4. **DEGP Model Initialization**:  
   - `X_train`: training points.  
   - `y_train_list`: function and derivative observations.  
   - `n_order`, `n_bases`, `der_indices`: derivative specifications.  
   - Kernel and normalization options (`kernel`, `kernel_type`, `normalize`).

5. **Hyperparameter Optimization**:  
   Performed using multiple restarts and a particle swarm (`n_restarts`, `swarm_size`).

6. **Prediction and Evaluation**:  
   Posterior mean predictions are computed on a test grid and compared to the true function for error analysis.

.. jupyter-execute::

    def train_degps(X_train, y_train_list, n_order, n_bases, der_indices):
        gp_model = degp(
            X_train, y_train_list, n_order, n_bases, der_indices,
            normalize=normalize_data, kernel=kernel, kernel_type=kernel_type
        )
        params = gp_model.optimize_hyperparameters(n_restart_optimizer=n_restarts, swarm_size=swarm_size, verbose=False)
        return gp_model, params

    gp_model, params = train_degps(X_train, y_train_list, n_order, n_bases, der_indices)

Evaluate Model
--------------

.. jupyter-execute::

    def evaluate_2d_model(gp_model, params, true_function, test_grid_resolution, lb_x, ub_x, lb_y, ub_y):
        x_lin = np.linspace(lb_x, ub_x, test_grid_resolution)
        y_lin = np.linspace(lb_y, ub_y, test_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
        X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
        y_pred = gp_model.predict(X_test, params, calc_cov=False)
        y_true = true_function(X_test, alg=np)

        y_true_flat, y_pred_flat = y_true.flatten(), y_pred.flatten()
        errors = np.abs(y_true_flat - y_pred_flat).reshape((test_grid_resolution, test_grid_resolution))
        nrmse = utils.nrmse(y_true_flat, y_pred_flat)
        rmse = np.sqrt(np.mean((y_true_flat - y_pred_flat)**2))
        max_error = np.max(np.abs(y_true_flat - y_pred_flat))

        return {
            'X_test': X_test, 'X1_grid': X1_grid, 'X2_grid': X2_grid,
            'y_pred': y_pred, 'y_true': y_true, 'nrmse': nrmse,
            'rmse': rmse, 'max_error': max_error, 'errors': errors
        }

    results = evaluate_2d_model(gp_model, params, true_function, test_grid_resolution, lb_x, ub_x, lb_y, ub_y)
    print(f"NRMSE: {results['nrmse']:.6f}, RMSE: {results['rmse']:.6f}, Max Error: {results['max_error']:.6f}")

Visualize 2D Results
--------------------

.. jupyter-execute::

    plotting_helper.make_plots(
        X_train, y_train_list, results['X_test'], results['y_pred'],
        true_function, X1_grid=results['X1_grid'], X2_grid=results['X2_grid'],
        n_order=n_order, n_bases=n_bases, der_indices=der_indices
    )
