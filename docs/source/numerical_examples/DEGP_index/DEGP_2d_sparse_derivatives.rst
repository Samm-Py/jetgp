2D Arbitrary Order Derivative-Enhanced Gaussian Process With Derivative Selection Numerical Example
====================================================================================================

This tutorial demonstrates **2D DEGP regression** with **selective derivative inclusion**. 
Instead of including all derivatives up to a given order, we strategically choose only 
the most informative derivatives. This balances **predictive accuracy** and **computational cost**.

Key concepts covered:
- Selective derivative strategies: 'gradient_only', 'main_derivatives', 'complete'
- Computational trade-offs in multi-dimensional derivative spaces
- Comparative performance evaluation of different strategies
- 2D visualization of DEGP models with selective derivatives

.. contents::
   :local:
   :depth: 2

Setup
-----

We begin by importing the necessary packages and setting plotting parameters. 
These libraries provide tools for numerical computation (`numpy`), combinatorics (`itertools`), 
hypercomplex automatic differentiation (`pyoti`), DEGP modeling (`full_degp`), 
and visualization (`matplotlib`).

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

Define Parameters
-----------------

Here we define the **domain, training grid, and DEGP hyperparameters**. These parameters 
control the size of the training set, the derivative order included, kernel choice, and 
optimization settings. Defining them upfront ensures that the experiment is fully reproducible.

.. jupyter-execute::

    # Domain
    lb_x, ub_x = -1.0, 1.0
    lb_y, ub_y = -1.0, 1.0

    # Training grid
    num_pts_per_axis = 5

    # DEGP hyperparameters
    n_order = 2
    n_bases = 2
    kernel = "SE"
    kernel_type = "anisotropic"
    normalize_data = True
    n_restarts = 15
    swarm_size = 150

    # Test grid resolution
    test_grid_resolution = 25

Parameter Explanation
---------------------

- **Domain Bounds (`lb_x`, `ub_x`, `lb_y`, `ub_y`)**: Specify the input space limits.
- **`num_pts_per_axis`**: Controls the sparsity of training points. Total points = num_pts_per_axis².
- **`n_order`**: Maximum derivative order included in DEGP.
- **`n_bases`**: Number of hypercomplex bases (equal to input dimension).
- **`kernel` / `kernel_type`**: Covariance kernel specification for DEGP.
- **`normalize_data`**: Normalizes observations before fitting, improving numerical stability.
- **`n_restarts` / `swarm_size`**: Hyperparameter optimization settings (particle swarm).
- **`test_grid_resolution`**: Number of points per axis for evaluation and visualization.

Defining Derivative Selection Strategies
----------------------------------------

Selective derivative inclusion is key to balancing **accuracy** and **computational efficiency**. 
We define three strategies:

1. **Gradient Only (`gradient_only`)**: Only first-order derivatives are included. Minimal cost, 
   captures local slope information.

2. **Main Derivatives (`main_derivatives`)**: First-order derivatives plus main second-order 
   derivatives (no mixed second-order terms). Moderate cost, captures curvature along primary axes.

3. **Complete (`complete`)**: All derivatives up to `n_order`. High cost, includes full interaction 
   between dimensions.

Mathematically, for a 2D function :math:`f(x_1, x_2)`:

.. math::

    \text{gradient\_only} = \Big\{
        \frac{\partial f}{\partial x_1},\;
        \frac{\partial f}{\partial x_2}
    \Big\}

.. math::

    \text{main\_derivatives} = \Big\{
        \frac{\partial f}{\partial x_1},\;
        \frac{\partial f}{\partial x_2},\;
        \frac{\partial^2 f}{\partial x_1^2},\;
        \frac{\partial^2 f}{\partial x_2^2}
    \Big\}

.. math::

    \text{complete} = \Big\{
        f,\;
        \frac{\partial f}{\partial x_1},\;
        \frac{\partial f}{\partial x_2},\;
        \frac{\partial^2 f}{\partial x_1^2},\;
        \frac{\partial^2 f}{\partial x_1 \partial x_2},\;
        \frac{\partial^2 f}{\partial x_2^2}
    \Big\}

These correspond to **multi-indices** used in `utils.gen_OTI_indices` and 
the hypercomplex tagging scheme in `oti.array`.

.. jupyter-execute::

    def define_strategies(n_order, n_bases):
        strategies = {}
        strategies['gradient_only'] = [[[[1,1]], [[2,1]]]]
        strategies['main_derivatives'] = [[[[1,1]], [[2,1]]], [[[1,2]], [[2,2]]]]
        strategies['complete'] = utils.gen_OTI_indices(n_bases, n_order)

        for name, der_indices in strategies.items():
            count = sum(len(group) for group in der_indices)
            print(f"Strategy: {name.upper()}, Derivatives per point: {count}, " +
                  f"Trade-off: {'Low' if count <= 2 else 'Medium' if count <= 4 else 'High'}")
        return strategies

    strategies = define_strategies(n_order, n_bases)

Generate 2D Training Grid
-------------------------

We generate a uniform grid of training points in 2D. Each point will be augmented with 
hypercomplex tags for derivative evaluation. Using a structured grid allows for easier 
interpretation of DEGP performance.

.. jupyter-execute::

    def create_training_grid(lb_x, ub_x, lb_y, ub_y, num_pts_per_axis):
        x_vals = np.linspace(lb_x, ub_x, num_pts_per_axis)
        y_vals = np.linspace(lb_y, ub_y, num_pts_per_axis)
        return np.array(list(itertools.product(x_vals, y_vals)))

    X_train = create_training_grid(lb_x, ub_x, lb_y, ub_y, num_pts_per_axis)
    print(f"Training points shape: {X_train.shape}")

Define the True Function
------------------------

We use a **nonlinear test function** with oscillatory and polynomial components. This challenges 
the DEGP to capture both smooth curvature and high-frequency behavior.

.. jupyter-execute::

    def true_function(X, alg=oti):
        x1, x2 = X[:, 0], X[:, 1]
        return x1**2 * x2 + alg.cos(10*x1) + alg.cos(10*x2)

Generate Training Data with Selective Derivatives
-------------------------------------------------

We augment the 2D training points with hypercomplex tags and compute only the selected 
derivatives according to the chosen strategy. This yields the **training dataset for DEGP**, 
including function values and derivative observations.

.. jupyter-execute::

    def generate_training_data(X_train, strategy_name, strategies, true_function):
        der_indices = strategies[strategy_name]
        X_train_pert = oti.array(X_train)
        for i in range(n_bases):
            X_train_pert[:, i] += oti.e(i+1, order=n_order)
        y_train_hc = true_function(X_train_pert)
        y_train_list = [y_train_hc.real]
        for group in der_indices:
            for sub_group in group:
                y_train_list.append(y_train_hc.get_deriv(sub_group))
        total_obs = sum(d.shape[0] for d in y_train_list)
        print(f"Total observations created for '{strategy_name}': {total_obs}")
        return y_train_list

    # Example: using 'main_derivatives'
    strategy_name = 'main_derivatives'
    y_train_list = generate_training_data(X_train, strategy_name, strategies, true_function)

Train and Evaluate DEGP Model
-----------------------------

We train the DEGP model using the constructed training data. **Hyperparameter optimization** ensures 
good predictive performance. Evaluations are performed on a fine 2D test grid for visualization.

.. jupyter-execute::

    def train_degps(X_train, y_train_list, der_indices):
        gp_model = degp(
            X_train, y_train_list, n_order, n_bases, der_indices,
            normalize=normalize_data, kernel=kernel, kernel_type=kernel_type
        )
        params = gp_model.optimize_hyperparameters(
            n_restart_optimizer=n_restarts,
            swarm_size=swarm_size, verbose=False
        )
        return gp_model, params

    gp_model, params = train_degps(X_train, y_train_list, strategies[strategy_name])

Evaluate Model
--------------

We compute **posterior predictions** on a 2D test grid, compare against the true function, 
and calculate error metrics including **NRMSE**, **RMSE**, and **maximum absolute error**.

.. jupyter-execute::

    def evaluate_2d_model(gp_model, true_function, test_grid_resolution):
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
        max_error = np.max(errors)

        return {
            'X_test': X_test, 'X1_grid': X1_grid, 'X2_grid': X2_grid,
            'y_pred': y_pred, 'y_true': y_true, 'nrmse': nrmse,
            'rmse': rmse, 'max_error': max_error, 'errors': errors
        }

    results = evaluate_2d_model(gp_model, true_function, test_grid_resolution)
    print(f"NRMSE: {results['nrmse']:.6f}, RMSE: {results['rmse']:.6f}, Max Error: {results['max_error']:.6f}")

Visualize Results
-----------------

Contour plots of **predictions, true function, and errors** provide an intuitive understanding 
of the DEGP performance. Training points are overlaid to show coverage of the input space.

.. jupyter-execute::

    plotting_helper.make_plots(
        X_train, y_train_list, results['X_test'], results['y_pred'],
        true_function, X1_grid=results['X1_grid'], X2_grid=results['X2_grid'],
        n_order=n_order, n_bases=n_bases, der_indices=strategies[strategy_name]
    )

Comparative Evaluation of Derivative Strategies
-----------------------------------------------

We now **compare the three strategies** to illustrate the trade-off between predictive accuracy 
and computational cost:

- **Gradient Only**: Minimal derivatives, fastest to compute.
- **Main Derivatives**: Captures key curvature information, moderate cost.
- **Complete**: Full derivative information, highest accuracy but most expensive.

.. jupyter-execute::

    strategy_names = ['gradient_only', 'main_derivatives', 'complete']
    results_summary = {}

    for strategy_name in strategy_names:
        print(f"Processing strategy: {strategy_name}")
        y_train_list = generate_training_data(X_train, strategy_name, strategies, true_function)
        gp_model, params = train_degps(X_train, y_train_list, strategies[strategy_name])
        results = evaluate_2d_model(gp_model, true_function, test_grid_resolution)
        results_summary[strategy_name] = {
            'nrmse': results['nrmse'],
            'rmse': results['rmse'],
            'max_error': results['max_error'],
            'n_observations': sum(d.shape[0] for d in y_train_list),
            'X_test': results['X_test'],
            'y_pred': results['y_pred'],
            'errors': results['errors']
        }

    # Display summary table
    print(f"{'Strategy':<20}{'NRMSE':<12}{'RMSE':<12}{'Max Error':<12}{'Observations'}")
    print("-"*70)
    for name, metrics in results_summary.items():
        print(f"{name:<20}{metrics['nrmse']:<12.6f}{metrics['rmse']:<12.6f}{metrics['max_error']:<12.6f}{metrics['n_observations']}")

Visual Comparison of Strategies
--------------------------------

Finally, we generate **error heatmaps** for all strategies to visualize where predictions 
perform well or poorly. This highlights regions where derivative selection most affects accuracy.

.. jupyter-execute::

    import math
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    all_errors = [results_summary[name]['errors'].reshape((test_grid_resolution, test_grid_resolution))
                for name in strategy_names]
    min_error = 10**-6  # avoid zero for log scale
    max_error = max(err.max() for err in all_errors)

    n_strategies = len(strategy_names)
    n_cols = 2
    n_rows = math.ceil(n_strategies / n_cols)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 4*n_rows), sharex=True, sharey=True)
    axs = axs.flatten()

    for i, strategy_name in enumerate(strategy_names):
        ax = axs[i]
        errors = results_summary[strategy_name]['errors'].reshape((test_grid_resolution, test_grid_resolution))
        im = ax.imshow(errors, extent=[lb_x, ub_x, lb_y, ub_y], origin='lower',
                    cmap='viridis', norm=LogNorm(vmin=min_error, vmax=max_error))
        ax.set_title(f"{strategy_name} - Max Error: {results_summary[strategy_name]['max_error']:.4f}", fontsize=11)
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.grid(False)

    for j in range(i+1, n_rows*n_cols):
        fig.delaxes(axs[j])

    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='vertical')
    cbar.set_label('Absolute Error (log scale)')

    fig.suptitle('Prediction Error Heatmaps for Different Derivative Strategies', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])
    plt.show()
