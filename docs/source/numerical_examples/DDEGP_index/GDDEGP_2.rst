================================================================================
DEGP Tutorial: Directional Derivative-Enhanced Gaussian Processes (GD-DEGP)
================================================================================

This tutorial demonstrates an advanced application of DEGP that utilizes **directional
derivatives** instead of standard partial derivatives. In this version, each training
point has multiple rays (gradient and perpendicular directions), all processed via a
global perturbation methodology.

Key concepts covered:
- Using multiple directional derivatives ("rays") per point.
- Generating training points with Latin Hypercube Sampling.
- Performing global pointwise directional automatic differentiation with pyoti.
- Training and visualizing a 2D GD-DEGP model.

Setup
-----

.. jupyter-execute::

    import numpy as np
    import pyoti.sparse as oti
    import itertools
    from full_gddegp.gddegp import gddegp
    from scipy.stats import qmc
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.lines import Line2D
    import utils

    plt.rcParams.update({'font.size': 12})

Configuration
-------------

.. jupyter-execute::

    n_order = 1
    n_bases = 2
    num_directions_per_point = 2
    num_training_pts = 10
    domain_bounds = ((-5.0, 10.0), (0.0, 15.0))
    test_grid_resolution = 50
    normalize_data = True
    kernel = "SE"
    kernel_type = "anisotropic"
    n_restarts = 15
    swarm_size = 250
    random_seed = 1
    np.random.seed(random_seed)

True Function and Gradient
--------------------------

.. jupyter-execute::

    def true_function(X, alg=np):
        x, y = X[:, 0], X[:, 1]
        a, b, c, r, s, t = 1.0, 5.1/(4*np.pi**2), 5.0/np.pi, 6.0, 10.0, 1.0/(8*np.pi)
        return a*(y - b*x**2 + c*x - r)**2 + s*(1-t)*alg.cos(x) + s

    def true_gradient(x, y):
        a, b, c, r, s, t = 1.0, 5.1/(4*np.pi**2), 5.0/np.pi, 6.0, 10.0, 1.0/(8*np.pi)
        gx = 2*a*(y - b*x**2 + c*x - r)*(-2*b*x + c) - s*(1-t)*np.sin(x)
        gy = 2*a*(y - b*x**2 + c*x - r)
        return gx, gy

.. jupyter-execute::

    def clipped_arrow(ax, origin, direction, length, bounds, color="black"):
        x0, y0 = origin
        dx, dy = direction * length
        xlim, ylim = bounds
        tx = np.inf if dx==0 else (xlim[1]-x0)/dx if dx>0 else (xlim[0]-x0)/dx
        ty = np.inf if dy==0 else (ylim[1]-y0)/dy if dy>0 else (ylim[0]-y0)/dy
        t = min(1.0, tx, ty)
        ax.arrow(x0, y0, dx*t, dy*t, head_width=0.25, head_length=0.35, fc=color, ec=color)

Training Data Generation
------------------------

.. jupyter-execute::

    def generate_training_data():
        sampler = qmc.LatinHypercube(d=n_bases, seed=random_seed)
        unit_samples = sampler.random(n=num_training_pts)
        X_train = qmc.scale(unit_samples,
                            [b[0] for b in domain_bounds],
                            [b[1] for b in domain_bounds])

        # Create multiple rays per point: gradient + perpendicular
        rays_list = [[] for _ in range(num_directions_per_point)]
        for x, y in X_train:
            gx, gy = true_gradient(x, y)
            theta_grad = np.arctan2(gy, gx)
            theta_perp = theta_grad + np.pi/2

            ray_grad = np.array([np.cos(theta_grad), np.sin(theta_grad)]).reshape(-1,1)
            ray_perp = np.array([np.cos(theta_perp), np.sin(theta_perp)]).reshape(-1,1)

            rays_list[0].append(ray_grad)
            rays_list[1].append(ray_perp)

        # Apply global perturbations using hypercomplex tags
        X_pert = oti.array(X_train)
        for i in range(num_directions_per_point):
            e_tag = oti.e(i+1, order=n_order)
            for j in range(len(rays_list[i])):
                perturbation = oti.array(rays_list[i][j]) * e_tag
                X_pert[j, :] += perturbation.T

        # Evaluate function with hypercomplex AD and truncate
        f_hc = true_function(X_pert, alg=oti)
        for combo in itertools.combinations(range(1, num_directions_per_point+1), 2):
            f_hc = f_hc.truncate(combo)

        y_train_list = [f_hc.real.reshape(-1,1)]
        der_indices_to_extract = [[[1,1]], [[2,1]]]
        for idx in der_indices_to_extract:
            y_train_list.append(f_hc.get_deriv(idx).reshape(-1,1))

        return {'X_train': X_train,
                'y_train_list': y_train_list,
                'rays_list': rays_list,
                'der_indices': der_indices_to_extract}

Model Training
--------------

.. jupyter-execute::

    def train_model(training_data):
        rays_array = [np.hstack(training_data['rays_list'][i]) for i in range(num_directions_per_point)]
        gp_model = gddegp(training_data['X_train'],
                          training_data['y_train_list'],
                          n_order=[1,1],
                          rays_array=rays_array,
                          der_indices=training_data['der_indices'],
                          normalize=normalize_data,
                          kernel=kernel,
                          kernel_type=kernel_type)

        params = gp_model.optimize_hyperparameters(n_restart_optimizer=n_restarts, swarm_size=swarm_size, verbose = False)
        return gp_model, params

Evaluation
----------

.. jupyter-execute::

    def evaluate_model(gp_model, params, training_data):
        gx = np.linspace(domain_bounds[0][0], domain_bounds[0][1], test_grid_resolution)
        gy = np.linspace(domain_bounds[1][0], domain_bounds[1][1], test_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(gx, gy)
        X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
        N_test = X_test.shape[0]

        dummy_ray = np.array([[1.0],[0.0]])
        rays_pred = [np.hstack([dummy_ray for _ in range(N_test)]) for _ in range(num_directions_per_point)]
        y_pred_full = gp_model.predict(X_test, rays_pred, params, calc_cov=False, return_deriv=False)
        y_pred = y_pred_full[:N_test]

        y_true = true_function(X_test, alg=np)
        nrmse_val = utils.nrmse(y_true.flatten(), y_pred.flatten())
        return {'X_test': X_test, 'X1_grid': X1_grid, 'X2_grid': X2_grid,
                'y_pred': y_pred, 'y_true': y_true, 'nrmse': nrmse_val,
                'training_data': training_data}

Visualization
-------------

.. jupyter-execute::

    def visualize_results(results):
        training_data = results['training_data']
        X_train, rays_list = training_data['X_train'], training_data['rays_list']
        X1, X2 = results['X1_grid'], results['X2_grid']

        fig, axs = plt.subplots(1,3,figsize=(18,5))

        cf1 = axs[0].contourf(X1, X2, results['y_pred'].reshape(X1.shape), levels=30, cmap='viridis')
        axs[0].scatter(X_train[:,0], X_train[:,1], c='red', s=40, edgecolors='k', zorder=5)
        xlim, ylim = (domain_bounds[0], domain_bounds[1])
        for i in range(num_directions_per_point):
            for pt, ray in zip(X_train, rays_list[i]):
                clipped_arrow(axs[0], pt, ray.flatten(), length=.5, bounds=(xlim,ylim), color='black')
        axs[0].set_title("GD-DEGP Prediction")
        fig.colorbar(cf1, ax=axs[0])

        cf2 = axs[1].contourf(X1, X2, results['y_true'].reshape(X1.shape), levels=30, cmap='viridis')
        axs[1].set_title("True Function")
        fig.colorbar(cf2, ax=axs[1])

        abs_error = np.abs(results['y_pred'].flatten() - results['y_true'].flatten()).reshape(X1.shape)
        abs_error_clipped = np.clip(abs_error, 1e-6, None)
        log_levels = np.logspace(np.log10(abs_error_clipped.min()), np.log10(abs_error_clipped.max()), num=100)
        cf3 = axs[2].contourf(X1, X2, abs_error_clipped, levels=log_levels, norm=LogNorm(), cmap='magma_r')
        fig.colorbar(cf3, ax=axs[2])
        axs[2].set_title("Absolute Error (log scale)")

        for ax in axs:
            ax.set_xlabel("$x_1$")
            ax.set_ylabel("$x_2$")
            ax.set_aspect("equal")

        custom_lines = [Line2D([0],[0], marker='o', color='w', markerfacecolor='red', markeredgecolor='k', markersize=8, label='Train Points'),
                        Line2D([0],[0], color='black', lw=2, label='Gradient Ray Direction')]
        fig.legend(handles=custom_lines, loc='lower center', ncol=2, frameon=False, fontsize=12, bbox_to_anchor=(0.5,-0.02))
        plt.tight_layout(rect=[0,0.05,1,1])
        plt.show()

Run Tutorial
------------

.. jupyter-execute::

    training_data = generate_training_data()
    gp_model, params = train_model(training_data)
    results = evaluate_model(gp_model, params, training_data)
    visualize_results(results)
    print(f"Final NRMSE: {results['nrmse']:.6f}")
