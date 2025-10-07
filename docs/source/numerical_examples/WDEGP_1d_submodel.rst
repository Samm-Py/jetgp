Grouped Submodel Derivative-Enhanced Gaussian Process Implementation
=====================================================================

This tutorial demonstrates a **grouped submodel derivative-enhanced Gaussian
Process (DEGP)** using a procedural approach. Training points are divided into
submodels with different derivative orders, and predictions are obtained by
combining them with a weighted GP framework.

Key Features
------------

- Groups of training points form distinct submodels.
- Each submodel can have its own derivative structure.
- Combines regional information into a global prediction.
- 1D function approximation using grouped derivative information.

Module Imports
--------------

.. jupyter-execute::

    import numpy as np
    import matplotlib.pyplot as plt
    import pyoti.sparse as oti
    from wdegp.wdegp import wdegp
    import utils
    import plotting_helper

Experiment Configuration
------------------------

.. jupyter-execute::

    config = {
        "n_order": 3,
        "n_bases": 1,
        "lb_x": 0.5,
        "ub_x": 2.5,
        "num_points": 10,
        "kernel": "SE",
        "kernel_type": "anisotropic",
        "normalize": True,
        "n_restart_optimizer": 15,
        "swarm_size": 200,
        "test_points": 250,
        "random_seed": None,
        "submodel_groups": [[0,1,2,3,4], [5,6,7,8,9]],
        "submodel_orders": [3, 3],
    }

Generating Training Points
--------------------------

.. jupyter-execute::

    def generate_training_points(cfg):
        return np.linspace(cfg["lb_x"], cfg["ub_x"], cfg["num_points"]).reshape(-1, 1)

    X_train = generate_training_points(config)

Creating Submodel Structures
----------------------------

.. jupyter-execute::

    def create_submodel_structure(cfg):
        submodel_indices = cfg["submodel_groups"]
        derivative_specs = [
            utils.gen_OTI_indices(cfg["n_bases"], order)
            for order in cfg["submodel_orders"]
        ]
        return submodel_indices, derivative_specs

    submodel_indices, derivative_specs = create_submodel_structure(config)

Preparing Grouped Submodel Data
-------------------------------

.. jupyter-execute::

    def prepare_grouped_submodel_data(X_train, submodel_indices, derivative_specs, true_function):
        y_function_values = true_function(X_train, alg=np)
        submodel_data = []
        for k, indices in enumerate(submodel_indices):
            X_sub_oti = oti.array(X_train[indices])
            for i in range(X_sub_oti.shape[1]):
                for j in range(X_sub_oti.shape[0]):
                    X_sub_oti[j,i] += oti.e(i+1, order=config["n_order"])
            y_with_derivatives = oti.array([true_function(x, alg=oti) for x in X_sub_oti])
            submodel_training_data = [y_function_values]
            current_der_spec = derivative_specs[k]
            for i in range(len(current_der_spec)):
                for j in range(len(current_der_spec[i])):
                    derivative_value = y_with_derivatives.get_deriv(
                        current_der_spec[i][j]
                    ).reshape(-1,1)
                    submodel_training_data.append(derivative_value)
            submodel_data.append(submodel_training_data)
        return submodel_data

Evaluating Example Function
---------------------------

.. jupyter-execute::

    def oscillatory_function_with_trend(X, alg=oti):
        x1 = X[:,0]
        return alg.sin(10*np.pi*x1)/(2*x1) + (x1-1)**4

    submodel_data = prepare_grouped_submodel_data(
        X_train, submodel_indices, derivative_specs, oscillatory_function_with_trend
    )

Building and Optimizing the GP
------------------------------

.. jupyter-execute::

    def build_and_optimize_gp(X_train, submodel_data, submodel_indices, derivative_specs, cfg):
        gp_model = wdegp(
            X_train,
            submodel_data,
            cfg["n_order"],
            cfg["n_bases"],
            submodel_indices,
            derivative_specs,
            normalize=cfg["normalize"],
            kernel=cfg["kernel"],
            kernel_type=cfg["kernel_type"]
        )
        params = gp_model.optimize_hyperparameters(
            n_restart_optimizer=cfg["n_restart_optimizer"],
            swarm_size=cfg["swarm_size"], verbose = False
        )
        return gp_model, params

    gp_model, params = build_and_optimize_gp(
        X_train, submodel_data, submodel_indices, derivative_specs, config
    )

Evaluating the Model
--------------------

.. jupyter-execute::

    def evaluate_model(gp_model, cfg, true_function):
        X_test = np.linspace(cfg["lb_x"], cfg["ub_x"], cfg["test_points"]).reshape(-1,1)
        y_pred, y_cov, submodel_vals, submodel_cov = gp_model.predict(
            X_test, params, calc_cov=True, return_submodels=True
        )
        y_true = true_function(X_test, alg=np)
        nrmse = utils.nrmse(y_true, y_pred)
        return {
            "X_test": X_test,
            "y_pred": y_pred,
            "y_true": y_true,
            "y_cov": y_cov,
            "submodel_vals": submodel_vals,
            "submodel_cov": submodel_cov,
            "nrmse": nrmse
        }

    results = evaluate_model(gp_model, config, oscillatory_function_with_trend)
    print(f"Final NRMSE: {results['nrmse']:.6f}")

Visualizing Results
-------------------

.. jupyter-execute::

    plotting_helper.make_submodel_plots(
        X_train, submodel_data, results['X_test'], results['y_pred'],
        oscillatory_function_with_trend, cov=results['y_cov'],
        n_order=config["n_order"], n_bases=config["n_bases"],
        plot_submodels=True,
        submodel_vals=results['submodel_vals'],
        submodel_cov=results['submodel_cov']
    )

Analyzing Submodel Contributions
--------------------------------

.. jupyter-execute::

    def analyze_submodel_contributions(results, X_train, submodel_groups, true_function):
        X_test = results["X_test"]
        submodel_vals = results["submodel_vals"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        colors = plt.cm.viridis(np.linspace(0, 0.85, len(submodel_vals)))
        for i, submodel_pred in enumerate(submodel_vals):
            label = f"Submodel {i+1} (Points {submodel_groups[i]})"
            ax1.plot(X_test.ravel(), submodel_pred.ravel(), color=colors[i], alpha=0.8, linewidth=1.5, label=label)

        ax1.scatter(X_train.ravel(), true_function(X_train, alg=np),
                    color='red', s=50, edgecolor='black', label='Training Points', zorder=10)
        ax1.set_title("Individual Submodel Predictions")
        ax1.set_ylabel("f(x)")
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)

        ax2.plot(X_test.ravel(), results["y_true"].ravel(), 'b-', linewidth=2, label='True Function')
        ax2.plot(X_test.ravel(), results["y_pred"].ravel(), 'r--', linewidth=2, label='Combined GP Prediction')

        std_dev = np.sqrt(results["y_cov"])
        ax2.fill_between(X_test.ravel(), results["y_pred"].ravel() - 2*std_dev,
                         results["y_pred"].ravel() + 2*std_dev, alpha=0.2, color='red', label='95% Confidence')

        ax2.scatter(X_train.ravel(), true_function(X_train, alg=np),
                    color='red', s=50, edgecolor='black', label='Training Points', zorder=10)
        ax2.set_title("Combined Model Prediction with Uncertainty")
        ax2.set_xlabel("x")
        ax2.set_ylabel("f(x)")
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    analyze_submodel_contributions(results, X_train, config["submodel_groups"], oscillatory_function_with_trend)
