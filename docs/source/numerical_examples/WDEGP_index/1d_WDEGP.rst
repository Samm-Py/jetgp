Weighted  Derivative-Enhanced Gaussian Process Tutorial (Individual Submodels)
===============================================================================

This tutorial demonstrates a **1D individual submodel derivative-enhanced Gaussian Process (DEGP)**
implemented in a procedural style. Each training point forms its own submodel with all derivatives
up to a specified order. Predictions are combined using a weighted GP framework.

Key Features
------------

- One submodel per training point
- Uniform derivative structure across all submodels
- Comprehensive directional derivative computation
- Weighted combination of individual submodel predictions
- 1D function approximation

---

Import required packages
------------------------

.. jupyter-execute::

   import numpy as np
   import matplotlib.pyplot as plt
   import pyoti.sparse as oti
   from wdegp.wdegp import wdegp
   import utils
   import plotting_helper

---

Define example function
-----------------------

.. jupyter-execute::

   def oscillatory_function_with_trend(X, alg=oti):
       x1 = X[:, 0]
       return alg.sin(10 * np.pi * x1) / (2 * x1) + (x1 - 1) ** 4

---

Set experiment parameters
-------------------------

.. jupyter-execute::

   n_order = 2
   n_bases = 1
   lb_x = 0.5
   ub_x = 2.5
   num_points = 10
   test_points = 250
   kernel = "SE"
   kernel_type = "anisotropic"
   normalize = True
   n_restart_optimizer = 15
   swarm_size = 50
   random_seed = 42

   np.random.seed(random_seed)

---

Generate training points
------------------------

.. jupyter-execute::

   X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)
   print("Training points:", X_train.ravel())

---

Create individual submodel structure
-----------------------------------

.. jupyter-execute::

   submodel_indices = [[i] for i in range(num_points)]
   base_derivative_indices = utils.gen_OTI_indices(n_bases, n_order)
   derivative_specs = [base_derivative_indices for _ in range(num_points)]

   print(f"Number of submodels: {len(submodel_indices)}")
   print(f"Derivative types per submodel: {len(base_derivative_indices)}")

---

Compute derivatives and prepare submodel data
---------------------------------------------

.. jupyter-execute::

   y_function_values = oscillatory_function_with_trend(X_train, alg=np)

   submodel_data = []

   for k, idx in enumerate(submodel_indices):
       X_point = X_train[idx]
       X_oti = oti.array(X_point)
       for i in range(n_bases):
           for j in range(X_oti.shape[0]):
               X_oti[j, i] += oti.e(i + 1, order=n_order)

       y_hc = oti.array([oscillatory_function_with_trend(x, alg=oti) for x in X_oti])
       derivatives = []
       for i in range(len(base_derivative_indices)):
           for j in range(len(base_derivative_indices[i])):
               derivatives.append(y_hc.get_deriv(base_derivative_indices[i][j]).reshape(-1, 1))
       submodel_data.append([y_function_values] + derivatives)

---

Build weighted derivative-enhanced GP
-------------------------------------

.. jupyter-execute::

   gp_model = wdegp(
       X_train,
       submodel_data,
       n_order,
       n_bases,
       submodel_indices,
       derivative_specs,
       normalize=normalize,
       kernel=kernel,
       kernel_type=kernel_type
   )

---

Optimize hyperparameters
------------------------

.. jupyter-execute::

   params = gp_model.optimize_hyperparameters(
       n_restart_optimizer=n_restart_optimizer,
       swarm_size=swarm_size, verbose = False
   )
   print("Optimized hyperparameters:", params)

---

Evaluate model performance
--------------------------

.. jupyter-execute::

   X_test = np.linspace(lb_x, ub_x, test_points).reshape(-1, 1)
   y_pred, y_cov, submodel_vals, submodel_cov = gp_model.predict(
       X_test, params, calc_cov=True, return_submodels=True
   )
   y_true = oscillatory_function_with_trend(X_test, alg=np)
   nrmse = utils.nrmse(y_true, y_pred)
   print(f"NRMSE: {nrmse:.6f}")

---

Visualize combined prediction
-----------------------------

.. jupyter-execute::

   plt.figure(figsize=(10, 6))
   plt.plot(X_test, y_true, 'b-', label='True Function', linewidth=2)
   plt.plot(X_test.flatten(), y_pred.flatten(), 'r--', label='GP Prediction', linewidth=2)
   plt.fill_between(X_test.ravel(),
                    (y_pred.flatten() - 2*np.sqrt(y_cov)).ravel(),
                    (y_pred.flatten() + 2*np.sqrt(y_cov)).ravel(),
                    color='red', alpha=0.3, label='95% Confidence')
   plt.scatter(X_train, y_function_values, color='black', label='Training Points')
   plt.title("Weighted Individual Submodel DEGP")
   plt.xlabel("x")
   plt.ylabel("f(x)")
   plt.legend()
   plt.grid(alpha=0.3)
   plt.show()

---

Analyze individual submodel contributions
-----------------------------------------

.. jupyter-execute::

   colors = plt.cm.tab10(np.linspace(0, 1, len(submodel_vals)))
   plt.figure(figsize=(10, 6))
   for i, color in enumerate(colors):
       plt.plot(X_test.flatten(), submodel_vals[i].flatten(), color=color, alpha=0.6, label=f'Submodel {i+1}')
   plt.title("Individual Submodel Predictions")
   plt.xlabel("x")
   plt.ylabel("f(x)")
   plt.legend()
   plt.grid(alpha=0.3)
   plt.show()

---

Summary
-------

This experiment demonstrates a **1D individual submodel DEGP** without classes. Each training point contributes its full derivative information, and predictions are combined through a weighted GP framework. This approach captures **local behavior** while achieving **global approximation accuracy**.
