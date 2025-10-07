Derivative-Enhanced Gaussian Process (DEGP)
===========================================

Overview
--------
The **Derivative-Enhanced Gaussian Process (DEGP)** extends standard Gaussian Process (GP) regression
by incorporating derivative information at training points. By doing so, the model is able to:

- Capture local nonlinear behavior more accurately.
- Improve predictions in regions where function samples are sparse.
- Incorporate higher-order information to refine uncertainty estimates and predictive gradients.

This tutorial demonstrates DEGP for 1D functions using **first-order derivatives only**, and then with **both first- and second-order derivatives**.

Example 1: 1D First-Order Derivatives Only
------------------------------------------

**Objective:** Learn a 1D function \(f(x) = \sin(x)\) using both the function values and its first derivative \(\cos(x)\).

Data Requirements
~~~~~~~~~~~~~~~~~
DEGP requires three main inputs:

1. **X_train**: Locations of training points, shape `(n_points, n_features)`.  
   For a 1D function, `n_features = 1`.

2. **y_train**: List of arrays containing function values and derivative values at each training point.  
   - `y_train[0]` → function values \(f(x_i)\)  
   - `y_train[1]` → first-order derivatives \(f'(x_i)\)  

3. **der_indices**: Specifies which derivative corresponds to which hypercomplex tag.  
   - For first-order derivative in 1D: `[[[1,1]]]`.

.. jupyter-execute::

   import numpy as np
   from full_degp.degp import degp

   # Training inputs
   X_train = np.array([[0.0], [0.5], [1.0]])
   print("X_train:", X_train)

   # Function values
   y_func = np.sin(X_train)

   # First-order derivatives
   y_deriv1 = np.cos(X_train)

   # Combine function values and derivatives into a list
   y_train = [y_func, y_deriv1]
   print("\ny_train[0] (function values):", y_func)
   print("y_train[1] (first derivatives):", y_deriv1)

   # Derivative indices
   der_indices = [[[[1, 1]]]]
   print("\nder_indices:", der_indices)

Initialization
~~~~~~~~~~~~~~
To initialize a DEGP model:

- Specify the **training data** `X_train` and `y_train`.
- Define the **derivative order** `n_order=[1]`.
- Provide the **derivative indices**.
- Select a **kernel** (`SE`, `Matern`, etc.) and whether it is **anisotropic**.

.. jupyter-execute::

   model = degp(X_train, y_train, n_order=1, n_bases = 1, der_indices=der_indices,
                normalize=True, kernel="SE", kernel_type="anisotropic")
   print("DEGP model initialized:", model)

Training
~~~~~~~~
DEGP is trained by **maximizing the Marginal Log Likelihood (MLL)**.  
The hyperparameters are optimized with either **Particle Swarm Optimization (PSO)** or **JADE**.

.. jupyter-execute::

   params = model.optimize_hyperparameters(n_restart_optimizer=10, swarm_size=100)
   print("Optimized hyperparameters:", params)

Prediction
~~~~~~~~~~
Once trained, DEGP can make predictions at new input locations. For first-order derivative models, predictions can include function values only, derivatives, or both.

.. jupyter-execute::

   X_test = np.linspace(0, 1, 50).reshape(-1, 1)
   y_pred = model.predict(X_test, params, calc_cov=False, return_deriv=False)
   print("Predicted function values:", y_pred)

Example 2: 1D First- and Second-Order Derivatives
-------------------------------------------------

**Objective:** Learn the same function \(f(x) = \sin(x)\), but now using both first- and second-order derivatives to further constrain the GP.  
Including second-order derivatives allows DEGP to better capture the **curvature** of the function.

Data Requirements
~~~~~~~~~~~~~~~~~
- `X_train`: Training locations (same as Example 1).
- `y_train`: Now includes three arrays:
  - `y_train[0]` → function values \(f(x_i)\)
  - `y_train[1]` → first derivatives \(f'(x_i)\)
  - `y_train[2]` → second derivatives \(f''(x_i)\)

- `der_indices`: Specifies which hypercomplex tags correspond to each derivative.  
  - First-order: `[[[1,1]]]`  
  - Second-order: `[[[1,2]]]`

.. jupyter-execute::

   X_train = np.array([[0.0], [0.5], [1.0]])

   y_func = np.sin(X_train)
   y_deriv1 = np.cos(X_train)
   y_deriv2 = -np.sin(X_train)

   y_train = [y_func, y_deriv1, y_deriv2]

   der_indices = [[[[1,1]], [[1,2]]]]
   print("X_train:", X_train)
   print("y_train shapes:", [v.shape for v in y_train])
   print("der_indices:", der_indices)

Initialization
~~~~~~~~~~~~~~
.. jupyter-execute::

   model = degp(X_train, y_train, n_order=2, n_bases =1,  der_indices=der_indices,
                normalize=True, kernel="SE", kernel_type="anisotropic")
   print("DEGP model initialized:", model)

Training
~~~~~~~~
.. jupyter-execute::

   params = model.optimize_hyperparameters(n_restart_optimizer=10, swarm_size=100)
   print("Optimized hyperparameters:", params)

Prediction
~~~~~~~~~~
.. jupyter-execute::

   X_test = np.linspace(0, 1, 50).reshape(-1, 1)
   y_pred = model.predict(X_test, params, calc_cov=False, return_deriv=True)
   print("Predicted function values + derivatives:", y_pred)

Notes
-----
- **X_train** must be 2D, even for 1D functions.
- **y_train** is always a list of arrays: `[f_values, derivative_values...]`.
- **der_indices** defines which derivatives are included.
- DEGP supports **higher-order derivatives** by extending `y_train` and `der_indices`.
- Hyperparameters are tuned via **MLL maximization** using PSO or JADE.
