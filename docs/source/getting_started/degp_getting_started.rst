Derivative-Enhanced Gaussian Process (DEGP)
===========================================

Overview
--------
The **Derivative-Enhanced Gaussian Process (DEGP)** extends standard Gaussian Process (GP) regression
by incorporating derivative information at training points. This enables the model to:

- Capture local nonlinear behavior more accurately.
- Improve predictions in regions where function samples are sparse.
- Refine uncertainty estimates and predictive gradients using higher-order information.

This tutorial demonstrates DEGP for a 1D function using **first-order derivatives only**, and then with **both first- and second-order derivatives**.

------------------------------------------------------------

Example 1: 1D First-Order Derivatives Only
------------------------------------------

**Objective:**  
Learn a 1D function :math:`f(x) = \sin(x)` using both the function values and its first derivative :math:`f'(x) = \cos(x)`.

### Step 1: Import required packages

.. jupyter-execute::

   import numpy as np
   from full_degp.degp import degp

   print("Modules imported successfully.")

### Step 2: Define training data

We define three training points at `x = [0.0, 0.5, 1.0]`.  
The corresponding function values and derivatives are computed using sine and cosine.

.. jupyter-execute::

   X_train = np.array([[0.0], [0.5], [1.0]])
   y_func = np.sin(X_train)
   y_deriv1 = np.cos(X_train)
   y_train = [y_func, y_deriv1]

   print("X_train:", X_train)
   print("Function values:", y_func)
   print("First derivatives:", y_deriv1)

### Step 3: Define derivative indices

`der_indices` specifies which derivatives correspond to hypercomplex tags.
For a single first derivative in 1D: `[[[[1, 1]]]]`.

.. jupyter-execute::

   der_indices = [[[[1, 1]]]]
   print("der_indices:", der_indices)

### Step 4: Initialize DEGP model

We create a DEGP model for first-order derivatives.  
Key settings:
- `n_order=1` → first-order derivatives  
- `kernel="SE"` → Squared Exponential kernel  
- `kernel_type="anisotropic"` → allows per-dimension length scales  

.. jupyter-execute::

   model = degp(X_train, y_train, n_order=1, n_bases=1,
                der_indices=der_indices, normalize=True,
                kernel="SE", kernel_type="anisotropic")

   print("DEGP model initialized.")

### Step 5: Optimize hyperparameters

Hyperparameters are optimized by maximizing the Marginal Log Likelihood (MLL)
using Particle Swarm Optimization (PSO) with 10 restarts and swarm size 100.

.. jupyter-execute::

   params = model.optimize_hyperparameters(n_restart_optimizer=10, swarm_size=100)
   print("Optimized hyperparameters:", params)

### Step 6: Make predictions

Predict the function at 50 evenly spaced test points.  
Set `return_deriv=False` to predict only function values.

.. jupyter-execute::

   X_test = np.linspace(0, 1, 50).reshape(-1, 1)
   y_pred = model.predict(X_test, params, calc_cov=False, return_deriv=False)
   print("Predicted function values:", y_pred)

------------------------------------------------------------

Example 2: 1D First- and Second-Order Derivatives
-------------------------------------------------

**Objective:**  
Learn the same function :math:`f(x) = \sin(x)`, now including both first and second derivatives.  
Adding second derivatives helps the GP better represent curvature.

### Step 1: Define training data

We use the same training locations, but include:
- Function values \( f(x) \)
- First derivatives \( f'(x) = \cos(x) \)
- Second derivatives \( f''(x) = -\sin(x) \)

.. jupyter-execute::

   X_train = np.array([[0.0], [0.5], [1.0]])
   y_func = np.sin(X_train)
   y_deriv1 = np.cos(X_train)
   y_deriv2 = -np.sin(X_train)

   y_train = [y_func, y_deriv1, y_deriv2]
   print("y_train shapes:", [v.shape for v in y_train])

### Step 2: Define derivative indices

The indices now include both first- and second-order derivative tags.

.. jupyter-execute::

   der_indices = [[[[1, 1]], [[1, 2]]]]
   print("der_indices:", der_indices)

### Step 3: Initialize DEGP model for second-order derivatives

We set `n_order=2` to include up to second derivatives.

.. jupyter-execute::

   model = degp(X_train, y_train, n_order=2, n_bases=1,
                der_indices=der_indices, normalize=True,
                kernel="SE", kernel_type="anisotropic")

   print("DEGP model (2nd order) initialized.")

### Step 4: Optimize hyperparameters

.. jupyter-execute::

   params = model.optimize_hyperparameters(n_restart_optimizer=10, swarm_size=100)
   print("Optimized hyperparameters:", params)

### Step 5: Make predictions (function and derivatives)

Set `return_deriv=True` to get both function values and derivatives.

.. jupyter-execute::

   X_test = np.linspace(0, 1, 50).reshape(-1, 1)
   y_pred = model.predict(X_test, params, calc_cov=False, return_deriv=True)
   print("Predicted function and derivatives:", y_pred)

------------------------------------------------------------

Notes
-----
- **X_train** must be 2D even for 1D functions.  
- **y_train** is always a list of arrays: `[f_values, derivative_values, ...]`.  
- **der_indices** defines which derivatives correspond to which hypercomplex tags.  
- DEGP supports higher-order derivatives by extending `y_train` and `der_indices`.  
- Hyperparameters are tuned by maximizing the **Marginal Log Likelihood (MLL)** using PSO or JADE.
