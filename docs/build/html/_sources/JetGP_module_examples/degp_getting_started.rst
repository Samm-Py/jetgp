Derivative-Enhanced Gaussian Process (DEGP)
===========================================

Overview
--------
The **Derivative-Enhanced Gaussian Process (DEGP)** extends standard Gaussian Process (GP) regression by incorporating derivative information at training points. This enables the model to:

- Capture local nonlinear behavior more accurately
- Improve predictions in regions where function samples are sparse
- Refine uncertainty estimates and predictive gradients using higher-order information

This tutorial demonstrates DEGP on 1D and 2D functions, first using **first-order derivatives only**, and then including **second-order derivatives** for improved curvature representation.

The examples also illustrate how DEGP predictions can be validated at training points and visualized over a range of inputs.

---

Example 1: 1D First-Order Derivatives Only
-------------------------------------------

Overview
~~~~~~~~
Learn a 1D function :math:`f(x) = \sin(x)` using both the function values and its first derivative :math:`f'(x) = \cos(x)`. Incorporating derivatives allows the GP to respect the slope of the function, leading to more accurate interpolation.

---

Step 1: Import required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   import numpy as np
   from full_degp.degp import degp
   import matplotlib.pyplot as plt

   print("Modules imported successfully.")

**Explanation:**  
We import ``numpy`` for numerical operations, ``matplotlib.pyplot`` for visualization, and the DEGP class from the ``full_degp`` package.

---

Step 2: Define training data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   X_train = np.array([[0.0], [0.5], [1.0]])
   y_func = np.sin(X_train)
   y_deriv1 = np.cos(X_train)
   y_train = [y_func, y_deriv1]

   print("X_train:", X_train)
   print("y_train:", y_train)

**Explanation:**  
Here, we define three training points and compute their function values and first derivatives. ``y_train`` is a list containing both values and derivatives, which the DEGP model will use for training.

---

Step 3: Define derivative indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   der_indices = [[[[1, 1]]]]
   print("der_indices:", der_indices)

**Explanation:**  
``der_indices`` tells the DEGP model which derivatives to include. In 1D, the first order derivative is labeled as ``[[1,1]]``.

---

Step 4: Initialize DEGP model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   model = degp(X_train, y_train, n_order=1, n_bases=1,
                der_indices=der_indices, normalize=True,
                kernel="SE", kernel_type="anisotropic")

   print("DEGP model initialized.")

**Explanation:**  
- ``n_order=1`` specifies first-order derivatives
- ``kernel="SE"`` uses a Squared Exponential kernel
- ``kernel_type="anisotropic"`` allows the model to learn separate length scales for each dimension
- ``normalize=True`` ensures that both function values and derivatives are scaled for numerical stability

---

Step 5: Optimize hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   params =  model.optimize_hyperparameters(
        optimizer='jade',
        pop_size = 100,
        n_generations = 15,
        local_opt_every = None,
        debug = False
        )
   print("Optimized hyperparameters:", params)

**Explanation:**  
Hyperparameters of the kernel (length scale, variance, noise, etc.) are optimized by maximizing the marginal log likelihood. Particle Swarm Optimization (PSO) is used here for robustness, with multiple restarts to avoid local minima.

---

Step 6: Validate predictions at training points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   y_train_pred = model.predict(X_train, params, calc_cov=False, return_deriv=True)
   y_func_pred = y_train_pred[:len(X_train)]
   y_deriv_pred = y_train_pred[len(X_train):]

   abs_error_func = np.abs(y_func_pred.flatten() - y_func.flatten())
   abs_error_deriv = np.abs(y_deriv_pred.flatten() - y_deriv1.flatten())

   print("Absolute error (function) at training points:", abs_error_func)
   print("Absolute error (derivative) at training points:", abs_error_deriv)

**Explanation:**  
We first check predictions at training points to ensure that the DEGP model exactly interpolates function values and derivatives, as expected for noiseless training data.

---

Step 7: Visualize predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   X_test = np.linspace(0, 1, 100).reshape(-1, 1)
   y_test_pred = model.predict(X_test, params, calc_cov=False, return_deriv=True)

   y_func_test = y_test_pred[:len(X_test)]
   y_deriv_test = y_test_pred[len(X_test):]

   plt.figure(figsize=(12,5))

   plt.subplot(1,2,1)
   plt.plot(X_test, np.sin(X_test), 'k--', label='True f(x)')
   plt.plot(X_test, y_func_test, 'r-', label='GP prediction')
   plt.scatter(X_train, y_func, c='b', s=50, label='Training points')
   plt.xlabel('x')
   plt.ylabel('f(x)')
   plt.title('Function and GP Prediction')
   plt.legend()

   plt.subplot(1,2,2)
   plt.plot(X_test, np.cos(X_test), 'k--', label="True f'(x)")
   plt.plot(X_test, y_deriv_test, 'r-', label='GP derivative prediction')
   plt.scatter(X_train, y_deriv1, c='b', s=50, label='Training points')
   plt.xlabel('x')
   plt.ylabel("f'(x)")
   plt.title('Derivative and GP Prediction')
   plt.legend()

   plt.tight_layout()
   plt.show()

**Explanation:**  
The plots compare DEGP predictions with the true function and derivatives. Notice how including derivative information allows the GP to more accurately capture the slope, even between training points.

---

Summary
~~~~~~~
This example demonstrates **first-order DEGP** on a 1D function. By incorporating derivative information at training points, the GP achieves:

- Exact interpolation of both function values and slopes at training locations
- Improved accuracy between training points
- Better extrapolation behavior near boundaries

---

Example 2: 1D First- and Second-Order Derivatives
--------------------------------------------------

Overview
~~~~~~~~
Improve the GP model by incorporating second-order derivatives. This gives the GP information about curvature, enhancing interpolation and derivative predictions.

---


Step 1: Define training data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   X_train = np.array([[0.0], [0.5], [1.0]])
   y_func = np.sin(X_train)
   y_deriv1 = np.cos(X_train)
   y_deriv2 = -np.sin(X_train)
   y_train = [y_func, y_deriv1, y_deriv2]

   print("X_train:", X_train)
   print("y_train:", y_train)

**Explanation:**  
The second derivative is included in the training data. DEGP can now enforce both slope and curvature.

---

Step 2: Define derivative indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   der_indices = [[[[1, 1]], [[1, 2]]]]
   print("der_indices:", der_indices)

**Explanation:**  
The first order derivative is labeled ``[[1,1]]`` and the second order derivative ``[[1,2]]``. This labeling maps derivatives to the structure used internally by DEGP.

---

Step 3: Initialize DEGP model for second-order derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   model = degp(X_train, y_train, n_order=2, n_bases=1,
                der_indices=der_indices, normalize=True,
                kernel="SE", kernel_type="anisotropic")

   print("DEGP model (2nd order) initialized.")

**Explanation:**  
Setting ``n_order=2`` instructs the model to incorporate both first- and second-order derivatives.

---

Step 4: Optimize hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   params = model.optimize_hyperparameters(
        optimizer='jade',
        pop_size = 100,
        n_generations = 15,
        local_opt_every = None,
        debug = True
        )
   print("Optimized hyperparameters:", params)

**Explanation:**  
Hyperparameters are optimized using PSO with multiple restarts for robustness.

---

Step 5: Validate predictions at training points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   y_train_pred = model.predict(X_train, params, calc_cov=False, return_deriv=True)

   n_train = len(X_train)
   y_func_pred = y_train_pred[:n_train]
   y_deriv1_pred = y_train_pred[n_train:2*n_train]
   y_deriv2_pred = y_train_pred[2*n_train:]

   abs_error_func = np.abs(y_func_pred.flatten() - y_func.flatten())
   abs_error_deriv1 = np.abs(y_deriv1_pred.flatten() - y_deriv1.flatten())
   abs_error_deriv2 = np.abs(y_deriv2_pred.flatten() - y_deriv2.flatten())

   print("Absolute error (function) at training points:", abs_error_func)
   print("Absolute error (1st derivative) at training points:", abs_error_deriv1)
   print("Absolute error (2nd derivative) at training points:", abs_error_deriv2)

**Explanation:**  
With second derivatives included, DEGP predictions should match both slopes and curvature at training points.

---

Step 6: Visualize predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   X_test = np.linspace(0, 1, 100).reshape(-1, 1)
   y_test_pred = model.predict(X_test, params, calc_cov=False, return_deriv=True)

   n_test = len(X_test)
   y_func_test = y_test_pred[:n_test]
   y_deriv1_test = y_test_pred[n_test:2*n_test]
   y_deriv2_test = y_test_pred[2*n_test:]

   plt.figure(figsize=(18,5))

   plt.subplot(1,3,1)
   plt.plot(X_test, np.sin(X_test), 'k--', label='True f(x)')
   plt.plot(X_test, y_func_test, 'r-', label='GP prediction')
   plt.scatter(X_train, y_func, c='b', s=50, label='Training points')
   plt.xlabel('x')
   plt.ylabel('f(x)')
   plt.title('Function')
   plt.legend()

   plt.subplot(1,3,2)
   plt.plot(X_test, np.cos(X_test), 'k--', label="True f'(x)")
   plt.plot(X_test, y_deriv1_test, 'r-', label='GP derivative prediction')
   plt.scatter(X_train, y_deriv1, c='b', s=50, label='Training points')
   plt.xlabel('x')
   plt.ylabel("f'(x)")
   plt.title('1st Derivative')
   plt.legend()

   plt.subplot(1,3,3)
   plt.plot(X_test, -np.sin(X_test), 'k--', label="True f''(x)")
   plt.plot(X_test, y_deriv2_test, 'r-', label='GP 2nd derivative prediction')
   plt.scatter(X_train, y_deriv2, c='b', s=50, label='Training points')
   plt.xlabel('x')
   plt.ylabel("f''(x)")
   plt.title('2nd Derivative')
   plt.legend()

   plt.tight_layout()
   plt.show()

**Explanation:**  
The three-panel plot shows function, first derivative, and second derivative predictions. Including higher-order derivatives improves the model's ability to interpolate curvature between points.

---

Summary
~~~~~~~
This example demonstrates **second-order DEGP** on a 1D function. By incorporating both first and second derivatives:

- The GP captures local curvature information
- Predictions are more accurate in regions with high curvature
- Derivative predictions are smoother and more reliable

---

Example 3: 2D First-Order Derivatives
--------------------------------------

Overview
~~~~~~~~
Learn a 2D function :math:`f(x,y) = \sin(x)\cos(y)` using function values and first derivatives with respect to :math:`x` and :math:`y`.

---

Step 1: Import required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   import numpy as np
   from full_degp.degp import degp
   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D

   print("Modules imported successfully for 2D example.")

**Explanation:**  
Additional imports for 3D visualization are included.

---

Step 2: Define training data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   X1 = np.array([0.0, 0.5, 1.0])
   X2 = np.array([0.0, 0.5, 1.0])
   X1_grid, X2_grid = np.meshgrid(X1, X2)
   X_train = np.column_stack([X1_grid.flatten(), X2_grid.flatten()])

   y_func = np.sin(X_train[:,0]) * np.cos(X_train[:,1])
   y_deriv_x = np.cos(X_train[:,0]) * np.cos(X_train[:,1])
   y_deriv_y = -np.sin(X_train[:,0]) * np.sin(X_train[:,1])
   y_train = [y_func.reshape(-1,1), y_deriv_x.reshape(-1,1), y_deriv_y.reshape(-1,1)]

   print("X_train shape:", X_train.shape)
   print("y_train shapes:", [v.shape for v in y_train])

**Explanation:**  
We define a 3×3 2D grid. Function values and first derivatives in both dimensions are included, allowing DEGP to learn the local slopes along each axis.

---

Step 3: Define derivative indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   der_indices = [[[[1, 1]], [[2, 1]]]]
   print("der_indices:", der_indices)

**Explanation:**  
The derivative indices specify first derivatives with respect to :math:`x` (``[[1,1]]``) and :math:`y` (``[[2,1]]``).

---

Step 4: Initialize DEGP model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   model = degp(X_train, y_train, n_order=1, n_bases=2,
                der_indices=der_indices, normalize=True,
                kernel="SE", kernel_type="anisotropic")

   print("2D DEGP model initialized.")

**Explanation:**  
The model uses two bases (``n_bases=2``) to capture multi-dimensional derivative information effectively.

---

Step 5: Optimize hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   params = model.optimize_hyperparameters(
        optimizer='jade',
        pop_size = 100,
        n_generations = 15,
        local_opt_every = None,
        debug = False
        )
   print("Optimized hyperparameters:", params)

**Explanation:**  
Hyperparameter optimization for 2D problems may take longer due to increased dimensionality.

---

Step 6: Validate predictions at training points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   y_train_pred = model.predict(X_train, params, calc_cov=False, return_deriv=True)

   n_train = len(X_train)
   y_func_pred = y_train_pred[:n_train]
   y_deriv_x_pred = y_train_pred[n_train:2*n_train]
   y_deriv_y_pred = y_train_pred[2*n_train:]

   print("Predictions at training points (function):", y_func_pred)
   print("Predictions at training points (deriv x):", y_deriv_x_pred)
   print("Predictions at training points (deriv y):", y_deriv_y_pred)

**Explanation:**  
Predictions are split into function values and derivatives for each dimension.

---

Step 7: Compute absolute errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   abs_error_func = np.abs(y_func_pred.flatten() - y_func)
   abs_error_dx = np.abs(y_deriv_x_pred.flatten() - y_deriv_x)
   abs_error_dy = np.abs(y_deriv_y_pred.flatten() - y_deriv_y)

   print("Absolute error (function) at training points:", abs_error_func)
   print("Absolute error (deriv x) at training points:", abs_error_dx)
   print("Absolute error (deriv y) at training points:", abs_error_dy)

**Explanation:**  
Verification that the model exactly interpolates training data.

---

Step 8: Generate test predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   x1_test = np.linspace(0, 1, 50)
   x2_test = np.linspace(0, 1, 50)
   X1_test, X2_test = np.meshgrid(x1_test, x2_test)
   X_test = np.column_stack([X1_test.flatten(), X2_test.flatten()])

   y_test_pred = model.predict(X_test, params, calc_cov=False, return_deriv=True)
   n_test = len(X_test)

   y_func_test = y_test_pred[:n_test].reshape(50,50)
   y_deriv_x_test = y_test_pred[n_test:2*n_test].reshape(50,50)
   y_deriv_y_test = y_test_pred[2*n_test:].reshape(50,50)

**Explanation:**  
A dense 50×50 test grid is created for visualization purposes.

---

Step 9: Visualize 2D predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   import matplotlib.pyplot as plt

   # Compute absolute errors for function and first derivatives
   abs_error_func = np.abs(y_func_test - (np.sin(X1_test) * np.cos(X2_test)))
   abs_error_dx   = np.abs(y_deriv_x_test - (np.cos(X1_test) * np.cos(X2_test)))
   abs_error_dy   = np.abs(y_deriv_y_test - (-np.sin(X1_test) * np.sin(X2_test)))

   # Setup figure
   fig, axs = plt.subplots(1, 3, figsize=(18, 5))

   # Titles and error arrays
   titles = [
      r'Absolute Error: $f(x,y)$',
      r'Absolute Error: $\partial f / \partial x$',
      r'Absolute Error: $\partial f / \partial y$'
   ]
   errors = [abs_error_func, abs_error_dx, abs_error_dy]
   cmaps  = ['Reds', 'Greens', 'Blues']

   # Plot each contour
   for ax, err, title, cmap in zip(axs, errors, titles, cmaps):
      cs = ax.contourf(X1_test, X2_test, err, levels=50, cmap=cmap)
      ax.scatter(X_train[:,0], X_train[:,1], c='k', s=40)
      ax.set_title(title, fontsize=14)
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      fig.colorbar(cs, ax=ax)

   plt.tight_layout()
   plt.show()

**Explanation:**  
The contour plots show absolute errors for the function and its first derivatives. Including derivatives along each dimension allows the GP to capture local slopes, resulting in smoother and more accurate surfaces.

---

Summary
~~~~~~~
This example demonstrates **2D first-order DEGP**. By incorporating partial derivatives:

- The GP learns directional slopes in the 2D space
- Predictions are accurate across the entire domain
- The model captures local gradient information effectively

---

Example 4: 2D Second-Order (Main) Derivatives
----------------------------------------------

Overview
~~~~~~~~
Learn a 2D function :math:`f(x,y) = \sin(x)\cos(y)` using function values, first derivatives, and **second-order derivatives** :math:`\frac{\partial^2 f}{\partial x^2}` and :math:`\frac{\partial^2 f}{\partial y^2}` (excluding mixed partials).

---

Step 1: Import required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   import numpy as np
   from full_degp.degp import degp
   import matplotlib.pyplot as plt
   from mpl_toolkits.mplot3d import Axes3D

   print("Modules imported successfully for 2D example with second-order derivatives.")

**Explanation:**  
Same imports as the previous 2D example.

---

Step 2: Define training data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   X1 = np.array([0.0, 0.5, 1.0])
   X2 = np.array([0.0, 0.5, 1.0])
   X1_grid, X2_grid = np.meshgrid(X1, X2)
   X_train = np.column_stack([X1_grid.flatten(), X2_grid.flatten()])

   # True function and derivatives
   y_func = np.sin(X_train[:,0]) * np.cos(X_train[:,1])
   y_deriv_x = np.cos(X_train[:,0]) * np.cos(X_train[:,1])
   y_deriv_y = -np.sin(X_train[:,0]) * np.sin(X_train[:,1])
   y_deriv_xx = -np.sin(X_train[:,0]) * np.cos(X_train[:,1])
   y_deriv_yy = -np.sin(X_train[:,0]) * np.cos(X_train[:,1])

   y_train = [
      y_func.reshape(-1,1),
      y_deriv_x.reshape(-1,1),
      y_deriv_y.reshape(-1,1),
      y_deriv_xx.reshape(-1,1),
      y_deriv_yy.reshape(-1,1)
   ]

   print("X_train shape:", X_train.shape)
   print("y_train shapes:", [v.shape for v in y_train])

**Explanation:**  
This dataset includes function values, first derivatives, and second-order main derivatives, allowing DEGP to leverage curvature information for improved learning accuracy.

---

Step 3: Define derivative indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   der_indices = [
       [ [[1, 1]], [[2, 1]] ],  # first-order derivatives
       [ [[1, 2]], [[2, 2]] ]   # second-order derivatives
   ]
   print("der_indices:", der_indices)

**Explanation:**  
The derivative indices specify both first-order and second-order partial derivatives with respect to :math:`x` and :math:`y`.

---

Step 4: Initialize DEGP model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   model = degp(X_train, y_train, n_order=2, n_bases=2,
                der_indices=der_indices, normalize=True,
                kernel="SE", kernel_type="anisotropic")

   print("2D DEGP model with second-order derivatives initialized.")

**Explanation:**  
Setting ``n_order=2`` enables the model to use both first and second-order derivative information.

---

Step 5: Optimize hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   params =   model.optimize_hyperparameters(
        optimizer='jade',
        pop_size = 100,
        n_generations = 15,
        local_opt_every = None,
        debug = False
        )
   print("Optimized hyperparameters:", params)

**Explanation:**  
Hyperparameters are optimized to balance function values and multiple orders of derivatives.

---

Step 6: Validate predictions at training points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   y_train_pred = model.predict(X_train, params, calc_cov=False, return_deriv=True)

   n_train = len(X_train)

   # First-order
   y_func_pred    = y_train_pred[:n_train]
   y_deriv_x_pred = y_train_pred[n_train:2*n_train]
   y_deriv_y_pred = y_train_pred[2*n_train:3*n_train]

   # Second-order main derivatives
   y_deriv_xx_pred = y_train_pred[3*n_train:4*n_train]
   y_deriv_yy_pred = y_train_pred[4*n_train:5*n_train]

**Explanation:**  
Predictions are organized by derivative order and type.

---

Step 7: Compute absolute errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   # First-order errors
   abs_error_func = np.abs(y_func_pred.flatten() - y_func)
   abs_error_dx   = np.abs(y_deriv_x_pred.flatten() - y_deriv_x)
   abs_error_dy   = np.abs(y_deriv_y_pred.flatten() - y_deriv_y)

   # Second-order main derivative errors
   abs_error_dxx  = np.abs(y_deriv_xx_pred.flatten() - y_deriv_xx)
   abs_error_dyy  = np.abs(y_deriv_yy_pred.flatten() - y_deriv_yy)

   # Print absolute errors
   print("Absolute error (function)       :", abs_error_func)
   print("Absolute error (derivative x)  :", abs_error_dx)
   print("Absolute error (derivative y)  :", abs_error_dy)
   print("Absolute error (second x-x)    :", abs_error_dxx)
   print("Absolute error (second y-y)    :", abs_error_dyy)

**Explanation:**  
Verification that all derivatives are correctly interpolated at training points.

---

Step 8: Generate test predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   x1_test = np.linspace(0, 1, 50)
   x2_test = np.linspace(0, 1, 50)
   X1_test, X2_test = np.meshgrid(x1_test, x2_test)
   X_test = np.column_stack([X1_test.flatten(), X2_test.flatten()])

   y_test_pred = model.predict(X_test, params, calc_cov=False, return_deriv=True)
   n_test = len(X_test)

   y_func_test = y_test_pred[:n_test].reshape(50,50)
   y_deriv_x_test = y_test_pred[n_test:2*n_test].reshape(50,50)
   y_deriv_y_test = y_test_pred[2*n_test:3*n_test].reshape(50,50)
   y_deriv_xx_test = y_test_pred[3*n_test:4*n_test].reshape(50,50)
   y_deriv_yy_test = y_test_pred[4*n_test:5*n_test].reshape(50,50)

**Explanation:**  
All predictions (function and derivatives) are computed and reshaped for visualization.

---

Step 9: Visualize absolute errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   import matplotlib.pyplot as plt

   # Compute absolute errors
   abs_error_func = np.abs(y_func_test - (np.sin(X1_test) * np.cos(X2_test)))
   abs_error_dx   = np.abs(y_deriv_x_test - (np.cos(X1_test) * np.cos(X2_test)))
   abs_error_dy   = np.abs(y_deriv_y_test - (-np.sin(X1_test) * np.sin(X2_test)))
   abs_error_xx   = np.abs(y_deriv_xx_test - (-np.sin(X1_test) * np.cos(X2_test)))
   abs_error_yy   = np.abs(y_deriv_yy_test - (-np.sin(X1_test) * np.cos(X2_test)))

   # Setup figure
   fig, axs = plt.subplots(1, 5, figsize=(24, 5))

   # Titles and errors
   titles = [
      r'Absolute Error: $f(x,y)$',
      r'Absolute Error: $\partial f / \partial x$',
      r'Absolute Error: $\partial f / \partial y$',
      r'Absolute Error: $\partial^2 f / \partial x^2$',
      r'Absolute Error: $\partial^2 f / \partial y^2$'
   ]

   errors = [abs_error_func, abs_error_dx, abs_error_dy, abs_error_xx, abs_error_yy]
   cmaps  = ['Reds', 'Greens', 'Blues', 'Greens', 'Blues']

   # Plot all
   for ax, err, title, cmap in zip(axs, errors, titles, cmaps):
      cs = ax.contourf(X1_test, X2_test, err, levels=50, cmap=cmap)
      ax.scatter(X_train[:,0], X_train[:,1], c='k', s=40)
      ax.set_title(title, fontsize=12)
      ax.set_xlabel('x')
      ax.set_ylabel('y')
      fig.colorbar(cs, ax=ax)

   plt.tight_layout()
   plt.show()

**Explanation:**  
Including second-order derivatives helps DEGP capture curvature and local convexity/concavity information, enhancing interpolation and extrapolation accuracy near sparse training regions.

---

Summary
~~~~~~~
This example demonstrates **2D second-order DEGP**. By incorporating both first and second-order partial derivatives:

- The GP captures local curvature in multiple dimensions
- Predictions account for surface bending and convexity
- Accuracy is significantly improved in regions with high curvature
- The model provides reliable derivative predictions across the domain

**Key Benefits of Higher-Order Derivatives:**

- More accurate interpolation between sparse training points
- Better extrapolation behavior
- Improved gradient predictions for optimization applications
- Reduced uncertainty in high-curvature regions