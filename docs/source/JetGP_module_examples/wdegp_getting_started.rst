Weighted Derivative-Enhanced Gaussian Process (WDEGP)
====================================================

Weighted Individual Submodel Framework
--------------------------------------

Overview
~~~~~~~~
The **Weighted Derivative-Enhanced Gaussian Process (WDEGP)** extends the DEGP framework by constructing **individual Gaussian Process submodels**. Each submodel is trained to interpolate the function values at all training locations and derivative information at a subset of training locations. The global prediction is obtained through a **weighted aggregation** of these local models.

**Submodel Construction:**

Each submodel :math:`k` is a Gaussian Process that interpolates:

- The function values at **all** training points :math:`\{\mathbf{x}_1, \mathbf{x}_2, \ldots, \mathbf{x}_N\}`
- Derivative information (up to a specified order) at selected training locations

This ensures that each submodel :math:`y_k(\mathbf{x})` satisfies both the global function interpolation conditions and its assigned local derivative constraints.

**Global Model:**

The global prediction is formed as a weighted combination of the submodels:

.. math::
   y(\mathbf{x}) = \sum_{k=1}^{M} w_k(\mathbf{x}) \, y_k(\mathbf{x})

where :math:`M` is the number of submodels and :math:`1 \leq M \leq N`

**Global Interpolation Guarantee:**

The weighting functions :math:`w_k(\mathbf{x})` are constructed to satisfy a partition of unity with the Kronecker delta property:

.. math::
   \sum_{k=1}^{M} w_k(\mathbf{x}_i) = 1 \quad \text{and} \quad w_j(\mathbf{x}_i) = \delta_{ij}

This means that at each training point :math:`\mathbf{x}_i`:

- Only the corresponding submodel :math:`j=i` contributes (with weight :math:`w_i(\mathbf{x}_i) = 1`)
- All other submodels have zero weight (:math:`w_k(\mathbf{x}_i) = 0` for :math:`k \neq i`)

**Consequence:** Since each submodel interpolates its training data, and the weighting scheme ensures that only submodel :math:`i` contributes at training point :math:`\mathbf{x}_i`, the global WDEGP model inherits the interpolation properties of the individual submodels. Therefore, the global model is **mathematically guaranteed** to interpolate both function values and derivatives at all training points where these constraints are imposed.

This formulation enables:

- **Improved scalability with problem dimension**: Each submodel is trained on a localized subset of data rather than the full global dataset, significantly reducing computational cost in high-dimensional problems
- Local adaptation to nonlinearities and curvature  
- Consistent inclusion of higher-order derivative information  
- Smooth global approximations that respect local dynamics  
- Parallelizable submodel evaluations

---

Key Data Structures: ``submodel_indices`` and ``derivative_specs``
------------------------------------------------------------------

Understanding how to specify which training points use which derivatives is critical for WDEGP. The framework uses two parallel data structures:

**``submodel_indices``**: Specifies which training point indices are associated with each derivative type in each submodel.

**``derivative_specs``** (also called ``der_indices``): Specifies which derivative types are used by each submodel.

These two structures must have matching shapes—each derivative type specification in ``derivative_specs`` corresponds to a list of training point indices in ``submodel_indices``.

**Important**: Indices do **not** need to be contiguous. You can directly use indices like ``[0, 2, 4, 6, 8]`` without any reordering of your training data.

Structure Overview
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # General structure:
   submodel_indices = [
       [indices_for_deriv_type_0, indices_for_deriv_type_1, ...],  # Submodel 0
       [indices_for_deriv_type_0, indices_for_deriv_type_1, ...],  # Submodel 1
       ...
   ]
   
   derivative_specs = [
       [deriv_type_0_spec, deriv_type_1_spec, ...],  # Submodel 0
       [deriv_type_0_spec, deriv_type_1_spec, ...],  # Submodel 1
       ...
   ]

**Access pattern**: ``submodel_indices[submodel_idx][deriv_type_idx]`` gives the list of training point indices that have the derivative type specified by ``derivative_specs[submodel_idx][deriv_type_idx]``.

Detailed Example
~~~~~~~~~~~~~~~~

Consider a problem with 10 training points and 2 submodels where:

- **Submodel 0**: Uses 1st-order derivatives at points [1,2,3] and 2nd-order derivatives at points [4,5,6]
- **Submodel 1**: Uses 1st-order derivatives at points [4,5,6] and 2nd-order derivatives at points [1,2,3]

.. code-block:: python

   # Submodel indices: which points have which derivative type
   submodel_indices = [
       [[1, 2, 3], [4, 5, 6]],  # Submodel 0: 1st order at [1,2,3], 2nd order at [4,5,6]
       [[4, 5, 6], [1, 2, 3]]   # Submodel 1: 1st order at [4,5,6], 2nd order at [1,2,3]
   ]
   
   # Derivative specifications: what derivative types each submodel uses
   derivative_specs = [
       [[[[1, 1]]], [[[1, 2]]]],    # Submodel 0: 1st order [[[1,1]]], 2nd order [[[1,2]]]
       [[[[1, 1]]], [[[1, 2]]]]     # Submodel 1: 1st order [[[1,1]]], 2nd order [[[1,2]]]
   ]

**Interpretation:**

- ``submodel_indices[0][0] = [1, 2, 3]`` means Submodel 0 has the derivative type ``derivative_specs[0][0] = [[[1,1]]]`` (1st order) at training points 1, 2, and 3
- ``submodel_indices[0][1] = [4, 5, 6]`` means Submodel 0 has the derivative type ``derivative_specs[0][1] = [[[1,2]]]`` (2nd order) at training points 4, 5, and 6
- ``submodel_indices[1][0] = [4, 5, 6]`` means Submodel 1 has the derivative type ``derivative_specs[1][0] = [[[1,1]]]`` (1st order) at training points 4, 5, and 6
- ``submodel_indices[1][1] = [1, 2, 3]`` means Submodel 1 has the derivative type ``derivative_specs[1][1] = [[[1,2]]]`` (2nd order) at training points 1, 2, and 3

**Key Features:**

1. **Non-contiguous indices**: Indices like ``[1, 2, 3]`` or ``[0, 2, 4, 6, 8]`` can be used directly without reordering training data
2. **Different indices per derivative type**: Within a single submodel, different derivative types can be applied at different training points
3. **Flexible submodel design**: Each submodel can have a completely different configuration of derivative types and indices

Simple Case: Same Indices for All Derivative Types
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When all derivative types in a submodel use the same training points (the most common case), the structure simplifies:

.. code-block:: python

   # 10 submodels, each at one training point with all derivatives at that point
   submodel_indices = [[[i], [i]] for i in range(10)]  # Same index for each deriv type
   
   derivative_specs = [[[[[1, 1]]], [[[1, 2]]]] for _ in range(10)]  # Same specs for all

Or for a single submodel using all points:

.. code-block:: python

   # Single submodel with derivatives at points [2, 3, 4, 5]
   submodel_indices = [[[2, 3, 4, 5], [2, 3, 4, 5]]]  # Same indices for both deriv types
   
   derivative_specs = [[[[[1, 1]]], [[[1, 2]]]]]  # 1st and 2nd order derivatives

---

Derivative Predictions
----------------------

WDEGP supports derivative predictions via the ``return_deriv`` parameter in the ``predict`` method. This allows direct prediction of derivatives without using finite differences.

**Requirements for ``return_deriv=True``:**

All submodels must have the **same derivative specifications** (same derivative types). The indices can be different - only the derivative types need to match.

**Single submodel case:**

For a single submodel, ``return_deriv=True`` works straightforwardly - you can predict derivatives at any test point.

.. code-block:: python

   # Single submodel with derivatives at some training points
   derivative_specs = [[[[[1, 1]]], [[[1, 2]]]]]
   
   # Can predict derivatives at any test point
   y_pred, y_cov = gp_model.predict(X_test, params, calc_cov=True, return_deriv=True)

**Multi-submodel case:**

When multiple submodels are used, derivative predictions are available if all submodels share the **same derivative specifications** (same derivative types). The indices can be completely different across submodels.

.. code-block:: python

   # Both submodels have same derivative types (indices can differ)
   derivative_specs = [
       [[[[1, 1]]], [[[1, 2]]]],    # Submodel 0
       [[[[1, 1]]], [[[1, 2]]]]     # Submodel 1
   ]
   
   # return_deriv=True works even with different indices!
   y_pred, y_cov = gp_model.predict(X_test, params, calc_cov=True, return_deriv=True)

**When ``return_deriv=True`` is NOT available:**

If submodels have **different derivative specifications**, ``return_deriv=True`` cannot be used:

.. code-block:: python

   # Different derivative specs - return_deriv NOT available
   derivative_specs = [
       [[[[1, 1]]]],              # Submodel 0: only 1st order
       [[[[1, 2]]]]               # Submodel 1: only 2nd order
   ]
   # Must use finite differences or access individual submodels

**Verification approaches:**

1. **Direct prediction** (preferred): Use ``return_deriv=True`` when all submodels share the same derivative types
2. **Individual submodels**: Use ``return_submodels=True`` to access each submodel's predictions separately
3. **Finite differences**: Apply finite differences to the weighted function predictions (fallback)

**Usage:**

.. code-block:: python

   # Standard prediction (function values only)
   y_pred, y_cov = gp_model.predict(X_test, params, calc_cov=True)
   # y_pred shape: (1, num_points)
   
   # With derivative predictions (when derivative_specs match across submodels)
   y_pred, y_cov = gp_model.predict(X_test, params, calc_cov=True, return_deriv=True)
   # y_pred shape: (num_deriv_types + 1, num_points)
   # Row 0: function values at all points
   # Row 1: 1st derivative at all points
   # Row 2: 2nd derivative at all points (if applicable)
   # etc.
   
   # With individual submodel outputs
   y_pred, y_cov, submodel_vals, submodel_cov = gp_model.predict(
       X_test, params, calc_cov=True, return_submodels=True
   )

---

Example 1: 1D Weighted DEGP with Individual Submodels
------------------------------------------------------

This tutorial demonstrates WDEGP on a **1D oscillatory function with trend**, using second-order derivatives to enhance smoothness and predictive accuracy. In particular, we consider 10 training points and construct 10 corresponding submodels—one centered at each training point.

Since all submodels use the same derivative types, we can use ``return_deriv=True`` to directly verify derivative interpolation without finite differences.

Step 1: Import required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   import numpy as np
   import matplotlib.pyplot as plt
   from jetgp.wdegp.wdegp import wdegp
   import jetgp.utils as utils

**Explanation:**  
We import the required modules for numerical operations, visualization, and the WDEGP framework.

---

Step 2: Define the example function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   import sympy as sp
   
   # Define function symbolically for exact derivatives
   x_sym = sp.symbols('x')
   f_sym = sp.sin(10 * sp.pi * x_sym) / (2 * x_sym) + (x_sym - 1)**4
   
   # Compute derivatives symbolically
   f1_sym = sp.diff(f_sym, x_sym)
   f2_sym = sp.diff(f_sym, x_sym, 2)
   
   # Convert to callable NumPy functions
   f_fun = sp.lambdify(x_sym, f_sym, "numpy")
   f1_fun = sp.lambdify(x_sym, f1_sym, "numpy")
   f2_fun = sp.lambdify(x_sym, f2_sym, "numpy")

**Explanation:**  
This benchmark function combines an oscillatory component with a smooth polynomial trend. We use SymPy for exact symbolic differentiation.

---

Step 3: Set experiment parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::
   
   n_bases = 1
   n_order = 2   
   lb_x = 0.5
   ub_x = 2.5
   num_points = 10

**Explanation:**  
Since this is a one-dimensional problem, we set `n_bases = 1`. We use second-order derivatives, thus `n_order = 2`. The 10 training points will be uniformly distributed between 0.5 and 2.5.

---

Step 4: Generate training points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)
   print("Training points:", X_train.ravel())

**Explanation:**  
Each training point will form the center of a local submodel that contributes to the overall WDEGP prediction.

---

Step 5: Create individual submodel structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   # Each submodel corresponds to one training point
   # Each submodel has both 1st and 2nd order derivatives at its single point
   # Structure: submodel_indices[submodel_idx][deriv_type_idx] = list of point indices
   submodel_indices = [[[i], [i]] for i in range(num_points)]
   
   # Derivative specifications: 1st order [[1,1]] and 2nd order [[1,2]] for each submodel
   # All submodels use the SAME derivative types - this enables return_deriv=True
   derivative_specs = [utils.gen_OTI_indices(n_bases, n_order) for _ in range(num_points)]

   print(f"Number of submodels: {len(submodel_indices)}")
   print(f"Derivative types per submodel: {len(derivative_specs[0])}")
   print(f"\nExample - Submodel 0:")
   print(f"  submodel_indices[0] = {submodel_indices[0]}")
   print(f"  derivative_specs[0] = {derivative_specs[0]}")
   print(f"  Meaning: 1st order deriv at point {submodel_indices[0][0]}, 2nd order deriv at point {submodel_indices[0][1]}")
   print(f"\nSince all submodels share the same derivative_specs, return_deriv=True is available!")

**Explanation:**  
In this example, each submodel corresponds to a single training point. The ``submodel_indices`` structure shows that for each submodel, both derivative types (1st and 2nd order) are applied at the same single point.

**Key insight**: Since all submodels use the same ``derivative_specs`` (both have ``[[1,1]]`` and ``[[1,2]]``), we can use ``return_deriv=True`` to get derivative predictions directly from the weighted model.

---

Step 6: Compute function values and derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   # Compute function values at all training points
   y_function_values = f_fun(X_train.flatten()).reshape(-1, 1)

   # Prepare submodel data
   submodel_data = []
   
   for k in range(num_points):
       xval = X_train[k, 0]
       
       # Compute derivatives at this point
       d1 = np.array([[f1_fun(xval)]])  # First derivative
       d2 = np.array([[f2_fun(xval)]])  # Second derivative
       
       # Each submodel gets: [all function values, 1st derivs, 2nd derivs]
       submodel_data.append([y_function_values, d1, d2])

   print("Function values shape:", y_function_values.shape)
   print("Number of submodels:", len(submodel_data))
   print("\nSubmodel 0 data structure:")
   print(f"  Element 0 (function values): shape {submodel_data[0][0].shape}")
   print(f"  Element 1 (1st derivatives): shape {submodel_data[0][1].shape}")
   print(f"  Element 2 (2nd derivatives): shape {submodel_data[0][2].shape}")

**Explanation:**  
This step computes the analytic first and second derivatives using SymPy. The full set of function values is shared across all submodels, while each submodel includes only the derivatives evaluated at its corresponding training point.

---

Step 7: Build weighted derivative-enhanced GP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   normalize = True
   kernel = "SE"
   kernel_type = "anisotropic"

   gp_model = wdegp(
       X_train,
       submodel_data,
       n_order,
       n_bases,
       derivative_specs,
       derivative_locations = submodel_indices,
       normalize=normalize,
       kernel=kernel,
       kernel_type=kernel_type
   )

**Explanation:**  
The **WDEGP model** is constructed with:

- **Training locations** (`X_train`): The spatial coordinates where function and derivative data are available
- **Submodel data** (`submodel_data`): The function values and derivatives for each submodel
- **Submodel indices** (`submodel_indices`): Specifies which training points have which derivative types for each submodel
- **Derivative specifications** (`derivative_specs`): Defines which derivative types are included in each submodel
- **Kernel configuration**: Squared exponential (SE) kernel with anisotropic length scales

---

Step 8: Optimize hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   params = gp_model.optimize_hyperparameters(
        optimizer='pso',
        pop_size = 100,
        n_generations = 15,
        local_opt_every = None,
        debug = False
        )
   print("Optimized hyperparameters:", params)

**Explanation:**  
The kernel hyperparameters are tuned by maximizing the **log marginal likelihood**.

---

Step 9: Evaluate model performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   X_test = np.linspace(lb_x, ub_x, 250).reshape(-1, 1)
   y_pred, y_cov, submodel_vals, submodel_cov = gp_model.predict(
       X_test, params, calc_cov=True, return_submodels=True
   )
   y_true = f_fun(X_test.flatten())
   nrmse = utils.nrmse(y_true, y_pred)
   print(f"NRMSE: {nrmse:.6f}")

---

Step 10: Verify interpolation using direct derivative predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   # Since all submodels share the same derivative_specs, we can use return_deriv=True
   # Output shape: (num_deriv_types + 1, num_points)
   y_pred_train = gp_model.predict(X_train, params, calc_cov=False, return_deriv=True)
   
   print(f"Prediction shape with return_deriv=True: {y_pred_train.shape}")
   print("  Row 0: function values")
   print("  Row 1: first derivatives")
   print("  Row 2: second derivatives")
   
   # Extract predictions
   pred_func = y_pred_train[0, :]  # Function values
   pred_d1 = y_pred_train[1, :]    # First derivatives
   pred_d2 = y_pred_train[2, :]    # Second derivatives
   
   # Compute analytic values
   analytic_func = y_function_values.flatten()
   analytic_d1 = np.array([f1_fun(X_train[i, 0]) for i in range(num_points)])
   analytic_d2 = np.array([f2_fun(X_train[i, 0]) for i in range(num_points)])
   
   print("\n" + "=" * 70)
   print("Interpolation verification using return_deriv=True:")
   print("=" * 70)
   
   print("\nFunction value interpolation:")
   for i in range(num_points):
       error = abs(pred_func[i] - analytic_func[i])
       print(f"  Point {i} (x={X_train[i, 0]:.3f}): Abs Error = {error:.2e}")
   
   print("\nFirst derivative interpolation:")
   for i in range(num_points):
       error = abs(pred_d1[i] - analytic_d1[i])
       rel_error = error / abs(analytic_d1[i]) if analytic_d1[i] != 0 else error
       print(f"  Point {i} (x={X_train[i, 0]:.3f}): Pred={pred_d1[i]:.6f}, Analytic={analytic_d1[i]:.6f}, Rel Error={rel_error:.2e}")
   
   print("\nSecond derivative interpolation:")
   for i in range(num_points):
       error = abs(pred_d2[i] - analytic_d2[i])
       rel_error = error / abs(analytic_d2[i]) if analytic_d2[i] != 0 else error
       print(f"  Point {i} (x={X_train[i, 0]:.3f}): Pred={pred_d2[i]:.6f}, Analytic={analytic_d2[i]:.6f}, Rel Error={rel_error:.2e}")
   
   print("\n" + "=" * 70)
   print("Summary:")
   print(f"  Max function error: {np.max(np.abs(pred_func - analytic_func)):.2e}")
   print(f"  Max 1st deriv error: {np.max(np.abs(pred_d1 - analytic_d1)):.2e}")
   print(f"  Max 2nd deriv error: {np.max(np.abs(pred_d2 - analytic_d2)):.2e}")
   print("=" * 70)

**Explanation:**  
Since all submodels share the same derivative **types** (``[[1,1]]`` and ``[[1,2]]``), we can use ``return_deriv=True`` to directly verify derivative interpolation without finite differences. The output shape is ``(num_deriv_types + 1, num_points)`` where each row contains a different output type.

---

Step 11: Visualize combined prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

Step 12: Analyze individual submodel contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

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

**Explanation:**  
Each submodel focuses on the local region surrounding its training point. The weighted combination yields a globally smooth and accurate approximation.

---

Summary
~~~~~~~
This tutorial demonstrates the **Weighted Derivative-Enhanced Gaussian Process (WDEGP)** for 1D function approximation using **individual derivative-informed submodels**. 

**Key takeaways:**

1. **Flexible indexing**: Submodel indices do not need to be contiguous
2. **Direct derivative predictions**: When all submodels share the same derivative types, use ``return_deriv=True`` for direct derivative predictions without finite differences
3. **Weighted aggregation**: WDEGP combines local submodels to capture both local detail and global smoothness


Example 2: 1D Sparse Weighted DEGP with Selective Derivative Observations
--------------------------------------------------------------------------

Overview
~~~~~~~~
This example demonstrates a **sparse derivative-enhanced Gaussian Process (WDEGP)** where derivatives are only included at a **subset of training points**. This approach is useful when derivative information is expensive to obtain or only available at select locations.

The sparse formulation constructs a **single submodel** that incorporates:

- Function values at all training points
- Derivatives at selected points only (non-contiguous indices are allowed)

Since this is a single submodel, ``return_deriv=True`` is available for direct derivative predictions at the training locations where derivatives were provided.

---

Step 1: Import required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt
    from jetgp.wdegp.wdegp import wdegp
    import jetgp.utils as utils

---

Step 2: Set experiment parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    n_order = 2
    n_bases = 1
    lb_x = 0.5
    ub_x = 2.5
    num_points = 10

---

Step 3: Define the symbolic function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    x = sp.symbols('x')
    f_sym = sp.sin(10 * sp.pi * x) / (2 * x) + (x - 1)**4

    f1_sym = sp.diff(f_sym, x)
    f2_sym = sp.diff(f_sym, x, 2)

    f_fun = sp.lambdify(x, f_sym, "numpy")
    f1_fun = sp.lambdify(x, f1_sym, "numpy")
    f2_fun = sp.lambdify(x, f2_sym, "numpy")

---

Step 4: Generate training points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)
    print("Training points:", X_train.ravel())

---

Step 5: Define sparse derivative structure with non-contiguous indices
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Sparse derivative selection: only include derivatives at these points
    # Note: Indices do NOT need to be contiguous!
    derivative_indices = [2, 3, 4, 5]
    
    # Single submodel with derivatives at the selected points
    # Structure: [[indices for 1st order, indices for 2nd order]]
    submodel_indices = [[derivative_indices, derivative_indices]]
    
    # Derivative specs: 1st and 2nd order for this submodel
    derivative_specs = [utils.gen_OTI_indices(n_bases, n_order)]
    
    print(f"Number of submodels: {len(submodel_indices)}")
    print(f"Derivative observation points: {derivative_indices}")
    print(f"submodel_indices structure: {submodel_indices}")
    print(f"derivative_specs: {derivative_specs}")

**Explanation:**  
WDEGP allows **any valid training point indices** without requiring them to be contiguous. Here we select points [2, 3, 4, 5] directly.

The ``submodel_indices`` structure ``[[derivative_indices, derivative_indices]]`` indicates that this single submodel uses both 1st-order and 2nd-order derivatives at the same set of points.

---

Step 6: Compute function values and sparse analytic derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Function values at all training points
    y_function_values = f_fun(X_train.flatten()).reshape(-1, 1)
    
    # Derivatives only at selected points
    d1_sparse = np.array([[f1_fun(X_train[idx, 0])] for idx in derivative_indices])
    d2_sparse = np.array([[f2_fun(X_train[idx, 0])] for idx in derivative_indices])
    
    # Submodel data: [function values at ALL points, derivs at selected points]
    submodel_data = [[y_function_values, d1_sparse, d2_sparse]]
    
    print(f"Function values shape: {y_function_values.shape} (all {num_points} points)")
    print(f"1st derivatives shape: {d1_sparse.shape} (only {len(derivative_indices)} points)")
    print(f"2nd derivatives shape: {d2_sparse.shape} (only {len(derivative_indices)} points)")

**Explanation:**  
Function values are computed at all training points, while derivatives are only computed at the selected indices. The derivative arrays have length equal to the number of selected points.

---

Step 7: Build weighted derivative-enhanced GP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    kernel = "SE"
    kernel_type = "anisotropic"
    normalize = True

    gp_model = wdegp(
        X_train,
        submodel_data,
        n_order,
        n_bases,
        derivative_specs,
        derivative_locations=submodel_indices,
        normalize=normalize,
        kernel=kernel,
        kernel_type=kernel_type
    )

---

Step 8: Optimize hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    params = gp_model.optimize_hyperparameters(
        optimizer='pso',
        pop_size = 100,
        n_generations = 15,
        local_opt_every = 15,
        debug = False
        )
    print("\nOptimized hyperparameters:", params)

---

Step 9: Evaluate model performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    X_test = np.linspace(lb_x, ub_x, 250).reshape(-1, 1)
    y_pred, y_cov = gp_model.predict(X_test, params, calc_cov=True)
    y_true = f_fun(X_test.flatten())
    nrmse = np.sqrt(np.mean((y_true - y_pred.flatten())**2)) / (y_true.max() - y_true.min())
    print(f"\nNRMSE: {nrmse:.6f}")

---

Step 10: Verify interpolation using direct derivative predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   # Verify function value interpolation at all training points
   y_pred_func = gp_model.predict(X_train, params, calc_cov=False)
   
   print("Function value interpolation errors:")
   print("=" * 70)
   for i in range(num_points):
       error_abs = abs(y_pred_func[0, i] - y_function_values[i, 0])
       status = "WITH derivs" if i in derivative_indices else "no derivs"
       print(f"Point {i} (x={X_train[i, 0]:.3f}, {status}): Abs Error = {error_abs:.2e}")
   
   # Verify derivative interpolation at sparse points using return_deriv=True
   # We query only the points where derivatives were provided
   X_deriv_points = X_train[derivative_indices]
   y_pred_with_derivs = gp_model.predict(X_deriv_points, params, calc_cov=False, return_deriv=True)
   
   # Output structure: (num_deriv_types + 1, num_points)
   # Row 0: function values, Row 1: 1st derivatives, Row 2: 2nd derivatives
   print(f"\nPrediction shape with return_deriv=True: {y_pred_with_derivs.shape}")
   print("  Row 0: function values")
   print("  Row 1: first derivatives")
   print("  Row 2: second derivatives")
   
   print("\n" + "=" * 70)
   print(f"Derivative verification at sparse points {derivative_indices} using return_deriv=True:")
   print("=" * 70)
   
   # Extract predictions - each row is a different output type
   pred_func = y_pred_with_derivs[0, :]   # Function values
   pred_d1 = y_pred_with_derivs[1, :]     # First derivatives
   pred_d2 = y_pred_with_derivs[2, :]     # Second derivatives
   
   print("\nFirst derivative interpolation (direct from GP):")
   for local_idx, global_idx in enumerate(derivative_indices):
       analytic = d1_sparse[local_idx, 0]
       predicted = pred_d1[local_idx]
       error = abs(predicted - analytic)
       rel_error = error / abs(analytic) if analytic != 0 else error
       print(f"  Point {global_idx} (x={X_train[global_idx, 0]:.3f}): Pred={predicted:.6f}, Analytic={analytic:.6f}, Rel Error={rel_error:.2e}")
   
   print("\nSecond derivative interpolation (direct from GP):")
   for local_idx, global_idx in enumerate(derivative_indices):
       analytic = d2_sparse[local_idx, 0]
       predicted = pred_d2[local_idx]
       error = abs(predicted - analytic)
       rel_error = error / abs(analytic) if analytic != 0 else error
       print(f"  Point {global_idx} (x={X_train[global_idx, 0]:.3f}): Pred={predicted:.6f}, Analytic={analytic:.6f}, Rel Error={rel_error:.2e}")
   
   print("\n" + "=" * 70)
   print("Note: Derivative predictions are only available at points where derivatives")
   print("      were used in training. For other points, use finite differences.")
   print("=" * 70)

**Explanation:**  
Since this is a single submodel, we can use ``return_deriv=True`` to directly obtain derivative predictions. The derivatives are predicted at the same points where they were provided during training.

For points without derivative training data, derivative predictions would need to use finite differences on the function predictions.

---

Step 11: Visualize combined prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_true, 'b-', label='True Function', linewidth=2)
    plt.plot(X_test.flatten(), y_pred.flatten(), 'r--', label='GP Prediction', linewidth=2)
    plt.fill_between(X_test.ravel(),
                     (y_pred.flatten() - 2*np.sqrt(y_cov)).ravel(),
                     (y_pred.flatten() + 2*np.sqrt(y_cov)).ravel(),
                     color='red', alpha=0.3, label='95% Confidence')
    plt.scatter(X_train, y_function_values, color='black', label='Training Points')
    plt.scatter(X_train[derivative_indices], y_function_values[derivative_indices], 
                color='orange', s=100, marker='s', label='Derivative Points', zorder=5)
    plt.title("Sparse WDEGP with Non-Contiguous Derivative Indices")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

**Explanation:**  
Orange squares highlight the training points where derivative information was included. The model effectively interpolates despite using derivatives at only a subset of points.

---

Summary
~~~~~~~
This tutorial demonstrates **sparse derivative observations** with **non-contiguous indices** in the WDEGP framework:

**Key takeaways:**

1. **Non-contiguous indices**: Use indices like ``[2, 3, 4, 5]`` directly without reordering data
2. **Direct derivative predictions**: Use ``return_deriv=True`` for direct verification at training points
3. **Sparse efficiency**: Reduce computational cost by including derivatives only where needed


Example 3: 1D Weighted DEGP with Multiple Submodels
----------------------------------------------------

Overview
~~~~~~~~
This example demonstrates how to construct **multiple submodels** in WDEGP, each with its own set of derivative observations at different training points. We create two submodels:

- **Submodel 0**: Uses derivatives at training points [0, 2, 4, 6, 8]
- **Submodel 1**: Uses derivatives at training points [1, 3, 5, 7, 9]

Both submodels have access to function values at **ALL** training points, but each uses derivatives only at its designated subset.

Since both submodels use the **same derivative types** (1st and 2nd order), ``return_deriv=True`` is available for the weighted model.

---

Step 1: Import required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt
    from jetgp.wdegp.wdegp import wdegp
    import jetgp.utils as utils

---

Step 2: Set experiment parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    n_order = 2
    n_bases = 1
    lb_x = 0.5
    ub_x = 2.5
    num_points = 10
    
    np.random.seed(42)

---

Step 3: Define the symbolic function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    x = sp.symbols('x')
    f_sym = sp.sin(10 * sp.pi * x) / (2 * x) + (x - 1)**4

    f1_sym = sp.diff(f_sym, x)
    f2_sym = sp.diff(f_sym, x, 2)

    f_fun = sp.lambdify(x, f_sym, "numpy")
    f1_fun = sp.lambdify(x, f1_sym, "numpy")
    f2_fun = sp.lambdify(x, f2_sym, "numpy")

---

Step 4: Generate training points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)
    print("Training points:", X_train.ravel())

---

Step 5: Define multi-submodel structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Submodel 0: derivatives at even indices [0, 2, 4, 6, 8]
    # Submodel 1: derivatives at odd indices [1, 3, 5, 7, 9]
    submodel0_indices = [0, 2, 4, 6, 8]
    submodel1_indices = [1, 3, 5, 7, 9]
    
    # Structure: submodel_indices[submodel_idx][deriv_type_idx] = list of point indices
    # Both derivative types (1st and 2nd order) use the same indices within each submodel
    submodel_indices = [
        [submodel0_indices, submodel0_indices],  # Submodel 0: both deriv types at [0,2,4,6,8]
        [submodel1_indices, submodel1_indices]   # Submodel 1: both deriv types at [1,3,5,7,9]
    ]
    
    # Both submodels use the same derivative specifications
    # This enables return_deriv=True for the weighted model!
    base_deriv_specs = utils.gen_OTI_indices(n_bases, n_order)
    derivative_specs = [base_deriv_specs, base_deriv_specs]
    
    print(f"Number of submodels: {len(submodel_indices)}")
    print(f"Submodel 0 derivative indices: {submodel0_indices}")
    print(f"Submodel 1 derivative indices: {submodel1_indices}")
    print(f"\nsubmodel_indices structure:")
    print(f"  Submodel 0: {submodel_indices[0]}")
    print(f"  Submodel 1: {submodel_indices[1]}")
    print(f"\nderivative_specs (same for both): {derivative_specs[0]}")
    print(f"\nSince both submodels share derivative_specs, return_deriv=True is available!")

**Explanation:**  
We partition the training points into two groups: even indices for Submodel 0, odd indices for Submodel 1. Both submodels use the same derivative types (1st and 2nd order), which enables direct derivative predictions from the weighted model.

---

Step 6: Compute function values and derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Function values at ALL training points (shared by both submodels)
    y_function_values = f_fun(X_train.flatten()).reshape(-1, 1)
    
    # Submodel 0: derivatives at even indices
    d1_submodel0 = np.array([[f1_fun(X_train[idx, 0])] for idx in submodel0_indices])
    d2_submodel0 = np.array([[f2_fun(X_train[idx, 0])] for idx in submodel0_indices])
    
    # Submodel 1: derivatives at odd indices
    d1_submodel1 = np.array([[f1_fun(X_train[idx, 0])] for idx in submodel1_indices])
    d2_submodel1 = np.array([[f2_fun(X_train[idx, 0])] for idx in submodel1_indices])
    
    # Package data for each submodel
    # Each gets: [function values at ALL points, derivs at its points]
    submodel_data = [
        [y_function_values, d1_submodel0, d2_submodel0],  # Submodel 0
        [y_function_values, d1_submodel1, d2_submodel1]   # Submodel 1
    ]
    
    print("Submodel data structure:")
    print(f"  Submodel 0: func vals ({y_function_values.shape}), d1 ({d1_submodel0.shape}), d2 ({d2_submodel0.shape})")
    print(f"  Submodel 1: func vals ({y_function_values.shape}), d1 ({d1_submodel1.shape}), d2 ({d2_submodel1.shape})")

---

Step 7: Build weighted derivative-enhanced GP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    kernel = "SE"
    kernel_type = "anisotropic"
    normalize = True

    gp_model = wdegp(
        X_train,
        submodel_data,
        n_order,
        n_bases,
        derivative_specs,
        derivative_locations = submodel_indices,
        normalize=normalize,
        kernel=kernel,
        kernel_type=kernel_type
    )

---

Step 8: Optimize hyperparameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    params = gp_model.optimize_hyperparameters(
        optimizer='jade',
        pop_size = 100,
        n_generations = 15,
        local_opt_every = None,
        debug = False
        )
    print("Optimized hyperparameters:", params)

---

Step 9: Evaluate model performance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    X_test = np.linspace(lb_x, ub_x, 250).reshape(-1, 1)
    y_pred, y_cov, submodel_vals, submodel_cov = gp_model.predict(
        X_test, params, calc_cov=True, return_submodels=True
    )
    y_true = f_fun(X_test.flatten())
    nrmse = np.sqrt(np.mean((y_true - y_pred.flatten())**2)) / (y_true.max() - y_true.min())
    print(f"NRMSE: {nrmse:.6f}")

---

Step 10: Verify interpolation using direct derivative predictions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   # Since both submodels share the same derivative_specs, we can use return_deriv=True
   y_pred_train = gp_model.predict(X_train, params, calc_cov=False, return_deriv=True)
   
   # Extract predictions - shape is (num_deriv_types + 1, num_points)
   pred_func = y_pred_train[0, :]  # Function values
   pred_d1 = y_pred_train[1, :]    # First derivatives
   pred_d2 = y_pred_train[2, :]    # Second derivatives
   
   # Compute analytic values
   analytic_func = y_function_values.flatten()
   analytic_d1 = np.array([f1_fun(X_train[i, 0]) for i in range(num_points)])
   analytic_d2 = np.array([f2_fun(X_train[i, 0]) for i in range(num_points)])
   
   print("=" * 70)
   print("Interpolation verification using return_deriv=True:")
   print("=" * 70)
   
   print("\nFunction value interpolation (all points):")
   for i in range(num_points):
       error = abs(pred_func[i] - analytic_func[i])
       submodel = "SM0" if i in submodel0_indices else "SM1"
       print(f"  Point {i} ({submodel}, x={X_train[i, 0]:.3f}): Error = {error:.2e}")
   
   print("\nFirst derivative interpolation:")
   print("  Submodel 0 points (even indices):")
   for idx in submodel0_indices:
       rel_error = abs(pred_d1[idx] - analytic_d1[idx]) / abs(analytic_d1[idx]) if analytic_d1[idx] != 0 else abs(pred_d1[idx] - analytic_d1[idx])
       print(f"    Point {idx}: Pred={pred_d1[idx]:.6f}, Analytic={analytic_d1[idx]:.6f}, Rel Error={rel_error:.2e}")
   print("  Submodel 1 points (odd indices):")
   for idx in submodel1_indices:
       rel_error = abs(pred_d1[idx] - analytic_d1[idx]) / abs(analytic_d1[idx]) if analytic_d1[idx] != 0 else abs(pred_d1[idx] - analytic_d1[idx])
       print(f"    Point {idx}: Pred={pred_d1[idx]:.6f}, Analytic={analytic_d1[idx]:.6f}, Rel Error={rel_error:.2e}")
   
   print("\nSecond derivative interpolation:")
   print("  Submodel 0 points (even indices):")
   for idx in submodel0_indices:
       rel_error = abs(pred_d2[idx] - analytic_d2[idx]) / abs(analytic_d2[idx]) if analytic_d2[idx] != 0 else abs(pred_d2[idx] - analytic_d2[idx])
       print(f"    Point {idx}: Pred={pred_d2[idx]:.6f}, Analytic={analytic_d2[idx]:.6f}, Rel Error={rel_error:.2e}")
   print("  Submodel 1 points (odd indices):")
   for idx in submodel1_indices:
       rel_error = abs(pred_d2[idx] - analytic_d2[idx]) / abs(analytic_d2[idx]) if analytic_d2[idx] != 0 else abs(pred_d2[idx] - analytic_d2[idx])
       print(f"    Point {idx}: Pred={pred_d2[idx]:.6f}, Analytic={analytic_d2[idx]:.6f}, Rel Error={rel_error:.2e}")
   
   print("\n" + "=" * 70)
   print("Summary:")
   print(f"  Max function error: {np.max(np.abs(pred_func - analytic_func)):.2e}")
   print(f"  Max 1st deriv error: {np.max(np.abs(pred_d1 - analytic_d1)):.2e}")
   print(f"  Max 2nd deriv error: {np.max(np.abs(pred_d2 - analytic_d2)):.2e}")
   print("=" * 70)

**Explanation:**  
Since both submodels share the same derivative types (``[[1,1]]`` and ``[[1,2]]``), we can use ``return_deriv=True`` to directly verify derivative interpolation. The weighted model correctly interpolates derivatives at all training points, even though each submodel only has derivative information at half the points.

---

Step 11: Visualize combined prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_true, 'b-', label='True Function', linewidth=2)
    plt.plot(X_test.flatten(), y_pred.flatten(), 'r--', label='GP Prediction', linewidth=2)
    plt.fill_between(X_test.ravel(),
                     (y_pred.flatten() - 2*np.sqrt(y_cov)).ravel(),
                     (y_pred.flatten() + 2*np.sqrt(y_cov)).ravel(),
                     color='red', alpha=0.3, label='95% Confidence')
    
    # Show submodel points with different colors
    plt.scatter(X_train[submodel0_indices], y_function_values[submodel0_indices], 
                color='green', s=100, marker='o', label='Submodel 0 points', zorder=5)
    plt.scatter(X_train[submodel1_indices], y_function_values[submodel1_indices],
                color='purple', s=100, marker='s', label='Submodel 1 points', zorder=5)
    
    plt.title("Weighted DEGP with Two Submodels")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

---

Step 12: Visualize individual submodel contributions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_true, 'b-', label='True Function', linewidth=2, alpha=0.3)
    plt.plot(X_test.flatten(), submodel_vals[0].flatten(), 'g-', 
             label='Submodel 0 Prediction', linewidth=2, alpha=0.7)
    plt.plot(X_test.flatten(), submodel_vals[1].flatten(), 'purple', linestyle='--',
             label='Submodel 1 Prediction', linewidth=2, alpha=0.7)
    plt.scatter(X_train[submodel0_indices], y_function_values[submodel0_indices], 
                color='green', s=100, marker='o', zorder=5)
    plt.scatter(X_train[submodel1_indices], y_function_values[submodel1_indices],
                color='purple', s=100, marker='s', zorder=5)
    plt.title("Individual Submodel Predictions")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

**Explanation:**  
Each submodel makes predictions across the entire domain. The weighted combination emphasizes each submodel's contribution near its training locations, producing a smooth global prediction.

---

Summary
~~~~~~~
This tutorial demonstrates **multiple submodels** with **non-contiguous indices**:

**Key takeaways:**

1. **Non-contiguous indices**: Use ``[0, 2, 4, 6, 8]`` and ``[1, 3, 5, 7, 9]`` directly
2. **Shared derivative specs**: When all submodels use the same derivative types, ``return_deriv=True`` provides direct derivative predictions
3. **Flexible partitioning**: Divide training points between submodels in any way that suits your problem
4. **No data reordering required**: The framework handles arbitrary index assignments


Example 4: Heterogeneous Derivative Indices Within Submodels
-------------------------------------------------------------

Overview
~~~~~~~~
This example demonstrates that within a single submodel, **different derivative types can have different indices**. This is useful when:

- Higher-order derivatives are only available at certain locations
- Computational resources should be allocated strategically
- Different data sources provide different derivative information

Since both submodels still use the **same derivative types** (1st and 2nd order), ``return_deriv=True`` remains available.

---

Step 1: Import required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt
    from jetgp.wdegp.wdegp import wdegp
    import jetgp.utils as utils

---

Step 2: Set experiment parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    n_order = 2
    n_bases = 1
    lb_x = 0.5
    ub_x = 2.5
    num_points = 10
    
    np.random.seed(42)

---

Step 3: Define the symbolic function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    x = sp.symbols('x')
    f_sym = sp.sin(10 * sp.pi * x) / (2 * x) + (x - 1)**4

    f1_sym = sp.diff(f_sym, x)
    f2_sym = sp.diff(f_sym, x, 2)

    f_fun = sp.lambdify(x, f_sym, "numpy")
    f1_fun = sp.lambdify(x, f1_sym, "numpy")
    f2_fun = sp.lambdify(x, f2_sym, "numpy")

---

Step 4: Generate training points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    X_train = np.linspace(lb_x, ub_x, num_points).reshape(-1, 1)
    print("Training points:", X_train.ravel())

---

Step 5: Define heterogeneous submodel structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Submodel 0: 1st order derivs at [0,2,4,6,8], 2nd order derivs at [4,5,6]
    # Submodel 1: 1st order derivs at [1,3,5,7,9], 2nd order derivs at [3,4,5]
    
    # Note: Different indices for different derivative types within each submodel!
    submodel_indices = [
        [[0, 2, 4, 6, 8], [2, 4, 6]],  # Submodel 0: 1st order at 5 pts, 2nd order at 3 pts
        [[1, 3, 5, 7, 9], [3, 7, 9]]   # Submodel 1: 1st order at 5 pts, 2nd order at 3 pts
        ]
    
    # Both submodels still use the same derivative TYPES (1st and 2nd order)
    # So return_deriv=True IS available in this case
    derivative_specs = [
        [[[[1, 1]]], [[[1, 2]]]],  # Submodel 0: 1st order, 2nd order
        [[[[1, 1]]], [[[1, 2]]]]   # Submodel 1: 1st order, 2nd order
    ]
    
    print("Submodel structure:")
    print(f"  Submodel 0:")
    print(f"    1st order derivs {derivative_specs[0][0]} at points {submodel_indices[0][0]}")
    print(f"    2nd order derivs {derivative_specs[0][1]} at points {submodel_indices[0][1]}")
    print(f"  Submodel 1:")
    print(f"    1st order derivs {derivative_specs[1][0]} at points {submodel_indices[1][0]}")
    print(f"    2nd order derivs {derivative_specs[1][1]} at points {submodel_indices[1][1]}")
    print(f"\nBoth submodels use same derivative TYPES - return_deriv=True is available!")

**Explanation:**  
This example shows that within a single submodel, **different derivative types can have different indices**:

- Submodel 0 has 1st-order derivatives at 5 points but 2nd-order derivatives at only 3 points
- This allows fine-grained control over where expensive higher-order derivatives are computed

Since both submodels use the same derivative **types** (even though at different indices), ``return_deriv=True`` is still available.

---

Step 6: Compute function values and derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Function values at ALL training points
    y_function_values = f_fun(X_train.flatten()).reshape(-1, 1)
    
    # Submodel 0 derivatives
    d1_sm0_indices = submodel_indices[0][0]
    d2_sm0_indices = submodel_indices[0][1]
    d1_submodel0 = np.array([[f1_fun(X_train[idx, 0])] for idx in d1_sm0_indices])
    d2_submodel0 = np.array([[f2_fun(X_train[idx, 0])] for idx in d2_sm0_indices])
    
    # Submodel 1 derivatives
    d1_sm1_indices = submodel_indices[1][0]
    d2_sm1_indices = submodel_indices[1][1]
    d1_submodel1 = np.array([[f1_fun(X_train[idx, 0])] for idx in d1_sm1_indices])
    d2_submodel1 = np.array([[f2_fun(X_train[idx, 0])] for idx in d2_sm1_indices])
    
    # Package data
    submodel_data = [
        [y_function_values, d1_submodel0, d2_submodel0],
        [y_function_values, d1_submodel1, d2_submodel1]
    ]
    
    print("Submodel data structure:")
    print(f"  Submodel 0: func ({y_function_values.shape}), d1 ({d1_submodel0.shape}), d2 ({d2_submodel0.shape})")
    print(f"  Submodel 1: func ({y_function_values.shape}), d1 ({d1_submodel1.shape}), d2 ({d2_submodel1.shape})")

---

Step 7: Build and optimize GP
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    gp_model = wdegp(
        X_train,
        submodel_data,
        n_order,
        n_bases,
        derivative_specs,
        derivative_locations =submodel_indices,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic"
    )
    
    params = gp_model.optimize_hyperparameters(
        optimizer='jade',
        pop_size=100,
        n_generations=15,
        debug=False
    )
    print("Optimized hyperparameters:", params)

---

Step 8: Evaluate and verify
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    X_test = np.linspace(lb_x, ub_x, 250).reshape(-1, 1)
    y_pred, y_cov = gp_model.predict(X_test, params, calc_cov=True)
    y_true = f_fun(X_test.flatten())
    nrmse = np.sqrt(np.mean((y_true - y_pred.flatten())**2)) / (y_true.max() - y_true.min())
    print(f"NRMSE: {nrmse:.6f}")
    
    # Verify using return_deriv=True (available because derivative_specs match)
    y_pred_train = gp_model.predict(X_train, params, calc_cov=False, return_deriv=True)
    
    # Extract predictions - shape is (num_deriv_types + 1, num_points)
    pred_func = y_pred_train[0, :]  # Function values
    pred_d1 = y_pred_train[1, :]    # First derivatives
    pred_d2 = y_pred_train[2, :]    # Second derivatives
    
    # Compute analytic values
    analytic_func = y_function_values.flatten()
    analytic_d1 = np.array([f1_fun(X_train[i, 0]) for i in range(num_points)])
    analytic_d2 = np.array([f2_fun(X_train[i, 0]) for i in range(num_points)])
    
    print("\n" + "=" * 70)
    print("Verification using return_deriv=True:")
    print("=" * 70)
    print(f"Max function error: {np.max(np.abs(pred_func - analytic_func)):.2e}")
    print(f"Max 1st deriv error: {np.max(np.abs(pred_d1 - analytic_d1)):.2e}")
    print(f"Max 2nd deriv error: {np.max(np.abs(pred_d2 - analytic_d2)):.2e}")
    
    print("\nSubmodel 0 derivative verification:")
    print(f"  1st order at {d1_sm0_indices}:")
    for idx in d1_sm0_indices:
        rel_err = abs(pred_d1[idx] - analytic_d1[idx]) / abs(analytic_d1[idx]) if analytic_d1[idx] != 0 else abs(pred_d1[idx] - analytic_d1[idx])
        print(f"    Point {idx}: Rel Error = {rel_err:.2e}")
    print(f"  2nd order at {d2_sm0_indices}:")
    for idx in d2_sm0_indices:
        rel_err = abs(pred_d2[idx] - analytic_d2[idx]) / abs(analytic_d2[idx]) if analytic_d2[idx] != 0 else abs(pred_d2[idx] - analytic_d2[idx])
        print(f"    Point {idx}: Rel Error = {rel_err:.2e}")
    
    print("\nSubmodel 1 derivative verification:")
    print(f"  1st order at {d1_sm1_indices}:")
    for idx in d1_sm1_indices:
        rel_err = abs(pred_d1[idx] - analytic_d1[idx]) / abs(analytic_d1[idx]) if analytic_d1[idx] != 0 else abs(pred_d1[idx] - analytic_d1[idx])
        print(f"    Point {idx}: Rel Error = {rel_err:.2e}")
    print(f"  2nd order at {d2_sm1_indices}:")
    for idx in d2_sm1_indices:
        rel_err = abs(pred_d2[idx] - analytic_d2[idx]) / abs(analytic_d2[idx]) if analytic_d2[idx] != 0 else abs(pred_d2[idx] - analytic_d2[idx])
        print(f"    Point {idx}: Rel Error = {rel_err:.2e}")

**Explanation:**
Since both submodels share the same derivative types (``[[1,1]]`` and ``[[1,2]]``), we can use ``return_deriv=True`` for direct verification. The weighted model correctly interpolates all derivatives at all training points.

---

Step 9: Visualize
~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_true, 'b-', label='True Function', linewidth=2)
    plt.plot(X_test.flatten(), y_pred.flatten(), 'r--', label='GP Prediction', linewidth=2)
    plt.fill_between(X_test.ravel(),
                     (y_pred.flatten() - 2*np.sqrt(y_cov)).ravel(),
                     (y_pred.flatten() + 2*np.sqrt(y_cov)).ravel(),
                     color='red', alpha=0.3, label='95% Confidence')
    plt.scatter(X_train, y_function_values, color='black', s=80, label='All training points')
    plt.title("WDEGP with Heterogeneous Derivative Indices")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

---

Summary
~~~~~~~
This tutorial demonstrates **heterogeneous derivative indices** within submodels:

**Key takeaways:**

1. **Different indices per derivative type**: Within a submodel, 1st-order and 2nd-order derivatives can be at different points
2. **Strategic allocation**: Concentrate expensive higher-order derivatives where most needed
3. **return_deriv availability**: Depends on whether derivative **types** (not indices) are shared across submodels
4. **Flexible design**: Mix and match derivative orders and locations as needed


Example 5: When ``return_deriv=True`` is NOT Available
------------------------------------------------------

Overview
~~~~~~~~
This example demonstrates a case where ``return_deriv=True`` is **NOT available** because submodels have **different derivative specifications**. In this case, derivative verification requires finite differences or accessing individual submodels.

---

Step 1-4: Setup (same as previous examples)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    import numpy as np
    import sympy as sp
    import matplotlib.pyplot as plt
    from jetgp.wdegp.wdegp import wdegp
    import jetgp.utils as utils
    
    x = sp.symbols('x')
    f_sym = sp.sin(10 * sp.pi * x) / (2 * x) + (x - 1)**4
    f1_sym = sp.diff(f_sym, x)
    f2_sym = sp.diff(f_sym, x, 2)
    
    f_fun = sp.lambdify(x, f_sym, "numpy")
    f1_fun = sp.lambdify(x, f1_sym, "numpy")
    f2_fun = sp.lambdify(x, f2_sym, "numpy")
    
    n_order = 2
    n_bases = 1
    num_points = 10
    X_train = np.linspace(0.5, 2.5, num_points).reshape(-1, 1)
    y_function_values = f_fun(X_train.flatten()).reshape(-1, 1)

---

Step 5: Define submodels with DIFFERENT derivative specs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Submodel 0: ONLY 1st order derivatives
    # Submodel 1: ONLY 2nd order derivatives
    # These are DIFFERENT derivative specs - return_deriv=True NOT available!
    
    submodel_indices = [
        [[0, 2, 4, 6, 8]],      # Submodel 0: only has 1st order
        [[1, 3, 5, 7, 9]]       # Submodel 1: only has 2nd order
    ]
    
    derivative_specs = [
        [[[[1, 1]]]],             # Submodel 0: only 1st order derivatives
        [[[[1, 2]]]]              # Submodel 1: only 2nd order derivatives
    ]
    
    print("Submodel structure with DIFFERENT derivative specs:")
    print(f"  Submodel 0: {derivative_specs[0]} at {submodel_indices[0][0]}")
    print(f"  Submodel 1: {derivative_specs[1]} at {submodel_indices[1][0]}")
    print(f"\nWARNING: No shared derivatives - return_deriv=True NOT available!")
    print("Must use finite differences for derivative verification on weighted model,")
    print("or access individual submodels via return_submodels=True.")

---

Step 6: Build model and prepare data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Compute derivatives for each submodel
    d1_submodel0 = np.array([[f1_fun(X_train[idx, 0])] for idx in submodel_indices[0][0]])
    d2_submodel1 = np.array([[f2_fun(X_train[idx, 0])] for idx in submodel_indices[1][0]])
    
    submodel_data = [
        [y_function_values, d1_submodel0],      # Submodel 0: func + 1st derivs only
        [y_function_values, d2_submodel1]       # Submodel 1: func + 2nd derivs only
    ]
    
    gp_model = wdegp(
        X_train,
        submodel_data,
        n_order,
        n_bases,
        derivative_specs,
        derivative_locations = submodel_indices,
        normalize=True,
        kernel="SE",
        kernel_type="anisotropic"
    )
    
    params = gp_model.optimize_hyperparameters(
        optimizer='jade', pop_size=100, n_generations=15, debug=False
    )

---

Step 7: Verify using finite differences (required when no shared derivs)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    print("=" * 70)
    print("Verification using FINITE DIFFERENCES (return_deriv not available)")
    print("=" * 70)
    
    h = 1e-6
    
    # Verify 1st derivatives on Submodel 0 points using individual submodel
    print("\nSubmodel 0 - 1st derivative verification (via individual submodel):")
    for local_idx, global_idx in enumerate(submodel_indices[0][0]):
        x_pt = X_train[global_idx, 0]
        
        X_plus = np.array([[x_pt + h]])
        X_minus = np.array([[x_pt - h]])
        
        _, sm_plus = gp_model.predict(X_plus, params, return_submodels=True)
        _, sm_minus = gp_model.predict(X_minus, params, return_submodels=True)
        
        # Use Submodel 0 predictions
        fd_d1 = (sm_plus[0][0, 0] - sm_minus[0][0, 0]) / (2 * h)
        analytic_d1 = d1_submodel0[local_idx, 0]
        rel_error = abs(fd_d1 - analytic_d1) / abs(analytic_d1) if analytic_d1 != 0 else abs(fd_d1 - analytic_d1)
        
        print(f"  Point {global_idx}: FD={fd_d1:.6f}, Analytic={analytic_d1:.6f}, Rel Error={rel_error:.2e}")
    
    # Verify 2nd derivatives on Submodel 1 points using individual submodel
    print("\nSubmodel 1 - 2nd derivative verification (via individual submodel):")
    for local_idx, global_idx in enumerate(submodel_indices[1][0]):
        x_pt = X_train[global_idx, 0]
        
        X_plus = np.array([[x_pt + h]])
        X_minus = np.array([[x_pt - h]])
        X_center = np.array([[x_pt]])
        
        _, sm_plus = gp_model.predict(X_plus, params, return_submodels=True)
        _, sm_minus = gp_model.predict(X_minus, params, return_submodels=True)
        _, sm_center = gp_model.predict(X_center, params, return_submodels=True)
        
        # Use Submodel 1 predictions
        fd_d2 = (sm_plus[1][0, 0] - 2*sm_center[1][0, 0] + sm_minus[1][0, 0]) / (h**2)
        analytic_d2 = d2_submodel1[local_idx, 0]
        rel_error = abs(fd_d2 - analytic_d2) / abs(analytic_d2) if analytic_d2 != 0 else abs(fd_d2 - analytic_d2)
        
        print(f"  Point {global_idx}: FD={fd_d2:.6f}, Analytic={analytic_d2:.6f}, Rel Error={rel_error:.2e}")

**Explanation:**  
When submodels have different derivative specifications (Submodel 0 has only 1st order, Submodel 1 has only 2nd order), there are no shared derivatives and ``return_deriv=True`` cannot be used on the weighted model.

To verify derivatives, we must either:

1. Use finite differences on the weighted prediction
2. Access individual submodels via ``return_submodels=True`` and verify each submodel's derivatives separately

---

Step 8: Visualize
~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    X_test = np.linspace(0.5, 2.5, 250).reshape(-1, 1)
    y_pred, y_cov = gp_model.predict(X_test, params, calc_cov=True)
    y_true = f_fun(X_test.flatten())
    
    plt.figure(figsize=(10, 6))
    plt.plot(X_test, y_true, 'b-', label='True Function', linewidth=2)
    plt.plot(X_test.flatten(), y_pred.flatten(), 'r--', label='GP Prediction', linewidth=2)
    plt.fill_between(X_test.ravel(),
                     (y_pred.flatten() - 2*np.sqrt(y_cov)).ravel(),
                     (y_pred.flatten() + 2*np.sqrt(y_cov)).ravel(),
                     color='red', alpha=0.3, label='95% Confidence')
    plt.scatter(X_train[submodel_indices[0][0]], y_function_values[submodel_indices[0][0]], 
                color='green', s=100, marker='o', label='SM0 (1st order only)', zorder=5)
    plt.scatter(X_train[submodel_indices[1][0]], y_function_values[submodel_indices[1][0]],
                color='purple', s=100, marker='s', label='SM1 (2nd order only)', zorder=5)
    plt.title("WDEGP with Different Derivative Specs (return_deriv NOT available)")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

---

Summary
~~~~~~~
This example demonstrates the limitations of ``return_deriv=True``:

**Key takeaways:**

1. **Shared derivative types required**: ``return_deriv=True`` only works when ALL submodels have the same derivative specifications
2. **Finite differences fallback**: When derivatives aren't shared, use finite differences
3. **Individual submodel access**: Use ``return_submodels=True`` to verify each submodel's derivatives separately
4. **Design consideration**: If you need direct derivative predictions, ensure all submodels use the same derivative types
