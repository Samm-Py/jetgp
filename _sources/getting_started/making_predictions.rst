Making Predictions
==================

Overview
--------
After optimizing hyperparameters, JetGP models can make predictions at new test locations.
The ``predict()`` method returns function value predictions and optionally their uncertainties 
(predictive variances/covariances) and derivative predictions.

All GP modules share a common prediction interface, with minor variations for directional 
derivative models and weighted variants.

------------------------------------------------------------

Basic Usage
-----------

After training and optimizing a GP model, predictions are made using the ``predict()`` method:

**Standard workflow:**

::

    # 1. Initialize the model
    gp = degp(
        X_train, y_train, n_order, n_bases=1, 
        der_indices=der_indices,
        derivative_locations=derivative_locations,
        normalize=True, kernel="SE", kernel_type="anisotropic"
    )
    
    # 2. Optimize hyperparameters
    params = gp.optimize_hyperparameters(
        optimizer='jade',
        pop_size=100,
        n_generations=15
    )
    
    # 3. Make predictions
    y_pred, y_var = gp.predict(
        X_test, params, calc_cov=True, return_deriv=False
    )

------------------------------------------------------------

Common Arguments
----------------

**X_test**  
Array of test input locations where predictions are desired.  
Shape: ``(n_test, n_dims)`` where ``n_test`` is the number of test points and 
``n_dims`` is the input dimensionality.

**params**  
Optimized hyperparameters returned by ``optimize_hyperparameters()``.  
These parameters define the trained GP model's covariance structure.

**calc_cov**  
Boolean flag determining whether to compute predictive uncertainty.

- ``True`` → Returns predictive variance at test points
- ``False`` → Returns only mean predictions (faster computation)

**Default:** ``False``

**return_deriv**  
Boolean flag controlling whether derivative predictions are returned alongside function predictions.

- ``True`` → Returns predictions for both function values and their derivatives in a row-wise format
- ``False`` → Returns only function value predictions

**Default:** ``False``

**Note:** Not all model variants support derivative predictions. When ``return_deriv=True``, the 
output is a 2D array where each row corresponds to a different output type (function values, 
then derivatives in order). See model-specific sections below for details.

**derivs_to_predict**  
List specifying which derivatives to predict when ``return_deriv=True``.  
Must be a subset of the derivatives used during training (as defined in ``der_indices``).

- ``None`` → Predicts all derivatives from training (default behavior)
- List of derivative specifications → Predicts only the specified derivatives

**Default:** ``None``

**Format:** Same nested list format as ``der_indices``, but flattened (no grouping by order required):

::

    # Predict only first-order partial derivatives
    derivs_to_predict = [[[1, 1]], [[2, 1]]]
    
    # Predict first-order and one second-order derivative
    derivs_to_predict = [[[1, 1]], [[2, 1]], [[1, 2]]]
    
    # Predict only mixed second-order derivative
    derivs_to_predict = [[[1, 1], [2, 1]]]

**Note:** The order of derivatives in ``derivs_to_predict`` determines the row order in the output.
Derivatives are returned in the same order they appear in this list.

------------------------------------------------------------

Return Values
-------------

The structure of returned values depends on the model type and input arguments.

**Standard Models (DEGP)**

For standard derivative-enhanced models, the return signature is:

::

    y_pred, y_var = gp.predict(X_test, params, calc_cov=True, return_deriv=False)

**Returns:**

- ``y_pred`` : array, shape ``(n_test,)``  
  Predicted function values at test locations
  
- ``y_var`` : array, shape ``(n_test,)`` (if ``calc_cov=True``)  
  Predictive variance at each test location

**If calc_cov=False:**

::

    y_pred = gp.predict(X_test, params, calc_cov=False, return_deriv=False)

**Returns:**

- ``y_pred`` : array, shape ``(n_test,)``  
  Predicted function values at test locations (no uncertainty)

**If return_deriv=True:**

::

    y_pred, y_var = gp.predict(X_test, params, calc_cov=True, return_deriv=True)

**Returns:**

- ``y_pred`` : array, shape ``(num_derivs + 1, n_test)``  
  Row-wise predictions including function values and all derivatives.
  
  **Structure:**
  
  - Row 0: Function value predictions
  - Row 1: First derivative component predictions
  - Row 2: Second derivative component predictions
  - ... and so on for all derivative components
  
  where ``num_derivs`` is the total number of derivative components specified in ``der_indices``.
  
- ``y_var`` : array, shape ``(num_derivs + 1, n_test)`` (if ``calc_cov=True``)  
  Predictive variance corresponding to each row in ``y_pred``, following the same structure.

**Example with return_deriv=True:**

::

    # Model with first-order derivatives in 2D
    # der_indices = [[[[1, 1]], [[2, 1]]]]  # Two first-order derivatives
    # num_derivs = 2
    
    X_test = np.random.rand(10, 2)  # 10 test points
    y_pred, y_var = gp.predict(X_test, params, calc_cov=True, return_deriv=True)
    
    # y_pred.shape = (3, 10)  # (2 + 1) rows, 10 columns
    # y_var.shape = (3, 10)
    
    # Extract predictions by row:
    func_pred = y_pred[0, :]     # Function predictions (row 0)
    deriv1_pred = y_pred[1, :]   # ∂f/∂x₁ predictions (row 1)
    deriv2_pred = y_pred[2, :]   # ∂f/∂x₂ predictions (row 2)
    
    # Similarly for variances:
    func_var = y_var[0, :]
    deriv1_var = y_var[1, :]
    deriv2_var = y_var[2, :]

**With higher-order derivatives:**

::

    # Model with first and second-order derivatives in 2D
    # der_indices = [
    #     [[[1, 1]], [[2, 1]]],                      # 2 first-order
    #     [[[1, 2]], [[1,1],[2,1]], [[2, 2]]]        # 3 second-order
    # ]
    # num_derivs = 5
    
    X_test = np.random.rand(10, 2)
    y_pred, y_var = gp.predict(X_test, params, calc_cov=True, return_deriv=True)
    
    # y_pred.shape = (6, 10)  # (5 + 1) rows, 10 columns
    
    func_pred = y_pred[0, :]       # f(x)
    d1_dx1 = y_pred[1, :]          # ∂f/∂x₁
    d1_dx2 = y_pred[2, :]          # ∂f/∂x₂
    d2_dx1 = y_pred[3, :]          # ∂²f/∂x₁²
    d2_dx1dx2 = y_pred[4, :]       # ∂²f/∂x₁∂x₂
    d2_dx2 = y_pred[5, :]          # ∂²f/∂x₂²

**With derivs_to_predict (selective derivative predictions):**

::

    # Model trained with first and second-order derivatives in 2D
    # der_indices = [
    #     [[[1, 1]], [[2, 1]]],                      # 2 first-order
    #     [[[1, 2]], [[1, 1], [2, 1]], [[2, 2]]]    # 3 second-order
    # ]
    
    # Predict only first-order derivatives
    derivs_to_predict = [[[1, 1]], [[2, 1]]]
    
    y_pred, y_var = gp.predict(
        X_test, params, 
        calc_cov=True, 
        return_deriv=True,
        derivs_to_predict=derivs_to_predict
    )

**Returns:**

- ``y_pred`` : array, shape ``(len(derivs_to_predict) + 1, n_test)``  
  Row 0 contains function predictions, subsequent rows contain only the 
  requested derivatives in the order specified.
  
- ``y_var`` : array, shape ``(len(derivs_to_predict) + 1, n_test)`` (if ``calc_cov=True``)

**Example with selective derivatives:**

::

    X_test = np.random.rand(10, 2)
    
    # Full model has 5 derivatives, but we only want 2
    derivs_to_predict = [[[1, 1]], [[2, 2]]]  # ∂f/∂x₁ and ∂²f/∂x₂²
    
    y_pred, y_var = gp.predict(
        X_test, params, 
        calc_cov=True, 
        return_deriv=True,
        derivs_to_predict=derivs_to_predict
    )
    
    # y_pred.shape = (3, 10)  # (2 + 1) rows, not (5 + 1)
    
    func_pred = y_pred[0, :]     # Function values
    d1_dx1 = y_pred[1, :]        # ∂f/∂x₁ (first in derivs_to_predict)
    d2_dx2 = y_pred[2, :]        # ∂²f/∂x₂² (second in derivs_to_predict)

**Reordering derivatives:**

The output order matches ``derivs_to_predict``, not the original ``der_indices`` order:

::

    # Original training order: ∂f/∂x₁, ∂f/∂x₂, ∂²f/∂x₁², ...
    
    # Request in different order
    derivs_to_predict = [[[2, 1]], [[1, 1]]]  # ∂f/∂x₂ first, then ∂f/∂x₁
    
    y_pred, _ = gp.predict(X_test, params, return_deriv=True, derivs_to_predict=derivs_to_predict)
    
    # y_pred[1, :] is now ∂f/∂x₂ (not ∂f/∂x₁)
    # y_pred[2, :] is now ∂f/∂x₁ (not ∂f/∂x₂)

------------------------------------------------------------

Model-Specific Prediction Interfaces
-------------------------------------

**1. DEGP (Derivative-Enhanced GP)**

Standard interface with full support for function and derivative predictions.

**Example:**

::

    # Function predictions with uncertainty
    y_pred, y_var = gp.predict(
        X_test, params, calc_cov=True, return_deriv=False
    )
    
    # Function and derivative predictions with uncertainty
    y_pred, y_var = gp.predict(
        X_test, params, calc_cov=True, return_deriv=True
    )
    
    # Selective derivative predictions
    derivs_to_predict = [[[1, 1]]]  # Only ∂f/∂x₁
    y_pred, y_var = gp.predict(
        X_test, params, calc_cov=True, return_deriv=True,
        derivs_to_predict=derivs_to_predict
    )

**Returned shapes:**

- ``y_pred``: ``(n_test,)`` if ``return_deriv=False``
- ``y_pred``: ``(num_derivs + 1, n_test)`` if ``return_deriv=True`` and ``derivs_to_predict=None``
- ``y_pred``: ``(len(derivs_to_predict) + 1, n_test)`` if ``return_deriv=True`` with ``derivs_to_predict``
- ``y_var``: matches ``y_pred`` shape when ``calc_cov=True``

----

**2. DDEGP (Directional Derivative-Enhanced GP)**

Uses directional derivatives along **global directions** (same direction at all training points).

**Standard interface** (same as DEGP):

::

    y_pred, y_var = gp.predict(
        X_test, params, calc_cov=True, return_deriv=False
    )

**Returns:**

- ``y_pred`` : array, shape ``(n_test,)``  
  Predicted function values
  
- ``y_var`` : array, shape ``(n_test,)`` (if ``calc_cov=True``)  
  Predictive variance

**With derivative predictions:**

::

    y_pred, y_var = gp.predict(
        X_test, params, calc_cov=True, return_deriv=True
    )

**Returns:**

- ``y_pred`` : array, shape ``(num_derivs + 1, n_test)``  
  Row-wise predictions including function values and directional derivatives.
  Row 0 contains function predictions, subsequent rows contain directional 
  derivative predictions for each direction/order specified in ``der_indices``.
  
- ``y_var`` : array, shape ``(num_derivs + 1, n_test)`` (if ``calc_cov=True``)  
  Predictive variance for each prediction component

When ``return_deriv=True``, predictions include directional derivatives along the 
same global directions used during training.

**With derivs_to_predict:**

::

    # Predict only specific directional derivatives
    derivs_to_predict = [[[1, 1]]]  # Only first direction, first order
    y_pred, y_var = gp.predict(
        X_test, params, calc_cov=True, return_deriv=True,
        derivs_to_predict=derivs_to_predict
    )

----

**3. GDDEGP (Generalized Directional Derivative-Enhanced GP)**

Allows **point-specific directional derivatives** (different directions at each training point).

**Function predictions only:**

::

    y_pred, y_var = gp.predict(
        X_test, params, calc_cov=True, return_deriv=False
    )

**Returns:**

- ``y_pred`` : array, shape ``(n_test,)``  
  Predicted function values
  
- ``y_var`` : array, shape ``(n_test,)`` (if ``calc_cov=True``)  
  Predictive variance

**With derivative predictions (requires rays_predict):**

When ``return_deriv=True``, you must provide ``rays_predict`` specifying the directional 
vectors at each test point:

::

    y_pred, y_var = gp.predict(
        X_test, params,
        rays_predict=rays_predict,
        calc_cov=True,
        return_deriv=True
    )

**Additional argument:**

- ``rays_predict`` : list of arrays  
  Required when ``return_deriv=True``. A list of 2D arrays specifying directional 
  vectors at each test point. Structure: ``rays_predict[direction_idx]`` has shape 
  ``(d, n_test)`` where column ``j`` is the direction vector for test point ``j``.

**Returns:**

- ``y_pred`` : array, shape ``(num_derivs + 1, n_test)``  
  Row-wise predictions including function values and directional derivatives 
  along the directions specified in ``rays_predict``
  
- ``y_var`` : array, shape ``(num_derivs + 1, n_test)`` (if ``calc_cov=True``)  
  Predictive variance for each prediction component

**With derivs_to_predict:**

Selective derivative predictions work with ``rays_predict``. The number of directions 
in ``rays_predict`` should match the number of unique direction indices referenced 
in ``derivs_to_predict``:

::

    # Training used 2 directions with 1st and 2nd order derivatives
    # der_indices = [[[[1,1]], [[2,1]]], [[[1,2]], [[2,2]]]]
    
    # Predict only 1st-order derivatives along both directions
    derivs_to_predict = [[[1, 1]], [[2, 1]]]
    
    rays_predict = [rays_dir1, rays_dir2]  # Both directions needed
    
    y_pred, _ = gp.predict(
        X_test, params,
        rays_predict=rays_predict,
        return_deriv=True,
        derivs_to_predict=derivs_to_predict
    )

**Flexibility of GDDEGP Predictions:**

A key advantage of GDDEGP is that you can predict directional derivatives in **any direction** 
at test points, not just the directions used during training. The only restriction is that the 
**derivative order** must have been included in training.

::

    # Training used gradient-aligned directions
    # But predictions can use ANY direction:
    
    # Predict along x-axis
    rays_x = np.array([[1.0] * n_test, [0.0] * n_test])
    y_pred_x, _ = gp.predict(X_test, params, rays_predict=[rays_x], return_deriv=True)
    
    # Predict along y-axis  
    rays_y = np.array([[0.0] * n_test, [1.0] * n_test])
    y_pred_y, _ = gp.predict(X_test, params, rays_predict=[rays_y], return_deriv=True)
    
    # Predict along 45° diagonal
    rays_diag = np.array([[0.707] * n_test, [0.707] * n_test])
    y_pred_diag, _ = gp.predict(X_test, params, rays_predict=[rays_diag], return_deriv=True)

**Example:**

::

    import numpy as np
    
    # Define test points
    X_test = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])
    n_test = len(X_test)
    
    # Define directional vectors for predictions at each test point
    # rays_predict[direction_idx] has shape (d, n_test)
    rays_dir1 = np.array([
        [1.0, 0.707, 0.0],      # x-components for 3 test points
        [0.0, 0.707, 1.0]       # y-components for 3 test points
    ])  # Shape: (2, 3)
    
    rays_predict = [rays_dir1]  # List with one direction
    
    # Function predictions only (no rays_predict needed)
    y_pred, y_var = gp.predict(
        X_test, params, calc_cov=True, return_deriv=False
    )
    
    # Function + derivative predictions (rays_predict required)
    y_pred, y_var = gp.predict(
        X_test, params,
        rays_predict=rays_predict,
        calc_cov=True,
        return_deriv=True
    )
    
    # Extract components
    func_pred = y_pred[0, :]      # Function values
    deriv1_pred = y_pred[1, :]    # Directional derivative along rays_dir1

**Example with multiple directions:**

::

    # Two directional derivatives at each test point
    rays_dir1 = np.zeros((2, n_test))  # Gradient direction
    rays_dir2 = np.zeros((2, n_test))  # Perpendicular direction
    
    for i in range(n_test):
        grad = 2 * X_test[i]  # For f = x² + y²
        grad_norm = np.linalg.norm(grad)
        rays_dir1[:, i] = grad / grad_norm
        rays_dir2[:, i] = [-rays_dir1[1, i], rays_dir1[0, i]]
    
    rays_predict = [rays_dir1, rays_dir2]
    
    y_pred, y_var = gp.predict(
        X_test, params,
        rays_predict=rays_predict,
        calc_cov=True,
        return_deriv=True
    )
    
    func_pred = y_pred[0, :]      # Function values
    deriv1_pred = y_pred[1, :]    # Derivative along gradient
    deriv2_pred = y_pred[2, :]    # Derivative perpendicular to gradient

----

**4. WDEGP (Weighted Derivative-Enhanced GP)**

Weighted models combine predictions from multiple submodels and provide additional 
diagnostics for each submodel. WDEGP supports three submodel types via the 
``submodel_type`` parameter: ``'degp'``, ``'ddegp'``, or ``'gddegp'``.

**Standard prediction:**

::

    y_pred, y_cov = gp.predict(
        X_test, params, calc_cov=True
    )

**Returns:**

- ``y_pred`` : array, shape ``(n_test,)``  
  Weighted combination of all submodel predictions
  
- ``y_cov`` : array, shape ``(n_test,)`` (if ``calc_cov=True``)  
  Predictive variance from weighted model

**With derivative predictions:**

::

    y_pred, y_cov = gp.predict(
        X_test, params, calc_cov=True, return_deriv=True
    )

**Returns when return_deriv=True:**

- ``y_pred`` : array, shape ``(n_shared_derivs + 1, n_test)``  
  Row-wise predictions: row 0 is function values, subsequent rows are derivatives.
  Only includes derivatives that are shared across all submodels (see restrictions below).
  
- ``y_cov`` : array, shape ``(n_shared_derivs + 1, n_test)`` (if ``calc_cov=True``)  
  Row-wise predictive variances corresponding to ``y_pred``

**With derivs_to_predict:**

For weighted models, ``derivs_to_predict`` must specify derivatives that are shared 
across all submodels (for DEGP/DDEGP) or derivative orders shared across all submodels 
(for GDDEGP):

::

    # WDEGP with two submodels
    # Submodel 1 has: ∂f/∂x₁, ∂f/∂x₂
    # Submodel 2 has: ∂f/∂x₁, ∂f/∂x₂, ∂²f/∂x₁²
    # Shared: ∂f/∂x₁, ∂f/∂x₂
    
    # Valid: subset of shared derivatives
    derivs_to_predict = [[[1, 1]]]  # Only ∂f/∂x₁
    
    y_pred, y_cov = gp.predict(
        X_test, params, calc_cov=True, return_deriv=True,
        derivs_to_predict=derivs_to_predict
    )
    
    # Invalid: includes non-shared derivative
    derivs_to_predict = [[[1, 1]], [[1, 2]]]  # ∂²f/∂x₁² not shared - will raise error

**With submodel outputs:**

::

    y_pred, y_cov, submodel_vals, submodel_cov = gp.predict(
        X_test, params, calc_cov=True, return_submodels=True
    )

**Additional arguments:**

- ``return_submodels`` – Boolean flag to return individual submodel predictions (default: ``False``)

**Additional returns when return_submodels=True:**

- ``submodel_vals`` : list of arrays  
  Predictions from each individual submodel
  
- ``submodel_cov`` : list of arrays (if ``calc_cov=True``)  
  Predictive variances from each individual submodel

**WDEGP with GDDEGP Submodels (rays_predict required for derivatives):**

When using ``submodel_type='gddegp'`` and ``return_deriv=True``, you must provide
``rays_predict`` specifying directional vectors at each test point:

::

    # WDEGP with GDDEGP submodels - derivative predictions
    y_pred, y_cov = gp.predict(
        X_test, params,
        rays_predict=rays_predict,
        calc_cov=True,
        return_deriv=True
    )

The ``rays_predict`` structure is the same as for standalone GDDEGP:
``rays_predict[direction_idx]`` has shape ``(d, n_test)``.

**Example with GDDEGP submodels:**

::

    # Build rays_predict for all test points
    rays_dir1 = np.zeros((2, n_test))
    rays_dir2 = np.zeros((2, n_test))
    
    for i in range(n_test):
        grad = 2 * X_test[i]
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-10:
            rays_dir1[:, i] = grad / grad_norm
            rays_dir2[:, i] = [-rays_dir1[1, i], rays_dir1[0, i]]
        else:
            rays_dir1[:, i] = [1, 0]
            rays_dir2[:, i] = [0, 1]
    
    rays_predict = [rays_dir1, rays_dir2]
    
    # Predict with derivatives
    y_pred, y_cov = gp.predict(
        X_test, params,
        rays_predict=rays_predict,
        calc_cov=True,
        return_deriv=True
    )

**Derivative Prediction Restrictions:**

Derivative predictions in WDEGP have different restrictions depending on the submodel type.

**For DEGP and DDEGP submodels (direction-specific):**

Derivatives can only be predicted if they appear in **all** submodels, because predictions 
are tied to specific coordinate directions (DEGP) or global ray directions (DDEGP).

::

    # Example: Two DEGP submodels
    # Submodel 1 has derivatives: [[[1,1]]]              (∂f/∂x only)
    # Submodel 2 has derivatives: [[[1,1]], [[2,1]]]    (∂f/∂x and ∂f/∂y)
    
    # Global predictions can ONLY be made for [[1,1]] (∂f/∂x)
    # [[2,1]] (∂f/∂y) is not shared, so it's excluded from weighted predictions

**For GDDEGP submodels (order-specific):**

WDEGP with GDDEGP submodels is a special case. Since you can specify **any direction** 
via ``rays_predict``, the restriction is on **derivative order**, not direction.

::

    # Example: Two GDDEGP submodels
    # Submodel 1 trains on: 1st-order directional derivatives only
    # Submodel 2 trains on: 1st-order AND 2nd-order directional derivatives
    
    # Shared order: 1st-order only
    # You can predict 1st-order directional derivatives in ANY direction
    # You CANNOT predict 2nd-order derivatives (not shared across all submodels)

This means WDEGP with GDDEGP submodels offers maximum flexibility for derivative predictions:

- **Direction flexibility:** Predict directional derivatives along any direction you choose
- **Order restriction:** Only derivative orders present in ALL submodels can be predicted

::

    # Training used different gradient-aligned directions in each submodel
    # But at prediction time, you can use completely different directions:
    
    # Predict 1st-order derivatives along coordinate axes
    rays_x = np.array([[1.0] * n_test, [0.0] * n_test])
    rays_y = np.array([[0.0] * n_test, [1.0] * n_test])
    
    y_pred, _ = gp.predict(
        X_test, params,
        rays_predict=[rays_x, rays_y],  # Any directions work!
        return_deriv=True
    )
    
    # This works because 1st-order is shared across submodels
    # The specific directions don't need to match training

**Additional restrictions (all submodel types):**

1. **Training derivatives only:**  
   Predictions can only be made for derivative orders that were explicitly included in training.
   
   ::
   
       # Example: Single submodel with 1st-order derivatives
       # Can predict: f(x) and 1st-order derivatives
       # Cannot predict: 2nd-order derivatives (not used in training)

2. **Point-level availability:**  
   As long as **at least one training point** includes a specific derivative order, that order
   can be predicted anywhere in the input space via the GP.
   
   ::
   
       # Example: 10 training points
       # Only points [0, 2, 5] have 1st-order derivative information
       # GP can still predict 1st-order derivatives at any test point

3. **Alternative for unavailable derivatives:**  
   If a desired derivative order was not used in training, you can approximate it using finite 
   differences on the GP function predictions:
   
   ::
   
       # Approximate 2nd-order derivative using finite differences
       h = 1e-5
       X_test_plus = X_test + h
       X_test_minus = X_test - h
       
       f_plus = gp.predict(X_test_plus, params, calc_cov=False)
       f_center = gp.predict(X_test, params, calc_cov=False)
       f_minus = gp.predict(X_test_minus, params, calc_cov=False)
       
       d2f_approx = (f_plus - 2*f_center + f_minus) / (h**2)

**Summary of WDEGP Derivative Restrictions:**

.. list-table:: WDEGP derivative prediction restrictions by submodel type
   :header-rows: 1
   :widths: 20 40 40

   * - Submodel Type
     - Restriction Type
     - What Can Be Predicted
   * - ``'degp'``
     - Direction-specific
     - Only coordinate derivatives (∂f/∂xᵢ) shared by all submodels
   * - ``'ddegp'``
     - Direction-specific
     - Only global ray directions shared by all submodels
   * - ``'gddegp'``
     - Order-specific
     - Any direction, but only orders shared by all submodels

**Use case for return_submodels:**

Examining individual submodel predictions is useful for:

- **Visualization:** Understanding how different derivative subsets contribute to final predictions
- **Diagnostics:** Identifying if certain submodels dominate or perform poorly
- **Analysis:** Studying sensitivity to different derivative information

**Example:**

::

    # Get weighted prediction and individual submodel contributions
    y_pred, y_var, submodel_preds, submodel_vars = gp.predict(
        X_test, params, calc_cov=True, return_submodels=True
    )
    
    # Visualize submodel contributions
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    for i, sm_pred in enumerate(submodel_preds):
        plt.plot(X_test, sm_pred.flatten(), alpha=0.3, label=f'Submodel {i+1}')
    plt.plot(X_test, y_pred.flatten(), 'k-', linewidth=2, label='Weighted prediction')
    plt.legend()
    plt.show()
    
    # Get derivative predictions (shared derivatives/orders only)
    y_with_derivs, cov_with_derivs = gp.predict(
        X_test, params, calc_cov=True, return_deriv=True
    )
    
    # Extract function and derivative values by row
    func_pred = y_with_derivs[0, :]      # f(x)
    deriv1_pred = y_with_derivs[1, :]    # First derivative (if shared)

------------------------------------------------------------

Predictive Uncertainty
----------------------

**Understanding Predictive Variance**

The predictive variance quantifies uncertainty in the GP predictions:

- **High variance** indicates regions with:
  
  - Sparse or no nearby training data
  - High observation noise
  - Poor model fit
  
- **Low variance** indicates regions with:
  
  - Dense training data coverage
  - Low observation noise
  - High model confidence

**Computing variance:**

::

    y_pred, y_var = gp.predict(X_test, params, calc_cov=True)
    
    # Standard deviation (uncertainty in same units as prediction)
    y_std = np.sqrt(y_var)
    
    # 95% confidence interval (assuming Gaussian predictive distribution)
    confidence_interval = 1.96 * y_std

**Covariance vs. Variance:**

- When ``calc_cov=True`` with single test points or independent predictions, returns **variance** (scalar per test point)
- Some implementations may return full **covariance matrix** between test points for joint predictions
- Check specific model documentation for covariance matrix support

------------------------------------------------------------


Performance Considerations
--------------------------

**Computational Cost**

Prediction cost depends on:

1. **Number of training points** (n_train): Dominant factor in covariance computation
2. **Number of test points** (n_test): Linear scaling
3. **Dimensionality** (n_dims): Affects kernel evaluations
4. **Uncertainty computation** (calc_cov): Adds computational overhead
5. **Derivative predictions** (return_deriv): Increases complexity
6. **Number of derivatives** (derivs_to_predict): Fewer derivatives = faster computation

**Approximate scaling:**

- **Without variance:** O(n_train × n_test)
- **With variance:** O(n_train² × n_test) due to matrix operations
- **With derivatives:** Scales with number of derivative components
- **With derivs_to_predict:** Scales with len(derivs_to_predict), not total training derivatives

**Optimization tips:**

1. **Disable uncertainty** when not needed:
   
   ::
   
       y_pred = gp.predict(X_test, params, calc_cov=False)

2. **Use derivs_to_predict** to compute only needed derivatives:
   
   ::
   
       # Instead of computing all derivatives
       y_pred, _ = gp.predict(X_test, params, return_deriv=True)
       
       # Compute only what you need
       derivs_to_predict = [[[1, 1]]]  # Just one derivative
       y_pred, _ = gp.predict(X_test, params, return_deriv=True, derivs_to_predict=derivs_to_predict)

3. **Batch predictions** instead of sequential calls:
   
   ::
   
       # Efficient: single call
       y_pred = gp.predict(X_test_large, params)
       
       # Inefficient: multiple calls
       for x in X_test_large:
           y = gp.predict(x.reshape(1, -1), params)

4. **For large test sets**, consider splitting:
   
   ::
   
       # Split large test set into chunks
       chunk_size = 1000
       predictions = []
       for i in range(0, len(X_test), chunk_size):
           X_chunk = X_test[i:i+chunk_size]
           y_chunk = gp.predict(X_chunk, params, calc_cov=False)
           predictions.append(y_chunk)
       y_pred = np.concatenate(predictions)

------------------------------------------------------------

Common Issues and Troubleshooting
----------------------------------

**Issue 1: Predictions far from training data have high uncertainty**

This is **expected behavior**. GP predictions extrapolate poorly beyond the training data range.

**Solution:**

- Ensure test points are within or near the training data domain
- Add training data in regions where predictions are needed

**Issue 2: Negative variances**

Should not occur with properly implemented GPs, but can arise from:

- Numerical instability in covariance matrix inversion
- Ill-conditioned covariance matrices

**Solution:**

- Enable normalization: ``normalize=True``
- Check for duplicate training points
- Reduce derivative order if using high-order derivatives
- Verify hyperparameter optimization converged properly

**Issue 3: Very large or very small predictions**

Can occur when normalization is disabled and data scales vary widely.

**Solution:**

- Always use ``normalize=True`` for numerical stability
- Manually scale data if custom scaling is required

**Issue 4: Slow predictions**

**Solution:**

- Disable uncertainty computation if not needed (``calc_cov=False``)
- Reduce training set size if feasible (consider sparse GP methods)
- Disable derivative predictions if not required (``return_deriv=False``)
- Use ``derivs_to_predict`` to compute only the derivatives you need

**Issue 5: GDDEGP derivative predictions require rays_predict**

When using GDDEGP with ``return_deriv=True``, you must provide ``rays_predict``.

**Solution:**

::

    # Define directional vectors at each test point
    # rays_predict[direction_idx] has shape (d, n_test)
    rays_dir1 = np.zeros((d, n_test))
    # ... populate with appropriate directions ...
    
    rays_predict = [rays_dir1]  # Add more if multiple directions
    
    y_pred, y_var = gp.predict(
        X_test, params,
        rays_predict=rays_predict,
        calc_cov=True,
        return_deriv=True
    )

**Note:** ``rays_predict`` is only required when ``return_deriv=True``. For function-only 
predictions, simply omit it:

::

    # Function predictions only - no rays_predict needed
    y_pred, y_var = gp.predict(X_test, params, calc_cov=True, return_deriv=False)

**Issue 6: Understanding the row-wise derivative output format**

When ``return_deriv=True``, predictions are returned with shape ``(num_derivs + 1, n_test)``.

**Solution:**

- Remember the structure: rows = output types, columns = test points
- Row 0 is always function values
- Subsequent rows are derivatives in the order specified by ``der_indices`` (or ``derivs_to_predict`` if specified)
- Use row indexing to extract components:
  
  ::
  
      func_pred = y_pred[0, :]     # Function values
      deriv1_pred = y_pred[1, :]   # First derivative component
      deriv2_pred = y_pred[2, :]   # Second derivative component
      # And so on for each derivative component

**Issue 7: WDEGP derivative predictions missing some derivatives**

For DEGP/DDEGP submodels, WDEGP can only predict derivatives that are **shared across all submodels**.
For GDDEGP submodels, WDEGP can only predict derivative **orders** shared across all submodels.

**Solution:**

- For DEGP/DDEGP: Check that all submodels include the desired derivative in their ``der_indices``
- For GDDEGP: Check that all submodels include the desired derivative order
- If a derivative/order is only in some submodels, it cannot be predicted globally
- Consider restructuring submodels to share common derivatives or orders

**Issue 8: WDEGP with GDDEGP - can I predict in different directions than training?**

Yes! This is a key advantage of GDDEGP submodels.

**Solution:**

::

    # Training used gradient-aligned directions, but you can predict ANY direction:
    
    # Predict along x-axis (even if training never used this direction)
    rays_x = np.array([[1.0] * n_test, [0.0] * n_test])
    
    y_pred, _ = gp.predict(
        X_test, params,
        rays_predict=[rays_x],
        return_deriv=True
    )
    
    # Works as long as the derivative ORDER was used in training

**Issue 9: derivs_to_predict contains derivatives not in training**

``derivs_to_predict`` must be a subset of the derivatives used during training.

**Solution:**

- Verify each entry in ``derivs_to_predict`` exists in the flattened ``der_indices``
- Check that derivative specifications match exactly (including order of variable-order pairs)

::

    # Training derivatives
    der_indices = [[[[1, 1]], [[2, 1]]], [[[1, 2]]]]
    # Flattened: [[1,1]], [[2,1]], [[1,2]]
    
    # Valid derivs_to_predict
    derivs_to_predict = [[[1, 1]], [[1, 2]]]  # Both exist in training
    
    # Invalid: [[2,2]] was not in training
    derivs_to_predict = [[[1, 1]], [[2, 2]]]  # Error!

**Issue 10: Output shape unexpected when using derivs_to_predict**

When ``derivs_to_predict`` is specified, output shape is ``(len(derivs_to_predict) + 1, n_test)``, 
not ``(num_all_derivs + 1, n_test)``.

**Solution:**

- Remember that row count equals ``len(derivs_to_predict) + 1``
- Row order matches the order in ``derivs_to_predict``, not ``der_indices``

::

    # Training had 5 derivatives
    der_indices = [
        [[[1, 1]], [[2, 1]]],
        [[[1, 2]], [[1, 1], [2, 1]], [[2, 2]]]
    ]
    
    # Request only 2 derivatives in specific order
    derivs_to_predict = [[[2, 1]], [[1, 2]]]  # ∂f/∂x₂, then ∂²f/∂x₁²
    
    y_pred, _ = gp.predict(X_test, params, return_deriv=True, derivs_to_predict=derivs_to_predict)
    
    # y_pred.shape = (3, n_test), NOT (6, n_test)
    # y_pred[0, :] = function values
    # y_pred[1, :] = ∂f/∂x₂ (first in derivs_to_predict)
    # y_pred[2, :] = ∂²f/∂x₁² (second in derivs_to_predict)

------------------------------------------------------------

Best Practices
--------------

1. **Always enable normalization** during model initialization for numerical stability:
   
   ::
   
       gp = degp(..., normalize=True)

2. **Always provide derivative_locations** specifying which points have each derivative:
   
   ::
   
       # All derivatives at all points
       derivative_locations = []
       for i in range(len(der_indices)):
           for j in range(len(der_indices[i])):
               derivative_locations.append([k for k in range(len(X_train))])
       
       gp = degp(..., derivative_locations=derivative_locations)

3. **Compute uncertainty** for critical predictions to quantify confidence:
   
   ::
   
       y_pred, y_var = gp.predict(X_test, params, calc_cov=True)

4. **Visualize predictions** with confidence intervals to understand model behavior:
   
   ::
   
       y_std = np.sqrt(y_var)
       plt.fill_between(X_test.flatten(), y_pred - 2*y_std, y_pred + 2*y_std, alpha=0.3)

5. **For weighted models**, examine submodel predictions to diagnose model behavior:
   
   ::
   
       y_pred, y_var, sub_vals, sub_cov = gp.predict(..., return_submodels=True)

6. **Use appropriate prediction settings** based on application:
   
   - **Real-time applications:** ``calc_cov=False``, ``return_deriv=False``
   - **Uncertainty quantification:** ``calc_cov=True``
   - **Sensitivity analysis:** ``return_deriv=True``

7. **Validate predictions** against held-out test data when possible

8. **Check prediction uncertainty** - high uncertainty may indicate:
   
   - Insufficient training data
   - Extrapolation beyond training domain
   - Poor hyperparameter optimization

9. **When using return_deriv=True**, extract components using row indexing:
   
   ::
   
       # Function values (row 0)
       func_pred = y_pred[0, :]
       
       # Each derivative component (rows 1, 2, ...)
       deriv_i = y_pred[i + 1, :]

10. **For GDDEGP/WDEGP-GDDEGP with derivative predictions**, always normalize directional vectors:
    
    ::
    
        for i in range(n_test):
            rays_dir1[:, i] = rays_dir1[:, i] / np.linalg.norm(rays_dir1[:, i])

11. **Leverage GDDEGP flexibility** - predict directional derivatives in any direction:
    
    ::
    
        # Don't feel constrained to training directions
        # Use whatever directions are most meaningful for your analysis

12. **Profile prediction performance** for large-scale applications and optimize accordingly

13. **Use derivs_to_predict for efficiency** when only specific derivatives are needed:
    
    ::
    
        # Instead of computing all 10 derivatives and discarding 8:
        y_pred, _ = gp.predict(X_test, params, return_deriv=True)
        needed = y_pred[[0, 3, 7], :]  # Extract rows manually
        
        # Compute only what you need:
        derivs_to_predict = [[[1, 2]], [[2, 2]]]  # Just these two
        y_pred, _ = gp.predict(
            X_test, params, 
            return_deriv=True, 
            derivs_to_predict=derivs_to_predict
        )

14. **Match derivs_to_predict order to your analysis needs** - output rows follow your specified order:
    
    ::
    
        # If you need ∂f/∂x₂ before ∂f/∂x₁ for downstream processing:
        derivs_to_predict = [[[2, 1]], [[1, 1]]]  # Specify desired order

------------------------------------------------------------


Summary
-------

JetGP provides a flexible and powerful prediction interface across all model variants:

**Key Features:**

- **Unified interface** for function and derivative predictions
- **Optional uncertainty quantification** via ``calc_cov`` parameter
- **Selective derivative predictions** via ``derivs_to_predict`` parameter
- **Row-wise output format** when ``return_deriv=True``: shape ``(num_derivs + 1, n_test)``
- **Automatic denormalization** when ``normalize=True`` during training
- **Efficient vectorized operations** for batch predictions

**Output Format Summary:**

+------------------------------+---------------------------------------+
| Configuration                | Output Shape                          |
+==============================+=======================================+
| ``return_deriv=False``       | ``(n_test,)``                         |
+------------------------------+---------------------------------------+
| ``return_deriv=True``,       | ``(num_all_derivs + 1, n_test)``      |
| ``derivs_to_predict=None``   |                                       |
+------------------------------+---------------------------------------+
| ``return_deriv=True``,       | ``(len(derivs_to_predict) + 1,        |
| ``derivs_to_predict=[...]``  | n_test)``                             |
+------------------------------+---------------------------------------+

**Row Structure when return_deriv=True:**

- Row 0: Function value predictions
- Row 1: First derivative component (per ``der_indices`` or ``derivs_to_predict`` order)
- Row 2: Second derivative component
- ... (following specified order)

**Model-Specific Notes:**

+----------------+-------------------------------------------+------------------------------+
| Model          | ``rays_predict`` Required                 | Derivative Restriction       |
+================+===========================================+==============================+
| DEGP           | N/A                                       | Specific coordinates         |
+----------------+-------------------------------------------+------------------------------+
| DDEGP          | N/A (uses global ``rays`` from training)  | Specific global directions   |
+----------------+-------------------------------------------+------------------------------+
| GDDEGP         | Only when ``return_deriv=True``           | Any direction, trained order |
+----------------+-------------------------------------------+------------------------------+
| WDEGP (DEGP)   | N/A                                       | Shared coordinates only      |
+----------------+-------------------------------------------+------------------------------+
| WDEGP (DDEGP)  | N/A                                       | Shared directions only       |
+----------------+-------------------------------------------+------------------------------+
| WDEGP (GDDEGP) | Only when ``return_deriv=True``           | Any direction, shared order  |
+----------------+-------------------------------------------+------------------------------+

**Best Practices:**

1. Always use ``normalize=True`` during model initialization
2. Always provide ``derivative_locations`` specifying point-derivative associations
3. Compute uncertainty (``calc_cov=True``) for critical predictions
4. Visualize predictions with confidence intervals
5. Extract derivative components using row indexing when ``return_deriv=True``
6. Use ``return_submodels=True`` for weighted models to diagnose performance
7. Normalize directional vectors in GDDEGP when using ``rays_predict``
8. Leverage GDDEGP/WDEGP-GDDEGP flexibility to predict in any direction
9. Use ``derivs_to_predict`` for efficiency and custom derivative ordering
10. Batch large prediction sets for memory efficiency

**Typical Workflow:**

1. Initialize and train model with appropriate configuration (including ``derivative_locations``)
2. Optimize hyperparameters thoroughly
3. Make predictions with uncertainty quantification
4. Use ``derivs_to_predict`` to select specific derivatives if needed
5. Extract relevant components using row indexing (function, derivatives, submodels)
6. Visualize results with confidence bands
7. Validate against test data when available

For more information on model initialization and hyperparameter optimization, 
see the respective documentation sections.