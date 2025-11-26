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
        X_train, y_train, n_order, n_bases=1, der_indices=der_indices,
        normalize=True, kernel="SE", kernel_type="anisotropic"
    )
    
    # 2. Optimize hyperparameters
    params = gp.optimize_hyperparameters(
        optimizer='lbfgs',
        n_restart_optimizer=10
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

- ``True`` → Returns predictions for both function values and their derivatives in a stacked format
- ``False`` → Returns only function value predictions

**Default:** ``False``

**Note:** Not all model variants support derivative predictions. When ``return_deriv=True``, the order of derivative predictions in the stacked output follows the same order as specified in ``der_indices``. See model-specific sections below for details.

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

- ``y_pred`` : array, shape ``(n_test * (num_derivs + 1), 1)``  
  Stacked predictions including function values and all derivatives.
  The first ``n_test`` rows contain function value predictions,
  followed by ``n_test`` rows for each derivative component specified in ``der_indices``.
  
  **Structure:**
  
  - Rows 0 to n_test-1: Function value predictions
  - Rows n_test to 2*n_test-1: First derivative component predictions
  - Rows 2*n_test to 3*n_test-1: Second derivative component predictions
  - ... and so on for all derivative components
  
  where ``num_derivs`` is the total number of derivative components specified in ``der_indices``.
  
- ``y_var`` : array, shape ``(n_test * (num_derivs + 1), 1)`` (if ``calc_cov=True``)  
  Predictive variance corresponding to each row in ``y_pred``, following the same stacking structure.

**Example with return_deriv=True:**

::

    # Model with first-order derivatives in 2D
    # der_indices = [[[[1, 1]], [[2, 1]]]]  # Two first-order derivatives
    # num_derivs = 2
    
    X_test = np.random.rand(10, 2)  # 10 test points
    y_pred, y_var = gp.predict(X_test, params, calc_cov=True, return_deriv=True)
    
    # y_pred.shape = (30, 1)  # 10 * (2 + 1) = 30
    # y_var.shape = (30, 1)
    
    # Extract predictions:
    n_test = X_test.shape[0]
    func_pred = y_pred[:n_test]              # Function predictions (rows 0-9)
    deriv1_pred = y_pred[n_test:2*n_test]    # ∂/∂x₁ predictions (rows 10-19)
    deriv2_pred = y_pred[2*n_test:3*n_test]  # ∂/∂x₂ predictions (rows 20-29)
    
    # Similarly for variances:
    func_var = y_var[:n_test]
    deriv1_var = y_var[n_test:2*n_test]
    deriv2_var = y_var[2*n_test:3*n_test]

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
    
    # y_pred.shape = (60, 1)  # 10 * (5 + 1) = 60
    
    n_test = 10
    func_pred = y_pred[0*n_test:1*n_test]       # f(x)
    d1_dx1 = y_pred[1*n_test:2*n_test]          # ∂f/∂x₁
    d1_dx2 = y_pred[2*n_test:3*n_test]          # ∂f/∂x₂
    d2_dx1 = y_pred[3*n_test:4*n_test]          # ∂²f/∂x₁²
    d2_dx1dx2 = y_pred[4*n_test:5*n_test]       # ∂²f/∂x₁∂x₂
    d2_dx2 = y_pred[5*n_test:6*n_test]          # ∂²f/∂x₂²

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

**Returned shapes:**

- ``y_pred``: ``(n_test,)`` if ``return_deriv=False``
- ``y_pred``: ``(n_test * (num_derivs + 1), 1)`` if ``return_deriv=True``
- ``y_var``: matches ``y_pred`` shape when ``calc_cov=True``

----

**2. WDEGP (Weighted DEGP)**

Weighted models combine predictions from multiple submodels and provide additional 
diagnostics for each submodel. Derivative predictions are supported with some restrictions
(see below).

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

**Additional arguments:**

- ``return_deriv`` – Boolean flag to include derivative predictions (default: ``False``)

**Returns when return_deriv=True:**

- ``y_pred`` : array, shape ``(n_test * (1 + n_shared_derivs),)``  
  Stacked predictions: ``[f(x1), df/dx1(x1), ..., f(x2), df/dx2(x2), ...]``
  Only includes derivatives that are shared across all submodels (see restrictions below)
  
- ``y_cov`` : array, shape ``(n_test * (1 + n_shared_derivs),)`` (if ``calc_cov=True``)  
  Stacked predictive variances corresponding to ``y_pred``

**With submodel outputs:**

::

    y_pred, y_cov, submodel_vals, submodel_cov = gp.predict(
        X_test, params, calc_cov=True, return_submodels=True
    )

**Additional arguments:**

- ``return_submodels`` – Boolean flag to return individual submodel predictions (default: ``False``)

**Additional returns when return_submodels=True:**

- ``submodel_vals`` : array, shape ``(n_submodels, n_test)``  
  Predictions from each individual submodel
  
- ``submodel_cov`` : array, shape ``(n_submodels, n_test)`` (if ``calc_cov=True``)  
  Predictive variances from each individual submodel

**Derivative Prediction Restrictions:**

Derivative predictions in WDEGP have the following limitations:

1. **Shared derivatives only (multi-submodel case):**  
   For weighted predictions across multiple submodels, derivatives can only be predicted if 
   they appear in **all** submodels.
   
   ::
   
       # Example: Two submodels
       # Submodel 1 has derivatives: [[[1,1]]]              (∂f/∂x only)
       # Submodel 2 has derivatives: [[[1,1]], [[1,2]]]    (∂f/∂x and ∂²f/∂x²)
       
       # Global predictions can ONLY be made for [[1,1]] (∂f/∂x)
       # [[1,2]] (∂²f/∂x²) is not shared, so it's excluded from weighted predictions

2. **Training derivatives only (single submodel case):**  
   Predictions can only be made for derivatives that were explicitly included in training.
   
   ::
   
       # Example: Single submodel with [[1,1]]
       # Can predict: f(x) and ∂f/∂x
       # Cannot predict: ∂²f/∂x² (not used in training)

3. **Point-level availability:**  
   As long as **at least one training point** includes a specific derivative, that derivative
   can be predicted anywhere in the input space via the GP.
   
   ::
   
       # Example: 10 training points
       # Only points [0, 2, 5] have ∂f/∂x information
       # GP can still predict ∂f/∂x at any test point

4. **Alternative for unavailable derivatives:**  
   If a desired derivative was not used in training, you can approximate it using finite 
   differences on the GP function predictions:
   
   ::
   
       # Approximate ∂²f/∂x² using finite differences
       h = 1e-5
       X_test_plus = X_test + h
       X_test_minus = X_test - h
       
       f_plus, _ = gp.predict(X_test_plus, params)
       f_center, _ = gp.predict(X_test, params)
       f_minus, _ = gp.predict(X_test_minus, params)
       
       d2f_dx2_approx = (f_plus - 2*f_center + f_minus) / (h**2)

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
    for i in range(submodel_preds.shape[0]):
        plt.plot(X_test, submodel_preds[i], alpha=0.3, label=f'Submodel {i+1}')
    plt.plot(X_test, y_pred, 'k-', linewidth=2, label='Weighted prediction')
    plt.legend()
    plt.show()
    
    # Get derivative predictions (shared derivatives only)
    y_with_derivs, cov_with_derivs = gp.predict(
        X_test, params, calc_cov=True, return_deriv=True
    )
    
    # Extract function and derivative values
    n_test = len(X_test)
    n_derivs_per_point = len(y_with_derivs) // n_test
    
    y_pred_reshaped = y_with_derivs.reshape(n_derivs_per_point, n_test).T
    # y_pred_reshaped[i, 0] = f(x_i)
    # y_pred_reshaped[i, 1] = df/dx(x_i)  (first shared derivative)
    # y_pred_reshaped[i, 2] = d2f/dx2(x_i)  (second shared derivative, if available)
    # etc.

----

**3. DDEGP (Directional Derivative-Enhanced GP)**

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

- ``y_pred`` : array, shape ``(n_test * (num_derivs + 1), 1)``  
  Stacked predictions including function values and directional derivatives.
  The first ``n_test`` rows contain function predictions, followed by ``n_test`` rows 
  for each directional derivative order specified in ``der_indices``.
  
- ``y_var`` : array, shape ``(n_test * (num_derivs + 1), 1)`` (if ``calc_cov=True``)  
  Predictive variance for each prediction component

When ``return_deriv=True``, predictions include directional derivatives along the 
same global directions used during training.

----

**4. GDDEGP (Generalized Directional Derivative-Enhanced GP)**

Allows **point-specific directional derivatives** (different directions at each training point).

**Critical difference:** Requires ``rays_pred`` argument specifying prediction directions.

**Signature:**

::

    y_pred, y_var = gp.predict(
        X_test, rays_pred, params, calc_cov=True, return_deriv=False
    )

**Additional required argument:**

- ``rays_pred`` : array, shape ``(n_test, n_dims)`` or ``(n_test, n_rays, n_dims)``  
  Directional vectors for derivative predictions at test locations.  
  **Must be provided even if return_deriv=False.**

**Returns:**

- ``y_pred`` : array, shape ``(n_test,)`` when ``return_deriv=False``  
  Predicted function values
  
- ``y_var`` : array, shape ``(n_test,)`` (if ``calc_cov=True``)  
  Predictive variance

**With derivative predictions:**

::

    y_pred, y_var = gp.predict(
        X_test, rays_pred, params, calc_cov=True, return_deriv=True
    )

**Returns:**

- ``y_pred`` : array, shape ``(n_test * (num_derivs + 1), 1)``  
  Stacked predictions including function values and directional derivatives 
  along the directions specified in ``rays_pred``
  
- ``y_var`` : array, shape ``(n_test * (num_derivs + 1), 1)`` (if ``calc_cov=True``)  
  Predictive variance for each prediction component

When ``return_deriv=True``, returns predictions for directional derivatives along 
the directions specified in ``rays_pred``.

**Example:**

::

    import numpy as np
    
    # Define test points
    X_test = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])
    
    # Define directional vectors for predictions at each test point
    rays_pred = np.array([
        [1.0, 0.0],       # Direction at first test point
        [0.0, 1.0],       # Direction at second test point
        [0.707, 0.707]    # Direction at third test point (normalized)
    ])
    
    # Make predictions (rays_pred required even with return_deriv=False)
    y_pred, y_var = gp.predict(
        X_test, rays_pred, params, calc_cov=True, return_deriv=False
    )

**Important notes:**

- ``rays_pred`` must be provided even when ``return_deriv=False``
- Directional vectors in ``rays_pred`` should be normalized (unit length)
- Each test point can have different directional vectors

----

**5. WDDEGP (Weighted Directional Derivative-Enhanced GP)**

Combines weighted submodel framework with directional derivatives.

**Signature** (similar to GDDEGP but with submodel options):

::

    y_pred, y_var = gp.predict(
        X_test, rays_pred, params, calc_cov=True, return_submodels=False
    )

**With submodel outputs:**

::

    y_pred, y_var, submodel_vals, submodel_cov = gp.predict(
        X_test, rays_pred, params, calc_cov=True, return_submodels=True
    )

Similar to WDEGP, setting ``return_submodels=True`` provides individual submodel predictions 
for diagnostic and visualization purposes.

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

**Approximate scaling:**

- **Without variance:** O(n_train × n_test)
- **With variance:** O(n_train² × n_test) due to matrix operations
- **With derivatives:** Scales with number of derivative components

**Optimization tips:**

1. **Disable uncertainty** when not needed:
   
   ::
   
       y_pred = gp.predict(X_test, params, calc_cov=False)

2. **Batch predictions** instead of sequential calls:
   
   ::
   
       # Efficient: single call
       y_pred = gp.predict(X_test_large, params)
       
       # Inefficient: multiple calls
       for x in X_test_large:
           y = gp.predict(x.reshape(1, -1), params)

3. **For large test sets**, consider splitting:
   
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

**Issue 5: GDDEGP requires rays_pred but I only want function predictions**

**Solution:**

- ``rays_pred`` must always be provided for GDDEGP, even when ``return_deriv=False``
- Provide arbitrary directional vectors if derivative predictions are not needed:
  
  ::
  
      # Dummy rays (e.g., unit vectors in first dimension)
      rays_pred = np.zeros((X_test.shape[0], X_test.shape[1]))
      rays_pred[:, 0] = 1.0
      
      y_pred = gp.predict(X_test, rays_pred, params, calc_cov=False, return_deriv=False)

**Issue 6: Confusion about stacked derivative output shape**

When ``return_deriv=True``, predictions are returned in a stacked format.

**Solution:**

- Remember the structure: ``(n_test * (num_derivs + 1), 1)``
- Use slicing to extract components:
  
  ::
  
      n_test = X_test.shape[0]
      func_pred = y_pred[:n_test]
      deriv1_pred = y_pred[n_test:2*n_test]
      deriv2_pred = y_pred[2*n_

test:3*n_test]
      # And so on for each derivative component

------------------------------------------------------------

Best Practices
--------------

1. **Always enable normalization** during model initialization for numerical stability:
   
   ::
   
       gp = degp(..., normalize=True)

2. **Compute uncertainty** for critical predictions to quantify confidence:
   
   ::
   
       y_pred, y_var = gp.predict(X_test, params, calc_cov=True)

3. **Visualize predictions** with confidence intervals to understand model behavior:
   
   ::
   
       y_std = np.sqrt(y_var)
       plt.fill_between(X_test, y_pred - 2*y_std, y_pred + 2*y_std, alpha=0.3)

4. **For weighted models**, examine submodel predictions to diagnose model behavior:
   
   ::
   
       y_pred, y_var, sub_vals, sub_cov = gp.predict(..., return_submodels=True)

5. **Use appropriate prediction settings** based on application:
   
   - **Real-time applications:** ``calc_cov=False``, ``return_deriv=False``
   - **Uncertainty quantification:** ``calc_cov=True``
   - **Sensitivity analysis:** ``return_deriv=True``

6. **Validate predictions** against held-out test data when possible

7. **Check prediction uncertainty** - high uncertainty may indicate:
   
   - Insufficient training data
   - Extrapolation beyond training domain
   - Poor hyperparameter optimization

8. **When using return_deriv=True**, carefully extract components using proper slicing:
   
   ::
   
       n_test = X_test.shape[0]
       # Function values
       func_pred = y_pred[:n_test].ravel()
       # Each derivative component
       deriv_i = y_pred[i*n_test:(i+1)*n_test].ravel()

9. **For GDDEGP**, always normalize directional vectors:
   
   ::
   
       rays_pred = rays_pred / np.linalg.norm(rays_pred, axis=1, keepdims=True)

10. **Profile prediction performance** for large-scale applications and optimize accordingly

------------------------------------------------------------


Summary
-------

JetGP provides a flexible and powerful prediction interface across all model variants:

**Key Features:**

- **Unified interface** for function and derivative predictions
- **Optional uncertainty quantification** via ``calc_cov`` parameter
- **Stacked output format** when ``return_deriv=True``: shape ``(n_test * (num_derivs + 1), 1)``
- **Automatic denormalization** when ``normalize=True`` during training
- **Efficient vectorized operations** for batch predictions



**Best Practices:**

1. Always use ``normalize=True`` during model initialization
2. Compute uncertainty (``calc_cov=True``) for critical predictions
3. Visualize predictions with confidence intervals
4. Extract derivative components carefully using proper slicing when ``return_deriv=True``
5. Use ``return_submodels=True`` for weighted models to diagnose performance
6. Normalize directional vectors in GDDEGP/WDDEGP
7. Batch large prediction sets for memory efficiency

**Typical Workflow:**

1. Initialize and train model with appropriate configuration
2. Optimize hyperparameters thoroughly
3. Make predictions with uncertainty quantification
4. Extract relevant components (function, derivatives, submodels)
5. Visualize results with confidence bands
6. Validate against test data when available

For more information on model initialization and hyperparameter optimization, 
see the respective documentation sections.
```

This completes the comprehensive Making Predictions documentation!