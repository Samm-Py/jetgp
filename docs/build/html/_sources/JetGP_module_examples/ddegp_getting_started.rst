Directional Derivative-Enhanced Gaussian Process (DDEGP)
========================================================

Overview
--------
The **Directional Derivative-Enhanced Gaussian Process (DDEGP)** extends the standard DEGP framework by incorporating derivative information along **specified directional rays** rather than coordinate axes. This enables the model to:

- Capture local behavior along arbitrary directions in the input space
- Reduce the number of required derivative evaluations in high-dimensional problems
- Focus derivative information along problem-specific directions of interest
- Provide more flexible derivative observations compared to axis-aligned derivatives

This tutorial demonstrates DDEGP on the 2D Branin function using a **global basis of directional derivatives** applied at all training points. The directional approach is particularly useful when:

- Coordinate-aligned derivatives are expensive or unavailable
- The problem has known directional characteristics
- Reducing the total number of derivative evaluations is critical
- Working with high-dimensional spaces where full derivative information is impractical

---

Example 1: Global Directional GP on the Branin Function
--------------------------------------------------------

Overview
~~~~~~~~
This example demonstrates a **Directional-Derivative Enhanced Gaussian Process (DDEGP)** applied to the 2D Branin function. We use a **global set of directional rays** that are applied at all training points, allowing the GP to learn local behavior along these specific directions.

Key concepts covered:

- Using a **global basis of directional derivatives** for all training points
- **Latin Hypercube Sampling (LHS)** for efficient training data generation
- Training and evaluating the ``ddegp`` model
- Visualizing predictions with directional rays overlaid on the results

---

Step 1: Import required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    import numpy as np
    import sympy as sp
    from full_ddegp.ddegp import ddegp
    import utils
    from scipy.stats import qmc
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

**Explanation:**  
We import necessary modules for numerical operations, symbolic differentiation (``sympy``), the DDEGP model, Latin Hypercube Sampling (``qmc``), and visualization tools.

---

Step 2: Set configuration parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    n_order = 1
    n_bases = 2
    num_training_pts = 16
    domain_bounds = ((-5.0, 10.0), (0.0, 15.0))
    test_grid_resolution = 50

    # Global set of directional rays
    rays = np.array([
        [np.cos(np.pi/4), np.cos(np.pi/2), np.cos(3*np.pi/4)],
        [np.sin(np.pi/4), np.sin(np.pi/2), np.sin(3*np.pi/4)]
    ])

    normalize_data = True
    kernel = "SE"
    kernel_type = "anisotropic"
    random_seed = 1
    np.random.seed(random_seed)
    
    print("Configuration complete!")
    print(f"Number of training points: {num_training_pts}")
    print(f"Number of directional rays: {rays.shape[1]}")
    print(f"Ray directions:")
    for i in range(rays.shape[1]):
        angle_deg = np.arctan2(rays[1, i], rays[0, i]) * 180 / np.pi
        print(f"  Ray {i+1}: [{rays[0, i]:+.4f}, {rays[1, i]:+.4f}] (angle: {angle_deg:.1f}°)")

**Explanation:**  
We configure the experiment parameters:

- ``n_order=1``: First-order directional derivatives
- ``n_bases=2``: Two-dimensional input space
- ``rays``: Three directional vectors at 45°, 90°, and 135° angles
- ``kernel="SE"``: Squared Exponential kernel for smooth interpolation
- The directional rays define the directions along which derivatives will be computed at each training point

---

Step 3: Define the Branin function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    def branin_function(X, alg=np):
        """2D Branin function - a common benchmark for optimization."""
        x1, x2 = X[:, 0], X[:, 1]
        a, b, c, r, s, t = 1.0, 5.1/(4.0*np.pi**2), 5.0/np.pi, 6.0, 10.0, 1.0/(8.0*np.pi)
        return a * (x2 - b*x1**2 + c*x1 - r)**2 + s*(1 - t)*alg.cos(x1) + s

    # Define symbolic version for derivatives
    x1_sym, x2_sym = sp.symbols('x1 x2')
    a, b, c, r, s, t = 1.0, 5.1/(4.0*sp.pi**2), 5.0/sp.pi, 6.0, 10.0, 1.0/(8.0*sp.pi)
    f_sym = a * (x2_sym - b*x1_sym**2 + c*x1_sym - r)**2 + s*(1 - t)*sp.cos(x1_sym) + s

    # Compute gradients symbolically
    grad_x1 = sp.diff(f_sym, x1_sym)
    grad_x2 = sp.diff(f_sym, x2_sym)

    # Convert to NumPy functions
    f_func = sp.lambdify([x1_sym, x2_sym], f_sym, 'numpy')
    grad_x1_func = sp.lambdify([x1_sym, x2_sym], grad_x1, 'numpy')
    grad_x2_func = sp.lambdify([x1_sym, x2_sym], grad_x2, 'numpy')
    
    print("Branin function and symbolic derivatives defined!")

**Explanation:**  
The Branin function is a standard 2D test function with three global minima. We define both a numerical version and a symbolic version using SymPy. The symbolic version allows us to compute exact partial derivatives :math:`\frac{\partial f}{\partial x_1}` and :math:`\frac{\partial f}{\partial x_2}`, which are then converted to fast NumPy-compatible functions using ``lambdify``.

---

Step 4: Generate training data with directional derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Latin Hypercube Sampling for training points
    sampler = qmc.LatinHypercube(d=n_bases, seed=random_seed)
    unit_samples = sampler.random(n=num_training_pts)
    X_train = qmc.scale(unit_samples, [b[0] for b in domain_bounds], [b[1] for b in domain_bounds])

    # Compute function values
    y_func = f_func(X_train[:, 0], X_train[:, 1]).reshape(-1, 1)
    
    # Compute coordinate-aligned gradients
    grad_x1_vals = grad_x1_func(X_train[:, 0], X_train[:, 1]).reshape(-1, 1)
    grad_x2_vals = grad_x2_func(X_train[:, 0], X_train[:, 1]).reshape(-1, 1)
    
    # Compute directional derivatives using the chain rule
    # For each ray: d_ray = grad_x1 * ray[0] + grad_x2 * ray[1]
    directional_derivs = []
    for i in range(rays.shape[1]):
        ray_direction = rays[:, i]
        dir_deriv = (grad_x1_vals * ray_direction[0] + 
                     grad_x2_vals * ray_direction[1])
        directional_derivs.append(dir_deriv)
    
    # Package training data
    y_train_list = [y_func] + directional_derivs
    der_indices = [[[[1, 1]], [[2, 1]], [[3, 1]]]]
    
    print(f"Training data generated!")
    print(f"X_train shape: {X_train.shape}")
    print(f"Function values shape: {y_func.shape}")
    print(f"Number of directional derivative arrays: {len(directional_derivs)}")

**Explanation:**  
This step performs several key operations:

1. **Latin Hypercube Sampling**: Efficiently distributes training points across the domain
2. **Function evaluation**: Computes function values at all training points
3. **Gradient computation**: Evaluates partial derivatives :math:`\frac{\partial f}{\partial x_1}` and :math:`\frac{\partial f}{\partial x_2}` using the symbolic functions
4. **Directional derivatives**: Uses the chain rule to compute derivatives along each ray direction:

   .. math::
      \frac{\partial f}{\partial \mathbf{d}} = \frac{\partial f}{\partial x_1} d_1 + \frac{\partial f}{\partial x_2} d_2

   where :math:`\mathbf{d} = [d_1, d_2]` is the ray direction vector

5. **Data packaging**: Organizes function values and directional derivatives for the DDEGP model

The result is training data with function values and three directional derivatives at each of the 16 training points, computed analytically using SymPy's symbolic differentiation.

---

Step 5: Initialize and train the DDEGP model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Initialize the DDEGP model
    gp_model = ddegp(
        X_train, 
        y_train_list,
        n_order=n_order, 
        der_indices=der_indices,
        rays=rays, 
        normalize=normalize_data,
        kernel=kernel, 
        kernel_type=kernel_type
    )
    
    print("DDEGP model initialized!")
    print("Optimizing hyperparameters...")
    
    # Optimize hyperparameters
    params = gp_model.optimize_hyperparameters(
        optimizer='pso',
        pop_size=200,
        n_generations=15,
        local_opt_every=None,
        debug=False
    )
    
    print("Optimization complete!")
    print(f"Optimized parameters: {params}")

**Explanation:**  
The DDEGP model is initialized with:

- Training locations and derivative data
- Directional rays defining the derivative directions
- Squared Exponential kernel for smooth modeling

Hyperparameters are optimized using Particle Swarm Optimization (PSO) with a population of 200 for 15 generations to find optimal kernel parameters.

---

Step 6: Evaluate model on a test grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Create dense test grid
    x_lin = np.linspace(domain_bounds[0][0], domain_bounds[0][1], test_grid_resolution)
    y_lin = np.linspace(domain_bounds[1][0], domain_bounds[1][1], test_grid_resolution)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    print(f"Test grid: {test_grid_resolution}×{test_grid_resolution} = {len(X_test)} points")
    
    # Predict on test grid
    y_pred = gp_model.predict(X_test, params, calc_cov=False, return_deriv=False)
    
    # Compute ground truth and error
    y_true = branin_function(X_test, alg=np)
    nrmse = utils.nrmse(y_true, y_pred)
    abs_error = np.abs(y_true - y_pred)
    
    print(f"\nModel Performance:")
    print(f"  NRMSE: {nrmse:.6f}")
    print(f"  Max absolute error: {abs_error.max():.6f}")
    print(f"  Mean absolute error: {abs_error.mean():.6f}")

**Explanation:**  
The model is evaluated on a 50×50 grid covering the entire domain. The Normalized Root Mean Square Error (NRMSE) quantifies prediction accuracy across the test set.

---

Step 7: Verify interpolation of training data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

   # ------------------------------------------------------------
   # Verify function value interpolation at all training points
   # ------------------------------------------------------------
   y_func_values = y_train_list[0]  # Function values
   
   # Predict at training points (function values only)
   y_pred_train = gp_model.predict(X_train, params, calc_cov=False, return_deriv=False)
   
   print("Function value interpolation errors:")
   print("=" * 70)
   for i in range(num_training_pts):
       error_abs = abs(y_pred_train[0, i] - y_func_values[i, 0])
       error_rel = error_abs / abs(y_func_values[i, 0]) if y_func_values[i, 0] != 0 else error_abs
       print(f"Point {i} (x1={X_train[i, 0]:.4f}, x2={X_train[i, 1]:.4f}): "
             f"Abs Error = {error_abs:.2e}, Rel Error = {error_rel:.2e}")
   
   max_func_error = np.max(np.abs(y_pred_train.flatten() - y_func_values.flatten()))
   print(f"\nMaximum absolute function value error: {max_func_error:.2e}")
   
   # ------------------------------------------------------------
   # Verify directional derivative interpolation
   # ------------------------------------------------------------
   print("\n" + "=" * 70)
   print("Directional derivative interpolation verification:")
   print("=" * 70)
   print(f"Number of directional rays: {rays.shape[1]}")
   print(f"Ray directions:")
   for i in range(rays.shape[1]):
       angle_deg = np.arctan2(rays[1, i], rays[0, i]) * 180 / np.pi
       print(f"  Ray {i+1}: [{rays[0, i]:+.4f}, {rays[1, i]:+.4f}] (angle: {angle_deg:.1f}°)")
   print("=" * 70)
   
   # Predict with derivatives - returns concatenated vector [func_vals, deriv1_vals, deriv2_vals, deriv3_vals]
   y_pred_with_derivs = gp_model.predict(X_train, params, calc_cov=False, return_deriv=True)
   
   print(f"\nPrediction with derivatives shape: {y_pred_with_derivs.shape}")
   print(f"Expected: function values ({num_training_pts}) + 3 rays × {num_training_pts} = {num_training_pts * 4}")
   
   # Extract predicted values from concatenated vector
   # Format: [func_0, func_1, ..., func_15, ray1_0, ray1_1, ..., ray1_15, ray2_0, ..., ray2_15, ray3_0, ..., ray3_15]
   start_idx = 0
   pred_func_vals = y_pred_with_derivs[start_idx:start_idx + num_training_pts]
   start_idx += num_training_pts
   
   pred_deriv_ray1 = y_pred_with_derivs[start_idx:start_idx + num_training_pts]
   start_idx += num_training_pts
   
   pred_deriv_ray2 = y_pred_with_derivs[start_idx:start_idx + num_training_pts]
   start_idx += num_training_pts
   
   pred_deriv_ray3 = y_pred_with_derivs[start_idx:start_idx + num_training_pts]
   
   # Extract analytic derivatives from training data
   analytic_deriv_ray1 = y_train_list[1]
   analytic_deriv_ray2 = y_train_list[2]
   analytic_deriv_ray3 = y_train_list[3]
   
   # Verify each ray's directional derivatives
   pred_derivs = [pred_deriv_ray1, pred_deriv_ray2, pred_deriv_ray3]
   analytic_derivs = [analytic_deriv_ray1, analytic_deriv_ray2, analytic_deriv_ray3]
   
   for ray_idx in range(rays.shape[1]):
       print(f"\n{'-'*70}")
       print(f"Ray {ray_idx + 1} - Direction: [{rays[0, ray_idx]:+.4f}, {rays[1, ray_idx]:+.4f}]")
       print(f"{'-'*70}")
       
       pred_deriv = pred_derivs[ray_idx]
       analytic_deriv = analytic_derivs[ray_idx]
       
       for i in range(num_training_pts):
           error_abs = abs(pred_deriv[i, 0] - analytic_deriv[i, 0])
           error_rel = error_abs / abs(analytic_deriv[i, 0]) if analytic_deriv[i, 0] != 0 else error_abs
           
           print(f"Point {i} (x1={X_train[i, 0]:.4f}, x2={X_train[i, 1]:.4f}):")
           print(f"  Analytic: {analytic_deriv[i, 0]:+.6f}, Predicted: {pred_deriv[i, 0]:+.6f}")
           print(f"  Abs Error: {error_abs:.2e}, Rel Error: {error_rel:.2e}")
       
       max_deriv_error = np.max(np.abs(pred_deriv.flatten() - analytic_deriv.flatten()))
       print(f"\nMaximum absolute error for Ray {ray_idx + 1}: {max_deriv_error:.2e}")
   
   print("\n" + "=" * 70)
   print("Interpolation verification complete!")
   print("Relative errors should be close to machine precision (< 1e-6)")
   print("\n" + "=" * 70)
   print("SUMMARY:")
   print(f"  - Function values: enforced at all {num_training_pts} training points")
   print(f"  - Directional derivatives: {rays.shape[1]} rays at each training point")
   print(f"  - Total constraints: {num_training_pts} function values + "
         f"{num_training_pts * rays.shape[1]} directional derivatives")
   print(f"  - Prediction vector structure: [func_vals ({num_training_pts}), "
         f"ray1 ({num_training_pts}), ray2 ({num_training_pts}), ray3 ({num_training_pts})]")
   print(f"  - Ray 1 direction: [{rays[0, 0]:+.4f}, {rays[1, 0]:+.4f}] "
         f"(angle: {np.arctan2(rays[1, 0], rays[0, 0]) * 180 / np.pi:.1f}°)")
   print(f"  - Ray 2 direction: [{rays[0, 1]:+.4f}, {rays[1, 1]:+.4f}] "
         f"(angle: {np.arctan2(rays[1, 1], rays[0, 1]) * 180 / np.pi:.1f}°)")
   print(f"  - Ray 3 direction: [{rays[0, 2]:+.4f}, {rays[1, 2]:+.4f}] "
         f"(angle: {np.arctan2(rays[1, 2], rays[0, 2]) * 180 / np.pi:.1f}°)")
   print("=" * 70)

**Explanation:**  
This verification step ensures that the DDEGP model correctly interpolates both the
**function values** and **directional derivatives** at all training points.

Key verification steps:

1. **Function values**: Checked at all 16 training points using standard predictions
   
   - Predictions return shape ``(1, n_pts)``, so we use ``y_pred_train[0, i]`` to access values

2. **Directional derivatives**: Verified by requesting derivative predictions at training points
   
   - Predictions with derivatives return shape ``(n_pts * 4, 1)`` as a concatenated vector
   - We slice this into function values and three ray derivatives

3. **Vector structure**: The ``return_deriv=True`` returns a concatenated vector: 
   ``[func_0, ..., func_n, ray1_0, ..., ray1_n, ray2_0, ..., ray2_n, ...]``

4. **Relative errors**: Computed to assess accuracy relative to the magnitude of each quantity

Critical insights for DDEGP verification:

- **Array indexing**: Function predictions have shape ``(1, n_pts)`` and derivative predictions have shape ``(n_pts, 1)``
- **Concatenated output format**: Predictions with derivatives return a flat vector where:
  
  - Indices ``0`` to ``n_pts-1``: Function values
  - Indices ``n_pts`` to ``2*n_pts-1``: First directional derivative (ray 1)
  - Indices ``2*n_pts`` to ``3*n_pts-1``: Second directional derivative (ray 2)
  - And so on for each ray defined in ``der_indices``

- **No finite differences required**: The ``return_deriv=True`` flag enables direct derivative prediction
- **Global directional rays**: The same three ray directions are used at all training points
- **Directional derivative formula**: :math:`\frac{\partial f}{\partial \mathbf{d}} = \nabla f \cdot \mathbf{d}`
  where :math:`\mathbf{d}` is the ray direction vector
- **Multiple constraints per point**: Each training point contributes 1 function value + 3 directional derivatives
- **Symbolic differentiation accuracy**: Training data uses exact derivatives from SymPy

The verification confirms that:

- Function values interpolate to machine precision at training points
- Directional derivatives along each ray match the analytical values
- The GP correctly learned the directional derivative information
- Relative errors should be near machine precision (< :math:`10^{-6}`)

**Advantages of direct derivative prediction**:

1. **Exact verification**: No numerical approximation errors from finite differences
2. **Efficient**: Single prediction call returns all derivatives in one vector
3. **Consistent**: Uses the same prediction mechanism as during training
4. **Scalable**: Works equally well for any number of rays or training points

**Interpretation**: The directional derivative :math:`\frac{\partial f}{\partial \mathbf{d}}` 
represents the rate of change of the function along direction :math:`\mathbf{d}`. By verifying 
these derivatives at training points, we confirm the GP has correctly incorporated the local 
directional information into its learned representation.

---

Step 8: Visualize results with directional rays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Prepare visualization data
    gp_map = y_pred.reshape(X1_grid.shape)
    true_map = y_true.reshape(X1_grid.shape)
    abs_err = np.abs(gp_map - true_map)
    abs_err_clipped = np.clip(abs_err, 1e-8, None)
    
    # Create three-panel figure
    fig, axs = plt.subplots(1, 3, figsize=(19, 5), constrained_layout=True)

    # GP Prediction
    cf1 = axs[0].contourf(X1_grid, X2_grid, gp_map, cmap='viridis')
    fig.colorbar(cf1, ax=axs[0])
    axs[0].scatter(X_train[:, 0], X_train[:, 1], c='red', s=50, edgecolors='black')
    axs[0].set_title("GP Prediction")

    # True Function
    cf2 = axs[1].contourf(X1_grid, X2_grid, true_map, cmap='viridis')
    fig.colorbar(cf2, ax=axs[1])
    axs[1].scatter(X_train[:, 0], X_train[:, 1], c='red', s=50, edgecolors='black')
    axs[1].set_title("True Branin Function")

    # Absolute Error
    cf3 = axs[2].contourf(X1_grid, X2_grid, abs_err_clipped,
                           norm=LogNorm(), cmap='magma_r')
    fig.colorbar(cf3, ax=axs[2])
    axs[2].scatter(X_train[:, 0], X_train[:, 1], c='white', s=50, edgecolors='black')
    axs[2].set_title("Absolute Error (Log Scale)")

    # Draw directional rays at each training point
    ray_length = 0.8
    for ax, color in zip(axs, ['white', 'white', 'black']):
        for pt in X_train:
            for i in range(rays.shape[1]):
                direction = rays[:, i]
                ax.arrow(pt[0], pt[1], direction[0]*ray_length,
                         direction[1]*ray_length, head_width=0.3, head_length=0.4,
                         fc=color, ec=color)

    for ax in axs:
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_aspect("equal")
    
    plt.show()
    
    print(f"\nFinal NRMSE: {nrmse:.6f}")

**Explanation:**  
The three-panel visualization shows:

1. **Left panel**: DDEGP prediction with training points and directional rays
2. **Center panel**: True Branin function for comparison
3. **Right panel**: Absolute error on a logarithmic scale

The directional rays (white/black arrows) at each training point illustrate the directions along which derivative information was incorporated. This visualization helps understand how the directional derivative information influences the model's predictions.

---

Summary
~~~~~~~
This tutorial demonstrates the **Directional Derivative-Enhanced Gaussian Process (DDEGP)** framework with the following key features:

**Advantages of Directional Derivatives:**

- **Flexibility**: Derivatives can be specified along arbitrary directions, not just coordinate axes
- **Efficiency**: Fewer derivative evaluations needed compared to full gradient information
- **Problem-specific**: Directions can be chosen based on domain knowledge or optimization objectives
- **Scalability**: Particularly beneficial in high-dimensional spaces where full gradients are expensive

**Key Concepts:**

1. **Global directional rays**: The same set of directions is used at all training points
2. **Symbolic differentiation**: SymPy computes exact partial derivatives analytically
3. **Chain rule**: Directional derivatives are computed from coordinate gradients
4. **Latin Hypercube Sampling**: Ensures efficient coverage of the input space
5. **Direct derivative prediction**: Model can predict derivatives without finite differences
6. **Interpolation verification**: Confirms GP correctly learned directional derivative information
7. **Visualization**: Directional rays are displayed to show where derivative information is incorporated

**Implementation Notes:**

- Predictions return arrays with shape ``(1, n_pts)`` for function values
- Derivative predictions are returned as ``(n_pts * (1 + n_rays), 1)`` concatenated vector
- Use proper indexing: ``y_pred[0, i]`` for functions, ``deriv_pred[i, 0]`` for derivatives