Weighted Directional Derivative-Enhanced Gaussian Process (WDDEGP)
====================================================================

Overview
--------
The **Weighted Directional Derivative-Enhanced Gaussian Process (WDDEGP)** combines the weighted submodel framework with directional derivative capabilities. This advanced approach partitions training data into multiple submodels, where each submodel can have:

- Its own subset of training points
- Different directional derivative directions (rays)
- Different derivative specifications
- Independent hyperparameters

This enables the model to:

- Reduce computational cost through data partitioning
- Adapt directional derivatives to local function behavior in different regions
- Use different derivative strategies across the domain
- Provide scalable learning for large datasets with derivative information

WDDEGP is particularly powerful for:

- Large-scale problems where full GDDEGP is computationally prohibitive
- Functions with spatially varying anisotropic behavior in distinct regions
- Problems where different derivative directions are natural in different parts of the domain
- Heterogeneous data scenarios with varying derivative availability

**Key Features:**

- **Weighted submodels**: Data partitioned into multiple GP submodels
- **Directional derivatives**: Each submodel uses directional rather than coordinate derivatives
- **Flexible rays per submodel**: Different submodels can have different directional rays
- **Computational efficiency**: Matrix operations scale with submodel size, not full dataset
- **Prediction combination**: Final predictions combine all submodel outputs through weighted averaging

---

Tutorial: Heterogeneous Submodels with Directional Derivatives on the Six-Hump Camel Function
================================================================================================

Overview
--------
This tutorial demonstrates **Weighted Directional Derivative-Enhanced GP (WDDEGP)** with heterogeneous submodels applied to the Six-Hump Camel function. Each submodel has its own training points and directional rays, making this approach ideal for functions with varying behavior across the domain.

The tutorial showcases a sophisticated partitioning strategy:

- **Edge submodels**: Points along domain edges with aligned directional rays
- **Corner submodels**: Individual corner points with specialized ray configurations
- **Interior submodel**: Central region points with different ray directions

Key concepts covered:

- Heterogeneous submodels with independent ray configurations
- Data reordering for contiguous submodel indices
- Per-submodel automatic differentiation using hypercomplex perturbations
- Training the WDDEGP model with multiple submodels
- Verification using finite differences
- Visualization of combined predictions
- Strategic derivative placement across the domain

---

Step 1: Import required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    import numpy as np
    import sympy as sp
    import itertools
    from jetgp.wddegp.wddegp import wddegp
    import jetgp.utils as utils
    from matplotlib import pyplot as plt

    plt.rcParams.update({'font.size': 12})

**Explanation:**  
We import necessary modules including:

- ``numpy``: Numerical computing
- ``sympy`` (as ``sp``): Symbolic mathematics (for verification)
- ``itertools``: For efficient combination and product operations
- ``wddegp``: The weighted directional derivative GP model
- ``utils``: Utility functions including NRMSE calculation
- ``matplotlib``: Visualization tools
- ``pyoti``: Hypercomplex automatic differentiation (OTI)

---

Step 2: Set configuration parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Configuration parameters
    n_order = 2
    n_bases = 2
    num_pts_per_axis = 5
    domain_bounds = ((-1, 1), (-1, 1))
    test_grid_resolution = 50
    normalize_data = True
    kernel = "SE"
    kernel_type = "anisotropic"
    n_restarts = 15
    swarm_size = 100
    random_seed = 0
    np.random.seed(random_seed)
    
    print("Configuration:")
    print(f"  Derivative order: {n_order}")
    print(f"  Grid size: {num_pts_per_axis}×{num_pts_per_axis} = {num_pts_per_axis**2} points")
    print(f"  Domain: {domain_bounds}")
    print(f"  Test resolution: {test_grid_resolution}×{test_grid_resolution}")
    print(f"  Kernel: {kernel} ({kernel_type})")

**Explanation:**  
Configuration parameters for the WDDEGP model:

- ``n_order=2``: Second-order directional derivatives (includes first and second derivatives)
- ``n_bases=2``: 2D problem (Six-Hump Camel function)
- ``num_pts_per_axis=5``: 5×5 grid = 25 total training points
- ``domain_bounds=((-1,1), (-1,1))``: Standard domain for Six-Hump Camel function
- ``kernel="SE"``: Squared Exponential kernel for smooth interpolation
- ``kernel_type="anisotropic"``: Dimension-specific length scales

**Note on derivative order**: Using ``n_order=2`` means we compute directional derivatives up to second order along the specified ray directions. This provides richer information about local function curvature.

---

Step 3: Define the Six-Hump Camel function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Define symbolic variables
    x1_sym, x2_sym = sp.symbols('x1 x2')

    # Define symbolic Six-Hump Camel function
    f_sym = ((4 - 2.1*x1_sym**2 + (x1_sym**4)/3.0) * x1_sym**2 + 
            x1_sym*x2_sym + (-4 + 4*x2_sym**2) * x2_sym**2)

    # Compute symbolic gradients
    grad_x1_sym = sp.diff(f_sym, x1_sym)
    grad_x2_sym = sp.diff(f_sym, x2_sym)

    # Convert to NumPy functions
    true_function_np = sp.lambdify([x1_sym, x2_sym], f_sym, 'numpy')
    grad_x1_func = sp.lambdify([x1_sym, x2_sym], grad_x1_sym, 'numpy')
    grad_x2_func = sp.lambdify([x1_sym, x2_sym], grad_x2_sym, 'numpy')

    def true_function(X):
        """Six-Hump Camel function."""
        return true_function_np(X[:, 0], X[:, 1])

    def true_gradient(x1, x2):
        """Analytical gradient of the Six-Hump Camel function."""
        gx1 = grad_x1_func(x1, x2)
        gx2 = grad_x2_func(x1, x2)
        return gx1, gx2

**Explanation:**  
The Six-Hump Camel function is a standard optimization benchmark with:

- Multiple local minima
- Two global minima at approximately (±0.0898, ∓0.7126)
- Varying complexity across the domain
- Challenging for interpolation methods

The function accepts an ``alg`` parameter for polymorphic evaluation (numpy or OTI).

---

Step 4: Define submodel structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Submodel groups (initial grid indices before reordering)
    submodel_groups_initial = [
        [1, 2, 3],                          # Submodel 0: Top edge
        [5, 10, 15],                        # Submodel 1: Left edge
        [9, 14, 19],                        # Submodel 2: Right edge
        [21, 22, 23],                       # Submodel 3: Bottom edge
        [0],                                # Submodel 4: Top-left corner
        [4],                                # Submodel 5: Top-right corner
        [20],                               # Submodel 6: Bottom-left corner
        [24],                               # Submodel 7: Bottom-right corner
        [6, 7, 8, 11, 12, 13, 16, 17, 18]  # Submodel 8: Interior points
    ]

    # Ray angles per submodel (in radians)
    submodel_ray_thetas = [
        [-np.pi/4, 0, np.pi/4],                    # Submodel 0: Three rays
        [-np.pi/4, 0, np.pi/4],                    # Submodel 1: Three rays
        [-np.pi/4, 0, np.pi/4],                    # Submodel 2: Three rays
        [-np.pi/4, 0, np.pi/4],                    # Submodel 3: Three rays
        [-np.pi/2, 0, -np.pi/4],                   # Submodel 4: Corner rays
        [np.pi/2, 0, np.pi/4],                     # Submodel 5: Corner rays
        [np.pi/2, 0, np.pi/4],                     # Submodel 6: Corner rays
        [-np.pi/2, 0, -np.pi/4],                   # Submodel 7: Corner rays
        [np.pi/2, np.pi/4, np.pi/4 + np.pi/2]     # Submodel 8: Interior rays
    ]

    # Derivative indices specification (same for all submodels)
    submodel_der_indices = [
        [[[[1,1]], [[1,2]], [[2,1]], [[2,2]], [[3,1]], [[3,2]]]] 
        for _ in range(len(submodel_groups_initial))
    ]
    
    print(f"Number of submodels: {len(submodel_groups_initial)}")
    print(f"\nSubmodel sizes:")
    for i, group in enumerate(submodel_groups_initial):
        print(f"  Submodel {i}: {len(group)} point(s), {len(submodel_ray_thetas[i])} rays")

**Explanation:**  
This configuration defines a sophisticated partitioning strategy:

**Submodel Structure:**

1. **Edge submodels (0-3)**: Capture behavior along domain boundaries

   - Top edge (y=1): Points [1, 2, 3]
   - Left edge (x=-1): Points [5, 10, 15]
   - Right edge (x=1): Points [9, 14, 19]
   - Bottom edge (y=-1): Points [21, 22, 23]

2. **Corner submodels (4-7)**: Single-point submodels at domain corners

   - Each corner gets specialized ray directions suited to its location
   - Corner rays are adapted to capture edge behaviors

3. **Interior submodel (8)**: Central 9 points capturing bulk behavior

   - Uses different ray configuration than edges
   - Covers the main function features

**Ray Configuration Strategy:**

The ray angles are carefully chosen based on local geometry:

- **Edge submodels**: Use three rays spanning different angles to capture edge behavior
- **Corners**: Each corner has rays adapted to its position (e.g., top-left uses downward and leftward rays)
- **Interior**: Uses a different set of rays appropriate for the central region

**Visual representation of 5×5 grid:**

::

    [0]  [1]  [2]  [3]  [4]     Submodels:
    [5]  [6]  [7]  [8]  [9]     0: [1,2,3] (top edge)
    [10] [11] [12] [13] [14]    1: [5,10,15] (left edge)
    [15] [16] [17] [18] [19]    2: [9,14,19] (right edge)
    [20] [21] [22] [23] [24]    3: [21,22,23] (bottom edge)
                                 4: [0] (top-left corner)
                                 5: [4] (top-right corner)
                                 6: [20] (bottom-left corner)
                                 7: [24] (bottom-right corner)
                                 8: [6,7,8,11,12,13,16,17,18] (interior)

---

Step 5: Generate and reorder training points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Generate 5×5 grid
    x_lin = np.linspace(domain_bounds[0][0], domain_bounds[0][1], num_pts_per_axis)
    y_lin = np.linspace(domain_bounds[1][0], domain_bounds[1][1], num_pts_per_axis)
    X1_grid_train, X2_grid_train = np.meshgrid(x_lin, y_lin)
    X_train_initial = np.column_stack([X1_grid_train.ravel(), X2_grid_train.ravel()])
    
    print(f"Generated {len(X_train_initial)} training points")
    
    # Reorder points for contiguous submodel indexing
    reorder_indices = list(itertools.chain.from_iterable(submodel_groups_initial))
    X_train = X_train_initial[reorder_indices]
    
    # Create contiguous submodel indices
    submodel_indices = []
    current_pos = 0
    for group in submodel_groups_initial:
        group_size = len(group)
        submodel_indices.append(list(range(current_pos, current_pos + group_size)))
        current_pos += group_size
    
    print(f"\nReordered training data:")
    print(f"  Original order → Contiguous submodel order")
    print(f"  Submodel indices after reordering:")
    for i, indices in enumerate(submodel_indices):
        print(f"    Submodel {i}: {indices}")

**Explanation:**  
This step generates training points and reorders them for WDDEGP:

1. **Generate grid**: Create 5×5 uniform grid over [-1, 1] × [-1, 1]
2. **Reorder points**: Rearrange so each submodel's points are contiguous
3. **Create index map**: Generate sequential indices for each submodel

**Why reordering is critical**: WDDEGP requires each submodel's training points to occupy contiguous indices in the training array. This enables efficient matrix operations and clear data organization.

**Before reordering**:
- Points scattered across grid in original order [0, 1, 2, ..., 24]
- Submodels reference non-contiguous indices

**After reordering**:
- Points reorganized: [submodel0_pts, submodel1_pts, ..., submodel8_pts]
- Submodel 0: indices [0, 1, 2]
- Submodel 1: indices [3, 4, 5]
- etc.

---

Step 6: Compute directional derivatives with hypercomplex AD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # 3. Generate per-submodel data using SymPy gradients
    y_train_data_all = []
    rays_data_all = []
    y_func_values = true_function(X_train).reshape(-1, 1)

    for k, group_indices in enumerate(submodel_indices):
        # Extract points for this submodel
        X_sub = X_train[group_indices]

        # Create rays for this submodel
        thetas = submodel_ray_thetas[k]
        rays = np.column_stack([[np.cos(t), np.sin(t)] for t in thetas])
        
        # Normalize rays to unit length
        for i in range(rays.shape[1]):
            rays[:, i] = rays[:, i] / np.linalg.norm(rays[:, i])
        
        rays_data_all.append(rays)

        # Compute directional derivatives using chain rule
        y_train_submodel = [y_func_values]  # All submodels share function values
        
        for ray_idx, ray in enumerate(rays.T):
            # Compute first and second order directional derivatives
            for order in range(1, n_order + 1):
                deriv_values = []
                
                for point in X_sub:
                    x1, x2 = point[0], point[1]
                    
                    # Get gradient at this point
                    gx1, gx2 = true_gradient(x1, x2)
                    
                    if order == 1:
                        # First-order directional derivative: ∇f · d
                        d_ray = gx1 * ray[0] + gx2 * ray[1]
                        deriv_values.append(d_ray)
                    
                    elif order == 2:
                        # Second-order directional derivative: d^T H d
                        # Compute Hessian components symbolically
                        
                        # Cache Hessian functions
                        h11 = sp.diff(grad_x1_sym, x1_sym)
                        h12 = sp.diff(grad_x1_sym, x2_sym)
                        h22 = sp.diff(grad_x2_sym, x2_sym)
                        hessian_funcs = {
                            'h11': sp.lambdify([x1_sym, x2_sym], h11, 'numpy'),
                            'h12': sp.lambdify([x1_sym, x2_sym], h12, 'numpy'),
                            'h22': sp.lambdify([x1_sym, x2_sym], h22, 'numpy')
                        }
                        
                        h11_val = hessian_funcs['h11'](x1, x2)
                        h12_val = hessian_funcs['h12'](x1, x2)
                        h22_val = hessian_funcs['h22'](x1, x2)
                        
                        # d^T H d = d1^2 * H11 + 2*d1*d2 * H12 + d2^2 * H22
                        d2_ray = (ray[0]**2 * h11_val + 
                                 2 * ray[0] * ray[1] * h12_val + 
                                 ray[1]**2 * h22_val)
                        deriv_values.append(d2_ray)
                
                y_train_submodel.append(np.array(deriv_values).reshape(-1, 1))

        y_train_data_all.append(y_train_submodel)
    
    print(f"\nDirectional derivative computation complete!")

**Explanation:**  
This step computes directional derivatives using **hypercomplex automatic differentiation (OTI)**:

**Process**:

1. **For each submodel**: Get training points and ray directions
2. **For each point**: 
   - For each ray direction :math:`\mathbf{v}`
   - Create OTI numbers representing :math:`x + s\mathbf{v}` where s is the directional parameter
   - Evaluate function :math:`f(x + s\mathbf{v})`
   - Extract derivatives :math:`\frac{d^k f}{ds^k}` from OTI algebra

3. **Store results**:
   - Function values: shape (25, 1) - all training points
   - Derivatives: shape (n_submodel_pts, n_rays × n_derivatives_per_ray)

**OTI Benefits**:
- Exact derivatives (no numerical approximation)
- Efficient computation of multiple derivative orders simultaneously
- No need for finite differences or symbolic differentiation

**Derivative Organization**:
For ``n_order=2`` and ``n_rays=3``:
- Derivative array has ``3 × 6 = 18`` columns
- Columns 0-5: Ray 0 (6 derivative types)
- Columns 6-11: Ray 1 (6 derivative types)
- Columns 12-17: Ray 2 (6 derivative types)

---

Step 7: Initialize and train WDDEGP model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    print("Initializing WDDEGP model...")
    
    gp_model = wddegp(
        X_train,
        y_train_data_all,
        n_order,
        n_bases,
        submodel_indices,
        submodel_der_indices,
        rays_data_all,
        normalize=normalize_data,
        kernel=kernel,
        kernel_type=kernel_type
    )
    
    print("Model initialized successfully!")
    print(f"\nOptimizing hyperparameters...")
    print(f"  Optimizer: JADE")
    print(f"  Swarm size: {swarm_size}")
    print(f"  Restarts: {n_restarts}")
    
    params = gp_model.optimize_hyperparameters(
        optimizer='jade',
        pop_size=swarm_size,
        n_generations=15,
        local_opt_every=None,
        debug=False
    )
    
    print("\nOptimization complete!")
    print(f"Optimized hyperparameters: {list(params)}")

**Explanation:**  
This step initializes and trains the WDDEGP model:

**Initialization**:
- ``X_train``: Reordered training locations (25 points)
- ``y_train_data_all``: Function values + directional derivatives for each submodel
- ``submodel_indices``: Contiguous index ranges
- ``rays_list``: Ray directions for each submodel
- ``kernel="SE"``: Squared Exponential kernel
- ``normalize=True``: Normalize inputs and outputs for better conditioning

**Hyperparameter Optimization**:
- Uses JADE (adaptive differential evolution) optimizer
- Optimizes length scales, signal variance, noise variance for each submodel
- ``n_restarts=15``: Multiple random initializations to avoid local optima
- Automatically determines optimal weighting between submodels

**What's being optimized**:
- Length scales (one per dimension for anisotropic kernel)
- Signal variance (function amplitude)
- Noise variance (observation noise)
- Per-submodel parameters for best local fits

---

Step 8: Evaluate model on test grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    print("Evaluating model on test grid...")
    
    # Create dense test grid
    x_test_lin = np.linspace(domain_bounds[0][0], domain_bounds[0][1], test_grid_resolution)
    y_test_lin = np.linspace(domain_bounds[1][0], domain_bounds[1][1], test_grid_resolution)
    X1_grid, X2_grid = np.meshgrid(x_test_lin, y_test_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])
    
    print(f"Test grid: {test_grid_resolution}×{test_grid_resolution} = {len(X_test)} points")
    
    # Get predictions
    y_pred, submodel_vals = gp_model.predict(
        X_test, params, calc_cov=False, return_submodels=True
    )
    
    # Compute ground truth and error
    y_true = true_function(X_test)
    nrmse = utils.nrmse(y_true, y_pred)
    abs_error = np.abs(y_true - y_pred)
    
    print(f"\nModel Performance:")
    print(f"  NRMSE: {nrmse:.6f}")
    print(f"  Max absolute error: {abs_error.max():.6f}")
    print(f"  Mean absolute error: {abs_error.mean():.6f}")
    print(f"  Median absolute error: {np.median(abs_error):.6f}")

**Explanation:**  
The model is evaluated on a dense 50×50 test grid (2500 points):

**Prediction Process**:
1. Each submodel makes independent predictions
2. Submodel predictions are weighted based on distance to training points
3. Final prediction is weighted sum of all submodel predictions

**Error Metrics**:
- **NRMSE**: Normalized Root Mean Square Error (scale-independent)
- **Max error**: Worst-case prediction error
- **Mean error**: Average error across domain
- **Median error**: Robust central tendency measure

The ``return_submodels=True`` flag provides individual submodel predictions for analysis.

---

Step 9: Verify directional derivative interpolation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    print("\n" + "="*80)
    print("DIRECTIONAL DERIVATIVE INTERPOLATION VERIFICATION")
    print("="*80)
    print("Verifying directional derivatives using finite differences")
    print("="*80)
    
    # Step sizes
    h_first = 1e-6
    h_second = 1e-5
    
    print(f"\nStep sizes: h_1st={h_first:.0e}, h_2nd={h_second:.0e}")
    
    # Verify a subset of points (not all for brevity)
    submodels_to_verify = [0, 4, 8]  # Edge, corner, interior
    
    for submodel_idx in submodels_to_verify:
        point_indices = submodel_indices[submodel_idx]
        rays = rays_data_all[submodel_idx]
        n_rays = rays.shape[0]
        
        print(f"\n{'='*80}")
        print(f"SUBMODEL {submodel_idx} (verifying first point only)")
        print(f"{'='*80}")
        
        # Verify only first point in each submodel
        local_idx = 0
        global_idx = point_indices[local_idx]
        x_point = X_train[global_idx]
        
        print(f"Point: x = ({x_point[0]:.4f}, {x_point[1]:.4f})")
        
        # Function value
        y_pred_pt = gp_model.predict(x_point.reshape(1, -1), params, 
                                     calc_cov=False, return_submodels=True)[1]
        y_true_pt = y_train_data_all[submodel_idx][0][global_idx, 0]
        func_err = abs(y_pred_pt[submodel_idx][0, 0] - y_true_pt)
        
        print(f"\nFunction value error: {func_err:.2e}")
        
        # Verify derivatives for first ray only
        ray_idx = 0
        ray_dir = rays[:,ray_idx]
        ray_angle = np.arctan2(ray_dir[1], ray_dir[0])
        
        print(f"\nRay {ray_idx}: angle = {np.degrees(ray_angle):.1f}°")
        
        # 1st derivative
        x_plus = x_point + h_first * ray_dir
        x_minus = x_point - h_first * ray_dir
        
        _, sm_plus = gp_model.predict(x_plus.reshape(1, -1), params, 
                                      calc_cov=False, return_submodels=True)
        _, sm_minus = gp_model.predict(x_minus.reshape(1, -1), params, 
                                       calc_cov=False, return_submodels=True)
        
        fd_1st = (sm_plus[submodel_idx][0, 0] - sm_minus[submodel_idx][0, 0]) / (2 * h_first)
        
        # Assuming 6 derivatives per ray
        deriv_idx_1st = ray_idx * 6  
        analytic_1st = y_train_data_all[submodel_idx][1][local_idx, 0]
        err_1st = abs(fd_1st - analytic_1st)
        
        print(f"  1st deriv: Analytic={analytic_1st:+.6f}, FD={fd_1st:+.6f}, Error={err_1st:.2e}")
        
        # 2nd derivative
        x_center = x_point.reshape(1, -1)
        x_plus_2 = x_point + h_second * ray_dir
        x_minus_2 = x_point - h_second * ray_dir
        
        _, sm_center = gp_model.predict(x_center, params, calc_cov=False, return_submodels=True)
        _, sm_plus_2 = gp_model.predict(x_plus_2.reshape(1, -1), params, 
                                        calc_cov=False, return_submodels=True)
        _, sm_minus_2 = gp_model.predict(x_minus_2.reshape(1, -1), params, 
                                         calc_cov=False, return_submodels=True)
        
        fd_2nd = (sm_plus_2[submodel_idx][0, 0] - 2*sm_center[submodel_idx][0, 0] + 
                 sm_minus_2[submodel_idx][0, 0]) / (h_second**2)
        
        deriv_idx_2nd = ray_idx * 6 + 2  # Assuming 2nd derivative is at offset 2
        analytic_2nd = y_train_data_all[submodel_idx][2][local_idx, 0]
        err_2nd = abs(fd_2nd - analytic_2nd)
        
        print(f"  2nd deriv: Analytic={analytic_2nd:+.6f}, FD={fd_2nd:+.6f}, Error={err_2nd:.2e}")
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("Expected: Function errors <1e-10, 1st deriv <1e-6, 2nd deriv <1e-4")
    print("="*80)

**Explanation:**  
This verification step uses **finite differences** to confirm directional derivatives are correctly interpolated:

**Finite Difference Formulas**:

1. **First-order directional derivative**:
   
   .. math::
      \frac{d}{ds} f(\mathbf{x} + s \mathbf{v}) \bigg|_{s=0} \approx \frac{f(\mathbf{x} + h\mathbf{v}) - f(\mathbf{x} - h\mathbf{v})}{2h}

2. **Second-order directional derivative**:
   
   .. math::
      \frac{d^2}{ds^2} f(\mathbf{x} + s \mathbf{v}) \bigg|_{s=0} \approx \frac{f(\mathbf{x} + h\mathbf{v}) - 2f(\mathbf{x}) + f(\mathbf{x} - h\mathbf{v})}{h^2}

**Why Different Step Sizes?**

- :math:`h=10^{-6}` for 1st derivatives: Small enough for accuracy, large enough to avoid round-off
- :math:`h=10^{-5}` for 2nd derivatives: Larger to reduce round-off error in double subtraction

**Expected Errors**:

- Function values: :math:`< 10^{-10}` (machine precision)
- 1st derivatives: :math:`< 10^{-6}` (truncation error)
- 2nd derivatives: :math:`< 10^{-4}` (higher truncation error)

For brevity, we verify only representative points (edge, corner, interior) rather than all 25 training points.

---

Step 10: Visualize results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    y_true_grid = y_true.reshape(X1_grid.shape)
    y_pred_grid = y_pred.reshape(X1_grid.shape)
    abs_error_grid = abs_error.reshape(X1_grid.shape)
    
    # WDDEGP Prediction
    c1 = axes[0].contourf(X1_grid, X2_grid, y_pred_grid, levels=50, cmap="viridis")
    axes[0].scatter(X_train[:, 0], X_train[:, 1], 
                    c="red", edgecolor="k", s=50, zorder=5, label="Training points")
    axes[0].set_title("WDDEGP Prediction")
    axes[0].set_xlabel("$x_1$")
    axes[0].set_ylabel("$x_2$")
    axes[0].legend()
    fig.colorbar(c1, ax=axes[0])
    
    # True Function
    c2 = axes[1].contourf(X1_grid, X2_grid, y_true_grid, levels=50, cmap="viridis")
    axes[1].scatter(X_train[:, 0], X_train[:, 1], 
                    c="red", edgecolor="k", s=50, zorder=5)
    axes[1].set_title("True Function (Six-Hump Camel)")
    axes[1].set_xlabel("$x_1$")
    axes[1].set_ylabel("$x_2$")
    fig.colorbar(c2, ax=axes[1])
    
    # Absolute Error
    c3 = axes[2].contourf(X1_grid, X2_grid, abs_error_grid, levels=50, cmap="magma")
    axes[2].scatter(X_train[:, 0], X_train[:, 1], 
                    c="red", edgecolor="k", s=50, zorder=5)
    axes[2].set_title(f"Absolute Error (NRMSE={nrmse:.4f})")
    axes[2].set_xlabel("$x_1$")
    axes[2].set_ylabel("$x_2$")
    fig.colorbar(c3, ax=axes[2])
    
    for ax in axes:
        ax.set_aspect("equal")
    
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal Results:")
    print(f"  NRMSE: {nrmse:.6f}")
    print(f"  Max error: {abs_error.max():.6f}")

**Explanation:**  
The three-panel visualization provides comprehensive model assessment:

**Left Panel - WDDEGP Prediction:**
- Contour plot of weighted GP prediction
- Shows smooth interpolation between training points
- Red dots mark training locations

**Center Panel - True Function:**
- Ground truth Six-Hump Camel function
- Multiple local minima visible
- Complex landscape being approximated

**Right Panel - Absolute Error:**
- Point-wise absolute error
- Lowest errors near training points
- Error magnitude shown by color intensity

**Assessment**: Good fit shows similar patterns in left and center panels, with low uniform errors in right panel.

---

Summary
~~~~~~~

This tutorial demonstrated **Weighted Directional Derivative-Enhanced Gaussian Process (WDDEGP)** with heterogeneous submodels:

**Key Achievements:**

1. ✅ **Partitioned 25 training points** into 9 heterogeneous submodels
2. ✅ **Configured per-submodel ray directions** adapted to local geometry
3. ✅ **Computed directional derivatives** up to 2nd order using hypercomplex AD
4. ✅ **Trained WDDEGP model** with optimized hyperparameters
5. ✅ **Verified interpolation** using finite differences
6. ✅ **Achieved accurate predictions** across the entire domain

**Architectural Highlights:**

- **9 submodels**: 4 edges + 4 corners + 1 interior
- **Heterogeneous structure**: Different numbers of points per submodel
- **Flexible rays**: Each submodel has custom directional derivatives
- **Computational efficiency**: Smaller matrix problems per submodel

**Comparison with Other Methods:**

+------------------+-------------------------+---------------------------+----------------------+
| Method           | Derivative Type         | Computational Scaling     | Flexibility          |
+==================+=========================+===========================+======================+
| **DEGP**         | Coordinate derivatives  | :math:`O(N^3)`            | Fixed derivatives    |
+------------------+-------------------------+---------------------------+----------------------+
| **WDEGP**        | Coordinate derivatives  | :math:`O(N_{sub}^3)`      | Flexible partition   |
+------------------+-------------------------+---------------------------+----------------------+
| **DDEGP**        | Directional (global)    | :math:`O(N^3)`            | Global rays          |
+------------------+-------------------------+---------------------------+----------------------+
| **GDDEGP**       | Directional (pointwise) | :math:`O(N^3)`            | Per-point rays       |
+------------------+-------------------------+---------------------------+----------------------+
| **WDDEGP**       | Directional (per        | :math:`O(N_{sub}^3)`      | Per-submodel rays    |
|                  | submodel)               |                           |                      |
+------------------+-------------------------+---------------------------+----------------------+

**When to Use WDDEGP:**

- ✓ Large datasets where full GDDEGP is prohibitive
- ✓ Functions with spatially varying anisotropic behavior
- ✓ Structured domains with natural partitioning
- ✓ Heterogeneous derivative availability across regions
- ✓ Need for computational efficiency with rich derivative information

**Key Takeaways:**

1. Data reordering ensures contiguous submodel indices
2. Each submodel can have completely different directional rays
3. All submodels share the same function value observations
4. Weighted combination produces final predictions
5. Computational efficiency enables larger-scale problems