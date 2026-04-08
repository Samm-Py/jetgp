Sparse Directional Derivative-Enhanced Gaussian Process (Sparse DDEGP)
======================================================================

Overview
--------
The **Sparse Directional Derivative-Enhanced Gaussian Process (Sparse DDEGP)** extends the DDEGP framework by using a **sparse Cholesky (Vecchia) approximation** for the kernel matrix inverse during hyperparameter optimization. This enables efficient training on larger datasets while retaining the ability to incorporate directional derivative information along specified rays.

Key features:

- Uses **sparse Cholesky decomposition** based on the Vecchia approximation for scalable training
- Applies **maximin distance (MMD) ordering** of training points for optimal sparsity patterns
- Prediction uses exact dense Cholesky for accuracy
- Controlled by the ``rho`` parameter (neighborhood size) and ``use_supernodes`` (batched computation)
- Drop-in replacement for the dense DDEGP with only two additional parameters

---

Example 1: Global Directional GP on the Branin Function with Sparse Cholesky
------------------------------------------------------------------------------

Overview
~~~~~~~~
This example demonstrates the **Sparse DDEGP** applied to the 2D Branin function using a **global set of directional rays** (45, 90, and 135 degrees) applied at all training points. The sparse Cholesky approximation accelerates hyperparameter optimization while predictions remain exact.

---

Step 1: Import required packages
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    import numpy as np
    import sympy as sp
    from jetgp.full_ddegp_sparse.ddegp import ddegp
    import jetgp.utils as utils
    from scipy.stats import qmc
    from matplotlib import pyplot as plt
    from matplotlib.colors import LogNorm

**Explanation:**
We import the sparse DDEGP from ``full_ddegp_sparse`` instead of ``full_ddegp``. All other imports are identical.

---

Step 2: Set configuration parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    n_order = 1
    n_bases = 2
    num_training_pts = 16
    domain_bounds = ((-5.0, 10.0), (0.0, 15.0))
    test_grid_resolution = 25

    # Global set of directional rays (45, 90, 135 degrees)
    rays = np.array([
        [np.cos(np.pi/4), np.cos(np.pi/2), np.cos(3*np.pi/4)],
        [np.sin(np.pi/4), np.sin(np.pi/2), np.sin(3*np.pi/4)]
    ])

    normalize_data = True
    kernel = "SE"
    kernel_type = "anisotropic"
    random_seed = 1
    np.random.seed(random_seed)

    # Sparse Cholesky parameters
    rho = 3.0
    use_supernodes = True

    print("Configuration complete!")
    print(f"Number of training points: {num_training_pts}")
    print(f"Number of directional rays: {rays.shape[1]}")
    print(f"Sparse Cholesky: rho={rho}, use_supernodes={use_supernodes}")

**Explanation:**
Configuration is identical to the dense DDEGP, with two additional sparse Cholesky parameters:

- ``rho=3.0``: Neighborhood multiplier for the Vecchia approximation
- ``use_supernodes=True``: Enables batched block computation for speedup

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
The Branin function and its symbolic gradients are defined identically to the dense DDEGP example.

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
    directional_derivs = []
    for i in range(rays.shape[1]):
        ray_direction = rays[:, i]
        dir_deriv = (grad_x1_vals * ray_direction[0] +
                     grad_x2_vals * ray_direction[1])
        directional_derivs.append(dir_deriv)

    # Package training data
    y_train_list = [y_func] + directional_derivs
    der_indices = [[[[1, 1]], [[2, 1]], [[3, 1]]]]

    # Build derivative_locations: all rays at all points
    derivative_locations = []
    for i in range(len(der_indices)):
        for j in range(len(der_indices[i])):
            derivative_locations.append([k for k in range(len(X_train))])

    print(f"Training data generated!")
    print(f"X_train shape: {X_train.shape}")
    print(f"Function values shape: {y_func.shape}")
    print(f"Number of directional derivative arrays: {len(directional_derivs)}")

**Explanation:**
Training data generation is identical to the dense DDEGP. Directional derivatives are computed using the chain rule along each global ray direction.

---

Step 5: Initialize and train the Sparse DDEGP model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Initialize the Sparse DDEGP model
    gp_model = ddegp(
        X_train,
        y_train_list,
        n_order=n_order,
        der_indices=der_indices,
        derivative_locations=derivative_locations,
        rays=rays,
        normalize=normalize_data,
        kernel=kernel,
        kernel_type=kernel_type,
        rho=rho,
        use_supernodes=use_supernodes
    )

    print("Sparse DDEGP model initialized!")
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
The sparse DDEGP model is initialized identically to the dense version, with the addition of ``rho`` and ``use_supernodes``. During optimization, the sparse Cholesky approximation is used to compute the negative log marginal likelihood efficiently.

---

Step 6: Evaluate model on a test grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Create dense test grid
    x_lin = np.linspace(domain_bounds[0][0], domain_bounds[0][1], test_grid_resolution)
    y_lin = np.linspace(domain_bounds[1][0], domain_bounds[1][1], test_grid_resolution)
    X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
    X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

    print(f"Test grid: {test_grid_resolution}x{test_grid_resolution} = {len(X_test)} points")

    # Predict on test grid
    y_pred_full = gp_model.predict(X_test, params, calc_cov=False, return_deriv=False)
    y_pred = y_pred_full[0, :]  # Row 0: function values

    # Compute ground truth and error
    y_true = branin_function(X_test, alg=np)
    nrmse = utils.nrmse(y_true, y_pred)
    abs_error = np.abs(y_true - y_pred)

    print(f"\nModel Performance:")
    print(f"  NRMSE: {nrmse:.6f}")
    print(f"  Max absolute error: {abs_error.max():.6f}")
    print(f"  Mean absolute error: {abs_error.mean():.6f}")

**Explanation:**
Predictions use exact dense Cholesky for full accuracy. The NRMSE quantifies prediction accuracy across the test domain.

---

Step 7: Verify interpolation of training data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    # Verify function value interpolation
    y_pred_train = gp_model.predict(X_train, params, calc_cov=False, return_deriv=False)

    max_func_error = np.max(np.abs(y_pred_train[0, :] - y_func.flatten()))
    print(f"Maximum absolute function value error at training points: {max_func_error:.2e}")

    # Verify directional derivative interpolation
    y_pred_with_derivs = gp_model.predict(X_train, params, calc_cov=False, return_deriv=True)

    print(f"\nPrediction with derivatives shape: {y_pred_with_derivs.shape}")

    for ray_idx in range(rays.shape[1]):
        pred_deriv = y_pred_with_derivs[ray_idx + 1, :]
        analytic_deriv = y_train_list[ray_idx + 1].flatten()
        max_deriv_error = np.max(np.abs(pred_deriv - analytic_deriv))
        angle_deg = np.arctan2(rays[1, ray_idx], rays[0, ray_idx]) * 180 / np.pi
        print(f"Ray {ray_idx + 1} ({angle_deg:.0f} deg) max derivative error: {max_deriv_error:.2e}")

    print("\nInterpolation errors should be near machine precision (< 1e-6)")

**Explanation:**
Since predictions use exact dense Cholesky, both function values and directional derivatives should interpolate to machine precision at training points.

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
    axs[0].set_title("Sparse DDEGP Prediction")

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
The visualization is identical to the dense DDEGP. The directional rays (arrows) at each training point show the directions along which derivative information was incorporated.

---

Summary
~~~~~~~
This example demonstrates the **Sparse DDEGP** as a drop-in replacement for the dense DDEGP:

- **Same API**: Only ``rho`` and ``use_supernodes`` are added to the constructor
- **Same predictions**: Prediction uses exact dense Cholesky for full accuracy
- **Faster training**: Sparse Cholesky approximation accelerates hyperparameter optimization
- **Scalable**: The speedup grows with dataset size, making large-scale DDEGP practical

**Sparse Cholesky Parameters:**

+---------------------+----------+--------------------------------------------------------+
| Parameter           | Default  | Description                                            |
+=====================+==========+========================================================+
| ``rho``             | 3.0      | Neighborhood multiplier for Vecchia conditioning sets. |
|                     |          | Larger values give better approximation but less       |
|                     |          | sparsity.                                              |
+---------------------+----------+--------------------------------------------------------+
| ``use_supernodes``  | True     | Enable supernode aggregation for batched block          |
|                     |          | computation. Provides significant speedup.             |
+---------------------+----------+--------------------------------------------------------+
| ``supernode_lam``   | 1.5      | Controls supernode grouping aggressiveness.             |
+---------------------+----------+--------------------------------------------------------+
