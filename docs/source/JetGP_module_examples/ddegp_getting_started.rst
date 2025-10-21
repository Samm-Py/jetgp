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
    kernel = "RQ"
    kernel_type = "anisotropic"
    n_restarts = 15
    swarm_size = 200
    random_seed = 1
    np.random.seed(random_seed)

**Explanation:**  
We configure the experiment parameters:

- ``n_order=1``: First-order directional derivatives
- ``n_bases=2``: Two-dimensional input space
- ``rays``: Three directional vectors at 45°, 90°, and 225° angles
- ``kernel="RQ"``: Rational Quadratic kernel for flexible length-scale behavior
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

**Explanation:**  
The Branin function is a standard 2D test function with three global minima. We define both a numerical version and a symbolic version using SymPy. The symbolic version allows us to compute exact partial derivatives :math:`\frac{\partial f}{\partial x_1}` and :math:`\frac{\partial f}{\partial x_2}`, which are then converted to fast NumPy-compatible functions using ``lambdify``.

---

Step 4: Generate training data with directional derivatives
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    def generate_training_data():
        """Generate training data with LHS and compute directional derivatives using SymPy."""
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

        return {'X_train': X_train, 'y_train_list': y_train_list, 'der_indices': der_indices}

**Explanation:**  
This function performs several key steps:

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

    def train_model(training_data):
        """Initialize and train the ddegp model."""
        gp_model = ddegp(
            training_data['X_train'], 
            training_data['y_train_list'],
            n_order=n_order, 
            der_indices=training_data['der_indices'],
            rays=rays, 
            normalize=normalize_data,
            kernel=kernel, 
            kernel_type=kernel_type
        )
        params = gp_model.optimize_hyperparameters(
        optimizer='jade',
        pop_size = 100,
        n_generations = 15,
        local_opt_every = None,
        debug = True
        )
        return gp_model, params

**Explanation:**  
The DDEGP model is initialized with:

- Training locations and derivative data
- Directional rays defining the derivative directions
- Rational Quadratic kernel for flexible modeling

Hyperparameters are optimized using Particle Swarm Optimization with multiple restarts for robustness.

---

Step 6: Evaluate model on a test grid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    def evaluate_model(gp_model, params):
        """Evaluate model on a grid and compute NRMSE."""
        x_lin = np.linspace(domain_bounds[0][0], domain_bounds[0][1], test_grid_resolution)
        y_lin = np.linspace(domain_bounds[1][0], domain_bounds[1][1], test_grid_resolution)
        X1_grid, X2_grid = np.meshgrid(x_lin, y_lin)
        X_test = np.column_stack([X1_grid.ravel(), X2_grid.ravel()])

        y_pred = gp_model.predict(X_test, params, calc_cov=False, return_deriv=False)
        y_true = branin_function(X_test, alg=np)
        nrmse_val = utils.nrmse(y_true, y_pred)

        return {'X_test': X_test, 'X1_grid': X1_grid, 'X2_grid': X2_grid,
                'y_pred': y_pred, 'y_true': y_true, 'nrmse': nrmse_val}

**Explanation:**  
The model is evaluated on a 50×50 grid covering the entire domain. The Normalized Root Mean Square Error (NRMSE) quantifies prediction accuracy across the test set.

---

Step 7: Visualize results with directional rays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    def visualize_results(training_data, results):
        X_train = training_data['X_train']
        res = results
        fig, axs = plt.subplots(1, 3, figsize=(19, 5), constrained_layout=True)

        gp_map = res['y_pred'].reshape(res['X1_grid'].shape)
        true_map = res['y_true'].reshape(res['X1_grid'].shape)
        abs_err = np.abs(gp_map - true_map)
        abs_err_clipped = np.clip(abs_err, 1e-8, None)

        # GP Prediction
        cf1 = axs[0].contourf(res['X1_grid'], res['X2_grid'], gp_map, cmap='viridis')
        fig.colorbar(cf1, ax=axs[0])
        axs[0].scatter(X_train[:, 0], X_train[:, 1], c='red', s=50, edgecolors='black')
        axs[0].set_title("GP Prediction")

        # True Function
        cf2 = axs[1].contourf(res['X1_grid'], res['X2_grid'], true_map, cmap='viridis')
        fig.colorbar(cf2, ax=axs[1])
        axs[1].scatter(X_train[:, 0], X_train[:, 1], c='red', s=50, edgecolors='black')
        axs[1].set_title("True Branin Function")

        # Absolute Error
        cf3 = axs[2].contourf(res['X1_grid'], res['X2_grid'], abs_err_clipped,
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

**Explanation:**  
The three-panel visualization shows:

1. **Left panel**: DDEGP prediction with training points and directional rays
2. **Center panel**: True Branin function for comparison
3. **Right panel**: Absolute error on a logarithmic scale

The directional rays (white/black arrows) at each training point illustrate the directions along which derivative information was incorporated. This visualization helps understand how the directional derivative information influences the model's predictions.

---

Step 8: Run the complete tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. jupyter-execute::

    training_data = generate_training_data()
    gp_model, params = train_model(training_data)
    results = evaluate_model(gp_model, params)
    visualize_results(training_data, results)
    print(f"Final NRMSE: {results['nrmse']:.6f}")

**Explanation:**  
This executes the complete workflow:

1. Generate training data with directional derivatives
2. Train the DDEGP model
3. Evaluate on a test grid
4. Visualize results with directional rays overlaid

The final NRMSE provides a quantitative measure of model accuracy.

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
5. **Visualization**: Directional rays are displayed to show where derivative information is incorporated
